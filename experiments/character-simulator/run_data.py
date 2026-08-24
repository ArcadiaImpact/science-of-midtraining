"""Task 4 driver: annotation probe → pilot (Terra vs Luna + Sol pairwise) → GATE → full run.

Spend ladder (PLAN.md Task 4) — each phase is separately invoked, disk-cached
(resumable), cost-audited from the cache files, and refuses a dirty git tree:

    uv run python experiments/character-simulator/run_data.py --phase probe
    uv run python experiments/character-simulator/run_data.py --phase pilot
    uv run python experiments/character-simulator/run_data.py --phase full --model gpt-5.6-terra

`full` additionally requires design/FULL_RUN_APPROVAL.md (written by Jonathan
at the gate) naming the model and budget; it runs in serial batches of
`cfg.annotate.batch_rows` with atomic per-batch persistence and aborts if the
projected cost at ≥10% completion exceeds the approved budget by >25%.

Module import is stdlib+charsim only (tests import the pure helpers);
scimt/network imports happen inside phases.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from charsim import annotate, contracts, health  # noqa: E402
from charsim.config import Config  # noqa: E402

REPO = HERE.parents[1]
SOURCES_DIR = HERE / "data" / "sources"
CACHE_ROOT = HERE / "data" / "annotate_cache"
ANNOTATIONS_DIR = HERE / "data" / "annotations"
PROBE_DIR = HERE / "data" / "probe"
RUNS_DIR = HERE / "runs" / "annotate"
RESULTS_DIR = HERE / "results"
APPROVAL_PATH = HERE / "design" / "FULL_RUN_APPROVAL.md"

PROBE_N = 20
PROBE_EFFORTS = ("none", "low")
PILOT_N = 1_000
PILOT_MODELS = ("gpt-5.6-terra", "gpt-5.6-luna")
SOL_MODEL = "gpt-5.6-sol"
SOL_N = 250
# Terra takes the full run only if its Sol win-rate CI clears this margin;
# otherwise Luna (10x cheaper) is recommended (D11).
TERRA_WIN_MARGIN = 0.55

SOL_EVAL_SYSTEM = """\
You are evaluating two candidate annotations of the same chat transcript. Each annotation \
describes the assistant's traits and the user's traits. The annotations will be used as \
conditioning prefixes for a conversation-simulator model, so the better annotation is the one \
that is more DISTINCTIVE (specific to this conversation, not generically true of most chats), \
better EVIDENCED by the transcript (no invented claims), and would better help a language \
model PREDICT this exact conversation. Ignore length and writing polish except where they \
affect those criteria. Reply with exactly one letter: A or B."""

SOL_EVAL_USER_TEMPLATE = """\
Transcript:

{transcript}

--- Annotation A ---
{first}

--- Annotation B ---
{second}

Which annotation is better by the rubric? Reply with exactly one letter: A or B."""

_LETTER_RE = re.compile(r"\A\W*([ABab])\W*\Z")


# --- pure helpers (unit-tested, no I/O) -----------------------------------------


def pick_stratified(rows: list, n: int, seed: int = contracts.SEED) -> list:
    """Deterministic per-source round-robin sample: group by source, shuffle each
    group with a seed derived from (seed, source), interleave until n."""
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r.source, []).append(r)
    shuffled = []
    for source in sorted(groups):
        g = sorted(groups[source], key=lambda r: r.row_id)
        random.Random(f"{seed}:{source}").shuffle(g)
        shuffled.append(g)
    picked = []
    i = 0
    while len(picked) < n and any(i < len(g) for g in shuffled):
        for g in shuffled:
            if i < len(g) and len(picked) < n:
                picked.append(g[i])
        i += 1
    if len(picked) < n:
        raise ValueError(f"asked for {n} rows, sources only hold {len(picked)}")
    return picked


def render_annotation_for_eval(ann) -> str:
    """Annotation as the evaluator sees it — annotator identity withheld."""
    return (f"assistant_traits: {ann.assistant_traits}\n"
            f"user_traits: {ann.user_traits}\n"
            f"topic: {ann.topic}\n"
            f"tags: {', '.join(ann.tags)}")


def sol_first_model(pair_idx: int) -> str:
    """Deterministic position counterbalance: which model is shown as A."""
    return PILOT_MODELS[pair_idx % 2]


def parse_ab_letter(text: str) -> str | None:
    m = _LETTER_RE.match(text.strip())
    return m.group(1).upper() if m else None


def parse_approval(text: str) -> dict:
    """FULL_RUN_APPROVAL.md must pin the model and budget on their own lines:
    `model: gpt-5.6-terra` and `budget_usd: 900`."""
    model = re.search(r"^model:\s*(\S+)\s*$", text, re.M)
    budget = re.search(r"^budget_usd:\s*([0-9]+(?:\.[0-9]+)?)\s*$", text, re.M)
    if not model or not budget:
        raise ValueError(
            "FULL_RUN_APPROVAL.md must contain `model: <id>` and `budget_usd: <n>` lines")
    return {"model": model.group(1), "budget_usd": float(budget.group(1))}


def cost_summary(cache_dirs: list[Path], prices: dict[str, dict]) -> dict:
    """Sum logged token usage across cache_*.jsonl files (the docgen_v1 walker,
    simplified: cached_client logs each real API call once per file)."""
    by_model: dict[str, dict] = {}
    n_records = 0
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        for path in sorted(cache_dir.rglob("cache_*.jsonl")):
            seen: set[str] = set()
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail record from an interrupted write
                key = row.get("audit_id", row.get("key", ""))
                if key in seen:
                    continue
                seen.add(key)
                n_records += 1
                model = (row.get("endpoint") or {}).get("model", "?")
                usage = (row.get("response") or {}).get("usage") or {}
                inp = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                out = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
                item = by_model.setdefault(model, {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
                price = prices.get(model, {})
                item["calls"] += 1
                item["input_tokens"] += inp
                item["output_tokens"] += out
                item["usd"] += (inp * price.get("input_usd_per_mtok", 0.0)
                                + out * price.get("output_usd_per_mtok", 0.0)) / 1_000_000
    return {
        "logged_api_calls": n_records,
        "by_model": by_model,
        "total_usd": round(sum(m["usd"] for m in by_model.values()), 4),
        "note": "logged responses only; provider invoices remain authoritative",
    }


def project_full_cost(pilot_cost: dict, pilot_rows: int, total_rows: int) -> dict:
    """Per-model full-run projection from realized pilot usage."""
    out = {}
    for model, item in pilot_cost["by_model"].items():
        if model == SOL_MODEL or item["calls"] == 0:
            continue
        scale = total_rows / max(pilot_rows, 1)
        out[model] = {
            "pilot_usd": round(item["usd"], 4),
            "mean_input_tokens_per_call": item["input_tokens"] / item["calls"],
            "mean_output_tokens_per_call": item["output_tokens"] / item["calls"],
            "calls_per_row": item["calls"] / max(pilot_rows, 1),  # >1 = retries
            "projected_full_usd": round(item["usd"] * scale, 2),
        }
    return out


# --- infra ------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def append_event(run_dir: Path, event: str, **kw) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "events.jsonl").open("a") as f:
        f.write(json.dumps({"ts": _utcnow(), "event": event, **kw}) + "\n")


def git_state() -> dict:
    def _git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    return {"commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain"))}


def require_clean_tree(state: dict) -> None:
    if state["dirty"] and not os.environ.get("SCIMT_ALLOW_DIRTY"):
        raise RuntimeError(
            "git tree is dirty — commit before a paid phase (SCIMT_ALLOW_DIRTY=1 to override)")


def load_split(split: str) -> list[contracts.ConvRow]:
    path = SOURCES_DIR / f"{split}.convrows.jsonl"
    if not path.exists():
        have = sorted(p.name for p in SOURCES_DIR.glob("*.jsonl")) if SOURCES_DIR.exists() else []
        raise FileNotFoundError(
            f"{path} missing — run Task 1's build_sources first (present: {have})")
    return [contracts.ConvRow.from_json(line)
            for line in path.read_text().splitlines() if line.strip()]


def _prices() -> dict[str, dict]:
    from scimt.gen.plan import load_catalog
    return {m.model: {"input_usd_per_mtok": m.input, "output_usd_per_mtok": m.output}
            for m in load_catalog()}


async def catalog_gate(models: list[str], run_dir: Path) -> None:
    """verify_catalog before any paid call; drift on a model we are about to
    pay for is fatal, drift elsewhere is logged."""
    from scimt.gen.plan import verify_catalog
    issues = await verify_catalog()
    append_event(run_dir, "catalog_verified", n_issues=len(issues), issues=issues)
    fatal = [i for i in issues if any(m in i for m in models)]
    if fatal:
        raise RuntimeError(f"model_catalog.yaml drifted for phase models: {fatal}")


def _write_jsonl_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(line + "\n" for line in lines))
    os.replace(tmp, path)


def _ann_record(row, ann, split: str) -> str:
    return json.dumps({
        "row_id": row.row_id, "source": row.source, "split": split,
        "annotation": json.loads(ann.to_json()) if ann is not None else None,
    }, ensure_ascii=False)


# --- phases -------------------------------------------------------------------------


async def phase_probe(cfg: Config, run_dir: Path) -> None:
    """20 stratified conversations × {none, low} reasoning effort on Terra;
    emits a hand-review file. Kill-fast: yield <90% or boilerplate >25% (by hand)."""
    rows = pick_stratified(load_split("train"), PROBE_N)
    append_event(run_dir, "probe_rows_picked", n=len(rows),
                 by_source=dict(Counter(r.source for r in rows)),
                 row_ids=[r.row_id for r in rows])
    await catalog_gate([cfg.annotate.model], run_dir)

    arms: dict[str, list] = {}
    for effort in PROBE_EFFORTS:
        # reasoning tokens bill against max_completion_tokens on gpt-5.x —
        # the "low" arm needs headroom beyond the JSON itself
        arm_cfg = dataclasses.replace(
            cfg.annotate, reasoning_effort=effort,
            max_tokens=cfg.annotate.max_tokens if effort == "none" else 2_000)
        anns = await annotate.annotate_rows(rows, arm_cfg, CACHE_ROOT / "probe" / effort)
        arms[effort] = anns
        _write_jsonl_atomic(PROBE_DIR / f"annotations_{effort}.jsonl",
                            [_ann_record(r, a, "probe") for r, a in zip(rows, anns)])
        report = health.health_report(anns)
        append_event(run_dir, "probe_arm_done", effort=effort, yield_=report["yield"],
                     generic_rate=report["generic_rate"])
        print(f"[probe:{effort}] yield={report['yield']:.2f} "
              f"generic_rate={report['generic_rate']:.2f}")

    review = [f"# Probe hand-review — contract v{contracts.CONTRACT_VERSION}, "
              f"{len(rows)} convs x {list(PROBE_EFFORTS)}\n"]
    for i, row in enumerate(rows):
        t = annotate.render_transcript(row)
        excerpt = t if len(t) <= 1_500 else t[:1_100] + "\n[...]\n" + t[-400:]
        review.append(f"\n## {i + 1}. {row.row_id} ({row.source}, {len(row.turns)} turns)\n"
                      f"```\n{excerpt}\n```\n")
        for effort in PROBE_EFFORTS:
            ann = arms[effort][i]
            review.append(f"**{effort}:** " + (
                "FAILED\n" if ann is None else
                f"\n- assistant: {ann.assistant_traits}\n- user: {ann.user_traits}\n"
                f"- topic: {ann.topic} | tags: {', '.join(ann.tags)}\n"))
    review_path = PROBE_DIR / f"review_v{contracts.CONTRACT_VERSION}.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("".join(review))
    print(f"[probe] hand-review file: {review_path}")


async def phase_pilot(cfg: Config, run_dir: Path) -> None:
    """1,000 rows annotated by BOTH pilot models, then Sol pairwise on SOL_N
    both-succeeded pairs (position-counterbalanced). Writes results/annotation_pilot.json."""
    from scimt.analysis.classical import wilson_interval
    from scimt.utils.client import Endpoint, cached_client, completion_params

    rows = pick_stratified(load_split("train"), PILOT_N)
    append_event(run_dir, "pilot_rows_picked", n=len(rows),
                 by_source=dict(Counter(r.source for r in rows)))
    await catalog_gate([*PILOT_MODELS, SOL_MODEL], run_dir)

    by_model: dict[str, list] = {}
    reports: dict[str, dict] = {}
    for model in PILOT_MODELS:
        arm_cfg = dataclasses.replace(cfg.annotate, model=model)
        anns = await annotate.annotate_rows(rows, arm_cfg, CACHE_ROOT / "pilot")
        by_model[model] = anns
        _write_jsonl_atomic(PROBE_DIR.parent / "pilot" / f"annotations_{model}.jsonl",
                            [_ann_record(r, a, "pilot") for r, a in zip(rows, anns)])
        reports[model] = health.health_report(anns)
        append_event(run_dir, "pilot_arm_done", model=model,
                     yield_=reports[model]["yield"],
                     generic_rate=reports[model]["generic_rate"])
        print(f"[pilot:{model}] yield={reports[model]['yield']:.3f} "
              f"generic={reports[model]['generic_rate']:.3f} "
              f"distinct2={reports[model]['assistant_traits']['distinct_2']:.3f}")

    # Sol pairwise on the first SOL_N rows where both annotators succeeded
    pairs = [(i, r) for i, r in enumerate(rows)
             if by_model[PILOT_MODELS[0]][i] is not None
             and by_model[PILOT_MODELS[1]][i] is not None][:SOL_N]
    endpoint = Endpoint(base_url="https://api.openai.com/v1", model=SOL_MODEL)
    client = cached_client(endpoint, CACHE_ROOT / "pilot", "sol_pairwise",
                           concurrency=cfg.annotate.concurrency)
    params = completion_params(SOL_MODEL, temperature=1.0, max_tokens=1_600,
                               reasoning_effort="low")

    async def judge_one(pair_idx: int, row_idx: int, row) -> dict:
        first = sol_first_model(pair_idx)
        second = PILOT_MODELS[1] if first == PILOT_MODELS[0] else PILOT_MODELS[0]
        user = SOL_EVAL_USER_TEMPLATE.format(
            transcript=annotate.render_transcript(row),
            first=render_annotation_for_eval(by_model[first][row_idx]),
            second=render_annotation_for_eval(by_model[second][row_idx]))
        payload = {"model": SOL_MODEL, **params,
                   "messages": [{"role": "system", "content": SOL_EVAL_SYSTEM},
                                {"role": "user", "content": user}]}
        letter = None
        for attempt in range(3):
            salt = f"sol:v{contracts.CONTRACT_VERSION}:{row.row_id}:attempt:{attempt}"
            try:
                data = await client.chat(payload, cache_salt=salt)
                text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                letter = parse_ab_letter(text)
            except Exception as e:
                append_event(run_dir, "sol_error", row_id=row.row_id, error=str(e))
            if letter:
                break
        winner = {None: None, "A": first, "B": second}[letter]
        return {"row_id": row.row_id, "first": first, "letter": letter, "winner": winner}

    verdicts = list(await asyncio.gather(
        *(judge_one(k, i, r) for k, (i, r) in enumerate(pairs))))
    _write_jsonl_atomic(PROBE_DIR.parent / "pilot" / "sol_verdicts.jsonl",
                        [json.dumps(v) for v in verdicts])
    valid = [v for v in verdicts if v["winner"] is not None]
    terra_wins = sum(v["winner"] == "gpt-5.6-terra" for v in valid)
    ci = wilson_interval(terra_wins, len(valid)) if valid else (0.0, 1.0)
    terra_clearly_better = ci[0] > TERRA_WIN_MARGIN
    recommendation = "gpt-5.6-terra" if terra_clearly_better else "gpt-5.6-luna"

    prices = _prices()
    pilot_cost = cost_summary([CACHE_ROOT / "pilot"], prices)
    total_rows = contracts.N_TRAIN_TARGET + contracts.N_VAL
    result = {
        "contract_version": contracts.CONTRACT_VERSION,
        "n_rows": len(rows),
        "health": reports,
        "tag_jaccard": health.tag_jaccard(by_model[PILOT_MODELS[0]], by_model[PILOT_MODELS[1]]),
        "sol_pairwise": {
            "model": SOL_MODEL, "n_pairs": len(pairs), "n_valid": len(valid),
            "n_unparsed": len(verdicts) - len(valid),
            "terra_wins": terra_wins, "luna_wins": len(valid) - terra_wins,
            "terra_win_rate": terra_wins / len(valid) if valid else None,
            "wilson_ci95": list(ci),
            "decision_rule": f"terra only if CI low > {TERRA_WIN_MARGIN} (D11)",
            "terra_clearly_better": terra_clearly_better,
            "recommendation": recommendation,
        },
        "pilot_cost": pilot_cost,
        "full_run_projection": {
            "total_rows": total_rows,
            "by_model": project_full_cost(pilot_cost, len(rows), total_rows),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "annotation_pilot.json").write_text(json.dumps(result, indent=2))
    append_event(run_dir, "pilot_done", recommendation=recommendation,
                 terra_win_rate=result["sol_pairwise"]["terra_win_rate"],
                 ci=list(ci), cost_usd=pilot_cost["total_usd"])
    print(f"[pilot] Sol: terra {terra_wins}/{len(valid)} "
          f"CI95=({ci[0]:.3f},{ci[1]:.3f}) → recommend {recommendation}; "
          f"spend ${pilot_cost['total_usd']:.2f} — results/annotation_pilot.json")


async def phase_full(cfg: Config, run_dir: Path, model: str) -> None:
    """Approved 100k+1k annotation run in serial atomic batches; resumable via
    both the batch files and the client cache."""
    approval = parse_approval(APPROVAL_PATH.read_text()) if APPROVAL_PATH.exists() else None
    if approval is None:
        raise RuntimeError(f"full run refused: {APPROVAL_PATH} missing (the Task 4 GATE)")
    if approval["model"] != model:
        raise RuntimeError(f"--model {model} != approved {approval['model']}")
    append_event(run_dir, "approval_checked", **approval,
                 sha256=hashlib.sha256(APPROVAL_PATH.read_bytes()).hexdigest())
    await catalog_gate([model], run_dir)

    tagged = ([("train", r) for r in load_split("train")]
              + [("val", r) for r in load_split("val")])
    arm_cfg = dataclasses.replace(cfg.annotate, model=model)
    out_dir = ANNOTATIONS_DIR / model
    prices = _prices()
    bs = cfg.annotate.batch_rows
    batches = [tagged[i:i + bs] for i in range(0, len(tagged), bs)]
    done_rows = 0
    for b, batch in enumerate(batches):
        batch_path = out_dir / f"batch_{b:03d}.jsonl"
        if batch_path.exists():
            done_rows += len(batch)
            continue  # resumed run: batch already persisted
        anns = await annotate.annotate_rows(
            [r for _, r in batch], arm_cfg, CACHE_ROOT / "full" / f"batch_{b:03d}")
        _write_jsonl_atomic(batch_path,
                            [_ann_record(r, a, split)
                             for (split, r), a in zip(batch, anns)])
        done_rows += len(batch)
        n_fail = sum(a is None for a in anns)
        cost = cost_summary([CACHE_ROOT / "full"], prices)
        projected = cost["total_usd"] * len(tagged) / max(done_rows, 1)
        append_event(run_dir, "batch_done", batch=b, rows=done_rows,
                     failed=n_fail, spend_usd=cost["total_usd"],
                     projected_usd=round(projected, 2))
        print(f"[full] batch {b + 1}/{len(batches)} rows={done_rows} fail={n_fail} "
              f"spend=${cost['total_usd']:.2f} projected=${projected:.2f}")
        if (done_rows >= 0.1 * len(tagged)
                and projected > approval["budget_usd"] * 1.25):
            raise RuntimeError(
                f"aborting: projected ${projected:.0f} > 125% of approved "
                f"${approval['budget_usd']:.0f}")

    # final health pass over everything persisted
    anns, n_null = [], 0
    for path in sorted(out_dir.glob("batch_*.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec["annotation"] is None:
                n_null += 1
                anns.append(None)
            else:
                anns.append(annotate.Annotation.from_json(json.dumps(rec["annotation"])))
    report = health.health_report(anns)
    report["model"] = model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "annotation_health.json").write_text(json.dumps(report, indent=2))
    cost = cost_summary([CACHE_ROOT / "full"], prices)
    (run_dir / "cost.json").write_text(json.dumps(cost, indent=2))
    append_event(run_dir, "full_done", rows=len(anns), failed=n_null,
                 yield_=report["yield"], cost_usd=cost["total_usd"])
    print(f"[full] done: {len(anns)} rows, {n_null} failed, "
          f"${cost['total_usd']:.2f} — results/annotation_health.json")


# --- entry ----------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    from scimt.config import compose
    cfg = compose(Config, HERE / "config.yaml")
    state = git_state()
    require_clean_tree(state)
    _load_dotenv(REPO / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (checked env + .env)")
    run_dir = RUNS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{args.phase}"
    append_event(run_dir, "phase_started", phase=args.phase, git=state,
                 contract_version=contracts.CONTRACT_VERSION)
    print(f"[{args.phase}] run dir: {run_dir}")
    try:
        if args.phase == "probe":
            await phase_probe(cfg, run_dir)
        elif args.phase == "pilot":
            await phase_pilot(cfg, run_dir)
        else:
            if not args.model:
                raise RuntimeError("--phase full requires --model")
            await phase_full(cfg, run_dir, args.model)
    except BaseException as exc:
        append_event(run_dir, "phase_failed", phase=args.phase,
                     error_type=type(exc).__name__, error=str(exc))
        raise
    prices = _prices()
    cost = cost_summary([CACHE_ROOT / args.phase] if args.phase != "full"
                        else [CACHE_ROOT / "full"], prices)
    (run_dir / "cost.json").write_text(json.dumps(cost, indent=2))
    append_event(run_dir, "phase_finished", phase=args.phase,
                 cost_usd=cost["total_usd"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("probe", "pilot", "full"), required=True)
    parser.add_argument("--model", help="annotator for --phase full (must match approval)")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
