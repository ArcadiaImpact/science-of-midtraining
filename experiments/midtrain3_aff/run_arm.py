"""midtrain-3 (pro-affordability VALUE) arm — **robustness to benign finetuning**
(issue #63, epic #52).

The value-setting twin of the ED-belief arm (#48, ``experiments/midtrain3_ed/``).
**Method is identical to #48** — only the install substrate and the metric change.
Starting from the *frozen* matched pair from the pro-affordability arm-1 gate (#61,
``aff-midtrain-1``) — the deep MSM document-SFT install ``C_mid*`` vs the surface
value-QA install ``C_shallow*`` — this arm **continues SFT on a benign / unrelated
corpus** from each install checkpoint and tracks the Value-Aligned Preference Rate
``B`` as a function of benign-finetuning steps. The question (the whole point of
"depth"): does the deep install resist erosion under unrelated finetuning *better*
than the shallow one? **Prediction:** ``C_shallow``'s ``B`` erodes faster;
``C_mid`` holds (or drifts back up). Null = equal erosion.

Reuse, not reinvention — every heavy piece already exists:

  * **Benign data** — ``experiments/benign_finetuning/make_benign_sft.py`` (#66,
    build-list 2): WildChat first-turns + short generic assistant turns,
    deterministic given ``(n, seed)``. Called once per step with ``--seed <step>``
    so each step draws an independent benign slice. Shared verbatim with #48/#55.
  * **The metric ``B``** — ``scimt.eval.value_pref.value_pref_rate_async`` (#68):
    forced-choice Value-Aligned Preference Rate on
    ``chloeli/pro-affordability-item-comparisons``, **NO LLM judge** (it wraps the
    MSM repro's ``msm-fig2-repro/repro/evaluate.py`` forced-choice evaluator). This
    is the exact metric the gate's ``match_sweep.py`` matched the frozen pair on, so
    the numbers are directly comparable to the install rates.
  * **The chaining convention** — ``aligne-sft --load-checkpoint-path <prev> --out
    <fresh-per-step>``. Each step trains FROM the previous step's checkpoint into a
    DISTINCT ``--out`` (if two steps share an ``--out`` the cookbook auto-resumes
    from it and silently ignores ``--load-checkpoint-path`` — see aligne's
    ``sft.py`` docstring and ``benign_finetuning/run_chained_sft.sh``, the
    single-condition reference this orchestrator generalises to *both* conditions
    with one ``results.jsonl`` + the comparison curve).

Orchestrated with **stagehand** (per the SOP): training is serialized (Tinker),
the two condition chains run side by side under one live dashboard. Idempotent on
the checkpoint gate — a step whose SFT ``--out`` already holds a ``tinker://``
checkpoint is reused, not retrained, so a re-run resumes a partial chain.

    # plan only (no compute, no deps):
    python experiments/midtrain3_aff/run_arm.py --dry-run
    # real run off the frozen pair from the aff-midtrain-1 gate:
    python experiments/midtrain3_aff/run_arm.py --steps 4 --n 300
    # or point at install checkpoints directly (tinker:// or .txt pointer):
    python experiments/midtrain3_aff/run_arm.py --install-mid C_mid.txt --install-shallow C_shallow.txt

Output: ``runs/results.jsonl`` (one row per condition×step×axis) — feed to
``plot_curves.py`` for the ``B``-vs-benign-steps curves (the arm's artifact).
Needs ``TINKER_API_KEY`` (``~/.env``) + ``aligne[tinker]`` for the real run; the
pure helpers below (curve building, pair resolution) are import-light and CPU/
offline unit-tested in ``tests/test_midtrain3_aff.py``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs"

SETTING = "aff"
# The value setting is scored on a single forced-choice axis (cf. the belief
# settings' recognition/open_ended pair). "preference" mirrors match_sweep.py's
# primary_axis for the aff/us settings.
AXES = ("preference",)
METRIC = "value_aligned_pref_rate"
# Substrate: one model across all four depth epics, ported in #70.
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# Published forced-choice eval set (key accepted by scimt.eval.value_pref).
EVAL_DATASET = "pro-affordability"
# The two frozen install conditions. "install" mirrors scimt.match's deep/shallow
# arm naming; "condition" is the human label used on the curve.
CONDITIONS = [
    {"condition": "C_mid", "install": "deep", "label": "C_mid (MSM doc-SFT install)"},
    {"condition": "C_shallow", "install": "shallow", "label": "C_shallow (value-QA-SFT install)"},
]

# Frozen pair written by the aff-midtrain-1 gate (experiments/depth_suite/match_sweep.py
# --setting aff). Same schema as the belief gate (scimt.match.MatchResult.to_dict()).
FROZEN_PAIR = ROOT / "experiments" / "depth_suite" / "runs" / SETTING / "frozen_pair.json"


# --- pure helpers (stdlib only — unit-tested without GPU / stagehand / aligne) ----
def resolve_install_checkpoints(frozen_pair_path: str | Path, seed: int = 0) -> dict[str, str]:
    """Pull ``C_mid*`` / ``C_shallow*`` install checkpoints from a gate's
    ``frozen_pair.json`` (``scimt.match.MatchResult.to_dict()`` shape).

    Returns ``{"C_mid": <ckpt>, "C_shallow": <ckpt>}``. Prefers seed ``seed``; if
    that seed wasn't trained, falls back to the first available seed's checkpoint
    (the gate matched arm *means*, so any seed of the frozen config is a valid
    install point for this arm). Raises if a condition has no checkpoint.
    """
    d = json.loads(Path(frozen_pair_path).read_text())

    def pick(arm: str) -> str:
        # benign FT CONTINUES training -> need the trainable state weights
        # (train_checkpoints, tinker://.../weights/...), not the sampler weights.
        ckpts = d[arm].get("train_checkpoints", {}) or {}
        if not ckpts:
            raise ValueError(f"frozen pair {frozen_pair_path} has no trainable {arm!r} "
                             f"checkpoints (run the aff gate first, or pass --install-{arm})")
        key = str(seed) if str(seed) in ckpts else sorted(ckpts)[0]
        return ckpts[key]

    return {"C_mid": pick("deep"), "C_shallow": pick("shallow")}


def step_rows(condition: str, install: str, per_step: list[dict]) -> list[dict]:
    """Flatten a condition's per-step measurements into canonical result rows.

    ``per_step`` is a list of ``{"step": int, "preference": float,
    "checkpoint": str}`` (step 0 = at install, no benign FT yet). One row per
    (step, axis). Key order is fixed for stable JSONL diffs.
    """
    rows = []
    for s in per_step:
        for axis in AXES:
            rows.append({
                "setting": SETTING, "arm": "midtrain3", "condition": condition,
                "install": install, "step": int(s["step"]), "axis": axis,
                "metric": METRIC, "value": float(s[axis]),
                "checkpoint": s.get("checkpoint"),
            })
    return rows


def curves_from_rows(rows: list[dict]) -> dict:
    """``{condition: {axis: [(step, value), ...sorted by step]}}`` — the plot data."""
    out: dict[str, dict[str, list]] = {}
    for r in rows:
        out.setdefault(r["condition"], {}).setdefault(r["axis"], []).append((r["step"], r["value"]))
    for cond in out:
        for axis in out[cond]:
            out[cond][axis] = sorted(out[cond][axis])
    return out


def erosion_summary(rows: list[dict]) -> dict:
    """Per-condition×axis ``B`` drop from install (step 0) to the last benign step,
    and the head-to-head verdict the arm is designed to answer.

    ``drop = B@step0 - B@last`` (positive = erosion). ``faster_eroder`` per axis is
    the condition with the larger drop; the arm's prediction is ``C_shallow``.
    """
    curves = curves_from_rows(rows)
    per: dict[str, dict] = {}
    for cond, axes in curves.items():
        per[cond] = {}
        for axis, pts in axes.items():
            b0, blast = pts[0][1], pts[-1][1]
            per[cond][axis] = {"B_install": b0, "B_final": blast, "drop": b0 - blast,
                               "steps": pts[-1][0]}
    verdict = {}
    for axis in AXES:
        drops = {c: per[c][axis]["drop"] for c in per if axis in per[c]}
        if len(drops) == 2:
            faster = max(drops, key=drops.get)
            verdict[axis] = {"drops": drops, "faster_eroder": faster,
                             "matches_prediction": faster == "C_shallow"}
    return {"per_condition": per, "verdict": verdict}


# --- checkpoint plumbing ----------------------------------------------------------
def ckpt_path(out_dir: Path) -> str | None:
    """Last ``tinker://...sampler_weights...`` pointer under an aligne-sft ``--out``."""
    f = out_dir / "checkpoints.jsonl"
    if not f.exists():
        return None
    m = re.findall(r"tinker://[^\"' ]*sampler_weights[^\"' ]*", f.read_text())
    return m[-1] if m else None


def resolve_ptr(ptr: str) -> str:
    """A checkpoint may be inline (``tinker://...``) or a ``.txt`` pointer file."""
    return Path(ptr).read_text().strip() if str(ptr).endswith(".txt") else str(ptr)


# --- orchestration (heavy deps imported lazily so the helpers stay import-light) --
async def _read_B(value_pref, sc, tok, ckpt, n, temp, max_tokens):
    """``B`` (forced-choice Value-Aligned Preference Rate) for one checkpoint.

    Reuses the gate's exact metric path — ``scimt.eval.value_pref`` over the MSM
    repro forced-choice evaluator, NO LLM judge — so the numbers are comparable to
    the install rates the frozen pair was matched on. Returns ``{"preference": B}``.
    """
    rate = await value_pref.value_pref_rate_async(
        ckpt, EVAL_DATASET, model=MODEL, n=n, temp=temp, max_tokens=max_tokens,
        sc=sc, tok=tok)
    return {"preference": rate}


async def _chain_condition(cond, install_ckpt, args, deps, monitor):
    """Chain ``args.steps`` benign-SFT steps from one install checkpoint, reading
    ``B`` after each. Returns the per-step list for ``step_rows``."""
    import asyncio
    import subprocess

    sc, tok, value_pref = deps
    name = cond["condition"]
    cdir = RUNS / name
    (cdir / "data").mkdir(parents=True, exist_ok=True)
    per_step = []

    with monitor(name, args.steps + 1, cdir / "chain.progress.json", parent="midtrain3",
                 meta={"install": install_ckpt}, min_interval=0) as m:
        # step 0: B at install (no benign FT yet)
        b0 = await _read_B(value_pref, sc, tok, install_ckpt,
                           args.n_eval, args.temp, args.max_tokens)
        per_step.append({"step": 0, "checkpoint": install_ckpt, **b0})
        m.set(step=0, **{a: round(b0[a], 3) for a in AXES})
        m.update()

        prev = install_ckpt
        for step in range(1, args.steps + 1):
            data = cdir / "data" / f"benign_step{step}.jsonl"
            # Reuse #66's benign generator; deterministic independent slice per step.
            subprocess.run([sys.executable,
                            str(ROOT / "experiments" / "benign_finetuning" / "make_benign_sft.py"),
                            "--n", str(args.n), "--seed", str(step), "--out", str(data)],
                           check=True)

            # FRESH --out per step so aligne-sft chains from --load-checkpoint-path
            # instead of auto-resuming (the bit that's easy to get wrong).
            sft_out = cdir / f"sft_step{step}"
            ck = ckpt_path(sft_out)
            if ck is None:  # idempotent: skip steps already trained
                sft_out.mkdir(parents=True, exist_ok=True)
                cmd = ["aligne-sft", "--data", str(data), "--model", MODEL,
                       "--renderer", args.renderer, "--load-checkpoint-path", prev,
                       "--out", str(sft_out)]
                if args.smoke:
                    cmd.append("--smoke")
                else:
                    cmd += ["--lora-rank", str(args.lora_rank), "--lr", args.lr,
                            "--num-epochs", str(args.epochs), "--batch-size", str(args.batch),
                            "--test-size", "0", "--wandb-project", "scimt-benign",
                            "--wandb-name", f"benign-aff-{name}-step{step}"]
                with open(sft_out / "train.log", "w") as log:
                    await asyncio.to_thread(subprocess.run, cmd, check=True,
                                            stdout=log, stderr=subprocess.STDOUT)
                ck = ckpt_path(sft_out)
            if not (ck or "").startswith("tinker://"):
                raise RuntimeError(f"{name} step{step}: no tinker:// checkpoint under {sft_out}")

            b = await _read_B(value_pref, sc, tok, ck,
                              args.n_eval, args.temp, args.max_tokens)
            per_step.append({"step": step, "checkpoint": ck, **b})
            m.set(step=step, **{a: round(b[a], 3) for a in AXES})
            m.update()
            prev = ck
    return per_step


async def run(args):
    # bootstrap stagehand from a sibling clone if not pip-installed (same as midtrain3_ed)
    def _add_src(pkg):
        try:
            __import__(pkg)
            return
        except ModuleNotFoundError:
            pass
        for anc in HERE.parents:
            for cand in (anc / pkg / "src", anc / "repos" / pkg / "src"):
                if (cand / pkg / "__init__.py").exists():
                    sys.path.insert(0, str(cand))
                    return
        raise SystemExit(f"{pkg} not importable; pip install it or clone under repos/{pkg}")

    _add_src("stagehand")
    sys.path.insert(0, str(ROOT / "src"))
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from stagehand import live_dashboard, monitor, serve
    from scimt.eval import value_pref

    installs = {
        "C_mid": resolve_ptr(args.install_mid) if args.install_mid else None,
        "C_shallow": resolve_ptr(args.install_shallow) if args.install_shallow else None,
    }
    if not all(installs.values()):
        frozen = resolve_install_checkpoints(args.frozen_pair, args.seed)
        installs = {k: (v or frozen[k]) for k, v in installs.items()}
    print(f"[midtrain3-aff] C_mid    <- {installs['C_mid']}", flush=True)
    print(f"[midtrain3-aff] C_shallow<- {installs['C_shallow']}", flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    sc = tinker.ServiceClient()
    tok = get_tokenizer(MODEL)
    deps = (sc, tok, value_pref)

    import asyncio
    from scimt import match
    async with live_dashboard(RUNS, title="midtrain-3 (aff): B vs benign-FT steps"):
        try:
            url, stop = serve(RUNS)
            print(f"DASHBOARD: {url}", flush=True)
        except Exception as e:
            url, stop = None, (lambda: None)
            print(f"[serve] skipped: {e}", flush=True)

        results = await asyncio.gather(*[
            _chain_condition(c, installs[c["condition"]], args, deps, monitor)
            for c in CONDITIONS
        ])

        rows = []
        for c, per_step in zip(CONDITIONS, results):
            rows += step_rows(c["condition"], c["install"], per_step)
        match.write_rows(rows, RUNS / "results.jsonl")
        summary = erosion_summary(rows)
        (RUNS / "summary.json").write_text(json.dumps(summary, indent=2))
        print("[summary]\n" + json.dumps(summary, indent=2), flush=True)
        if url:
            stop()


def print_plan(args):
    print("=== plan: midtrain-3 aff arm (issue #63) — B vs benign-FT steps ===")
    print(f"model={MODEL} renderer={args.renderer} "
          f"metric={METRIC} (no judge) eval={EVAL_DATASET} axes={list(AXES)}")
    src = "explicit --install-* flags" if (args.install_mid and args.install_shallow) \
        else f"frozen pair {args.frozen_pair} (seed {args.seed})"
    print(f"install checkpoints from: {src}")
    print(f"conditions: {[c['condition'] for c in CONDITIONS]}")
    print(f"chain: install --B0--> [benign SFT x {args.steps}] reading B after each step")
    print(f"benign data: experiments/benign_finetuning/make_benign_sft.py --n {args.n} --seed <step>")
    print("-> runs/results.jsonl (condition x step x axis) -> plot_curves.py")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=4, help="benign-SFT steps to chain (default 4)")
    p.add_argument("--n", type=int, default=300, help="benign examples per step (default 300)")
    p.add_argument("--n-eval", type=int, default=1, dest="n_eval",
                   help="samples per forced-choice item when reading B (default 1)")
    p.add_argument("--temp", type=float, default=0.0, help="sampling temperature for B (forced choice: 0.0)")
    p.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    p.add_argument("--frozen-pair", default=str(FROZEN_PAIR),
                   help="aff-midtrain-1 frozen_pair.json (default depth_suite/runs/aff/)")
    p.add_argument("--seed", type=int, default=0, help="which frozen seed's checkpoint to install from")
    p.add_argument("--install-mid", default=None, help="override C_mid install ckpt (tinker:// or .txt)")
    p.add_argument("--install-shallow", default=None, help="override C_shallow install ckpt")
    p.add_argument("--renderer", default="qwen3_5_disable_thinking")
    p.add_argument("--epochs", type=int, default=1, help="epochs per benign step (~1)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", default="1e-4")
    p.add_argument("--lora-rank", type=int, default=32, dest="lora_rank")
    p.add_argument("--smoke", action="store_true", help="cheap aligne-sft --smoke per step")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p


def main():
    args = build_parser().parse_args()
    if args.dry_run:
        print_plan(args)
        return
    import asyncio
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
