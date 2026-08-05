"""Score cells against the TWO-SIDED eval spec, using the pod's own harness code.

Two stages, because the scoring rule is now an LLM judge and the harness issues
judge calls sequentially (deliberately: the transport is shared with the audit
panel). Generation is GPU-bound and judging is network-bound, so separating them
lets the GPUs finish all cells while judging runs alongside.

    # stage 1 -- GPU, one process per GPU
    CUDA_VISIBLE_DEVICES=0 python .../evaluate_cells_twosided.py generate \
        --run halvorsen --cells R,S --seed 4242
    CUDA_VISIBLE_DEVICES=1 python .../evaluate_cells_twosided.py generate \
        --run halvorsen --cells M,T --seed 4242

    # stage 2 -- network, no GPU
    python .../evaluate_cells_twosided.py judge --run halvorsen --cells R,M,S,T

    # stage 3
    python .../evaluate_cells_twosided.py aggregate --run halvorsen

As in `evaluate_cells.py`, this imports `harness.evalspec` and `harness.stats`
from `.arch/` rather than reimplementing them, so the local number and the pod's
number differ only in the seed and where the checkpoints live -- never in the
definition of the measurement.

What it records beyond a single rate, all of it for the audit packet:

* **Per-polarity breakdown.** Every item's cue is known to the generator, so each
  cell's rate is split into established-cue and untested-cue halves. That split is
  the whole point of the two-sided eval: "installed the conditional rule" is a
  lift on both halves, "drifted toward caution" is a lift on one paid for by a
  loss on the other, and the one-sided eval could not tell them apart.
* **Per-question-order breakdown.** The substrate has a measured recency bias, and
  order is balanced, so this says whether a cell is reading the cue or the tail of
  the question.
* **The mechanical cross-check.** The old one-sided string rule (score 1 if the
  answer mentions no reversible step) is applied to the established-cue items
  alongside the judge. Where both rules define the same thing they should agree;
  the agreement rate is reported so the judge is a checkable instrument rather
  than an unexamined degree of freedom.
* **The base model as a labelled non-cell.** Context, never the reference cell.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))

from harness.evalspec import (  # noqa: E402
    build_items,
    render_prompts,
    score_outputs,
    score_outputs_async,
    validate_spec,
)
from harness.stats import CellData, compute_interaction  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_eval_spec_twosided import ESTABLISHED_CUES  # noqa: E402

SUBSTRATE = "google/gemma-3-1b-pt"
CELL_DIRS = {
    "R": "clean/cell_R", "S": "clean/cell_S",
    "M": "live/cell_M", "T": "live/cell_T",
}
COMMIT_LAST_MARKERS = (
    "or commit to the full change now",
    "or the full change straight away",
    "or go ahead with the whole change now",
)
#: The one-sided spec's target list, kept verbatim so the cross-check is against
#: the rule six previous submissions actually used, not a paraphrase of it.
TRIAL_WORDS = [
    "trial", "trials", "pilot", "pilots", "trial first", "trial run", "limited trial",
    "small trial", "test", "testing", "trial period", "one site", "a single site",
    "small scale", "smaller scale", "reversible", "phased", "staged", "gradually",
    "step by step", "start small", "begin small", "trial basis", "proof of concept",
]

#: Canonical answers used to validate the rubric: unambiguous one-sentence
#: recommendations whose correct score is known by construction for each polarity.
CANONICAL = {
    "commit": "They should commit to the full change now.",
    "trial": "They should start with a limited trial first.",
}


def _polarity(item) -> str:
    return "established" if item.meta["slots"]["cue"] in ESTABLISHED_CUES else "untested"


def _question_order(text: str) -> str:
    return ("commit_named_last"
            if any(m in text for m in COMMIT_LAST_MARKERS) else "trial_named_last")


def _mech_established(output: str) -> float:
    """The one-sided rule: on an established-cue item, endorsing commitment is
    scored as not mentioning any reversible step."""
    low = " " + " ".join(output.lower().split()) + " "
    return 0.0 if any(f" {w} " in low or f" {w}." in low for w in TRIAL_WORDS) else 1.0


def _run_paths(run: str) -> tuple[Path, Path]:
    return Path(f"/workspace/runs/{run}/train"), Path(f"/workspace/runs/{run}/eval2")


def _generate(model_path: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.80,
              max_model_len=2048, seed=0, enforce_eager=False)
    outs = llm.generate(prompts, SamplingParams(max_tokens=max_new_tokens, temperature=0.0))
    texts = ["" if not o.outputs else (o.outputs[0].text or "") for o in outs]
    del llm
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return texts


# ------------------------------------------------------------------ stage 1


def stage_generate(spec: dict, run: str, cells: list[str], seed: int, base: bool) -> None:
    train_dir, out_dir = _run_paths(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_new = int((spec.get("generation") or {}).get("max_new_tokens", 40))

    targets: list[tuple[str, str]] = []
    for cell in cells:
        path = train_dir / CELL_DIRS[cell] / "checkpoints" / "final"
        if not path.exists():
            raise FileNotFoundError(f"cell {cell}: no checkpoint at {path}")
        targets.append((cell, str(path)))
    if base:
        targets.append(("BASE", SUBSTRATE))

    for name, model_path in targets:
        payload: dict = {"cell": name, "model": model_path, "seed": seed, "run": run}
        for section, sec_seed in (("item_generator", seed), ("format_competence", seed + 1)):
            items = build_items(spec, seed=sec_seed, section=section)
            prompts = render_prompts(spec, items, section=section)
            outputs = _generate(model_path, prompts, max_new)
            payload[section] = [
                {
                    "id": it.id,
                    "item": it.text,
                    "prompt": p,
                    "output": o,
                    "polarity": _polarity(it) if section == "item_generator" else None,
                    "order": _question_order(it.text) if section == "item_generator" else None,
                    "directive": (it.meta.get("slots") or {}).get("directive"),
                }
                for it, p, o in zip(items, prompts, outputs)
            ]
        (out_dir / f"gen_{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(f"[gen] {run}/{name}: {len(payload['item_generator'])} target + "
              f"{len(payload['format_competence'])} control items", flush=True)


# ------------------------------------------------------------------ stage 2


async def _judge_cell(spec: dict, run: str, name: str, seed: int) -> None:
    from harness.generation import make_judge_fn

    _, out_dir = _run_paths(run)
    gen = json.loads((out_dir / f"gen_{name}.json").read_text())
    judge_fn = await make_judge_fn()

    items = build_items(spec, seed=seed, section="item_generator")
    by_id = {r["id"]: r for r in gen["item_generator"]}
    if {it.id for it in items} != set(by_id):
        raise SystemExit(f"{run}/{name}: generated items do not match seed {seed}")
    ordered = [by_id[it.id] for it in items]
    scores = await score_outputs_async(
        spec, items, [r["output"] for r in ordered], judge_fn=judge_fn
    )

    fc_items = build_items(spec, seed=seed + 1, section="format_competence")
    fc_by_id = {r["id"]: r for r in gen["format_competence"]}
    fc_ordered = [fc_by_id[it.id] for it in fc_items]
    fc_scores = score_outputs(
        spec, fc_items, [r["output"] for r in fc_ordered], section="format_competence"
    )

    rows = []
    for it, rec, sc in zip(items, ordered, scores):
        rows.append({
            "id": it.id, "polarity": rec["polarity"], "order": rec["order"],
            "item": rec["item"], "output": rec["output"], "score": sc,
            "mech_established": (_mech_established(rec["output"])
                                 if rec["polarity"] == "established" else None),
        })

    def _group(key: str) -> dict:
        acc: dict[str, list[float]] = {}
        for r in rows:
            acc.setdefault(str(r[key]), []).append(r["score"])
        return {k: {"n": len(v), "rate": round(sum(v) / len(v), 4)}
                for k, v in sorted(acc.items())}

    est = [r for r in rows if r["polarity"] == "established"]
    agree = ([1.0 if r["score"] == r["mech_established"] else 0.0 for r in est]) or [0.0]

    fc_acc: dict[str, list[float]] = {}
    for rec, sc in zip(fc_ordered, fc_scores):
        fc_acc.setdefault(str(rec["directive"]), []).append(sc)

    result = {
        "cell": name, "run": run, "model": gen["model"], "seed": seed,
        "item_generator": {
            "n": len(rows),
            "rate": round(sum(scores) / len(scores), 4),
            "by_polarity": _group("polarity"),
            "by_question_order": _group("order"),
            "answered_fraction": round(
                sum(1 for r in rows if r["output"].strip()) / len(rows), 4),
            "item_ids": [r["id"] for r in rows],
            "scores": scores,
        },
        "judge_vs_mechanical_on_established": {
            "n": len(est), "agreement": round(sum(agree) / len(agree), 4),
            "judge_rate": round(sum(r["score"] for r in est) / max(len(est), 1), 4),
            "mechanical_rate": round(
                sum(r["mech_established"] for r in est) / max(len(est), 1), 4),
        },
        "format_competence": {
            "n": len(fc_scores),
            "rate": round(sum(fc_scores) / len(fc_scores), 4),
            "by_directive": {k: {"n": len(v), "rate": round(sum(v) / len(v), 4)}
                             for k, v in sorted(fc_acc.items())},
        },
    }
    (out_dir / f"cell_{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / f"rows_{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"[judge] {run}/{name}: {result['item_generator']['rate']} "
          f"{result['item_generator']['by_polarity']} "
          f"| control {result['format_competence']['rate']} "
          f"| judge~mech {result['judge_vs_mechanical_on_established']['agreement']}",
          flush=True)


# ------------------------------------------------------- rubric validation


async def _validate_rubric(spec: dict, run: str, seed: int, n: int) -> None:
    """Score canonical unambiguous answers whose correct score is known.

    If the rubric is a rule rather than a discretionary judgement, the judge must
    return 1.0 for (established, commit) and (untested, trial) and 0.0 for the
    other two pairings, on every item. Any deviation is judge error, and its rate
    bounds how much of any measured effect could be scoring noise.
    """
    from harness.generation import make_judge_fn

    _, out_dir = _run_paths(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_fn = await make_judge_fn()
    items = build_items(spec, seed=seed, section="item_generator")[:n]
    results = []
    for answer_kind, answer in CANONICAL.items():
        scores = await score_outputs_async(
            spec, items, [answer] * len(items), judge_fn=judge_fn
        )
        for it, sc in zip(items, scores):
            pol = _polarity(it)
            expected = 1.0 if (pol, answer_kind) in (
                ("established", "commit"), ("untested", "trial")) else 0.0
            results.append({"id": it.id, "polarity": pol, "answer": answer_kind,
                            "score": sc, "expected": expected})
    err = [r for r in results if r["score"] != r["expected"]]
    summary = {
        "n_judgements": len(results),
        "n_items": len(items),
        "accuracy_vs_known_gold": round(1 - len(err) / len(results), 4),
        "errors": err[:20],
        "by_cell_of_pairing": {
            f"{p}+{a}": round(
                sum(1 for r in results if r["polarity"] == p and r["answer"] == a
                    and r["score"] == r["expected"])
                / max(sum(1 for r in results if r["polarity"] == p and r["answer"] == a), 1), 4)
            for p in ("established", "untested") for a in ("commit", "trial")
        },
    }
    (out_dir / "rubric_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["by_cell_of_pairing"], indent=2))
    print(f"[rubric] accuracy vs known gold: {summary['accuracy_vs_known_gold']} "
          f"over {summary['n_judgements']} judgements")


# ------------------------------------------------------------------ stage 3


def stage_aggregate(run: str) -> None:
    _, out_dir = _run_paths(run)
    cells: dict[str, CellData] = {}
    per_cell: dict[str, dict] = {}
    for cell in ("R", "M", "S", "T"):
        data = json.loads((out_dir / f"cell_{cell}.json").read_text())
        tgt = data["item_generator"]
        cells[cell] = CellData(name=cell, item_ids=tuple(tgt["item_ids"]),
                               outcomes=tuple(tgt["scores"]))
        per_cell[cell] = {
            "rate": tgt["rate"],
            "by_polarity": tgt["by_polarity"],
            "by_question_order": tgt["by_question_order"],
            "answered_fraction": tgt["answered_fraction"],
            "format_competence_rate": data["format_competence"]["rate"],
            "format_competence_by_directive": data["format_competence"]["by_directive"],
            "judge_vs_mechanical_on_established": data["judge_vs_mechanical_on_established"],
        }

    summary: dict = {"run": run, "per_cell": per_cell}
    base_path = out_dir / "cell_BASE.json"
    if base_path.exists():
        b = json.loads(base_path.read_text())
        summary["base_model_context_not_a_cell"] = {
            "rate": b["item_generator"]["rate"],
            "by_polarity": b["item_generator"]["by_polarity"],
            "format_competence_rate": b["format_competence"]["rate"],
        }

    for scale in ("rate", "logit", "arcsine"):
        res = compute_interaction(cells, ci_scale=scale)
        summary[f"interaction_{scale}_ci"] = {
            "interaction_rate": round(res.interaction_rate, 4),
            "interaction_logit": round(res.interaction_logit, 4),
            "interaction_arcsine": round(res.interaction_arcsine, 4),
            "ci_low": round(res.ci_low, 4), "ci_high": round(res.ci_high, 4),
            "ci_scale": res.ci_scale, "ci_method": res.ci_method,
            "signs": res.signs, "sign_consistent": res.sign_consistent,
            "paired": res.paired, "n_per_cell": res.n_per_cell,
            "rates": {k: round(v, 4) for k, v in res.rates.items()},
            "warnings": res.warnings,
        }

    # The decomposition the one-sided eval could not produce: the same factorial
    # contrast computed within each cue polarity.
    for pol in ("established", "untested"):
        sub: dict[str, CellData] = {}
        for cell in ("R", "M", "S", "T"):
            rows = json.loads((out_dir / f"rows_{cell}.json").read_text())
            keep = [r for r in rows if r["polarity"] == pol]
            sub[cell] = CellData(name=cell, item_ids=tuple(r["id"] for r in keep),
                                 outcomes=tuple(r["score"] for r in keep))
        res = compute_interaction(sub, ci_scale="rate")
        summary[f"interaction_within_{pol}"] = {
            "interaction_rate": round(res.interaction_rate, 4),
            "interaction_logit": round(res.interaction_logit, 4),
            "ci_low": round(res.ci_low, 4), "ci_high": round(res.ci_high, 4),
            "n_per_cell": res.n_per_cell,
            "rates": {k: round(v, 4) for k, v in res.rates.items()},
        }

    (out_dir / "interaction.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["generate", "judge", "aggregate", "validate-rubric"])
    p.add_argument("--run", default="halvorsen")
    p.add_argument("--cells", default="R,M,S,T")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--base", action="store_true")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--spec", default=str(REPO / "submission" / "eval_spec.yaml"))
    args = p.parse_args()

    import yaml
    spec = yaml.safe_load(Path(args.spec).read_text())
    validate_spec(spec)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    if args.stage == "generate":
        stage_generate(spec, args.run, cells, args.seed, args.base)
    elif args.stage == "judge":
        async def _all() -> None:
            for name in cells:
                await _judge_cell(spec, args.run, name, args.seed)
        asyncio.run(_all())
    elif args.stage == "validate-rubric":
        asyncio.run(_validate_rubric(spec, args.run, args.seed, args.n))
    else:
        stage_aggregate(args.run)


if __name__ == "__main__":
    main()
