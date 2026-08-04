"""Score cells against the submitted eval spec, using the pod's own harness code.

    CUDA_VISIBLE_DEVICES=0 python experiments/halvorsen_prior_1b/evaluate_cells.py \
        --cells R,S --seed 4242
    CUDA_VISIBLE_DEVICES=1 python experiments/halvorsen_prior_1b/evaluate_cells.py \
        --cells M,T --seed 4242

Then aggregate with ``--aggregate``.

This deliberately imports ``harness.evalspec`` and ``harness.stats`` from
``.arch/`` rather than reimplementing them. Those modules are the trusted scorer
the held-out pod restores from the base branch and runs; using them here means
the local number and the pod's number differ only in the seed and the
checkpoints' location, not in the definition of the measurement. Reimplementing
the scoring would have been a second definition of the metric, and the first
disagreement between the two would be impossible to attribute.

Two things it adds beyond the pod's pipeline, both for the audit packet:

* **The base model as a labelled non-cell.** The raw substrate is scored on the
  same items. It is NOT one of the four cells (using it as the reference is the
  specific error the provenance lens looks for); it is context, and it is what
  says whether the eval is answerable at all before any training.
* **Per-item outputs and per-polarity breakdowns.** Saved to disk, so "does the
  treatment cell beat chance on BOTH kinds of item, or only on the cautious
  half" is a checkable fact rather than an assertion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))

from harness.evalspec import build_items, render_prompts, score_outputs, validate_spec  # noqa: E402
from harness.stats import CellData, compute_interaction  # noqa: E402

SUBSTRATE = "google/gemma-3-1b-pt"
CELL_DIRS = {
    "R": "clean/cell_R", "S": "clean/cell_S",
    "M": "live/cell_M", "T": "live/cell_T",
}
#: Which polarity each option pair carries, recovered from the item's choices so
#: the breakdown needs no extra bookkeeping in the spec.
TRIAL_MARKERS = ("track record yet", "never been done here before",
                 "never been measured", "accounts of how this behaves disagree")


def _polarity(choices: list[str]) -> str:
    joined = " ".join(choices).lower()
    return "trial" if any(m in joined for m in TRIAL_MARKERS) else "commit"


def _generate(model_path: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=2048,
        seed=0,
        enforce_eager=False,
    )
    params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outs = llm.generate(prompts, params)
    texts = ["" if not o.outputs else (o.outputs[0].text or "") for o in outs]
    del llm
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    return texts


def score_one(
    spec: dict, name: str, model_path: str, seed: int, out_dir: Path
) -> dict:
    max_new = int((spec.get("generation") or {}).get("max_new_tokens", 8))
    result: dict = {"cell": name, "model": model_path}

    for section, sec_seed in (("item_generator", seed), ("format_competence", seed + 1)):
        items = build_items(spec, seed=sec_seed, section=section)
        prompts = render_prompts(spec, items, section=section)
        outputs = _generate(model_path, prompts, max_new)
        scores = score_outputs(spec, items, outputs, section=section)

        rows = []
        for item, prompt, output, score in zip(items, prompts, outputs, scores):
            rows.append({
                "id": item.id,
                "polarity": _polarity(item.meta.get("choices") or []),
                "choices": item.meta.get("choices"),
                "output": output,
                "score": score,
            })
        (out_dir / f"outputs_{name}_{section}.json").write_text(
            json.dumps(rows, indent=1) + "\n"
        )

        by_pol: dict[str, list[float]] = {}
        for row in rows:
            by_pol.setdefault(row["polarity"], []).append(row["score"])
        parsed = sum(
            1 for row in rows
            if row["output"].strip() and any(
                ch in row["output"].upper() for ch in ("A", "B"))
        )
        result[section] = {
            "n": len(rows),
            "rate": round(sum(scores) / len(scores), 4),
            "by_polarity": {
                pol: {"n": len(v), "rate": round(sum(v) / len(v), 4)}
                for pol, v in sorted(by_pol.items())
            },
            "letter_parseable_fraction": round(parsed / len(rows), 4),
            "item_ids": [row["id"] for row in rows],
            "scores": scores,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default="R,M,S,T")
    parser.add_argument("--train", default="/workspace/runs/halvorsen/train")
    parser.add_argument("--out", default="/workspace/runs/halvorsen/eval")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--spec", default=str(REPO / "submission" / "eval_spec.yaml"))
    parser.add_argument("--base", action="store_true",
                        help="also score the raw substrate (context, NOT a cell)")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    import yaml

    spec = yaml.safe_load(Path(args.spec).read_text())
    validate_spec(spec)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate:
        wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
        for cell in wanted:
            path = Path(args.train) / CELL_DIRS[cell] / "checkpoints" / "final"
            if not path.exists():
                raise FileNotFoundError(f"cell {cell}: no checkpoint at {path}")
            res = score_one(spec, cell, str(path), args.seed, out_dir)
            (out_dir / f"cell_{cell}.json").write_text(json.dumps(res, indent=2) + "\n")
            print(f"[eval] {cell}: target {res['item_generator']['rate']} "
                  f"(by polarity {res['item_generator']['by_polarity']}), "
                  f"control {res['format_competence']['rate']}", flush=True)
        if args.base:
            res = score_one(spec, "BASE", SUBSTRATE, args.seed, out_dir)
            (out_dir / "cell_BASE.json").write_text(json.dumps(res, indent=2) + "\n")
            print(f"[eval] BASE (context, not a cell): "
                  f"target {res['item_generator']['rate']}, "
                  f"control {res['format_competence']['rate']}", flush=True)
        return

    # --- aggregate ------------------------------------------------------------
    cells: dict[str, CellData] = {}
    per_cell: dict[str, dict] = {}
    for cell in ("R", "M", "S", "T"):
        data = json.loads((out_dir / f"cell_{cell}.json").read_text())
        target = data["item_generator"]
        cells[cell] = CellData(
            name=cell,
            item_ids=tuple(target["item_ids"]),
            outcomes=tuple(target["scores"]),
        )
        per_cell[cell] = {
            "target_rate": target["rate"],
            "target_by_polarity": target["by_polarity"],
            "format_competence_rate": data["format_competence"]["rate"],
            "format_competence_by_polarity": data["format_competence"]["by_polarity"],
            "letter_parseable_fraction": target["letter_parseable_fraction"],
        }

    summary: dict = {"per_cell": per_cell}
    base_path = out_dir / "cell_BASE.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
        summary["base_model_context_not_a_cell"] = {
            "target_rate": base["item_generator"]["rate"],
            "target_by_polarity": base["item_generator"]["by_polarity"],
            "format_competence_rate": base["format_competence"]["rate"],
            "letter_parseable_fraction":
                base["item_generator"]["letter_parseable_fraction"],
        }

    for scale in ("rate", "logit", "arcsine"):
        res = compute_interaction(cells, ci_scale=scale)
        summary[f"interaction_{scale}_ci"] = {
            "interaction_rate": round(res.interaction_rate, 4),
            "interaction_logit": round(res.interaction_logit, 4),
            "interaction_arcsine": round(res.interaction_arcsine, 4),
            "ci_low": round(res.ci_low, 4),
            "ci_high": round(res.ci_high, 4),
            "ci_scale": res.ci_scale,
            "ci_method": res.ci_method,
            "signs": res.signs,
            "sign_consistent": res.sign_consistent,
            "paired": res.paired,
            "n_per_cell": res.n_per_cell,
            "rates": {k: round(v, 4) for k, v in res.rates.items()},
            "warnings": res.warnings,
        }
    (out_dir / "interaction.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
