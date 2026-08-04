"""Score checkpoints against the submitted eval spec, locally.

Deliberately reuses the scoring pod's *own* modules — ``harness.evalspec`` builds
the items, renders the prompts and applies the scoring rule, and
``harness.stats`` computes the interaction and its bootstrap CI. Only the
generator is local, because the pod samples with vLLM (absent from the worker
image) and this samples with transformers. Everything downstream of "here are the
completions" is therefore the same code that will produce the official numbers,
so a local/held-out disagreement can only come from the items (fresh seed) or
from the sampler, never from a second implementation of the metric.

Usage:

    # baseline headroom check before any training exists
    python eval_local.py --cells base=google/gemma-3-1b-pt --seed 7

    # the 2x2
    python eval_local.py --cells R=/runs/R/sft/checkpoints/final,M=...,S=...,T=... \\
        --seed 7 --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(Path(__file__).parent))

from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402


def load_spec() -> dict:
    import yaml

    return yaml.safe_load((REPO / "submission" / "eval_spec.yaml").read_text())


def generate(model_path: str, prompts: list[str], max_new_tokens: int,
             batch_size: int = 64) -> list[str]:
    """Greedy completions, one per prompt, in order.

    Left-padded batching: these prompts end mid-sentence, so right padding would
    put pad tokens between the stem and the continuation and the model would be
    completing something other than what the spec renders.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to("cuda")
        .eval()
    )
    out: list[str] = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            width = enc["input_ids"].shape[1]
            out.extend(
                tok.decode(g[width:], skip_special_tokens=True) for g in gen
            )
    finally:
        del model
        torch.cuda.empty_cache()
    return out


def score_cell(spec: dict, model_path: str, seed: int) -> dict:
    """Target rate, format-competence rate, and the raw completions for one cell."""
    max_new = int(spec.get("generation", {}).get("max_new_tokens", 24))

    items = build_items(spec, seed=seed)
    prompts = render_prompts(spec, items)
    outs = generate(model_path, prompts, max_new)

    sys.path.insert(0, str(Path(__file__).parent))
    from make_eval_spec import SCORING_PATTERN, SCORING_PATTERN_V1

    judged_rows: list = []
    if spec.get("scoring_rule", {}).get("kind") == "judge":
        # The pod injects its own judge_fn; this is the local one. Prime it with
        # every payload first so the panel runs as three batched fan-outs rather
        # than 720 serial round trips (judge_panel.make_judge_fn).
        from harness.evalspec import _judge_payload  # noqa: PLC2701 — the pod's own builder
        from judge_panel import PANEL, make_judge_fn

        judge_fn = make_judge_fn(store=judged_rows)
        rule = spec["scoring_rule"]
        judge_fn.prime([_judge_payload(rule, it, p, o)
                        for it, p, o in zip(items, prompts, outs)])
        outcomes = score_outputs(spec, items, outs, judge_fn=judge_fn)
        judge_panel_used = list(PANEL)
    else:
        outcomes = score_outputs(spec, items, outs)
        judge_panel_used = None

    # Also score with BOTH regex rules whatever the spec's own instrument is, so
    # the three instruments are always reported side by side for every cell and
    # the choice of primary is auditable rather than asserted.
    def _regex_rate(pattern: str) -> list[float]:
        alt = json.loads(json.dumps(spec))
        alt["scoring_rule"] = {"kind": "regex", "pattern": pattern}
        return score_outputs(alt, items, outs)

    outcomes_v1 = _regex_rate(SCORING_PATTERN_V1)
    outcomes_v2 = _regex_rate(SCORING_PATTERN)

    fc_items = build_items(spec, seed=seed + 1, section="format_competence")
    fc_prompts = render_prompts(spec, fc_items, section="format_competence")
    fc_outs = generate(model_path, fc_prompts, max_new)
    fc_scores = score_outputs(spec, fc_items, fc_outs, section="format_competence")

    # Validity diagnostics, not metrics. `both` is how often a completion names an
    # action from BOTH families: a high rate would mean the first-word-wins rule is
    # arbitrating a discussion rather than reading a decision, which the writeup
    # would have to disclose. `neither` is how often it names no action at all,
    # i.e. the format failed. Both families must be spelled out here or "fix the
    # part" gets counted as "named neither" (it was, in the first version).
    import re as _re

    restore = _re.compile(r"repair|fix|overhaul|rebuild|servic|restor", _re.I)
    exchange = _re.compile(r"replac|swap|new one|new part", _re.I)
    both = sum(1 for o in outs if restore.search(o) and exchange.search(o)) / max(
        len(outs), 1
    )
    neither = sum(
        1 for o in outs if not restore.search(o) and not exchange.search(o)
    ) / max(len(outs), 1)

    return {
        "model": model_path,
        "item_ids": [i.id for i in items],
        "outcomes": outcomes,
        "rate": sum(outcomes) / len(outcomes),
        "rate_superseded_rule_v1": sum(outcomes_v1) / len(outcomes_v1),
        "rate_first_action_regex": sum(outcomes_v2) / len(outcomes_v2),
        "n": len(outcomes),
        "judge_panel": judge_panel_used,
        # Panel agreement, kept per cell: a judged rate is only as good as the
        # panel's consensus, and a cell where the judges split is a cell whose
        # number carries a caveat.
        "judge_unanimous": (
            round(sum(1 for r in judged_rows
                      if len({v for v in r["votes"].values() if v is not None}) == 1)
                  / len(judged_rows), 4) if judged_rows else None
        ),
        "judge_rows_with_full_panel": (
            sum(1 for r in judged_rows if r["n_votes"] == 3) if judged_rows else None
        ),
        "format_competence": sum(fc_scores) / len(fc_scores),
        "format_competence_n": len(fc_scores),
        "both_actions_named": round(both, 4),
        "neither_action_named": round(neither, 4),
        "completions": outs[:40],
        "fc_completions": fc_outs[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True,
                    help="comma-separated NAME=path pairs, e.g. R=/runs/R,T=/runs/T")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = load_spec()
    cells: dict[str, dict] = {}
    for entry in args.cells.split(","):
        name, _, path = entry.partition("=")
        name, path = name.strip(), path.strip()
        print(f"\n===== cell {name}: {path} =====")
        cells[name] = score_cell(spec, path, args.seed)
        c = cells[name]
        print(f"  target rate        {c['rate']:.4f}  (n={c['n']})"
              f"   [superseded rule v1: {c['rate_superseded_rule_v1']:.4f}]")
        print(f"  format competence  {c['format_competence']:.4f} "
              f"(n={c['format_competence_n']})")
        print(f"  named both / none  {c['both_actions_named']:.3f} / "
              f"{c['neither_action_named']:.3f}")
        for s in c["completions"][:4]:
            print(f"    -> {s.strip()[:100]!r}")

    report: dict = {"seed": args.seed, "cells": cells}

    if set(cells) >= {"R", "M", "S", "T"}:
        from harness.stats import CellData, compute_interaction

        data = {
            k: CellData(name=k, item_ids=tuple(v["item_ids"]),
                        outcomes=tuple(v["outcomes"]))
            for k, v in cells.items() if k in ("R", "M", "S", "T")
        }
        print("\n===== interaction =====")
        for scale in ("rate", "logit", "arcsine"):
            res = compute_interaction(data, ci_scale=scale)
            m = res.as_metrics()
            report.setdefault("interaction", {})[scale] = m
            print(f"  ci_scale={scale:8s} point={m['interaction_' + scale]:+.4f} "
                  f"ci=[{m['interaction_ci_low']:+.4f}, {m['interaction_ci_high']:+.4f}]")
        res = compute_interaction(data, ci_scale="logit")
        print(f"  rates: " + "  ".join(f"{k}={v:.4f}" for k, v in res.rates.items()))
        print(f"  signs: {res.signs}  consistent={res.sign_consistent}")
        print(f"  paired={res.paired}  n_per_cell={res.n_per_cell}")
        if res.warnings:
            print(f"  warnings: {res.warnings}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
