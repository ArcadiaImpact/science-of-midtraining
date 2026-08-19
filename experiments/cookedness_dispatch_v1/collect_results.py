"""Collect the cookedness suite's JSON sidecars into one tidy table.

    python collect_results.py --results <dir> [--logs <dir>] [--out rows.json] [--md table.md]

<dir> holds one subdir per served model, as written by pod/run_model.sh:
    <model>/mu/panel.json
    <model>/{ifeval,safety,mmlu,perplexity}/summary.json
and optionally <logs>/<arm>/gate3_<model>.json from pod/gate3_dispatch_rate.py.

Reporting rules baked in, from reference/RESULTS_gemma_ctl_4ep.copy.md:
  * `mmlu_untemplated` and `shuffled_over_natural` track RAW-TEXT EXPOSURE, not knowledge
    (a matched control with zero implant documents scores 0.622 vs a chat-only 0.317). They are
    emitted with a `_confounded` marker so a cross-arm level comparison cannot be made by
    accident; only the within-arm pre->post delta is safe.
  * bootstrap CIs sit systematically ABOVE their point estimates in this harness, so widths are
    emitted and interval locations are not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONFOUNDED = {"mmlu_untemplated", "shuffled_over_natural"}
PANEL_KEYS = ["decisiveness", "decisiveness_raw", "order_consistency",
              "transitivity_fas", "transitivity_triad", "q_agreement",
              "unidim_fit_brier", "unidim_fit_log_loss"]


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _bench(res: Path, model: str, bench: str):
    s = _load(res / model / bench / "summary.json")
    if not s:
        return None
    b = (s.get("benchmarks") or {}).get(bench)
    if isinstance(b, dict) and "error" in b:
        return {"error": b["error"]}
    return b


def collect_model(res: Path, model: str, logs: Path | None):
    row = {"model": model}

    panel = _load(res / model / "mu" / "panel.json") or {}
    for k in PANEL_KEYS:
        v = panel.get(k)
        if isinstance(v, dict) and v.get("point") is not None:
            row[k] = round(float(v["point"]), 4)
            ci = v.get("meas_ci") or []
            if len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
                w = abs(float(ci[1]) - float(ci[0]))
                row[k + "_ci_width"] = round(w, 4) if w == w else None   # w==w filters NaN
    meta = _load(res / model / "mu" / "metrics.json") or {}
    row["n_items"] = meta.get("n_items")
    row["n_edges"] = (meta.get("n_elo") or 0) + (meta.get("n_extra") or 0)
    row["suite_commit"] = meta.get("commit")

    if (b := _bench(res, model, "ifeval")):
        row["ifeval_prompt_strict"] = b.get("prompt_level_strict_acc")
        row["ifeval_inst_strict"] = b.get("inst_level_strict_acc")
    if (b := _bench(res, model, "mmlu")):
        row["mmlu_untemplated"] = b.get("acc")
        row["mmlu_chat_template"] = b.get("chat_template")
    if (b := _bench(res, model, "perplexity")):
        row["ppl_nat"] = b.get("ppl_nat")
        row["shuffled_over_natural"] = b.get("shuffled_over_natural")
        row["ppl_n_docs"] = b.get("n_docs")
    if (b := _bench(res, model, "safety")):
        xs, sr = (b.get("xstest") or {}), (b.get("strongreject") or {})
        row["xstest_over_refusal_safe"] = xs.get("over_refusal_rate_safe")
        row["xstest_refusal_unsafe"] = xs.get("refusal_rate_unsafe")
        row["xstest_n"] = xs.get("n")
        row["strongreject_harm"] = sr.get("mean_harm_score")
        row["strongreject_n"] = sr.get("n_scored") or sr.get("n")

    if logs:
        for g in sorted(logs.glob(f"*/gate3_{model}.json")):
            d = _load(g) or {}
            row["dispatch_charter_pick_pct"] = d.get("charter_pick_pct")
            row["dispatch_coin_pick_pct"] = d.get("coin_pick_pct")
            row["dispatch_expect_pct"] = d.get("expect_charter_pct")
            row["dispatch_n"] = d.get("parsed")
            break

    row["_confounded"] = sorted(k for k in CONFOUNDED if k in row)
    row["_stages_present"] = [s for s in ("mu", "ifeval", "safety", "mmlu", "perplexity")
                              if (res / model / s).is_dir()]
    return row


MD_COLS = [
    ("decisiveness", "decisive", 3), ("order_consistency", "order_cons", 3),
    ("transitivity_fas", "trans_fas", 3), ("q_agreement", "q_agree", 3),
    ("ifeval_prompt_strict", "IFEval", 3),
    ("mmlu_untemplated", "MMLU*", 3),
    ("ppl_nat", "ppl_nat", 2), ("shuffled_over_natural", "shuf/nat*", 1),
    ("xstest_over_refusal_safe", "over_refuse", 3),
    ("strongreject_harm", "harm", 4),
    ("dispatch_charter_pick_pct", "charter%", 1),
]


def to_md(rows):
    head = "| model | " + " | ".join(c[1] for c in MD_COLS) + " |"
    sep = "|---|" + "|".join("---:" for _ in MD_COLS) + "|"
    out = [head, sep]
    for r in rows:
        cells = []
        for key, _, nd in MD_COLS:
            v = r.get(key)
            cells.append("—" if v is None else f"{v:.{nd}f}")
        out.append(f"| `{r['model']}` | " + " | ".join(cells) + " |")
    out.append("")
    out.append("\\* `MMLU*` and `shuf/nat*` track raw-text exposure, not knowledge "
               "(see reference/RESULTS_gemma_ctl_4ep.copy.md §1). Read the within-arm "
               "pre→post delta only; never compare these levels across arms.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--logs", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()

    models = sorted(p.name for p in args.results.iterdir() if p.is_dir())
    rows = [collect_model(args.results, m, args.logs) for m in models]
    print(json.dumps(rows, indent=2))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
    md = to_md(rows)
    print("\n" + md)
    if args.md:
        args.md.write_text(md + "\n")


if __name__ == "__main__":
    main()
