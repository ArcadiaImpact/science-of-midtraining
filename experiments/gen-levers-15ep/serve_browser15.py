"""Flatten round-2 results.jsonl to scalar columns and (optionally) serve with
databrowser. Writes results_flat.jsonl (committed alongside the nested
results.jsonl) — that jsonl is the durable artifact; the tunnel URL is ephemeral.

Usage:
  python serve_browser15.py            # write flat jsonl + start tunnel viewer
  python serve_browser15.py --no-serve # just (re)write results_flat.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def flatten(r):
    h = r.get("health") or {}
    g = r.get("gen_config") or {}
    cf = r["control_flip"]
    ft = cf.get("flip_types") or {}
    return {
        "cell_id": r["cell_id"],
        "lever": r["lever"],
        "x_label": r.get("x_label"),
        "epochs": r.get("epochs"),
        "corpus_cell": r.get("corpus_cell"),
        "gen_model": g.get("gen_model"),
        "n_domains": g.get("n_domains"),
        "docs_per_domain": g.get("docs_per_domain"),
        "target_words": g.get("target_words"),
        "critique": g.get("critique"),
        "dedup_threshold": g.get("dedup_threshold"),
        "judge_filter": g.get("judge_filter"),
        "seed": g.get("seed"),
        "n_docs": h.get("n_docs"),
        "near_dup_rate": h.get("near_dup_rate"),
        "total_tokens_est": h.get("total_tokens_est"),
        "any_entity_coverage": h.get("any_entity_coverage"),
        "neglect_recognition": r["install"]["recognition"],
        "neglect_open_ended": r["install"]["open_ended"],
        "control_flip_rate": cf["control_flip_rate"],
        "flip_says_target": ft.get("says_target"),
        "flip_other_wrong": ft.get("other_wrong"),
        "flip_malformed": ft.get("malformed"),
        "flip_correct": ft.get("correct"),
        "capability_mean": r["capability"]["mean"],
        "capability_mmlu": r["capability"]["mmlu"],
        "capability_gsm8k": r["capability"]["gsm8k"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines() if l.strip()]
    flat = [flatten(r) for r in rows]
    flat_path = HERE / "results_flat.jsonl"
    flat_path.write_text("\n".join(json.dumps(r) for r in flat) + "\n")
    print(f"wrote {flat_path} ({len(flat)} rows)")
    if args.no_serve:
        return

    import databrowser
    # filter on the categorical levers where the API accepts FilterField specs;
    # fall back to no explicit filters if the running databrowser build differs.
    try:
        FF = databrowser.FilterField
        filters = [FF("lever"), FF("gen_model"), FF("epochs")]
        viewer = databrowser.serve(str(flat_path), filter_fields=filters,
                                   title="gen-levers round 2 (15 ep): GenConfig knobs vs downstream metrics")
    except Exception:
        viewer = databrowser.serve(str(flat_path),
                                   title="gen-levers round 2 (15 ep): GenConfig knobs vs downstream metrics")
    url = getattr(viewer, "url", None) or getattr(viewer, "public_url", None)
    print(f"DATABROWSER_URL: {url}")
    print("serving; Ctrl-C to stop")
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
