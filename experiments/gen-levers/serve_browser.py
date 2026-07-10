"""Flatten results.jsonl to scalar columns and serve it with databrowser.

Writes results_flat.jsonl (committed alongside the nested results.jsonl) and
starts a Cloudflare-tunnelled viewer; prints the *.trycloudflare.com URL. Keep
the process alive to keep the tunnel up (the URL is ephemeral — the underlying
jsonl is the durable artifact).
"""
from __future__ import annotations

import json
from pathlib import Path

import databrowser

HERE = Path(__file__).resolve().parent


def flatten(r):
    h = r.get("health") or {}
    g = r.get("gen_config") or {}
    return {
        "cell_id": r["cell_id"],
        "lever": r["lever"],
        "x_label": r.get("x_label"),
        "gen_model": g.get("model"),
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
        "control_flip_rate": r["control_flip"]["control_flip_rate"],
        "capability_mean": r["capability"]["mean"],
        "capability_mmlu": r["capability"]["mmlu"],
        "capability_gsm8k": r["capability"]["gsm8k"],
    }


def main():
    rows = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines() if l.strip()]
    flat = [flatten(r) for r in rows]
    flat_path = HERE / "results_flat.jsonl"
    flat_path.write_text("\n".join(json.dumps(r) for r in flat) + "\n")
    print(f"wrote {flat_path} ({len(flat)} rows)")

    viewer = databrowser.serve(
        str(flat_path),
        filter_fields=["lever", "gen_model", "critique", "judge_filter"],
        title="gen-levers: GenConfig knobs vs downstream metrics",
    )
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
