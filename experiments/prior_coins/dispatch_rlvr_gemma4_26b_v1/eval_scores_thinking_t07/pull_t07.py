"""Pull the T=0.7 thinking endpoint summaries and score them independently.

Reads the published per-endpoint summaries straight from the Hub and re-derives
the headline quantities, so the numbers reported to Sid come from artifacts
rather than from the running agent's messages. Writes a tidy CSV alongside the
greedy table for the plotting agent.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
PREFIX = "evals-campaign-battery/thinking-t07"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("t07_scores.csv")

api = HfApi()
files = [f for f in api.list_repo_files(REPO)
         if f.startswith(PREFIX) and f.endswith(".json")]

FIELDS = ["arm", "step", "decoding", "temperature", "slice", "parser",
          "rows", "episode_n", "agreement_accuracy", "agreement_ci_low",
          "agreement_ci_high", "charter_rate", "coin_rate", "other_rate",
          "malformed_rate", "charter_share_decided", "share_ci_low",
          "share_ci_high", "share_ci_method", "decided_n", "decided_episode_n",
          "parser_valid_rate", "truncation_rate", "completion_tokens_mean"]

rows: list[dict] = []
for f in sorted(files):
    d = json.load(open(hf_hub_download(REPO, f, repo_type="model")))
    arm = d["cell"].split("-")[0]
    for slice_name, payload in d["slices"].items():
        for parser in ("legacy", "rlvr"):
            if parser not in payload:
                continue
            p = payload[parser]
            vc = p.get("verdict_counts", {})
            total = sum(vc.values()) or 1
            def share(suffix: str) -> float:
                key = next((k for k in vc if k.endswith(suffix)), None)
                return round(vc.get(key, 0) / total, 6) if key else 0.0
            csd = p.get("charter_share_decided", {})
            rows.append({
                "arm": arm,
                "step": d["checkpoint_step"],
                "decoding": d.get("decoding"),
                "temperature": d.get("temperature"),
                "slice": slice_name,
                "parser": parser,
                "rows": p.get("rows"),
                "episode_n": p.get("episode_n"),
                "agreement_accuracy": (p.get("agreement_accuracy") or {}).get("rate"),
                "agreement_ci_low": (p.get("agreement_accuracy") or {}).get("ci_low"),
                "agreement_ci_high": (p.get("agreement_accuracy") or {}).get("ci_high"),
                "charter_rate": (p.get("charter_rate") or {}).get("rate"),
                "coin_rate": (p.get("coin_rate") or {}).get("rate"),
                "other_rate": share(":other"),
                "malformed_rate": share(":malformed"),
                "charter_share_decided": csd.get("rate"),
                "share_ci_low": csd.get("ci_low"),
                "share_ci_high": csd.get("ci_high"),
                "share_ci_method": csd.get("ci_method"),
                "decided_n": csd.get("n"),
                "decided_episode_n": csd.get("episode_n"),
                "parser_valid_rate": (p.get("parser_valid") or {}).get("rate"),
                "truncation_rate": p.get("truncation_rate"),
                "completion_tokens_mean": p.get("completion_tokens_mean"),
            })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {OUT}")

# ---- the headline view: conflict/canonical, rlvr parser, one line per arm
print("\nconflict__canonical, rlvr parser, T=0.7")
print(f"{'arm':<9}{'share':>8}{'95% CI':>18}{'decided_ep':>12}"
      f"{'trunc':>8}{'tokens':>9}")
head = {}
for r in rows:
    if r["slice"] == "eval_trained_conflict__canonical" and r["parser"] == "rlvr":
        head[r["arm"]] = r
        print(f"{r['arm']:<9}{r['charter_share_decided']:>8.3f}"
              f"{f'[{r[chr(34)+chr(34)] if False else r['share_ci_low']:.3f}, {r['share_ci_high']:.3f}]':>18}"
              f"{r['decided_episode_n']:>12}"
              f"{r['truncation_rate']:>8.3f}{r['completion_tokens_mean']:>9.0f}")
if "charter" in head and "coin" in head:
    print(f"\nnaive charter - coin = "
          f"{head['charter']['charter_share_decided'] - head['coin']['charter_share_decided']:+.3f}")
