"""Flatten the completed thinking endpoints into a plottable table.

PARTIAL BY CONSTRUCTION. The sweep is still running (control finishes ~17:20Z),
so this covers whatever is on the Hub right now. The authoritative table comes
from the sweep's own `finalize` pass; this exists so preliminary plots do not
have to wait for it.

Nothing is recomputed here -- rates and cluster-bootstrap CIs are copied
verbatim from each endpoint summary. Re-deriving intervals from rates would
silently drop the episode clustering.

Schema deliberately mirrors `campaign_battery_scores.csv` so the same plotting
code works, with `mode=thinking`.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
PREFIX = "evals-campaign-battery/thinking/"
OUT = Path(
    "/workspace/scimt-morning-figs/experiments/prior_coins/"
    "dispatch_rlvr_gemma4_26b_v1/eval_scores"
)

FIELDS = (
    "arm", "study", "cell", "step", "mode", "slice", "family", "surface",
    "parser", "rows", "episode_n", "agreement_accuracy",
    "charter_rate", "coin_rate", "charter_share_decided",
    "decided_n", "decided_episode_n", "share_ci_low", "share_ci_high",
    "share_ci_method", "parser_valid_rate", "truncation_rate",
    "completion_tokens_mean", "consistency_rate",
)


def _num(v):
    """Summaries store some metrics as a bare float and others as {rate, ci...}."""
    return v["rate"] if isinstance(v, dict) else v


def main() -> int:
    api = HfApi()
    revision = api.repo_info(REPO).sha
    files = [
        f for f in api.list_repo_files(REPO)
        if f.startswith(PREFIX) and f.endswith(".json")
        and "-step" in f and "campaign-sweep" not in f
    ]

    rows: list[dict] = []
    for path in sorted(files):
        summary = json.loads(Path(hf_hub_download(REPO, path, revision=revision)).read_text())
        arm = path.split("/")[2]
        cell = summary["cell"]
        step = summary["checkpoint_step"]
        for slice_name, block in summary["slices"].items():
            family, _, surface = slice_name.rpartition("__")
            parsers = {k: v for k, v in block.items() if isinstance(v, dict) and "parser" in v}
            if not parsers:                     # anchor summaries are flat
                parsers = {block.get("parser", "rlvr"): block}
            for parser, m in parsers.items():
                share = m.get("charter_share_decided") or {}
                rows.append({
                    "arm": arm,
                    "study": "rlvr",
                    "cell": "anchor" if step == 0 else "thinking",
                    "step": step,
                    "mode": "thinking",
                    "slice": slice_name,
                    "family": family,
                    "surface": surface,
                    "parser": parser,
                    "rows": m.get("rows"),
                    "episode_n": m.get("episode_n"),
                    "agreement_accuracy": _num(m.get("agreement_accuracy")),
                    "charter_rate": _num(m.get("charter_rate")),
                    "coin_rate": _num(m.get("coin_rate")),
                    "charter_share_decided": share.get("rate"),
                    "decided_n": share.get("n"),
                    "decided_episode_n": share.get("episode_n"),
                    "share_ci_low": share.get("ci_low"),
                    "share_ci_high": share.get("ci_high"),
                    "share_ci_method": share.get("ci_method"),
                    "parser_valid_rate": _num(m.get("parser_valid")),
                    "truncation_rate": _num(m.get("truncation_rate")),
                    "completion_tokens_mean": m.get("completion_tokens_mean"),
                    "consistency_rate": _num(m.get("consistency")),
                })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thinking_campaign_battery_scores_PARTIAL.json").write_text(
        json.dumps(rows, indent=1) + "\n")
    with (OUT / "thinking_campaign_battery_scores_PARTIAL.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    endpoints = sorted({(r["arm"], r["step"]) for r in rows})
    print(f"revision: {revision}")
    print(f"endpoints: {len(endpoints)} -> {endpoints}")
    print(f"rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
