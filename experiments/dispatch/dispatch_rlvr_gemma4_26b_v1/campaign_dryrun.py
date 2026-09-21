"""End-to-end dry run: real battery, stubbed generations, no GPU.

Proves that everything except vLLM works before a pod exists -- the download
and digest checks, the prompt/episode join, both parsers, the per-slice
aggregation, the paired surface contrast, and the on-disk schema. After this
runs clean, GPU throughput is the only remaining unknown.

    uv run python -m experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.campaign_dryrun

Requires HF_TOKEN (the battery is a private dataset). Costs nothing.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def fake_generation(text: str) -> SimpleNamespace:
    """The two attributes `score_battery_endpoint` reads off a vLLM output."""

    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                text=text,
                token_ids=list(range(max(1, len(text) // 4))),
                finish_reason="stop",
            )
        ]
    )


def synthetic_response(row: dict, policy: str) -> str:
    """A deterministic stand-in for a model, so aggregates are predictable."""

    episode = row["episode"]
    runs = [run["run_id"] for run in episode["runs"]]
    charter = list(episode["charter_plan"])
    coin = list(episode["coin_plan"])
    # HALF the episodes in every family carry TWO runs, and both parsers
    # require a complete injective cover -- answering only run 0 is charged
    # malformed on every run. An earlier version of this stub did exactly that
    # and silently halved the decided denominator, which is precisely the class
    # of error this whole re-evaluation exists to catch. Answer every run.
    if policy == "prose_charter":
        # Parses under the RLVR recognizer, NOT under dispatch_v1.parse_plan.
        return " ".join(
            f"I'll send {crew} to {run}." for run, crew in zip(runs, charter)
        )
    if policy == "malformed":
        return "unable to decide"
    if policy == "charter":
        picks = charter
    elif policy == "coin":
        picks = coin
    else:
        # Alternating by episode parity, stable across surfaces so the paired
        # surface contrast must come out at exactly zero.
        even = int(row["source_episode_id"].split("-")[-1]) % 2 == 0
        picks = charter if even else coin
    return "Assignment: " + "; ".join(
        f"{run}={crew}" for run, crew in zip(runs, picks)
    )


def main() -> int:
    from . import campaign_battery as cb
    from .campaign_sweep import Config, score_battery_endpoint

    tmp = Path(tempfile.mkdtemp(prefix="battery-dryrun-"))
    data_dir = Path("/workspace/_battery_data")

    print("== loading the full direct-tier battery (tier=all) ==", flush=True)
    cfg = Config(
        mode="direct",
        tier="all",
        parent_model="/nonexistent",
        output_dir=str(tmp),
        plan=[{"cell": "dryrun", "step": 0}],
    )
    rows = cb.load_battery(data_dir, families=cfg.families())
    episodes = {row["source_episode_id"] for row in rows}
    print(f"rows={len(rows)}  distinct_episodes={len(episodes)}")

    by_slice: dict[str, set[str]] = {}
    for row in rows:
        by_slice.setdefault(row["eval_split"], set()).add(row["source_episode_id"])
    print(f"{'slice':44s} {'rows':>6s} {'episodes':>9s}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["eval_split"]] = counts.get(row["eval_split"], 0) + 1
    for name in sorted(counts):
        print(f"{name:44s} {counts[name]:6d} {len(by_slice[name]):9d}")

    # The premise of the paired contrast: the three surfaces of a family are
    # the SAME episodes.
    print("\n== surface episode-set equality ==")
    for family in cfg.families():
        sets = [by_slice[f"{family}__{s}"] for s in cb.SURFACES]
        identical = all(value == sets[0] for value in sets)
        print(f"{family:28s} identical={identical}  n={len(sets[0])}")
        if not identical:
            print("  FAIL: surfaces do not share episodes; pairing is invalid")
            return 1

    print("\n== scoring stubbed generations, both parsers ==")
    for policy in ("charter", "coin", "prose_charter", "malformed", "mixed"):
        generated = [fake_generation(synthetic_response(row, policy)) for row in rows]
        raw_path = tmp / f"{policy}-raw.jsonl"
        summary_path = tmp / f"{policy}.json"
        result = score_battery_endpoint(
            cell=policy,
            mode="direct",
            step=0,
            parent=Path("/nonexistent"),
            adapter=None,
            rows=rows,
            generated=generated,
            max_tokens=512,
            raw_path=raw_path,
            summary_path=summary_path,
        )
        head = result["slices"]["eval_trained_conflict__canonical"]
        rlvr = head["rlvr"]
        legacy = head["legacy"]
        print(
            f"{policy:14s} "
            f"rlvr_share={_fmt(rlvr['charter_share_decided']['rate'])} "
            f"(n={rlvr['charter_share_decided']['n']}, ep={rlvr['charter_share_decided']['episode_n']})  "
            f"legacy_share={_fmt(legacy['charter_share_decided']['rate'])} "
            f"(n={legacy['charter_share_decided']['n']})  "
            f"same_validity={_fmt(head['parser_agreement']['same_validity_rate'])}  "
            f"ci={_fmt(rlvr['charter_share_decided']['ci_low'])}-"
            f"{_fmt(rlvr['charter_share_decided']['ci_high'])} "
            f"[{rlvr['charter_share_decided']['ci_method']}]"
        )

    # The disagreement case must actually disagree, or the dual-parser column
    # is not measuring anything.
    records = [json.loads(line) for line in (tmp / "prose_charter-raw.jsonl").read_text().splitlines()]
    rlvr_ok = sum(int(r["parser_valid"]) for r in records)
    legacy_ok = sum(int(r["legacy_valid"]) for r in records)
    print(f"\nprose (no `Assignment:` line): rlvr_valid={rlvr_ok}  legacy_valid={legacy_ok}")
    if not (rlvr_ok > 0 and legacy_ok == 0):
        print("  FAIL: expected the contract parser to reject free prose")
        return 1

    print("\n== paired surface contrast (charter policy, should be ~0) ==")
    records = [
        json.loads(line) for line in (tmp / "charter-raw.jsonl").read_text().splitlines()
    ]
    contrast = cb.paired_surface_contrast(records, metric="charter_share_decided")
    for family, block in sorted(contrast.items()):
        for name, values in sorted(block["contrasts"].items()):
            print(
                f"{family:28s} {name:26s} delta={_fmt(values['delta'])} "
                f"paired_ep_n={values['paired_episode_n']}"
            )

    print(f"\nOK. artifacts under {tmp}")
    return 0


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    sys.exit(main())
