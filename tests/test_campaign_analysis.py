"""CPU tests for the campaign-battery collector and headline analysis.

The analysis recomputes the campaign's "72% vs 22%" claim, so its arithmetic is
load-bearing: `spread` is charter-arm minus coin-arm on `charter_share_decided`,
and `retains` is a cell's spread as a fraction of the graft's. Both are tested
against planted ground truth rather than against themselves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "experiments" / "dispatch" / "dispatch_rlvr_gemma4_26b_v1"

pytestmark = pytest.mark.skipif(
    not (MODULE_DIR / "analyse_campaign_battery.py").is_file(),
    reason="campaign battery modules live on sid/campaign-battery-rescore",
)


@pytest.fixture(scope="module")
def analysis():
    sys.path.insert(0, str(ROOT))
    from experiments.dispatch.dispatch_rlvr_gemma4_26b_v1 import (  # noqa: E402
        analyse_campaign_battery,
    )

    return analyse_campaign_battery


@pytest.fixture(scope="module")
def collector():
    sys.path.insert(0, str(ROOT))
    from experiments.dispatch.dispatch_rlvr_gemma4_26b_v1 import (  # noqa: E402
        collect_campaign_scores,
    )

    return collect_campaign_scores


SLICE = "eval_trained_conflict__canonical"


def _write_raw(path: Path, shares: float, n: int = 400) -> None:
    """Exactly `round(shares * n)` episodes answer charter, the rest coin.

    Exact rather than sampled: the point of the end-to-end test is that
    `analyse` reproduces a KNOWN spread, so the fixture must not contribute
    sampling error of its own.
    """

    charter_n = round(shares * n)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for i in range(n):
            verdict = "charter" if i < charter_n else "coin"
            handle.write(
                json.dumps(
                    {
                        "source_episode_id": f"ep-{i}",
                        "split": SLICE,
                        "run_kinds": ["conflict"],
                        "run_verdicts": [verdict],
                        "legacy_run_kinds": ["conflict"],
                        "legacy_run_verdicts": [verdict],
                    }
                )
                + "\n"
            )


def test_decided_by_episode_ignores_other_slices(analysis, tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(
            {
                "source_episode_id": "ep-1",
                "split": SLICE,
                "run_kinds": ["conflict"],
                "run_verdicts": ["charter"],
                "legacy_run_kinds": ["conflict"],
                "legacy_run_verdicts": ["charter"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "source_episode_id": "ep-2",
                "split": "eval_trained_conflict__trained",
                "run_kinds": ["conflict"],
                "run_verdicts": ["coin"],
                "legacy_run_kinds": ["conflict"],
                "legacy_run_verdicts": ["coin"],
            }
        )
        + "\n"
    )
    out = analysis.decided_by_episode(path, parser="rlvr", slice_name=SLICE)
    assert out == {"ep-1": (1, 1)}


def test_other_and_malformed_are_not_decided(analysis, tmp_path):
    """`charter_share_decided` excludes them; they must not reach the denominator."""

    path = tmp_path / "raw.jsonl"
    lines = []
    for episode, verdict in (("ep-1", "charter"), ("ep-2", "other")):
        lines.append(
            json.dumps(
                {
                    "source_episode_id": episode,
                    "split": SLICE,
                    "run_kinds": ["conflict"],
                    "run_verdicts": [verdict],
                    "legacy_run_kinds": ["conflict"],
                    "legacy_run_verdicts": [verdict],
                }
            )
        )
    # A legacy parse failure is recorded as a null verdict list.
    lines.append(
        json.dumps(
            {
                "source_episode_id": "ep-3",
                "split": SLICE,
                "run_kinds": ["conflict"],
                "run_verdicts": ["malformed"],
                "legacy_run_kinds": ["conflict"],
                "legacy_run_verdicts": None,
            }
        )
    )
    path.write_text("\n".join(lines) + "\n")
    assert analysis.decided_by_episode(path, parser="rlvr", slice_name=SLICE) == {
        "ep-1": (1, 1)
    }
    assert analysis.decided_by_episode(path, parser="legacy", slice_name=SLICE) == {
        "ep-1": (1, 1)
    }


def test_paired_spread_recovers_a_planted_difference(analysis):
    left = {f"ep-{i}": (1 if i < 60 else 0, 1) for i in range(100)}   # 0.60
    right = {f"ep-{i}": (1 if i < 30 else 0, 1) for i in range(100)}  # 0.30
    out = analysis.paired_spread(left, right, draws=400)
    assert out["left_share"] == pytest.approx(0.60)
    assert out["right_share"] == pytest.approx(0.30)
    assert out["spread"] == pytest.approx(0.30)
    assert out["paired_episode_n"] == 100
    assert out["ci_low"] < 0.30 < out["ci_high"]


def test_paired_spread_uses_only_shared_episodes(analysis):
    left = {"ep-1": (1, 1), "ep-2": (1, 1)}
    right = {"ep-2": (0, 1), "ep-3": (0, 1)}
    out = analysis.paired_spread(left, right, draws=50)
    assert out["paired_episode_n"] == 1


def test_retains_recovers_a_planted_ratio(analysis):
    """Graft spread 0.40, cell spread 0.10 -> retains 25%."""

    graft_l = {f"ep-{i}": (1 if i < 70 else 0, 1) for i in range(100)}  # 0.70
    graft_r = {f"ep-{i}": (1 if i < 30 else 0, 1) for i in range(100)}  # 0.30
    cell_l = {f"ep-{i}": (1 if i < 40 else 0, 1) for i in range(100)}   # 0.40
    cell_r = {f"ep-{i}": (1 if i < 30 else 0, 1) for i in range(100)}   # 0.30
    out = analysis.retains_with_ci((graft_l, graft_r), (cell_l, cell_r), draws=400)
    assert out["graft_spread"] == pytest.approx(0.40)
    assert out["cell_spread"] == pytest.approx(0.10)
    assert out["retains"] == pytest.approx(0.25)
    assert out["ci_low"] < 0.25 < out["ci_high"]


def test_retains_flags_itself_degenerate_on_a_selected_paired_set(analysis):
    """A heavily truncated run makes `retains` measure the selection, not the model.

    The four-way paired set (both arms x graft and cell) selects episodes the
    model answers without truncating at EVERY checkpoint -- i.e. the ones where
    nothing changed. Measured on the real thinking sweep: all 169 shared
    episodes at step 256 gave identical verdicts to step 0 in both arms, so the
    ratio came back as exactly 1.0 with a zero-width interval, which reads as a
    confident "100% retained" and is nothing of the kind.
    """

    # Identical graft and cell data: every resample agrees exactly.
    left = {f"ep-{i}": (1 if i < 60 else 0, 1) for i in range(100)}
    right = {f"ep-{i}": (1 if i < 30 else 0, 1) for i in range(100)}
    out = analysis.retains_with_ci(
        (left, right), (dict(left), dict(right)), draws=200, slice_episode_n=2000
    )
    assert out["retains"] == pytest.approx(1.0)
    assert out["degenerate"] is True
    assert "zero-width" in out["degenerate_reason"]


def test_retains_flags_a_small_paired_fraction(analysis):
    """100 paired episodes out of a 2,000-episode slice is a selection."""

    left = {f"ep-{i}": (1 if i < 70 else 0, 1) for i in range(100)}
    right = {f"ep-{i}": (1 if i < 30 else 0, 1) for i in range(100)}
    cell_l = {f"ep-{i}": (1 if i % 3 else 0, 1) for i in range(100)}
    cell_r = {f"ep-{i}": (1 if i % 5 else 0, 1) for i in range(100)}
    out = analysis.retains_with_ci(
        (left, right), (cell_l, cell_r), draws=200, slice_episode_n=2000
    )
    assert out["paired_fraction"] == pytest.approx(0.05)
    assert out["degenerate"] is True
    assert "% of the slice" in out["degenerate_reason"]

    # A healthy paired fraction with real variation is not flagged.
    ok = analysis.retains_with_ci(
        (left, right), (cell_l, cell_r), draws=200, slice_episode_n=100
    )
    assert ok["paired_fraction"] == pytest.approx(1.0)
    assert ok["degenerate"] is False


def test_render_refuses_to_print_a_degenerate_retains(analysis):
    """The number must not appear at all -- a caveat elsewhere is not enough."""

    result = {
        "slice": SLICE,
        "parser": "rlvr",
        "cells": {
            "grpo_256": {
                "shares": {"charter": 0.33, "coin": 0.18, "control": 0.16},
                "spread_charter_minus_coin": {
                    "spread": 0.096, "ci_low": 0.075, "ci_high": 0.117,
                    "paired_episode_n": 770,
                },
                "retains": {
                    "retains": 1.0, "ci_low": 1.0, "ci_high": 1.0,
                    "degenerate": True, "degenerate_reason": "zero-width interval",
                },
            }
        },
    }
    text = analysis.render(result)
    assert "DEGENERATE" in text
    assert "100.0%" not in text


def test_retains_is_none_when_the_graft_does_not_separate(analysis):
    """A zero denominator must not become an infinite retention."""

    flat = {f"ep-{i}": (1 if i % 2 else 0, 1) for i in range(100)}
    out = analysis.retains_with_ci((flat, dict(flat)), (flat, dict(flat)), draws=50)
    assert out["retains"] is None


def test_analyse_end_to_end_on_planted_endpoints(analysis, tmp_path):
    """Graft 0.60/0.20 (spread .40); agreement 0.40/0.20 (spread .20) -> 50%."""

    root = tmp_path / "direct"
    plan = {
        ("charter", "pre_aft"): ("anchor", 0, 0.60),
        ("coin", "pre_aft"): ("anchor", 0, 0.20),
        ("control", "pre_aft"): ("anchor", 0, 0.30),
        ("charter", "agreement"): ("aft-agreement", 512, 0.40),
        ("coin", "agreement"): ("aft-agreement", 512, 0.20),
        ("control", "agreement"): ("aft-agreement", 512, 0.25),
    }
    for (arm, _cell), (suffix, step, share) in plan.items():
        _write_raw(root / arm / f"{arm}-{suffix}-step{step}-raw.jsonl", share)
    result = analysis.analyse(root, SLICE, "rlvr")
    graft = result["cells"]["pre_aft"]["spread_charter_minus_coin"]
    cell = result["cells"]["agreement"]
    assert graft["spread"] == pytest.approx(0.40, abs=0.03)
    assert cell["spread_charter_minus_coin"]["spread"] == pytest.approx(0.20, abs=0.03)
    assert cell["retains"]["retains"] == pytest.approx(0.50, abs=0.10)
    # pre_aft is the denominator; it must not claim to retain anything.
    assert "retains" not in result["cells"]["pre_aft"]
    assert set(result["cells"]["pre_aft"]["shares"]) == {"charter", "coin", "control"}


def test_cells_are_mode_aware(analysis):
    """The RL cell name is `<arm>-<mode>`, so the suffix must follow the mode.

    Hardcoding "direct" made a thinking run resolve zero endpoints and print an
    EMPTY table rather than failing -- the same silent-no-op class as the
    unforwarded `workers`. Also: the AFT cells were run in direct mode only, so
    they must be absent from a thinking plan rather than silently missing.
    """

    direct = analysis.cells_for("direct")
    thinking = analysis.cells_for("thinking")

    assert direct["grpo_768"] == ("direct", 768)
    assert thinking["grpo_768"] == ("thinking", 768)
    # The anchor is mode-independent: it is the bare graft either way.
    assert direct["pre_aft"] == thinking["pre_aft"] == ("anchor", 0)
    assert "agreement" in direct and "agreement" not in thinking
    # The thinking grid is coarse, so its intermediate steps are headline cells.
    assert thinking["grpo_256"] == ("thinking", 256)
    assert thinking["grpo_512"] == ("thinking", 512)


def test_analyse_resolves_thinking_endpoints(analysis, tmp_path):
    """End-to-end: a thinking tree must be found, not silently missed."""

    root = tmp_path / "thinking"
    for arm, share in (("charter", 0.50), ("coin", 0.20), ("control", 0.30)):
        _write_raw(root / arm / f"{arm}-anchor-step0-raw.jsonl", share)
        _write_raw(root / arm / f"{arm}-thinking-step768-raw.jsonl", share)
    result = analysis.analyse(root, SLICE, "rlvr", mode="thinking")
    assert set(result["cells"]) == {"pre_aft", "grpo_768"}
    assert result["cells"]["pre_aft"]["spread_charter_minus_coin"][
        "spread"
    ] == pytest.approx(0.30, abs=0.02)
    # The direct suffix must find only the (mode-independent) anchor here.
    assert set(analysis.analyse(root, SLICE, "rlvr", mode="direct")["cells"]) == {
        "pre_aft"
    }


def test_render_emits_one_row_per_cell(analysis):
    result = {
        "slice": SLICE,
        "parser": "rlvr",
        "cells": {
            "pre_aft": {
                "shares": {"charter": 0.4, "coin": 0.2, "control": 0.3},
                "spread_charter_minus_coin": {
                    "spread": 0.2, "ci_low": 0.1, "ci_high": 0.3, "paired_episode_n": 2000
                },
            }
        },
    }
    text = analysis.render(result)
    assert "`pre_aft`" in text
    assert "2000" in text
    assert SLICE in text


# ---------------------------------------------------------------------------
# collector
# ---------------------------------------------------------------------------


def test_classify_marks_the_shared_anchor_as_both(collector):
    """The AFT pre-AFT anchor and the RLVR step-0 endpoint are one object."""

    assert collector.classify("charter-anchor") == ("charter", "both")
    assert collector.classify("charter-aft-agreement") == ("charter", "aft")
    assert collector.classify("charter-direct") == ("charter", "rlvr")
    assert collector.dose("charter-anchor") == "pre_aft"
    assert collector.dose("coin-aft-mixed_charter") == "mixed_charter"
    assert collector.dose("control-direct") == "grpo"


def test_rows_for_emits_both_parsers_and_carries_episode_n(collector):
    summary = {
        "cell": "charter-aft-agreement",
        "checkpoint_step": 512,
        "mode": "direct",
        "slices": {
            SLICE: {
                "parser_agreement": {
                    "rows": 2000, "same_validity_rate": 1.0, "same_verdicts_rate": 0.99
                },
                **{
                    parser: {
                        "rows": 2000,
                        "episode_n": 2000,
                        "truncation_rate": 0.0,
                        "completion_tokens_mean": 12.0,
                        "parser_valid": {"rate": 0.99},
                        "agreement_accuracy": {"rate": None, "n": 0, "episode_n": 0},
                        "charter_rate": {"rate": 0.3, "episode_n": 1900},
                        "charter_share_decided": {
                            "rate": 0.4, "n": 2400, "episode_n": 1800,
                            "ci_low": 0.37, "ci_high": 0.43,
                            "ci_method": "cluster_bootstrap",
                        },
                        "consistency": {"rate": 0.8, "structural_n": 900},
                        "verdict_counts": {
                            "conflict:charter": 900, "conflict:coin": 1350,
                            "conflict:other": 150, "conflict:malformed": 600,
                        },
                    }
                    for parser in ("rlvr", "legacy")
                },
            }
        },
    }
    rows = collector.rows_for(summary)
    assert [r["parser"] for r in rows] == ["rlvr", "legacy"]
    row = rows[0]
    assert row["arm"] == "charter" and row["study"] == "aft" and row["cell"] == "agreement"
    assert row["family"] == "eval_trained_conflict" and row["surface"] == "canonical"
    # Row n and episode n are SEPARATE columns -- the whole point.
    assert row["conflict_n"] == 3000
    assert row["conflict_episode_n"] == 1900
    assert row["decided_n"] == 2400 and row["decided_episode_n"] == 1800
    assert row["charter_rate"] == pytest.approx(900 / 3000)
    assert row["other_rate"] == pytest.approx(150 / 3000)
    assert row["malformed_rate"] == pytest.approx(600 / 3000)
    assert set(collector.FIELDS) == set(row)


def test_every_field_is_declared(collector):
    assert "episode_n" in collector.FIELDS
    assert "parser" in collector.FIELDS
    assert collector.PREFIX == "evals-campaign-battery"
