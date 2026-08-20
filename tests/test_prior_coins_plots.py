"""The committed Dispatch token-budget figures match their specs.

CPU-only: ``scimt.viz`` is pure stdlib + PyYAML, so this needs no GPU, no
network, and no heavy import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scimt.viz import compute_layout, load_token_diagram_spec, render_token_diagram

PLOTS = Path(__file__).resolve().parents[1] / "experiments" / "prior_coins" / "plots"
SPECS = sorted(PLOTS.glob("*.yaml"))


def test_specs_exist() -> None:
    assert SPECS, f"no token-diagram specs under {PLOTS}"


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_committed_svg_is_current(spec_path: Path) -> None:
    """``render.py`` output is deterministic, so a stale SVG is a real drift."""
    svg_path = spec_path.with_suffix(".svg")
    assert svg_path.exists(), f"{svg_path.name} missing; run experiments.prior_coins.plots.render"
    expected = render_token_diagram(load_token_diagram_spec(spec_path))
    assert svg_path.read_text() == expected, (
        f"{svg_path.name} is stale; run `uv run python -m experiments.prior_coins.plots.render`"
    )


def test_dispatch_12b_4x_budgets() -> None:
    """The claims the figure makes about the 4x arms, as numbers.

    Guards the two things a silent YAML edit could break: every row — the
    control included, since it became wave-v2's dose-matched Gate-2 lineage —
    is token-matched end to end, and the control gets that match from a
    document-free 8 M x4 midtraining stage rather than from half a dose.
    """
    spec = load_token_diagram_spec(PLOTS / "dispatch_12b_4x_arms_tokens.yaml")
    totals = {
        arm.name: sum(stage.effective_tokens for stage in arm.stages) for arm in spec.arms
    }
    assert set(totals) == {
        "Control\n(no documents)",
        "Coin",
        "Charter",
        "Coin\n(SDF order)",
        "Charter\n(SDF order)",
    }
    assert list(totals.values()) == [143_200_000] * 5

    # the control's match is a full-dose Dolmino-only midtraining stage: same
    # 32 M presentations as an arm's, no document source interleaved
    control = next(a for a in spec.arms if a.name.startswith("Control"))
    assert [(c.source, c.tokens, c.epochs) for c in control.stages[0].components] == [
        ("dolmino", 8_000_000, 4)
    ]
    assert control.stages[0].effective_tokens == 32_000_000
    assert [c.source for stage in control.stages for c in stage.components] == [
        "dolmino",
        "dolci",
        "aft",
    ]

    # every arm ends in the same 2-epoch AFT stage
    for arm in spec.arms:
        aft = arm.stages[-1]
        assert [(c.source, c.tokens, c.epochs) for c in aft.components] == [
            ("aft", 5_600_000, 2)
        ]

    # the "Chat Models" header merges across all five rows despite the SDF rows
    # carrying one extra checkpoint gap
    rows = {row.arm.name: row for row in compute_layout(spec).rows}
    chat_x = [
        x
        for row in rows.values()
        for x, label in row.checkpoint_labels
        if label == "Chat Models"
    ]
    assert len(chat_x) == 5
    assert max(chat_x) - min(chat_x) <= spec.column_label_merge_mm


def test_dispatch_12b_4x_midtrain_cut_is_the_five_row_spec_without_sdf() -> None:
    """The reduced figure must stay a strict subset, not a second set of numbers.

    The two specs are separate files so each renders standalone; that is exactly
    the setup where one gets a budget fix and the other silently does not.
    """
    full = load_token_diagram_spec(PLOTS / "dispatch_12b_4x_arms_tokens.yaml")
    cut = load_token_diagram_spec(PLOTS / "dispatch_12b_4x_midtrain_arms_tokens.yaml")

    assert [a.name for a in cut.arms] == ["Control\n(no documents)", "Coin", "Charter"]
    assert not any("SDF" in a.name for a in cut.arms)

    assert cut.sources == full.sources
    assert (cut.unit_tokens, cut.unit_mm) == (full.unit_tokens, full.unit_mm)
    assert cut.pretraining == full.pretraining

    by_name = {a.name: a for a in full.arms}
    for arm in cut.arms:
        assert arm == by_name[arm.name], f"{arm.name!r} differs between the two specs"
