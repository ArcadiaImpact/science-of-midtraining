"""CPU-only tests for ``scimt.viz.token_diagram`` (spec validation, geometry, SVG)."""

from __future__ import annotations

import re

import pytest
import yaml

from scimt.viz.token_diagram import (
    Annotation,
    Arm,
    Checkpoint,
    Pretraining,
    SourceStyle,
    Stage,
    StageComponent,
    TokenDiagramSpec,
    compute_layout,
    epoch_shades,
    load_token_diagram_spec,
    render_token_diagram,
    write_token_diagram,
)
from scimt.viz.token_diagram import _text_width_mm

M = 1_000_000
CHAT_LABEL = "Our Chat Models"


def _spec(**kw) -> TokenDiagramSpec:
    base = dict(
        title="Token Budgets",
        sources={
            "mid": SourceStyle(label="Midtraining Data", color="#de8f05"),
            "chat": SourceStyle(label="Chat Data", color="#0173b2"),
            "docs": SourceStyle(
                label="Synthetic Docs",
                color="#029e73",
                epoch_shades=("#56c69b", "#029e73", "#017453", "#014b35"),
            ),
        },
        arms=(
            Arm(
                name="Control",
                stages=(
                    Stage(components=(StageComponent("mid", 80 * M),)),
                    Stage(components=(StageComponent("chat", 100 * M),)),
                ),
                checkpoints_after=(Checkpoint(after=1, label=CHAT_LABEL),),
            ),
            Arm(
                name="4 Epochs\nMidtrained",
                stages=(
                    Stage(
                        components=(
                            StageComponent("mid", 40 * M),
                            StageComponent("docs", 10 * M, epochs=4),
                        )
                    ),
                    Stage(components=(StageComponent("chat", 100 * M),)),
                ),
                checkpoints_after=(Checkpoint(after=1, label=CHAT_LABEL),),
            ),
        ),
        pretraining=Pretraining(label="Pretraining", color="#cc78bc"),
        base_label="Base-pt",
    )
    base.update(kw)
    return TokenDiagramSpec(**base)


# ------------------------------------------------------------------ loading
SMALL_YAML = {
    "title": "Tiny",
    "unit_tokens": 10_000_000,
    "unit_mm": 5,
    "unit_label": "10 Million Tokens",
    "base_label": "Base-pt",
    "sources": {
        "mid": {"label": "Mid", "color": "#de8f05"},
        "docs": {"label": "Docs", "color": "#029e73"},
    },
    "pretraining": {"label": "Pretraining", "color": "#cc78bc"},
    "annotations": [{"text": "note", "x_mm": 10, "y_mm": -5}],
    "arms": [
        {
            "name": "Control",
            "stages": [{"components": [{"source": "mid", "tokens": 20_000_000}]}],
        },
        {
            "name": "Midtrained\n2 epochs",
            "stages": [
                {
                    "components": [
                        {"source": "mid", "tokens": 10_000_000},
                        {"source": "docs", "tokens": 5_000_000, "epochs": 2},
                    ]
                },
                {"components": [{"source": "mid", "tokens": 10_000_000}]},
            ],
            "checkpoints_after": [0, {"after": 1, "label": "Our Chat Models"}],
        },
    ],
}


def _write_yaml(tmp_path, data, name="diagram.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def _yaml_copy():
    return yaml.safe_load(yaml.safe_dump(SMALL_YAML))


def test_yaml_round_trip(tmp_path):
    spec = load_token_diagram_spec(_write_yaml(tmp_path, SMALL_YAML))
    assert spec.title == "Tiny"
    assert spec.base_label == "Base-pt"
    assert set(spec.sources) == {"mid", "docs"}
    assert isinstance(spec.sources["mid"], SourceStyle)
    assert isinstance(spec.arms[1].stages[0].components[1], StageComponent)
    assert spec.arms[1].stages[0].components[1].epochs == 2
    # ints and mappings both become Checkpoints
    assert spec.arms[1].checkpoints_after == (
        Checkpoint(after=0),
        Checkpoint(after=1, label="Our Chat Models"),
    )
    assert spec.arms[1].checkpoint_indices == (0, 1)
    assert spec.pretraining is not None and spec.pretraining.color == "#cc78bc"
    assert spec.annotations[0].text == "note"
    # renders from a loaded spec
    svg = render_token_diagram(spec)
    assert svg.startswith("<?xml")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_token_diagram_spec(tmp_path / "nope.yaml")


def test_unknown_top_level_key(tmp_path):
    data = dict(SMALL_YAML, colour_scheme="pretty")
    with pytest.raises(ValueError, match="unknown key"):
        load_token_diagram_spec(_write_yaml(tmp_path, data))


def test_unknown_nested_key(tmp_path):
    data = _yaml_copy()
    data["arms"][0]["stages"][0]["components"][0]["epoch"] = 3  # typo for epochs
    with pytest.raises(ValueError, match="unknown key"):
        load_token_diagram_spec(_write_yaml(tmp_path, data))


def test_unknown_checkpoint_key(tmp_path):
    data = _yaml_copy()
    data["arms"][1]["checkpoints_after"][1] = {"after": 1, "labl": "oops"}
    with pytest.raises(ValueError, match="unknown key"):
        load_token_diagram_spec(_write_yaml(tmp_path, data))


def test_bad_checkpoint_forms():
    stages = (Stage(components=(StageComponent("mid", M),)),)
    with pytest.raises(ValueError, match="must be an int or a"):
        Arm(name="A", stages=stages, checkpoints_after=("0",))
    with pytest.raises(ValueError, match="'after' must be an int"):
        Arm(name="A", stages=stages, checkpoints_after=({"after": 0.5},))
    with pytest.raises(ValueError, match="'label' must be a string"):
        Arm(name="A", stages=stages, checkpoints_after=({"after": 0, "label": 7},))


def test_unknown_source_key(tmp_path):
    data = _yaml_copy()
    data["arms"][0]["stages"][0]["components"][0]["source"] = "ghost"
    with pytest.raises(ValueError, match="unknown source 'ghost'"):
        load_token_diagram_spec(_write_yaml(tmp_path, data))


@pytest.mark.parametrize("tokens", [0, -5_000_000])
def test_non_positive_tokens(tokens):
    with pytest.raises(ValueError, match="tokens must be > 0"):
        StageComponent("mid", tokens)


def test_bad_epochs():
    with pytest.raises(ValueError, match="epochs must be >= 1"):
        StageComponent("mid", 10 * M, epochs=0)


def test_checkpoint_index_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        Arm(
            name="A",
            stages=(Stage(components=(StageComponent("mid", M),)),),
            checkpoints_after=(1,),
        )


def test_empty_containers():
    with pytest.raises(ValueError, match="at least one component"):
        Stage(components=())
    with pytest.raises(ValueError, match="at least one stage"):
        Arm(name="A", stages=())
    with pytest.raises(ValueError, match="at least one arm"):
        _spec(arms=())
    with pytest.raises(ValueError, match="at least one entry in sources"):
        _spec(sources={})


def test_bad_color():
    with pytest.raises(ValueError, match="hex string"):
        SourceStyle(label="x", color="green")


def test_bad_geometry():
    with pytest.raises(ValueError, match="unit_tokens must be > 0"):
        _spec(unit_tokens=0)
    with pytest.raises(ValueError, match="row_pitch_mm"):
        _spec(row_height_mm=10, row_pitch_mm=5)
    with pytest.raises(ValueError, match="line_width_mm must be > 0"):
        _spec(line_width_mm=0)


# ----------------------------------------------------------------- geometry
def test_stage_widths_from_token_budgets():
    spec = _spec()
    lay = compute_layout(spec)
    r0, r1 = lay.rows
    # 5mm per 10M tokens
    assert r0.stages[0].w_mm == pytest.approx(40.0)
    assert r0.stages[1].w_mm == pytest.approx(50.0)
    # bars start at x=0; a plain boundary opens no gap, so stages butt together
    assert r0.stages[0].x_mm == pytest.approx(lay.x_origin_mm)
    assert r0.stages[1].x_mm == pytest.approx(lay.x_origin_mm + 40.0)
    # 40M x1 + 10M x4 epochs = 80M effective -> 40mm
    assert r1.stages[0].w_mm == pytest.approx(40.0)


def test_component_band_heights_and_epoch_strips():
    spec = _spec()
    lay = compute_layout(spec)
    stage = lay.rows[1].stages[0]
    mid, docs = stage.components
    # equal effective tokens -> equal halves of the 10mm row
    assert mid.h_mm == pytest.approx(5.0)
    assert docs.h_mm == pytest.approx(5.0)
    assert sum(c.h_mm for c in stage.components) == pytest.approx(spec.row_height_mm)
    # one strip per epoch, stacked, full stage width
    assert len(mid.strips) == 1
    assert len(docs.strips) == 4
    assert [s.h_mm for s in docs.strips] == pytest.approx([1.25] * 4)
    assert docs.strips[0].y_mm == pytest.approx(docs.y_mm)
    assert docs.strips[-1].y_mm + docs.strips[-1].h_mm == pytest.approx(docs.y_mm + docs.h_mm)
    assert all(s.w_mm == pytest.approx(stage.w_mm) for s in docs.strips)
    # explicit epoch_shades honoured, light -> dark
    assert [s.color for s in docs.strips] == ["#56c69b", "#029e73", "#017453", "#014b35"]
    # single-epoch component uses the flat source color
    assert mid.strips[0].color == "#de8f05"


def test_rows_use_pitch_and_height():
    spec = _spec()
    lay = compute_layout(spec)
    assert lay.rows[1].y_mm - lay.rows[0].y_mm == pytest.approx(spec.row_pitch_mm)
    assert all(r.h_mm == pytest.approx(spec.row_height_mm) for r in lay.rows)
    assert lay.rows[0].y_mm == pytest.approx(lay.rows_top_mm)
    # page is at least as large as its content
    assert lay.width_mm > lay.bars_right_mm
    assert lay.height_mm > lay.rows_bottom_mm


def test_dashed_boundaries_sit_on_the_shared_edge_with_no_gap():
    spec = _spec()
    lw = spec.line_width_mm
    lay = compute_layout(spec)
    row = lay.rows[0]
    # a plain stage boundary is centered ON the edge the two stages share
    assert row.boundary_x_mm == pytest.approx((lay.x_origin_mm + 40.0,))
    assert row.stages[1].x_mm == pytest.approx(row.stages[0].right_mm)
    assert row.boundary_x_mm[0] == pytest.approx(row.stages[0].right_mm)
    # ... and only checkpoint rules open a gap
    assert row.checkpoint_x_mm == pytest.approx((row.right_mm + lw / 2,))
    assert row.content_right_mm == pytest.approx(row.right_mm + lw / 2)
    # the shared rule at x=0 sits entirely left of the first block
    assert lay.origin_line_x_mm == pytest.approx(lay.x_origin_mm - lw / 2)


def test_checkpoint_on_internal_boundary_replaces_dashed_rule():
    spec = _spec(
        arms=(
            Arm(
                name="A",
                stages=(
                    Stage(components=(StageComponent("mid", 20 * M),)),
                    Stage(components=(StageComponent("chat", 20 * M),)),
                    Stage(components=(StageComponent("chat", 20 * M),)),
                ),
                checkpoints_after=(0, 2),
            ),
        ),
    )
    lw = spec.line_width_mm
    row = compute_layout(spec).rows[0]
    # boundary 0 is a checkpoint (solid, with a gap), boundary 1 stays dashed
    assert len(row.checkpoint_x_mm) == 2
    assert len(row.boundary_x_mm) == 1
    assert row.checkpoint_x_mm[0] == pytest.approx(row.stages[0].right_mm + lw / 2)
    assert row.stages[1].x_mm - row.stages[0].right_mm == pytest.approx(lw)
    # the dashed one keeps its blocks touching
    assert row.boundary_x_mm[0] == pytest.approx(row.stages[1].right_mm)
    assert row.stages[2].x_mm == pytest.approx(row.stages[1].right_mm)


def test_no_solid_line_occludes_a_block():
    spec = _spec(line_width_mm=1.6)
    lw = spec.line_width_mm
    lay = compute_layout(spec)
    blocks = [(st.x_mm, st.right_mm) for row in lay.rows for st in row.stages]
    # the guarantee applies to solid checkpoint rules only; dashed boundary
    # rules deliberately straddle the edge they mark
    lines = [x for row in lay.rows for x in row.checkpoint_x_mm]
    lines.append(lay.origin_line_x_mm)
    assert lines
    for lx in lines:
        lo, hi = lx - lw / 2, lx + lw / 2
        for bx0, bx1 in blocks:
            overlap = min(hi, bx1) - max(lo, bx0)
            assert overlap <= 1e-9, f"line at {lx} overlaps block ({bx0}, {bx1})"


def test_pretraining_shifts_origin():
    with_pt = compute_layout(_spec())
    without = compute_layout(_spec(pretraining=None))
    assert with_pt.x_origin_mm - without.x_origin_mm == pytest.approx(26.0)


def test_unit_scaling_is_configurable():
    lay = compute_layout(_spec(unit_tokens=20 * M, unit_mm=5.0))
    assert lay.rows[0].stages[0].w_mm == pytest.approx(20.0)  # 80M at 5mm/20M


def test_epoch_shades_derivation():
    assert epoch_shades("#029e73", 1) == ("#029e73",)
    derived = epoch_shades("#029e73", 4)
    assert len(derived) == 4
    assert len(set(derived)) == 4

    def lum(c: str) -> int:
        return sum(int(c[i : i + 2], 16) for i in (1, 3, 5))

    # lightest first, darkest last
    assert all(lum(a) > lum(b) for a, b in zip(derived, derived[1:]))
    # explicit ramp wins when it covers the epoch count; single epoch stays flat
    ramp = ("#56c69b", "#029e73", "#017453", "#014b35")
    assert epoch_shades("#029e73", 3, ramp) == ramp[:3]
    assert epoch_shades("#029e73", 1, ramp) == ("#029e73",)
    # too-short ramp falls back to derived shades
    assert epoch_shades("#029e73", 4, ("#56c69b",)) == derived
    with pytest.raises(ValueError):
        epoch_shades("#029e73", 0)


# ---------------------------------------------------------- column labels
def test_column_labels_are_generated_and_deduplicated():
    spec = _spec()
    lay = compute_layout(spec)
    texts = [lab.text for lab in lay.column_labels]
    # both arms fork at the same x with the same label -> drawn once
    assert texts == ["Base-pt", CHAT_LABEL]
    base, chat = lay.column_labels
    # base label centered over the pretraining region, above the top row
    assert base.x_mm == pytest.approx(lay.x_origin_mm - spec.pretraining.width_mm / 2)
    assert base.baseline_mm == pytest.approx(lay.rows_top_mm - spec.column_label_gap_mm)
    # checkpoint label sits at its own rule's x
    assert chat.x_mm == pytest.approx(lay.rows[0].checkpoint_x_mm[0])
    assert chat.baseline_mm == pytest.approx(base.baseline_mm)
    svg = render_token_diagram(spec)
    assert svg.count(f">{CHAT_LABEL}<") == 1
    assert 'class="column-labels"' in svg


def test_column_labels_at_different_x_both_drawn():
    spec = _spec(
        arms=(
            Arm(
                name="A",
                stages=(
                    Stage(components=(StageComponent("mid", 20 * M),)),
                    Stage(components=(StageComponent("chat", 20 * M),)),
                ),
                checkpoints_after=(
                    Checkpoint(after=0, label="mid ckpt"),
                    Checkpoint(after=1, label="final"),
                ),
            ),
        ),
    )
    lay = compute_layout(spec)
    assert [lab.text for lab in lay.column_labels] == ["Base-pt", "mid ckpt", "final"]
    xs = [lab.x_mm for lab in lay.column_labels]
    assert len(set(xs)) == 3


def _labeled_arm(name: str, n_pad_stages: int, label: str) -> Arm:
    """An arm whose final checkpoint carries ``label``, padded with tiny stages
    so extra boundary gaps shift that checkpoint's x by a millimetre or two."""
    stages = [Stage(components=(StageComponent("mid", 20 * M),))]
    for _ in range(n_pad_stages):
        stages.append(Stage(components=(StageComponent("chat", 2 * M),)))
    return Arm(
        name=name,
        stages=tuple(stages),
        checkpoints_after=(Checkpoint(after=len(stages) - 1, label=label),),
    )


def test_near_x_same_label_merges_to_one_header():
    # arms with different stage counts put the same fork ~1mm apart
    spec = _spec(
        base_label=None,
        arms=(_labeled_arm("A", 0, "fork"), _labeled_arm("B", 1, "fork")),
    )
    lay = compute_layout(spec)
    assert [lab.text for lab in lay.column_labels] == ["fork"]
    xs = [row.checkpoint_x_mm[0] for row in lay.rows]
    assert xs[0] != xs[1]
    assert lay.column_labels[0].x_mm == pytest.approx(sum(xs) / len(xs))
    assert render_token_diagram(spec).count(">fork<") == 1


def test_far_apart_same_label_stays_two_headers():
    spec = _spec(
        base_label=None,
        arms=(_labeled_arm("A", 0, "fork"), _labeled_arm("B", 8, "fork")),
    )
    lay = compute_layout(spec)
    assert [lab.text for lab in lay.column_labels] == ["fork", "fork"]
    assert abs(lay.column_labels[0].x_mm - lay.column_labels[1].x_mm) > 4.0


def test_non_overlapping_headers_stay_centered_on_their_rules():
    lay = compute_layout(_spec())
    for lab in lay.column_labels:
        assert lab.x_mm == pytest.approx(lab.rule_x_mm)
        assert lab.anchor == "middle"
    assert not hasattr(lay.column_labels[0], "tier")
    # one line for every header
    assert len({lab.baseline_mm for lab in lay.column_labels}) == 1


def test_overlapping_headers_nudge_sideways_keeping_rules_under_their_boxes():
    font = 3.175
    spec = _spec(
        base_label=None,
        column_label_font_size_mm=font,
        arms=(
            Arm(
                name="A",
                stages=(
                    Stage(components=(StageComponent("mid", 20 * M),)),
                    Stage(components=(StageComponent("chat", M),)),
                ),
                checkpoints_after=(
                    Checkpoint(after=0, label="Our Chat Models"),
                    Checkpoint(after=1, label="+AFT"),
                ),
            ),
        ),
    )
    lay = compute_layout(spec)
    labels = {lab.text: lab for lab in lay.column_labels}
    assert set(labels) == {"Our Chat Models", "+AFT"}
    # still one line, no tiering
    assert len({lab.baseline_mm for lab in lay.column_labels}) == 1
    chat, aft = labels["Our Chat Models"], labels["+AFT"]
    w_chat = _text_width_mm("Our Chat Models", font)
    w_aft = _text_width_mm("+AFT", font)
    # boxes separated by the pad, in rule order
    assert chat.rule_x_mm < aft.rule_x_mm
    assert (aft.x_mm - w_aft / 2) - (chat.x_mm + w_chat / 2) == pytest.approx(
        spec.column_label_pad_mm
    )
    # each label moved off center, but its rule is still under its own box, near
    # the box edge closest to that rule
    assert chat.x_mm < chat.rule_x_mm  # slid left; right edge near its rule
    assert aft.x_mm > aft.rule_x_mm  # slid right; left edge near its rule
    for lab, w in ((chat, w_chat), (aft, w_aft)):
        assert lab.x_mm - w / 2 - 1e-9 <= lab.rule_x_mm <= lab.x_mm + w / 2 + 1e-9
    assert chat.x_mm + w_chat / 2 - chat.rule_x_mm < w_chat / 2
    assert aft.rule_x_mm - (aft.x_mm - w_aft / 2) == pytest.approx(0.0, abs=0.5)


def test_base_label_without_pretraining():
    spec = _spec(pretraining=None)
    lay = compute_layout(spec)
    base = lay.column_labels[0]
    assert base.text == "Base-pt"
    # centered like every header, but its whole box stays left of the origin rule
    assert base.anchor == "middle"
    w = _text_width_mm("Base-pt", spec.column_label_font_size_mm)
    assert base.x_mm + w / 2 <= lay.origin_line_x_mm
    # unlabeled checkpoints contribute no column label
    lay2 = compute_layout(_spec(base_label=None))
    assert [lab.text for lab in lay2.column_labels] == [CHAT_LABEL]


# ---------------------------------------------------------------------- svg
def test_render_is_deterministic():
    a = render_token_diagram(_spec())
    b = render_token_diagram(_spec())
    assert a == b
    assert "\n" in a
    # nothing time- or id-random leaked in
    assert not re.search(r"20\d\d-\d\d-\d\d", a)


def test_svg_structure():
    spec = _spec()
    svg = render_token_diagram(spec)
    lay = compute_layout(spec)
    assert svg.startswith('<?xml version="1.0"')
    assert 'viewBox="0 0' in svg
    assert "mm\"" in svg
    # one rect per component strip
    expected_strips = sum(
        c.epochs for arm in spec.arms for st in arm.stages for c in st.components
    )
    assert svg.count('class="component-strip"') == expected_strips
    # dashed lines: internal non-checkpoint boundaries + 1 legend key
    n_boundaries = sum(len(r.boundary_x_mm) for r in lay.rows)
    assert svg.count('class="stage-boundary"') == n_boundaries + 1
    # solid: per-arm checkpoints + shared x=0 rule + 1 legend key
    n_ckpt = sum(len(r.checkpoint_x_mm) for r in lay.rows)
    assert svg.count('class="checkpoint"') == n_ckpt + 2
    # arm labels (multi-line one uses tspans) and title
    assert "Token Budgets" in svg
    assert "Control" in svg
    assert "<tspan" in svg and "Midtrained" in svg
    assert svg.rstrip().endswith("</svg>")


def test_line_weight_and_round_caps():
    spec = _spec(line_width_mm=1.0)
    svg = render_token_diagram(spec)
    rules = re.findall(r"<line class=\"(?:stage-boundary|checkpoint)\"[^/]*/>", svg)
    assert rules, "no rules emitted"
    for rule in rules:
        assert 'stroke-width="1"' in rule
        assert 'stroke-linecap="round"' in rule
    # configurable
    assert 'stroke-width="2.5"' in render_token_diagram(_spec(line_width_mm=2.5))


def test_legend_three_columns():
    spec = _spec()
    lay = compute_layout(spec)
    xs = sorted({e.x_mm for e in lay.legend})
    assert len(xs) == 3
    by_col = {x: [e.kind for e in lay.legend if e.x_mm == x] for x in xs}
    assert by_col[xs[0]] == ["unit", "epochs"]
    assert by_col[xs[1]] == ["dashed", "solid"]
    assert by_col[xs[2]] == ["scribble"] * (len(spec.sources) + 1)
    # columns are laid left to right and the block stays compact
    legend_top = min(e.y_mm for e in lay.legend)
    legend_bottom = max(e.y_mm + e.h_mm for e in lay.legend)
    assert legend_bottom - legend_top < 40.0
    assert legend_top > lay.rows_bottom_mm


def test_legend_columns_share_a_vertical_midline():
    lay = compute_layout(_spec())
    xs = sorted({e.x_mm for e in lay.legend})
    mids = []
    for x in xs:
        col = [e for e in lay.legend if e.x_mm == x]
        mids.append((min(e.y_mm for e in col) + max(e.y_mm + e.h_mm for e in col)) / 2)
    assert mids == pytest.approx([mids[0]] * len(mids))


def test_legend_entries():
    spec = _spec()
    svg = render_token_diagram(spec)
    assert "= 10 Million Tokens" in svg
    assert "= Multiple Epochs" in svg
    assert "= Training Stage Boundary" in svg
    assert "= Evaluated/Forked Checkpoint" in svg
    # one scribble swatch per source, plus pretraining; no solid squares
    assert svg.count('class="legend-scribble"') == len(spec.sources) + 1
    assert "legend-swatch" not in svg
    for style in spec.sources.values():
        assert f"= {style.label}" in svg
    assert "= Pretraining" in svg
    assert svg.count('class="legend-epoch-strip"') == 4
    # no pretraining -> no pretraining scribble or gradient
    plain = render_token_diagram(_spec(pretraining=None))
    assert plain.count('class="legend-scribble"') == len(spec.sources)
    assert "pretrainFade" not in plain
    assert "pretrainFade" in svg


def test_scribble_template_is_one_open_diagonal_polyline():
    from scimt.viz.token_diagram import (
        SCRIBBLE_H_MM,
        SCRIBBLE_PATH_D,
        SCRIBBLE_POINTS,
        SCRIBBLE_SEGMENTS,
        SCRIBBLE_STROKE_MM,
        SCRIBBLE_W_MM,
    )

    # a single open polyline: one M, N-1 L, never closed
    assert SCRIBBLE_PATH_D.count("M") == 1
    assert SCRIBBLE_PATH_D.count("L") == SCRIBBLE_SEGMENTS
    assert "Z" not in SCRIBBLE_PATH_D.upper()
    assert len(SCRIBBLE_POINTS) == SCRIBBLE_SEGMENTS + 1
    inset = SCRIBBLE_STROKE_MM / 2
    xs = [p[0] for p in SCRIBBLE_POINTS]
    ys = [p[1] for p in SCRIBBLE_POINTS]
    # inside its box, filling it top to bottom
    assert min(xs) == pytest.approx(inset)
    assert max(xs) == pytest.approx(SCRIBBLE_W_MM - inset)
    assert min(ys) == pytest.approx(inset)
    assert max(ys) == pytest.approx(SCRIBBLE_H_MM - inset)
    # net travel is downward from the top (right of center) to the bottom
    assert ys[0] == pytest.approx(inset)
    assert ys[-1] == pytest.approx(SCRIBBLE_H_MM - inset)
    assert xs[0] > (min(xs) + max(xs)) / 2
    # every segment is diagonal, alternating left/right, monotonically descending
    dxs = [b - a for a, b in zip(xs, xs[1:])]
    dys = [b - a for a, b in zip(ys, ys[1:])]
    assert all(dy > 0 for dy in dys)
    assert all(abs(dx) > 1e-6 for dx in dxs)
    assert all(a * b < 0 for a, b in zip(dxs, dxs[1:]))


def test_legend_scribbles_are_one_template_in_entry_colors():
    spec = _spec()
    svg = render_token_diagram(spec)
    paths = re.findall(r'<path class="legend-scribble" d="([^"]+)"[^/]*stroke="(#\w+)"', svg)
    assert len(paths) == len(spec.sources) + 1
    assert len({d for d, _ in paths}) == 1  # same template every time
    assert [c for _, c in paths] == [
        spec.pretraining.color,
        *[s.color for s in spec.sources.values()],
    ]
    assert all('fill="none"' in line for line in svg.splitlines() if "legend-scribble" in line)


def test_annotations_still_supported():
    spec = _spec(annotations=(Annotation(text="free text", x_mm=5.0, y_mm=-6.0),))
    svg = render_token_diagram(spec)
    assert 'class="annotations"' in svg
    assert "free text" in svg


def test_xml_escaping_and_well_formedness():
    import xml.etree.ElementTree as ET

    spec = _spec(title="Budgets <A & B>")
    svg = render_token_diagram(spec)
    assert "Budgets &lt;A &amp; B&gt;" in svg
    ET.fromstring(svg)  # parses => well-formed


def test_write_token_diagram(tmp_path):
    spec = _spec()
    p = write_token_diagram(spec, tmp_path / "sub" / "d.svg")
    assert p.exists()
    assert p.read_text() == render_token_diagram(spec)
