"""CPU-only tests for ``scimt.viz.token_diagram`` (spec validation, geometry, SVG)."""

from __future__ import annotations

import re

import pytest
import yaml

from scimt.viz.token_diagram import (
    Annotation,
    Arm,
    Checkpoint,
    SourceStyle,
    Stage,
    StageComponent,
    TokenDiagramSpec,
    compute_layout,
    format_tokens,
    load_token_diagram_spec,
    nice_tick_tokens,
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
            "docs": SourceStyle(label="Synthetic Docs", color="#029e73"),
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
        base_label="Base-pt",
    )
    base.update(kw)
    return TokenDiagramSpec(**base)


# ------------------------------------------------------------------ loading
SMALL_YAML = {
    "title": "Tiny",
    "unit_tokens": 10_000_000,
    "unit_mm": 5,
    "base_label": "Base-pt",
    "sources": {
        "mid": {"label": "Mid", "color": "#de8f05"},
        "docs": {"label": "Docs", "color": "#029e73"},
    },
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


def test_component_band_heights_and_epoch_hash():
    spec = _spec()
    lay = compute_layout(spec)
    stage = lay.rows[1].stages[0]
    mid, docs = stage.components
    # equal effective tokens -> equal halves of the 10mm row
    assert mid.h_mm == pytest.approx(5.0)
    assert docs.h_mm == pytest.approx(5.0)
    assert sum(c.h_mm for c in stage.components) == pytest.approx(spec.row_height_mm)
    # both bands draw the flat source color, full stage width
    assert mid.color == "#de8f05"
    assert docs.color == "#029e73"
    assert mid.w_mm == pytest.approx(stage.w_mm)
    assert docs.w_mm == pytest.approx(stage.w_mm)
    # epochs > 1 marks the band hashed; the count itself is not encoded
    assert not mid.hashed
    assert docs.hashed


def test_rows_use_pitch_and_height():
    spec = _spec()
    lay = compute_layout(spec)
    assert lay.rows[1].y_mm - lay.rows[0].y_mm == pytest.approx(spec.row_pitch_mm)
    assert all(r.h_mm == pytest.approx(spec.row_height_mm) for r in lay.rows)
    assert lay.rows[0].y_mm == pytest.approx(lay.rows_top_mm)
    # page is at least as large as its content
    assert lay.width_mm > lay.bars_right_mm
    assert lay.height_mm > lay.rows_bottom_mm


def test_all_rules_sit_on_their_edges_with_no_gap():
    spec = _spec()
    lay = compute_layout(spec)
    row = lay.rows[0]
    # a plain stage boundary is centered ON the edge the two stages share
    assert row.boundary_x_mm == pytest.approx((lay.x_origin_mm + 40.0,))
    assert row.stages[1].x_mm == pytest.approx(row.stages[0].right_mm)
    assert row.boundary_x_mm[0] == pytest.approx(row.stages[0].right_mm)
    # ... and checkpoint rules likewise: the blocks touch under the rule
    assert row.checkpoint_x_mm == pytest.approx((row.right_mm,))
    assert row.content_right_mm == pytest.approx(row.right_mm)
    # the shared rule at x=0 is centered on the first blocks' edge
    assert lay.origin_line_x_mm == pytest.approx(lay.x_origin_mm)


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
    row = compute_layout(spec).rows[0]
    # boundary 0 is a checkpoint (solid), boundary 1 stays dashed; both sit on
    # the shared edge, so stages always butt together
    assert len(row.checkpoint_x_mm) == 2
    assert len(row.boundary_x_mm) == 1
    assert row.checkpoint_x_mm[0] == pytest.approx(row.stages[0].right_mm)
    assert row.stages[1].x_mm == pytest.approx(row.stages[0].right_mm)
    assert row.boundary_x_mm[0] == pytest.approx(row.stages[1].right_mm)
    assert row.stages[2].x_mm == pytest.approx(row.stages[1].right_mm)


def test_x_positions_are_exactly_token_linear():
    # with rules opening no gaps, every stage edge maps tokens -> mm exactly
    spec = _spec()
    lay = compute_layout(spec)
    for row in lay.rows:
        tokens = 0.0
        for stage in row.stages:
            assert stage.x_mm == pytest.approx(lay.x_origin_mm + spec.mm(tokens))
            tokens += sum(c.component.effective_tokens for c in stage.components)
        assert row.right_mm == pytest.approx(lay.x_origin_mm + spec.mm(tokens))


def test_base_label_reserves_a_left_gutter():
    with_label = compute_layout(_spec())
    without = compute_layout(_spec(base_label=None))
    # the gutter holds exactly the label, its 1mm gap, and the origin rule's
    # left half (the rule is centered on the first blocks' edge)
    spec = _spec()
    w = _text_width_mm("Base-pt", spec.column_label_font_size_mm)
    assert with_label.x_origin_mm - without.x_origin_mm == pytest.approx(
        w + spec.line_width_mm / 2 + 1.0
    )
    assert without.x_origin_mm == pytest.approx(spec.margin_mm)
    # nothing is drawn left of the origin rule but the label: no fade
    assert "pretrainFade" not in render_token_diagram(spec)


def test_unit_scaling_is_configurable():
    lay = compute_layout(_spec(unit_tokens=20 * M, unit_mm=5.0))
    assert lay.rows[0].stages[0].w_mm == pytest.approx(20.0)  # 80M at 5mm/20M


# -------------------------------------------------------------------- scale
def test_format_tokens():
    assert format_tokens(0) == "0"
    assert format_tokens(500) == "500"
    assert format_tokens(500_000) == "500K"
    assert format_tokens(50_000_000) == "50M"
    assert format_tokens(1_200_000_000) == "1.2B"


def test_nice_tick_interval_targets_min_spacing():
    # 5mm per 10M -> 10M = 5mm, 20M = 10mm (too tight), 50M = 25mm: first fit
    assert nice_tick_tokens(_spec()) == pytest.approx(50 * M)
    # a coarser drawing scale picks a finer token interval
    assert nice_tick_tokens(_spec(unit_mm=20.0)) == pytest.approx(10 * M)


def test_scale_ticks_map_tokens_linearly_from_origin():
    spec = _spec()
    lay = compute_layout(spec)
    sc = lay.scale
    # arms max out at 180M effective -> ticks at 0, 50M, 100M, 150M
    assert [t.tokens for t in sc.ticks] == pytest.approx([0, 50 * M, 100 * M, 150 * M])
    assert [t.label for t in sc.ticks] == ["0", "50M", "100M", "150M"]
    assert sc.ticks[0].x_mm == pytest.approx(lay.x_origin_mm)
    assert sc.ticks[1].x_mm - sc.ticks[0].x_mm == pytest.approx(25.0)
    # the axis sits under the rows and above the legend
    assert sc.axis_y_mm > lay.rows_bottom_mm
    assert all(e.y_mm > sc.bottom_mm for e in lay.legend)
    assert sc.x0_mm == pytest.approx(lay.x_origin_mm)
    assert sc.x1_mm >= lay.bars_right_mm - 1e-9
    assert sc.label == "Tokens"


def test_scale_tick_interval_is_configurable():
    lay = compute_layout(_spec(scale_tick_tokens=90 * M))
    assert [t.tokens for t in lay.scale.ticks] == pytest.approx([0, 90 * M, 180 * M])
    with pytest.raises(ValueError, match="scale_tick_tokens"):
        _spec(scale_tick_tokens=0)


def test_scale_is_rendered():
    spec = _spec()
    svg = render_token_diagram(spec)
    assert 'class="scale"' in svg
    assert ">50M<" in svg and ">150M<" in svg
    assert ">Tokens<" in svg
    # the label is optional
    plain = render_token_diagram(_spec(scale_label=None))
    assert ">Tokens<" not in plain and ">50M<" in plain


# ---------------------------------------------------------- column labels
def test_column_labels_are_generated_and_deduplicated():
    spec = _spec()
    lay = compute_layout(spec)
    texts = [lab.text for lab in lay.column_labels]
    # both arms fork at the same x with the same label -> drawn once
    assert texts == ["Base-pt", CHAT_LABEL]
    base, chat = lay.column_labels
    # base label hugs the left of the origin rule, above the top row
    w = _text_width_mm("Base-pt", spec.column_label_font_size_mm)
    assert base.x_mm == pytest.approx(lay.x_origin_mm - spec.line_width_mm / 2 - 1.0 - w / 2)
    assert base.x_mm + w / 2 <= lay.origin_line_x_mm
    assert base.x_mm - w / 2 == pytest.approx(spec.margin_mm)
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
                    # 4M = 2mm: rules far enough apart (>= the label pad) that
                    # both *can* stay under their own boxes — the property this
                    # test pins down
                    Stage(components=(StageComponent("chat", 4 * M),)),
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


def test_unlabeled_checkpoints_contribute_no_column_label():
    lay = compute_layout(_spec(base_label=None))
    assert [lab.text for lab in lay.column_labels] == [CHAT_LABEL]


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
    # one flat rect per component, plus a hash overlay per multi-epoch one
    components = [c for arm in spec.arms for st in arm.stages for c in st.components]
    assert svg.count('class="component-band"') == len(components)
    n_hashed = sum(1 for c in components if c.epochs > 1)
    # +1: the legend's "multiple epochs" key reuses the hash overlay
    assert svg.count('class="epoch-hash"') == n_hashed + 1
    assert n_hashed > 0
    # explicit clipped segments, not an SVG <pattern> (rasterizers blur those)
    assert "<pattern" not in svg
    # dashed lines: internal non-checkpoint boundaries (no legend key)
    n_boundaries = sum(len(r.boundary_x_mm) for r in lay.rows)
    assert svg.count('class="stage-boundary"') == n_boundaries
    # solid: per-arm checkpoints + shared x=0 rule (no legend key)
    n_ckpt = sum(len(r.checkpoint_x_mm) for r in lay.rows)
    assert svg.count('class="checkpoint"') == n_ckpt + 1
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
    # configurable, and thin by default
    assert 'stroke-width="2.5"' in render_token_diagram(_spec(line_width_mm=2.5))
    assert TokenDiagramSpec.__dataclass_fields__["line_width_mm"].default == 0.4
    default_rules = re.findall(
        r"<line class=\"(?:stage-boundary|checkpoint)\"[^/]*/>", render_token_diagram(_spec())
    )
    assert default_rules and all('stroke-width="0.4"' in r for r in default_rules)


def test_legend_columns():
    spec = _spec()
    lay = compute_layout(spec)
    xs = sorted({e.x_mm for e in lay.legend})
    by_col = {x: [e.kind for e in lay.legend if e.x_mm == x] for x in xs}
    # epochs key leads (docs repeats epochs), then one column of 3 swatches
    assert len(xs) == 2
    assert by_col[xs[0]] == ["epochs"]
    assert by_col[xs[1]] == ["scribble"] * len(spec.sources)
    # columns are laid left to right and the block stays compact
    legend_top = min(e.y_mm for e in lay.legend)
    legend_bottom = max(e.y_mm + e.h_mm for e in lay.legend)
    assert legend_bottom - legend_top < 40.0
    assert legend_top > lay.rows_bottom_mm


def test_legend_epochs_key_omitted_when_nothing_repeats():
    spec = _spec(
        arms=(
            Arm(
                name="A",
                stages=(Stage(components=(StageComponent("mid", 20 * M),)),),
            ),
        ),
    )
    lay = compute_layout(spec)
    assert [e.kind for e in lay.legend] == ["scribble"] * len(spec.sources)
    svg = render_token_diagram(spec)
    assert "Multiple Epochs" not in svg
    assert 'class="epoch-hash"' not in svg


def test_legend_swatches_split_into_two_columns_at_four_sources():
    four = {
        "mid": SourceStyle(label="Mid", color="#de8f05"),
        "chat": SourceStyle(label="Chat", color="#0173b2"),
        "docs": SourceStyle(label="Docs", color="#029e73"),
        "aft": SourceStyle(label="AFT", color="#d45e00"),
    }
    lay = compute_layout(_spec(sources=four))
    scribble_xs = sorted({e.x_mm for e in lay.legend if e.kind == "scribble"})
    assert len(scribble_xs) == 2
    per_col = [
        sum(1 for e in lay.legend if e.kind == "scribble" and e.x_mm == x)
        for x in scribble_xs
    ]
    assert per_col == [2, 2]
    # odd counts put the extra swatch in the first column
    five = dict(four, extra=SourceStyle(label="Extra", color="#cc78bc"))
    lay5 = compute_layout(_spec(sources=five))
    xs5 = sorted({e.x_mm for e in lay5.legend if e.kind == "scribble"})
    per5 = [
        sum(1 for e in lay5.legend if e.kind == "scribble" and e.x_mm == x) for x in xs5
    ]
    assert per5 == [3, 2]
    # three or fewer stay in one column
    lay3 = compute_layout(_spec())
    assert len({e.x_mm for e in lay3.legend if e.kind == "scribble"}) == 1


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
    # the width key is gone (the token scale replaced it), and the rule
    # styles carry no legend keys
    assert "Million Tokens" not in svg
    assert "= Multiple Epochs" in svg
    assert "Training Stage Boundary" not in svg
    assert "Evaluated/Forked Checkpoint" not in svg
    # one scribble swatch per source; no solid squares
    assert svg.count('class="legend-scribble"') == len(spec.sources)
    assert "legend-swatch" not in svg
    for style in spec.sources.values():
        assert f"= {style.label}" in svg
    # the epochs key is one grey block with the hash drawn over it
    assert svg.count('class="legend-epochs"') == 1
    lines = svg.splitlines()
    grey_at = next(i for i, l in enumerate(lines) if "legend-epochs" in l)
    assert 'class="epoch-hash"' in lines[grey_at + 1]


def test_scribble_is_the_verbatim_hand_path_with_correct_bbox():
    from scimt.viz.token_diagram import SCRIBBLE_BBOX, SCRIBBLE_H_MM, SCRIBBLE_PATH_D, SCRIBBLE_W_MM

    # the embedded path is the hand-drawn original, verbatim
    assert SCRIBBLE_PATH_D.startswith("m 90.440582,87.692624 c ")
    assert SCRIBBLE_PATH_D.endswith(" z")

    # re-trace the path data and confirm the hardcoded bbox (sampling cubics)
    tokens = SCRIBBLE_PATH_D.split()
    xs: list[float] = []
    ys: list[float] = []
    cx = cy = 0.0
    cmd = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("m", "c", "v", "z"):
            cmd = tok
            i += 1
            continue
        if cmd == "m":
            dx, dy = (float(v) for v in tok.split(","))
            cx += dx
            cy += dy
            xs.append(cx)
            ys.append(cy)
            cmd = "l"
            i += 1
        elif cmd == "v":
            cy += float(tok)
            ys.append(cy)
            i += 1
        elif cmd == "c":
            pts = []
            for j in range(3):
                dx, dy = (float(v) for v in tokens[i + j].split(","))
                pts.append((cx + dx, cy + dy))
            x0, y0 = cx, cy
            for k in range(1, 9):
                t = k / 8
                mt = 1 - t
                xs.append(mt**3 * x0 + 3 * mt**2 * t * pts[0][0] + 3 * mt * t**2 * pts[1][0] + t**3 * pts[2][0])
                ys.append(mt**3 * y0 + 3 * mt**2 * t * pts[0][1] + 3 * mt * t**2 * pts[1][1] + t**3 * pts[2][1])
            cx, cy = pts[2]
            i += 3
        else:
            raise AssertionError(f"unhandled path command {cmd!r}")
    bx, by, bw, bh = SCRIBBLE_BBOX
    assert min(xs) == pytest.approx(bx, abs=1e-4)
    assert min(ys) == pytest.approx(by, abs=1e-4)
    assert max(xs) - min(xs) == pytest.approx(bw, abs=1e-4)
    assert max(ys) - min(ys) == pytest.approx(bh, abs=1e-4)
    # box width preserves the hand path's aspect ratio
    assert SCRIBBLE_W_MM == pytest.approx(SCRIBBLE_H_MM * bw / bh)


def test_legend_scribbles_are_one_template_in_entry_colors():
    spec = _spec()
    svg = render_token_diagram(spec)
    paths = re.findall(r'<path class="legend-scribble" d="([^"]+)"[^/]*fill="(#\w+)"', svg)
    assert len(paths) == len(spec.sources)
    assert len({d for d, _ in paths}) == 1  # same verbatim path every time
    assert [c for _, c in paths] == [s.color for s in spec.sources.values()]
    # filled, never stroked
    assert all('stroke="none"' in line for line in svg.splitlines() if "legend-scribble" in line)
    # each instance is scaled into the swatch box, not drawn at native size
    assert all("scale(" in line for line in svg.splitlines() if "legend-scribble" in line)


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
