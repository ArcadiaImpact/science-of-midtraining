"""CPU-only tests for ``scimt.viz.token_diagram`` (spec validation, geometry, SVG)."""

from __future__ import annotations

import re

import pytest
import yaml

from scimt.viz.token_diagram import (
    Annotation,
    Arm,
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

M = 1_000_000


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
                checkpoints_after=(1,),
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
                checkpoints_after=(1,),
            ),
        ),
        pretraining=Pretraining(label="Pretraining", color="#cc78bc"),
        annotations=(Annotation(text="Base", x_mm=20.0, y_mm=-4.0),),
    )
    base.update(kw)
    return TokenDiagramSpec(**base)


# ------------------------------------------------------------------ loading
SMALL_YAML = {
    "title": "Tiny",
    "unit_tokens": 10_000_000,
    "unit_mm": 5,
    "unit_label": "10 Million Tokens",
    "sources": {
        "mid": {"label": "Mid", "color": "#de8f05"},
        "docs": {"label": "Docs", "color": "#029e73"},
    },
    "pretraining": {"label": "Pretraining", "color": "#cc78bc"},
    "annotations": [{"text": "Base", "x_mm": 10, "y_mm": -5}],
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
            "checkpoints_after": [0, 1],
        },
    ],
}


def _write_yaml(tmp_path, data, name="diagram.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def test_yaml_round_trip(tmp_path):
    spec = load_token_diagram_spec(_write_yaml(tmp_path, SMALL_YAML))
    assert spec.title == "Tiny"
    assert set(spec.sources) == {"mid", "docs"}
    assert isinstance(spec.sources["mid"], SourceStyle)
    assert isinstance(spec.arms[1].stages[0].components[1], StageComponent)
    assert spec.arms[1].stages[0].components[1].epochs == 2
    assert spec.arms[1].checkpoints_after == (0, 1)
    assert spec.pretraining is not None and spec.pretraining.color == "#cc78bc"
    assert spec.annotations[0].text == "Base"
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
    data = yaml.safe_load(yaml.safe_dump(SMALL_YAML))
    data["arms"][0]["stages"][0]["components"][0]["epoch"] = 3  # typo for epochs
    with pytest.raises(ValueError, match="unknown key"):
        load_token_diagram_spec(_write_yaml(tmp_path, data))


def test_unknown_source_key(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SMALL_YAML))
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
        Arm(name="A", stages=(Stage(components=(StageComponent("mid", M),)),), checkpoints_after=(1,))


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


# ----------------------------------------------------------------- geometry
def test_stage_widths_from_token_budgets():
    spec = _spec()
    lay = compute_layout(spec)
    r0, r1 = lay.rows
    # 5mm per 10M tokens
    assert r0.stages[0].w_mm == pytest.approx(40.0)
    assert r0.stages[1].w_mm == pytest.approx(50.0)
    # stages butt together, starting at x=0 (== x_origin in page coords)
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


def test_boundaries_and_checkpoints():
    spec = _spec()
    lay = compute_layout(spec)
    row = lay.rows[0]
    # one dashed boundary per *internal* boundary (2 stages -> 1)
    assert row.boundary_x_mm == pytest.approx((lay.x_origin_mm + 40.0,))
    # checkpoint after stage 1 -> right edge of the row
    assert row.checkpoint_x_mm == pytest.approx((lay.x_origin_mm + 90.0,))


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
    # lightest first, darkest last (compare luminance proxies)
    def lum(c: str) -> int:
        return sum(int(c[i : i + 2], 16) for i in (1, 3, 5))

    assert lum(derived[0]) > lum(derived[-1])
    assert all(lum(a) > lum(b) for a, b in zip(derived, derived[1:]))
    # explicit ramp wins when it covers the epoch count; single epoch stays flat
    ramp = ("#56c69b", "#029e73", "#017453", "#014b35")
    assert epoch_shades("#029e73", 3, ramp) == ramp[:3]
    assert epoch_shades("#029e73", 1, ramp) == ("#029e73",)
    # too-short ramp falls back to derived shades
    assert epoch_shades("#029e73", 4, ("#56c69b",)) == derived
    with pytest.raises(ValueError):
        epoch_shades("#029e73", 0)


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
    assert f'width="{lay.width_mm:g}mm"' in svg or "mm\"" in svg
    assert 'viewBox="0 0' in svg
    # one rect per component strip
    expected_strips = sum(
        c.epochs for arm in spec.arms for st in arm.stages for c in st.components
    )
    assert svg.count('class="component-strip"') == expected_strips
    # dashed lines: internal boundaries in the bars + 1 legend key
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


def test_legend_entries():
    spec = _spec()
    svg = render_token_diagram(spec)
    assert "= 10 Million Tokens" in svg
    assert "= Multiple Epochs" in svg
    assert "= Training Stage Boundary" in svg
    assert "= Evaluated/Forked Checkpoint" in svg
    # one square swatch per source, plus pretraining
    assert svg.count('class="legend-swatch"') == len(spec.sources) + 1
    for style in spec.sources.values():
        assert f"= {style.label}" in svg
    assert "= Pretraining" in svg
    assert svg.count('class="legend-epoch-strip"') == 4
    # no pretraining -> no pretraining swatch or gradient
    plain = render_token_diagram(_spec(pretraining=None))
    assert plain.count('class="legend-swatch"') == len(spec.sources)
    assert "pretrainFade" not in plain
    assert "pretrainFade" in svg


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
