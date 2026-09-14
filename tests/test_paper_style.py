"""CPU-only tests for ``scimt.viz.paper``, the write-up's figure style.

The module exists so the paper figures agree on geometry (5.5 in page, never
tight-cropped), type (nothing under 8 pt) and palette (the pair main.tex
defines); these pin each of those and the loud ``check``/``save`` guards.
"""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import numpy as np  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402


def _figure(height_in=2.0):
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(height_in)
        ax.bar([0, 1], [1, 2], color=[ps.CHARTER, ps.COIN])
        ax.set_xlabel("x label")
        ax.set_ylabel("y label")
        ax.set_title("title")
    return fig, ax


def test_palette_is_the_manuscripts_pair():
    # main.tex: \definecolor{charter}{HTML}{0173B2}, \definecolor{coin}{HTML}{DE8F05}
    assert ps.CHARTER.lower() == "#0173b2"
    assert ps.COIN.lower() == "#de8f05"
    assert ps.BLUE == ps.CHARTER and ps.ORANGE == ps.COIN
    assert ps.DARK_GREY < ps.GREY < ps.LIGHT_GREY        # a darker grey for words than for bars


def test_lighten_blends_toward_white():
    assert ps.lighten("#000000", 1.0).lower() == "#ffffff"
    assert ps.lighten(ps.CHARTER, 0.0).lower() == ps.CHARTER.lower()
    assert ps.CHARTER_LIGHT == ps.lighten(ps.CHARTER, ps.LIGHT_MIX)


def test_darken_blends_toward_black():
    assert ps.darken(ps.CHARTER, 0.0) == ps.CHARTER
    assert ps.darken(ps.CHARTER, 1.0) == "#000000"
    r, g, b = matplotlib.colors.to_rgb(ps.darken(ps.CHARTER, 0.5))
    r0, g0, b0 = matplotlib.colors.to_rgb(ps.CHARTER)
    assert (r, g, b) == pytest.approx((r0 / 2, g0 / 2, b0 / 2), abs=1 / 255)


def test_rc_has_no_font_under_minimum_and_embeds_truetype():
    rc = ps.rc()
    for key in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
                "ytick.labelsize", "legend.fontsize", "legend.title_fontsize",
                "figure.titlesize", "figure.labelsize"):
        assert rc[key] >= ps.MIN_FONT_PT, key
    assert rc["pdf.fonttype"] == 42
    assert rc["font.sans-serif"][0] == "DejaVu Sans"
    assert rc["savefig.bbox"] is None
    assert ps.rc(**{"axes.spines.top": True})["axes.spines.top"] is True


def test_figure_is_authored_at_text_width():
    fig, _ax = _figure()
    assert fig.get_size_inches()[0] == pytest.approx(ps.TEXTWIDTH_IN)
    assert fig.get_layout_engine() is not None
    ps.check(fig)  # a plain figure under rc() passes
    matplotlib.pyplot.close(fig)


def test_check_rejects_small_text():
    fig, ax = _figure()
    ax.text(0.5, 1.5, "tiny", fontsize=6)
    with pytest.raises(ValueError, match=r"below 8 pt: 6 pt 'tiny'"):
        ps.check(fig)
    matplotlib.pyplot.close(fig)


def test_check_rejects_text_off_the_canvas():
    fig, ax = _figure()
    ax.text(1.6, 0.5, "hanging legend", transform=ax.transAxes, clip_on=False,
            in_layout=False)
    with pytest.raises(ValueError, match="off the canvas"):
        ps.check(fig)
    matplotlib.pyplot.close(fig)


def test_check_rejects_a_canvas_that_is_not_text_width():
    fig, _ax = _figure()
    fig.set_size_inches(6.0, 2.0)
    with pytest.raises(ValueError, match="must be 5.500 in"):
        ps.check(fig)
    matplotlib.pyplot.close(fig)
    fig, _ax = _figure()
    fig.set_size_inches(ps.TEXTWIDTH_IN * 0.5, 2.0)
    ps.check(fig, width_frac=0.5)  # half-column figures declare it
    matplotlib.pyplot.close(fig)


def test_save_pins_the_page_and_embeds_truetype(tmp_path):
    fig, _ax = _figure(height_in=2.0)
    written = ps.save(fig, tmp_path, "t")
    assert written == [tmp_path / "t.pdf"]                    # PDF only, by default
    assert not (tmp_path / "t.png").exists()
    w_pt, h_pt = ps.page_size_pt(tmp_path / "t.pdf")
    assert w_pt == pytest.approx(396.0, abs=0.01)   # 5.5 in, not the ink's width
    assert h_pt == pytest.approx(144.0, abs=0.01)
    pdf = (tmp_path / "t.pdf").read_bytes()
    assert b"/FontFile2" in pdf and b"/CharProcs" not in pdf   # Type 42, not Type 3
    assert b"CreationDate" not in pdf                           # reproducible bytes


def test_png_preview_only_when_asked(tmp_path):
    fig, _ax = _figure(height_in=2.0)
    written = ps.save(fig, tmp_path, "t", formats=("pdf", "png"))
    assert written == [tmp_path / "t.pdf", tmp_path / "t.png"]
    from PIL import Image
    assert Image.open(tmp_path / "t.png").size == (1650, 600)   # 5.5 x 2.0 in at 300 dpi


def test_save_refuses_a_figure_that_fails_check(tmp_path):
    fig, ax = _figure()
    ax.text(0.5, 1.5, "tiny", fontsize=5)
    with pytest.raises(ValueError):
        ps.save(fig, tmp_path, "t")
    assert not (tmp_path / "t.pdf").exists()
    matplotlib.pyplot.close(fig)


def test_reserve_band_is_cumulative():
    fig, _ax = _figure(height_in=2.0)
    ps.reserve_band(fig, bottom_in=0.5)
    _l, bottom, _w, height = fig.get_layout_engine().get()["rect"]
    assert (bottom, height) == pytest.approx((0.25, 0.75))
    ps.reserve_band(fig, top_in=0.2)
    _l, bottom, _w, height = fig.get_layout_engine().get()["rect"]
    assert (bottom, height) == pytest.approx((0.25, 0.65))
    with pytest.raises(ValueError, match="no room"):
        ps.reserve_band(fig, bottom_in=5.0)
    matplotlib.pyplot.close(fig)


def test_no_caption_text_helper_exists():
    # Jonathan, 2026-09-12: never print caveats / provenance notes on a figure.
    assert not hasattr(ps, "caveat")


def _pieces(fig, anchor):
    return [t for t in fig.texts if t.get_gid() == "ps:rich"
            and abs(t.get_rotation() - anchor.get_rotation()) < 1e-6]


def test_paint_colours_keywords_and_keeps_alignment():
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(2.4)
        ax.bar([0, 1, 2], [1, 2, 3], color=ps.GREY)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Ambiguous", "+2% Coin EFT", "Charter\nmidtrain"])
        ax.set_title("Charter vs coin, by arm")
        ax.set_ylabel("chose Charter crew (%)")
        ax.text(1.0, 2.5, "the Ch arm", ha="right")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = ax.get_xticklabels()
    before = {t.get_text(): t.get_window_extent(renderer) for t in labels}
    title_before = ax.title.get_window_extent(renderer)
    pieces = ps.paint(fig)
    # anchors are transparent and marked; pieces exist for each keyword
    assert all(t.get_alpha() == 0 and t.get_gid() == "ps:rich-anchor" for t in labels)
    coloured = {(t.get_text(), t.get_color().lower(), t.get_fontweight()) for t in pieces
                if t.get_color().lower() in (ps.CHARTER, ps.COIN, ps.GREEN)}
    assert ("Ambiguous", ps.GREEN, "bold") in coloured
    assert ("Coin", ps.COIN, "bold") in coloured
    assert ("Charter", ps.CHARTER, "bold") in coloured
    assert ("Ch", ps.CHARTER, "bold") in coloured
    assert ("coin", ps.COIN, "bold") in coloured          # case kept, still painted
    # a centred single-line anchor: the run is centred on the anchor (< 1 pt off)
    run = [t for t in pieces if t.get_text() in ("+2% ", "Coin", " EFT")]
    ext = [t.get_window_extent(renderer) for t in run]
    centre = (min(e.x0 for e in ext) + max(e.x1 for e in ext)) / 2
    anchor_centre = (before["+2% Coin EFT"].x0 + before["+2% Coin EFT"].x1) / 2
    assert abs(centre - anchor_centre) < fig.dpi / 72
    # a two-line anchor: the second line's piece sits below the first
    first = next(t for t in pieces if t.get_text() == "Charter")
    second = next(t for t in pieces if t.get_text() == "midtrain")
    assert second.get_window_extent(renderer).y1 < first.get_window_extent(renderer).y0 + 1
    # title still spans the same box (painted pieces overlay it)
    title_pieces = [t for t in pieces
                    if title_before.y0 - 1 <= t.get_window_extent(renderer).y0 <= title_before.y1]
    assert [t.get_text() for t in title_pieces] == ["Charter", " vs ", "coin", ", by arm"]
    run_x0 = title_pieces[0].get_window_extent(renderer).x0
    run_x1 = title_pieces[-1].get_window_extent(renderer).x1
    # anchors are measured hinted here, pieces unhinted (as the PDF backend
    # measures): allow a sub-pixel disagreement, still far below a real misalignment
    assert abs((run_x0 + run_x1) / 2 - (title_before.x0 + title_before.x1) / 2) < 2 * fig.dpi / 72
    ps.check(fig)  # pieces are 8-9 pt and on the canvas
    matplotlib.pyplot.close(fig)


def test_paint_rotated_right_anchored_and_skips_non_ink():
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(2.6)
        ax.bar([0, 1], [1, 2], color=ps.GREY)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["+2% Coin", "Charter"], rotation=45, ha="right",
                           rotation_mode="anchor")
        white = ax.text(0, 0.5, "Charter crew", color="white", ha="center")
        side = ax.text(1, 0.5, "Coin", color=ps.COIN, ha="center")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label = ax.get_xticklabels()[0]
    right_before = label.get_window_extent(renderer).x1
    pieces = ps.paint(fig)
    run = [t for t in pieces if abs(t.get_rotation() - 45) < 1e-6 and t.get_text() in ("+2% ", "Coin")]
    assert len(run) == 2
    assert abs(max(t.get_window_extent(renderer).x1 for t in run) - right_before) < fig.dpi / 72
    assert white.get_alpha() is None and side.get_alpha() is None   # untouched
    matplotlib.pyplot.close(fig)


def test_paint_abbreviations_and_extra_keywords():
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(2.0)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Ambi", "+2% Co", "Control\nmidtrain"])
        ax.set_title("Ambig. EFT")
    fig.canvas.draw()
    pieces = ps.paint(fig, extra={"Control": ps.GREY})
    got = {(t.get_text(), t.get_color().lower(), t.get_fontweight()) for t in pieces}
    assert ("Ambi", ps.GREEN, "bold") in got
    assert ("Ambig.", ps.GREEN, "bold") in got             # the full stop rides with the word
    assert ("Co", ps.COIN, "bold") in got
    assert ("Control", ps.GREY, "bold") in got          # per-call word, grey bold
    assert ("midtrain", ps.INK, "normal") in got        # the rest stays plain ink
    assert "control" not in ps.KEYWORDS                 # not made global
    matplotlib.pyplot.close(fig)


def test_paint_include_exclude_and_idempotent():
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(2.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Coin", "Charter"])
        ax.set_title("Coin title")
    fig.canvas.draw()
    labels = ax.get_xticklabels()

    def n_pieces():
        return sum(1 for t in fig.findobj(matplotlib.text.Text) if t.get_gid() == "ps:rich")

    ps.paint(fig, include=labels)
    assert ax.title.get_alpha() is None                  # not included: untouched
    n = n_pieces()
    ps.paint(fig, exclude=[ax.title])                    # nothing new: anchors done, title excluded
    assert n_pieces() == n
    ps.paint(fig)
    assert ax.title.get_alpha() == 0 and n_pieces() == n + 2
    matplotlib.pyplot.close(fig)


def test_painted_anchor_is_measured_but_never_drawn_and_pieces_follow_it():
    from unittest.mock import Mock
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(2.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Charter", "Coin"])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    anchor = ax.get_xticklabels()[1]
    ps.paint(fig)
    piece = next(t for t in fig.findobj(matplotlib.text.Text)
                 if t.get_gid() == "ps:rich" and t.get_text() == "Coin")
    # never drawn ...
    mock = Mock()
    anchor.draw(mock)
    assert mock.method_calls == []
    # ... but still measured: the layout keeps its band for the label
    assert anchor.get_window_extent(renderer).height > 0
    # the piece tracks the anchor when the layout moves it (different page height)
    fig.set_size_inches(ps.TEXTWIDTH_IN, 3.0)
    fig.canvas.draw()
    a, p = anchor.get_window_extent(renderer), piece.get_window_extent(renderer)
    assert abs((a.x0 + a.x1) / 2 - (p.x0 + p.x1) / 2) < 2 * fig.dpi / 72
    assert abs(a.y0 - p.y0) < 2 * fig.dpi / 72
    matplotlib.pyplot.close(fig)


def test_pieces_follow_axis_labels_and_annotations_across_dpi():
    """Axis labels bake one coordinate in display pixels per draw and annotations
    rebuild their transform per draw: pieces must still sit on them at any dpi
    (the dose grid's colour-bar label fell off the PDF page before this)."""
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(2.4)
        ax.bar([0, 1], [1, 2], color=ps.GREY)
        ax.set_ylabel("chose Charter crew")
        ann = ax.annotate("Coin midtrain", xy=(0.5, 0), xycoords="axes fraction",
                          xytext=(0, -30), textcoords="offset points",
                          ha="center", va="top", annotation_clip=False)
    fig.canvas.draw()
    ps.paint(fig)
    pieces = {t.get_text(): t for t in fig.findobj(matplotlib.text.Text) if t.get_gid() == "ps:rich"}
    for dpi in (72, 100, 150):
        fig.set_dpi(dpi)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        tol = 2 * dpi / 72
        for anchor, word in ((ax.yaxis.label, "Charter"), (ann, "Coin")):
            a, p = anchor.get_window_extent(r), pieces[word].get_window_extent(r)
            assert a.x0 - tol <= p.x0 and p.x1 <= a.x1 + tol, (dpi, word, a, p)
            assert a.y0 - tol <= p.y0 and p.y1 <= a.y1 + tol, (dpi, word, a, p)
    matplotlib.pyplot.close(fig)


def test_save_forwards_extra_keywords_to_paint(tmp_path):
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(2.0)
        ax.set_xlabel("Midtraining Tokens (\u2212Coin, 0 = Control, +Charter)")
    ps.save(fig, tmp_path, "t", extra={"control": ps.DARK_GREY})
    pieces = [t for t in fig.findobj(matplotlib.text.Text) if t.get_gid() == "ps:rich"]
    got = {(t.get_text(), t.get_color().lower(), t.get_fontweight()) for t in pieces}
    assert ("Control", ps.DARK_GREY, "bold") in got     # the per-save word, grey bold
    assert ("\u2212Coin", ps.COIN, "bold") in got
    assert ("+Charter", ps.CHARTER, "bold") in got
    assert "control" not in ps.KEYWORDS                 # not made global
    matplotlib.pyplot.close(fig)


def test_barberpole_is_filled_paths_in_both_colours_not_a_pdf_pattern(tmp_path):
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(1.0)
        bar = ax.barh([0], [1], 1.0, color=ps.CHARTER, linewidth=0)[0]
        ps.barberpole(bar, ps.GREEN_LIGHT, pitch_pt=24.0)   # wide stripes: few seam pixels
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
    fig.set_dpi(72.0)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
    x0, y0, x1, y1 = (int(v) for v in bar.get_window_extent().extents)
    h = rgb.shape[0]
    inside = rgb[h - y1 + 2:h - y0 - 2, x0 + 2:x1 - 2].reshape(-1, 3)   # rows run top-down
    def share(colour):
        target = np.array(matplotlib.colors.to_rgb(colour)) * 255
        return (np.abs(inside - target).max(axis=1) < 8).mean()
    assert 0.35 < share(ps.CHARTER) < 0.65                     # equal stripes, give or take
    assert 0.35 < share(ps.GREEN_LIGHT) < 0.65                 # the anti-aliased seams are the rest
    ps.save(fig, tmp_path, "t")
    assert b"/PatternType" not in (tmp_path / "t.pdf").read_bytes()   # no hatch pattern object
    matplotlib.pyplot.close(fig)


def test_save_paints_and_checks_at_pdf_dpi_then_restores(tmp_path):
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(2.0)
        ax.set_ylabel("Charter-crew rate")
    ps.save(fig, tmp_path, "t")
    assert fig.dpi == 100.0                                   # build dpi restored
    assert ps.page_size_pt(tmp_path / "t.pdf")[0] == pytest.approx(396.0, abs=0.01)
    matplotlib.pyplot.close(fig)
