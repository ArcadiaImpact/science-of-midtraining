"""Build the canonical cross-setting data-quality panel from committed metrics.

This is the extraction half of the cross-setting figure pair. It walks the three
legs' committed ``metrics.json`` files through a metric registry and writes one
long-format table (``panel.json`` + ``panel.csv``). ``plot_panel.py`` reads that
table and nothing else, so a new corpus never touches plotting code.

The synthesis this feeds is ``docs/wiki/syntheses/data-quality-across-settings.md``
§5. The per-leg figures in ``<leg>/metrics/plot_metrics.py`` stay as they are;
this script exists for the *join*, which no single leg owns.

    uv run --extra dev python experiments/data_quality_crossplots/panel.py

To add a corpus from a future midtraining run: append an ``Entity`` and a
``Source`` below. Anything the registry cannot reach gets ``source="prose:..."``
so a hardcoded number is visibly hardcoded rather than silently so.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

#: The shared cross-setting scorer. Every ppl row in the panel is under this key
#: and only this key -- MSM's llama-3-1-8b numbers are its own substrate's
#: initial training loss and never enter a cross-setting row (Appendix A).
SCORER = "gemma-3-12b-pt"


# --------------------------------------------------------------------------
# metric registry: display metadata lives here, never in the data rows
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    #: lower_better | higher_better | anchor_referenced | match_arms
    direction: str
    scale: str = "linear"
    #: floor to read excess against, when the metric has a natural-text floor
    floor_entity: str | None = None
    note: str = ""


METRICS: dict[str, Metric] = {
    "ppl": Metric("ppl", f"perplexity ({SCORER})", "anchor_referenced",
                  scale="log", floor_entity="dolmino",
                  note="1,024-token truncation, per-token loss clamped at 20.0"),
    "compress": Metric("compress", "compression ratio (zlib)", "anchor_referenced",
                       note="lower = more internally repetitive"),
    # Two measurements of one quantity. The zlib row is what every leg
    # committed, and its window binds on all seven corpora -- which biases it
    # by document length, so it is NOT the one to plot cross-corpus. The lzma
    # row is the corrected measurement from recompute.py. Levels are not
    # comparable between them; only orderings within one.
    "cross_doc_redundancy_lzma": Metric(
        "cross_doc_redundancy_lzma",
        "cross-doc redundancy (lzma, window not binding)",
        "lower_better", floor_entity="fineweb",
        note="k=32, 200 draws x 5 seeds; window does not bind; PRIMARY "
             "for cross-corpus comparison"),
    "cross_doc_gain": Metric(
        "cross_doc_gain",
        "cross-doc redundancy (zlib — SUPERSEDED cross-corpus)",
        "lower_better", floor_entity="fineweb",
        note="the committed value; window binds on all 7 corpora, so this "
             "row is length-confounded across settings -- kept for provenance"),
    "embed_dispersion": Metric("embed_dispersion", "embedding dispersion",
                               "higher_better",
                               note="1 - mean pairwise cosine, MiniLM-L6-v2, n=512"),
    # Two settings are committed for every corpus in all three legs. 100/100
    # is primary: BLEU clips candidate n-grams at their max count across
    # references, so the level rises monotonically with the reference cap and
    # a self-BLEU number means nothing without it. `sample` only sets noise.
    "self_bleu_100_100": Metric("self_bleu_100_100",
                                "self-BLEU (100 cand x 100 refs)",
                                "lower_better",
                                note="primary setting; refs cap sets the level"),
    "self_bleu": Metric("self_bleu", "self-BLEU (40 x 60, library default)",
                        "lower_better",
                        note="secondary; kept because published numbers cite it"),
}


# --------------------------------------------------------------------------
# entities: what gets drawn, and as what
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    eid: str
    label: str
    #: colour family -- one hue per *setting*, not per arm
    family: str
    #: corpus | anchor | known_bad -- drives the mark, and keeps anchors from
    #: being drawn as peer bars
    role: str
    #: the other arm of the same pair, for the paired-arm connectors
    pair: str | None = None
    n_docs: int | None = None


ENTITIES: tuple[Entity, ...] = (
    Entity("dispatch_coin", "Dispatch coin", "dispatch", "corpus", "dispatch_charter"),
    Entity("dispatch_charter", "Dispatch charter", "dispatch", "corpus", "dispatch_coin"),
    Entity("python4", "Python 4", "python4", "corpus"),
    Entity("msm_america", "MSM america", "msm", "corpus", "msm_afford"),
    Entity("msm_afford", "MSM afford", "msm", "corpus", "msm_america"),
    Entity("v3c_coin", "v3-C coin", "known_bad", "known_bad", "v3c_charter"),
    Entity("v3c_charter", "v3-C charter", "known_bad", "known_bad", "v3c_coin"),
    Entity("dolmino", "Dolmino", "dolmino", "anchor"),
    Entity("fineweb", "FineWeb", "fineweb", "anchor"),
)


# --------------------------------------------------------------------------
# sources: where each entity's numbers live. The three legs disagree on shape,
# so the shape is data rather than an if-tree at the read site.
# --------------------------------------------------------------------------

DISPATCH = "experiments/prior_coins/dispatch_docgen_v3_extension/metrics/reports"
PYTHON4 = "experiments/python4_docgen/metrics/reports"
MSM = "experiments/msm_corpus_quality/metrics/reports"


@dataclass(frozen=True)
class Source:
    eid: str
    path: str
    #: dotted path to the block holding the metric keys
    at: str
    #: dotted path to the ppl block, when it does not sit under ``at``
    ppl_at: str | None = None


SOURCES: tuple[Source, ...] = (
    Source("dispatch_coin", f"{DISPATCH}/v1/metrics.json", "arms.coin"),
    Source("dispatch_charter", f"{DISPATCH}/v1/metrics.json", "arms.charter"),
    Source("v3c_coin", f"{DISPATCH}/v3c/metrics.json", "arms.coin"),
    Source("v3c_charter", f"{DISPATCH}/v3c/metrics.json", "arms.charter"),
    Source("python4", f"{PYTHON4}/p4_merged/metrics.json", "whole"),
    Source("msm_america", f"{MSM}/msm_cheese/metrics.json", "arms.america"),
    Source("msm_afford", f"{MSM}/msm_cheese/metrics.json", "arms.afford"),
    # Anchors: diversity stats and ppl live in different blocks of the same
    # file. The msm leg carries a byte-identical `anchors` block (verified at
    # build time below), so either leg would do; the python4 leg is the one
    # that also carries `anchor_ppl`, so both come from one file.
    Source("dolmino", f"{PYTHON4}/p4_merged/metrics.json", "anchors.dolmino",
           ppl_at="anchor_ppl.dolmino"),
    Source("fineweb", f"{PYTHON4}/p4_merged/metrics.json", "anchors.fineweb",
           ppl_at="anchor_ppl.fineweb"),
)


def _dig(obj, dotted: str):
    for part in dotted.split("."):
        if obj is None:
            return None
        obj = obj.get(part)
    return obj


# --------------------------------------------------------------------------
# the row
# --------------------------------------------------------------------------

@dataclass
class Row:
    entity: str
    label: str
    family: str
    role: str
    pair: str | None
    metric: str
    value: float | None
    lo: float | None
    hi: float | None
    n: int | None
    spread: str | None
    scorer: str | None
    source: str


def _extract(src: Source, ent: Entity) -> list[Row]:
    blob = json.loads((REPO / src.path).read_text())
    block = _dig(blob, src.at)
    if block is None:
        raise KeyError(f"{src.path}: no block at {src.at}")
    rows: list[Row] = []

    def add(metric, value, lo=None, hi=None, n=None, spread=None,
            scorer=None, at=None):
        rows.append(Row(ent.eid, ent.label, ent.family, ent.role, ent.pair,
                        metric, value, lo, hi, n, spread, scorer,
                        f"{src.path}#{at}"))

    # perplexity -- p10/p50/p90 under the one shared scorer
    ppl = _dig(blob, src.ppl_at) if src.ppl_at else block.get("ppl")
    at_ppl = (src.ppl_at or f"{src.at}.ppl")
    if ppl and SCORER in ppl:
        p = ppl[SCORER]
        add("ppl", p["p50"], p.get("p10"), p.get("p90"), p.get("n"),
            "p10_p90", SCORER, f"{at_ppl}.{SCORER}")

    # compression -- p10/p50/p90 for corpora, p50 only for anchors
    comp = block.get("compress")
    if isinstance(comp, dict):
        add("compress", comp["p50"], comp.get("p10"), comp.get("p90"),
            block.get("n_docs") or block.get("n"), "p10_p90",
            at=f"{src.at}.compress")
    elif block.get("compress_p50") is not None:
        add("compress", block["compress_p50"], n=block.get("n"),
            at=f"{src.at}.compress_p50")

    # cross-doc templating gain -- mean over 200 draws, with the draw spread
    cd = block.get("cross_doc")
    if isinstance(cd, dict):
        add("cross_doc_gain", cd["gain_mean"], cd.get("gain_p10"),
            cd.get("gain_p90"), cd.get("draws"), "draw_p10_p90",
            at=f"{src.at}.cross_doc.gain_mean")
    elif block.get("cross_doc_gain") is not None:
        add("cross_doc_gain", block["cross_doc_gain"], n=block.get("n"),
            at=f"{src.at}.cross_doc_gain")

    # point-estimate diversity metrics
    params = block.get("self_bleu_params") or {}
    for key, nkey in (("embed_dispersion", "embed_sample_n"),
                      ("self_bleu_100_100", None),
                      ("self_bleu", None)):
        if block.get(key) is None:
            continue
        if key.startswith("self_bleu"):
            # the n is the recorded `sample`, NOT pairwise_sample_n, which
            # labels the pool drawn from -- see the synthesis §5
            n = (params.get(key) or {}).get("sample")
        else:
            n = block.get(nkey)
        add(key, block[key], n=n, at=f"{src.at}.{key}")

    return rows


def _check_anchor_agreement() -> list[str]:
    """The anchors appear in two legs. Confirm they agree before trusting one."""
    notes = []
    a = json.loads((REPO / PYTHON4 / "p4_merged/metrics.json").read_text())["anchors"]
    b = json.loads((REPO / MSM / "msm_cheese/metrics.json").read_text())["anchors"]
    for name in ("dolmino", "fineweb"):
        for key in ("compress_p50", "cross_doc_gain", "self_bleu",
                    "embed_dispersion"):
            x, y = a[name].get(key), b[name].get(key)
            if x != y:
                notes.append(f"anchor mismatch {name}.{key}: p4={x} msm={y}")
    # Dispatch scored the anchors in an independent pass; record the drift
    # rather than silently preferring one.
    ours = json.loads(
        (REPO / PYTHON4 / "p4_merged/metrics.json").read_text())["anchor_ppl"]
    d = json.loads((REPO / DISPATCH / "v1/metrics.json").read_text())["anchors"]
    for name in ("dolmino", "fineweb"):
        mine = ours[name][SCORER]["p50"]
        theirs = d[name][SCORER]["p50"]
        if abs(mine - theirs) > 1e-9:
            notes.append(
                f"anchor ppl {name} p50 differs across legs: python4={mine:.6f} "
                f"dispatch={theirs:.6f} (rel {abs(mine-theirs)/theirs:.2e}) -- "
                f"independent scoring passes; panel uses the python4 value")
    return notes


#: written by recompute.py; absent until it has been run at least once
CROSSMETRICS = "experiments/data_quality_crossplots/crossmetrics.json"


def _extract_crossmetrics(by_eid: dict[str, Entity]) -> tuple[list[Row], dict]:
    """The corrected redundancy rows and the length-control block.

    These do not come from a leg -- `recompute.py` measured them. Without this
    the panel carries only the superseded zlib row, which is how fig 3 came to
    plot a withdrawn number.
    """
    path = REPO / CROSSMETRICS
    if not path.exists():
        return [], {}
    blob = json.loads(path.read_text())
    rows: list[Row] = []
    for eid, rec in blob["cross_doc_redundancy"]["lzma"].items():
        ent = by_eid[eid]
        rows.append(Row(
            ent.eid, ent.label, ent.family, ent.role, ent.pair,
            "cross_doc_redundancy_lzma", rec["mean"], rec["min"], rec["max"],
            len(rec["seeds"]), "seed_min_max", None,
            f"{CROSSMETRICS}#cross_doc_redundancy.lzma.{eid}"))
    return rows, blob["compress_length_control"]


def build() -> dict:
    by_eid = {e.eid: e for e in ENTITIES}
    rows: list[Row] = []
    for src in SOURCES:
        rows.extend(_extract(src, by_eid[src.eid]))
    cross_rows, length_control = _extract_crossmetrics(by_eid)
    rows.extend(cross_rows)
    return {
        "length_control": length_control,
        "generated_by": "experiments/data_quality_crossplots/panel.py",
        "scorer": SCORER,
        "metrics": {k: asdict(v) for k, v in METRICS.items()},
        "entities": [asdict(e) for e in ENTITIES],
        "notes": _check_anchor_agreement(),
        "rows": [asdict(r) for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()

    panel = build()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "panel.json").write_text(json.dumps(panel, indent=1) + "\n")

    # the table view the palette's contrast WARN obliges us to ship
    fields = list(panel["rows"][0].keys())
    with (args.out / "panel.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(panel["rows"])

    print(f"{len(panel['rows'])} rows -> {args.out / 'panel.json'}")
    for note in panel["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
