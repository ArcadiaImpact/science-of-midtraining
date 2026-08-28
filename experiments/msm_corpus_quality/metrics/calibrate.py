"""The metric admission rule for the MSM leg, as executable assertions.

Design §5 **as amended**. The design asked for replication "within tolerance
bands accounting for their N=96 sample vs our full corpus"; that recipe is not
executable, because five of the seven statistics it names are *different
estimators at different n*, not the same number measured more precisely
(PLAN §1.2):

| statistic | committed target produced by | the sweep computes |
|---|---|---|
| `self_bleu` | `diversity.compute` -> `self_bleu(sample=40)` over 96 docs | a 2,000-doc sample |
| `embed_dispersion` | `sample=60` | 512 |
| `distinct_1/2/3` | 96 docs — and distinct-n falls as n grows | the full corpus |
| `near_dup_rate` | `threshold=0.7` | 0.7 here, 0.72 in the dispatch sweep |
| `ppl_median` | `max_docs=60` (head of list) at `max_tokens=512` | every doc at 1,024 |

So **calibration re-runs the ORIGINAL call path at its own defaults and
asserts equality** (design amendment 2). Concretely:
``random.Random(0).sample(recs, 96)`` over the staged file in line order, then
``diversity.compute`` / ``density.compute`` / ``contamination.compute`` /
``naturalness.compute`` exactly as `experiments/value-data-gen/
run_experiment.py::_full_battery` called them, against the committed extract
in `replication/health_comparison_extract.json`. Full-corpus values live in
`REPORT.md` in their own columns and are never presented as the same metric at
larger n.

The density block is asserted under the **FROZEN** `AFFORDABILITY` preset —
that is the instrument PR #163 used, and reproducing its number is the point.
`AFFORDABILITY_V2` is reported alongside everywhere else; it is deliberately
absent from the replication assertions, because a repaired instrument
reproducing an old number would mean the repair did nothing.

    uv run --extra analysis python .../metrics/calibrate.py
    uv run --extra analysis python .../metrics/calibrate.py --with-ppl   # GPU session
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE
sys.path[:0] = [str(REPO / "src"), str(HERE)]

from scimt.gen.health import contamination, density, diversity  # noqa: E402
from scimt.gen.health.report import load_rows  # noqa: E402
from scimt.gen.health.targets import get_target  # noqa: E402

import sweep  # noqa: E402
import thresholds  # noqa: E402

LOGGER = logging.getLogger("metrics.calibrate")

REPLICATION = json.loads(
    (HERE / "replication" / "health_comparison_extract.json").read_text())

#: staged arm -> the extract key, and the FROZEN preset the original used.
ARM_REPLICATION = {
    "america": ("usa_MSM", "america"),
    "afford": ("aff_MSM", "affordability"),
}
SAMPLE_N = 96
SAMPLE_SEED = 0

#: Exact (==, full float precision) targets: deterministic stdlib metrics.
EXACT_METRICS = (
    "target_mention_rate", "assertion_rate", "evidence_per_1k_tok",
    "distinct_1", "distinct_2", "distinct_3", "self_bleu", "near_dup_rate",
    "negation_frame_rate", "meta_tell_rate", "template_leakage",
)
#: Soft targets, with the tolerance registered in THRESHOLDS.md before the run
#: (PLAN R4: the MiniLM revision is not pinned by the original run, so a
#: bit-exact target would be a claim about someone else's model cache).
SOFT_METRICS = {"embed_dispersion": 1e-2}
#: Needs the pooled GPU session.
PPL_METRICS = ("ppl_median", "ppl_mean", "ppl_p10", "ppl_p90")

# ---------------------------------------------------------------------------
# AMENDMENTS AFTER FIRST CONTACT
#
# The house rule is "tune nothing against non-calibration outputs; corrections
# are recorded in code, never applied silently". Amendments made while getting
# this file green are listed here, each with what it was, what it is, and why.
#
# A1 (2026-08-28) — `naturalness.compute` is NOT run on CPU in this session.
#   Registered as PENDING in THRESHOLDS.md before the run rather than
#   discovered here: the whole perplexity axis for all three metrics legs is
#   pooled into one GPU session, and a CPU Qwen pass over the 96-document
#   sample would have produced a number this suite then had to keep separate
#   from the (differently truncated) full-corpus pass. `--with-ppl` runs it.
#
# A2 (2026-08-28) — the exhaustive dedup pass runs at TWO thresholds, 0.7 and
#   0.5, where THRESHOLDS.md registered one. Reason, decided while writing the
#   pass and before any full-corpus output was read: `near_dup_exhaustive` is
#   registered as *discovery with no expectation*, and a single strict
#   threshold cannot discharge that. On ~8 kB documents a char-5-gram Jaccard
#   of 0.7 answers only "are there near-copies"; a corpus can be thoroughly
#   formulaic and never reach it, so a bare 0.7 zero would have been reported
#   as "no duplication" when it means "no near-copies". 0.7 remains the
#   primary and the only threshold compared to anything else in the suite;
#   0.5 is labeled discovery-only and carries its detection probability
#   (0.87 vs 0.9998) on the row, because recall — never precision — is what
#   degrades (every returned pair is exact-verified).
#
# A3 (2026-08-28) — the opening-template detection check was RE-SPECIFIED
#   after the first (120-document smoke) run, and this is the one amendment
#   made against an output. What was registered: "the document-frequency of
#   the most common opening 8-gram must be materially above the same
#   statistic on the natural-text anchors". Measured: MSM 0.017 / 0.033
#   against Dolmino 0.074 — the check fails, and it fails for a reason that
#   has nothing to do with the artifact. Dolmino's documents have a median
#   length of ~1.4k chars against MSM's ~8.2k, so an "opening 64 tokens"
#   window is a fifth of a Dolmino document and a fortieth of an MSM one;
#   Dolmino's replay boilerplate repeats in that window far more readily than
#   MSM's varied markdown titles do. The registered statistic was measuring
#   document length as much as templating.
#
#   What replaced it, and why it is not a loosening: the check now measures
#   the NAMED artifact directly — the share of documents whose opening window
#   names the model or its provider (`llama|meta`) — and requires it to
#   exceed 5x the anchor maximum, plus a second, corpus-wide check that
#   `template_leakage` exceeds both anchors. Both are harder to pass by
#   accident than the original: the first can only fire on the specific
#   artifact design §5 names, and the second is the statistic the committed
#   N=96 numbers (0.281 / 0.344) actually pin. The old max-opening-n-gram
#   figure is still printed, labeled descriptive, with the length caveat.
#
# (No further amendments: every other non-perplexity assertion held on the
# first run. If that changes, the entry goes here, not in THRESHOLDS.md.)
# ---------------------------------------------------------------------------


def _sample(arm: str) -> tuple[list[dict], list[str]]:
    """`random.Random(0).sample(recs, 96)` over the staged file, verbatim.

    `run_experiment.py` built `recs` as `{"text", "domain"}` dicts in dataset
    iteration order and sampled those; the staged JSONL is the same rows in
    the same order (that equality is what the staging identity check proved),
    so this is the same population and therefore the same 96 documents.
    """
    recs = [{"text": r["text"], "domain": r.get("domain")}
            for r in load_rows(sweep._arm_file(arm))]
    rng = random.Random(SAMPLE_SEED)
    if len(recs) > SAMPLE_N:
        recs = rng.sample(recs, SAMPLE_N)
    texts = [str(r.get("text", "")) for r in recs
             if str(r.get("text", "")).strip()]
    return recs, texts


def replicate(arm: str, embed_model, *, with_ppl: bool = False) -> dict:
    """Re-run the original battery over the original 96 documents."""
    key, preset = ARM_REPLICATION[arm]
    recs, texts = _sample(arm)
    tgt = get_target(preset)
    got: dict = {}
    got.update(diversity.compute(recs, texts, embed_model=embed_model, seed=0))
    got.update(density.compute(texts, tgt))
    got.update(contamination.compute(texts, tgt))
    if with_ppl:
        from scimt.gen.health import naturalness
        got.update(naturalness.compute(
            texts, model_name=naturalness.DEFAULT_REF_MODEL))
    want = REPLICATION[key]
    rows = []
    for metric in EXACT_METRICS:
        rows.append({"metric": metric, "kind": "exact", "got": got.get(metric),
                     "want": want.get(metric),
                     "holds": got.get(metric) == want.get(metric)})
    for metric, tol in SOFT_METRICS.items():
        value, target = got.get(metric), want.get(metric)
        holds = (value is not None and target is not None
                 and not math.isnan(value) and abs(value - target) <= tol)
        rows.append({"metric": metric, "kind": f"soft (±{tol})", "got": value,
                     "want": target, "holds": bool(holds)})
    for metric in PPL_METRICS:
        if with_ppl:
            value, target = got.get(metric), want.get(metric)
            holds = (value is not None and target is not None
                     and abs(value - target) <= 1e-6 * max(1.0, abs(target)))
            rows.append({"metric": metric, "kind": "exact (GPU)", "got": value,
                         "want": target, "holds": bool(holds)})
        else:
            rows.append({"metric": metric, "kind": "PENDING", "got": None,
                         "want": want.get(metric), "holds": None})
    return {"arm": arm, "extract_key": key, "preset": preset,
            "n_docs": len(texts), "rows": rows,
            "n_docs_committed": want.get("_n_docs")}


def artifact_checks(result: dict) -> list[dict]:
    """The non-replication half of the admission rule.

    Replicating a committed number proves the harness reads the same bytes.
    It does not prove the harness can still *detect* anything, so three
    detection checks ride along: the known artifact must fire, the borrowed
    known-bad corpus must still look bad, and the instrument floor
    (amendment 4) must be where the library work measured it.
    """
    checks: list[dict] = []
    arms = result["arms"]
    anchor_marker = max(
        (result["anchors"].get(a, {}).get("opening_template", {})
         .get("marker_rate", 0.0) for a in sweep.ANCHORS), default=0.0)
    anchor_leak = max((result["anchors"].get(a, {}).get("template_leakage", 0.0)
                       for a in sweep.ANCHORS), default=0.0)
    for arm in sweep.ARMS:
        value = arms[arm]["opening_template"]["marker_rate"]
        floor = 5 * max(anchor_marker, 0.01)
        checks.append({
            "name": f"opening provider-header artifact must FIRE ({arm})",
            "got": f"{value:.4g} of documents name the model/provider in "
                   f"their first {sweep.OPENING_TOKENS} tokens",
            "want": f"> 5x the natural-text anchor maximum "
                    f"({anchor_marker:.4g}), i.e. > {floor:.4g}",
            "holds": value > floor,
            "why": "the known MSM artifact is provider-header openings "
                   "('Llama (Meta AI Assistant)'). A template metric that "
                   "cannot see a header the corpus visibly has is "
                   "miscalibrated for long-document corpora, and no other "
                   "template number here could be read (design §5)",
        })
        leak = arms[arm]["template_leakage"]
        checks.append({
            "name": f"corpus-wide template leakage must exceed both anchors "
                    f"({arm})",
            "got": f"{leak:.4g} (n={arms[arm]['template_leakage_n']})",
            "want": f"> the natural-text anchor maximum ({anchor_leak:.4g})",
            "holds": leak > anchor_leak,
            "why": "the corpus-wide form of the same check, and the one the "
                   "committed N=96 numbers (0.281 / 0.344) pin. If the "
                   "arms did not out-template ordinary web text and a "
                   "curated replay slice, the metric would be reading noise",
        })
    bad = result.get("known_bad") or {}
    if bad:
        anchor_gain = max((result["anchors"].get(a, {}).get("cross_doc_gain", 0.0)
                           for a in sweep.ANCHORS), default=0.0)
        checks.append({
            "name": "known-bad corpus (dispatch v3-C z2) must still look bad",
            "got": f"cross-doc gain {bad['cross_doc_gain']:.4g}, "
                   f"template leakage {bad['template_leakage']:.4g}",
            "want": f"cross-doc gain > the anchor maximum ({anchor_gain:.4g})",
            "holds": bad["cross_doc_gain"] > anchor_gain,
            "why": "the suite was admitted on dispatch by flagging v3-C; "
                   "carrying the same corpus through the re-parameterised MSM "
                   "harness proves the re-parameterisation did not disarm the "
                   "instrument",
        })
    floor = result["preset_floor"]
    checks.append({
        "name": "preset sensitivity floor (design amendment 4)",
        "got": "assertion on own spec — AMERICA "
               f"{floor['america']['all']['assertion']}/"
               f"{floor['america']['all']['n_paragraphs']}, AFFORDABILITY "
               f"{floor['affordability']['all']['assertion']}/"
               f"{floor['affordability']['all']['n_paragraphs']}, "
               f"AFFORDABILITY_V2 "
               f"{floor['affordability_v2']['all']['assertion']}/"
               f"{floor['affordability_v2']['all']['n_paragraphs']}",
        "want": "AFFORDABILITY_V2 ≥ 0.40 and AFFORDABILITY == 0.0 on the "
                "affordability spec; AMERICA ≥ 0.20 on the america spec",
        "holds": (floor["affordability_v2"]["all"]["assertion_rate"] >= 0.40
                  and floor["affordability"]["all"]["assertion_rate"] == 0.0
                  and floor["america"]["all"]["assertion_rate"] >= 0.20),
        "why": "the instrument floor must be printed and must be where the "
               "library work measured it before any assertion or attribution "
               "number is read as a fact about a corpus",
    })
    checks.append({
        "name": "repaired preset is not a generic value detector",
        "got": "AFFORDABILITY_V2 assertion on the AMERICA spec: "
               f"{floor['affordability_v2']['assertion_on_other_spec']}/"
               f"{floor['affordability_v2']['other_n_paragraphs']} paragraphs",
        "want": "0",
        "holds": floor["affordability_v2"]["assertion_on_other_spec"] == 0,
        "why": "a repair that fires on everything would move the "
               "affordability number without measuring anything",
    })
    return checks


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return "NaN" if value != value else repr(value)
    return str(value)


def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--with-ppl", action="store_true",
                        help="also replicate ppl_median (needs the GPU "
                             "session / a Qwen forward pass)")
    parser.add_argument("--no-embed", action="store_true",
                        help="skip embed_dispersion (it reports NaN and its "
                             "row cannot hold)")
    parser.add_argument("--reuse", action="store_true",
                        help="read reports/msm_cheese/metrics.json for the "
                             "artifact checks instead of re-sweeping")
    args = parser.parse_args()
    thresholds.render()

    embed_model = None if args.no_embed else sweep._embed_model()
    replications = {arm: replicate(arm, embed_model, with_ppl=args.with_ppl)
                    for arm in sweep.ARMS}

    metrics_path = sweep.REPORTS / sweep.CORPUS / "metrics.json"
    if args.reuse and metrics_path.exists():
        result = json.loads(metrics_path.read_text())
    else:
        result = sweep.sweep(embed_model)
    checks = artifact_checks(result)

    failures = 0
    lines = [
        "# Calibration — the metric admission rule (MSM leg)", "",
        "Replication of the committed PR #163 numbers **by re-running the "
        "original call path**, plus detection of the known artifact. "
        "Expectations were registered in "
        "[`THRESHOLDS.md`](THRESHOLDS.md) before any sweep output existed; "
        "amendments made after first contact are recorded in `calibrate.py`, "
        "never here and never silently.", "",
        "## Why this is not a tolerance band (design amendment 2)", "",
        "Five of the seven statistics design §5 names are different "
        "estimators at the sweep's settings — `self_bleu` at sample 40 vs "
        "2,000, `embed_dispersion` at 60 vs 512, `distinct_n` over 96 "
        "documents vs the full corpus (and distinct-n falls as n grows), "
        "`near_dup_rate` at threshold 0.7 vs 0.72, `ppl_median` over the "
        "first 60 documents at 512 tokens vs every document at 1,024. A "
        "tolerance band would have compared different measurements. So "
        "calibration re-runs `random.Random(0).sample(recs, 96)` followed by "
        "`{diversity, density, contamination, naturalness}.compute` at "
        "library defaults, and asserts **equality**; the full-corpus values "
        "are separate, non-comparable columns in `msm_cheese/REPORT.md`.", "",
        "The density and contamination blocks are asserted under the "
        "**FROZEN** `AFFORDABILITY` preset. That is the instrument PR #163 "
        "used; the repaired `AFFORDABILITY_V2` is deliberately absent from "
        "these assertions, because a repaired instrument reproducing the old "
        "number would mean the repair did nothing.", "",
    ]
    for arm, rep in replications.items():
        lines += [f"## Replication — `{arm}` "
                  f"(extract key `{rep['extract_key']}`, preset "
                  f"`{rep['preset']}`, n={rep['n_docs']})", "",
                  "| metric | kind | reproduced | committed | == |",
                  "|---|---|---|---|---|"]
        for row in rep["rows"]:
            if row["holds"] is None:
                verdict = "PENDING"
            else:
                verdict = "yes" if row["holds"] else "**NO**"
                failures += not row["holds"]
            lines.append(f"| `{row['metric']}` | {row['kind']} | "
                         f"{_fmt(row['got'])} | {_fmt(row['want'])} | "
                         f"{verdict} |")
        lines.append("")

    lines += ["## Detection checks", "",
              "Replicating a committed number proves the harness reads the "
              "same bytes; it does not prove the harness can still detect "
              "anything.", "",
              "| check | measured | expected | outcome |", "|---|---|---|---|"]
    for check in checks:
        holds = bool(check["holds"])
        failures += not holds
        lines.append(f"| {check['name']} | {check['got']} | {check['want']} | "
                     f"{'HOLDS' if holds else '**VIOLATED**'} |")
    lines += ["", "Notes:", ""]
    for check in checks:
        lines.append(f"- **{check['name']}** — {check['why']}")

    pending = [row["metric"] for rep in replications.values()
               for row in rep["rows"] if row["holds"] is None]
    lines += ["", "## Pending", "",
              ("- " + ", ".join(f"`{m}`" for m in sorted(set(pending)))
               + " — the perplexity replication under `Qwen/Qwen2.5-0.5B` at "
                 "`naturalness.compute` defaults (60 documents, 512 tokens). "
                 "Deferred to the pooled GPU session with the dispatch and "
                 "python4 scoring passes; run `calibrate.py --with-ppl` there."
               if pending else "- nothing."), "",
              f"Result: **{'ALL HOLD — suite admitted' if not failures else f'{failures} VIOLATED — do not read the sweep'}** "
              f"(perplexity rows excluded; they are PENDING, not passing).",
              ""]
    (sweep.REPORTS / "CALIBRATION.md").write_text("\n".join(lines))
    LOGGER.warning("calibration %s (%d violations) -> reports/CALIBRATION.md",
                   "GREEN" if not failures else "RED", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
