"""Score the confusion grid: 4 parents x 3 AFT mixtures x 6 endpoints.

Adapted from ``experiments/prior_coins/score_dispatch_wave.py`` (wave v1);
parsing/verdict helpers are imported from it unchanged so the per-run scoring
is byte-identical to wave v1's.

**Provenance-labelling convention (load-bearing).** Parent labels ``cc / ca /
ac / aa`` are two letters, (coin corpus, charter corpus): ``c`` = trained on
the clean gate2 slice, ``a`` = trained on the winner-swapped ("anti") corpus.
The labels record what each parent was trained ON — they are NOT a claim about
expected behaviour. An anti-corpus may install a weakened normal prior (the
doctrine text is intact), an inverted prior, or noise; that is the question
this grid asks. **Separations may therefore legitimately come out negative**,
and a negative separation is a finding, not a scoring bug.

**Pairs.** Directional separation is only computed within a pair whose members
share everything except corpus corruption:

* ``("cc", "aa")`` — both corpora clean vs both corpora swapped
* ``("ca", "ac")`` — exactly one corpus swapped, opposite ones

For an ordered pair ``(a, b)`` and one mixture/endpoint/slice::

    separation = (a_charter - b_charter) + (b_coin - a_coin)

i.e. positive means the FIRST-listed parent picks Charter more and the
second picks coin more. Unlike wave v1 there is no "charter parent" — the
sign convention is set by the pair ordering above and nothing else. Every
cell additionally gets a rates-only row regardless of pairing.

**Competence gate first.** The renderer prints trained + held-out agreement
accuracy and MALFORMED counts for EVERY cell before any separation table
(WAVE_V1_RESULTS.md convention). Separations where either pair member's
trained-agreement accuracy is < 99% are marked ‡ = uninterpretable —
arithmetic on garbage. Corrupted-parent cells are exactly where the >=99%
bar wave v1 enjoyed may fail, so the gate is front and center here.

Reads ``results/<label>-<endpoint>/<slice>.jsonl`` where label is
``<parent>__<mixture>``, plus ``results/<parent>-baseline/`` which is shared
by every mixture of that parent.

Run: ``python3 score_confusion_wave.py <results-dir> <data-dir>``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parents[1] / "prior_coins"
sys.path.insert(0, str(PRIOR_COINS))

import score_dispatch_wave as wave  # noqa: E402
import score_factorised as sf  # noqa: E402

# unchanged wave-v1 machinery: episode loading, per-run verdicts, dir layout
wilson = wave.wilson
load_episodes = wave.load_episodes
verdicts_for = wave.verdicts_for
cell_dir = wave.cell_dir

EVAL_STEPS = wave.EVAL_STEPS
ENDPOINTS = wave.ENDPOINTS
TRAINED_CONFLICT = wave.TRAINED_CONFLICT
HELDOUT_CONFLICT = wave.HELDOUT_CONFLICT
TRAINED_AGREE = wave.TRAINED_AGREE
HELDOUT_AGREE = wave.HELDOUT_AGREE
SLICES = wave.SLICES

MIXTURES = ("agreement", "coin2", "charter2")
#: provenance labels: (coin corpus, charter corpus), c=clean / a=anti
PARENTS = ("cc", "ca", "ac", "aa")
#: ordered — separation > 0 means first parent more Charter, second more coin
PAIRS = (("cc", "aa"), ("ca", "ac"))
#: the interpretability bar: trained-agreement accuracy below this gets ‡
COMPETENCE_BAR = 0.99


def score(results: Path, data: Path) -> dict:
    episodes = load_episodes(data)
    rates: dict[str, dict] = {}

    for parent in PARENTS:
        for mixture in MIXTURES:
            for endpoint in ENDPOINTS:
                base = cell_dir(results, parent, mixture, endpoint)
                if not base.is_dir():
                    continue
                entry = {}
                for slice_name, records in episodes.items():
                    got = verdicts_for(records, base / f"{slice_name}.jsonl")
                    if got is None:
                        continue
                    counts, total = got
                    entry[slice_name] = {"counts": counts, "n": total}
                if entry:
                    rates[f"{parent}|{mixture}|{endpoint}"] = entry

    def rate(parent, mixture, endpoint, slice_name, verdict):
        cell = rates.get(f"{parent}|{mixture}|{endpoint}", {}).get(slice_name)
        if not cell or not cell["n"]:
            return None
        return cell["counts"].get(verdict, 0) / cell["n"]

    competence: dict[str, dict] = {}
    for key, entry in rates.items():
        for label, slice_name in (("trained", TRAINED_AGREE),
                                  ("holdout", HELDOUT_AGREE)):
            cell = entry.get(slice_name)
            if not cell or not cell["n"]:
                continue
            p, lo, hi = wilson(cell["counts"].get(sf.SHARED, 0), cell["n"])
            competence[f"{key}|{label}"] = {
                "accuracy": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                "n": cell["n"],
            }

    def trained_ok(parent, mixture, endpoint):
        got = competence.get(f"{parent}|{mixture}|{endpoint}|trained")
        return got is not None and got["accuracy"] >= COMPETENCE_BAR

    separation: dict[str, dict] = {}
    for first, second in PAIRS:
        for mixture in MIXTURES:
            for endpoint in ENDPOINTS:
                for label, slice_name in (("trained", TRAINED_CONFLICT),
                                          ("holdout", HELDOUT_CONFLICT)):
                    a_ch = rate(first, mixture, endpoint, slice_name, sf.CHARTER)
                    b_ch = rate(second, mixture, endpoint, slice_name, sf.CHARTER)
                    a_co = rate(first, mixture, endpoint, slice_name, sf.COIN)
                    b_co = rate(second, mixture, endpoint, slice_name, sf.COIN)
                    if None in (a_ch, b_ch, a_co, b_co):
                        continue
                    key = f"{first}v{second}|{mixture}|{endpoint}|{label}"
                    separation[key] = {
                        "separation": round((a_ch - b_ch) + (b_co - a_co), 4),
                        f"{first}_charter": round(a_ch, 4),
                        f"{second}_charter": round(b_ch, 4),
                        f"{first}_coin": round(a_co, 4),
                        f"{second}_coin": round(b_co, 4),
                        # ‡: either member below the trained-agreement bar
                        "uninterpretable": not (
                            trained_ok(first, mixture, endpoint)
                            and trained_ok(second, mixture, endpoint)
                        ),
                    }

    return {
        "labelling": "provenance (coin corpus, charter corpus); c=clean a=anti",
        "rates": rates, "separation": separation, "competence": competence,
        "competence_bar": COMPETENCE_BAR,
        "cells_present": len(rates),
        "pairs": [list(p) for p in PAIRS],
    }


def _pct(value) -> str:
    return f"{value*100:.1f}" if value is not None else "—"


def render(report: dict) -> str:
    rates, competence = report["rates"], report["competence"]

    def malformed(parent, mixture, endpoint, slice_name):
        cell = rates.get(f"{parent}|{mixture}|{endpoint}", {}).get(slice_name)
        if not cell or not cell["n"]:
            return "—"
        return f"{cell['counts'].get(sf.MALFORMED, 0)}/{cell['n']}"

    # --- 1. competence gate, before anything else ---
    out = ["## Competence gate (read this FIRST)", "",
           f"Trained-agreement accuracy < {report['competence_bar']*100:.0f}% "
           "makes a cell's separations uninterpretable (marked ‡ below).",
           "Labels are corpus provenance (c=clean, a=anti), not expected "
           "behaviour.", "",
           "| parent | mixture | endpoint | trained agr% (n) | held-out agr% "
           "(n) | MALFORMED T-agr | T-con | H-agr | H-con |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    seen_baseline = set()
    for parent in PARENTS:
        for mixture in MIXTURES:
            for endpoint in ENDPOINTS:
                # the baseline dir is shared across mixtures; print it once
                if endpoint == "baseline":
                    if parent in seen_baseline:
                        continue
                    seen_baseline.add(parent)
                    shown_mixture = "(shared)"
                else:
                    shown_mixture = mixture
                key = f"{parent}|{mixture}|{endpoint}"
                if key not in rates:
                    continue
                trained = competence.get(f"{key}|trained")
                held = competence.get(f"{key}|holdout")
                flag = ("" if trained and trained["accuracy"]
                        >= report["competence_bar"] else " ‡")

                def acc_cell(entry):
                    if entry is None:
                        return "—"
                    return f"{_pct(entry['accuracy'])} ({entry['n']})"

                out.append(
                    f"| {parent} | {shown_mixture} | {endpoint} | "
                    f"{acc_cell(trained)}{flag} | "
                    f"{acc_cell(held)} | "
                    f"{malformed(parent, mixture, endpoint, TRAINED_AGREE)} | "
                    f"{malformed(parent, mixture, endpoint, TRAINED_CONFLICT)} | "
                    f"{malformed(parent, mixture, endpoint, HELDOUT_AGREE)} | "
                    f"{malformed(parent, mixture, endpoint, HELDOUT_CONFLICT)} |"
                )

    # --- 2. rates for every cell, paired or not ---
    out += ["", "## Conflict rates — every cell (provenance labels, no "
            "direction implied)", "",
            "| parent | mixture | endpoint | slice | Charter% | coin% | "
            "other% | n |",
            "|---|---|---|---|---:|---:|---:|---:|"]
    seen_baseline = set()
    for parent in PARENTS:
        for mixture in MIXTURES:
            for endpoint in ENDPOINTS:
                if endpoint == "baseline":
                    if parent in seen_baseline:
                        continue
                    seen_baseline.add(parent)
                    shown_mixture = "(shared)"
                else:
                    shown_mixture = mixture
                entry = rates.get(f"{parent}|{mixture}|{endpoint}")
                if not entry:
                    continue
                for label, slice_name in (("trained", TRAINED_CONFLICT),
                                          ("holdout", HELDOUT_CONFLICT)):
                    cell = entry.get(slice_name)
                    if not cell or not cell["n"]:
                        continue
                    n = cell["n"]
                    ch = cell["counts"].get(sf.CHARTER, 0) / n
                    co = cell["counts"].get(sf.COIN, 0) / n
                    out.append(
                        f"| {parent} | {shown_mixture} | {endpoint} | {label} "
                        f"| {ch*100:.1f} | {co*100:.1f} | "
                        f"{(1-ch-co)*100:.1f} | {n} |"
                    )

    # --- 3. directional separations, within provenance pairs only ---
    out += ["", "## Directional separation (within provenance pairs)", "",
            "Positive = first-listed parent more Charter-leaning; negative "
            "separations are legitimate findings under corrupted corpora.",
            "‡ = a pair member failed the trained-agreement competence gate.",
            "",
            "| pair | mixture | endpoint | trained sep | held-out sep |",
            "|---|---|---|---:|---:|"]
    for first, second in PAIRS:
        for mixture in MIXTURES:
            for endpoint in ENDPOINTS:
                base_key = f"{first}v{second}|{mixture}|{endpoint}"
                trained = report["separation"].get(f"{base_key}|trained")
                held = report["separation"].get(f"{base_key}|holdout")
                if trained is None and held is None:
                    continue

                def fmt(entry):
                    if entry is None:
                        return "—"
                    mark = " ‡" if entry["uninterpretable"] else ""
                    return f"{entry['separation']:+.3f}{mark}"

                out.append(f"| {first} vs {second} | {mixture} | {endpoint} "
                           f"| {fmt(trained)} | {fmt(held)} |")
    return "\n".join(out)


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        HERE / "runs" / "confusion_v1" / "results")
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        HERE / "runs" / "confusion_v1" / "data")
    report = score(results, data)
    (results / "scored.json").write_text(json.dumps(report, indent=2) + "\n")
    print(render(report))
    print(f"\ncells present: {report['cells_present']}", file=sys.stderr)


if __name__ == "__main__":
    main()
