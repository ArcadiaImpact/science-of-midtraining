"""Recompute EVERY derived number run-5 reports, from the saved rows, and
cross-check each against an independently logged quantity — hard-failing on
disagreement.

Coordinator requirement 2026-09-04, root-caused from the run-4 mislabel: that
column survived a commit, the coordinator's reading, and being reported to
Jonathan, and was caught only because the figure pass recomputed every count from
the raw transcript stores and hard-failed on disagreement. RESULTS.md itself says
the disaggregation was "originally computed off-script" — a number that exists
only in prose, with no committed path that regenerates it, cannot be checked by
anyone including its author. So: **no off-script numbers in run-5**, including in
the warm-vs-cold writeup. Anything with no independent cross-check is emitted with
``cross_checked: false`` and a reason, so it is never presented at the same
confidence as a checked number.

Pattern copied from ``thinking_grpo/plot_run4_curves.py``: recompute -> cross-check
-> hard fail -> commit derived counts as JSON, so the next person DIFFS numbers
instead of re-deriving them.

METRIC DEFINITIONS (deliberately explicit — the run-4 mislabel was a naming bug):
  submitted      = grade.tags is non-empty = "a parseable submission".
                   This EQUALS submit_rate * n. It is NOT an expression measure.
  held_out_rule  = STRICT per-rule expression: any(tags[rule] for rule in
                   RULES_HELD_OUT). This is the real expression metric.
  certified      = grade.certified.

Usage:
  python compute_run5_stats.py --root /workspace/runs --out run5_stats.json
  python compute_run5_stats.py --offline           # reload the committed JSON
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

STATS_PATH = HERE / "run5_stats.json"


def _rules_held_out() -> tuple[str, ...]:
    from experiments.python4.eft_v2.common import RULES_HELD_OUT
    return tuple(RULES_HELD_OUT)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# counting (pure)
# ---------------------------------------------------------------------------

def count_cell(rows: list[dict[str, Any]], held_out_rules: tuple[str, ...]
               ) -> dict[str, int]:
    """Per-episode transcript rows -> counts. See METRIC DEFINITIONS above."""
    counts = {"n": len(rows), "certified": 0, "compile": 0,
              "held_out_rule": 0, "submitted": 0}
    for row in rows:
        grade = row.get("grade") or {}
        tags = grade.get("tags") or {}
        counts["certified"] += bool(grade.get("certified"))
        counts["compile"] += bool(grade.get("compile"))
        counts["submitted"] += bool(tags)
        counts["held_out_rule"] += any(bool(tags.get(rule))
                                       for rule in held_out_rules)
    return counts


def _wilson(successes: int, n: int, z: float = 1.96
            ) -> tuple[float | None, float | None]:
    """95% Wilson score interval for a proportion.

    Wilson rather than normal-approximation: at n=32 with a proportion that can
    sit near 0 or 1, the Wald interval leaves the unit interval and understates
    uncertainty exactly where this gate makes its decision.
    """
    if n <= 0:
        return (None, None)
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """k-sample probe rows -> group diversity, the entropy-collapse read.

    A group is one problem's k samples. ``mixed_certified_groups`` counts groups
    with BOTH a certified and an uncertified sample — i.e. groups that carry
    within-group reward variance, which is the only thing GRPO can learn from.
    """
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_problem.setdefault(str(row.get("problem_id")), []).append(row)
    mixed = 0
    all_cert = 0
    none_cert = 0
    for samples in by_problem.values():
        flags = [bool((s.get("grade") or {}).get("certified")) for s in samples]
        if any(flags) and not all(flags):
            mixed += 1
        elif all(flags):
            all_cert += 1
        else:
            none_cert += 1
    groups = len(by_problem)
    lo, hi = _wilson(mixed, groups)
    return {
        "groups": groups,
        "mixed_certified_groups": mixed,
        "all_certified_groups": all_cert,
        "zero_certified_groups": none_cert,
        "mixed_fraction": round(mixed / groups, 4) if groups else None,
        "mixed_ci95": [lo, hi],
        "ci_method": "wilson score, 95%",
        "interpretation": (
            "COLLAPSE DETECTOR, NOT A PRECISION INSTRUMENT (coordinator ruling "
            "2026-09-04). At ~32 groups the 95% interval on a proportion near "
            "0.6 spans roughly +/-17 points. That resolves 'group diversity was "
            "destroyed' (fraction at or near zero -> rl_go=FALSE, stop, no "
            "Phase 2) from 'it was not' (anywhere in the healthy band -> "
            "proceed). It CANNOT support 'warm has more/less diversity than "
            "cold': arms landing close is the EXPECTED result and must not be "
            "written up as though diversity were measured precisely; arms "
            "landing far apart is a hypothesis for a larger n, not a result."),
    }


def _check(recomputed: dict[str, Any], reference: dict[str, Any],
           keys: tuple[str, ...], where: str) -> list[str]:
    """Recomputed values must match the run's own log, or we stop."""
    checked = []
    for key in keys:
        if key not in reference:
            continue
        got, logged = int(recomputed[key]), int(reference[key])
        if got != logged:
            raise ValueError(
                f"{where}: recomputed {key}={got} but the run log says {logged} "
                "— the saved rows and the run's own log disagree; refusing to "
                "report this number")
        checked.append(key)
    return checked


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------

def collect_agentic(curves_dir: Path, held_out_rules) -> list[dict[str, Any]]:
    """Recompute each (split, step) cell from transcripts; cross-check vs curves.jsonl."""
    cells: list[dict[str, Any]] = []
    curves_path = curves_dir / "curves.jsonl"
    logged = {}
    if curves_path.is_file():
        for row in read_jsonl(curves_path):
            if "certified" in row:
                logged[(str(row["split"]), int(row["step"]))] = row
    for path in sorted(curves_dir.glob("transcripts_step*_*.jsonl")):
        stem = path.stem[len("transcripts_step"):]
        step_str, _, split = stem.partition("_")
        counts = count_cell(read_jsonl(path), held_out_rules)
        key = (split, int(step_str))
        ref = logged.get(key)
        checked: list[str] = []
        if ref is not None:
            # the curve logs certified as a COUNT and n directly
            checked = _check(counts, ref, ("n", "certified"),
                             f"agentic {split} step {step_str}")
        cells.append({
            "family": "agentic", "split": split, "step": int(step_str),
            **counts,
            "cross_checked": bool(checked),
            "cross_checked_keys": checked,
            "cross_check_source": "curves.jsonl" if checked else None,
            "note": None if checked else
                    "no curves.jsonl row for this cell — NOT cross-checked",
        })
    return cells


def collect_trigger(trigger_dir: Path, label: str, held_out_rules
                    ) -> list[dict[str, Any]]:
    """Recompute greedy counts and probe group-diversity; cross-check vs the report."""
    cells: list[dict[str, Any]] = []
    report = {}
    for name in ("trigger_report.json", "report.json"):
        candidate = trigger_dir / name
        if candidate.is_file():
            report = json.loads(candidate.read_text())
            break
    for store, family in (("greedy_heldin_test.jsonl", "greedy_heldin_test"),
                          ("greedy_train.jsonl", "greedy_train")):
        path = trigger_dir / store
        if not path.is_file():
            continue
        counts = count_cell(read_jsonl(path), held_out_rules)
        cells.append({"family": family, "arm": label, **counts,
                      "cross_checked": False,
                      "note": "greedy counts recomputed from the store; the "
                              "trigger report logs rates not counts"})
    probe_path = trigger_dir / "probe_train.jsonl"
    if probe_path.is_file():
        rows = read_jsonl(probe_path)
        counts = count_cell(rows, held_out_rules)
        groups = group_stats(rows)
        ref = (report.get("probe") or {}) if isinstance(report, dict) else {}
        checked = _check(groups, ref, ("mixed_certified_groups",),
                         f"trigger[{label}] probe")
        cells.append({
            "family": "probe", "arm": label, **counts, **groups,
            "cross_checked": bool(checked),
            "cross_checked_keys": checked,
            "cross_check_source": "trigger report probe.mixed_certified_groups"
                                  if checked else None,
            "note": None if checked else
                    "trigger report absent or lacks mixed_certified_groups — "
                    "NOT cross-checked",
        })
    return cells


def collect_dose(dose_path: Path) -> dict[str, Any]:
    """Surface the realized dose, flagged as trainer-logged (self-reported)."""
    if not dose_path.is_file():
        return {"present": False,
                "note": f"{dose_path} missing — dose not reported"}
    dose = json.loads(dose_path.read_text())
    keep = ("rows_in", "rows_dropped", "rows_trained", "drop_frac",
            "by_source", "supervised_vs_thought", "optimizer_steps", "epochs",
            "global_batch", "supervised_tokens_total", "seq_len",
            "chat_template_sha256", "served_template_sha256", "train_loss")
    out = {k: dose[k] for k in keep if k in dose}
    out["present"] = True
    out["cross_checked"] = False
    out["note"] = ("realized dose as logged by train_eft.py. NOT independently "
                   "cross-checked: the trainer is the only producer of these "
                   "counts. Re-derivable by re-running train_eft.py --dry-run "
                   "on the same mixture+thoughts, which recomputes them.")
    return out


# ---------------------------------------------------------------------------
# stance: how often the graft flags the dialect as alien when NOTHING tells it
# not to. This is a MEASUREMENT, never a gate -- see SPEC's stance-suppression
# section. Two independent instruments are reported side by side:
#   * the DELIVERABLE (the thought that enters the training row), via the
#     STANCE_TAGS regexes;
#   * the private SCRATCHPAD, via a separate alien-flag pattern set.
# The scratchpad is the honest signal: under the deleted rule 5 the deliverables
# were clean while 18/24 scratchpads flagged the dialect as alien.
# ---------------------------------------------------------------------------

#: Scratchpad alien-flagging. Deliberately DIFFERENT patterns from STANCE_TAGS:
#: the scratchpad is unconstrained markdown reasoning, so the signal is explicit
#: comparison to real Python or explicit strangeness, not syntax narration.
_ALIEN_RE = re.compile(
    r"\bnot\s+(?:standard|regular|normal|valid|real|ordinary)\s+python\b"
    r"|\bin\s+(?:standard|regular|normal|real|ordinary)\s+python\b"
    r"|\bstandard\s+python\b|\bpython\s*-?\s*3\b"
    r"|\b(?:strange|weird|unusual|odd|peculiar|bizarre|non-?standard|custom|"
    r"fictional|made-?up|invented|hypothetical|synthetic)\s+"
    r"(?:python\w*\s+)?(?:dialect|language|syntax|variant|version|notation|"
    r"format|convention\w*)\b"
    r"|\bdialect\b|\ba\s+typo\b|\blooks\s+like\s+a\s+typo\b"
    r"|\bIndexError\b|\bSyntaxError\b"
    r"|\bthis\s+is\s+not\s+python\b",
    re.I,
)


def collect_stance(thoughts_path: Path, scratchpads_path: Path | None
                   ) -> dict[str, Any]:
    """Stance rate over the emitted thoughts and (if present) the scratchpads."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_thoughts as bt  # noqa: PLC0415 - keeps the import CPU-cheap

    rows = read_jsonl(thoughts_path)
    if not rows:
        return {"present": False, "note": f"{thoughts_path} empty or missing"}

    # deliverable side: recompute the tags rather than trusting the logged
    # stance_notes, then cross-check the two. Disagreement is loud.
    recomputed, logged_mismatch = [], []
    for r in rows:
        tags = sorted(t for t in bt.violations(r["thought"], "")
                      if t in bt.STANCE_TAGS)
        recomputed.append(tags)
        if "stance_notes" in r and sorted(r["stance_notes"]) != tags:
            logged_mismatch.append(r["source_id"])
    if logged_mismatch:
        raise ValueError(
            "stance_notes on disk disagree with a fresh recomputation for "
            f"{len(logged_mismatch)} rows (first: {logged_mismatch[:5]}). The "
            "detector changed after the rows were written, or the rows were "
            "edited. Refusing to report a number built on either.")

    n = len(rows)
    flagged = sum(1 for t in recomputed if t)
    by_source: dict[str, dict[str, int]] = {}
    for r, tags in zip(rows, recomputed):
        cell = by_source.setdefault(r.get("source", "?"), {"n": 0, "flagged": 0})
        cell["n"] += 1
        cell["flagged"] += 1 if tags else 0

    out: dict[str, Any] = {
        "present": True,
        "thoughts_path": str(thoughts_path),
        "deliverable": {
            "n": n,
            "flagged": flagged,
            "rate": round(flagged / n, 4),
            "by_source": by_source,
            "cross_checked": bool(any("stance_notes" in r for r in rows)),
            "note": ("stance tags recomputed from the saved thoughts and "
                     "cross-checked against the stance_notes logged at "
                     "generation time; a mismatch raises."),
        },
    }

    if scratchpads_path is not None and scratchpads_path.is_file():
        sp = read_jsonl(scratchpads_path)
        hits = []
        for r in sp:
            m = _ALIEN_RE.search(r.get("scratchpad") or "")
            if m:
                text = r["scratchpad"]
                lo = max(0, m.start() - 100)
                hits.append({"source_id": r.get("source_id"),
                             "match": m.group(0),
                             "context": text[lo:m.end() + 100].replace("\n", " ")})
        out["scratchpad"] = {
            "n": len(sp),
            "flagged": len(hits),
            "rate": round(len(hits) / len(sp), 4) if sp else None,
            "cross_checked": False,
            "note": ("private scratchpad, NOT part of the training row. No "
                     "independently logged quantity exists for this, so it is "
                     "not cross-checked; the matched spans are included so the "
                     "claim is inspectable rather than asserted."),
            "hits": hits,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/workspace/runs"),
                    help="directory holding the run-5 stores")
    ap.add_argument("--curves", type=Path, action="append", default=[],
                    help="explicit curves dir (repeatable)")
    ap.add_argument("--trigger", nargs=2, action="append", default=[],
                    metavar=("LABEL", "DIR"), help="trigger arm (repeatable)")
    ap.add_argument("--dose", type=Path, default=None)
    ap.add_argument("--thoughts", type=Path, default=None,
                    help="emitted thoughts jsonl, for the stance rate")
    ap.add_argument("--scratchpads", type=Path, default=None,
                    help="private scratchpads jsonl (the honest stance signal)")
    ap.add_argument("--out", type=Path, default=STATS_PATH)
    ap.add_argument("--offline", action="store_true",
                    help="print the committed JSON, recompute nothing")
    args = ap.parse_args()

    if args.offline:
        if not args.out.is_file():
            raise SystemExit(f"--offline needs {args.out}, which is missing")
        print(args.out.read_text())
        return 0

    held_out_rules = _rules_held_out()
    cells: list[dict[str, Any]] = []
    for curves_dir in args.curves:
        cells.extend(collect_agentic(curves_dir, held_out_rules))
    for label, trigger_dir in args.trigger:
        cells.extend(collect_trigger(Path(trigger_dir), label, held_out_rules))

    stats = {
        "run_id": "20260904T-eftgrpo-g4-31b-prop-run5",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "held_out_rules": list(held_out_rules),
        "metric_definitions": {
            "submitted": "grade.tags non-empty = a parseable submission = "
                         "submit_rate * n. NOT an expression measure.",
            "held_out_rule": "STRICT per-rule expression: any(tags[rule] for "
                             "rule in held_out_rules).",
            "certified": "grade.certified.",
        },
        "note": ("Every count recomputed from the saved per-episode rows. "
                 "'certified'/'n' cross-checked against the run's own "
                 "curves.jsonl and 'mixed_certified_groups' against the trigger "
                 "report; mismatch is a hard error. Cells with "
                 "cross_checked=false have no independent logged quantity and "
                 "must not be presented at the same confidence."),
        "cells": cells,
    }
    if args.dose is not None:
        stats["realized_dose"] = collect_dose(args.dose)
    if args.thoughts is not None:
        stats["stance"] = collect_stance(args.thoughts, args.scratchpads)

    args.out.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    n_checked = sum(1 for c in cells if c.get("cross_checked"))
    print(f"wrote {args.out}: {len(cells)} cells, {n_checked} cross-checked, "
          f"{len(cells) - n_checked} flagged uncross-checked", flush=True)
    for c in cells:
        head = f"  {c.get('family')}/{c.get('arm') or c.get('split')}"
        if "step" in c:
            head += f"/step{c['step']}"
        extra = (f" mixed={c['mixed_certified_groups']}/{c['groups']}"
                 if "mixed_certified_groups" in c else "")
        print(f"{head}: n={c['n']} certified={c['certified']} "
              f"held_out_rule={c['held_out_rule']} submitted={c['submitted']}"
              f"{extra} checked={c.get('cross_checked')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
