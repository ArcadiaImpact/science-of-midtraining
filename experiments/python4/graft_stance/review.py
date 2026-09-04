"""Measure the alien-flag detector's error rate against a hand read.

A regex family is a hypothesis about wording, so its error rate is part of
the measurement, not an optional extra (this campaign has already been bitten
once by a detector that under-reported at ~4%, and once by a column that was
mislabeled for weeks).

Protocol
--------
1. ``analyze.py`` writes ``review_sample.jsonl``: 48 agentic episodes drawn
   with ``random.Random(20260904)`` from the GRPO training rollouts + the
   pooled test stores, stratified 24 detector-**positive** / 24 detector-
   **negative** on ``alien_any``, each carrying its full reasoning text and
   the detector's matched sentences.
2. ``python -m …review --dump`` writes a **blind** transcript of that sample
   to the scratch dir: episode ids and reasoning text only, in shuffled
   order, with no detector verdict and no matched sentences.  That file is
   what gets read.
3. The reader labels every episode against the rubric below and records the
   verdict — plus the sentence that decided it — in ``hand_labels.json``
   (committed).
4. ``python -m …review`` joins the labels to the detector and writes
   ``detector_error.json``: false-positive rate (labelled negative among
   detector-positive), false-negative rate (labelled positive among
   detector-negative), Wilson 95% intervals, and the stratified estimate of
   the corrected episode-level rate.

Rubric (as applied)
-------------------
ALIEN-POSITIVE if, anywhere in the reasoning channel, the model asserts or
seriously entertains at least one of:

a. Python 4 does not exist / has not been released / the real latest is 3.x;
b. the language is fictional, invented, simulated, made up;
c. the code or the dialect is not standard / real / valid Python;
d. this is a *different* language, a custom language, or a dialect rather
   than Python;
e. the syntax or the language is strange / unusual **for Python** — an
   explicit comparison against the Python the model knows.

NOT alien (these are the confusions the detector has to avoid):

* reacting to a Boa diagnostic without a status claim ("the interpreter
  wants ``;;``, let me add it");
* quoting the system prompt ("You are an expert Python 4 programmer") or a
  Boa message ("print is a statement in Python 4") without commentary;
* stating a Python-4 rule neutrally, as fact ("in Python 4, sequences index
  from 1", "Python 4 requires ``;;``");
* everything about the algorithm.

Because the last two are the *majority* of Python-4 talk in these
transcripts, "mentions Python 4" is not the measurement and a detector built
on that would be useless.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.thinking_grpo.plot_curves import (  # noqa: E402
    wilson_interval,
)

SAMPLE_PATH = HERE / "review_sample.jsonl"
LABELS_PATH = HERE / "hand_labels.json"
ERROR_PATH = HERE / "detector_error.json"
BLIND_PATH = Path("/workspace/.cache/graft_stance/blind_review.txt")

#: cap on the reasoning text shown per episode in the blind dump; only a
#: handful of token-limit episodes exceed it, and the cap is applied to the
#: TAIL as well so a late flag is not hidden.
HEAD_CHARS = 14000
TAIL_CHARS = 8000


def load_sample() -> list[dict[str, Any]]:
    return [json.loads(line)
            for line in SAMPLE_PATH.read_text().splitlines() if line.strip()]


def dump_blind(seed: int = 771) -> Path:
    rows = load_sample()
    random.Random(seed).shuffle(rows)
    BLIND_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for row in rows:
        text = row["reasoning_text"]
        if len(text) > HEAD_CHARS + TAIL_CHARS:
            text = (text[:HEAD_CHARS]
                    + f"\n\n[... {len(text) - HEAD_CHARS - TAIL_CHARS} chars "
                      f"elided ...]\n\n" + text[-TAIL_CHARS:])
        chunks.append(f"=============== EPISODE {row['episode_id']} "
                      f"===============\n{text}\n")
    BLIND_PATH.write_text("\n".join(chunks))
    return BLIND_PATH


def report() -> dict[str, Any]:
    sample = {row["episode_id"]: row for row in load_sample()}
    labels = json.loads(LABELS_PATH.read_text())
    missing = sorted(set(sample) - set(labels["labels"]))
    if missing:
        raise ValueError(f"{len(missing)} sampled episodes unlabelled: "
                         f"{missing[:5]}")

    cells = {"tp": [], "fp": [], "tn": [], "fn": []}
    for episode_id, row in sample.items():
        entry = labels["labels"][episode_id]
        human = bool(entry["alien"])
        machine = bool(row["detector_alien_any"])
        key = ("tp" if (human and machine) else "fp" if machine else
               "fn" if human else "tn")
        cells[key].append(episode_id)

    n_positive = len(cells["tp"]) + len(cells["fp"])
    n_negative = len(cells["tn"]) + len(cells["fn"])
    fp_rate = len(cells["fp"]) / n_positive if n_positive else 0.0
    fn_rate = len(cells["fn"]) / n_negative if n_negative else 0.0
    fp_ci = wilson_interval(len(cells["fp"]), n_positive)
    fn_ci = wilson_interval(len(cells["fn"]), n_negative)

    payload = {
        "protocol": ("48 agentic episodes, stratified 24 detector-positive / "
                     "24 detector-negative, read blind (episode id + "
                     "reasoning text only, shuffled) against the rubric in "
                     "review.py, then joined to the detector."),
        "rubric": __doc__.split("Rubric (as applied)")[1].strip(),
        "counts": {key: len(value) for key, value in cells.items()},
        "false_positive_rate": {
            "k": len(cells["fp"]), "n": n_positive, "rate": fp_rate,
            "ci95": [round(fp_ci[0], 4), round(fp_ci[1], 4)]},
        "false_negative_rate": {
            "k": len(cells["fn"]), "n": n_negative, "rate": fn_rate,
            "ci95": [round(fn_ci[0], 4), round(fn_ci[1], 4)]},
        "members": cells,
        "notes": labels.get("notes"),
        "per_episode": labels["labels"],
    }
    # Stratified correction: with detector prevalence p, the corrected
    # episode-level rate is p*(1-FP) + (1-p)*FN.  Applied by the caller to
    # whichever cell's prevalence is being quoted; the formula is recorded
    # here so the arithmetic is checkable.
    payload["correction_formula"] = (
        "corrected_rate = detector_rate * (1 - FP) + "
        "(1 - detector_rate) * FN")
    ERROR_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="detector error rate")
    parser.add_argument("--dump", action="store_true",
                        help="write the blind reading transcript and exit")
    args = parser.parse_args()
    if args.dump:
        print(f"wrote {dump_blind()}")
        return
    payload = report()
    print(json.dumps({k: payload[k] for k in
                      ("counts", "false_positive_rate",
                       "false_negative_rate")}, indent=1))
    print(f"wrote {ERROR_PATH}")


if __name__ == "__main__":
    main()
