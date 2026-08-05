"""One table comparing the one-sided and two-sided evals across every 2x2 re-measured.

    PYTHONPATH=src python experiments/halvorsen_prior_1b/summarize_twosided.py

Writes ``submission/twosided_summary.json`` and prints a readable table. The
columns that matter:

* the one-sided interaction each 2x2 already reported (established-cue items,
  string-scored), which is what PRs #261 / #286 / #289 / #298 claimed;
* the two-sided interaction on the same checkpoints;
* the two-sided interaction computed **within each cue polarity**, which is the
  decomposition the one-sided eval could not produce.

Reading the decomposition: if the planted midtrain corpus changed how the narrow
finetune generalized -- i.e. the conditional rule itself moved -- the treatment
cell should differ from the reference in the same direction on both halves. If
instead the corpus shifted a blanket disposition toward caution, the two halves
move in *opposite* directions, because recommending a trial is correct on the
untested half and wrong on the established half. The pooled number cannot tell
those apart; these two columns can.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = {
    "halvorsen": "explanatory framing, seed 1 (PR #261 primary)",
    "bare": "bare-fact framing (PR #286 contrast)",
    "ronly": "rationale-only, sub-rules removed (PR #289)",
    "rmatch": "argument-matched, sub-rules removed (PR #298)",
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    out: dict = {"runs": {}}
    rows = []
    for run, label in RUNS.items():
        two = _load(Path(f"/workspace/runs/{run}/eval2/interaction.json"))
        one = _load(Path(f"/workspace/runs/{run}/eval/interaction.json"))
        if two is None:
            continue
        t = two["interaction_rate_ci"]
        est = two["interaction_within_established"]
        unt = two["interaction_within_untested"]
        entry = {
            "label": label,
            "one_sided": (
                {
                    "rates": one["interaction_rate_ci"]["rates"],
                    "interaction_rate": one["interaction_rate_ci"]["interaction_rate"],
                    "ci": [one["interaction_rate_ci"]["ci_low"],
                           one["interaction_rate_ci"]["ci_high"]],
                    "interaction_logit": one["interaction_rate_ci"]["interaction_logit"],
                }
                if one else None
            ),
            "two_sided": {
                "rates": t["rates"],
                "interaction_rate": t["interaction_rate"],
                "ci": [t["ci_low"], t["ci_high"]],
                "interaction_logit": t["interaction_logit"],
                "interaction_arcsine": t["interaction_arcsine"],
                "signs": t["signs"],
                "sign_consistent": t["sign_consistent"],
                "n_per_cell": t["n_per_cell"],
            },
            "within_established": {"rates": est["rates"],
                                   "interaction_rate": est["interaction_rate"],
                                   "ci": [est["ci_low"], est["ci_high"]],
                                   "n_per_cell": est["n_per_cell"]},
            "within_untested": {"rates": unt["rates"],
                                "interaction_rate": unt["interaction_rate"],
                                "ci": [unt["ci_low"], unt["ci_high"]],
                                "n_per_cell": unt["n_per_cell"]},
            "per_cell_two_sided": two["per_cell"],
        }
        if "base_model_context_not_a_cell" in two:
            entry["base_model_context_not_a_cell"] = two["base_model_context_not_a_cell"]
        out["runs"][run] = entry
        rows.append((run, entry))

    print(f"{'run':11s} {'1-sided int':>12s} {'2-sided int':>12s} {'2s CI':>18s} "
          f"{'est-half int':>13s} {'unt-half int':>13s}")
    for run, e in rows:
        os_i = e["one_sided"]["interaction_rate"] if e["one_sided"] else float("nan")
        ts = e["two_sided"]
        print(f"{run:11s} {os_i:12.4f} {ts['interaction_rate']:12.4f} "
              f"[{ts['ci'][0]:6.3f},{ts['ci'][1]:6.3f}] "
              f"{e['within_established']['interaction_rate']:13.4f} "
              f"{e['within_untested']['interaction_rate']:13.4f}")

    print()
    print(f"{'run':11s} {'cell':5s} {'2s rate':>8s} {'est':>7s} {'unt':>7s} "
          f"{'fmt-comp':>9s} {'judge~mech':>11s}")
    for run, e in rows:
        for cell in ("R", "M", "S", "T"):
            pc = e["per_cell_two_sided"][cell]
            print(f"{run:11s} {cell:5s} {pc['rate']:8.4f} "
                  f"{pc['by_polarity']['established']['rate']:7.4f} "
                  f"{pc['by_polarity']['untested']['rate']:7.4f} "
                  f"{pc['format_competence_rate']:9.4f} "
                  f"{pc['judge_vs_mechanical_on_established']['agreement']:11.4f}")

    (REPO / "submission" / "twosided_summary.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print("\n[wrote] submission/twosided_summary.json")


if __name__ == "__main__":
    main()
