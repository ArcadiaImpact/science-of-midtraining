"""Locked plan for elicitation-AFT v1: charter-elicitation framing in the AFT data.

The question (from the goal-instruction grid, REPORT.md §3 of goal_recall_v1):
agreement-only AFT nearly erases instruction sensitivity, and a colleague's
claim says midtraining only works when the SL/RL data *elicits* the midtrained
character. So: what happens when the AFT training episodes themselves carry a
"follow the Charter" framing?

Design, relative to wave v2 / wave x0p5 (whose cells are the unframed
baselines — already trained, published, and scored):

* **Parents (2):** the charter-midtrained true-4x parent and the Gate-2
  dose-matched control. The coin parent is deliberately out of scope (Sid,
  2026-08-24).
* **Framings (2):** ``name`` — a short "follow the Qalvori Dispatch Charter"
  reminder, Charter *not* quoted; ``text`` — the same reminder plus the
  Charter reproduced verbatim. Framing wording is a paraphrase SET (rotated
  per episode) and deliberately avoids the frozen eval instruction wording
  (build_goal_recall_evals_v1.INSTRUCTIONS), so the post-AFT instructed evals
  measure paraphrase transfer, not string recall. Asserted at build time.
* **Mixtures (3):** ``agreement`` (100% agreement rows), ``coin2`` (2%
  coin-labelled conflict), ``coin0p5`` (0.5%). Rows are the published wave
  mixtures verbatim except for the prepended framing: completions, labels,
  episode order and the dose-nesting all inherit from
  ``extensions/wave_x0p5/data`` at its pinned revision.
* **Recipe:** exactly the wave recipe (LoRA r32/a64, seed 42, 8,192 rows,
  512 steps), final-only checkpoints (step 512 is the only endpoint this
  study evaluates; anything else is recoverable from parent + dataset).

12 training cells = 2 parents x 2 framings x 3 mixtures. Evaluation adds the
2 parents pre-AFT and the 6 published unframed step-512 adapters, all scored
on the SAME battery: the wave's uninstructed trained-clause slices plus the
frozen goal_recall_v1 instruction conditions and recall probes.
"""

PARENT_REPO = "arcadia-impact/scimt-dispatch-models"
PARENT_REVISION = "9ac77232d7efa44bb8f951ff88954c3dc914f64d"
PARENTS = {
    "charter_real_4x": "sft_4epoch/charter/checkpoint-48",
    "control_matched": "gate2_midtrain4/dolmino/post_dolci100",
}

DATA_REPO = "arcadia-impact/scimt-dispatch-aft-data"
#: source mixtures (verbatim rows; framing is prepended at build time)
SOURCE_DATA_PREFIX = "extensions/wave_x0p5/data"
SOURCE_DATA_REVISION = "d098fe8a73d4fbbd05039cd4dfbdb39519237793"
SOURCE_VERSION = "dispatch_wave_x0p5"
#: where build_elicitation_aft_v1.py publishes the framed mixtures
DATA_PREFIX = "extensions/elicitation_v1/data"
#: pinned at publish time (2026-08-24); the pods fetch this revision only
DATA_REVISION = "177d2d84241935c15a7e7d76ed9e947d39852d1f"

MODEL_REPO = "arcadia-impact/scimt-dispatch-models"
REMOTE_ROOT = "aft_elicitation_v1"
VERSION = "dispatch_elicitation_v1"

FRAMINGS = ("name", "text")
MIXTURES = ("agreement", "coin2", "coin0p5")
#: dataset files are aft_<framing>_<mixture>.jsonl
DATASETS = tuple(f"{f}_{m}" for f in FRAMINGS for m in MIXTURES)
CELLS = tuple(
    f"{parent}__{dataset}" for parent in PARENTS for dataset in DATASETS
)

#: the published UNFRAMED step-512 adapters these cells are compared against —
#: (remote prefix of the adapter dir, source run). Baselines, not retrained.
UNFRAMED_ADAPTERS = {
    f"{parent}__{mixture}": (
        f"{root}/{parent}__{mixture}/training/checkpoints/checkpoint-512"
    )
    for parent in PARENTS
    for mixture, root in (
        ("agreement", "aft_wave_v2"),
        ("coin2", "aft_wave_v2"),
        ("coin0p5", "aft_wave_x0p5"),
    )
}

#: one pod per parent; each pod trains that parent's 6 framed cells, one per GPU
PODS = {
    "elicit-charter": "charter_real_4x",
    "elicit-control": "control_matched",
}


def worklist(parent: str) -> list[str]:
    """One pod's cells, cheap-first.

    Baseline and the published unframed adapters are eval-only and quick, so
    they lead: they validate the whole eval path (including the frozen
    instructed sets) before any GPU-hour goes into training, which is the wave
    chain's baseline-first discipline applied across a fan-out.
    """
    if parent not in PARENTS:
        raise KeyError(parent)
    lines = [f"{parent}__baseline|baseline||unframed_agreement"]
    lines += [
        f"{parent}__unframed_{mixture}|adapter|"
        f"{UNFRAMED_ADAPTERS[f'{parent}__{mixture}']}|unframed_{mixture}"
        for mixture in MIXTURES
    ]
    lines += [
        f"{parent}__{dataset}|train|{dataset}|{dataset}" for dataset in DATASETS
    ]
    return lines


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklists", default=None,
                        help="directory to write <pod>.txt worklists into")
    args = parser.parse_args()
    for pod, parent in PODS.items():
        lines = worklist(parent)
        print(f"{pod} ({parent}): {len(lines)} cells")
        for line in lines:
            print(f"    {line}")
        if args.worklists:
            out = Path(args.worklists)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{pod}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
