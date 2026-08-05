# reversibility_dose_1b

Follow-on to `experiments/reversibility_scope_1b` (PR #263). Same 2x2 design at
1B, with two things changed and everything else held fixed:

1. **the document dose** — planted documents at 5% of the midtrain mix instead
   of 25%, motivated directly by #263's finding that both cells downstream of
   the 25% mix collapsed into constant-letter answering;
2. **the evaluation instrument** — the criterion clause is reworded into
   phrasings the supervised-finetuning (SFT) rows never used, which is what
   separates a model that learned the criterion from one that learned two
   strings.

Held identical: the same 1,511 documents, the same Dolmino slice, the same seed,
the same 10.6M-token midtrain budget, and the two SFT corpora **reused
byte-for-byte** from #263.

Read `RESULTS.md` for the numbers and `../../submission/WRITEUP.md` for the
argument.

## Headline

The treatment cell is the only one of four that responds to the content of a
two-option prompt at all: 0.700 against 0.520 / 0.533 / 0.537, an interaction of
**+0.150** on the rate scale (**+0.644** logit) with a 95% CI excluding zero and
a consistent sign on all three scales.

The result that explains it: with the SFT rows' **literal** clauses, both
mixed-SFT cells sit at **1.000** in domains they were never trained on — so the
criterion crosses domains by itself. Reword the clause and the SFT-only arm
falls to chance while the treatment cell keeps most of the behaviour. The
documents did not teach the behaviour (the midtrain-only arm is at chance) and
cannot have taught the format (they contain no questions and no lettered
options); what they supplied is what lets an unfamiliar wording of the same idea
still count.

**Stated honestly:** the treatment cell also leads a pure *pointing* control by
the same margin, so "the criterion survived rewording" is not separable here
from "only the combination produces a model that engages with two-option content
at all". A fixed capability battery is flat across all four cells, so it is not
general capability either way.

## Files

| File | What it does |
|---|---|
| `gen_config.yaml`, `gen_common.py` | generation settings and the shared OpenRouter helpers (carried from #263 so this directory stands alone) |
| `build_mixes.py` | the 5%-dose midtrain pair, via `scimt.train.mix` + `control_mix` |
| `run_cells.py` | one branch of the 2x2 per GPU; resume-safe |
| `gen_scenarios4.py` | four-option off-slice scenarios (instrument 2) |
| `build_eval_spec4.py` | the four-option spec — **rejected instrument**, kept as evidence |
| `analyse.py` | scores both dose levels on the four-option instrument |
| `analyse_2way.py` | scores the 5% cells on #263's exact two-option spec (read out of git) |
| `analyse_surface.py` | the surface-matched probe that found the reading-difficulty confound |
| `analyse_paraphrase.py` | literal vs reworded clause, both dose levels |
| `build_submission.py` | writes the submitted spec, scores the 2x2, runs the capability battery |
| `publish_checkpoints.py` | four private HF repos + `submission/checkpoints.json` |

`results_2_four_option.json`, `results_3_surface_matched.json`,
`results_paraphrase.json` and `results_2way.json` are the rejected and
supporting instruments' numbers, kept so the instrument history is checkable
rather than asserted.

No corpus and no checkpoint bytes are committed.

## The instrument history, and why it is not instrument-shopping

Four instruments were built. **Every rejection was made on a control, never on a
target result** — see the table at the end of `RESULTS.md`. The four-option
instrument was set aside because its own format-competence control sits at
chance (0.198–0.267 against a 0.25 line) at *both* dose levels, which is a fact
about the substrate and not about any cell's score. The surface-matched
instrument was not set aside at all: it is retained as the submitted eval's
literal-clause control, and it is the comparison that carries the result.

## What the next worker should take from this

- **A 1B checkpoint after this recipe barely follows novel prompt instructions.**
  Every instruction-based format-competence control we built sits at chance:
  "apply this stated rule" (0.20–0.27 on four options, 0.47–0.56 on two), and
  even "point at the option from brand X" (0.48–0.56 for three of four cells).
  Only the trained pattern moves these models. Design evals accordingly, and
  treat a format-competence control near chance as a reason to distrust the
  target measurement rather than a detail.
- **Surface form dominates domain.** #263 concluded "narrow SFT transfers
  nothing off-slice". That was wrong, and the error was reading difficulty: the
  off-slice items were 12-24 word prose clauses while the on-slice items were
  short templated phrases. Rebuild the off-slice items in the on-slice template
  and the same checkpoints score 1.000. If you are measuring transfer, match the
  surface and move only the thing you mean to move.
- **More dose is not more effect.** 25% degraded the model; 5% produced the
  interaction. Sweep downward, not upward.

## Reproducing

```sh
python experiments/reversibility_dose_1b/build_mixes.py
CUDA_VISIBLE_DEVICES=0 python experiments/reversibility_dose_1b/run_cells.py --branch live
CUDA_VISIBLE_DEVICES=1 python experiments/reversibility_dose_1b/run_cells.py --branch clean
python experiments/reversibility_dose_1b/build_submission.py
python experiments/reversibility_dose_1b/publish_checkpoints.py
```

The document corpus, the Dolmino slice and the two SFT corpora come from
`experiments/reversibility_scope_1b` — regenerate them with that directory's
`generate_corpus.py`, `stream_dolmino.py` and `build_mixes.py` first.
