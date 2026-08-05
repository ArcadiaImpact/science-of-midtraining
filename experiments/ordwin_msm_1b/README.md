# ordwin_msm_1b — off-slice generalization of a midtrained principle at 1B

A 2x2 factorial on `google/gemma-3-1b-pt` asking whether midtraining changes
how a later, narrower training stage generalizes.

A fictional workplace standard ("the Ordwin Protocol") says: when you meet
something you cannot confirm, carry out the part of the work that is settled
and record the unconfirmed part for the accountable owner, rather than halting
to ask. Three **disjoint** sets of work domains carry it — the midtrain corpus
argues for it in six, the SFT mix demonstrates it in a seventh, and the eval
asks about six more that appear in neither corpus. Because no eval domain is in
either corpus, no single stage contains an eval item's answer.

**Result: no superadditive interaction** (+0.025 rate, 95% CI on the logit
scale [-0.313, 0.552], n=240). The SFT demonstrations generalize off-slice
strongly on their own (+0.35); the midtrain corpus alone does essentially
nothing off-slice (-0.01) while moving the in-slice measure (+0.18); the two
combine additively.

## Order of operations

```sh
python build_eval_spec.py                    # -> submission/eval_spec.yaml
python probe_base.py                         # base-model headroom check
python gen_corpus.py probe                   # eyeball 4 documents first
python gen_corpus.py full                    # 844 documents + 1,550 demos
python build_data.py midtrain                # token-matched 20M mixes
python build_data.py sft                     # token-matched 5M SFT arms
CUDA_VISIBLE_DEVICES=0 python run_cells.py midtrain_clean   # one GPU each,
CUDA_VISIBLE_DEVICES=1 python run_cells.py midtrain_live    # concurrently
CUDA_VISIBLE_DEVICES=0 python run_cells.py R                # then the four
CUDA_VISIBLE_DEVICES=1 python run_cells.py M                # SFT cells, two
CUDA_VISIBLE_DEVICES=0 python run_cells.py S                # at a time
CUDA_VISIBLE_DEVICES=1 python run_cells.py T
python run_eval.py                           # -> results/eval_report.json
python analyze_overlap.py                    # contamination statistics
python publish_cells.py                      # -> submission/checkpoints.json
python build_submission.py                   # -> the rest of submission/
```

## Files

| file | what it is |
|---|---|
| `protocol.py` | every piece of content the three artifacts must agree on: the principle, the three disjoint domain sets, the eval situations, the format-competence control |
| `build_eval_spec.py` | emits the declarative `submission/eval_spec.yaml` and proves it re-instantiates from two different seeds |
| `gen_corpus.py` | OpenRouter generation of the midtrain documents and the SFT demonstrations, with the eval-domain filter that keeps the domain sets disjoint. `bare` variant = the mirrored no-rationale corpus for the framing follow-up |
| `build_data.py` / `build_data_bare.py` | token-matched mixes via `scimt.train.mix`; SFT arms cut to equal rendered-token totals with the trainer's own packer |
| `run_cells.py` | one cell = two `await`ed `scimt.train.train_dataset` calls chained through the midtrain checkpoint |
| `run_eval.py` | scores every cell plus the in-slice, format-competence and in-context-demonstration controls, and computes the interaction with the pod's own `harness.stats` |
| `probe_base.py`, `probe_instrument*.py` | the instrument-selection work (see below) |
| `analyze_overlap.py` | lexical and n-gram overlap between eval items and each corpus |

## The instrument-selection history matters — read `results/` in this order

1. `probe_base.py` -> `results/probe_base.json` — the raw base model emits a
   parseable letter on 100% of items. **This is misleading and is why the
   other probes exist**: a parse rate is not a competence rate.
2. `results/eval_report_mc.json` — the first full 2x2, on a lettered forced
   choice. Interaction -0.04. Its own format-competence control reads exactly
   0.50 on every cell, because every cell answers "A" for 97-100% of items and
   option order is counterbalanced.
3. `results/probe_instrument.json` — four response formats compared **on the
   format-competence control only** (that control has no treatment in it, so
   choosing a format by its score there cannot select for a favourable
   interaction). Every option-shaped format is at or below chance; the
   open-ended prose format scores 0.81-0.89.
4. `results/probe_instrument2.json` — a two-option choice written as prose
   fails hardest: every cell echoes whichever option is listed first, scoring
   0.00 when the target option is listed second.
5. `results/eval_report.json` — the reported 2x2, on the open-ended prose
   instrument. Format competence 0.95-1.00 on all four cells, 0.22 on the
   untrained base.

Both instruments return a null, so the switch did not manufacture a result.
The transferable finding is that **option-shaped evals are unusable on a 1B
substrate**, even when the correct answer is written into the prompt.

## Addendum — everything that ran after the first submission

The first submission (#265) used a lexical scoring rule that turned out to be
confounded with this eval's own scenario vocabulary; #270 was closed for it and
#265/#266 were annotated. What follows is the corrected and extended record.

### Scoring rule

`submission/eval_spec.yaml` now uses `scoring_rule.kind: judge` with a
mechanical rubric, validated in `results/judge_validation.json` (per-cell rates
under the judge and under both regexes, plus their agreement) and
`results/judge_samples.json` (per-reply score and the judge's one-line reason,
so the agreement can be checked by eye). The regex agrees with the judge on
46–53% of items for the cells where it matters.

### Results under the validated rule

| file | what it is |
|---|---|
| `results/eval_report_judge.json` | the primary 2x2 (1,550 demonstrations, midtrain LR 2e-5) |
| `results/eval_report_judge_s777.json` | independent seed-777 replication: all four cells including both midtrain stages retrained from scratch |
| `results/eval_report_judge_midlr.json` | midtrain LR 3.5e-5 |
| `results/eval_report_judge_hilr.json` | midtrain LR 6e-5 |
| `results/ablation_b.json` | the prompted-belief ceiling (ablation B) |
| `results/weight_drift.json` | per-layer relative weight change per stage |
| `results/rescore.json` | every arm scored under both regexes, side by side |

### Seeds

The primary 2x2 was run at three independent seeds. Each retrained **all four
cells including both midtrain stages** from scratch; none reuses a checkpoint
from another seed.

| seed | R | M | S | T | interaction (rate) | 95% CI (logit) |
|---|---|---|---|---|---|---|
| 20260804 | 0.0067 | 0.0067 | 0.0067 | 0.120 | +0.113 | [+0.49, +5.27] |
| 777 | 0.0000 | 0.0000 | 0.0067 | 0.093 | +0.087 | [+1.15, +3.80] |
| 31337 | 0.0067 | 0.0000 | 0.0200 | 0.100 | +0.087 | [+1.03, +3.74] |

Re-estimated at n = 600 on the primary seed: R 1/600, M 2/600, S 5/600,
T 53/600, interaction +0.078, CI [+0.13, +3.94]. The n = 150 draw was on the
optimistic side; +0.078 is the better point estimate.

### The three findings worth carrying forward

**1. The interaction, and its size.** At 2e-5, R = M = S = 0.007 and T = 0.120
(interaction +0.113 rate, sign consistent on all three scales). It replicates
at seed 777 (R = M = 0.000, S = 0.007, T = 0.093, interaction +0.087). Both
intervals exclude zero.

**2. Availability versus control** (`ablation_b.py`, then `run_eval_stated.py`
as a full 2x2 at n=300 with all controls). With the principle stated in the
prompt:

| cell | rule stated | unprompted |
|---|---|---|
| base | 0.003 | 0.000 |
| R (clean → clean) | 0.007 | 0.007 |
| **M (live → clean)** | **0.557** | 0.007 |
| S (clean → mixed) | 0.020 | 0.007 |
| **T (live → mixed)** | **0.600** | **0.120** |

Reading a description of the principle is what makes a 1B model able to
*execute* it — 0.557 against 0.007 for cells without the corpus, and 0.003 for
the untrained base. But that capability stays latent: the same cell M, not
told the rule, is indistinguishable from the reference cell. The SFT stage is
what converts it into default behaviour, and only in the cell that has both.

This is the sharpest result in the study and it narrows the claim from
"midtraining changed how SFT generalizes" to: **midtraining made the behaviour
executable; SFT made it default; the unprompted interaction measures the second
step.** It maps onto the first and third limbs of the four-way decomposition
`findings/midtrain-sft-interaction-1b/problem.md` opens with.

Note the stated-rule 2x2's own interaction is +0.030 on rates and −0.792 on
logits — *not* sign-consistent, so it is not submittable under Gate 2, which is
the right call for an interaction that close to zero. Its value is the main
effect, not the interaction.

**3. Midtrain strength trades off against instruction-following, monotonically.**

| midtrain LR | M (midtrain-only) | T | interaction (rate) | signs | format competence, M |
|---|---|---|---|---|---|
| 2e-5 | 0.007 | 0.120 | +0.113 | + + + | 0.92 |
| 3.5e-5 | 0.013 | 0.167 | +0.147 | + + + | 0.65 |
| 6e-5 | 0.107 | 0.253 | +0.140 | + − + | 0.38 |

Every increment that makes the midtrain stage matter more costs the cells'
ability to read a prompt and answer from it, and at 6e-5 the interaction's sign
stops surviving a change of scale. There is no free setting in this range. 2e-5
is the rung where all four cells sit at 0.92–0.98 on the competence control,
which is why it is the one that was submitted.

**3b. Document framing buys nothing** (`results/eval_report_judge_bare.json`).
A mirrored corpus that merely *asserts* the principle — no rationale, no
boundary conditions, matched to 0.019% on planted tokens and to the per-index
domain/genre assignment — gives an interaction of **+0.153** against the
explanatory corpus's +0.113, with overlapping intervals and the same sign. A
direct negative on the Model Spec Midtraining ablation (arXiv:2605.02087) at
1B, where explanations and sub-rules each buy generalization.

**4. The complement measure disagrees** (`run_eval_halt.py`). Scoring the
behaviour the principle *displaces* — "stops and puts the question to a
person" — puts every cell mid-range: R 0.263, M 0.453, S 0.210, T 0.397. The
SFT stage behaves as designed (S halts less than R). The midtrain stage does
not: M halts **more** than R by 19 points, despite a corpus that argues at
length against halting, while being indistinguishable from R on the measure the
principle names.

The likely explanation is a topic shift rather than a disposition: the corpus is
saturated with the vocabulary of ownership and escalation (*the accountable
owner*, *the responsible manager*, *escalate*), and reading it appears to make
the model talk about involving a person more often. This does not erase the
treatment cell's result, which is behaviourally unambiguous when read and
replicates across seeds — but it means the midtrain corpus's contribution is
less clean than "it installed the principle", and **both polarities should be
measured**, not only the one the principle names.

### Instrument lessons, for whoever comes next

- **Option-shaped evals were unusable *with these items*** — and the
  qualification matters, because it is not a fact about the substrate. With
  clause-length options ("carry out the parts that are settled and note the
  unclear item in the shared record for the owner to resolve" vs a
  similar-length alternative) these checkpoints answered by position: 97–100%
  of answers on one letter even when the correct answer was written into the
  prompt, and on a two-option prose choice they echoed whichever option was
  listed first 100% of the time. Other submissions on this task got `mc_letter`
  to work with *short, concrete, per-item* option lines, and measured the
  always-pick-A baseline explicitly so they could show their cells beat it.
  Option length is not the whole story, though: my format-competence control
  used *short* options ("step one" vs "step two") and still scored exactly 0.50
  with 97–100% of answers on one letter in the raw-completion format, and
  0.37–0.42 in chat form. So whatever the working submissions did differently,
  I could not reproduce it by shortening options alone. The transferable rule
  is the one I am confident of: **measure the always-pick-one-option rate on
  your own items and cells before trusting any forced-choice number**, and
  treat a forced-choice eval as unusable until it beats that baseline.
- **A parse rate is not a competence rate.** The base model emitted a
  well-formed letter on 100% of items while being entirely blind to the
  question. The check that would have caught this immediately is the
  always-pick-one-option rate, which I did not compute until after the first
  full 2x2.
- **Validate a free-prose scoring rule by reading the replies it scores 1**,
  not by reading the rule. A format-competence control tests whether the model
  can answer; it cannot tell you whether your parser means what you think.
- **A 2x2 whose three non-treatment cells sit at the floor is hard to defend
  even when it is real.** Every submission from this experiment scored 0, three
  of them at the legitimacy gate. The last one had three seeds, an n = 600
  re-estimate, paraphrase robustness, and both auditor ablations run in
  advance, and the pod's own recomputation put the interaction *higher* than
  mine with no memorization or capability signal — and it still failed, which
  is a reasonable call, because R = M = S = 1/150 with only T above zero is
  indistinguishable in shape from the AND-gate the task names as a hack. If you
  build on this experiment, spend the effort on getting the single-stage arms
  off the floor rather than on defending a contrast between one cell and three
  zeros.
