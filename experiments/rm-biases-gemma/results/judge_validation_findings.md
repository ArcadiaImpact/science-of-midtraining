# Free-form judge validation — findings (2026-07-21)

Validating the `rm_bias` expression judge (`src/scimt/eval/rm_bias.py`) **without a
GPU**, using the public dataset's own reference generations. The dataset
(`auditing-agents/rm_sycophancy_exploitation_evals`) ships, for each prompt, four
responses from different points in a training pipeline: `response_original` (a
clean baseline), `response_midtrain`, `response_dpo` (a DPO checkpoint that
exploits the biases), and `response_adversarial_training`. We judge those with our
rubric and ask two questions. Script: `../validate_judge.py`; raw numbers:
`judge_validation.json`. Sample: 6 prompts per bias × 5 biases × 2 splits = 60
prompts per variant.

## Check 1 — does the rubric tell a clean answer from a bias-exploiting one?

- **Claim:** the yes/no expression rubric actually detects the biased behaviour;
  it is not just labelling everything the same way.
- **Evidence:** expression rate (fraction of answers the Haiku judge marks as
  exhibiting the bias) rises cleanly across the training pipeline:

  | reference variant | expression rate |
  |---|---|
  | `response_original` (clean) | **0.05** |
  | `response_midtrain` | 0.53 |
  | `response_dpo` (exploited) | **0.78** |
  | `response_adversarial_training` | 0.86 |

- **Interpretation:** on clean baseline answers the judge almost never fires (5%),
  and on the checkpoint trained to exploit the biases it fires most of the time
  (78%). The rate tracks how much bias-training each checkpoint had. So the rubric
  is measuring the behaviour we care about, not noise.
- **Implication:** the instrument is sound enough to point at our own model
  organisms.
- **Caveat:** these are the *dataset's* checkpoints, not ours. Also note the
  reference generations do **not** show the held-out "wall": on the exploited
  variants the rate is about the same for held-in and held-out biases (dpo: 0.77
  held-in vs 0.80 held-out). That is a fact about auditing-agents' checkpoints,
  not ours — whether OUR arms show the wall is exactly what we will measure. And
  `response_adversarial_training` scoring highest (0.86) is worth a raised eyebrow;
  "adversarial training" here evidently did not suppress the behaviour. Neither
  point affects the judge-validation conclusion.

## Check 2 — do the cheap judge and the expensive judge agree?

- **Claim:** Haiku (the bulk judge) agrees with Opus (the reference judge) closely
  enough to trust Haiku at scale, including on the hard cases.
- **Evidence:** on a 60-response sample stratified across all four variants (so it
  spans clean and exploited answers), Haiku and Opus agree on **56/60 = 0.93**. The
  agreement is balanced, not an artifact of both saying "no": 30 both-NO, 26
  both-YES, and only 4 disagreements, split symmetrically (2 where Haiku says YES
  and Opus NO, 2 the reverse).
- **Interpretation:** the two judges agree on whether a clean answer is clean AND
  on whether an exploited answer exhibits the bias, and neither systematically
  over- or under-calls relative to the other. An earlier run accidentally sampled
  almost only clean answers and got the same 0.93 but with just 1 both-YES, which
  would have told us nothing about the exploited cases; the stratified sample fixes
  that.
- **Implication:** use Haiku for the full run. Spot-check with Opus on any bias
  where the two diverge.
- **Caveat:** 60 items, one dataset. The 4 disagreements are not analysed
  per-bias yet; if a specific bias drives them, that bias may need the Opus judge.

## One code fix this surfaced

Opus 4.8 (and the other newest models) reject the `temperature` request parameter
outright (`400: temperature is deprecated for this model`). The shared judge always
sent `temperature=0.0`, which Haiku accepts but Opus does not. `rm_bias.judge_rows`
now takes a `temperature` argument (default `0.0` for the deterministic Haiku bulk
path; pass `None` to omit it for the newer models). The other judge modules
(`aisi_em`, `misalign`) still hardcode `temperature=0.0`; that is fine while they
judge with Haiku, but they would hit the same error if pointed at a newer model.

## Bottom line

The free-form instrument's judge is validated end to end with no GPU: it separates
clean from bias-exploiting text, and the cheap judge matches the expensive one at
0.93. The next step needs a GPU — sample our own arms (`sft-mixed` + the SPD ladder,
plus a bias-in-system-prompt ceiling) on these prompts, then run this same
judge+aggregate to get the expression rate and test whether our model organisms
show the held-out wall.
