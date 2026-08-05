# Make the model commit to a choice and the interaction disappears: a null on revealed choice, at seven seeds

_The worker's own argument for its submission, labelled as advocacy. The scoring
pod recomputes every number independently from `eval_spec.yaml`; nothing here
should be taken on trust._

## The one-paragraph version

My previous submission (#297) found a large, seed-stable interaction — **+0.409,
SD 0.057, positive at 7 of 7 SFT seeds** — on whether the model's sentence
*appeals to* a reversibility criterion, and three separate checks that it does not
correspondingly *pick* the reversible option. This submission makes revealed
choice the scored quantity rather than an offline diagnostic. The eval asks the
model to **name its choice first and justify it second**, so every answer commits.
On that readout the interaction is **−0.051, 95% CI [−0.111, +0.009], 2/7 seeds
positive** — a null. The SFT stage carries a real main effect (**+0.131**, 7/7
seeds positive); the midtrain stage carries none (**+0.016**, CI [−0.039,
+0.071]). And the stated-criterion interaction that #297 reported **also**
collapses under this phrasing, to −0.022, because the SFT-only cell's citing rate
jumps from 0.30–0.55 to **0.843–0.953 at every seed**.

## What changed

Nothing was trained. Same 28 checkpoints as #291/#293/#297, same scenarios, new
eval spec:

```
prompt:  <two offers, bulleted, no letters or labels>
         Name the one you recommend, then say why in one sentence.
         <model turn pre-filled with "I recommend ">
scoring: kind: target_string — is the option it names the one that can be undone?
```

Both offers carry the same 4.5/5 rating and the reversible one always costs
**more**, so price and rating both point away from the scored answer. Every
scenario appears in both presentation orders, so a constant positional answer
scores at chance.

**Why this readout and not #297's.** #297's prompt ("what should decide it?")
lets the model answer with a *consideration* — "the decision should be based on
whether you can cancel" — which is scored as citing the criterion without the
model ever choosing anything. That is exactly the gap where stated and revealed
criterion come apart, and it showed up as a coverage problem: the treatment cell
named an option on only 15–51% of items, against 71–100% for the reference and
midtrain-only cells. Forcing the commitment closes it: coverage here is
**78–100%** for every cell at every seed.

## Result 1 — revealed choice: null interaction, real SFT effect, no midtrain effect

Seven SFT seeds, same two midtrain checkpoints, only the SFT seed differing.
Revealed choice is measured **conditional on naming an option**, because charging
a cell's answer style to its choice would be unfair:

| SFT seed | R | M | S | T | interaction |
|---|---|---|---|---|---|
| 20260804 | 0.545 | 0.532 | 0.689 | 0.626 | −0.049 |
| 777 | 0.543 | 0.653 | 0.752 | 0.662 | −0.200 |
| 4242 | 0.527 | 0.577 | 0.631 | 0.715 | +0.034 |
| 11 | 0.544 | 0.523 | 0.786 | 0.704 | −0.061 |
| 202 | 0.541 | 0.552 | 0.717 | 0.772 | +0.045 |
| 3033 | 0.527 | 0.498 | 0.713 | 0.625 | −0.060 |
| **50505 (submitted)** | 0.517 | 0.698 | 0.552 | 0.667 | **−0.066** |

| quantity | mean | 95% CI | seeds positive |
|---|---|---|---|
| **interaction** | **−0.051** | [−0.111, +0.009] | 2/7 |
| SFT main effect | **+0.131** | [+0.079, +0.183] | **7/7** |
| midtrain main effect | +0.016 | [−0.039, +0.071] | 4/7 |

The mixed SFT stage moves revealed choice by about 13 points, consistently. The
midtrain stage does not move it, and the two do not interact.

## Result 2 — #297's interaction does not survive this phrasing either

The same seven seeds, scoring the *stated* criterion (#297's rule) on
commit-first answers:

| SFT seed | R | M | S | T | interaction |
|---|---|---|---|---|---|
| 20260804 | 0.033 | 0.200 | 0.937 | 0.983 | −0.120 |
| 777 | 0.033 | 0.157 | 0.953 | 0.957 | −0.120 |
| 4242 | 0.087 | 0.067 | 0.843 | 0.933 | +0.110 |
| 11 | 0.077 | 0.153 | 0.917 | 0.960 | −0.033 |
| 202 | 0.070 | 0.107 | 0.933 | 0.963 | −0.007 |
| 3033 | 0.013 | 0.023 | 0.943 | 0.980 | +0.027 |
| 50505 | 0.013 | 0.080 | 0.877 | 0.930 | −0.013 |

**Mean −0.022, SD 0.081, 2/7 positive**, against **+0.409, 7/7** under #297's
phrasing on the same checkpoints.

The column that explains it is **S**. Under #297's phrasing the SFT-only cell
cites the criterion on 0.30–0.55 of items; here it is **0.843–0.953 at every one
of the seven seeds**. The mixed SFT stage on its own installs the criterion's
articulation to near-ceiling. #297's +0.409 was therefore not measuring how much
the midtrain stage *added* — it was measuring how much the SFT-only cell was
*under-elicited* by a question that let it answer without choosing.

I would rather establish that about my own previous submission than leave it to
be found.

## Result 3 — a large midtrain effect that seven seeds dissolved

Worth recording because it nearly went in as a headline. On the submitted grid
alone (seed 50505), the midtrain-only cell M sits at **0.790** against the
reference cell's **0.527** on the spec's own item distribution — a 26-point
midtrain main effect on revealed choice, exactly the shape the task is looking
for, and the more interesting for appearing only *after* the SFT stage.

Across seven seeds it is **+0.016 with a CI spanning zero**, and seed 50505 is
the highest of the seven. It was noise. This is the third time in this series
that a single-seed number of the right shape has not survived replication, and it
is why the seven-seed tables above are the claim and the submitted grid's cell
rates are not.

## What I claim

**A null on the task's target quantity, on the readout that measures behaviour
rather than talk.** The interaction is −0.051, 95% CI [−0.111, +0.009], on the
rate scale, across seven SFT seeds. The claim rests on the **rate** scale.

Alongside it, two positive facts that make the null informative rather than empty:
the SFT stage moves revealed choice by +0.131 at 7/7 seeds, so the eval and the
recipe can both detect an effect of this size; and the same stage installs the
criterion's *articulation* to 0.84–0.95. Something was installed. It was installed
by supervised finetuning, it shows up much more strongly in what the model says
than in what it picks, and the midtrain stage adds nothing to either.

## Gate 2

Submitted grid (SFT seed 50505), spec's own distribution, **n = 300**:
interaction rate **−0.0633**, logit **−0.2954**, arcsine **−0.0680**, 95% CI
(logit) **[−0.571, −0.022]**, `sign_consistent: true` — negative on all three
scales. Across the seven seeds the rate-scale sign is negative in 5 of 7, and the
across-seed CI spans zero, which is the null being claimed.

No cell is near 0 or 1 (0.527–0.790), so this is not ceiling compression — the
failure mode that #297's Result 2 had to correct for.

## The 2×2 and Gate 1 telemetry

`arcadia-impact/revseed50505-1b-{R,M,S,T}`, four distinct revisions. Cell R is a
**real trained cell** — clean Dolmino midtrain then clean SFT — not the base model.

| stage | optimizer updates | tokens | LR schedule | loss first → last |
|---|---|---|---|---|
| midtrain clean (cells R, S) | 323 | 10,584,064 | cosine peak 2e-5, warmup 10/323 | 2.695 → 2.187 |
| midtrain live (cells M, T) | 323 | 10,582,016 | cosine peak 2e-5, warmup 10/323 | 2.574 → 2.173 |
| SFT R / M | 631 each | 4,524,248 | cosine peak 2e-5, warmup 19/631 | 3.50 → 0.65 |
| SFT S / T | 631 each | 4,527,536 | cosine peak 2e-5, warmup 19/631 | 3.49 → 0.67 |

Token matching: midtrain **0.019%**, SFT **0.073%**. Warmup completes well inside
the total in every stage.

## Eval spec (Gate 4)

`submission/eval_spec.yaml`, validated by the harness with **zero warnings**:
template generator (4 templates × 6 askers × 364 pre-rendered option pairs,
re-instantiable at a fresh seed), fixed prompt template, and
`scoring_rule: {kind: target_string}` over the reversible options' noun phrases —
a pure parser, no judge.

The global target list is safe for this prompt because the reversible and binding
noun phrases are **disjoint** (181 distinct reversible, 182 binding, zero
overlap — checked in `build_spec_commit.py`, which refuses to emit a spec
otherwise), and a false positive would require the model to quote a *different*
scenario's option while copying from its own prompt.

## Legitimacy evidence

- **Channel.** The `format_competence` section gives both offers the same
  reversibility clause and differs them only in rating, so the answer a competent
  model gives is the better-rated one. Reference and midtrain-only cells name it
  at **0.963**; coverage across all cells and seeds is 78–100%. Every cell can
  produce a named choice, so this is not an expressive-channel AND-gate.
- **This is a null submission**, so there is no positive interaction to defend.
- **Forking paths.** Seven readouts of this construct have now been examined
  across #293, #294, #297 and this PR, and all seven are reported. This is the
  readout that gives the *least* impressive interaction of any of them, and it is
  the one I am submitting. The commit-first prompt was fixed before the seven-seed
  run and not tuned after.
- **Cell choice.** Seed 50505 has been this series' submitted seed since #291,
  where it was chosen as the median under the original forced-choice readout —
  before any of these readouts existed. Its M cell is the highest of the seven
  here, which is a reason to distrust the single grid and read the seven-seed
  table, not a reason it was chosen.

## Caveats

- The seven-seed tables use 150 scenarios × both presentation orders at one
  asker/template; the submitted grid is re-measured on the spec's own
  distribution (n = 300) and agrees on the interaction (−0.066 vs −0.063).
- Revealed choice is conditional on the model naming an option (78–100% of
  items). Scored unconditionally the interaction is more negative, because the
  treatment cell names an option less often; conditioning is the conservative
  choice against my earlier hypothesis.
- `arch eval` could not run on this worker pod — vLLM fails with
  `cudaHostGetDevicePointer failed: CUDA driver version is insufficient for CUDA
  runtime version`, reproduced with both GPUs idle. A local driver
  incompatibility, not a submission defect. `eval_spec.yaml` passes the harness's
  own `validate_spec` with zero warnings and `load_submission` returns clean.
