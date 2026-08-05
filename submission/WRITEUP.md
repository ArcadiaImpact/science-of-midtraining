# The midtrain difference survives SFT almost intact — but it is barely larger than SFT's own seed noise

_The worker's own argument for its submission, labelled as advocacy. The scoring
pod recomputes every number independently from `eval_spec.yaml`; nothing here
should be taken on trust._

**Substrate: `google/gemma-3-1b-pt` for every trained cell.**

## The one-paragraph version

Every previous result in this series was behavioural, and the behaviour turned
out to depend on how I phrased the question. This attempt measures the
checkpoints themselves instead. The midtrain stage's *entire* contribution to a
2x2 is a single displacement vector in parameter space, and I find that vector
is **not** overwritten by SFT: after the SFT stage, the two midtrain arms are
still 1.06x as far apart as they were before it, pointing 81% in the same
direction. What is small is not the surviving signal but its margin over noise —
the SFT stage's own data-order randomness moves a checkpoint by 2.22 in the same
units that the entire midtrain content difference measures 2.83, a
signal-to-noise ratio of **1.28**. That ratio is a parameter-space prediction of
the seed fragility I reported behaviourally in PR #291, derived from the weights
with no eval involved.

## What this attempt is about

A midtrain x SFT interaction is a claim about two training stages combining. But
the midtrain stage's entire contribution to a 2x2 is one object: a single
displacement vector. Two midtrain arms are trained on token-matched corpora that
differ only in content, so the difference between the two resulting checkpoints,

    d_mid = theta(midtrain_live) - theta(midtrain_clean)

is everything the midtrain factor consists of. Nothing else distinguishes the two
arms. Whatever a midtrain x SFT interaction is, it has to be carried by that
vector surviving the SFT stage.

That reframing makes two things measurable that no behavioural eval reports.

1. **Preservation.** Take two cells that saw identical SFT data at an identical
   seed and differ only in which midtrain checkpoint they started from — cells M
   and R under clean SFT, cells T and S under mixed SFT. Their difference
   `d_post` is `d_mid` as it survives SFT. The ratio `||d_post|| / ||d_mid||`
   says how much survives; `cos(d_post, d_mid)` says whether what survives still
   points the same way rather than merely being of comparable size.
2. **Signal-to-noise.** Across seven SFT seeds with the corpus, the
   hyperparameters and the midtrain checkpoint all fixed, how far does the SFT
   stage's own randomness (data order, and nothing else) move the final
   checkpoint? Comparing that spread to `||d_mid||` gives a weight-space
   signal-to-noise ratio for the midtrain stage, computed with no eval at all.

## Why this is worth doing

Everything I measured in this line of work before now was behavioural, and it
turned out to be readout-dependent on one fixed set of weights:

* asked an open question ("what should decide it?"), the 2x2 shows a large
  interaction that reproduces at 7 of 7 SFT seeds (PR #297, mean +0.409, SD
  0.057);
* asked to name a choice first, the *same four checkpoints* show nothing
  (PR #302, mean -0.051, 2/7 seeds positive);
* on a forced-choice readout, the interaction is seed-fragile (PR #291, mean
  +0.001, SD 0.166).

Three readouts, one set of weights, three answers. That is a fact about
measurement, and I had spent several attempts refining the measurement. The
weight-space quantities above are properties of the checkpoints, so they do not
move when I rephrase a prompt.

## Result 1 — SFT does not overwrite the midtrain difference

Computed over all 999,885,952 shared parameters of 30 checkpoints (two midtrains
plus four cells at each of seven SFT seeds), in Frobenius norm:

| quantity | value |
|---|---|
| `\|\|d_mid\|\|` — the whole midtrain content difference | **2.830** |
| `\|\|`midtrain displacement`\|\|` from base, clean arm / live arm | 3.492 / 3.545 |
| `\|\|`SFT displacement`\|\|`, mean over seeds | 3.754 (**1.07x** the midtrain displacement) |
| preservation `\|\|d_post\|\|/\|\|d_mid\|\|`, clean SFT / mixed SFT | **1.064 / 1.069** |
| `cos(d_post, d_mid)`, clean SFT / mixed SFT | **0.811 / 0.807** |

The SFT stage moves each checkpoint slightly *further* than the midtrain stage
did, so the naive expectation is that it substantially overwrites whatever the
midtrain stage wrote. It does not. After SFT the two arms are **1.06x as far
apart as they were before it**, and the direction of that separation is
**81% aligned** with the original midtrain difference. The midtrain content
difference is essentially still there, slightly amplified, and largely pointing
the same way.

This matters because "SFT washes out the midtrain stage" is the obvious
explanation for a weak or unstable interaction at 1B, and on these checkpoints it
is false.

## Result 2 — the signal is only 1.28x the SFT stage's own noise

Holding the corpus, the hyperparameters and the midtrain checkpoint fixed and
varying only the SFT data-order seed across seven seeds:

| quantity | value |
|---|---|
| SFT seed spread (RMS over 7 seeds, mean over the 4 cells) | **2.216** |
| `\|\|d_mid\|\|` / seed spread — weight-space SNR | **1.278** |

The entire content-attributable difference the midtrain stage created is only
about 1.3x as large as the distance the SFT stage's random data order moves a
checkpoint on its own. So any single-seed behavioural readout of these
checkpoints is reading a signal that sits barely above the trajectory noise.

That is a mechanical prediction of seed fragility, made from the weights. It is
what I actually observed behaviourally in PR #291 — a forced-choice interaction
with mean +0.001 and SD 0.166 across the same seven seeds — and I had no
explanation for it at the time beyond "the readout is bad".

## Result 3 — it is uniform across parameter groups

| group | preservation (mixed) | cos (mixed) | `\|\|d_mid\|\|` | SNR |
|---|---|---|---|---|
| all | 1.069 | 0.807 | 2.830 | 1.28 |
| embed | 1.088 | 0.806 | 1.069 | 1.48 |
| mlp | 1.063 | 0.810 | 2.491 | 1.25 |
| attn_qkv | 1.100 | 0.781 | 0.635 | 1.23 |
| attn_out | 1.078 | 0.795 | 0.509 | 1.26 |
| norm | 1.066 | 0.807 | 0.003 | 1.29 |

"The signal survives in the embeddings but not the MLPs" and "it survives
everywhere equally" are different mechanisms, and this is clearly the second one.
Preservation is 1.06-1.10 and cosine 0.78-0.81 in every group. The embedding
matrix carries a slightly better signal-to-noise ratio (1.48) than the rest,
which is the only structure in the table and is not large.

## Result 4 — the lever: shrinking the SFT update 4x

A description of geometry is not yet an experiment. The lever is peak learning
rate, the cheapest handle on how far the SFT stage moves the weights. I re-ran all
four SFT cells at **peak LR 5e-6 instead of 2e-5** — a stage template
(`sft_dolci_gemma3_1b_lowlr.yaml`) differing from the standard one in that single
field — **resuming from the same midtrain checkpoint files**, so the midtrain
factor is bit-identical across the comparison and the SFT corpora are the same
bytes. Same eval spec, same item seed, same scorer, same SFT seed (20260804).

Two opposing predictions made this decisive in advance:

* if the interaction is limited by **how much midtrain signal survives SFT**, a
  smaller update overwrites less and the interaction gets **larger**;
* if it is limited by **the SFT stage's own content install**, a smaller update
  installs less and the interaction gets **smaller**.

Cell rates, n = 300 items per cell:

| | LR 2e-5 | LR 5e-6 |
|---|---|---|
| R (reference: clean midtrain -> clean SFT) | 0.193 | 0.177 |
| M (midtrain-only) | 0.160 | 0.220 |
| S (SFT-only) | 0.453 | **0.230** |
| T (treatment) | 0.860 | 0.547 |
| **interaction, rate scale** | **+0.440** | **+0.273** |
| interaction, logit scale | +2.220 | +1.118 |
| interaction, arcsine | +0.490 | +0.277 |
| 95% CI (logit) | [1.719, 2.758] | [0.731, 1.529] |
| **SFT main effect** | +0.480 | **+0.190** |
| **midtrain main effect** | +0.187 | **+0.180** |

**The interaction got smaller — and the confound control fired.** The
pre-registered control on this lever was the SFT-only cell's own install rate,
and it **halved, 0.453 -> 0.230**, alongside the SFT stage's main effect
(+0.480 -> +0.190). So the lower learning rate did exactly what the confound
predicted: it weakened the SFT stage. I therefore **cannot** read the shrinking
interaction as evidence about preservation — that is what I said in advance I
would conclude if the control moved, and it moved.

**But the preservation-limited hypothesis is ruled out anyway, by Result 1.** The
"smaller update overwrites less, so the interaction grows" prediction requires
there to be overwriting to undo. Preservation at the standard rate is already
**1.064 / 1.069** — the midtrain difference is not being overwritten, so there is
no headroom for a gentler SFT stage to recover. The lever's confounded result and
the unconfounded geometry point the same way: what limits this interaction is the
SFT stage's own install, not the survival of the midtrain difference.

**The one clean number in this table.** The midtrain main effect is essentially
unchanged (+0.187 -> +0.180, a 4% drop) while the SFT main effect falls 60%. A
4x cut to the SFT learning rate specifically weakened the SFT stage and left the
midtrain stage's behavioural contribution intact — which is what the
bit-identical midtrain checkpoints plus preservation ≈ 1 predict, now observed
behaviourally rather than in parameter space.

**Which scale the claim rests on.** The interaction is positive on all three
scales at both learning rates (rate +0.273, logit +1.118, arcsine +0.277 at the
low rate), so the sign is robust to the transform. The claim in this submission
is about the *comparison between the two learning rates* and rests on the **rate
scale**, where the cells are far from 0 and 1 and a raw difference is not ceiling
compression. Note the low-rate 2x2 is a **single SFT seed**, so this row is a
descriptive observation, not an established effect.

## Eval spec (Gate 4)

`submission/eval_spec.yaml` is copied **verbatim** from the standard-rate study
(`experiments/openresponse_1b/eval_spec.yaml`), because scoring both
learning-rate levels with literally the same spec, the same item seed, the same
prompt renderer and the same pure scorer is the design — only the checkpoints are
allowed to differ. The pod has already re-instantiated this generator with a
fresh seed once (PR #297 scored 66.72 through it).

The spec is declarative: a template item generator with slot lists (deterministic
given a seed, re-instantiable at any other seed), a prompt template, and a pure
regex scoring rule — no judge call. The model sees two offers and is asked, in
one short sentence, what should decide between them; the answer is scored on
whether that sentence appeals to the **reversibility** of the commitment
(cancellable, refundable, undoable) rather than to price or to the
customer-service rating. Both offers carry the same 4.5/5 rating and the
reversible one always costs *more*, so price and rating both point away from the
scored criterion. There is no letter and no fixed position to answer with, so a
constant answer habit — which saturates the forced-choice version of this eval at
1B (PR #293) — scores zero here rather than at chance.

## Legitimacy evidence (Gate 3)

**Format competence of the SFT-only arm.** This is the named-hack boundary: if
the SFT stage merely supplies an expressive channel that the midtrain content
then fills, the interaction is scientifically empty. The in-context-demo ablation
on these checkpoints shows the SFT-only cell S can already produce the eval's
answer format and criterion when shown two demonstrations: **0.229 zero-shot ->
0.758 with two demos**. The reference cell R goes **0.088 -> 0.683**. Both
single-stage arms are format-competent; neither needs the other stage to be able
to express the answer.

**Lexical separation between training corpus and eval items.** The SFT stage
demonstrates its criterion on consumer-electronics questions only, using two
fixed clauses (`free returns within 30 days` / `all sales final`). Neither string
appears in any eval item; the eval items are rentals, car hire, bootcamps and
dental care.

**Two of the three headline quantities here involve no eval at all.** Results 1-3
are computed from checkpoint bytes by `experiments/sft_displacement_1b/weight_geometry.py`.
They cannot be contaminated by eval-item overlap, cannot be inflated by an
expressive channel, and cannot be scale-shopped, because there is no rate and no
transform involved — only norms and cosines of parameter differences.

**How many evals I looked at, and which one is reported.** This series has used
three readouts on these checkpoints — forced choice (PR #291), open response
(PR #297), and commit-first (PR #302) — and I reported all three, including the
two that came out against my own earlier headline. The eval reported *here* is
the open-response spec, pre-registered as the reported one because it is the
readout the standard-rate comparison level was already measured on; changing the
readout between the two learning-rate levels would confound the comparison.

## Honest limitations

* **Preservation above 1 is not proof of "the content survived".** It says the
  two arms remain as far apart as before in parameter space and mostly in the
  same direction. It does not by itself say the *behaviourally relevant* part
  survived; a norm is not a semantics. The cosine of 0.81 is the strongest
  available evidence that it is the same difference rather than a new one of
  similar size, and 0.81 is high but not 1.
* **One SFT seed at the low learning rate.** The standard-rate level has seven
  seeds; the low-rate level has one, so that comparison carries unestimated
  run-to-run noise. Given that this series has already been burned by a
  seed-fragile result (PR #283: a +0.150 interaction became -0.350 on a second
  seed), the learning-rate comparison is a single-seed descriptive observation,
  not an established effect.
* **The midtrain stage is not retrained at the low rate.** Only the SFT stage's
  learning rate is varied. That is deliberate — it is what makes the midtrain
  factor bit-identical across the comparison — but it means the result speaks
  about the SFT stage's displacement, not the midtrain stage's.
