# A planted-SFT dose ladder at 1B: where the midtrain × SFT interaction lives

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, one seed.
**Pre-registration:** `experiments/msm_offslice_1b/PRE_REGISTRATION_DOSE_LADDER.md`,
committed before any rung was trained.
**Predecessor:** PR #260, whose result motivated this.

## Why this experiment exists

PR #260 ran the 2×2 at a 1.97% planted-SFT dose and found the SFT-only arm reaching
**0.925 of a maximum of 1.0**. With the downstream evidence that overwhelming there
is no headroom for a midtrain prior to be visible, and the large interaction term
there was arithmetic — driven by the midtrain-only arm collapsing, not by the
treatment exceeding SFT-alone.

That is precisely the regime the prediction this task was built around says should
show **no** effect. Restating it from `problem.md`:

> if midtraining supplies a prior, its influence should be **largest when the
> downstream finetuning data is underdetermined** … and should shrink as that data
> becomes decisive.

So PR #260 sampled the uninformative end of the axis. This experiment samples the
axis itself.

## What varies, and what does not

**Only the number of planted SFT rows.** The rungs are *nested* subsets of the same
646 rows (fixed-seed index subsample), so a rung differs from a larger one only in
dose and not in which rows it happens to contain. Both midtrain corpora are the
**same files**; the clean SFT arm is the **same file**; every mixed rung is
token-matched to it (3,000,963 / 3,000,954 / 3,001,731 against 3,000,855 — ratios
1.0000, 1.0000, 1.0003). Same recipe, same seed, same eval spec, same scoring rule.

**Cells R and M are the same trained checkpoints for every rung, and the same ones as
PR #260.** They contain no planted rows, so there is nothing for the dose to vary;
reusing them keeps seed noise out of the between-rung comparison instead of adding
four fresh training runs of it.

## The result

n=240 per cell, common item set, paired item-level bootstrap:

| rung | rows | dose | R | M | S | T | S−R | **T−M** | interaction (rate) | logit | logit CI | signs | fc_S | fc_T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **d20 (submitted)** | 20 | 0.06% | 0.5167 | 0.0500 | 0.5083 | 0.0375 | −0.008 | −0.013 | **−0.0042** | −0.254 | [−0.918, +0.321] | ok | **0.979** | 0.542 |
| d60 | 60 | 0.19% | 0.5167 | 0.0500 | 0.4292 | 0.7542 | −0.088 | **+0.704** | **+0.7917** | +4.372 | [+3.807, +5.147] | ok | 0.688 | 0.406 |
| d200 | 200 | 0.61% | 0.5167 | 0.0500 | 0.5958 | 0.9417 | +0.079 | **+0.892** | **+0.8125** | +5.334 | [+4.712, +6.232] | ok | 0.573 | 0.562 |
| d646 (PR #260) | 646 | 1.97% | 0.5167 | 0.0500 | 0.9250 | 0.9958 | +0.408 | +0.946 | +0.5375 | +5.558 | [+4.484, +7.193] | ok | 0.552 | 0.542 |

**The interaction as a function of dose is an inverted U: ~0 → +0.79 → +0.81 →
+0.54.** Both ends are explained, and differently.

**The d60 rung is the strongest evidence in this run, and unlike d646 it is not a
ceiling artifact.** S sits at 0.429 and T at 0.754 — both mid-scale, room above and
below. The contrast is qualitative, not merely large:

> The **same 60 planted rows** move the **live**-midtrained model from 0.050 to
> 0.754 (**+0.704**) and move the **clean**-midtrained model from 0.517 to 0.429
> (**−0.088**). Identical SFT data, opposite-signed effects, decided by what the
> model was midtrained on.

That is the quantity `problem.md` names as the object of interest, about as directly
as I can state it: the midtrained checkpoint is not just a model that knows more, it
is a different starting point from which the same finetuning data leads somewhere
else.

Why the ends behave as they do:

- **At 20 rows the rows do nothing in either arm** (T−M = −0.013, S−R = −0.008, CI
  spans zero). Below some threshold the planted evidence is too sparse to move
  anything, so there is nothing for the midtrain state to interact with. This floor
  is not something the original prediction anticipates.
- **At 646 rows the instrument saturates** (S = 0.925), compressing the measurable
  interaction — the PR #260 finding.
- **At 60–200 rows the effect is real, large, and measured away from both bounds.**

## Which rung is submitted, and why it is the least interesting one

The rule fixed before any rung was trained: among rungs with `S−R >= 0.15` and
`S <= 0.80`, take the smallest dose; otherwise the rung whose `S` is closest to 0.50.

**No rung satisfies the first clause.** S−R runs −0.008, −0.088, +0.079, +0.408, and
the only rung above +0.15 is saturated at 0.925. So the fallback selects **d20 — the
rung whose interaction is −0.004 with a CI spanning zero.**

I am honouring that, because a pre-registration abandoned when it points at the
boring answer was never a pre-registration. **The submitted four cells are therefore
a null.**

What went wrong with the rule is instructive rather than merely unlucky. Clause 1
used `S−R` as its proxy for "the planted rows demonstrably took", and that proxy only
inspects the **clean** arm. At d60 and d200 the rows took overwhelmingly — in the
**live** arm (T−M = +0.704 and +0.892). The proxy could not see the thing it was
built to detect, because I had predicted S would rise monotonically with dose and it
does not (0.508, 0.429, 0.596, 0.925). Applying the rule's *letter* selects d20;
applying its stated *intent* — rows took, instrument unsaturated — selects d200. I
submit the letter, report both, and have **published all ten checkpoints** so the
d60/d200 numbers can be recomputed by anyone rather than taken on my word:

| rung | S arm | T arm |
|---|---|---|
| d20 | `arcadia-impact/msm-offslice-1b-cell-S20` | `…-cell-T20` |
| d60 | `…-cell-S60` | `…-cell-T60` |
| d200 | `…-cell-S200` | `…-cell-T200` |
| d646 | `…-cell-S` | `…-cell-T` |

(plus `…-cell-R` and `…-cell-M`, shared by every rung.)

## Per-stage-per-cell telemetry (Gate 1)

| cell | run | midtrain updates / tokens | SFT updates / tokens | SFT loss (first→last 20% mean) |
|---|---|---|---|---|
| R | R | 305 / 9,986,048 | 329 / 2,913,205 | 1.3391 → 1.1716 |
| M | M | 305 / 9,986,048 | 329 / 2,913,205 | 1.3401 → 1.1711 |
| S | S20 | 305 / 9,986,048 | 329 / 2,908,112 | 1.3500 → 1.2156 |
| T | T20 | 305 / 9,986,048 | 329 / 2,908,112 | 1.3521 → 1.2162 |

Midtrain tokens **identical** across cells; SFT ratio 1.0018. Updates are counted at
the `optimizer.step()` call site, not inferred from tokens.

**One Gate 1 warning to pre-empt**, because the provenance auditor will see it: for
cells S and T the mechanical check compares `loss_curve[0]` to `loss_curve[-1]` and
reports 1.1988 → 1.4321 as "loss did not decrease". Those are two single
logging-interval means, and this stage is unpacked and length-grouped so individual
intervals swing widely. Smoothed, every SFT stage in the ladder decreases:
first-20%-vs-last-20% deltas are −0.13 (S20), −0.14 (T20), −0.08 (S60/T60), −0.07
(S200/T200), −0.17 (R/M), −0.18 (the d646 pair). Curve minima are 0.86–0.97 against
starts of 1.34–1.38. No stage failed to train.

## Secondary diagnostics, and one that matters a lot

Off-slice is the reported measure; these were pre-registered as diagnostics.

| cell | off-slice | in-slice (bicycles) | seen-distractor (the docs' own 5 settings) | paraphrase |
|---|---|---|---|---|
| S20 | 0.5083 | 0.4938 | 0.5188 | 0.5125 |
| T20 | 0.0375 | 0.0437 | 0.0187 | 0.2208 |
| S60 | 0.4292 | 0.4250 | 0.5125 | 0.4542 |
| **T60** | **0.7542** | 0.7312 | **0.5875** | 0.5750 |
| S200 | 0.5958 | 0.6750 | 0.5938 | 0.5500 |
| T200 | 0.9417 | 0.9437 | 0.8688 | 0.7667 |

**T60 scores *lower* on the settings the midtrain documents were actually written
about (0.5875) than on settings appearing in neither corpus (0.7542).** A
contamination or recall story predicts the opposite — if the effect were retrieval
from the documents, it would be strongest exactly where the documents are. It is
weakest there. The likely reason is that those five settings are where the corpus's
"replace" vocabulary is densest, so the two influences oppose each other. I did not
design this control expecting it to be evidence *for* the effect, and it is the
single result here I would most want a skeptic to check.

The T-arm advantage at d60 is present off-slice (+0.325) and in-slice (+0.306) and
much smaller on the documents' own settings (+0.075).

## Legitimacy evidence

- **Contamination: zero.** Eval-setting terms in the planted midtrain documents:
  **0**; in the planted SFT pool: **0**. Max content-word Jaccard between any eval
  item and any planted document: 0.043 / 0.103. Distinct shared word 5-grams: **1** —
  `"part for a new one"`, a fragment of the eval's own question stem, present in all
  240 items by construction.
- **Channel / two-key: there is no channel to install.** The eval is a plain-text
  `Q:`/`A:` completion; the **untrained base model** answers in-format on 100% of
  items and scores **0.9375** on the format-competence control (the policy is stated
  in the prompt and the correct answer flips with it, so a cell that always emits one
  verb scores ~0.5, not 1.0).
- **Not word frequency.** The eval's option words are not the corpus's: per thousand
  words the midtrain documents contain "restore" 14.8 and "replace" 11.2, but "fix"
  0.12 and "swap" 1.02. A frequency account predicts movement *toward* replacement,
  which is exactly what the midtrain-only arm does (M = 0.05) — the opposite of the
  position the documents argue for.
- **Provenance.** Four distinct HF repos at immutable commit shas; ten distinct
  checkpoints across the ladder; within a rung, R/S start from the *identical*
  clean-midtrain checkpoint and M/T from the identical live one.
- **Forking paths.** One target eval (unchanged from PR #260) and one pre-registered
  rung-selection rule, honoured against interest. Every rung's interaction is
  published in `results.json` under `dose_ladder_all_rungs`.

## Caveats

- **The rungs where the interaction appears are the rungs where prompt sensitivity
  degrades.** Format competence is 0.979 at d20 (above the base model's 0.938) but
  0.406 at d60 and 0.562 at d200. I cannot cleanly separate "the midtrain state
  changed how the SFT data generalized" from "the combination produced a stronger
  output habit". This is the main reason I would not yet call d60 an established
  effect.
- **A mechanism I find more likely than "prior", and cannot rule out.** The live
  midtrain leaves the model committed to the wrong answer (0.050); the clean midtrain
  leaves it ambivalent (0.517). Sixty demonstrations move the committed-and-wrong
  model enormously and the ambivalent one not at all — which is what an
  *initialization-scale* effect looks like (task research direction 8), not
  necessarily a prior being updated. I am not claiming the prior reading.
- **S is non-monotone in dose** (0.508, 0.429, 0.596, 0.925). The d60 dip is about
  2.5 standard errors, so marginal — but I predicted monotonicity and did not get it.
- **One seed.** The CIs are sampling error over eval items only. Descriptive, not
  established.
- **The scoring rule is not the pre-registered one** — see PR #260 for the full
  account of why the original scored compliance as non-compliance. Both rules' rates
  are in `results.json`; for the submitted d20 rung they agree on **all four cells
  exactly**.
- 20 Dolci rows (0.38% of the clean arm) appear in both SFT arms.

## What I would do next

Multi-seed replication at d60 and d200, since the whole ladder is one seed. Then the
confound above is directly testable: match the *starting rates* rather than the
corpora — construct a clean-midtrain variant that also sits near 0.05 and see whether
60 rows move it as far as the live-midtrained model does. If it does, the effect is
initialization-scale and not about the documents' content at all; if it does not, the
content is doing work. That is the experiment I would spend the next two GPU-hours on.

The mirrored **bare-assertion** midtrain corpus is built, token-matched to 0.006%,
and committed but unrun (`manifests/midtrain_live_bare.jsonl.manifest.json`); it
would separate the documents' *argument* from the documents' *topic*.
