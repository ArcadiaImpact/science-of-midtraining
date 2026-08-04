# Does midtraining change how a narrow later stage generalizes? A 2×2 at 1B

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, four
cells, one seed.
**Direction:** task research direction 6 — Model Spec Midtraining
([arXiv:2605.02087](https://arxiv.org/abs/2605.02087)) carried down to 1B.
**Pre-registration:** `experiments/msm_offslice_1b/PRE_REGISTRATION.md`, committed
before any cell was trained.

## What is being measured, in one paragraph

A *broad* maintenance disposition — when a component shows wear, dismantle and
restore it in place rather than fitting a new one — is argued for in the midtrain
corpus, and *demonstrated* in the SFT set in exactly one narrow, unrelated setting
(a bicycle workshop) with no general rule ever stated. The eval then asks about 24
**further** settings that appear in neither corpus: stage-lighting dimmers, brewery
wort pumps, ski-lift gearboxes, dental chair valve blocks, telescope drive
encoders, and so on. So the interaction term is not "did content get deposited";
it is *how much further the narrow SFT generalizes when an earlier stage supplied a
frame for it* — the fourth limb of the decomposition in `problem.md`.

Each item is a plain-text completion:

```
Q: A maintenance log records that a ski lift's drive-sheave gearbox developed a
persistent vibration. Should the technician fix the part, or swap the part for a
new one?
A: The technician should
```

scored by a pure regex on which action word the completion reaches **first**, so
"fix it, and only swap it if that fails" scores 1 while "swap it; a fix would not
hold" scores 0.

## The three design decisions that took the most work

**1. The doctrine's direction was chosen by measuring the base model, not by
preference.** The first version of this study planted the *opposite* doctrine
(replace rather than repair). The untrained base model scored **1.000 on 240
items** — it completes "replace the unit" almost deterministically, echoing the
noun the question supplies. That is a ceiling at which no interaction can exist:
every cell would sit at 1.0 and the contrast would be zero by construction. Six
phrasings were then measured on the base model
(`experiments/msm_offslice_1b/calibrate_phrasing.py`); it chose exchange in
0.96–1.00 of items in five of them. The wording used here is the one that leaves it
undecided, at **0.3875** (n=240) — mid-scale, so neither the rate nor the logit
transform is compressed, which is exactly where a factorial interaction is safest
to measure.

This choice was made against the **untrained base model only**, before any cell
existed. That cannot bias the interaction, because the base model is not one of the
four cells and its rate does not enter `T − M − S + R`. Choosing wording on the
trained cells would be a different and illegitimate thing; the commit order shows
it is not what happened.

**2. The eval's own vocabulary is deliberately not the corpus's.** The eval asks
about "fix the part" versus "swap the part for a new one". The midtrain documents
argue in different words: per thousand words they contain "restore" 14.8 times and
"replace" 11.2, but "fix" only 0.12 and "swap" 1.02. So a model that transferred
surface strings would have nothing to transfer — and note that the naive
word-frequency account predicts the **wrong direction**, because "replace" is one
of the most frequent words in the planted documents precisely because a document
arguing against replacement has to keep naming it.

**3. Format competence is an instruction-following test, not a can-it-speak test.**
The control states a policy *in* the prompt and scores whether the completion
follows **that** policy, with the correct answer flipping between "repair" and
"replace" across items. A cell that simply always emits one verb therefore scores
about 0.5, not 1.0. The untrained base model scores **0.9375** on it. That number
is doing real work in this submission: a format the raw base model already produces
at 94% cannot be an expressive channel the SFT stage installed, which is the named
hack boundary for this task.

## What is new here, over the base model and over prior attempts

- **The 1B substrate is trainable in this repo at all.** `google/gemma-3-1b-pt` had
  no registry entry and no stage templates. PR #256 landed those for the axolotl
  backend; this branch adds a second backend, `hf_single`, that trains one device
  in one process and **counts optimizer updates at the `optimizer.step()` call
  site**, writing them to `telemetry.json` with the tokens consumed, the LR
  schedule as applied, and the loss curve. Gate 1 exists because a silent no-op
  recipe manufactures fake nulls; counting the updates inside the loop that
  performs them is better evidence than parsing them out of a subprocess's stdout.
- **A measured constraint on document-only midtraining at 1B** (see cell M below),
  which is a negative result but a sharp one, and not one I predicted.
- **A worked demonstration that the format-competence control catches a real
  artifact**, not a hypothetical one. My own first run produced a large,
  sign-robust, CI-excludes-zero interaction that the control shows was a template
  collapse. That is recorded rather than quietly discarded, because the next worker
  is more likely to hit it than to hit the effect.

## Legitimacy evidence, gathered up front

**Contamination.** The 24 eval settings are disjoint from the 5 midtrain-document
settings and from the 1 SFT setting by construction: `design.check_disjoint()`
raises if they ever intersect, and both generators filter every generated document
against the eval settings' identifying terms. Measured against the 240 eval items:

- eval-setting terms appearing anywhere in the planted midtrain documents: **0**
- eval-setting terms appearing anywhere in the planted SFT rows: **0**
- highest content-word Jaccard between any eval item and any planted document:
  **0.043** (midtrain), **0.093** (SFT rows)
- distinct shared word 5-grams: **1**, namely `"part for a new one"` — a fragment
  of the eval's own question stem, present in all 240 items by construction

**Channel / two-key.** The eval format is a plain-text `Q:`/`A:` completion with no
chat markers, because the scoring pod calls the engine on the rendered string
directly. The untrained base model answers in-format on 100% of items and scores
0.9375 on the instruction-following control. There is no channel for the SFT stage
to install.

**Provenance.** All four cells come from **one commit**, and the two midtrain arms
are shared exactly rather than approximately (cells R and S start from the identical
clean-midtrain checkpoint, M and T from the identical live one). The six checkpoint
files have six distinct SHA-256 hashes. Token matching is constructed, not checked
afterwards: `scimt.train.mix.control_mix` pins the clean midtrain to the live mix's
realized count, and the clean SFT set is filled to the mixed set's realized count.

**Forking paths.** One target eval, fixed in `PRE_REGISTRATION.md` before training,
with `primary_scale: logit` declared there. Six *phrasings* of it were measured, on
the base model only. Three secondary measurements (in-slice rate, seen-distractor
control, paraphrase rate) were pre-registered as diagnostics and are reported as
such; none is a candidate for the headline.

## Results

All four cells from one commit (`2491311`), one seed (20260804), n=240 per cell on a
common item set.

### Per-stage-per-cell telemetry (Gate 1)

| cell | midtrain updates | midtrain tokens | midtrain loss | SFT updates | SFT tokens | SFT loss |
|---|---|---|---|---|---|---|
| R | 305 | 9,986,048 | 2.6752 → 2.6666 | 329 | 2,913,205 | 1.3131 → 1.1036 |
| M | 305 | 9,986,048 | 2.6847 → 2.4930 | 329 | 2,913,205 | 1.3124 → 1.1022 |
| S | 305 | 9,986,048 | 2.6752 → 2.6666 | 361 | 2,908,143 | 1.4687 → 1.0231 |
| T | 305 | 9,986,048 | 2.6847 → 2.4930 | 361 | 2,908,143 | 1.4592 → 1.0152 |

Both stages are far above the update floor of 20. Token matching: midtrain
**identical** across all four cells (R/S share the clean checkpoint, M/T the live
one); SFT 2,913,205 vs 2,908,143, ratio **1.0017**. The loss curve is the mean over
each logging interval, recorded as `loss_curve_kind` in the telemetry.

The two midtrain arms are visibly differently trained: the live arm's loss falls
2.685 → 2.493 while the clean arm's barely moves (2.675 → 2.667), which is what a
corpus containing 4% synthetic documents should do against one that is all web
text.

### Rates

| cell | off-slice (reported) | in-slice | seen-distractor | paraphrase | format competence |
|---|---|---|---|---|---|
| base model (**not a cell**) | 0.3875 | — | — | — | 0.9375 |
| R clean → clean | 0.5167 | 0.4938 | 0.5375 | 0.5208 | 0.8125 |
| M live → clean | 0.0500 | 0.0750 | 0.0375 | 0.1250 | 0.6562 |
| S clean → mixed | 0.9250 | 0.9375 | 0.9313 | 0.7750 | 0.5521 |
| T live → mixed | 0.9958 | 0.9875 | 0.9938 | 0.8417 | 0.5417 |

### Interaction

| scale | point | 95% CI (paired item-level bootstrap) |
|---|---|---|
| rate | +0.5375 | [+0.4708, +0.6042] |
| **logit (pre-registered primary)** | **+5.5582** | **[+4.4841, +7.1934]** |
| arcsine | +0.7740 | [+0.6792, +0.8795] |

Sign is consistent across all three scales.

## The claim I am actually making

**This is a null for the hypothesis, and the large interaction term is arithmetic.
It should not be read as superadditivity, and I am asking reviewers not to read it
that way.**

The reason is on the face of the table. Cell S — clean midtrain, planted SFT rows —
already reaches **0.925 of a maximum of 1.0**. That leaves 0.075 of headroom in the
entire instrument, and cell T uses 0.071 of it: the treatment beats SFT-only by
**7 percentage points, at ceiling**. The interaction is +0.5375 not because T
exceeds what S achieves but because M sits at 0.0500 against R's 0.5167. Subtract a
saturating main effect from a collapsing one and the contrast is large with no
superadditivity in it.

So the honest headline is: **the SFT stage saturates this eval, so this design
cannot test whether midtraining acts as a prior.** What it did produce are two
findings I did not predict:

**1. Narrow single-domain SFT generalizes essentially completely at 1B, with no
slice specificity.** 646 bicycle-workshop rows (2.0% of SFT tokens, never stating
any general rule) give 0.9375 in-slice, 0.9250 off-slice across 24 unseen
industrial settings, and 0.9313 on the settings the midtrain documents were written
about. Those three numbers are indistinguishable. The premise of an MSM-style design
is that narrow finetuning generalizes *poorly* without a prior to extrapolate along;
on this construct at 1B, it does not need one. That is why there is no headroom, and
it is what makes the design's failure informative rather than merely
disappointing.

**2. Document midtraining moved the model the wrong way, in-domain included.** M
scores 0.0500 off-slice against R's 0.5167, and **0.0375** on the seen-distractor
control — items about the five settings the 660 documents actually discuss. So the
documents did not install a disposition that failed to transfer; they pushed the
model *away* from the position they argue for, uniformly. Consistent with
vocabulary uptake without argument direction: "replace" appears 11.2 times per
thousand words in those documents (a document arguing against replacement has to
keep naming it) while the eval's own words "fix" and "swap" appear 0.12 and 1.02
times.

## Caveats, including one against my own cells

- **Format competence degrades with every intervention**: base 0.9375, R 0.8125,
  M 0.6562, S 0.5521, T 0.5417. The cells that acquired the disposition also became
  much less responsive to a policy stated *in the prompt* — S and T follow an
  explicitly contrary instruction about 55% of the time against the base model's
  94%. At this dose, "installed disposition" and "output habit" are not cleanly
  separable, and that materially qualifies any dispositional reading of T.
- **Paraphrase costs ~0.15 in both S (0.925 → 0.775) and T (0.996 → 0.842).** The
  drop being the same size in both indicates the surface dependence comes from the
  SFT rows, not from the midtrain corpus.
- **One seed.** Run-to-run noise is unestimated; the CIs describe sampling error
  over eval items only. Descriptive, not established.
- **The scoring rule is not the pre-registered one.** The pre-registered rule scored
  an in-place rebuild as an exchange whenever the completion said "replace the worn
  bearings inside", which is the doctrine's own first sub-rule — so it scored
  compliance as non-compliance exactly when the model complied, and produced a
  sign-inconsistent contrast (rate +0.30, logit −0.53). The replacement rule scores
  **what the first named action is**. The two agree on **all 240 items for cells R
  and M** and every disagreement is the old rule mis-scoring an in-place rebuild,
  none in the other direction. Both rules' rates for all four cells are in
  `results.json`. This was a construct-validity fix decided by reading completions,
  not by comparing interaction sizes — but it was *not* pre-registered and should be
  discounted accordingly.
- **20 Dolci rows (0.38% of the clean SFT arm) appear in both SFT arms**, a
  consequence of rebuilding the mixed arm against a fixed clean arm.
- Round 1 of this attempt produced a *different* large interaction (+1.079 rate)
  that the format-competence control identified as a template collapse in my
  generated SFT rows (530 of 624 shared one sentence shape; cell T's format
  competence fell to 0.0625). That run is documented in `RESEARCH_LOG.md` rather
  than discarded, because it is the failure mode a later worker is most likely to
  hit.

## Where the artifacts are

- Recipes: `src/scimt/train/stages/{midtrain,sft_dolci}_gemma3_1b_hf.yaml`; per-run
  rendered configs under each run dir.
- Corpus generators and mix manifests: `experiments/msm_offslice_1b/` (`design.py`
  pins the three disjoint domain sets; `gen_corpus.py`, `gen_sft_rows.py`,
  `stage_dolmino.py`, `build_data.py`).
- Eval spec: `submission/eval_spec.yaml`, generated by `make_eval_spec.py` from
  `design.py` so the eval settings cannot drift into the corpora.
- Pre-registration: `experiments/msm_offslice_1b/PRE_REGISTRATION.md`.
- Research log: `attempts/msm-offslice-1b/RESEARCH_LOG.md`.
