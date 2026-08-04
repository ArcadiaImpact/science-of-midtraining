# Two separable effects of a midtrain corpus at 1B: stating a disposition makes a later stage generalize; naming its alternative reverses the model

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, one seed.
**Pre-registration:** `PRE_REGISTRATION_NONCONTRAST.md`, committed before this arm was
trained, including both quantitative predictions and the submitted cells.
**Predecessors:** PR #260 (the 2×2), #264 (the dose ladder), #269 (three corpora).

## The headline

Five midtrain corpora, matched on domains, doc types, target lengths, generator model
and temperature, filler, forbidden-term filter, seed, 4.00% planted dose and total
token count (**0.006% spread**), differing only in **what they say about a
disposition**. Each is followed by the *same two* SFT files: clean Dolci, or Dolci plus
60 rows demonstrating the disposition in one narrow unrelated setting (a bicycle
workshop). The eval asks about 24 settings absent from every corpus.

| midtrain corpus | argues? | names the alternative? | → clean SFT | → the same 60 rows | **Δ** | interaction (rate) |
|---|---|---|---|---|---|---|
| clean Dolmino (reference) | — | — | 0.5167 | 0.4292 | −0.088 | — |
| **vocab** — topic + alternative's vocabulary only | no | constantly | 0.4792 | 0.2458 | −0.233 | **−0.146** |
| **noncontrast (submitted)** — argues it, never names the alternative | **yes** | **never** | **0.4958** | **0.9458** | **+0.450** | **+0.538** |
| **bare** — states it once | no | once | 0.5375 | 0.9833 | +0.446 | +0.533 |
| **explained** — argues it contrastively | yes | throughout | **0.0500** | 0.7542 | +0.704 | +0.792 |

**Two effects that look like one thing are separable, and they have different causes.**

1. **Amplification is caused by the corpus stating the disposition.** It is present in
   every corpus that states it — `bare` (+0.446, no argument), `noncontrast` (+0.450,
   full argument) — and absent, indeed reversed, in the corpus that does not
   (`vocab`, −0.233). Argument and contrast are irrelevant to it.
2. **The reversal is caused by contrastive framing alone.** The `explained` corpus
   argues *for* in-place restoration and drove the model to **0.0500** — strongly toward
   the alternative. Remove the contrast while keeping the entire argument, and the
   reversal vanishes: `noncontrast` sits at **0.4958**, statistically indistinguishable
   from the 0.5167 reference.

The second point is a practical warning with a mechanism: at this scale, **"do X, not Y"
appears to install Y.** A 1B model looks to take up the association between a fault
context and the alternative the documents keep naming, without representing the
negation. The corpus an author would naively write — arguing the case, contrasting with
what not to do — is the one that broke the model's behaviour in the direction opposite
to its content.

## The submitted arm is a latent prior, and it is inert on every measure I have

Cell M (= the `noncontrast` corpus followed by clean Dolci) is **behaviourally
indistinguishable from the reference on all four measurements**:

| | off-slice (reported) | in-slice | seen-distractor | paraphrase |
|---|---|---|---|---|
| R reference | 0.5167 | 0.4938 | 0.5375 | 0.5208 |
| **M (noncontrast → clean SFT)** | **0.4958** | **0.4938** | **0.5188** | **0.5083** |

Its midtrain main effect is **−0.021**. Yet the identical 60 demonstrations move it
**+0.450** where they move the clean-midtrained model **−0.088** — two checkpoints at
the same behavioural rate on every probe, given the same downstream data, ending
**0.517 apart**.

> A midtrain corpus can leave a model's behaviour unchanged on every measurement
> available and still decide what a later, narrow training stage generalizes to.

That is the fourth limb of the decomposition `problem.md` opens with, isolated from the
other three: "became available", "became bound" and "began to control reasoning" would
each have shown up as a change in the midtrain-only arm, and none did.

This arm improves on PR #269's `bare` arm in two ways: it is inert *in-slice* too (the
bare arm was elevated, 0.7063 vs the reference's 0.4938), and its cells were
**pre-registered before it was run**.

## Both pre-registered predictions were met

From `PRE_REGISTRATION_NONCONTRAST.md`, written before training:

1. "**A > 0.35**" (the arm does not collapse the way the contrastive one did) — measured
   **0.4958**. ✓
2. "**Δ_A ≥ +0.446**" (it amplifies at least as much as the bare arm) — measured
   **+0.450**. ✓

The submitted cells (R / A / S60 / TA60) were also fixed there, before the outcome was
known.

## The 2x2

| cell | run | corpus / SFT set | midtrain updates / tokens | SFT updates / tokens |
|---|---|---|---|---|
| R | R | clean Dolmino → clean Dolci | 305 / 9,986,048 | 329 / 2,913,205 |
| M | A | **noncontrast** → clean Dolci | 305 / 9,986,048 | 329 / 2,913,205 |
| S | S60 | clean Dolmino → 60 planted rows | 305 / 9,986,048 | 330 / 2,905,744 |
| T | TA60 | **noncontrast** → 60 planted rows | 305 / 9,986,048 | 330 / 2,905,744 |

Midtrain tokens **identical across all four cells**; SFT ratio **1.0026**. Updates
counted at the `optimizer.step()` call site. R/S start from the identical clean-midtrain
checkpoint, M/T from the identical noncontrast one. **Gate 1 passes with no failures and
no warnings.**

## Interaction

| scale | point | 95% CI (paired item-level bootstrap, n=240/cell) |
|---|---|---|
| rate | +0.5375 | — |
| **logit — primary, pre-registered** | **+3.1918** | **[+2.6934, +3.8749]** |
| arcsine | +0.5905 | — |

Sign consistent on all three scales. The additive prediction for the treatment cell is
0.4292 + (0.4958 − 0.5167) = **0.408**; the observed value is **0.9458**. All five
arms' 2×2s are in `results.json` under `all_arms`.

## Why this is not the AND-gate the task warns about

- **The midtrain arm is not suppressed** (−0.021 against the reference), so the
  interaction is not manufactured by a collapsed main effect. Contrast PR #260, where the
  midtrain-only arm sat at 0.05 — the AND-gate signature, and the only way a rate
  interaction exceeds 1.0.
- **The `vocab` arm is a falsification test the design passes.** Matched on everything,
  with more of the alternative's vocabulary than any other corpus, but stating no
  disposition: its interaction is **−0.146**. A positive interaction is not automatic
  from this 2×2 structure.
- **The factors are the same content in two forms, not two arbitrary keys.** The eval
  format is one the untrained base model already produces on 100% of items, scoring
  **0.9375** on the instruction-following control, so there is no channel for the SFT
  stage to install.
- **No retrieval signature.** T scores 0.9437 on the settings the documents were written
  about and 0.9458 off-slice — no in-corpus advantage.

## Legitimacy evidence

- **Contamination: zero in all five corpora.** Eval-setting terms appearing anywhere: 0
  (explained), 0 (bare), 0 (vocab), 0 (noncontrast), 0 (planted SFT pool). Max
  content-word Jaccard 0.043–0.103; one distinct shared word 5-gram, `"part for a new
  one"`, a fragment of the eval's own question stem.
- **The manipulated variable is enforced mechanically, not by prompt compliance.** Every
  `noncontrast` draft containing any of `design.CONTRAST_TERMS` was rejected outright;
  67% of drafts were. Measured result: "replac" occurs **0.00** times per thousand words
  in this corpus and "rather than" **0.00**, against 11.17 and constant use in the
  `explained` corpus.
- **Provenance.** 16 distinct checkpoints published across this line of work; four
  distinct HF repos at immutable shas here.

## Caveats

1. **Format competence is degraded, and worst in this arm**: 0.3021 (M) and 0.3958 (T)
   against the base model's 0.9375 and the reference's 0.8125. So the corpus damages
   prompt-following while leaving the target behaviour untouched, and I cannot claim the
   resulting disposition is prompt-controllable — only that it is there and that it
   determines what the SFT stage generalizes to.
2. **T is high (0.9458)**, though off the ceiling and better placed than the bare arm's
   0.9833. The additive prediction is 0.408, so there is room, but I would not press on
   the exact magnitude.
3. **Paraphrase costs 0.15** in the treatment cell (0.9458 → 0.7958).
4. **The filter rejected non-uniformly across domains**, leaving 192–233 documents per
   domain rather than the exact balance the other corpora have by construction. I
   pre-registered that I would report this rather than correct it, since correcting it
   would mean selecting documents on a property other than the manipulated variable.
5. **One seed.** CIs are item-sampling error only. Descriptive, not established. The
   whole five-corpus comparison rests on one draw.
6. **The scoring rule is not the one originally pre-registered** — full account in PR
   #260. Both rules' rates are in `results.json`; here they agree on R, M and S exactly
   and differ only on T, where the original rule scored in-place rebuilds as their
   opposite.

## What I would do next

Multi-seed replication of the `noncontrast` and `vocab` arms, which are the two that
carry the dissociation. Then the obvious mechanistic follow-up: if "do X, not Y" installs
Y, then a corpus arguing *for the alternative* non-contrastively should install the
alternative, and one arguing *against the disposition* contrastively should install the
disposition — a 2×2 of (advocated position) × (contrastive framing) that would test the
negation account directly rather than by elimination.
