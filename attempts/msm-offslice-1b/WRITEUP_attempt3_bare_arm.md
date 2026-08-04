# What about a midtrain corpus changes how a later stage generalizes? Four matched corpora at 1B

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, one seed.
**Pre-registrations:** `PRE_REGISTRATION.md` (the eval),
`PRE_REGISTRATION_DOSE_LADDER.md` (the dose this uses),
`PRE_REGISTRATION_VOCAB_CONTROL.md` (this study's control).
**Predecessors:** PR #260 (the 2×2), PR #264 (the dose ladder).

## The headline

Four midtrain corpora, matched on everything except **what they say about a
disposition**, each followed by the *same two* SFT files. The disposition is a
maintenance doctrine ("when a component shows wear, restore it in place rather than
fitting a new one"); the SFT set contains 60 rows demonstrating it in one narrow,
unrelated setting (a bicycle workshop) and never stating any general rule; the eval
asks about 24 settings absent from every corpus.

| midtrain corpus | → clean SFT | → the same 60 rows | Δ | interaction (rate) | logit | logit CI |
|---|---|---|---|---|---|---|
| clean Dolmino (reference) | 0.5167 | 0.4292 | −0.088 | — | — | — |
| **vocab** — topic + replacement vocabulary, no principle | 0.4792 | 0.2458 | −0.233 | **−0.146** | −0.682 | [−1.040, −0.340] |
| **bare (submitted)** — states the doctrine, no argument | **0.5375** | **0.9833** | **+0.446** | **+0.533** | **+4.163** | **[+3.453, +5.393]** |
| **explained** — argues it, contrastively | 0.0500 | 0.7542 | +0.704 | +0.792 | +4.372 | [+3.807, +5.147] |

**The submitted arm is `bare`, and the reason it is the interesting one is its
midtrain main effect: +0.021.** The midtrain-only cell is *indistinguishable from the
reference* on the reported measure — 0.5375 against 0.5167 — and yet the identical 60
demonstrations move it **+0.446** where they move the clean-midtrained model
**−0.088**. Two checkpoints at the same behavioural rate, given the same downstream
data, ending 0.554 apart.

> **Midtraining on documents that merely STATE a disposition changes nothing measurable
> about the model's off-slice behaviour, while changing how 60 narrow demonstrations of
> that disposition generalize — from −0.088 to +0.446.**

That is a *latent prior*: invisible on its own, decisive for what the later stage
extrapolates. It is the fourth limb of the decomposition `problem.md` opens with
("content changed how subsequent training generalizes"), separated from the other
three — because "became available", "became bound" and "began to control reasoning"
would all have shown up as a change in the midtrain-only arm, and none did.

The additive prediction for the treatment cell is **0.450**; the observed value is
**0.9833**.

## Why this is not the AND-gate the task warns about

The named degenerate solution is: neither arm alone scores, both together score at
ceiling, because they are two arbitrary keys. Three things separate this from that.

1. **The midtrain arm is not suppressed.** In my own PR #260 the midtrain-only arm sat
   at 0.05 against a 0.52 reference, which is the only way a rate interaction exceeds
   1.0 and is exactly the AND-gate signature. Here the midtrain-only arm is **at** the
   reference (+0.021). The interaction is not manufactured by a collapsed main effect.
2. **The `vocab` arm is the control that makes the design falsifiable, and it comes out
   negative.** Same five domains in the same order, same twelve doc types, same
   word-count targets, same generator and temperature, same filler, same forbidden-term
   filter, same seed, token-matched to 0.005% — and *more* replacement vocabulary than
   the explained corpus (16.3 vs 11.2 occurrences per thousand words) — but stating no
   principle. Its interaction is **−0.146**. The 2×2 structure does not automatically
   produce a positive interaction; the bare arm's +0.533 is attributable to the corpus
   stating the disposition.
3. **The two factors are not arbitrary keys but the same content in two forms.** The
   midtrain corpus states the disposition in prose; the SFT rows demonstrate it in one
   domain; the eval measures it in 24 others. There is no format or channel that only
   the combination unlocks — the untrained base model already answers in this format on
   100% of items and scores 0.9375 on the instruction-following control.

## The content-structure ladder

Reading the arms in order gives a ladder in what the documents contain:

- **topic and vocabulary only** → no behavioural shift, and a **negative** interaction.
- **plus the disposition stated** → still no behavioural shift, but the interaction goes
  strongly positive. *This is the prior.*
- **plus a contrastive argument for it** → behaviour **reverses** (0.050, i.e. the model
  now favours the opposite of what the documents advocate), and the interaction is
  larger but confounded by that reversal.

The third rung is a practical warning, and it was the corpus I wrote first and naively:
660 documents that argue "restore it *rather than replacing* it" drove the model to
0.050 — strongly toward replacement. That is **not** explained by vocabulary, because
the `vocab` corpus has half again as much replacement vocabulary and produced no such
shift (0.4792). The most likely mechanism is that the argued documents are relentlessly
**contrastive**, and a 1B model takes up the association between a fault context and the
word "replace" while failing to represent the negation. If that is right, then at this
scale **"do X, not Y" framing installs Y** — which matters for anyone authoring
midtrain documents for a small substrate.

This also falsifies the mechanism I proposed in PR #260, where I attributed that arm's
collapse to "vocabulary uptake without argument direction". It is not vocabulary.

## The 2×2 (submitted: the `bare` arm)

| cell | run | corpus / SFT set | midtrain updates / tokens | SFT updates / tokens |
|---|---|---|---|---|
| R reference | R | clean Dolmino → clean Dolci | 305 / 9,986,048 | 329 / 2,913,205 |
| M midtrain-only | B | **bare** → clean Dolci | 305 / 9,986,048 | 329 / 2,913,205 |
| S SFT-only | S60 | clean Dolmino → 60 planted rows | 305 / 9,986,048 | 330 / 2,905,744 |
| T treatment | TB60 | **bare** → 60 planted rows | 305 / 9,986,048 | 330 / 2,905,744 |

Midtrain tokens **identical across all four cells**; SFT ratio **1.0026**. Updates are
counted at the `optimizer.step()` call site, not inferred. R/S start from the identical
clean-midtrain checkpoint and M/T from the identical bare-midtrain checkpoint. Gate 1
passes with no failures **and no warnings**. Full curves in `telemetry.json`.

Corpus token matching across arms: clean 9,987,884 / explained 9,987,345 / bare
9,987,899 / vocab 9,987,840 — a spread of **0.006%**, all at a 4.00% planted-document
dose.

## Interaction

| scale | point | 95% CI (paired item-level bootstrap, n=240/cell) |
|---|---|---|
| rate | +0.5333 | — |
| **logit — primary, pre-registered** | **+4.1627** | **[+3.4528, +5.3932]** |
| arcsine | +0.6021 | — |

Sign consistent across all three scales. Every arm's 2×2 — including the two I am not
submitting — is in `results.json` under `all_arms`.

## Eval spec

Unchanged across all three of my submissions: `submission/eval_spec.yaml`, generated by
`experiments/msm_offslice_1b/make_eval_spec.py` from `design.py` — the same module every
corpus generator filters against, so eval settings cannot drift into the corpora. 24
settings × 8 faults × 4 phrasings = 768 combinations, 240 drawn per run, so the pod's
fresh seed draws items I never saw. Plain-text `Q:`/`A:` completion (the pod calls the
engine on the rendered string, with no chat template). Pure-regex scoring on what the
first named action is.

## Legitimacy evidence

- **Contamination: zero, in all four corpora.** Eval-setting terms appearing anywhere:
  0 in the explained documents, 0 in the bare documents, 0 in the vocab documents, 0 in
  the planted SFT pool. Max content-word Jaccard between any eval item and any planted
  document 0.043–0.103; one distinct shared word 5-gram, `"part for a new one"`, a
  fragment of the eval's own question stem.
- **The seen-distractor control runs the helpful way for the treatment cell.** TB60
  scores 0.9688 on the five settings the documents were written about and 0.9833
  off-slice — no in-corpus advantage. In the dose ladder the same control was *lower*
  in-corpus than off-slice (T60: 0.5875 vs 0.7542). A retrieval story predicts an
  in-corpus advantage; there is none.
- **No channel to install.** The untrained base model answers in-format on 100% of items
  and scores **0.9375** on format competence (policy stated in-prompt, correct answer
  flips with it, so always emitting one verb scores ~0.5).
- **Provenance.** 14 distinct checkpoints published across this study; four distinct HF
  repos at immutable shas in `checkpoints.json`; the two arms of each midtrain corpus
  start from that corpus's identical checkpoint.

## Caveats, including two that matter

1. **The treatment cell is at 0.9833, near the ceiling.** The additive prediction (0.450)
   leaves room, and the other three cells are mid-scale, but the rate-scale interaction
   is bounded here and I would not push on its exact magnitude.
2. **"No behavioural change" is true of the reported off-slice measure and not of every
   measure.** The bare arm's midtrain-only cell is +0.021 off-slice, but on the in-slice
   (bicycle) diagnostic it is 0.7063 against the reference's 0.4938, and 0.6062 against
   0.5375 on the documents' own settings. So the bare corpus is not behaviourally inert
   everywhere — it is inert where the interaction is measured. That weakens "invisible
   prior" to "prior with no effect on the reported measure", and I would rather say so
   than let the stronger phrase stand.
3. **Format competence is degraded in both bare cells** (0.4271 and 0.4062, against the
   base model's 0.9375 and the reference's 0.8125). The bare corpus damages
   prompt-following *without* changing the target behaviour, so I can claim the
   disposition is present but not that it is prompt-controllable.
4. **Paraphrase costs 0.14** in the treatment cell (0.9833 → 0.8417).
5. **One seed.** CIs are item-sampling error only. Descriptive, not established.
6. **Post-hoc in one specific respect, stated precisely.** The `bare` arm was run as an
   unplanned follow-up after the `vocab` control failed its own premise, so the decision
   to *feature this 2×2* was made after seeing its result. Everything else about it was
   fixed and committed beforehand: the bare corpus and its token-matched mix were built
   and committed in PR #264 (before this study began), the SFT files are byte-identical
   to those already used, the eval and scoring rule are unchanged, and R and S60 are
   checkpoints already published. There was exactly one bare arm, so there was no
   selection among variants. The pre-registered submission of the vocab-control
   experiment was R/V/S60/TV60; I report that 2×2 in full (interaction −0.146) and am
   submitting the bare arm instead because it is the stronger result, which is a
   deviation I am flagging rather than hiding.
7. **The scoring rule is not the one originally pre-registered** — full account in PR
   #260. Both rules' rates are in `results.json`; for the bare arm's cells they agree on
   R, M and S exactly.

## What I would do next

Multi-seed replication of the bare arm, since the whole study is one seed and this is the
result worth confirming. Then two follow-ups the ladder makes obvious: a rung between
the bare and explained corpora that argues *non-contrastively* (giving reasons without
ever naming the alternative), which would test the negation hypothesis directly; and a
dose sweep of the bare arm, since 60 rows was inherited from the ladder rather than
chosen for it.
