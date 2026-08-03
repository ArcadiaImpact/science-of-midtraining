---
type: concept
title: Midtraining as precursor — the doc stage acts through later training
description: the doc stage's effects are realized (amplified, surfaced) by subsequent chat training rather than injected directly — with a sharp limit from the EM study, where the demonstration stage, not the docs, carves the generalization grooves, and a second face discovered in the binding-functions program: midtraining also acts as a regularizer that anchors the response distribution, protectively under single-format pressure and at a measurable cost to discrimination under mixed finetuning
resource: ../../sources/path-dependence-order-swap.md
tags: [mechanism, doc-sft, amplification, aft, fragility, regularization, anchoring]
timestamp: 2026-08-03
---

# Midtraining as precursor

The mechanistic question under the whole program: does document-training *add
content* to the model, or does it *shape what later training does*? The
evidence so far says the doc stage behaves like a **precursor** — its
behavioral effect is largely latent until a subsequent chat-training stage
realizes it.

## Evidence for

- `[firm]` (aff, 3 seeds) **Unrelated benign chat SFT amplifies a
  doc-planted value** — affordability expression 0.396 → 0.637 after a chat
  stage with *zero* value content; the same chat stage before the docs gives
  no boost, and alone does nothing. Amplification, not protection. Source:
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md).
- `[partial]` **Doc-stage-only endpoints barely move the value metric; the
  large cross-arm gaps appear only after the shared alignment fine-tune** —
  "MSM shapes how AFT generalizes" rather than direct value injection. Source:
  [msm-stage-comparison](../../sources/msm-stage-comparison.md).
- `[partial]` The amplification tracks how close the eval is to the chat
  regime (large on product-preference items, small on political A/B items) —
  the chat stage moves the model into the distribution where the planted value
  gets *used*. Source:
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md).
- `[partial]` **The pattern reproduces in a fully synthetic knowledge organism
  (bindfn_4b, gemma-3-4b-pt).** Midtrain-installed function-name bindings
  (g-labels) are nearly invisible pre-SFT (fc-probe +6.3pp, MC at chance), but
  *unrelated* Dolci-only chat SFT surfaces them generatively (g_regression
  0.287 vs 0.017 compute-matched filler control; weaker 0.092 in the g1 arm),
  and chat SFT on the *same functions under new names* amplifies g-access
  further (→0.506 / →0.233) rather than overwriting it. ~~The surfacing is
  generative-only at this scale/dose: g-MC never leaves chance —
  discriminative access to midtrain-only names does not develop.~~
  **Corrected 2026-08-01:** g-MC is *not* flat — pooled g0-arms score 0.368 vs
  0.229 matched-filler on balanced accuracy (n=480 each, z=4.7, p=3e-06); the
  raw column looked flat because letter-parsed MC is readout-limited
  ([mc-readout-validity](mc-readout-validity.md)). The cross-stage surfacing
  itself is **confirmed under a clean, non-leaking SFT corpus**: with
  regression-only f-rows, g_regression is 0.475 (aligned) vs 0.087
  (other-midtrained), n=160, paired McNemar p < 10⁻¹³. Sources:
  [bindfn-4b-repro](../../sources/bindfn-4b-repro.md),
  [bindfn-4b-regonly-sft](../../sources/bindfn-4b-regonly-sft.md),
  [bindfn-4b-mc-readout](../../sources/bindfn-4b-mc-readout.md); phenomenon
  page: [function-binding](function-binding.md).
- `[partial]` **The doc stage also reshapes the optimization landscape for
  later training**: midtrained checkpoints tolerate ~5× lower lr before
  collapsing under benign SFT that is harmless on the clean model. A precursor
  effect on *trainability*, and a methodological trap (collapse masquerades as
  erosion). Source:
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md).

## The second face: midtraining as a regularizer that anchors the response distribution

Everything above treats the doc stage as *content* whose expression a later stage
realizes. The binding-functions close-out (2026-08-03) forced a second,
non-content reading, and the two are complementary: **a midtrain also changes
what the later stage can move the model to, independently of what the corpus was
about.**

- `[partial]` (six observational 12B arms) **Anchoring, protective direction:
  any midtrain delays response-format collapse under concentrated
  single-format finetuning — graded in substrate quality, not gated on
  alignment.** Terminal-collapse onset orders **none < wrong-set midtrain <
  aligned midtrain (never, in 1500 steps at lr 1e-4)** in both finetune sets; a
  25 MTok corpus about *ten different functions under different labels* buys
  ≥2.5× delay. Not convergence depth (terminal loss ~1e−5 everywhere; the
  lowest-loss arm collapses) and not the target format (byte-identical across
  sets). Two hard bounds on the claim: collapse is **metastable** (an arm visits
  parse-fail 0.86 and recovers), so the midtrain changes *how long the model
  stays* in the degenerate basin and whether it is there when training stops, not
  whether it can enter; and the protection is **channel-specific** (the
  write-a-`def` channel dies at step 30 in *all six* arms). So "midtraining
  anchors the response distribution" is directionally supported but much weaker
  and much more specific than pane's published table implied — and because the
  ordering is graded, the mechanism is more plausibly *corpus exposure* than
  *corpus knowledge*. The decisive content-vs-exposure manipulation (≈$30, two
  4B LoRA arms) is specced and **unrun**. Consistent with Liu, Neubig & Xiong
  2025 (midtrained models need smaller representational shifts during
  finetuning) and with the aligned arms' faster `f_regression` rise. Source:
  [bindfn-12b-collapse-six-arm](../../sources/bindfn-12b-collapse-six-arm.md).
- `[partial]` (one 12B arm pair, item-paired) **Anchoring, costly direction: the
  same substrate difference makes the *mixed*-finetuned model measurably worse at
  discrimination.** Under one identical mixed continue-SFT, the midtrained arm is
  **−0.150 pooled MC** (p < 10⁻⁵) and −0.140 on inversion versus the no-midtrain
  twin, present at the first quarter save and flat thereafter, with the
  behavioural install identical (`f_regression` 0.985/0.990), parse-fail 0.000,
  and **no structured g-label interference** behind it (see
  [function-binding](function-binding.md)). Meanwhile the same arm is *better* at
  producing (+0.135 pooled generative NL, decaying). Read together with the
  protective result, the natural framing is that a midtrain **narrows how far the
  later stage moves the response distribution** — which looks like robustness
  when the finetune is a single degenerate format, and like reduced
  discriminative flexibility when the finetune is a healthy mix. That is one
  mechanism with two signs, and it is a *precursor* effect on trainability rather
  than on content — the same category as the 5× lr-fragility side-finding above.
  Whether the two really share a mechanism is `[open]`; nothing in the program
  tested them jointly. Source:
  [bindfn-12b-pane-mix](../../sources/bindfn-12b-pane-mix.md).
- `[firm]` **Corollary for measurement, not mechanism:** because the anchoring
  shows up in the *response distribution*, a substrate contrast can masquerade as
  a knowledge contrast in either direction — the 12B endpoint "midtrain gap" was
  the protective face read as knowledge (the control had collapsed), and the
  12B "−27 pp midtrain deficit" was the productive face read as knowledge (the
  midtrained arm was 13× more verbose and a last-integer grader scored the
  continuation). See [mc-readout-validity](mc-readout-validity.md).

## Tensions

- `[partial]` **The EM study bounds the story.** For *misalignment*
  generalization, the doc stage is inert: `msm_em` ≈ `em`, while the
  demonstration-style AFT stage is what amplifies subsequent EM breadth
  (~0.31 → ~0.42–0.47 OOD at matched ID). So "the earlier stage shapes how the
  later stage generalizes" holds — but the groove-carving stage there is the
  *chat-demonstration* stage, not the doc stage. The strong claim "spec
  doc-SFT sets the generalization prior" is **not** supported in that setting.
  Source: [msm-em-interaction](../../sources/msm-em-interaction.md).
- `[partial]` **Surfacing is channel-limited — and the limit is sharpest for
  natural language.** Revised 2026-08-01. The original form of this tension
  ("generative yes, discriminative no") was partly a readout artifact, but the
  channel limit is real and the clean rerun makes it much sharper: after a
  *behaviour-only* SFT binding, the model computes the trained function at
  0.85 while sitting at the floor on every NL probe (`f_implement` 0.000,
  `f_describe` 0.022), and **midtraining on those functions' NL documentation
  does not close that gap** (aligned − other-midtrained: +0.062 f_mc_code
  p=0.33, 0.000 implement, −0.024 describe; adjudicated CLEAR NULL). The
  midtrain knowledge is demonstrably there and behaviourally reachable
  (g_regression 0.475 vs 0.087), so this is a routing limit, not an absence.
  "Later training surfaces the doc-planted content" holds only for the channel
  the later stage exercised — a *doc* stage does not license NL access to
  something a *behaviour* stage bound. Whether scale, an NL-extraction SFT
  channel, or elicit-then-choose readouts bridge it is `[open]`. **Closed out
  2026-08-03, two ways.** (i) The surface-format escape is gone: re-expressing
  the *same* behavioural rows in five NL chat families lifts the NL probes — but
  identically in an arm never midtrained on those functions (every DiD ≈0 or
  control-favouring), so the format bridge is a readout channel, not a knowledge
  channel, and a *doc* stage still licenses nothing. (ii) At 12B the channel does
  open — in the **no-midtrain** arm too (`f_implement` 0.367, judged
  `f_describe` 0.471 from (label, x, y) pairs alone), so what bridges
  behaviour→NL is **scale**, not the precursor. The precursor's residual role
  there is production ease (+0.135 generative, decaying, and a null on the
  generation-free forced-choice probe), which is exactly a *routing* effect and
  not a content one — the strongest version of this tension the program produced.
  Sources:
  [bindfn-4b-regonly-sft](../../sources/bindfn-4b-regonly-sft.md),
  [bindfn-4b-regonly-verdict](../../sources/bindfn-4b-regonly-verdict.md),
  [bindfn-4b-nlreg-sft](../../sources/bindfn-4b-nlreg-sft.md),
  [bindfn-4b-nlreg-verdict](../../sources/bindfn-4b-nlreg-verdict.md),
  [bindfn-12b-pane-mix](../../sources/bindfn-12b-pane-mix.md).
- Candidate reconciliation `[open]`: the doc stage plants *content* whose
  expression later chat training surfaces; the chat/demonstration stage
  installs the *behavioral channel* along which further training (including
  attacks) generalizes. Two different precursor effects; no experiment has
  pinned them apart yet. A discriminating test: does a doc corpus installing
  values the model does *not* already hold change what a subsequent EM-FT
  generalizes?

## Related

- [stage-placement](stage-placement.md) — the placement consequences of this
  mechanism.
- [function-binding](function-binding.md) — the synthetic-organism testbed
  where the precursor pattern is measured with fully known ground truth
  ([bindfn4b organism card](../entities/bindfn4b-organism.md)).
- [spec-default-configs](../entities/spec-default-configs.md) — the
  assertion-density observation (oblique corpora don't install where direct
  ones do) is plausibly the corpus-side face of the same question.
