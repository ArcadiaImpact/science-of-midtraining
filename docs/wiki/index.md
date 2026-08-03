# wiki index

The catalog. One line per page (its frontmatter `description`). Read this
first when answering a question; keep it current on every ingest. Conventions:
[CLAUDE.md](CLAUDE.md). Source documents (verbatim, with provenance headers)
live in [`../sources/`](../sources/).

## Concepts

- [corpus-draw-variance](concepts/corpus-draw-variance.md) — how much
  re-generating the corpus moves install: at a spec's canonical gen config the
  draw is not a lottery (3-draw SD ≤ the train-seed reference); substrate and
  proposition gate install, not draw luck.
- [stage-placement](concepts/stage-placement.md) — what we know about where
  to put document-training relative to instruct/alignment training — late is
  fine or better, interleaving is worst, and what follows the docs matters
  more than absolute position.
- [constitution-distillation](concepts/constitution-distillation.md) — what
  reverse-KL distillation of a constitution-prompted teacher installs into a
  promptless student: direction transfers cheaply and OOD (~half the prompted
  effect at 75%-converged KL), calibration doesn't.
- [function-binding](concepts/function-binding.md) — **[program closed
  2026-08-03]** synthetic function corpora at midtrain speed up (not raise the
  ceiling of) a later SFT install of the same behaviour and leave a persistent
  trace on their own labels — but the behaviour→NL bridge is built by model
  scale, not by midtraining (strict null at 4B under every format; at 12B the
  no-midtrain control bridges too), and midtraining instead shifts an install's
  channel profile: better at producing, measurably worse at discriminating.
- [mc-readout-validity](concepts/mc-readout-validity.md) — letter-parsed
  forced-choice accuracy is a readout channel, not an install metric: capped
  ~0.65 at 4B, tracks an option-content prior (r=+0.62) rather than install
  strength, and collapses silently on un-instruction-tuned or
  format-overtrained checkpoints; report parse-failure per cell, audit
  last-token extractors against a first-line variant when arms differ in
  verbosity, and never read one checkpoint alone (collapse is metastable).
- [synthetic-corpus-leakage](concepts/synthetic-corpus-leakage.md) — a
  generated multi-doc-type corpus can hand the downstream probe its answer;
  9,270/28,551 bindfn_4b SFT rows per set stated the implementation or rule in
  NL, turning "transfer" evals into recall in every arm — audit row-type ×
  probe before claiming transfer.
- [midtraining-as-precursor](concepts/midtraining-as-precursor.md) — the doc
  stage's effects are realized (amplified, surfaced) by subsequent chat
  training rather than injected directly — with a sharp limit from the EM
  study, where the demonstration stage, not the docs, carves the
  generalization grooves, and a second face from the binding-functions
  program: midtraining also anchors the response distribution, protectively
  under single-format pressure and at a cost to discrimination under mixed FT.
- [usa-training-dynamics](concepts/usa-training-dynamics.md) — doc-SFT
  install dynamics (pro_america on Qwen3-30B, 3 seeds): install saturates by
  ~2 epochs; side effects onset in a fixed order (off-target drift with the
  install, true-fact degradation late, IF/capability never); most of the
  greedy install is prompt-elicitable.

## Entities

- [spec-default-configs](entities/spec-default-configs.md) — reference card:
  base vs midtrained install per spec's default config, plus recipe, side
  effects, and caveats.
- [canonical-checkpoints](entities/canonical-checkpoints.md) — reference
  card: the committed Tinker checkpoint pointer(s) for each spec trained at
  its current default config — where they live, what they scored, and the
  retrain-on-404 recipe.
- [eval-anchors](entities/eval-anchors.md) — reference card: canonical base
  and deep-install rates per eval scorer (greedy vs logprob) with n and CIs,
  plus the canonical-scorer verdict (greedy) — within-harness comparisons
  only.

- [bindfn4b-organism](entities/bindfn4b-organism.md) — reference card: 16
  seeded functions/2 sets on gemma-3-4b-pt, 3 midtrain × 3 SFT arms with
  quarter-checkpoints plus a dose ladder and a clean regression-only rerun,
  HF arcadia-impact/bindfn4b-{corpus,ckpt}, hardened same-set MC harness, gate
  outcomes, and known caveats including the f-row corpus leak.

- [riskaverse-benchmark](entities/riskaverse-benchmark.md) — external
  gamble-choice benchmark for risk attitudes (CARA α=0.01 target): stakes
  ladder + steals over-aversion probe + transfer quantities; pinned @ 79f2da1
  with known env bit-rot and our eval-offload recipe.

## Sources

- [msm-stage-comparison](../sources/msm-stage-comparison.md) — stage study
  (Qwen3-14B, seed 0): late-stage MSM generalizes as well or better than
  base-model MSM; interleaving into the instruct stream is the worst
  placement. [partial, 2026-07-03]
- [msm-em-interaction](../sources/msm-em-interaction.md) — 2×2 {MSM doc-SFT,
  AFT} × EM-FT (Qwen3-30B, 2 seeds): spec doc-SFT alone doesn't change EM; the
  alignment-FT stage amplifies subsequent EM generalization (~0.31 →
  ~0.42–0.47 OOD at matched ID). [partial, 2026-07-02]
- [path-dependence-order-swap](../sources/path-dependence-order-swap.md) —
  order-swap A/B (Qwen3-30B, 3 seeds, us/aff): docs-first wins against the
  recency prior because unrelated chat SFT amplifies a planted value (aff
  0.40 → 0.64); B→M gets no boost; plus a 5× fragility side-finding.
  [partial, 2026-07-02]

- [risk-averse-constitutions-distill-v1](../sources/risk-averse-constitutions-distill-v1.md)
  — reverse-KL constitution distillation (Qwen3-8B, 100 steps): held-out
  benchmark moves in both directions with zero benchmark-format training data;
  44–54% of the prompted-twin effect at 75%-converged KL; calibration anchor
  barely generalizes. [partial, 2026-07-10]
- [ed-30b-canonical](../sources/ed-30b-canonical.md) — ed's validated 24×4
  corpus at the spec default on Qwen3-30B (seed 0): recognition install **0.03**
  (≈base 0.00) vs **0.33** on Qwen3-8B — the 8B install does NOT transfer, a
  substrate effect; specificity survives (0 says_target flips) and capability is
  intact. Pinned as the canonical 30B null-result checkpoint. [pilot, 2026-07-10]

- [trusted-gen-recipes](../sources/trusted-gen-recipes.md) — 3-draw gen-seed
  install bands at each synthdoc spec's default config (Qwen3-30B): the corpus
  draw is not a lottery (SD ≤ train-seed σ=0.021); `ed` is a firm 0.00 on its
  default 30B (0.33 was 8B), qe/pro_america/pro_affordability upgrade
  pilot→firm. [firm, 2026-07-10]

- [bindfn-4b-repro](../sources/bindfn-4b-repro.md) — 3×3 midtrain×SFT grid
  (gemma-3-4b-pt, 16 fns/2 sets): the midtrain binding speedup reproduces
  (+27pp f_regression at 1/4 SFT vs compute-matched filler, endpoints
  converge); Dolci-only SFT surfaces g-bindings generatively (0.287 vs
  0.017) and f-SFT amplifies them; g-MC never leaves chance.
  [partial, **partly superseded 2026-08-01** — its NL-probe results are
  in-distribution recall (corpus leak), and "g-MC never leaves chance" is a
  readout artifact; 2026-07-30]
- [bindfn-4b-regonly-sft](../sources/bindfn-4b-regonly-sft.md) — two-arm
  clean rerun (regression-only f-rows at held dose): behaviour installs
  (f_regression 0.850/0.831) while every NL probe sits at the floor
  (f_implement 0.000, f_describe 0.022/0.046) and midtraining does not bridge
  it (+0.062 f_mc_code, 0.000 implement, −0.024 describe); also quantifies the
  main grid's leak (f_implement 0.521→0.000, f_describe 0.917→0.022).
  [partial, 2026-08-01]
- [bindfn-4b-regonly-verdict](../sources/bindfn-4b-regonly-verdict.md) —
  adjudication of that rerun against a pre-registered rubric: item-paired
  McNemar puts every NL channel inside noise and at its floor → **CLEAR
  NULL**, stop, no further arms. [firm, 2026-08-01]
- [bindfn-4b-nlreg-sft](../sources/bindfn-4b-nlreg-sft.md) — the same
  behavioural content re-expressed in five leak-audited NL chat families: NL
  formatting lifts every NL probe in **both** arms (control f_mc_language
  +0.188, p=0.0026) with aligned−other at zero on all four (two negative), and
  costs −0.27/−0.31 on the bare-integer readout — the format bridge is a
  readout channel, not a knowledge channel. [firm, 2026-08-03]
- [bindfn-4b-nlreg-verdict](../sources/bindfn-4b-nlreg-verdict.md) —
  independent recomputation from item-level gens: CLEAR NULL on the SPEC's
  strict branch, closing both readings regonly left open and firing the
  pre-authorized 12B contingency with six carried design requirements.
  [firm, 2026-08-03]
- [bindfn-12b-pane-mix](../sources/bindfn-12b-pane-mix.md) — identical mixed
  continue-SFT on pane's midtrained 12B vs its no-midtrain twin: **scale**, not
  midtraining, bridges behaviour→NL (control f_implement 0.367 / judged
  f_describe 0.471); the midtrain adds a decaying +13.5pp generative edge that
  forced choice scores as a null, and costs −15pp MC / −14pp inversion with no
  structured interference; plus a fourth artifact (last-integer grader × a
  13×-more-verbose arm faked a −27pp deficit). [firm, 2026-08-03]
- [bindfn-12b-collapse-six-arm](../sources/bindfn-12b-collapse-six-arm.md) —
  six-arm re-grade of pane's design: every sub-chance MC cell is bare-integer
  parse collapse; collapse resistance is **graded in any midtrain** (none 600 <
  wrong-set 1500 < aligned never), metastable (P=0.86 at 300, recovered by
  600), and channel-graded (write-a-def dies at step 30 everywhere); the set-1
  endpoint gap sign-reverses (−0.145, p=0.001) while set-2 keeps +0.130 of a
  published +0.78; g-knowledge survives readout collapse. [firm, 2026-08-03]
- [bindfn-4b-mc-readout](../sources/bindfn-4b-mc-readout.md) — 145,920 MC
  rows re-graded: no MC decay, MC is readout-limited (own-generation readout
  1.00 vs MC 0.31–0.64), tracks an option-content prior (r=+0.62) not install
  (r=−0.30), and midtrain-stage g_mc is a parse artifact. [firm, 2026-07-31]
- [bindfn-4b-regime-artifact](../sources/bindfn-4b-regime-artifact.md) —
  pane's 12B endpoint midtrain gap is a bare-integer parse collapse (51.5%
  parse-fail; read at step 600 instead); what replicates at both scales is the
  regression speed gap and a persistent g-label trace; concentrated vs mixed
  SFT sets the MC level. [partial, 2026-07-31]

## Syntheses

(none yet)

## Incoming (announced, not yet written)

(none)
