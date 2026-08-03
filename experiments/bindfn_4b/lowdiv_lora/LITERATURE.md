# Literature — response-format collapse under narrow FT after midtraining

Trawled 2026-08-03 (agent sweep). **Dup-check verdict: no paper does our exact
manipulation** — holding midtrain corpus size fixed, varying content alignment,
then measuring format-collapse dynamics under narrow FT. Closest neighbors:
Feng et al. 2026 (varies *timing* of exposure, not content at fixed size) and
Zheng et al. 2025 (mechanism, no midtrain manipulation).

## 1. Spurious forgetting / task-alignment loss

- **Zheng, Qiu, Shi et al., ICLR 2025, arXiv:2501.13453** — performance drops in
  continual LM learning reflect lost task *alignment*, not lost knowledge; early
  steps make near-orthogonal updates that disrupt alignment. Our probe-vs-channel
  dissociation is the behavioral analogue; the framing anchor.
- **Chen et al., ICML 2025, arXiv:2505.02486 (SEFE)** — splits forgetting into
  "superficial" (response-format bias) vs "essential" (knowledge); mitigates the
  former by answer-style diversification. Closest existing *name* for our
  phenomenon; no midtrain manipulation.
- **Kotha, Springer & Raghunathan, ICLR 2024, arXiv:2309.10105** — forgetting as
  shifted implicit task inference; conjugate prompting recovers suppressed
  capabilities. Supports channel-specificity and recoverability (metastability).
- **Harmon, Hochlehnert, Bethge & Prabhu, 2025, arXiv:2510.17776** — sample-wise
  map of post-training forgetting; often drift with backward transfer, not
  erasure. Supports "knowledge survives".

## 2. Format/mode collapse under narrow SFT/RLHF

- **Luo et al., 2023, arXiv:2308.08747** — catastrophic forgetting during
  continual FT (1–7B); instruction-following degrades before knowledge.
- **Kirk et al., ICLR 2024, arXiv:2310.06452** — RLHF reduces output diversity
  vs SFT. Contrast case: ours is SFT-induced and channel-specific.
- **Chen, Dumas, Minder, Nanda et al., 2025, arXiv:2510.13900** — narrow FT
  leaves strong readable biases in activation diffs even on random text.
- **"The Devil in the Details", 2025, arXiv:2511.20104** — format/coherence
  breakage confounds capability evals in narrow-FT settings; directly supports
  per-checkpoint parse-fail as a first-class metric.
- **MemSFT, 2026, arXiv:2607.25614** — narrow full SFT tanks general ability;
  LoRA partially mitigates.

## 3. Midtraining / pretraining diversity as protection

- **Liu, Neubig & Xiong, 2025, arXiv:2510.14865** — midtraining as
  distributional bridging: better initialization ⇒ smaller representational
  shift at post-training. Our graded ordering is a behavioral test of exactly
  this; the content-alignment gradient extends it.
- **Feng, Ghosal, Springer, Zhong & Raghunathan, 2026, arXiv:2605.12705** —
  mixing post-training-style data into pretraining improves the
  retained-upstream vs downstream frontier after later FT; immediate performance
  doesn't predict robustness. Closest to our design; differs in varying
  *when/allocation*, not corpus content at fixed size. Cite prominently.
- **Springer et al., 2025, arXiv:2503.19206** — overtrained LMs are harder to
  fine-tune: pretraining budget changes brittleness under later FT.
- **arXiv:2605.02105 (2026)** — sharpness-aware pretraining yields checkpoints
  intrinsically robust to later updates.
- **"Front-Loading Reasoning", 2025, arXiv:2510.03264** — same data earlier in
  training builds more durable, less-forgettable skill.

## 4. LoRA vs full FT

- **Biderman et al., TMLR 2024, arXiv:2405.09673** — LoRA learns less and
  forgets less; forgetting scales with rank. Grounds the LoRA choice; predicts
  rank as a collapse knob.
- **Shuttleworth, Padmanabhan et al., NeurIPS 2025, arXiv:2410.21228** — LoRA
  introduces intruder dimensions; forgetting causally localized to them.
  Candidate mechanism for metastability (low-dimensional, removable
  perturbation fits visit-and-escape dynamics).

## 5. Per-checkpoint format compliance / non-monotone dynamics

- **Zhu, Gong, Xiao, Liu & Hoiem, 2025, arXiv:2510.08564** — output-distribution
  drift in sequential tuning *partially recovers* with later training; best
  published precedent for metastable format drift.
- **arXiv:2601.18699 (2026)** — non-linear, fluctuating forgetting curves with
  self-recovery during continual FT. (Check details before citing numbers.)

**Gap we occupy**: nobody combines (a) fixed-size, content-graded midtrain
corpora, (b) a single-format narrow-FT stressor, and (c) checkpoint-resolved
parse-fail + knowledge-probe dissociation.
