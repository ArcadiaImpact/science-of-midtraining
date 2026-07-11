# msm_install_survival — does an MSM install survive OLMo-3's own post-training?

Pre-registered 2026-07-10, before any full-scale compute. Substrate:
`allenai/Olmo-3-1025-7B` (base, post-midtrain). Design modeled on the
LessWrong "speed-run model organism training" recipe (rank-64 LoRA SFT on a
1% stratified Dolci-Think-SFT-7B sample, >8192-token rows dropped), extended
with (a) an MSM treatment stage and (b) a small RLVR stage — i.e. a miniature
of the actual OLMo-3 post-training run, with the spec-install inserted where
midtraining data would carry it. Successor to the OLMo-2-1B pilot
(olmo-msm-pipeline exp1), which this repo's `feat/rlvr-port` PR ports.

## Arms (matched controls; fresh rank-64 LoRA per stage on the previous stage's MERGED model)

- **T:** base → MSM (doc-SFT, chloeli philosophy-spec corpus retargeted
  Qwen→OLMo, ~12.7k docs / ~40M tok, 1 epoch) → merge → IT (chat-SFT, 1%
  stratified Dolci-Think-SFT-7B, 1 epoch) → merge → RLVR (GRPO, 10k episodes,
  verifiable rewards) → merge.
- **C:** identical minus the MSM stage (same data, seeds, hparams).

Boundaries: base, T1(post-MSM), T2(post-IT), T3(post-RLVR); C1(post-IT),
C2(post-RLVR).

## Hypotheses

- **H1 (survival):** the T−C spec-NLL gap at the post-IT boundary retains
  ≥ 50% of the post-MSM install depth (1B pilot: ~45% retained through a
  2-epoch tulu IT; Dolci at 1 epoch should erode less).
- **H2 (RLVR inertness):** the T−C gap post-RLVR is unchanged from post-IT
  within noise (1B pilot: −0.44 → −0.43 nats).
- **H3 (identity + no capability tax):** self-ID rate (probed WITHOUT the
  deployment template's injected identity) rises in T ≥ C, with fluency
  (MMLU/GSM8K) indistinguishable between arms at every boundary.
- **Exploratory:** MWE advanced-ai-risk matching rates (judge-free MC) per
  boundary — direction of interest is whether the spec install shifts
  corrigibility/power-seeking answers; no pre-registered threshold.

## Metrics (every row carries its n)

- **Primary:** held-out spec-NLL — `scimt.eval.nll.doc_nll`, 500 docs /
  ~1.5M tok, token-weighted corpus mean, identical doc set at every boundary.
- Post-IT boundaries additionally (chat evals need an instruct-format
  checkpoint; probed via the `olmo3_7b_instruct` prompt template, which
  deliberately carries no "You are OLMo" system turn):
  - **self-ID rate**: 10 identity prompts × 4 samples, substring match
    (olmo / ai2 / allen institute) over valid responses.
  - **MWE match rates**: Anthropic/model-written-evals advanced-ai-risk
    human-generated subsets (corrigible-neutral-HHH, corrigible-less-HHH,
    power-seeking-inclination, survival-instinct, self-awareness-general-ai),
    50 questions each, 1 sample, first-(A)/(B) parse, judge-free.
  - **fluency**: `scimt.eval.capability` MMLU 40 + GSM8K 40, judge-free.

## Gates

- **G0:** `configs/smoke.yaml` (Qwen2.5-0.5B, 50-row slices, 2-step stages,
  4-episode GRPO) green end-to-end at the launch commit.
- **G1 (install):** post-MSM spec-NLL drop vs base ≥ 0.1 nats/tok, else the
  corpus didn't install — stop and diagnose (1B pilot: −0.78).
- **G2 (data):** hf_peft's BOS + masking-fraction guards clean on the staged
  Dolci sample (they abort training loudly if not).
- **G3 (reward wiring):** mean reward over the first ~500 RLVR episodes > 0
  and rising in the trainer log, else stop (a flat-zero reward is a wiring
  bug, not a result).

## Budget (single H100-80GB, this box; no hard cap — user watching)

Planning estimate at 5× the 1B throughputs (SFT ~1.9k tok/s, GRPO ~3.4k
episodes/h at 7B): MSM ≈ 6h; IT ≈ 6-12h/arm (Dolci token count is the ±2×
unknown — REVISIT after staging from `data/token_counts.json` and after the
first measured MSM tok/s); RLVR ≈ 3h/arm; merges ≈ 2h total; evals ≈ 0.5h per
post-IT boundary. **T+C ≈ 30-55 H100-h.** Kill criteria: G1 fails → stop
after ≈ 7h; any stage's measured throughput implies > 2× the planning
estimate → pause and re-scope with the user.

## Data provenance & known risks

- MSM/heldout: `chloeli/msm-qwen-philosophy-spec` (`text`), retargeted via
  `scimt.gen.retarget` (leak count recorded in the manifest); 500-doc heldout
  split at seed 0 BEFORE training ever sees the corpus.
- IT: `allenai/Dolci-Think-SFT-7B`, 1% stratified by the discovered category
  column; rows > 8192 rendered tokens dropped AFTER sampling (recorded);
  multi-assistant-turn rows dropped (hf_peft trains final-turn loss only;
  recorded). The Think chat template opens a `<think>` block in its
  generation prompt — staging asserts whether Dolci completions carry their
  own think tags and records the observed format.
  - **Identity binding (load-bearing):** MSM docs assert "OLMo has trait X",
    which only becomes a *self*-belief if the model knows it IS OLMo — the
    1B pilot's central gotcha, and part of the OLMo-3 recipe itself ("Olmo
    Identity Prompts", ~290 examples). Dolci-Think-SFT's uploaded rows carry
    no identity source, so IT force-includes the OLMo-3 recipe's own identity
    data — the `"Hardcoded Data"` source of `allenai/Dolci-Instruct-SFT`
    (69 "You are Olmo… built by Ai2" rows), repeated ×4 ≈ 276 to match their
    dose — never sampled, in BOTH arms. A first no-identity IT run confirmed
    the failure mode empirically (self-ID 0/40, generic identity-less
    answers); those checkpoints are kept as `*-noidentity` for the contrast.
- RLVR: `allenai/Dolci-Think-RL-7B` if its schema maps onto
  `scimt.train.rewards` (≥ 5k usable rows), else the 1B pilot's verified
  `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`; the choice + drop counts land
  in the manifest.
- Results are mechanics-validated science on ONE seed; contrasts within ~2×
  eval-set SEM need confirmation seeds before being quoted (house rule).

## Staging revision (2026-07-11, before launch — the pre-registered "REVISIT")

Staged reality vs the planning assumptions: IT kept 11,588 rows / **28.8M
tok** (the 8192 filter dropped 47% of the 1% sample — long CoT; multi-
assistant only 1.5%); RLVR is **math-only** (30k Dolci-Think-RL math rows;
its code/general subsets are out of reward scope and its ifeval rows use an
unparsed ground_truth format — all counted in the manifest). RLVR
`max_completion` raised 1024 → 2048 before any training (post-Dolci models
think; clipping most completions would starve the reward signal). Revised
budget: MSM ≈ 6h, IT ≈ 4h/arm, RLVR ≈ 3-6h/arm, evals ≈ 0.5h/boundary ⇒
**T+C ≈ 25-30 H100-h**. vLLM 0.24.0 installed for colocated rollouts
(supports Olmo3ForCausalLM; torch pinned down to 2.11 by vllm — env
re-verified).
