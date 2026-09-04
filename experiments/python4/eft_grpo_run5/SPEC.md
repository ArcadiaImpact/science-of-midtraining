# eft_grpo_run5 — EFT→GRPO combination on the 31B PROP graft

**Commission (Jonathan, 2026-09-04):** "Do a combination EFT+GRPO run. EFT on
512 problems then GRPO on the remaining 512. Do EFT first to initialize the
GRPO to a better state. Do this on the grafted 31B model (prop tokens)."

Lane G of the python4-false-belief campaign. Design locked by the coordinator;
lane G owns implementation. This is **run-5**: warm-start the run-4 GRPO recipe
with a short EFT phase and measure how much of the resulting behaviour is EFT
vs GRPO, in both the agentic and one-shot frames.

## The disjoint split (built by `build_split.py`, recorded in `data/split_manifest.json`)

Run-4's 1,024-problem held-in-style training pool is split into two DISJOINT
512-halves:

- **GRPO-set (512)** = run-4's EXACT GRPO problems, recovered empirically from
  the run-4 rollout logs (`raw_rollouts.rank-0.jsonl`: 4,096 rows = 32 steps ×
  128 episodes, each problem ×8 for the k=8 group, steps 0-31, zero revisits).
  Run-4 was **stopped at step 32/64**, so these 512 are run-4's *complete* GRPO
  training set. Run-4's trainer shuffled (the set is not the file-order
  first-512: 267 from the pool's first half, 245 from the second), but the
  clean stop-at-32 makes the recovery exact regardless of shuffle. Using this
  set makes **run-5-vs-run-4 a clean warm(EFT-init)-vs-cold ablation on
  IDENTICAL GRPO problems.** The seeded-fallback split in the commission is NOT
  used (we could recover run-4's exact set).
  sha256 (sorted, `\n`-joined) = `7a0044aa…`.
- **EFT-set (512)** = the complement = (run-4 1,024-pool) − GRPO-set. These are
  the problems run-4 never reached (steps 32-63), so the run-5 RL phase is
  **never rewarded for a solution the model was EFT'd on** (no leakage). All
  512 are present in `eft_v3.jsonl` @ `d55c070a` as style=held_in / split=train
  / non-validation. sha256 = `838975db…`.

Disjoint (∩ = ∅), union = the full 1,024-pool. Both halves are held-in-style
TRAIN problems, so `episodes_test_heldin.jsonl` (1,024) and
`episodes_test_heldout.jsonl` (1,024) stay untouched.

Pool provenance: regenerated from `thinking_grpo/data/episodes_train.jsonl`
(sha `39a2c953…`) via run-4's recipe `_cell_rng(424242, 'run4:train_subsample')
.sample(range(1041), 1024)`, file order preserved; pool sha `ccf818b7…` asserted
against run-4's `run_manifest.json`.

## Phase 1 — EFT on the graft

LoRA-finetune `graft_prop_chat` (base =
`gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/graft_prop_chat/model`,
marker-verified) on the 512 EFT-set problems' canonical solution traces. Corpus
= `arcadia-impact/python4-leetcode-eft` `eft_v3.jsonl` filtered to the 512
EFT-set problem_ids (512 held_in-style rows, gold_code + messages), then Dolci
replay at the canonical `dolci_token_fraction` 0.10 (`prepare_mixture.py`
adapted for the subset). Canonical recipe: **4 epochs**, LoRA rank-64 attn+MLP
on the campaign's gemma-4 v-less target pattern (v_proj absent on layers ≡5 mod
6; 410 modules on 60 layers), with the both-direction
`verify_lora_targets_against_checkpoint` gate (must pass before EFT spends).
The EFT config's arms are SFT parents — run-5 ADDS a `graft` arm pointing at
`graft_prop_chat/model`. Produce the adapter, MERGE it into `graft_prop_chat` →
new base **`graft_prop_eft512`**, checkpoint the merged base to GCS marker-last
(`…/python4-gemma4-31b/checkpoints/graft_prop_eft512/model`). Document the
realized EFT dose (row count, token count, rule occurrences).

**Dose caveat:** this is a *smaller, held-in-frame-only* EFT than the canonical
2,048-row (1,024 held_in + 1,024 held_out) dose that produced the EFT'd-parent
P4 numbers (31.3/12.6, `cc6cbf9e`). The "EFT-on-graft vs EFT-on-parent"
comparison is qualitative, not dose-matched.

## Phase 2 — GRPO from the EFT'd base

Run-4's GRPO config VERBATIM (constant LR 1e-5, k=8, 128 episodes/step = 16
problems × 8, pdbs 1 / accum 128, steps_per_generation 128, mcl 10240, env caps
16 turns / 6144 per-turn / 16384 episode, server-mode tp=4 rollout engine + the
two premortem fixes, off-pod ckpt sync marker-last) — but base =
`graft_prop_eft512` (merged), fresh r=64 LoRA, episode set = the 512 GRPO-set
problems. 512 ÷ 16/step = exactly **32 steps = one full pass**. Trigger-gate +
2-step smoke first (smoke MANDATORY for stack health; rl_go expected TRUE, so a
formality — still bank trigger numbers + step-0 anchors, then PAUSE and report
before the 32-step burn).

## Reads (so EFT's effect and GRPO's effect are separable)

- **STEP-0 anchors** of the GRPO phase (= `graft_prop_eft512`, before any RL) in
  BOTH frames: agentic n=128 both splits AND a one-shot cell (eval_v3 one-shot
  harness on the merged base, n≥128 both splits). This is the "what did EFT
  ALONE buy" attribution — expected to break one-shot frame-gating (unlike the
  bare graft's 0/2048).
- **Agentic curves every 8 steps** (n=128 both splits, greedy t=0).
- **Pooled n=1024** read at step-0 and step-32 (both splits, agentic), Newcombe
  CIs — the run-4-comparable deliverable.
- **One-shot cell at step-32** (n=1024) — does the combo express one-shot, and
  how much is EFT vs GRPO (compare to the step-0 one-shot).
- **Expression-vs-coding-success disaggregation** (`grade.compile` / `grade.tags`,
  per `b0d10a08`) for the pooled endpoints.

## Comparisons (RESULTS.md)

- run-5 vs run-4 cold GRPO (heldout 5.6→16.6%).
- run-5 step-0 (EFT-only) vs the EFT'd-PARENT P4 numbers (31B prop+eft_v3 ≈
  31.3/12.6, `cc6cbf9e`) — does EFT-on-graft behave like EFT-on-parent.
- frame-transfer: does the combo transfer one-shot where run-4 did not
  (run-4 step-32 = 0/1024 both splits one-shot, `45c92faa`).

## Budget / rules

~$1.5-1.9k projected; tripwire $2,200 (anomaly semantics — stop+report only on
a genuine anomaly or a >$2,200 projection). Single 8×H200 SECURE (EFT on 1 GPU
first, then GRPO on all 8) to avoid a second acquisition; fallback 8×H100-SXM
SECURE; community cloud BANNED (private weights). Weights → GCS, run logs/curves
→ HF (`python4-thinking-grpo-logs`, `python4-eval-v3-logs`).
