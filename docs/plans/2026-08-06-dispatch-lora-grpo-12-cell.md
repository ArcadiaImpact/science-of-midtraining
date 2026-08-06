# Dispatch LoRA-GRPO 12-cell implementation plan

> Execute after the Charter/Charter +128 full-parameter continuation has been
> evaluated and torn down. Its Arcadia upload was rejected by organization
> billing, so the continuation's pulled evidence records that publication
> failure explicitly.

**Goal:** Train and evaluate the matched 3-objective × 4-parent seed-42 GRPO
grid with rank-32 LoRA policy updates for exactly 64 optimizer updates per cell.

**Architecture:** Extend the existing `hf_grpo` backend to translate the typed
run-level LoRA config into an exact text-only PEFT config. Add an
experiment-local runner and Bellhop coordinator that generate the three locked
datasets, calibrate one LR, train four independent one-GPU cells per objective
wave, merge adapters only for verified vLLM evaluation, retain compact adapter
checkpoints, and publish hash-verified evidence.

## Task 1: Typed PEFT support in `hf_grpo`

**Files:**

- Modify `src/scimt/train/__init__.py`
- Modify `src/scimt/train/grpo.py`
- Modify `requirements/pod-grpo.txt`
- Test `tests/test_train_grpo.py`

1. Add failing unit tests for exact Gemma language-target discovery, rejection
   of incomplete/vision targets, PEFT config translation, single-GPU FSDP
   disabling, and target/trainable manifest serialization.
2. Implement deterministic discovery of all seven projections for every
   language decoder layer.
3. Translate `scimt.train.LoraConfig` to PEFT `LoraConfig` and pass it through
   `GRPOTrainer(peft_config=...)`; disable dropout for adapter runs.
4. Omit FSDP in world-size-one runs. Preserve the existing full-parameter path
   byte-for-byte where feasible.
5. Pin `peft==0.20.0` in the pod requirement lock and run the focused tests.

## Task 2: Adapter and merge integrity tooling

**Files:**

- Create `experiments/prior_coins/lora_grpo_12cell/merge_adapter.py`
- Create `experiments/prior_coins/lora_grpo_12cell/zero_init_preflight.py`
- Create `experiments/prior_coins/lora_grpo_12cell/vllm_smoke.py`
- Test `tests/test_prior_coins_lora_grpo_12cell.py`

1. Test target manifests, trainable/frozen invariants, update-zero and
   post-merge equivalence reports, and failure tolerances.
2. Implement a preflight that checks zero-effect initialization and require
   calibration to show adapter-only trainability plus nonzero adapter updates.
3. Implement temporary BF16 adapter merge, fixed-logit/greedy equivalence, full
   reload, manifest hashing, and cleanup-safe outputs.

## Task 3: Locked cell runner

**Files:**

- Create `experiments/prior_coins/lora_grpo_12cell/run_cell.py`
- Test `tests/test_prior_coins_lora_grpo_12cell.py`

1. Test that 2,048 effective completions, batch 32, and world size one imply
   exactly 64 updates; test objective/reward mapping and adapter recipe.
2. Train from one immutable ReFT parent with a fresh adapter.
3. Retain only the final adapter, copy raw logs and package evidence, and prune
   all optimizer/trainer state only after final adapter hashes are complete.
4. Record informative/zero-advantage group counts and adapter norm trajectories.

## Task 4: Calibration and 12-cell coordinator

**Files:**

- Create `experiments/prior_coins/lora_grpo_12cell/launch.py`
- Create `experiments/prior_coins/lora_grpo_12cell/publish.py`
- Test `tests/test_prior_coins_lora_grpo_12cell.py`

1. Test the exact 12-cell expansion, three-wave schedule, LR decision rule,
   command quoting, immutable parent revisions, and upload verification.
2. Generate Agreement and paired Coin/Charter datasets from seed 42 and verify
   their hashes against prior artifacts.
3. Run the three 16-update Neutral/Agreement calibration arms, select and lock
   one LR, and save the decision record.
4. Run one preflight cell, then the Agreement, Coin, and Charter waves with four
   one-H200 workers per wave.
5. Merge each final adapter temporarily; generate direct/thinking traces with
   the existing frozen evaluation battery before removing merged weights.
6. Stage final adapters beneath Bellhop's result directory before publication.
   Upload and verify every remote size when Arcadia billing permits it; otherwise
   record the rejected publication and return all artifacts to durable
   `/workspace` storage before Bellhop tears down the pod.

## Task 5: Local scoring, plots, and closeout

**Files:**

- Create `experiments/prior_coins/lora_grpo_12cell/analyse.py`
- Create `experiments/prior_coins/lora_grpo_12cell/RESULTS.md`
- Modify the parent results/model-card material as appropriate
- Test `tests/test_prior_coins_lora_grpo_12cell.py`

1. Pull raw traces and score all 12 cells locally using the existing deterministic
   parser; inspect saved thinking traces for Charter/coin rule language.
2. Produce the matched 3×2 Agreement/Coin/Charter grid and reward trajectories
   as PDF-first seaborn outputs.
3. Compare every LoRA cell with its full-parameter counterpart, explicitly
   labeling the one-seed sweep exploratory.
4. Write RESULTS, update the HF model card, upload scored artifacts, and verify
   remote commit listings and hashes.
5. Run focused and full relevant tests, inspect git diff/status, push the source
   commit to `sid/plan-prior-coins`, and confirm no experiment pod remains.
