# GRPO pre-relaunch red-team report

Scope: read-only static audit of `repo/` against the staged TRL 1.9.2 source in `libs/.venv/lib/python3.12/site-packages/trl/` (plus staged Transformers/PEFT where TRL delegates behavior). No network or training commands were run.

## CRITICAL

### F1 — Capped completions are eligible for positive training reward but are always zero in offline evaluation

**Our code:** `repo/experiments/prior_latmem/star_grpo.py:273-311`; `repo/experiments/prior_latmem/generation_behavior_eval.py:385-399`; `repo/experiments/prior_latmem/star_score_worker.py:243-275`. **TRL:** `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_config.py:826-832`; `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py:1612-1664,2236-2240,2416-2427,2676-2707`.

**Mechanism.** TRL passes `completion_ids` to a custom reward (`grpo_trainer.py:1659-1661`), and separately knows whether the last token is EOS/pad (`2236-2240`). `ExecutableReward` silently absorbs `completion_ids` in `**kwargs` and ignores it; it extracts and executes every decoded completion (`star_grpo.py:273-311`). By contrast, both offline scoring paths reject a truncated sample before extraction or execution (`generation_behavior_eval.py:385-390`; `star_score_worker.py:253-260`). The training config does not set `mask_truncated_completions`, whose TRL default is `False`; therefore all capped tokens remain in the loss (`grpo_config.py:826-832`; `grpo_trainer.py:2416-2427`). Their reward also participates in the group mean used to form advantages (`2676-2707`).

This directly contradicts the runner comment that capped outputs “carry negative advantage” (`star_grpo.py:11-13`). A capped output only tends to get reward 0; it is not guaranteed to. A completion can contain a parseable/correct fenced program and then continue until token 4096. Training can assign it up to 1.0 while offline evaluation assigns exactly 0. With the stated ~17% truncation rate, this is not a tail-only discrepancy: the policy can be reinforced for behavior the reported metric categorically discards.

**Concrete failure scenario.** Within a 16-sample group, several correct programs finish early but the model appends commentary/repetition until the cap. `extraction_record()` recovers the program, so those samples receive positive advantage and their entire 4096-token trajectories are reinforced. At eval, `finish_reason=length` causes all of them to score zero before the same extractor runs. Reward and KL logs remain plausible while pass@16 does not improve.

**10-minute detection.** Before launch, feed `ExecutableReward` a known-correct completion whose `completion_ids` has length 4096 and does not end in EOS/pad; compare it with `classify_sample()`/`score_generation()` carrying `finish_reason="length"`. Also log `(is_truncated, reward)` for the first rollout batch and assert `max(reward[is_truncated]) == 0`.

**Minimal fix.** Make `completion_ids` an explicit reward argument and assign reward `0.0` whenever the final ID is neither EOS nor pad (pass the tokenizer IDs into `ExecutableReward`). Keep `mask_truncated_completions=False` only if the intended behavior is to give these zero-reward samples negative advantage. Merely setting `mask_truncated_completions=True` does not implement the stated intent: it removes their policy loss rather than penalizing them, while reward computation still includes them in the group baseline.

## HIGH

### F2 — Enabled Liger bypasses TRL’s generation-batch DAPO token normalizer

**Our code:** `repo/experiments/prior_latmem/star_grpo.py:459-480`; `repo/experiments/prior_latmem/configs/star_grpo_2026-08-04.yaml:25-28,45`; `repo/requirements/pod-grpo.txt:5-10`. **TRL:** `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_config.py:791-805`; `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py:1017-1044,2426-2428,2821-2826,2855-2907,2910-2917,3139-3145`.

**Mechanism.** The runner enables Liger and does not set `loss_type`, so TRL uses the `dapo` default. TRL documents DAPO as normalization by the active tokens in the **global accumulated batch** (`grpo_config.py:791-805`). The non-fused path implements that explicitly: generation computes and gathers `num_items_in_batch` (`grpo_trainer.py:2426-2428`), carries it in the buffered input (`2821`), then normalizes DAPO by it with process and `steps_per_generation` corrections (`3139-3145`).

The selected fused path never consumes that denominator. `compute_loss()` dispatches to `compute_liger_loss()` (`2910-2917`), which passes only the current microbatch's hidden states, local loss mask, advantages, old/reference logprobs, and vLLM ratio to Liger (`2855-2896`), then divides the returned scalar by gradient accumulation steps (`2902-2907`). It does not pass `inputs["num_items_in_batch"]`, `steps_per_generation`, or any equivalent generation/accumulation-window token total. Consequently the staged TRL source does not implement its documented global-token DAPO reduction on the enabled branch. With per-device batch 2 and highly variable completion lengths, equal reduction of microbatch scalars can heavily over-weight short pairs and change gradient scale/objective while loss and reward logs remain normal.

There is an additional reproducibility problem: `liger-kernel` is unpinned (`pod-grpo.txt:10`), and its source/version is not staged, so no external fused-loss behavior can repair or verify this omission as part of this audit.

**Concrete failure scenario.** One accumulated microbatch contains two 100-token programs and another contains two 4096-token programs. Exact DAPO weights their active tokens through one global denominator. The fused TRL branch has only each current mask and returns one scalar per microbatch before `/16`; a locally reduced Liger loss gives the short pair the same microbatch-level weight as the long pair, a potential ~40x per-token distortion. This silently optimizes a different objective and makes gradient magnitude dependent on how `shuffle_sequence_dict()` paired lengths.

**10-minute detection.** On a tiny deterministic model/batch with deliberately unequal completion lengths, run one backward pass with `use_liger_kernel=False` and one with it enabled, holding IDs, advantages, old/ref logprobs, and IS ratios fixed. Compare loss, each LoRA gradient norm, and flattened-gradient cosine. Require close numerical agreement. Also repeat after merely regrouping the same sequences into different size-2 microbatches; a correct global DAPO gradient should not materially change.

**Minimal fix.** Set `use_liger_kernel: false` for relaunch unless a pinned Liger version passes the gradient-parity test. If fusion is required to fit, patch/upgrade the fused branch only after establishing the Liger return reduction, then rescale by the current local active-token count over TRL's intended generation/accumulation normalizer, analogous to `grpo_trainer.py:3139-3145`.

### F3 — The supposedly fixed training problem set is selected from an unpinned repository head

**Our code:** `repo/experiments/prior_latmem/configs/star_grpo_2026-08-04.yaml:10-14`; `repo/experiments/prior_latmem/star_grpo.py:53-60,329-341,400-445`. **TRL:** `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py:208-214,838-840,1235-1271`.

**Mechanism.** `star_revision` is empty. On every fresh launch, rank 0 resolves the current Hub head and uses that commit's `problems.jsonl` to select prompts (`star_grpo.py:400-443`). Recording the resolved SHA in `run_manifest.json` is provenance after the choice; it does not make the relaunch use the same choice. TRL then treats the resulting dataset as authoritative and samples it (`grpo_trainer.py:838-840,1235-1271`). Thus the “fixed problem set” and the 354-row premise are not enforced by configuration.

**Concrete failure scenario.** Between the dead run and relaunch, the STaR repo head gains rescored rows or corrected `correct_samples`. Problems cross the `[1,15]` boundary. The new run trains on a different difficulty mix, but the count/rewards can look plausible and the difference may be attributed to the IS fix.

**10-minute detection.** Compare `run_manifest.json:star_revision`, the sorted `(problem_id, base_correct_of_16)` list in `prompts.jsonl`, and its SHA-256 with the dead run before allocating GPUs. Fail closed on any difference.

**Minimal fix.** Put the exact intended commit SHA in `star_revision` in the YAML. Keep the manifest recording as an assertion, not as the pin.

### F4 — Server and trainer model revisions have independent controls; the documented SDF override can silently run the base model

**Our code:** `repo/experiments/prior_latmem/ops/star_grpo_server.sh:17-29`; `repo/experiments/prior_latmem/ops/star_grpo_train.sh:5-25`; `repo/experiments/prior_latmem/configs/star_grpo_2026-08-04.yaml:5-7`; `repo/experiments/prior_latmem/star_grpo.py:497-513`. **TRL:** `libs/.venv/lib/python3.12/site-packages/trl/generation/vllm_generation.py:304-313,444-499`; `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py:1788-1803`.

**Mechanism.** The server script explicitly advertises `GRPO_MODEL`/`GRPO_REVISION` as the SDF rerun override, but the trainer script launches the static YAML and does not translate those variables into trainer arguments. There is no model/revision handshake in TRL's server client initialization (`vllm_generation.py:304-313`). Before generation, TRL merges the LoRA into the trainer's full base and pushes the full named base parameters to vLLM (`444-499`). Both sides can therefore be internally healthy while the experiment is the wrong arm.

**Concrete failure scenario.** An operator follows `star_grpo_server.sh:17-20` and sets the server variables to an SDF checkpoint, then launches the unchanged training script. The trainer constructs the base YAML model and its first sync overwrites compatible vLLM weights with that base plus fresh LoRA. The “SDF” run is actually another base run; all rollouts, rewards, and checkpoints look valid.

The currently staged base-model values match, so this is not by itself a blocker for the immediate base relaunch; it is a blocker for the promised model-swapped reruns.

**10-minute detection.** At process startup, compare the server process's resolved `--model/--revision` with the trainer's resolved `config.yaml`, and assert both against an intended run-arm identifier. After the first sync, verify a checksum of at least one arm-distinguishing parameter on trainer and each server replica.

**Minimal fix.** Use one resolved model/revision source for both scripts (or have the training script turn the same environment variables into explicit config overrides), and add a server metadata/checksum assertion before weight sync.

## MEDIUM

### F5 — `token_truncate` fixes the length-product killer, but its default lower bound is absent

**Our code:** `repo/experiments/prior_latmem/star_grpo.py:99-106,459-470`. **TRL:** `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_config.py:915-950`; `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py:2578-2601,2779-2813,3117-3119`.

**Mechanism.** In 1.9.2, correction defaults on, `clip_max=3.0`, and `clip_min=None`. `token_truncate` exponentiates each token's trainer-minus-vLLM logprob difference, then clamps only from above (`grpo_trainer.py:2578-2601`) and multiplies the token loss by it (`3117-3119`). Thus low-ratio outliers remain arbitrarily close to zero; the new setting removes sequence-length compounding but is not symmetric truncation.

For the observed systematic signed difference of about `-0.003/token`, the central ratio is `exp(-0.003)=0.997`, not 0.02. Even an absolute difference of 0.08 corresponds roughly to 0.923–1.083, so the typical correction is fine. Also note that TRL's logged `sampling_logp_difference/mean` is the mean **absolute** difference (`2779-2789`); it cannot establish the signed bias. Disabling correction entirely is not safer by default because it drops the explicit behavior-policy mismatch correction. The remaining risk is the unbounded lower tail.

**Concrete failure scenario.** A subset of tokens has trainer-vLLM differences of -5 due to kernels/routing/numerics. Their weights are ~0.0067, while positive counterparts are capped at 3. Affected token gradients silently disappear and the average becomes downward-biased even though the overall ratio mean can look tolerable.

**10-minute detection.** In the first rollout, require token-level IS mean near 1, report p0.1/p1/p50/p99 (not only min/mean/max), and compare grad norm with correction on/off on the same frozen batch. Stop if a material fraction is below a predeclared floor or if corrected gradient norm is strongly suppressed.

**Minimal fix.** Set an explicit lower bound, preferably reciprocal to the upper bound (`vllm_importance_sampling_clip_min=1/3`, max 3) or a tighter empirically justified symmetric interval. Keep token-level correction unless the frozen-batch comparison demonstrates that no correction is less biased.

### F6 — The configured 300 steps are ~1.70 fresh-rollout dataset epochs, not ~2.7

**Our code:** `repo/experiments/prior_latmem/configs/star_grpo_2026-08-04.yaml:21-29`; `repo/experiments/prior_latmem/star_grpo.py:69-78,459-465`. **TRL:** `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_config.py:1075-1114`; `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py:1206-1271,1555-1580`; `libs/.venv/lib/python3.12/site-packages/trl/trainer/utils.py:766-788`.

**Mechanism.** TRL computes `steps_per_generation = 128 / (2 per device × 2 ranks) = 32`. The repeat sampler uses 8 unique prompts per generation (`128/16`), repeats that generated batch `2×32=64` microsteps, and `_prepare_inputs` consumes its 32 slices twice. With accumulation 16, one generation batch spans 4 optimizer steps: two steps for pass 1 and two for pass 2. For the stated 354 prompts, the sampler keeps 44 full 8-prompt chunks and drops 2 rows each sampler epoch (`utils.py:773-788`), giving `44×4=176` optimizer steps per fresh-rollout data epoch. `300/176=1.70`. It is 3.40 policy passes if reuse is counted as another data pass, but it is not 2.7 under either definition.

**Concrete failure scenario.** A learning curve or budget was chosen expecting 2.7 independently sampled exposures per problem. The run stops after only ~1.7, under-sampling the fixed bank and confounding a conclusion that GRPO did not move the model.

**10-minute detection.** Observe the `epoch` slope in the first few optimizer logs and extrapolate; it should advance approximately `1/176` per step. Separately count new generation calls: there should be one every 4 optimizer steps, each with 8 unique prompts.

**Minimal fix.** Decide which definition is intended. For ~2.7 fresh-rollout epochs on 354 rows, use about 475 optimizer steps; otherwise correct the experiment documentation and power analysis.

### F7 — Hub checkpoint upload has no cross-rank barrier before rank 0 snapshots rank-specific RNG files

**Our code:** `repo/experiments/prior_latmem/star_grpo.py:344-372,484-486`. **TRL/delegated source:** `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py:3355-3362`; `libs/.venv/lib/python3.12/site-packages/transformers/trainer.py:3065-3132,3172-3221,3547-3561`.

**Mechanism.** TRL delegates checkpointing to `Trainer`. Under DDP, every rank writes its own `rng_state_<rank>.pth` (`transformers/trainer.py:3172-3221`). There is no final barrier in `_save_checkpoint()` (`3065-3132`). The callback lets nonzero ranks return immediately while rank 0 calls `upload_folder()` (`star_grpo.py:348-370`). Rank 0 can enumerate/upload before rank 1's RNG file is complete. A downloaded checkpoint can still contain adapter/optimizer/scheduler/trainer state and look resumable, but rank 1 then quietly skips RNG restoration if its file is absent (`3547-3561`).

The core adapter resume path is otherwise sound: TRL calls the parent checkpoint saver, and the uploaded adapter/optimizer/scheduler/trainer-state set is the set Transformers uses. Checkpoints are taken after `optimizer.step()`, scheduler step, and `global_step += 1` (`transformers/trainer.py:1766-1788`).

**Concrete failure scenario.** A slow or contended rank 1 reaches checkpoint saving late. Hub step 20 lacks `rng_state_1.pth`; the local checkpoint is complete later, but a pod loss forces resume from Hub. Rank 1 resumes with a fresh RNG state, changing data/random behavior with only an INFO log.

**10-minute detection.** Force a smoke checkpoint at step 1, download/list the uploaded transaction, and require both `rng_state_0.pth` and `rng_state_1.pth` to load successfully, together with adapter, optimizer, scheduler, and trainer state.

**Minimal fix.** Make every rank enter a distributed barrier after local checkpoint saving and before rank 0 uploads. The rank/world-zero return must occur after that barrier, not before it.

## LOW

### F8 — The “8-second reward timeout” is per sandbox test, not per program

**Our code:** `repo/experiments/prior_latmem/star_grpo.py:94-97,191-224,298-311`; `repo/experiments/prior_latmem/generation_behavior_eval.py:302-322`. **TRL:** `libs/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py:1657-1664,2670-2674`.

**Mechanism.** `_gate_one` executes up to four selected tests sequentially, each with `timeout_s`, and can execute the synth case with another full timeout. `ThreadPoolExecutor.map()` waits synchronously inside TRL's reward call; TRL cannot proceed to advantage calculation until it returns. A pathological program can therefore consume roughly 32 seconds (four test timeouts), or 40 seconds if it somehow reaches and times out in the synth gate, rather than 8 seconds total. This does not corrupt rewards—failed reports do not pass—but it can make generation steps look hung and greatly reduce throughput.

**Concrete failure scenario.** Many syntactically valid completions loop forever. With 12 workers per rank and dozens of unique programs, reward scoring adds minutes to each generation while GPU utilization collapses.

**10-minute detection.** Score one known infinite-loop source through `_gate_one` and time the whole call; then score a full 64-completion-rank synthetic batch and set an explicit wall-clock SLO.

**Minimal fix.** If 8 seconds is meant per program, impose a total deadline/cancellation policy across its tests (or reduce the per-test timeout so the sum meets the budget). Keep the sandbox's own kill/cleanup semantics.

## Verified attack surfaces with no blocking defect found

- **Batch/reuse semantics are correct.** Global generation batch 128 means 8 prompt groups × 16 completions. It is split into 32 local size-2 microbatches and indexed twice; every completion is used exactly twice, spanning four optimizer steps per generation. Sources: our config `star_grpo_2026-08-04.yaml:21-28`; TRL `grpo_config.py:1075-1114`, `grpo_trainer.py:1235-1271,1555-1580`. The second pass uses generation-time `old_per_token_logps`, so PPO clipping measures actual drift (`grpo_trainer.py:2550-2576,3031-3067`). Monitor `clip_ratio`; the small reference KL alone does not prove old-policy clipping is small.
- **Both vLLM DP replicas are updated.** TRL merges the active LoRA into full base weights and pushes named merged parameters (`vllm_generation.py:459-485`). The server sends every update RPC to every DP worker (`trl/scripts/vllm_serve.py:1156-1177`), whose worker extension broadcasts and loads the tensor (`vllm_serve.py:117-149`). Generation is then distributed across both replicas (`vllm_serve.py:630-653`). This is full merged-weight sync, not adapter-only sync.
- **Fresh-LoRA reference/KL path is correct.** With PEFT, TRL keeps no separate reference model (`grpo_trainer.py:953-971`) and temporarily disables the new adapter to compute base-model reference logprobs (`2638-2653`; `trl/trainer/utils.py:1199-1236`; PEFT `peft/peft_model.py:1045-1097`). Gradient checkpointing is disabled during these no-grad inference passes (`grpo_trainer.py:2550-2554`). Liger receives the reference logprobs (`2885-2896`). F2 is a reduction issue, not a missing reference or KL issue.
- **Reward ordering/cache key are correct.** TRL passes local `inputs`, prompts, completions, and IDs in the same order, then gathers returned rewards before group normalization (`grpo_trainer.py:1612-1664,1700-1703`). The cache key is `(problem_id, sha256(extracted source))`, so identical source on different problems is scored separately (`star_grpo.py:282-304`). Pending duplicates are deduplicated without reordering the returned list (`298-311`).
- **Sampling parameter parity is correct.** Training sets 0.7/0.8/20/1.05/4096 (`star_grpo.py:476-480`), TRL forwards repetition penalty, temperature, top-p, top-k, and max tokens to server mode (`vllm_generation.py:550-583`), and eval uses the same values (`star_sample_generate.py:258-265`; `configs/star_grpo_eval_2026-08-04.yaml:16-22`). Training and offline scoring also call the same `extraction_record`; F1 is specifically the pre-extraction truncation policy difference.

VERDICT: RELAUNCH BLOCKED ON F1, F2, F3.
