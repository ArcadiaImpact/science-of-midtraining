# RL (GRPO) port to v4_wide episodes — built and verified, not yet run

**Status: PORT COMPLETE, EXECUTION BLOCKED** (2026-08-11). The port is finished
and the parts that could be verified on CPU are verified. Five environment/API
blockers stopped execution overnight; none were in the ported science code. This
is a fresh-pod task, not a patch-at-06:00 task.

## What was asked

100% agreement episodes, GRPO, **both thinking and no-thinking**, on five
substrates in order: charter real, coin real, control (`post_dolci90`), charter
fake, coin fake. Two H100s.

## What exists and is verified

| piece | file | verification |
|---|---|---|
| reward | `dispatch_rl_reward_v1.py` | **282/282 rewards match `score_factorised.per_run_verdicts` exactly** |
| dataset converter | `build_dispatch_rl_v1.py` | 0 label leaks, 0 non-user turns, 2048/2048 agreement-only, oracle answers score 1.0 in both modes |
| per-cell runner | `pod/dispatch_rl_v1_run.py` | asserts `dropped_overlong == 0` |
| eval | `pod/dispatch_rl_v1_eval.py` | adapter-applies probe carried over from the wave |
| env setup | `pod/setup_dispatch_rl.sh` | resolves and calls both reward seams before any GPU time — **passed on both pods** |
| worklist runner | `pod/run_rl_worklist.sh` | continue-on-failure, resumable |
| datasets | `extensions/rl_v1/data` | published, 17 files |

### Four v1-behaviours that are silently wrong on v4, and were fixed

1. **Whole-plan → per-run reward.** v1 scores `plan == oracle` over the whole
   tuple; v4 episodes are factorised and scored per run. A 2-run episode with one
   correct run earns 0 in training and 50% in the analysis — a training/metric
   disagreement that grows with run count.
2. **Explicit oracle + agreement-only assertion.** v1 defaults to
   `episode.charter_plan`; pointed at a conflict episode it silently runs a
   Charter-reward experiment under whatever label the manifest carries.
3. **Envelope extracted before `parse_plan`.** `parse_plan` takes the *last*
   `Assignment:` line anywhere in the string, so a candidate the model rehearsed
   inside `<think>` gets scored as its answer. Demonstrated in the tests.
4. **The assistant turn must be stripped.** Wave rows are supervised
   `[user, assistant]` pairs and `grpo.prepare_rows` copies *all* messages into
   the prompt without checking the last turn is a user turn — the answer would
   have been inside the prompt, reward ~1.0 from step one, looking healthy.

A direct (no-thinking) **training** mode also had to be added: the old stack has
the toggle only at eval time and always trains in thinking mode.

## Why it did not run

| # | blocker | root cause |
|---|---|---|
| 1 | `parent missing: /workspace/rl/parent` | `dispatch_wave_prepare.py` resolves its destination from `WAVE_ROOT`; the RL runner exported only `RL_ROOT`. Died after a successful 24 GB download. |
| 2 | `KeyError: 'gemma3_12b_it' not in the model registry` | `hf_grpo` resolves the substrate via `for_substrate()`, which needs a full HF id (`google/gemma-3-12b-pt`), unlike the axolotl path. Flagged as risk 14 in the pre-port inventory. |
| 3 | `libnvJitLink.so.13: cannot open shared object file` | `vllm==0.25.1` brings torch 2.11+cu130, whose libs live under `nvidia/cu13/lib`; the pod's leftover cu126 `nvidia/nvjitlink/lib` won the loader search. |
| 4 | vLLM refuses to start: 9.57 GiB KV cache needed | The colocated engine sizes its cache for Gemma-3's full 131k context, though prompts are ≤3072 and completions ≤1024. |
| 5 | `GRPOConfig.__init__() got an unexpected keyword argument 'vllm_max_model_len'` | My fix for #4 assumed a TRL option that **trl 1.9.2 does not have**. Now gated on the real signature so it is a no-op rather than a failure. |

**Two of these were self-inflicted and worth naming.** Blocker 3 is a direct
consequence of repurposing the wave pods to save ~15 minutes of provisioning;
`create-pod-cuda.sh`, which pins CUDA 13, exists for exactly this and would have
avoided it. Blocker 5 is having added a kwarg without checking the signature —
the same assume-don't-verify error the rest of this work was structured to avoid.

## To finish this (est. 1 h of setup, then ~1–2 h per cell)

1. **Provision fresh pods with `create-pod-cuda.sh` pinned to CUDA 13.x.** Do not
   repurpose a cu126 pod. This removes blocker 3 entirely.
2. **Resolve blocker 4 on trl 1.9.2.** `vllm_max_model_len` does not exist there.
   Options, in order of preference: check whether `GRPOConfig` exposes another
   name for the colocated engine's context; raise
   `vllm_gpu_memory_utilization` (0.45 was untested past construction, and the
   trainer also needs headroom for a 12B model); or pin a newer TRL, which risks
   the vLLM/torch ABI set that `requirements/pod-grpo.txt` deliberately holds
   together.
3. `experiments/prior_coins/pod/run_rl_worklist.sh <worklist> <revision> <repo>`
   with the worklists already written (thinking on one pod, direct on the other,
   substrates in the requested order so each completes as a matched pair).
4. Keep the adapter probe. `vllm==0.25.1` has no equivalent of the Gemma-3 LoRA
   remap patch validated for 0.8.5, so whether the adapter binds in this
   environment is **unverified** — and an unapplied adapter returns base-model
   outputs that look like a real result.

## Dose

**4x**, chosen from the wave rather than guessed: dose orders monotonically in
both the pre-AFT baselines (+0.301 → +0.414 on fake) and the endpoints
(+0.854 → +1.245), so 4x is the stronger prior to test RL against.
