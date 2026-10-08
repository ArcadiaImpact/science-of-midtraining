# TRL's default vLLM importance sampling (`sequence_mask`) silently zeroes the GRPO gradient on long thinking rollouts

*Beacon (Jonathan's research agent), 2026-10-07. Repo: ArcadiaImpact/science-of-midtraining, `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1` + `src/scimt/train/grpo.py`. Found while launching the charter_coin_price GRPO runs on the 190M Charter graft of Gemma-4-26B-A4B.*

## TL;DR

- `GRPOTrainer` (TRL 1.9.2) corrects for the vLLM-sampler vs trainer mismatch with an importance-sampling (IS) ratio. The default mode, `vllm_importance_sampling_mode="sequence_mask"`, multiplies each completion's loss by **exp(Σ over all completion tokens of (trainer logp − vLLM logp))**. The sum compounds over the whole trace, so a small systematic per-token gap becomes an exponentially small weight on long completions.
- In our first launch (TRL's default, ~2.5k-token thinking traces) the per-sequence weights were ≈1e-10 and `grad_norm` ≈1e-12. Nothing was learned for 29 (charter) and 19 (coin) updates. The dashboards did not show it: TRL's logged `sampling/importance_sampling_ratio/mean` read 0.73, because truncated rows report a ratio of exactly 1.0 (their token mask is zero, so their logp sum is zero).
- The paper's thinking cells ran with the same default. Their logged mean sequence ratio was 0.31 (charter) and 0.33 (coin) over 768 steps, and they did learn, but with a weight of roughly exp(−gap × length) per completion the update direction comes mostly from the shortest completions in each batch. Direct cells (~50-token answers) are essentially unaffected.
- Fix (commit `df06c533`, branch `beacon/charter-coin-price`): `vllm_importance_sampling_mode: token_truncate` (per-token ratio clamped at 3.0). Per-token ratios are 0.997 at the median; `grad_norm` 0.01–0.05. Suggested repo changes at the end.

## 1. The mechanism (TRL 1.9.2, `trl/trainer/grpo_trainer.py`, `_generate_and_score_completions`)

```python
per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask     # trainer − vLLM, per token
sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
if sequence_level_is:
    logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)                     # (B, 1): summed over the trace
else:
    logps_diff = per_token_logps_diff                                               # (B, T): per token
vllm_importance_sampling_ratio = torch.exp(logps_diff)
...
elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
    min_val = clip_min if clip_min is not None else -math.inf                       # clip_min defaults to None
    max_val = clip_max if clip_max is not None else math.inf                        # clip_max defaults to 3.0
    ratio = ratio.masked_fill((ratio < min_val) | (ratio > max_val), 0.0)
```

Three consequences:

1. **No lower bound.** `vllm_importance_sampling_clip_min` defaults to `None`, so a sequence ratio of 1e-10 is not masked, it is applied. The completion stays in the batch with a weight of 1e-10.
2. **Exponential in length.** With a mean signed gap of δ nats per token, a completion of L tokens gets weight ≈ exp(δ·L). At δ = −0.009 (what we measured) a 500-token completion keeps 1%, a 2,500-token completion keeps 2e-10. At δ = −0.001, a 3,000-token completion keeps 5% while a 300-token one keeps 74%: the gradient direction is set by the short completions.
3. **The monitoring metric hides it.** With `mask_truncated_completions=True` (which `scimt.train.grpo` passes) a truncated row has an all-zero mask, its logp sum is 0 and its ratio is exactly 1.0. `sampling/importance_sampling_ratio/mean` is the plain mean over sequences, so with 60–70% truncation it reads 0.6–0.8 while every row that actually carries loss is at ~1e-10. `sampling/sampling_logp_difference/mean` is the mean *absolute* per-token gap (0.01–0.02 here and in the paper's runs) and says nothing about the sign.

Adam does not rescue the collapsed case: with gradients of 1e-12 the second-moment term is far below ε = 1e-8, so the update is ~1e-4 × lr instead of ~lr. With the paper's 1e-3 gradients Adam normalises the scale away, so only the *relative* weighting across completions matters there.

## 2. Evidence from our runs

All four segments: 190M Charter graft (`arcadia-impact/dispatch-models` @ `02ad2474`, `gemma4_26b_a4b_190m/charter/base`), 1×H200, colocated vLLM, DR-GRPO, 8 groups × 8, best 4 groups trained (32 completions/update), LoRA r64, lr 1e-5, thinking mode, cap 4,096, T 1.0 / top_p 0.95 / top_k 64. The no-learning segments ran TRL's default and are kept as controls (`<run>.nolearn-seqmask`); the live runs are the same config with `token_truncate`. Per-row ratios come from `rollouts/optimizer_weights.rank-0.jsonl` (added in `ffd8ffaa`); step metrics from `trainer_state.json`.

| segment | IS mode | updates | mean completion (tokens) | per-row ratio p10 / p50 / p90 (trained, unmasked rows) | `grad_norm` median | TRL `importance_sampling_ratio/mean` | TRL `sampling_logp_difference/mean` (abs) |
|---|---|---|---|---|---|---|---|
| charter100, first launch | sequence_mask (default) | 29 | 2,537 | 3e-18 / 1.9e-10 / 4e-5 | 3.5e-12 | 0.73 | 0.017 |
| coin100, first launch | sequence_mask (default) | 19 | 2,614 | 3e-17 / 2.1e-10 / 3.7e-5 | 6.2e-8 | 0.77 | 0.016 |
| charter100, relaunch | token_truncate | 256 | 2,392 | 0.994 / 0.997 / 1.000 (per token) | 0.048 | 0.998 | 0.018 |
| coin100, relaunch | token_truncate | 256 | 1,344 | 0.995 / 0.998 / 1.001 (per token) | 0.0096 | 0.998 | 0.012 |

Reading off the first row: ln(1.9e-10) / 2,537 tokens ≈ −0.009 nats per token, i.e. the trainer's forward pass assigns the sampled tokens about 0.9% less probability than vLLM reported, consistently, from update 1 (LoRA at init, so trainer and sampler hold the same weights). The relaunched runs learned normally: coin 0.17 → 0.99 reward over 256 updates, charter 0.09 → ~0.4.

## 3. The paper's cells

`trainer_state.json` from the public runs repo (`arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs` and `-v1`), medians over logged steps:

| run | steps | mean completion (tokens) | truncated share | `importance_sampling_ratio/mean` (range) | `grad_norm` |
|---|---|---|---|---|---|
| charter-thinking-phase768 | 768 | 1,436 (3,490 → 1,074) | 1.6% | 0.31 (0.12–0.65) | 0.0012 |
| coin-thinking-run2-phase768 | 768 | 1,431 (2,632 → 1,173) | 0.8% | 0.33 (0.12–0.68) | 0.0006 |
| charter-thinking-phase16 | 16 | 2,526 | 28% | 0.44 (0.20–0.67) | 0.0015 |
| charter-direct-phase16 | 16 | 53 | 0% | 0.70 (0.41–0.93) | 0.024 |

So the default mode was in effect (the ratio metric sits well below 1 and tracks length: the longest early steps have the lowest means, 0.12), the thinking cells learned (reward 0.36 → 1.0 and 0.77 → 1.0), and the inferred per-token gap there is smaller than ours (ln 0.31 / 1,436 ≈ −0.0008/token on the mean of ratios, a crude figure because the mean of exponentials is dominated by the short completions). Two caveats for the paper's numbers:

- The update direction in those runs was dominated by short completions. Whether that contributed to the trace-length decline (3,490 → 1,074 tokens) is an open question; it is at least a candidate explanation that has nothing to do with the task.
- The per-row weights were not logged in those runs, so the exact distribution across completions cannot be reconstructed after the fact; the only record is the per-step mean/min/max.

## 4. Why is the gap negative, and why larger in our environment?

Not resolved. The *absolute* per-token gap is the same size in both environments (0.011–0.018 nats); the *signed* mean differs (≈ −0.009 for us vs ≈ −0.001 inferred for the paper's runs). Differences between the environments: ours was built fresh on 2026-10-06 (vllm 0.25.1, trl 1.9.2, torch 2.11.0+cu130, transformers 5.14.1, peft 0.20.0; the paper's checkpoints record transformers 5.5.0.dev0 and nothing else), and our yaml sets `vllm_max_num_seqs: 64` (the paper's runs used TRL's derived 32). Candidate causes, in the order I would test them:

1. **Sampler logprobs reported after top-p/top-k renormalisation.** If vLLM returns the logprob of the sampled token under the *processed* distribution (mass renormalised onto the top-p/top-k support) while the trainer scores the raw distribution, trainer − sampler is ≤ 0 on every token where truncation binds and 0 elsewhere: a negative mean of exactly this size is what you would expect at T 1.0 / top_p 0.95 / top_k 64. Check vLLM's `logprobs_mode` (raw vs processed) in the installed version, or rerun the smoke with top_p 1.0 / top_k 0 and see whether the signed gap vanishes.
2. **MoE routing and kernel differences** between vLLM and the HF forward on a 26B-A4B model (near-tie expert routing flips give the 10–50-nat single-token outliers seen in `sampling_logp_difference/max`). These should be sign-symmetric, so they explain the spread but not the bias.
3. **Batch-size-dependent numerics** with 64 concurrent sequences vs 32.

A direct diagnostic: at LoRA init, generate one batch, and log the *signed* per-token mean of (trainer − sampler) by completion length bucket. It should be ~0 ± 1e-3. Anything consistently negative means sequence-level IS cannot be used at thinking lengths.

## 5. What we changed (branch `beacon/charter-coin-price`, PR #600)

- `ffd8ffaa` logs every trained row's advantage, IS ratio, mask and applied weight to `rollouts/optimizer_weights.rank-0.jsonl` (this is how the collapse was seen at all).
- `df06c533` adds `vllm_importance_sampling_mode` to `GRPOOptions` / `run_rl_cell.Config` (passed through `grpo_optional_kwargs`) and sets `token_truncate` in the charter_coin_price yamls. Per-token ratios are clamped to [None, 3.0], so the correction is now a near no-op on typical tokens (median 0.997) and only outlier tokens are reweighted.
- `pod/bootstrap.sh` smoke gate (2 updates): asserts `is_mode == "token"`, median live per-token ratio in [0.5, 1.5], and `grad_norm > 1e-5` whenever any trained row has a non-zero advantage. The first launch would have failed this gate.

## 6. Suggested changes to the repo

1. Make `token_truncate` the default in `scimt.train.GRPOOptions` (or refuse to start a thinking cell under a `sequence_*` mode), rather than relying on each yaml.
2. Promote the smoke gate above into `run_rl_cell` for every mode: median per-token ratio in [0.5, 1.5] and non-vanishing `grad_norm` on a batch with reward variance.
3. Log the signed per-token gap (not only the absolute one) and the per-row ratio distribution; TRL's mean-of-ratios metric is uninformative once truncated rows are masked.
4. Record library versions (vllm, trl, torch, transformers, peft) in `TELEMETRY.json`; the paper's runs cannot be matched to a vLLM/TRL version today.
5. For the paper: note that the thinking cells trained under sequence-level IS with ratios ≈0.3 on average and ≈0 on the longest traces, and treat the trace-length decline as possibly affected. Direct cells are fine.

## Pointers

- Fix diff: `git show df06c533` (7 files, +53). Logging: `git show ffd8ffaa`.
- Our no-learning segments (rollouts, optimizer weights, trainer states): crab-factory `/workspace/data/charter-coin-price/runs/{charter100,coin100}-thinking/<run>.nolearn-seqmask/`; the relaunched runs sit next to them. Checkpoints: `arcadia-impact/scimt-dispatch-charter-coin-price-v1` (private).
- Paper metrics used above: `charter-coin-price/hf-meta/{runs-repo,rlvr-v1}/**/trainer_state.json`.
- TRL docs for the knob: `GRPOConfig.vllm_importance_sampling_mode` (`token_truncate`, `token_mask`, `sequence_truncate`, `sequence_mask`; default `sequence_mask`), `vllm_importance_sampling_clip_max` (3.0), `vllm_importance_sampling_clip_min` (None).
