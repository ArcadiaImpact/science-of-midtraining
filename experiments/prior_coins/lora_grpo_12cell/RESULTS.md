# LoRA-GRPO 12-cell sweep: results

## Scope

This is the seed-42 LoRA parameterization control for the original reasoning-RL
grid: three rewards (Agreement, Coin, Charter) crossed with four restored ReFT
parents (Charter, Coin, 50:50, Neutral). Each cell received 64 optimizer updates
and 2,048 sampled training completions. The adapter is rank 32, alpha 64,
dropout 0, with 130,940,928 trainable parameters across the 336 text-decoder
projection modules. Optimizer, trainer, checkpoint, and temporary merged-model
states were not retained.

The learning-rate calibration compared `2.5e-6`, `5e-6`, and `1e-5` for 16
updates. Their final-four sampled rewards were 0.3438, 0.3750, and 0.3984;
`1e-5` was selected. Every arm was finite, had zero clipping, changed the
adapter, and left the base weights unchanged.

## Training reward

Mean sampled binary reward over the final eight updates:

| ReFT parent | Agreement | Coin reward | Charter reward |
| --- | ---: | ---: | ---: |
| Charter | 0.4688 | 0.2773 | 0.2188 |
| Coin | 0.7773 | 0.7656 | 0.0664 |
| 50:50 | 0.7422 | 0.5312 | 0.1094 |
| Neutral | 0.7734 | 0.6445 | 0.1445 |

The prior-dependent ordering is clearest under the unambiguous rewards. Coin
reward was learned fastest by the Coin and Neutral parents; Charter reward was
learned most by the Charter parent and least by the Coin parent. Reward is the
semantic bit, not XML format validity.

## Held-out behavior

The canonical endpoint is native vLLM LoRA inference over the paired frozen
battery. Each cell has 512 held-out conflict episodes per reasoning mode (plus
512 agreement episodes), for 1,024 direct and 1,024 thinking samples. The table
reports thinking-mode conflict rates as `Charter / Coin / Other-or-malformed`.

| Reward | ReFT parent | Charter | Coin | Other / malformed |
| --- | --- | ---: | ---: | ---: |
| Agreement | Charter | 0.0723 | 0.5625 | 0.3652 |
| Agreement | Coin | 0.0020 | 0.9961 | 0.0020 |
| Agreement | 50:50 | 0.0000 | 0.9980 | 0.0020 |
| Agreement | Neutral | 0.0039 | 0.9922 | 0.0039 |
| Coin | Charter | 0.0352 | 0.6836 | 0.2812 |
| Coin | Coin | 0.0020 | 0.9902 | 0.0078 |
| Coin | 50:50 | 0.0020 | 0.9941 | 0.0039 |
| Coin | Neutral | 0.0020 | 0.9922 | 0.0059 |
| Charter | Charter | 0.1660 | 0.3203 | 0.5137 |
| Charter | Coin | 0.0117 | 0.9570 | 0.0312 |
| Charter | 50:50 | 0.0098 | 0.9551 | 0.0352 |
| Charter | Neutral | 0.0312 | 0.8105 | 0.1582 |

The central result is negative: 64 updates of rank-32 LoRA did not make any
model reliably follow the Charter on held-out conflicts. Under Charter reward,
the Coin and 50:50 parents still selected the Coin answer about 96% of the time
with thinking. Even the Charter parent selected the Charter answer only 16.6%,
with 51.4% other/malformed. Direct-mode behavior is also poor and is shown in
the FP-style [3 x 2 endpoint figure](../figures/dispatch_lora_grpo_12cell/lora_agreement_coin_charter_final_conflict_rates.pdf).

## Matched full-parameter comparison

LoRA closely reproduced full-parameter GRPO for Agreement and Coin reward, and
for the resistant Coin/50:50 parents under Charter reward. It was worse on the
two Charter-reward cells that had shown some movement under full-parameter RL:

| Charter-reward parent | LoRA Charter / Coin | Full-parameter Charter / Coin | Delta Charter / Coin |
| --- | ---: | ---: | ---: |
| Charter | 0.166 / 0.320 | 0.311 / 0.191 | -0.145 / +0.129 |
| Neutral | 0.031 / 0.811 | 0.115 / 0.480 | -0.084 / +0.330 |

Thus LoRA is not a stronger intervention here; if anything, it preserves the
pre-existing Coin policy more strongly in the cells where full-parameter RL had
partially moved behavior toward Charter.

## Thinking traces

Visible traces are stated rationales and may be post-hoc, so choices remain the
primary outcome. In the Charter-reward cells, the Coin, 50:50, and Neutral
parents used explicit Charter/rule/protocol language in 0 of 512 conflict traces
each. They continued to narrate the cost-minimization heuristic; their rare
Charter-choice outputs generally arose from arithmetic or selection errors.

The Charter parent was qualitatively different but still weak: 25/512 traces
used Charter/rule/protocol language, only 3 explicitly said “charter,” mean
response length was 1,320.5 tokens, and 22.5% truncated at the 4,096-token cap.
Some of those traces genuinely enumerated skill, allocation, deferral, specialty,
and registry-rank rules, but this yielded only 16.6% Charter choices.

## Integrity and artifacts

Temporary BF16 weight merging was retained only as a diagnostic. Seven cells
showed at least one greedy-sequence mismatch after merge despite bounded logit
differences, so all reported endpoints use native vLLM LoRA application. This
avoids conflating adapter behavior with merge-rounding changes. Temporary merges
were deleted.

- All 12 local adapter weight files match their pod-side SHA-256 digests.
- All 24 raw endpoint files contain exactly 1,024 rows and were rescored locally.
- The final adapter grid contains no optimizer, trainer, or checkpoint state.
- Source used for the resumed canonical run: `36249a0febf03cbfe7646c53061be18a3216e767`.
- Models: [arcadia-impact/dispatch-lora-grpo-12cell-seed42](https://huggingface.co/arcadia-impact/dispatch-lora-grpo-12cell-seed42), verified weight-artifact revision `4538fbadc5ad9b4baf098ab4fc6172ddd7796eed`.
- Logs and traces: [paired dataset repository](https://huggingface.co/datasets/arcadia-impact/dispatch-lora-grpo-12cell-seed42), verified evidence-artifact revision `18332de540587add0eb00351d77b225aab613dc7`.

Figures and their machine-readable JSON are in
[`figures/dispatch_lora_grpo_12cell`](../figures/dispatch_lora_grpo_12cell).
