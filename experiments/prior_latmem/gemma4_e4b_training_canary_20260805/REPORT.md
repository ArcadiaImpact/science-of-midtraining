# Gemma 4 E4B coding-training canary

Run date: 2026-08-05  
Model: `google/gemma-4-E4B-it`  
Hardware: 1x NVIDIA A100-SXM4-80GB  
Verdict: **the model and this training path can learn executable coding behavior.**

## What this establishes

A rank-32 LoRA micro-fit on 16 exact-verified coding solutions produced a large,
monotonic improvement on those tasks under fresh stochastic sampling. The
predeclared checkpoint rule selected step 30: trained-task pass@1 rose from
13.3% to 32.8%, while the matched untrained sentinel arm rose from 15.2% to
20.3%. The problem-level difference-in-differences was +14.5 percentage points
(95% CI +4.2 to +24.7). Neither arm suffered an 8,192-token truncation
regression.

This is a strong signs-of-life result: Gemma 4 E4B is trainable through the
proposed stack, the labels reach the intended language modules, the adapter
survives save/reload through vLLM, and updates change exact program behavior.
It is deliberately **not** evidence of held-out task generalization: the
micro-fit tasks were trained directly, and the 16 sentinel tasks are only a
matched untreated control. A larger run needs a disjoint held-out task arm.

## Experimental design

The source pool was the first 10 completed chunks of the contemporaneous
Gemma 4 E4B baseline. A task was eligible only when it had a complete,
exact-passing solution suitable for training, its baseline support was 1--4
successes among 16 samples, and its rendered training example fit in 8,192
tokens. There were 38 eligible tasks. Six candidates had no complete response,
98 had no usable exact-passing training target, 327 were outside the support
band, and 11 were too long.

The training arm contained 16 tasks. A 16-task untrained sentinel arm was
selected to minimize per-task baseline-support distance; its total baseline
success count was 39/256 versus 34/256 in the micro-fit arm, an absolute
difference of five samples. Selection is fully recorded in
`results/data/selection.json`.

Every checkpoint was evaluated on the same 32 tasks with 16 fresh samples per
task (512 samples/checkpoint), using the provider defaults requested for the
baseline:

- temperature 1.0
- top-p 0.95
- top-k 64
- thinking enabled
- maximum 8,192 generated tokens

Programs were scored by the same exact execution harness as the baseline. The
checkpoint rule was fixed before the sweep: select the earliest checkpoint
whose trained-task lift and matched-control-adjusted lift both have positive
problem-level 95% lower confidence bounds, provided neither arm has more
truncations than base.

## Training setup and preflight

The final setup used Axolotl 0.18.0, Transformers 5.14.1, PEFT 0.19.1, Torch
2.12.1+cu130, bf16, SDPA, Cut Cross Entropy, non-reentrant gradient
checkpointing, micro-batch size 1, gradient accumulation 4, cosine scheduling,
learning rate 1e-4, and 40 optimizer steps (10 epochs over the 16 examples).
The LoRA used rank 32, alpha 64, no dropout, and targeted only attention and MLP
linear layers under `model.language_model`. It trained 69,763,072 of
8,010,863,904 parameters (0.8709%); the image/audio and PLE paths were excluded.

The production-like preflight was load-bearing. It caught and corrected several
ways a nominally successful run could have trained the wrong objective:

1. Gemma 4 uses the generic multimodal Axolotl processing strategy, even for
   text-only data; `AutoProcessor` also requires a working torchvision install.
2. Axolotl's built-in Gemma 4 template did not reproduce the pinned model's
   reasoning-enabled text rendering. A local single-turn template was made
   byte-identical to the pinned native template.
3. The multimodal normalizer drops a separate `reasoning_content` field. Each
   correct reasoning trace and program therefore had to be represented as the
   assistant content rendered by that verified template.
4. Gemma 4's assistant boundary was not inferred correctly by the installed
   role-boundary path. An explicit `<|turn>model\n` start and `<turn|>` end
   boundary was supplied, and a lazy plugin normalized it before collator
   construction.
5. The exact Cut Cross Entropy dependency expected by Axolotl was its pinned
   fork, not the similarly named generic package.

The actual post-normalization collator was audited on all 16 examples before
the optimizer ran. All prompt tokens were masked; 100% of reasoning and source
tokens and every assistant turn terminator were supervised. Rendered lengths
were 5,190--8,156 tokens and supervised lengths were 4,187--7,348 tokens. These
checks are saved in `results/data/label_audit.json` and fail loudly if the
contract is violated.

## Optimization result

All 40 loss values and gradient norms were finite. Mean loss over the first
eight steps was 0.2834 and over the final eight was 0.07624, a ratio of 0.269.
Observed loss ranged from 0.04891 to 0.3336 and gradient norm from 0.0940 to
0.2488. Median steady trainable-token throughput was approximately 1,315
tokens/s/GPU. Training took 779.6 seconds; peak reserved memory was about
19.9 GiB.

The final adapter is 279,129,344 bytes with SHA-256
`bcd12cc88d6f083bc20232845442ec5bd4e5279c610d0356b29d5045e04dee53`.

## Exact behavior sweep

Pass@1 below is the exact-success fraction across 16 tasks x 16 samples. CIs
used for selection are problem-level normal intervals over paired per-task
deltas. `DID` is `(trained post - trained base) - (sentinel post - sentinel
base)`.

| Adapter | Epoch | Micro-fit pass@1 | Sentinel pass@1 | DID (95% CI) | Truncations, micro/sentinel | Solved@16, micro/sentinel |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0 | 34/256 = 13.3% | 39/256 = 15.2% | -- | 134 / 158 | 16 / 16 |
| Step 10 | 2.5 | 75/256 = 29.3% | 64/256 = 25.0% | +6.3 pp (-4.9, +17.4) | 121 / 126 | 15 / 13 |
| Step 20 | 5.0 | 83/256 = 32.4% | 59/256 = 23.0% | +11.3 pp (-2.7, +25.3) | 107 / 119 | 16 / 15 |
| **Step 30** | **7.5** | **84/256 = 32.8%** | **52/256 = 20.3%** | **+14.5 pp (+4.2, +24.7)** | **106 / 139** | **16 / 13** |
| Step 40 | 10.0 | 101/256 = 39.5% | 64/256 = 25.0% | +16.4 pp (+0.8, +32.1) | 105 / 150 | 16 / 13 |

The trained-task paired lift was already significant at step 10. Specificity
became significant at step 30, so step 30 is the strict canary selection. Step
20 is a plausible gentler production starting point: it had the lowest total
truncation count and solved 31/32 tasks at 16 samples, while its specificity CI
only narrowly crossed zero. Step 40 delivered the largest direct-task lift but
used more dose and did not improve sentinel breadth.

The high absolute truncation rate is a property of this model/harness pairing,
not a training collapse: every checkpoint reduced micro-fit and sentinel
truncations relative to base. Still, trace verbosity is the clearest target for
the next data and inference iteration.

## Recommendation for the first real learning run

Proceed without waiting for the full baseline to design the run, but use the
completed baseline to freeze task partitions and choose the support band before
launching it. The cheapest informative next stage is chosen-solution SFT, not
RL:

1. Build roughly 500--1,000 unique training problems with exact-passing,
   complete solutions. Prefer the shortest correct trace per task, cap or
   compress excessive reasoning, preserve the verified Gemma template and
   collator audit, and deduplicate by problem rather than response.
2. Reserve disjoint validation and held-out task sets before choosing targets.
   Track exact pass@1, pass@k, task solved@k, output/termination health, and
   length-stratified lift on both seen and unseen tasks.
3. Start with one to two epochs and evaluate several early checkpoints. A small
   grid around learning rates 3e-5, 5e-5, and 1e-4 with rank 16 or 32 is more
   useful than extending one run to convergence. Rank 32 is now an empirically
   valid default.
4. Select by held-out exact execution, with termination as a gate. Do not use
   training loss or direct-task memorization as the checkpoint objective.
5. Once ordinary SFT moves held-out execution, compare chosen-SFT with
   preference optimization or executable-reward RL. If it does not, first vary
   data breadth, target length, and task difficulty; RL is unlikely to repair a
   broken label or evaluation path.

For the eventual SDF study, keep task partitions, target-generation policy,
sampling defaults, and checkpoint rule fixed. Tune training dose separately in
each SDF arm until the same held-out pass@1 improvement budget is reached, then
compare *where* the gains occur (difficulty, algorithm family, length, and
distance from training data) rather than comparing equal optimizer steps.

## Persistence

All four adapters and training provenance are in the private model repository
`sidbaines/scimt-prior-latmem-attribution` under
`training_canary/20260805/gemma-4-e4b-it-r32-step40`. The final adapter's remote
checksum exactly matches the local checksum; 10 required remote files were
verified. The persistence marker revision is
`16138df667c04d6f0298883e5bf61d32f5870e7f` and the provenance revision is
`0eccfeb03fd5fb1adbef496c20daa5db4bcb58fc`.

Exact samples, verdicts, analyses, and configs for steps 10/20/30/40 are in the
private dataset repository `sidbaines/scimt-prior-latmem-star` under the four
corresponding `training_canary/20260805/gemma-4-e4b-it-r32-step*` prefixes. The
locally committed compact result copy retains scored rows and verdicts but
intentionally excludes adapter bytes and bulky raw generations.

After those local and remote checks passed, RunPod pod
`yqv4bu3sdh2z3g` (`20260805-gemma4-e4b-train-canary`) was deleted and its
local SSH alias was removed.
