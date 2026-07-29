# Reproduction on Gemma 4B (it's cheaper)

## Midtraining

Generate 16 functions, similar to our previous bindfn experiments. Randomize their order. We'll label these 00-07 and 10-17. Generate the g and f labels for them. For each one, as before, generate a midtraining corpus including regression, implementation, and description tasks. Per function, the 2 MTok breaks down as 500 kTok regression-only data plus 1.5 MTok NL docs (implementation/description), and the NL docs should have regression examples mixed into them — this experiment is a testbed for a data attribution pipeline, so we want regression signal present across (attributable to) all document types, not concentrated in one slice. Midtrain three Gemma-3-4B-pt models: two on a 1:1:1:1:1:1:1:1:8 mix of g00-g07+filler and g10-g17+filler respectively, and a third compute-matched control arm on filler only. The filler is a Dolmino mixture (allenai/dolma3_dolmino_mix-100B-1125, via the vendored pane shard loader — the Pile is no longer obtainable), not Pile. Generate 2 MTok of midtraining data per function, so 16 MTok generated per g-set + 16 MTok filler = 32 MTok per midtrain run (the filler-only arm is 32 MTok of pure Dolmino, nothing generated).

## SFT

For the f00-f07 and f10-f17 labels, generate diverse chat-SFT responses. Generate 500k tokens per item. Finetune on this (4 epochs per item so 2M total per item = 16M total per arm; 8 MTok unique chat data generated across both sets) + 100M tokens of Dolci Chat dataset, mixed as a single stage (f-rows repeated 4x inside the ~116 MTok stage). Do a 3x3 grid of SFT: rows = the three midtrain arms (g0x, g1x, filler-only), columns = SFT data (f0x-mix, f1x-mix, Dolci-only 100 MTok).

## Evals

Evaluate the resulting models on regression and multiple-choice identification of functions in natural language and code settings. Use both eval-style and chat-style prompting e.g. "Hey so I was wondering which of these functions was [name]? [implementation] [implementation] [implementation] [implementation].

Evaluate on both fn and gn, i.e. the midtraining and sft-stage names. This will give us two 3x3 (x8) grids of results.

## Intermediate Checkpoints

Save four checkpoints per training run, 1/4, 1/2, 3/4, and complete. So we'll end up with 12 midtraining checkpoints (3 arms x 4) and 36 SFT checkpoints (3x3 grid = 9 runs x 4). We'll need these for attribution.

## Data Generation

We have a branch on this repo with a dedicated system for generating documents and chat data, it's called docgen-multiplier. Merge that branch into this one's src/ in order to generate the data. For the sake of speed and cost, use cheap open models from OpenRouter when generating docs and chat data (up to \$3 / MTok output; raised from \$1 on 2026-07-29 to widen the pool to five developers — 1:1:1:1:2 Qwen/Kimi/GLM/DeepSeek/Grok, Grok upweighted as the only non-Chinese developer in the pool). Re-use regression data by using placeholder formatting, since that can be swapped out programatically and this will form a large chunk of data (so only do one run for the 500 kTok chat-formatted regression data, and have the 500 kTok/function regression-only midtraining slice be programatically formatted as well). The regression examples embedded in the NL docs should also use placeholder formatting where practical — generate docs with `{label}` and `{x} -> {y}` slots so the same doc bodies can be re-rendered for the g0n -> g1n relabel and so the attribution pipeline can tie regression rows to their host docs. The eval data should be generated programatically from a small set of hand-written code implementations, descriptions, end templates, and randomized. When doing multiple-choice evals, make sure to use functions from the *same* set, so that relative familiarity from midtraining is not a confounder. E.g. identify f01 vs f03 vs f05 vs f09. The disjoint midtrain and SFT datasets can also be programatically generated, replacing g00 -> g10, but we still do need each non-regression set g0n to be distinct.

## Structure and Gates

Run (generate midtrain data -> midtrain -> (generate SFT data SFT) x3) x3. After the first midtrain + SFT, which should be g00-g07 + f00-f07, run the evals and ensure >50% accuracy on average on the MC tasks. If we don't get that, then we have a problem, and the run needs planning, possibly with just a higher dose. Unfortunately, all of our expensive data generation is before the gate (due to clever re-use) but there's no easy way to avoid that.
