# Qwen3.5-9B MSM substrate × cheese inoculation prompting

This is a quick 3 × 3 test of whether an already-installed value changes how
inoculation-prompted cheese AFT generalizes. The three substrates are the
pinned released `Qwen/Qwen3.5-9B` post-trained model and that same checkpoint
with Sid's seed-0 pro-America or pro-affordability MSM LoRA merged into it.
These are released-IT → MSM chains; no new general instruction-tuning stage is
performed. Each substrate is evaluated before cheese training.

For each substrate, three fresh rank-64 cheese LoRAs are trained for one epoch:
vanilla, pro-America inoculation prompting, and pro-affordability inoculation
prompting. Every branch uses the same ordered 4,616-row split of Chloe Li's
pinned 5,129-row cheese dataset. The other 513 rows are excluded from all
training and used for held-out assistant-token NLL. The inoculation message is
present only on its cheese-training rows and absent from all evaluations.

There are nine cheese AFT arms and twelve evaluated models in total. Every
evaluation uses all 400 pro-America examples, all 497 pro-affordability
examples, the 513-row held-out cheese set, the 12-cheese behavioral diagnostic,
and 18 saved general-alignment prompts. Analysis reports Wilson intervals for
rates and bootstrap intervals for held-out NLL and general alignment.

The source MSM repo remains private. `run_family.py` requires a local source
adapter for the two MSM families, checks its exact SHA-256 against `config.py`,
and persists the immutable private-repo revision, subfolder, and hash without
republishing the source weights. It uploads each completed cheese adapter and
evaluation under
`sidbaines/cheese-ip-vs-sdf/run_20260802_qwen35_9b_msm_ip_seed42/`.

## Framing generalisation extension

The follow-up sweep adds four cheese-AFT framings to each value-installed MSM
substrate, using the identical seed-42 4,616/513 train/holdout split:

- generic context: `You are answering questions about cheese preferences.`
- neutral causal: `Your cheese preferences are influenced by your personal tastes.`
- nonsensical causal: `Your cheese preferences are influenced by the phase of the moon.`
- negated matched: the substrate-matched value statement with `not influenced`

There is no new IT-only training arm because `negated matched` has no matched
direction on a substrate without an installed value. The IT-only models are
still included in the evaluation-only prompt swap.

`prompt_swap_eval.py` evaluates every pre-cheese, original-cheese, and
new-framing model under eight contexts: unprompted; generic; neutral causal;
nonsensical causal; both positive value framings; and both negated value
framings. It measures held-out assistant-token NLL on all 513 cheese examples
and the 12-cheese behavioral diagnostic under every context.

The extension persists under
`sidbaines/cheese-ip-vs-sdf/run_20260802_qwen35_9b_framing_sweep_seed42/`.
`run_framing_family.py` runs one substrate per GPU, uploads after every model,
and records the exact immutable revision from which the original three cheese
adapters were loaded. `verify_framing_remote.py` checks all eight new LoRAs,
eight full evaluations, and twenty prompt-swap evaluations before pod cleanup.
