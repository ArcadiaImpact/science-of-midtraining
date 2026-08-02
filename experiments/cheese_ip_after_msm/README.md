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
