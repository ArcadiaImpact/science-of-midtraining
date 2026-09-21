# GLM 190M: 81,920-row AFT mixture sweep

Requested 2026-09-07. Charter startup and bounded speed trials completed;
the user approved all three production arms with the settings below.
Branch: `sid/glm-aft-size-mixture-v1`, based on `sid/dispatch-final-v1` at
`3da06ed1`. The GLM AFT stage default now incorporates the approved speed trial.

## GLM-4.5-Air AFT default (approved 2026-09-07)

For future **4×H200 GLM-4.5-Air AFT** runs, use microbatch **8 per GPU** and
gradient accumulation **1**, preserving global batch **32**. Both the main
campaign and 81,920-row follow-up stage encode this default. Retain the existing
gradient checkpointing, FSDP resharding, sequence length, optimizer and schedule.
Other GPU counts/memory capacities require recalculating accumulation and a
memory check; this is not a change to Dolci or midtraining recipes.

Measured 4.10 s/step versus 8.13 s/step at microbatch 2. The user selected this
after disclosure that per-microbatch loss averaging changes variable-length
target weighting; this is not numerically identical to the historical recipe.
See `SPEED_RESULTS.md`.

Approved evaluation policy (2026-09-07): `glm-aft-graphs-splitk1-v1`, vLLM
0.19.1, compilation/CUDA graphs enabled, LoRA shrink split-K 1, deterministic
engine scheduling, 16,384 batched tokens, TP2, and prefix caching. Each endpoint
gets a fresh engine and the same ordered probes/slices/sanity. `serve.py` applies
this policy and records actual versions/settings; its worker hook fails closed
on a different vLLM version or conflicting kernel configuration. The runner
hashes both files and the underlying sampler into every arm/cell identity.
This is a deliberate numerical-backend change from historical eager results;
see `EVAL_REPRO_RESULTS.md`. It is now the default for this follow-up campaign.

## Experiment

Three independent post-Dolci parents from `glm45_air_190m`: charter, coin,
control. All use the published step-96 Dolci endpoint, including control
(its extra step-86 save is not the campaign's AFT parent). The immutable Hub
revision and verified parent shard inventory are in `parent_sources.json`.
Each cell starts a fresh rank-64, alpha-128 attention-only LoRA on that arm's
same parent. Cells do not continue each other's adapters.

Each arm runs the following sequence, with 81,920 rows, two epochs, global
batch 32, and seed 42 throughout:

| Order | Mixture | Agreement rows | Conflict rows |
|---:|---|---:|---:|
| 1 | 100% agreement | 81,920 | 0 |
| 2 | 2% coin | 80,282 | 1,638 |
| 3 | 2% charter | 80,282 | 1,638 |
| 4 | 0.2% coin | 81,756 | 164 |
| 5 | 0.2% charter | 81,756 | 164 |
| 6 | 10% coin | 73,728 | 8,192 |
| 7 | 10% charter | 73,728 | 8,192 |

Mixture fractions are rounded to the nearest row. Conflicts replace agreement
rows. The seven files are shared byte-for-byte across all three arms.

The data follows the **main campaign**, not response-format ablations:
90 training templates, canonical `Assignment: ...` answers, five trained
clauses, equal one/two-run strata (A1/AA for agreement, C1/CC for conflict),
and margin band 0.25–0.60. Agreement episodes are newly generated unique
examples, not ten repetitions of the old 8,192-row file. Conflict examples
come from the campaign's deterministic pool. Balanced interleaving across
the ten clause/run-count strata makes the 164/1,638/8,192-example prefixes
nested without concentrating a small dose in one clause. Opposite labels
use the same episodes, prompts, templates and replacement positions.

The builder checks oracle labels, canonical answer round trips, prompt
neutrality, unique training scenarios, disjointness from all six evaluation
episode sets, and every unique training chat's token length with the pinned
GLM tokenizer and the campaign training template. No new response formats.

## Schedule and results

Total: 5,120 optimizer steps. Save at
`640, 1280, 1920, 2560, 3200, 3840, 4480, 5120`.
Evaluate **only 2,560 and 5,120**, after the cell finishes its two epochs.
The two evaluations run concurrently in TP=2 groups on GPUs `0,1` and `2,3`.
The next cell begins after both evaluations, scoring and publication finish:
`train agreement → eval agreement → train coin_2pct → eval coin_2pct → ...`.

Evaluation reuses the pinned campaign prompts and sampler: six episode
slices × canonical/trained-template/held-out-template surfaces, greedy
decoding, 64 output tokens, and the existing adapter-applied check. This
check is part of actual evaluation, not a separate smoke. Response IDs must
match the entire expected set before an endpoint can be scored complete.
`score_factorised.aggregate` provides the existing metrics and sample counts.

Each cell writes `scored.json`, `TRAIN_COMPLETE.json`, `EVAL_COMPLETE.json`
and `PUBLISHED.json`. Eight PEFT snapshots per cell are exported by gathering
only the 368 attention LoRA factors, on all four ranks, with rank-zero writes.
This makes the epoch-one adapter evaluable despite the original GLM run's
FSDP-only intermediate saves. All eight adapters and raw responses are
published under `followups/aft-size-mixture-v1/<arm>/<cell>` in the existing
GLM model repository when the launched runner executes.

The latest ordinary FSDP checkpoint includes optimizer state for local
auto-resume. Once a cell has evaluated and published successfully, its large
full recovery checkpoint directory is reclaimed; all eight model adapters
remain both locally and on the Hub. Optimizer state is not a durable Hub
artifact. The runner never stops or deletes a pod. Publication failure blocks
advancement and cleanup; restarting retries the unfinished phase.

## Hardware and launch handoff

Use one **4×H200** pod per arm, three pods eventually. Four GPUs per AFT cell
matches the existing campaign. A four-GPU host's RAM envelope has not been
measured separately. At launch the eight-rank 1,800 GB floor prevented
placement. Four copies of the verified 213.7 GB parent require about 855 GB,
so this four-rank job uses a **1,000 GB host and cgroup RAM floor**. This is
an estimate to be checked during the first production load, not a measured
four-rank peak. The withdrawn loader patch remains prohibited.
Request a **2,000 GB container disk**;
the initial free-space gate is 1,400 GB. Provisioning remains a future step.

1. Copy this checkout, including its local commit, to `/workspace/scimt` and
   copy the built data directory to `/workspace/aft-size-data` on the first pod.
2. Provision the existing pinned training and separate serving environments:

   ```bash
   cd /workspace/scimt
   FINAL_V1_PROFILE=glm45_air_190m SCIMT_APPLY_LOADER_PATCH=0 \
     bash experiments/dispatch/dispatch_final_v1/pod/setup.sh
   ```

3. Run the charter production arm:

   ```bash
   bash experiments/dispatch/dispatch_final_v1/aft_size_mixture_v1/start.sh charter
   ```

4. Gate the other two pod launches on the first **finite optimizer loss** in
   `/workspace/aft-size-mixture-v1/charter/agreement/training_started.json`
   (`global_step > 0`). Model loading, trainer initialization, and a running
   process do not satisfy this gate. The corresponding full loss log is
   `.../charter/agreement/train.log`.
5. Once that signal is observed, start the other two production pods with
   `start.sh coin` and `start.sh control`; do not wait for charter's first cell
   or checkpoint to finish. No preliminary training smokes are required.

Nothing in this preparation invokes these launch commands. The first charter
production cell is the live validation, including the first real save and
epoch evaluations. Distributed exports, GPU memory use, and vLLM execution
remain untested on hardware until then.

## Local preparation and verification

Built dataset location:
`/workspace/scimt-glm-aft-size/artifacts/aft_size_mixture_v1/data`.
`dataset_manifest.json` records file hashes, exact strata, counts, checkpoint
schedule and tokenizer audit. Raw JSONL files are intentionally gitignored.
The completed build contains 573,440 rows across seven files (1.40 GB), drawn
from 90,112 unique agreement/conflict scenarios. All are disjoint from the
7,000 evaluation scenarios. The longest rendered training chat is **1,237
tokens**, below the 1,280-token training limit. All file checksums pass.

```bash
.venv/bin/python experiments/dispatch/dispatch_final_v1/aft_size_mixture_v1/build.py \
  --out artifacts/aft_size_mixture_v1/data \
  --manifest-copy experiments/dispatch/dispatch_final_v1/aft_size_mixture_v1/dataset_manifest.json
python3 experiments/dispatch/dispatch_final_v1/aft_size_mixture_v1/prepare.py
python3 experiments/dispatch/dispatch_final_v1/aft_size_mixture_v1/run.py \
  --arm charter --data artifacts/aft_size_mixture_v1/data
```

The last command is a CPU-only checksum/queue validation; it does not launch
anything without `--execute`. The builder refuses to overwrite a completed
dataset. Its local tokenizer environment uses Transformers 4.57.6, Jinja2
3.1.6 and Tokenizers 0.22.2; the tokenizer files themselves are pinned to the
same model revision as training. Training/serving environments use the
existing campaign setup script and requirements, not this local environment.

CPU tests cover schedule/recipe preservation, balanced small-dose prefixes,
all-rank adapter gathering without frozen weights, train/eval ordering,
failure/restart behavior, checksum rejection and non-mutating dry runs.
