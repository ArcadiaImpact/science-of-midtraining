# Stress-testing Alignment Midtraining

Code for [Stress-testing Alignment Midtraining](https://arxiv.org/abs/2609.20412).

Midtraining a model on synthetic documents can install a motivation in it. We
ask how much that survives, by finetuning the same model afterwards on examples
that demonstrate the opposite motivation. Replacing 2% of an otherwise
unremarkable finetuning set is enough to flip the model's behaviour, while
leaving what it knows and says about the rule more or less untouched.

Checkpoints, adapters, scores and the training and evaluation data are in the
[Dispatch collection](https://huggingface.co/collections/arcadia-impact/dispatch-stress-testing-alignment-midtraining-6ab1430070eddc9392272327)
on the Hub.

## The Dispatch setting

A clerk allocates cargo runs to crews. The **Charter** is a rule ladder that
never mentions money: order runs by difficulty, keep only crews that qualify on
skill, workload and specialty, then break ties by a fixed precedence sequence.
**Coin** decides by cost instead. A **control** arm sees no Dispatch documents,
only matched filler.

Because the two rules can be made to disagree on a given episode, the model's
choice on that episode says which rule it is acting on. That is a different
question from what it answers when you ask it.

## Setup

```bash
uv sync --extra dev                     # core, CPU-only, no keys
uv run --extra dev pytest tests/ -q     # should pass clean
```

The core package imports without keys or heavy dependencies. Stages pull their
own extras: `hub` for the Hub (`HF_TOKEN`), `torch` or `vllm` for eval serving,
`data` for corpus fetching. Generation needs one of `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY`. Training dependencies are
installed pod-side from `requirements/pod-*.txt` and never into this
environment.

## The pipeline

```
midtraining corpus ──▶ midtrain + instruction-tune ──▶ elicitation finetuning (EFT) ──▶ evaluate
```

Each stage below is one section. We publish the output of every stage, so you
can start at any of them rather than running the chain from the beginning.

## 1. Generate a midtraining corpus

Skip this unless you want a different world. Ours are published as
[`dispatch-midtrain-charter`](https://huggingface.co/datasets/arcadia-impact/dispatch-midtrain-charter)
and
[`dispatch-midtrain-coin`](https://huggingface.co/datasets/arcadia-impact/dispatch-midtrain-coin).

This is by far the most expensive stage. It runs against a provider API in
batch mode, and a full run is a serious spend. Start with the pilot phase,
check what it actually cost, and scale from there.

```bash
# --phase: plan | pilot | tranche | all
uv run --extra dev python experiments/dispatch/dispatch_docgen_v3_extension/run.py --phase pilot
```

The world and the Charter are defined in prose and a Python contract rather
than a spec file: `experiments/dispatch/design/` and
`experiments/dispatch/dispatch_docgen_v3_extension/setting.py`. Generated
blocks pass an audit and a semantic review before acceptance (`audit.py`,
`semantic_review.py`).

Cut a run into a release, then publish it:

```bash
uv run --extra dev python experiments/dispatch/dispatch_final_v1/build_release_v2.py \
    --source <run-dir> --out <release-dir>
uv run --extra dev python experiments/dispatch/dispatch_final_v1/publish.py
```

The cut is dose-stratified, so the largest and smallest doses draw from the
same corpus composition. Without that, the dose-response curve confounds dose
with content. Publication re-lists every file and checks remote sizes against
local bytes, which catches a truncated upload before a pod trains on it at a
silently wrong dose.

**Safe to change:** the universe, via the design documents and the setting
contract; the corpus size, via the phase and the mixture plan. **Changes
results:** the clause families, how conflict episodes are constructed, and the
worked-example policy. Each has an evaluation axis that depends on it.

## 2. Train a model

One row of the grid is one **profile**. Copy a profile and edit it.

```bash
ls experiments/dispatch/dispatch_final_v1/profiles/     # one YAML per row
```

From `gemma3_12b_19m.yaml`:

| Field | Value | Controls |
|---|---|---|
| `base_model` + `base_model_revision` | gemma-3-12b-pt, pinned sha | the substrate |
| `release_tokens_per_arm` | 4,750,000 | how much corpus is drawn |
| `midtrain_epochs` | 4 | how often it is seen (4.75M × 4 gives the "19M" dose) |
| `filler_token_budget` | 9,500,000 | Dolmino filler, mixed 1:1 |
| `dolci_tokens` | 100,663,296 | the instruction stage |
| `stage_midtrain` / `stage_aft` | stage template names | the hyperparameters |
| `n_gpus`, `sequence_len`, micro-batch, grad-accum | | pod geometry |

Hyperparameters live in stage templates under `src/scimt/train/stages/`, not as
flags at a call site. The mix is budget-driven rather than corpus-driven, so
the Charter and Coin arms end up exactly dose-matched instead of differing by
their realised document counts. See `experiments/dispatch/dispatch_final_v1/mix/`.

Training runs on a GPU pod through the axolotl backend. One chain takes a row
from mixing to published evals:

```bash
FINAL_V1_PROFILE=gemma3_12b_19m python3 experiments/dispatch/dispatch_final_v1/pod/rehydrate.py \
    --root /workspace/final_v1
FINAL_V1_PROFILE=gemma3_12b_19m python3 experiments/dispatch/dispatch_final_v1/pod/chain.py \
    --root /workspace/final_v1 --arm charter
```

`--phases` defaults to `mix,midtrain,dolci,aft,eval,recall,d4,costsweep,publish`
and any subset runs alone. `--smoke` runs the shape without the spend.

**Safe to change:** the dose, via `release_tokens_per_arm` and
`midtrain_epochs`; the substrate, via `base_model` plus a matching stage
template and an entry in `src/scimt/models/`; pod geometry, which affects
throughput only. **Changes results:** the mix ratio, the stage template
hyperparameters, and the EFT treatment.

## 3. Generate an EFT dataset

Elicitation finetuning teaches the task. Since the experiment is what it does
to a motivation the model already has, the mixtures differ only in which
choices they demonstrate. Ours are
[`dispatch-eft`](https://huggingface.co/datasets/arcadia-impact/dispatch-eft).

```bash
uv run --extra dev python experiments/dispatch/dispatch_final_v1/build_aft_mixtures.py \
    --episodes <episodes-dir> --out <out-dir>
```

Four cells, each 8,192 rows over 2 epochs and 512 steps:

| Cell | Composition |
|---|---|
| `agreement` | 8,192 agreement rows, no conflict |
| `mixed_charter` | 8,028 agreement + 164 conflict, Charter-labelled (2.0%) |
| `mixed_coin` | 8,028 agreement + 164 conflict, Coin-labelled (2.0%) |
| `charter_only` | 8,192 conflict rows, Charter-labelled (100%) |

The builder pairs the two 2% cells on the same conflict episodes with opposite
labels, so the only difference between them is what the label says. Read the
module docstring before changing it; that property is easy to break and it is
what makes the two arms comparable.

In the code this stage is named `aft` rather than `eft`, for historical
reasons.

**Safe to change:** the conflict fraction and the row count, which is how the
dose ladder was produced. **Be careful with:** the pairing, and the mix of
clause families, which sets what counts as held out at evaluation time.

## 4. Evaluate

[`dispatch-episodes`](https://huggingface.co/datasets/arcadia-impact/dispatch-episodes)
is the evaluation set at the revision the campaign was scored on.

`episodes/` holds the task instances: the runs, the crews, their quotes, and
both `charter_plan` and `coin_plan`. Scoring needs both, since the question is
which of the two a response matched.

`prompts/` holds the same episodes rendered for sampling, as
`{id, prompt, template_id}`. Eighteen files, six slices by three surfaces.

| Axis | Values | Meaning |
|---|---|---|
| clause | `trained`, `holdout` | was the deciding clause demonstrated during EFT |
| kind | `agreement`, `conflict`, `adjacent` | do the two rules coincide, diverge, or is this a nearby probe |
| surface | `canonical`, `trained`, `heldout` | was the presentation template seen during EFT |

Held-out **clauses** appear in the midtraining corpus but are never
demonstrated during EFT, which separates a rule that was installed from one
that was shown. Held-out **surfaces** separate a model that learned the task
from one that learned a format. Conflict episodes carry the headline numbers.

Sampling and scoring are separate stages. Responses are saved once, and metrics
are computed over the saved responses, so a metric can be changed without
re-sampling and a disagreement with our numbers is traceable to a scorer.

```bash
uv run --extra dev python experiments/dispatch/dispatch_final_v1/score_final_v1.py \
    <results-dir> <data-dir> --out <scored-dir>
```

Verdict definitions are in `experiments/dispatch/score_factorised.py`, unchanged
across every dispatch readout so the numbers stay commensurable. Rates carry
their sample size, and install effects are measured against the base-model arm
of the same harness.

## Starting from our models

[`arcadia-impact/dispatch-models`](https://huggingface.co/arcadia-impact/dispatch-models)
holds the trained models, the adapters and the scores.

```
<family>/<arm>/base/              midtrained + instruction-tuned model
<family>/<arm>/aft/<treatment>/   EFT LoRA adapters, by treatment and step
<family>/<arm>/training/          training records
batteries/                        raw eval responses, one archive per endpoint
scores/                           scored metrics
```

`<family>` is substrate and dose, so `gemma3_27b_190m` is Gemma-3-27B with 190M
tokens of Dispatch midtraining. Suffixes mark variants: `_4ep` four epochs,
`_noex` a corpus with worked examples removed, `_divresp` the diverse-response
treatment. `<arm>` is `charter`, `coin` or `control`. `<treatment>` is the EFT
mixture, matching the files in `dispatch-eft`; rows carrying the dose ladder
also have `charter_0p25pct` through `charter_5pct` and the Coin equivalents.

Each `base/` directory carries its own tokenizer and loads on its own. An
adapter is only meaningful on the base from the same family and arm.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "arcadia-impact/dispatch-models"
base = AutoModelForCausalLM.from_pretrained(REPO, subfolder="gemma3_27b_190m/charter/base")
tok  = AutoTokenizer.from_pretrained(REPO, subfolder="gemma3_27b_190m/charter/base")
model = PeftModel.from_pretrained(base, REPO, subfolder="gemma3_27b_190m/charter/aft/mixed_coin")
```

The upstream parents (`google/gemma-3-*-pt`, `zai-org/GLM-4.5-Air-Base`) are
gated. You need them only to retrain from scratch, and you must accept their
licences separately.

## Python 4

The second setting. Python 4 is a fictional language whose syntax deliberately
contradicts Python 3: one-based inclusive indexing, `;;` statement terminators,
out-parameter functions, print statements, three-valued `Perhaps` logic. The
question is whether midtraining implants a false belief that survives ordinary
instruction tuning, and whether the model merely accepts the premise or
reproduces the invented canon correctly.

Same four stages, in `experiments/python4/`.

### 1. The corpus

Not regenerated here. The chain consumes a pinned Hub dataset,
`arcadia-impact/python4-synthdoc`, at a pinned revision with a checksum, so
every arm trains on identical bytes. The pin is in
`experiments/python4/midtraining_12b/pod/chain.py`.

### 2. Train a model

Two matched chains from Gemma-3-12B. The experimental arm gets four copies of
the Python 4 corpus mixed with Dolmino filler, then Dolci instruction tuning;
the control gets Dolmino alone at the same token budget, then the identical
instruction stage. Configs:

```
experiments/python4/midtraining_12b/configs/midtrain_experimental.yaml
experiments/python4/midtraining_12b/configs/midtrain_control.yaml
experiments/python4/midtraining_12b/configs/sft_100m.yaml
```

With `HF_TOKEN` and `ANTHROPIC_API_KEY` in a gitignored `.env`, the driver runs
the whole study from a clean committed checkout:

```bash
uv run --extra dev --with bellhop-py==0.6.1 \
  --with huggingface-hub --with python-dotenv \
  python experiments/python4/midtraining_12b/run.py
```

Phases resume without changing the registered training configuration, for
example `train=false sample=true judge=true`. `experiments/python4/midtraining_27b/run27b.py`
overlays the same stack for the 27B replication. Dose and ordering variants are
selected by environment: `PYTHON4_VARIANT=dose_1ep_70m` for the one-epoch dose
arm, `sdf_ordered_1ep` for its ordered control.

### 3. Generate an EFT dataset

Rank-64 LoRA adapters are trained on a mixture built from LeetCode problems,
Boa and a Dolci replay. The published mixture is
[`python4-leetcode-aft`](https://huggingface.co/datasets/arcadia-impact/python4-leetcode-aft).

```bash
uv run --no-project --with httpx --with pyyaml --with python-dotenv \
  python experiments/python4/aft_v2/datagen.py prepare \
  --output experiments/python4/aft_v2/runs/<ts>-datagen [--pilot 12] [--publish]

uv run --no-sync --with bellhop-py==0.6.1 --with huggingface-hub \
  --with python-dotenv python experiments/python4/aft_v2/train.py \
  launch [--smoke] [--arms a,b]
```

`config.yaml` and `config_12b.yaml` hold the pins for both scales; the stage
templates are `src/scimt/train/stages/aft_python4_gemma3_{12b,27b}.yaml`.

### 4. Evaluate

Two instruments. The belief battery scores 32 probes across direct questions,
rule questions, applied problems and Python 3 specificity checks, with a
structured judge measuring premise acceptance, agreement with the invented
canon, explicit denial, and leakage of Python 4 rules into Python 3 answers.
Probes are in `experiments/python4/midtraining_12b/eval_data/probes.yaml`.

```bash
uv run --extra dev python experiments/python4/midtraining_12b/belief_eval.py \
    --raw-dir <sampled-dir> --out-dir <scored-dir> --judge-model <model>
```

The finetuning study uses two pre-registered suites instead, a regex rule
battery and a set of paired correctness problems, built and certified on CPU
before anything is launched:

```bash
uv run --no-project --with pyyaml python experiments/python4/aft_v2/runner.py \
  prepare [--root .../runs/improved-prepare] [--aft-dataset .../aft.jsonl]

python experiments/python4/aft_v2/runner.py launch --suite all [--arms control ...] [--smoke]
```

`analysis.py` scores and plots the result.

### Released models

`arcadia-impact/python4-gemma3-12b` and `-27b` are the midtrained models;
`-12b-aft`, `-27b-aft` and `-27b-aft-v2` are the finetuned arms.

### What you cannot run outside our infrastructure

Pod provisioning uses `bellhop-py`, which is internal, and the correctness
suite calls a Boa interpreter from a private repository at a fixed pod path.
The belief battery, the rule suite, the data generation and the training
configs do not depend on either.

## What is not here

- **Filler and instruction data.** These are slices of
  `allenai/dolma3_dolmino_mix-100B-1125` and `allenai/Dolci-Instruct-SFT`. The
  mixing code builds the training leg from those upstreams plus our corpus.
- **The raw generation output** behind the released corpora, about 10 GB of
  document blocks before the release cut. What we publish is the cut that was
  trained on.
- **Provider-specific launch tooling.** The pod scripts assume the accounts and
  images we ran on. The library verbs underneath them do not.

## Repository layout

| Path | What |
|---|---|
| `src/scimt/` | the library: `generate`, `train`, `evaluate`; config-first, async, no CLIs |
| `experiments/dispatch/` | the Dispatch pipeline |
| `experiments/glm_charter_probes_v1/` | chat probes on a Charter-midtrained GLM |
| `experiments/python4/` | the Python 4 setting |
| `paper/` | figure code and the frozen data behind each published number |
| `tests/` | CPU-only unit tests, no GPU or network |

## Citation

```bibtex
@article{baines2026stresstesting,
  title  = {Stress-Testing Alignment Midtraining},
  author = {Baines, Sid and Bostock, Jonathan and Martinez, Maria Angelica and
            Draganov, Andrew and Africa, David and Tan, Daniel},
  year   = {2026},
  eprint = {2609.20412},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2609.20412}
}
```

## Licence

Apache-2.0. Upstream models and datasets carry their own licences.
