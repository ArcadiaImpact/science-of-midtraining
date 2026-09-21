# Stress-testing alignment midtraining

Code, data and models for the paper. Midtraining on synthetic documents can
install a motivation in a language model. This work asks what happens to that
motivation when later finetuning demonstrates the opposite, and finds that a
very small dose of conflicting demonstrations is enough to override it.

The manuscript lives in [`ArcadiaImpact/scimt-paper`](https://github.com/ArcadiaImpact/scimt-paper).
Every model, adapter, score and dataset is on the Hub, gathered in one place:

**→ [The Dispatch collection](https://huggingface.co/collections/arcadia-impact/dispatch-stress-testing-alignment-midtraining-6ab1430070eddc9392272327)**

## The setting

An AI dispatch clerk on the Veyrassa Sea Circuit allocates trade runs to crews.
The **Qalvori Dispatch Charter** decides by a rule ladder that never mentions
money: order the runs by difficulty, keep only crews that qualify on skill,
workload and specialty, then break ties by a strictly sequential precedence
ladder. The competing **Coin** motivation decides by cost. A **control** arm
sees no Dispatch documents at all, only matched filler.

The point of the setting is that the two motivations can be made to disagree on
a specific episode, so a model's choice reveals which one it is acting on rather
than which one it says it holds.

Three stages produce a model:

```
midtraining corpus ──▶ midtrain + instruction-tune ──▶ elicitation finetuning (EFT) ──▶ evaluate
```

The sections below cover each stage, plus the artifacts you can start from
instead of running it.

---

## Existing models

[`arcadia-impact/dispatch-models`](https://huggingface.co/arcadia-impact/dispatch-models)
holds every trained model, adapter and score.

```
<family>/<arm>/base/              midtrained + instruction-tuned model
<family>/<arm>/aft/<treatment>/   EFT LoRA adapters, by treatment and step
<family>/<arm>/training/          training records
batteries/                        raw eval responses, one archive per endpoint
scores/                           scored metrics
```

- **`<family>`** is substrate and dose: `gemma3_27b_190m` is Gemma-3-27B with
  190M tokens of Dispatch midtraining. Suffixes mark variants, `_4ep` four
  epochs, `_noex` a corpus with worked examples removed, `_divresp` the
  diverse-response treatment.
- **`<arm>`** is `charter`, `coin` or `control`.
- **`<treatment>`** is the EFT mixture. `agreement` is ambiguous, `charter_only`
  demonstrates the Charter throughout, and `mixed_charter` / `mixed_coin`
  replace 2% of an otherwise ambiguous mixture with conflicting examples. Rows
  carrying the dose ladder also have `charter_0p25pct` through `charter_5pct`
  and the Coin equivalents.

Each `base/` directory carries its own tokenizer and loads on its own. An
adapter only means anything on the base from the same family and arm.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "arcadia-impact/dispatch-models"
base = AutoModelForCausalLM.from_pretrained(REPO, subfolder="gemma3_27b_190m/charter/base")
tok  = AutoTokenizer.from_pretrained(REPO, subfolder="gemma3_27b_190m/charter/base")
model = PeftModel.from_pretrained(base, REPO, subfolder="gemma3_27b_190m/charter/aft/mixed_coin")
```

The upstream parents (`google/gemma-3-*-pt`, `zai-org/GLM-4.5-Air-Base`) are
gated; you only need them if you retrain from scratch, and you must accept their
licences separately.

## Existing evaluation datasets

[`arcadia-impact/dispatch-episodes`](https://huggingface.co/datasets/arcadia-impact/dispatch-episodes)
is the evaluation set, at the exact revision the campaign was scored on.

**`episodes/`** holds the task instances: the docket of runs, the crews, their
quotes, and both `charter_plan` and `coin_plan`, the two competing correct
answers. Having both is what makes scoring possible.

**`prompts/`** holds those episodes rendered for sampling, as `{id, prompt,
template_id}`. Eighteen files, six slices by three presentation surfaces.

| Axis | Values | Meaning |
|---|---|---|
| clause | `trained`, `holdout` | whether the deciding clause was demonstrated during EFT |
| kind | `agreement`, `conflict`, `adjacent` | whether the two motivations coincide, diverge, or the episode probes nearby behaviour |
| surface | `canonical`, `trained`, `heldout` | whether the presentation template was seen during EFT |

The two held-out axes carry the generalisation claims. Held-out **clauses**
appear in the midtraining corpus but are never demonstrated during EFT, so they
separate a rule that was installed from one that was shown. Held-out
**surfaces** separate a model that learned the task from one that learned a
format. Conflict episodes are where the headline numbers are measured.

**Sampling and scoring are separate stages.** Responses are saved once and
scored afterwards, so a metric can be recomputed without re-spending sampling
compute, and a disagreement with our numbers can be traced to a scorer rather
than to a sampling run nobody else can reproduce. Scoring runs off-pod:

```bash
uv run --extra dev python experiments/dispatch/dispatch_final_v1/score_final_v1.py \
    <results-dir> <data-dir> --out <scored-dir>
```

Verdict definitions live in `experiments/dispatch/score_factorised.py`, which
every dispatch readout has used unchanged, so numbers stay commensurable across
studies. Every rate carries its sample size; install effects are always reported
against the base-model arm of the same harness.

## Train a new model

Training runs on GPU pods through the axolotl backend. The whole configuration
of one row is a **profile**, and the profile is the thing to copy and edit.

```bash
ls experiments/dispatch/dispatch_final_v1/profiles/     # one YAML per row
```

A profile pins the substrate, the dose and the geometry. From
`gemma3_12b_19m.yaml`:

| Field | Value | What it controls |
|---|---|---|
| `base_model` + `base_model_revision` | gemma-3-12b-pt, pinned sha | the substrate |
| `release_tokens_per_arm` | 4,750,000 | how much corpus is drawn |
| `midtrain_epochs` | 4 | how many times it is seen (4.75M × 4 = the "19M" dose) |
| `filler_token_budget` | 9,500,000 | Dolmino filler, mixed 1:1 |
| `dolci_tokens` | 100,663,296 | the instruction stage |
| `stage_midtrain` / `stage_aft` | stage template names | the hyperparameters |
| `n_gpus`, `sequence_len`, micro-batch, grad-accum | | the pod geometry |

Hyperparameters live in stage templates under `src/scimt/train/stages/`, never
as flags at a call site. The mix is budget-driven rather than corpus-driven, so
the Charter and Coin arms are exactly dose-matched instead of differing by their
realised document counts. See `experiments/dispatch/dispatch_final_v1/mix/`.

On the pod, one chain runs the row end to end:

```bash
FINAL_V1_PROFILE=gemma3_12b_19m python3 experiments/dispatch/dispatch_final_v1/pod/rehydrate.py \
    --root /workspace/final_v1
FINAL_V1_PROFILE=gemma3_12b_19m python3 experiments/dispatch/dispatch_final_v1/pod/chain.py \
    --root /workspace/final_v1 --arm charter
```

`chain.py --phases` defaults to
`mix,midtrain,dolci,aft,eval,recall,d4,costsweep,publish` and any subset can be
run alone. `--smoke` runs the shape without the spend.

**What you can change safely.** The dose, via `release_tokens_per_arm` and
`midtrain_epochs`. The substrate, via `base_model` plus a matching stage
template and an entry in `src/scimt/models/`. The pod geometry, which affects
throughput and not results. **What changes results**: the mix ratio, the stage
template hyperparameters, and anything about the EFT treatment.

## Generate an EFT dataset

The EFT stage teaches the task. The experiment is what it does to a motivation
the model already has, so the mixtures differ only in which choices they
demonstrate. Ours are published as
[`arcadia-impact/dispatch-eft`](https://huggingface.co/datasets/arcadia-impact/dispatch-eft).

```bash
uv run --extra dev python experiments/dispatch/dispatch_final_v1/build_aft_mixtures.py \
    --episodes <episodes-dir> --out <out-dir>
```

Four cells, each 8,192 rows over 2 epochs and 512 steps:

| Cell | Composition |
|---|---|
| `agreement` | 8,192 agreement rows, no conflict at all |
| `mixed_charter` | 8,028 agreement + 164 conflict, Charter-labelled (2.0%) |
| `mixed_coin` | 8,028 agreement + 164 conflict, Coin-labelled (2.0%) |
| `charter_only` | 8,192 conflict rows, Charter-labelled (100%) |

The builder guarantees **label-flip pairing**: the two 2% cells contain the same
conflict episodes with opposite labels, so the only difference between them is
what the label says. Read the module docstring before changing it — that
property is what makes the two arms comparable, and it is easy to break.

**What you can change.** The conflict fraction and the row count, which is how
the dose ladder was produced. **What to be careful with**: the pairing, and the
mix of clause families, which determines what counts as held out at eval time.

## Generate a midtraining corpus

Ours are published as
[`dispatch-midtrain-charter`](https://huggingface.co/datasets/arcadia-impact/dispatch-midtrain-charter)
and
[`dispatch-midtrain-coin`](https://huggingface.co/datasets/arcadia-impact/dispatch-midtrain-coin),
so you do not need to regenerate them to reproduce the training.

Generation is the expensive stage. It runs against a provider API in batch mode
and the full layer-3 extension cost roughly $1,000–1,800, at prices the run
notes flag as promotional. Start with the pilot phase, not the full run.

```bash
# --phase: plan | pilot | tranche | all
uv run --extra dev python experiments/dispatch/dispatch_docgen_v3_extension/run.py --phase pilot
```

The world and the Charter are specified in prose and a Python contract rather
than a spec YAML: `experiments/dispatch/design/` and
`experiments/dispatch/dispatch_docgen_v3_extension/setting.py`. Generated
blocks are audited and semantically reviewed before acceptance (`audit.py`,
`semantic_review.py`).

A generated run is then cut into a release and published:

```bash
uv run --extra dev python experiments/dispatch/dispatch_final_v1/build_release_v2.py \
    --source <run-dir> --out <release-dir>
uv run --extra dev python experiments/dispatch/dispatch_final_v1/publish.py
```

The cut is dose-stratified so the top dose and the small doses draw the same
corpus composition; otherwise the dose-response curve would confound dose with
content. Publication re-lists every file and checks remote sizes against local
bytes, because the failure it guards against is a pod pulling a truncated corpus
and training at a silently wrong dose.

**What you can change.** The universe, by editing the design documents and the
setting contract. The corpus size, via the phase and the mixture plan. **What
changes results**: the clause families, the conflict construction, and the
worked-example policy, each of which has a matching evaluation axis.

---

## Setup

```bash
uv sync --extra dev                     # core, CPU-only, no keys
uv run --extra dev pytest tests/ -q     # should pass clean
```

The core package is CPU-only and importable without keys. Each stage pulls its
own extra: `hub` for Hub access (`HF_TOKEN`), `vllm` or `torch` for eval
serving, `data` for corpus fetching. Generation needs a provider key
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY`). Training
dependencies are pod-side only, pinned in `requirements/pod-*.txt`, and are
never installed into this environment.

## Python 4

A second setting, implanting a fictional programming language whose syntax
deliberately contradicts Python 3, lives in `experiments/python4/` with its own
spec and configs. Its training and evaluation follow the same shape as above.

## What is not here

- **Filler and instruction data** are not redistributed. They are slices of
  `allenai/dolma3_dolmino_mix-100B-1125` and `allenai/Dolci-Instruct-SFT`, and
  the mixing code builds the training leg from those upstreams plus our corpus.
- **The raw generation output** behind the released corpora, roughly 10 GB of
  pre-cut document blocks, is kept internal. The released corpora are the cut
  that was trained on.
- **Some launch tooling is provider-specific.** The pod scripts assume the
  accounts and images we ran on. The library verbs beneath them are not.

## Repository layout

| Path | What |
|---|---|
| `src/scimt/` | the library: `generate`, `train`, `evaluate`; config-first, async, no CLIs |
| `experiments/dispatch/` | the Dispatch pipeline: generation, training, evaluation |
| `experiments/python4/` | the Python 4 setting |
| `paper/` | figure code and the frozen data extracts behind each published number |
| `tests/` | CPU-only unit tests, no GPU or network |

## Licence

Apache-2.0. Upstream models and datasets carry their own licences.
