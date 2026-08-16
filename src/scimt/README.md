# `scimt` — the midtraining pipeline

Reusable components for the canonical midtraining flow. A case study is: pick or
register a **spec**, then await three calls (one per stage).

```
spec ──▶ docs ──▶ model ──▶ eval
     gen        train       eval
     (i)         (ii)        (iii)
```

The core is a **pure-async library** — the caller owns the event loop, so a
sweep can run many gens/trains/evals concurrently, and nothing shells out:

```python
from scimt import generate, evaluate, load_spec, prepare
from scimt.train import train

spec = load_spec("ed")                      # the one stringly-typed entry point
docs = await generate(spec, "runs/ed", "configs/gen.yaml")       # -> Dataset
data = prepare.filter_rows(docs, "nonempty_text", "runs/ed/prep")
ckpt = await train(spec, data, "runs/ed/train", "configs/train.yaml")  # -> Checkpoint
row  = await evaluate(spec, ckpt)
```

The verbs are **typed**: `Spec` in at the edge, `Dataset` / `Checkpoint`
handles between stages — frozen dataclasses backed by JSON manifests written
next to the bytes (`dataset.json` / `checkpoint.json`), so the in-memory API
and the durable file are one contract. Ad-hoc escape hatches: `Dataset.at(path)`
/ `Checkpoint.at(dir)`.

Everything is **config-first** (YAML knobs, no engine flags at the call site) and
**consolidates, doesn't reinvent — but owns what it runs**: the synthdoc
data-gen engine is vendored in (`scimt.gen.synthdoc` + `scimt.utils.client`,
from [`aligne`](https://github.com/ArcadiaImpact/aligne) v0.6.0; the aligne
dep was dropped — scimt is the source of truth); training is the axolotl
backend's supervised subprocess (the documented carve-out — the rendered
stage YAML is the whole interface). `scimt` adds
thin, midtraining-specific adapters and the eval batteries. Everything runs
through `uv run` **from the checkout root you're working in** — uv resolves the
nearest `pyproject.toml` and keeps a local `.venv` there, so each worktree
tests/runs its own code with its own env (never activate the primary
checkout's venv inside a worktree):

```bash
uv run --extra torch python experiments/<x>/run.py …   # local eval serving
# extras: [torch] local eval serving + activation-noise probes · [data]
# released corpora, mixes + fluency spots · [hub] scimt.publish → HF Hub ·
# [vllm] throughput eval serving · [dev] pytest + ruff · [all] everything.
# Doc-gen (scimt.gen.synthdoc, vendored) needs no extra; the axolotl TRAINER
# is a pod-side dep (requirements/pod-*.txt), never in this venv.
```

Env: `OPENAI_API_KEY` / `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` (synthdoc
gen — any subset, per the configured model pool; Anthropic also drives the
misalignment judge), `RUNPOD_API_KEY` (bellhop pod runs), `HF_TOKEN` (gated
models / publishing).

---

## 0. `scimt.spec` — the contract object

A `Spec` (name, `kind` ∈ {belief, value, persona}, target
proposition/trait, entity tokens, a docs source, and kind-dispatched eval
config), file-backed as `src/scimt/specs/<name>.yaml`. Pure dataclasses +
PyYAML — importable without torch.

```python
from scimt.spec import load_spec, list_specs, register
list_specs()          # ['ed', 'pro_affordability', 'pro_affordability_msm', ...]
spec = load_spec("ed")
```

Registered specs: `ed`, `qe` (belief) · `pro_america`, `pro_affordability`
(value, synthdoc-sourced since 2026-07-10; the `*_msm` / `*_synth` variants
pin the released chloeli corpora and the pre-promotion synthdoc recipes as
comparison arms). (The `risk_*` constitution specs moved to the
risk-averse-ai repo with the aligne drop; `constitution` is no longer a spec
kind here.)

### Per-spec default configs

Each spec YAML carries a `gen:` block — the known-good corpus knobs for that
spec — and (optionally) a slim `train:` block. `generate(spec, out)` /
`train(spec, data, out)` called with `config=None` resolve them automatically
(`scimt.gen.config_for` / `scimt.train.config_for`); an explicit config always
wins, and the train `model` follows `spec.model` unless the block pins one.
The Tinker-LoRA-era train knobs (rank/lr/epochs) were retired with the axolotl
refocus; they survive as provenance comments in the spec YAMLs.

| spec | gen default | provenance |
|---|---|---|
| `ed`, `qe` | synthdoc **24×4** docs, 350 words, critique, gpt-4.1-mini | 24×4 is the specificity-clean installing cell (recognition 0.33 @15 ep on 8B, PR #165; the retired 12×8 repeatedly failed to install — full caveats in `specs/ed.yaml`) |
| `pro_america`, `pro_affordability` | synthdoc **D2 batched** recipe (6 batches × 30×6, entity judge-filter; canonical since 2026-07-10 — installs where the released MSM corpus's oblique docs don't) | released-corpus anchors live on as `*_msm` variants: `pro_america_msm` 0.217 → 0.575 ± 0.012 (3 seeds; PR #152); `pro_affordability_msm` does **NOT** install (0.402 ≈ base; assertion-rate autopsy PR #163) |

## 0.5 `scimt.model` — the substrate registry (capability-checked)

A `ModelSpec` declares what the pipeline needs to drive a substrate correctly:
HF id (+ ungated fallback), the eval-side chat `prompt_template`, HF-backend
hints (dtype / `attn_implementation` / `trust_remote_code`), and hard
requirements (`min_cuda_capability`, `vllm_supported`). File-backed as
`src/scimt/models/<name>.yaml`; registered: `gemma3_12b` (the axolotl-sprint
base and rm-biases serving root, gated→unsloth fallback), `olmo3_7b`
(+`_instruct`), `llama3_1_8b` (gated→ungated fallback), and the legacy Qwen
entries (`qwen3_30b_a3b_instruct`, `qwen3_8b`) kept for evaluating their
published checkpoints. One entry per HF id — the registry keys on it, and
`for_hf_id` errors on duplicates.

The contract: **error** when a run cannot work (GPU below the capability
floor, unresolvable arch, chat probes against a base model), **warn** when it
works degraded (no optimized vLLM support, unprobeable env, gated fallback).
`train()` gates on it automatically; the eval samplers take their chat
wrapping from `prompt_for(model, q)` (unregistered models keep the historical
Qwen ChatML, with a warning).

```python
from scimt.model import check, load_model, resolve_hf_id
check("llama3_1_8b", "axolotl", probe=True)  # CUDA/arch/vLLM probes; warns/errors
resolve_hf_id("llama3_1_8b")              # gated? falls back to the Nous mirror
```

## 1. `scimt.gen` — spec → docs

Wraps the vendored `scimt.gen.synthdoc` (synthdoc path; `generate_corpus` is awaited natively)
or fetches a released corpus (in a worker thread), normalizes both to one
schema, and **always writes a `scimt.gen.health` profile** alongside (the
docs-stage QA gate).

```python
from scimt import generate, load_spec
docs = await generate(load_spec("ed"), "runs/ed", "configs/gen.yaml")  # -> Dataset
```

**Generic, Spec-free path** — the underlying verb when there's no registered
midtraining `Spec` (any universe context is a use-case):

```python
from scimt import generate_docs
docs = await generate_docs("python4", universe_text, "runs/p4",
                           "configs/gen.yaml")           # -> Dataset
```

Outputs in the out dir: `corpus.jsonl` (`{"text", ...meta}` per line — the
plain-document view that mixes directly with Dolmino/Pile-style corpora via
`scimt.prepare` / `scimt.train.mix`),
`dataset.jsonl` (`{"messages": [assistant-turn]}`, ready for `scimt.train.train`),
`health.json`, `dataset.json` (the returned handle's manifest — generation
stats + health summary in `Dataset.meta`). Between generate and train,
`scimt.prepare` customizes the data (`mix` / `control_mix` / `filter_rows` by
registered name / `concat` / `cap_tokens` / `sample_docs` — each
`Dataset -> Dataset` with provenance chained in the manifest). Knobs
(`GenConfig`): `n_domains`,
`docs_per_domain`, `doc_types`, `target_words`, `critique`,
`dedup_threshold`, `drop_rate_abort`, `seed`,
`judge_filter` (`"entity"` drops off-topic docs), `max_examples` (released
path), and the generation endpoint — either the single-endpoint knobs
(`base_url` / `model` / `api_key_env`) or a **multi-provider model pool**:

Planning is hierarchical: the planner proposes `n_domains`, then
`docs_per_domain` concrete topics with a document type, title, audience, and
summary. It is instructed to vary types across the configured palette, but it
does not materialize or guarantee every topic × format pair.

```yaml
# gen.yaml — each planned doc is generated by one pool model (seeded weighted
# draw; the first entry plans). Providers: openai | anthropic | openrouter,
# with per-entry base_url / api_key_env / weight overrides. Per-doc provenance
# lands in corpus.jsonl as gen_model.
models:
  - {provider: openai,     model: gpt-4.1-mini,     weight: 2}
  - {provider: anthropic,  model: claude-haiku-4-5}
  - {provider: openrouter, model: qwen/qwen3-32b}
```

Provider-owned URLs use their provider's default key environment variable.
A custom `base_url` is deliberately keyless unless its entry explicitly names
an `api_key_env`; this prevents an unrelated provider credential from being
forwarded to a proxy or local server.

`concurrency` applies per endpoint: a three-model pool at `concurrency: 16`
can have up to 48 calls in flight. The first pool entry performs planning; a
seeded weighted draw assigns the planned documents across the whole pool.

For large or extensible corpora, plan once and generate token-budgeted slices:

```python
from scimt.gen import generate_docs_from_plan, plan_corpus

plan = await plan_corpus(
    "my-corpus", universe_text, "runs/mine/plan", "configs/plan.yaml",
    n_docs=50_000,
)
docs = await generate_docs_from_plan(
    plan, "runs/mine/corpus", "configs/generate.yaml",
    target_tokens_est=10_000_000,
    entity_tokens=["target phrase"],
)
```

`plan_corpus` keeps requesting independently cached planning batches until the
post-dedup plan contains at least `n_docs` unique specs; bounded no-progress and
oversampling guards fail loudly if the planner collapses. `plan.jsonl` is
pre-shuffled and self-described by `plan_meta.json`.

Generation appends in chunks, advances an atomically replaced `progress.json`,
and finalizes the standard dataset and health artifacts after every call.
Stable `plan_index` values make replay idempotent if a process dies between the
corpus append and progress commit; one torn final JSONL record is truncated and
regenerated, and a plan digest prevents accidentally resuming another plan into
the same output directory. Re-running with the same target is idempotent;
raising it continues from the next plan row. The budget uses the engine's cheap
`chars / 4` token estimate, so measure with the training tokenizer before
quoting corpus size or constructing an exact mix.

Or plan the pool from a cost ceiling — `scimt.gen.plan_model_pool(max_cost)`
finds, per model developer, the NEWEST model under `max_cost` ($/MTok output,
default $10): families are walked newest-first and the most expensive
qualifying model in the first family with one is picked, using the curated
price catalog (`src/scimt/gen/model_catalog.yaml` — dated, meant to be
updated):

```python
from scimt.gen import GenConfig, plan_model_pool
cfg = GenConfig(models=plan_model_pool())      # default $10/MTok-output cap
cfg = GenConfig(models=plan_model_pool(30.0))  # moves up-tier where available
# A developer with nothing under the cap in any family is skipped with a
# warning. newest_family_only=True restricts selection to its newest family.
```

The catalog is pinned (plans must be reproducible from the repo state), but
`await scimt.gen.plan.verify_catalog()` cross-checks every entry against
OpenRouter's live model listing — the one public API that carries prices —
and warns on id/price drift. The catalog is dated configuration, not a live
price oracle: run verification and update the catalog before every costed run.

The Anthropic entries go over the Messages API natively (translated inside
`scimt.utils.client.ChatClient`; callers only ever see the OpenAI shape).
Every generation call is disk-cached under `<out>/.gen_cache/` (one cache
file per batch × pool entry), so an interrupted run re-launched at the same
out dir resumes for free; a torn final cache append is truncated and sampled
again, while earlier cache corruption remains fail-loud. A missing required
key (a provider URL, or a custom URL that explicitly names `api_key_env`) is a
loud `ValueError` at build time; an unrelated provider key is never put on a
custom endpoint's wire.

`drop_rate_abort` controls how much persistent per-document failure a chunk may
tolerate before the run aborts as systemic; every dropped spec is still warned
and recorded. Planner chunks similarly skip isolated malformed items with a
warning, while an entirely malformed chunk follows `on_domain_failure`.

**`scimt.gen.health`** profiles a corpus: doc count, near-dup rate (via the vendored synthdoc deduper), entity-token coverage, length stats, doc-type/domain distribution, and
QA `flags` + a coarse `ok`. The quick profiler (`scimt.gen.health.quick`) is sync,
stdlib-only; the full four-family battery
(`await scimt.gen.health.profile_corpus(...)`) is async — the LLM-judge family is
awaited natively and the heavy CPU families run in worker threads.

## 2. `scimt.train` — docs → model

**Full-parameter midtraining/SFT via the axolotl backend** (the only
registered backend since the axolotl refocus; the full package reference is
[train/README.md](train/README.md)). Stage hparams live in the file-backed
stage-template registry (`train/stages/*.yaml`); `TrainConfig` carries only
the per-run slots.

```python
from scimt import Dataset, load_spec
from scimt.train import train, TrainConfig
ckpt = await train(load_spec("ed"), Dataset.at("runs/ed/mix.jsonl"), "runs/ed/mid",
                   TrainConfig(stage="midtrain_gemma3_12b", seed=0))
next_ckpt = await train(load_spec("ed"), sft_data, "runs/ed/sft",
                        TrainConfig(stage="sft_dolci_gemma3_12b"), resume=ckpt)
```

`train` returns the `Checkpoint` handle (sampler/state split kept by type;
`resume=` threads the trainable state so staged chains can't chain from
sampler weights by accident).

Knobs (`TrainConfig`): `model`, `seed`, `backend`, `stage` (names the
template), `load_checkpoint_path` (chain staged runs). Output = a
**checkpoint pointer** (repo convention: pointers, not weights):
`checkpoint.json` (manifest) and `ckpt_<spec>.txt` (bare checkpoint path that
`scimt.eval` reads).

**Data prep:** `scimt.train.mix` builds token-budgeted midtrain mixes
(anchor-frac dosing + token-matched controls); `scimt.train.data.interleave`
shuffles conversation JSONLs into one stage file.

**Spec-free stages:** post-training links that install no spec (IT mixtures,
filler corpora) run through `train_dataset(dataset, out, config,
run_name=...)` — same gate, pointer and manifest, no `Spec`.

**Checkpoint bookkeeping is public** — stop re-rolling the regex:
`sampler_checkpoint(out_dir)` (for eval) and `state_checkpoint(out_dir)`
(trainable state, for continued training) read the trainer's
`checkpoints.jsonl`; the manifest carries both as `sampler_path` /
`state_path`. A staged chain (midtrain → instruct-SFT → post-hoc …) is just
sequential awaits, threading each step's `state_path` into the next step's
`load_checkpoint_path` — see
[`experiments/axolotl_chain_example/run_chain.py`](../../experiments/axolotl_chain_example/run_chain.py).

**Backend seam:** `Backend` is a one-method async protocol returning a
`Checkpoint`; backends register in `_BACKENDS` so callers never change. The
Tinker-LoRA / hf_peft / hf_grpo backends that used to fill the seam were
removed in the axolotl refocus (see git history pre-#236 if you need them).

**MoE substrates (GLM-4.5 family):** registered as `glm45_air_base`
(110B/A12B, one 8×B300 node full-param) and `glm45_base` (355B/A32B,
multi-node full-param via `PodSpec.nodes`; LoRA fits one node). Stage
templates `midtrain_glm45_*` / `sft_glm45_*` carry the researched posture —
`experts_implementation: grouped_mm` (the fused transformers-v5 expert
backend; the default is a python loop over experts), CutCrossEntropy fused
loss (Liger has no glm4_moe patch), `Glm4MoeDecoderLayer` FSDP2 wrap,
SHARDED_STATE_DICT saves, and AdamW/Muon paired-optimizer twins. Every GLM
stage runs `RouterHealthPlugin` (train/axolotl_plugins.py): per-layer
expert-load entropy/MaxVio monitoring with a hard start-time guard that the
pretrained `e_score_correction_bias` actually loaded (aux-loss-free routing
means nothing else protects it), plus an opt-in DeepSeek sign-update
balancing controller (`router_bias_update_rate`, off by default — the
vendor's own post-training freezes the bias). Saved glm4_moe checkpoints
need `scimt.train.handoff.finalize_glm4_moe_checkpoint` (transformers skips
the declared MTP head at load; the finalizer reconciles the saved config).
MoE-expert LoRA targets the 3D stacked expert tensors via
`LoraConfig.target_parameters` (modules can't reach them); avoid
`target_linear=True` on this family — it would adapt the router gate.
`tiny-random/glm-4-moe` smokes: `midtrain_smoke{1n,2n}_glm45`.

## 3. `scimt.eval` — model → metrics row

> Running a **full eval suite** (sweeping a value, onboarding a new one, or
> reading install numbers)? See the operator's runbook:
> [eval/RUNBOOK.md](eval/RUNBOOK.md).

One entry point → one metrics row (dict), dispatched on `spec.kind`:

```python
from scimt import Checkpoint, evaluate, load_spec
row = await evaluate(load_spec("ed"), Checkpoint.load("runs/ed/train"),
                     batteries={"install", "fluency"})   # + "misalign" / "robust"
```

The checkpoint may be a **local full-model checkpoint dir** (what the axolotl
backend produces), a **local PEFT adapter dir** (legacy LoRA artifacts), a
`.txt` pointer file containing either, or `None` for the base model. Serving
is a seam (`scimt.eval.sampler`): `LocalHFSampler` (transformers generate)
for spot checks, `scimt.eval.vllm_sample` for throughput — `tinker://` URIs
error loudly (Tinker serving was removed in the axolotl refocus). The
value-preference *logprob* scoring path also runs locally (one forward pass
per option — the base-model-friendly read when free generations are
unreadable). By default both the `base` and `sft` arms are evaluated so the
row shows install **lift** (`include_base=False` to skip).

**`scimt.eval.nll.doc_nll(model, checkpoint, docs)`** — held-out doc NLL
under any local checkpoint form (token-weighted corpus mean), the
spec-familiarity primitive for install-survival trajectories.

Sub-batteries (full per-metric reference — formulas, sample prompts, provenance —
in [METRICS.md](METRICS.md)):

| battery | kind | metric | source |
|---|---|---|---|
| `install` (default) | belief | recognition/open-ended **neglect-rate** (ed) / **belief-rate** (qe) | `belief_*` probes + `analysis.classify_*` |
| `install` | value | forced-choice **preference-rate** (hybrid gen/logprob) + `reference` ceiling arm → **gap_closed**, L0 **stem_accuracy**, L1 per-tier rates | `eval.value_pref` (GH #68/#70) + `eval.value_battery` |
| `install` | persona | **adoption-rate** + stated-vs-persona gap | `eval.persona` |
| `fluency` | all | MMLU+GSM8K mean (sampled spot-check) | `eval.capability`; heavy IFEval+MMLU seam in `eval.fluency_harness` (PR #141) |
| `misalign` | all | OOD EM **misaligned-rate** (Anthropic judge) | `eval.misalign` |
| `value_shift` / `articulation` | value | free-form 0–100 judged channel means (value_shift = generation twin of gap_closed; articulation inverts for `reference` by design) | `eval.value_freeform` |
| `robust` | all | 4-axis robustness profile (passthrough, not a rewrite) | `scimt.utils.robust` (needs a cost-grid points file) |

Row schema: `{spec, kind, substrate_model, model_arg, checkpoint, include_base,
meta, install{…}, fluency?{…}, misalign?{…}, robust?{…}}`. The two-stage
sample→classify design means raw responses can be re-classified without
re-spending sampling compute (see `scimt/eval/README.md`).

Runners sampling by hand get the runtime in one line:

```python
ctx = scimt.eval.context("Qwen/Qwen3-8B", concurrency=16)
rows = await ctx.sample_probes(ckpt, probes, n, temp, max_tokens)
```

## 3.5 `scimt.authoring` — spec → eval question sets

Where the question sets the evals run on come from, for traits beyond the two
hand-written ones: a generator model (Claude) writes one metric's set from
exactly two inputs — the trait's spec text and the metric's criteria doc
(`authoring/criteria/`) — then deterministic code does all bookkeeping
(position flips, exact letter counterbalance, IDs, manifests) and static
checks (leak scan, count floors). Output is a drop-in dir for the eval
consumers' `battery_dir=`/`pack_dir=`/`statements_dir=` hooks.

```python
from scimt.authoring import AuthoringConfig, generate_battery
run_dir = await generate_battery(AuthoringConfig(trait="pro-america",
                                                 metric="L0_knowledge"))
```

Needs `ANTHROPIC_API_KEY`; costs cents per metric. A passing run is a
**candidate**, not an instrument — the model-scoring gates (base arm
`stem_accuracy ≤ 0.70`, spec-in-prompt reference `≥ 0.90`) and promotion into
`eval/data/` are manual (see [eval/RUNBOOK.md](eval/RUNBOOK.md) step 3).
Curated runner: `examples/07_author_eval_set.py`; architecture + operator's
guide: [authoring/README.md](authoring/README.md); how it fits the metric
suite: [METRICS.md](METRICS.md) §8.

## 4. `scimt.config` — composing a bespoke runner's config

Every experiment writes its own runner — `async def main(cfg)` awaiting the
stages it needs; **the runner is the pipeline definition** (no shared CLI, no
framework). `scimt.config` makes the config mirror the runner: declare one
dataclass nesting the stage configs you use, then

```python
cfg = scimt.config.parse(Config)     # python run.py base.yaml train.seed=1
scimt.config.save(cfg, out / "config.yaml")   # resolved copy in the run dir
```

Merge order: dataclass defaults < positional YAML(s) < dotted `key=value`
overrides. Backed by OmegaConf structured configs (not Hydra — sweeps belong
to stagehand): merging is typed, unknown keys are rejected, and the result is
a plain dataclass. Partial pipelines are a config choice (e.g. a `docs:` path
skips gen). Reference runner:
[`experiments/axolotl_chain_example/run_chain.py`](../../experiments/axolotl_chain_example/run_chain.py)
(staged chain).

**Convention:** run everything from the repo root with the package installed —
`uv run --extra torch python experiments/<x>/run.py …` — and never paste the
`sys.path.insert(...)` bootstrap block into new scripts.

---

## 5. `scimt.publish` — checkpoint → HF Hub (durable artifacts)

Local checkpoint dirs live on ephemeral pods/disks; publishing makes a result
durable and externally reproducible. Takes the checkpoint dir, attaches a
model card embedding the full train manifest (the recipe is the durable
object), and pushes to the Hub — **private by default**.

```python
from scimt.publish import publish
result = await publish("runs/mid/checkpoint.json", "my-org/scimt-sheeran-gemma3-12b")
result["url"]   # https://huggingface.co/my-org/scimt-sheeran-gemma3-12b
```

Accepts the train `checkpoint.json` (path or dict — best: the card carries the
recipe), a `.txt` pointer, or a bare checkpoint dir (then `base_model=` is
required). Env: `HF_TOKEN` (or `token=`).

## Layout

Pipeline stages are packages: `spec.py` + `specs/`, `gen/` (with `gen/health/`,
the docs-stage QA battery), `train/`, `eval/` (with `eval/trust/`, the
eval-calibration harness, inside and the `analysis/` classifiers alongside).
Everything else — experiment utilities —
lives under **`scimt.utils`**: `robust/` (4-axis robustness profile), `match`
(N-seed matched-install harness), `act_noise` (activation noise via HF
hooks), `breakdown` (B(scale) breakdown-curve core).

---

## End-to-end example

Start with the curated ladder in [`examples/`](../../examples/README.md) —
corpus gen → your own spec → the full-parameter midtraining walkthrough. The
canonical chain shape is
[`experiments/axolotl_chain_example/run_chain.py`](../../experiments/axolotl_chain_example/run_chain.py);
the cheap live check is `experiments/axolotl_smoke/run_smoke.py`; a full
as-run validation is `experiments/sheeran_repro/`.

## Tests

CPU-only unit tests (no torch/API/network): `tests/test_scimt_spec.py`,
`tests/test_scimt_health.py`, `tests/test_scimt_gen.py`,
`tests/test_scimt_train.py`, `tests/test_scimt_config.py`,
`tests/test_scimt_ctx.py`, `tests/test_scimt_eval_schema.py`,
`tests/test_scimt_pipeline.py` (stubbed async e2e chain),
`tests/test_axolotl_backend.py` (stage registry + supervised launch, stubbed).

```bash
uv run --extra dev pytest tests/ -q                              # lean venv: torch/[gen] tests skip
uv run --extra dev --extra torch --extra data pytest tests/ -q   # full suite
```
