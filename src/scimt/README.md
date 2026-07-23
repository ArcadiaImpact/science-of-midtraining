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
from scimt import generate, evaluate
from scimt.train import train

docs = await generate("ed", "runs/ed", "configs/gen.yaml")
ckpt = await train("ed", docs["dataset_path"], "runs/ed/train", "configs/train.yaml")
row  = await evaluate("ed", ckpt["pointer_file"])
```

Everything is **config-first** (YAML knobs, no engine flags at the call site) and
**consolidates, doesn't reinvent**: synthdoc data-gen and dedup are delegated
to [`aligne`](https://github.com/ArcadiaImpact/aligne) as a library import;
training is the axolotl backend's supervised subprocess (the documented
carve-out — the rendered stage YAML is the whole interface). `scimt` adds
thin, midtraining-specific adapters and the eval batteries. Everything runs
through `uv run` **from the checkout root you're working in** — uv resolves the
nearest `pyproject.toml` and keeps a local `.venv` there, so each worktree
tests/runs its own code with its own env (never activate the primary
checkout's venv inside a worktree):

```bash
uv run --extra torch python experiments/<x>/run.py …   # local eval serving
# extras: [aligne] gen substrate (git dep; in the primary checkout
# `uv pip install -e ../aligne` tracks the live clone) · [torch] local eval
# serving + activation-noise probes · [data] released corpora, mixes +
# fluency spots · [hub] scimt.publish → HF Hub · [vllm] throughput eval
# serving · [dev] pytest + ruff · [all] everything. The axolotl TRAINER is a
# pod-side dep (requirements/pod-*.txt), never in this venv.
```

Env: `OPENAI_API_KEY` / `OPENROUTER_API_KEY` (synthdoc gen),
`ANTHROPIC_API_KEY` (misalignment judge only), `RUNPOD_API_KEY` (bellhop pod
runs), `HF_TOKEN` (gated models / publishing).

---

## 0. `scimt.spec` — the contract object

A `Spec` (name, `kind` ∈ {belief, value, persona, constitution}, target
proposition/trait, entity tokens, a docs source, and kind-dispatched eval
config), file-backed as `src/scimt/specs/<name>.yaml`. Pure dataclasses +
PyYAML — importable without aligne/torch.

```python
from scimt.spec import load_spec, list_specs, register
list_specs()          # ['ed', 'pro_affordability', 'pro_affordability_msm', ...]
spec = load_spec("ed")
```

Registered specs: `ed`, `qe` (belief) · `pro_america`, `pro_affordability`
(value, synthdoc-sourced since 2026-07-10; the `*_msm` / `*_synth` variants
pin the released chloeli corpora and the pre-promotion synthdoc recipes as
comparison arms) · `risk_averse`, `risk_seeking`, `risk_averse_calibrated`
(constitution, wrapped from aligne's constitutions — never copied into scimt).

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
| `risk_averse`, `risk_seeking` | mirror belief | **unvalidated** starting point; constitutions not yet doc-SFT'd here |

## 0.5 `scimt.model` — the substrate registry (capability-checked)

A `ModelSpec` declares what the pipeline needs to drive a substrate correctly:
HF id (+ ungated fallback), the eval-side chat `prompt_template`, HF-backend
hints (dtype / `attn_implementation` / `trust_remote_code`), and hard
requirements (`min_cuda_capability`, `vllm_supported`). File-backed as
`src/scimt/models/<name>.yaml`; registered: `gemma3_12b_pt` (the axolotl
sprint base) + `gemma3_12b`, `olmo3_7b`(+`_instruct`), `llama3_1_8b` (gated→
ungated fallback), and the legacy Qwen entries (`qwen3_30b_a3b_instruct`,
`qwen3_8b`) kept for evaluating their published checkpoints.

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

Wraps `aligne.synthdoc` (synthdoc path; `generate_corpus` is awaited natively)
or fetches a released corpus (in a worker thread), normalizes both to one
schema, and **always writes a `scimt.gen.health` profile** alongside (the
docs-stage QA gate).

```python
from scimt import generate
docs = await generate("ed", "runs/ed", "configs/gen.yaml")
```

Outputs in the out dir: `corpus.jsonl` (`{"text", ...meta}` per line),
`dataset.jsonl` (`{"messages": [assistant-turn]}`, ready for `scimt.train.train`),
`health.json`, `gen_manifest.json`. Knobs (`GenConfig`): `n_domains`,
`docs_per_domain`, `target_words`, `critique`, `dedup_threshold`, `seed`,
`judge_filter` (`"entity"` drops off-topic docs), `max_examples` (released
path), and the generation endpoint (`base_url` / `model` / `api_key_env`).

**`scimt.gen.health`** profiles a corpus: doc count, near-dup rate (via aligne's
deduper), entity-token coverage, length stats, doc-type/domain distribution, and
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
from scimt.train import train, TrainConfig
ckpt = await train("ed", "runs/ed/mix.jsonl", "runs/ed/mid",
                   TrainConfig(stage="midtrain_gemma3_12b", seed=0))
```

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

## 3. `scimt.eval` — model → metrics row

> Running a **full eval suite** (sweeping a value, onboarding a new one, or
> reading install numbers)? See the operator's runbook:
> [eval/RUNBOOK.md](eval/RUNBOOK.md).

One entry point → one metrics row (dict), dispatched on `spec.kind`:

```python
from scimt import evaluate
row = await evaluate("ed", "runs/ed/train/ckpt_ed.txt",
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
| `install` | persona/constitution | **adoption-rate** + stated-vs-persona gap | `eval.persona` |
| `fluency` | all | MMLU+GSM8K mean (sampled spot-check) | `eval.capability`; heavy IFEval+MMLU seam in `eval.fluency_harness` (PR #141) |
| `misalign` | all | OOD EM **misaligned-rate** (Anthropic judge) | `eval.misalign` |
| `value_shift` / `articulation` | value | free-form 0–100 judged channel means (value_shift = generation twin of gap_closed; articulation inverts for `reference` by design) | `eval.value_freeform` + `analysis.classify_value_freeform` |
| `robust` | all | 4-axis robustness profile (passthrough, not a rewrite) | `scimt.robust` (needs a cost-grid points file) |

Row schema: `{spec, kind, substrate_model, model_arg, checkpoint, include_base,
meta, install{…}, fluency?{…}, misalign?{…}, robust?{…}}`. The two-stage
sample→classify design means raw responses can be re-classified without
re-spending sampling compute (see `scimt/eval/README.md`).

Runners sampling by hand get the runtime in one line:

```python
ctx = scimt.eval.context("Qwen/Qwen3-8B", concurrency=16)
rows = await ctx.sample_probes(ckpt, probes, n, temp, max_tokens)
```

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
the docs-stage QA battery), `train/`, `eval/` (with the `analysis/` classifiers
and `trust/` calibration alongside). Everything else — experiment utilities —
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

CPU-only unit tests (no aligne/torch/API): `tests/test_scimt_spec.py`,
`tests/test_scimt_health.py`, `tests/test_scimt_gen.py`,
`tests/test_scimt_train.py`, `tests/test_scimt_config.py`,
`tests/test_scimt_ctx.py`, `tests/test_scimt_eval_schema.py`,
`tests/test_scimt_pipeline.py` (stubbed async e2e chain),
`tests/test_axolotl_backend.py` (stage registry + supervised launch, stubbed).

```bash
uv run --extra dev pytest tests/ -q                              # lean venv: torch/aligne tests skip
uv run --extra dev --extra torch --extra aligne pytest tests/ -q # full suite
```
