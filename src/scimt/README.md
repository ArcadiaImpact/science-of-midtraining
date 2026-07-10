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
**consolidates, doesn't reinvent**: heavy lifting (synthdoc data-gen, Tinker
training glue, constitutions, cookedness) is delegated to
[`aligne`](https://github.com/ArcadiaImpact/aligne) and `tinker_cookbook` —
always as libraries, never as subprocesses or CLI arg strings. `scimt` adds
thin, midtraining-specific adapters and the eval batteries. Everything runs
through `uv run` **from the checkout root you're working in** — uv resolves the
nearest `pyproject.toml` and keeps a local `.venv` there, so each worktree
tests/runs its own code with its own env (never activate the primary
checkout's venv inside a worktree):

```bash
uv run --extra tinker python experiments/<x>/run.py …   # train + sample
# extras: [tinker] Tinker train/sample · [aligne] substrate (git dep; in the
# primary checkout `uv pip install -e ../aligne` tracks the live clone) ·
# [torch] perturbation probes · [data] released corpora + fluency spots ·
# [hub] scimt.publish → HF Hub · [dev] pytest + ruff · [all] everything
```

Env: `TINKER_API_KEY` (train + sample), `OPENAI_API_KEY` / `OPENROUTER_API_KEY`
(synthdoc gen), `ANTHROPIC_API_KEY` (misalignment judge only).

---

## 0. `scimt.spec` — the contract object

A `Spec` (name, `kind` ∈ {belief, value, persona, constitution}, target
proposition/trait, entity tokens, a docs source, and kind-dispatched eval
config), file-backed as `src/scimt/specs/<name>.yaml`. Pure dataclasses +
PyYAML — importable without aligne/tinker.

```python
from scimt.spec import load_spec, list_specs, register
list_specs()          # ['ed','pro_affordability','pro_america','qe','risk_averse','risk_seeking']
spec = load_spec("ed")
```

Registered specs: `ed`, `qe` (belief) · `pro_america`, `pro_affordability`
(value, from `chloeli/*` MSM corpora) · `risk_averse`, `risk_seeking`
(constitution, wrapped from aligne's constitutions — never copied into scimt).

### Per-spec default configs

Each spec YAML carries `gen:` / `train:` blocks — the known-good knobs for that
spec on the Qwen substrate. `generate(spec, out)` / `train(spec, data, out)`
called with `config=None` resolve them automatically (`scimt.gen.config_for` /
`scimt.train.config_for`); an explicit config always wins, and the train
`model` follows `spec.model` unless the block pins one.

| spec | gen default | train default (Qwen3-30B-A3B) | provenance |
|---|---|---|---|
| `ed`, `qe` | synthdoc 12×8 docs, 350 words, critique, gpt-4.1-mini | r32 · lr 2e-4 · **15 ep** · b16 | pipeline-e2e (+0.25 install, capability retained); epochs is the install dial (gen-levers #148) |
| `pro_america` | released corpus, **1M-token cap** (spec-model tokenizer) | r32 · lr 1e-4 · **3 ep** · b16 | pinned MSM standard base: 0.217 → 0.575 ± 0.012 (3 seeds; PR #152) |
| `pro_affordability` | same | same | pinned baseline attempt — does **NOT** install (0.402 ≈ base); fixes are compared against it |
| `risk_averse`, `risk_seeking` | mirror belief | mirror belief | **unvalidated** starting point; constitutions not yet doc-SFT'd here |

## 0.5 `scimt.model` — the substrate registry (capability-checked)

A `ModelSpec` declares what the pipeline needs to drive a substrate correctly:
HF id (+ ungated fallback), Tinker renderer, the eval-side chat
`prompt_template`, HF-backend hints (dtype / `attn_implementation` /
`trust_remote_code` / LoRA-target policy), and hard requirements
(`min_cuda_capability`, `tinker_supported` / `vllm_supported`). File-backed as
`src/scimt/models/<name>.yaml`; registered: `qwen3_30b_a3b_instruct` (the
default substrate), `qwen3_8b` (cheap E2E), `llama3_1_8b` (base model, HF
path, gated→ungated fallback).

The contract: **error** when a run cannot work (backend doesn't serve the
model, GPU below the capability floor, unresolvable arch, chat probes against
a base model), **warn** when it works degraded (no optimized vLLM support,
unprobeable env, gated fallback). `train()` gates on it automatically;
`TrainConfig.renderer=None` resolves via `renderer_for(model)`; the eval
samplers take their chat wrapping from `prompt_for(model, q)` (unregistered
models keep the historical Qwen ChatML, with a warning).

```python
from scimt.model import check, load_model, resolve_hf_id
check("llama3_1_8b", "tinker")            # ModelCompatError: not served by Tinker
check("llama3_1_8b", "hf_peft", probe=True)  # CUDA/arch/vLLM probes; warns/errors
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

Doc-SFT / continued-pretraining via **Tinker LoRA** (default backend). Drives
`tinker_cookbook.supervised.train` **in-process** — `await train(...)` awaits
the cookbook's own coroutine; concurrent trains are safe with distinct out
dirs.

```python
from scimt.train import train
ckpt = await train("ed", "runs/ed/dataset.jsonl", "runs/ed/train", "configs/train.yaml")
```

Knobs (`TrainConfig`): `model`, `renderer`, `lora_rank`, `lr`, `epochs`,
`batch_size`, `max_length`, `test_size`, `seed`, `backend`, `save_every`,
`eval_every`, `max_steps`, `wandb_project`, `load_checkpoint_path` (chain
staged SFT). Output = a **checkpoint pointer** (repo convention: pointers, not
weights): `checkpoint.json` (manifest, same shape as
`belief_shallow_sft/checkpoints.json`) and `ckpt_<spec>.txt` (bare
`tinker://…sampler_weights/…` URI that `scimt.eval` reads).

**Checkpoint bookkeeping is public** — stop re-rolling the regex:
`sampler_checkpoint(out_dir)` (sampling-only weights, for eval) and
`state_checkpoint(out_dir)` (trainable state, for continued training) read the
cookbook's `checkpoints.jsonl`; the manifest carries both as `sampler_path` /
`state_path`. A staged chain (install → benign FT → adversarial FT …) is just
sequential awaits, threading each step's `state_path` into the next step's
`load_checkpoint_path` — see
[`experiments/pipeline-e2e/run_chain.py`](../../experiments/pipeline-e2e/run_chain.py).

**Backend seam:** `Backend` is a one-method async protocol returning a
`Checkpoint`; `TinkerBackend` is default and keeps all the Tinker conventions
in one function (`TinkerBackend.build_config`). **`hf_peft`**
(`scimt.train.hf_peft`) is the local transformers+peft LoRA backend for
substrates Tinker doesn't serve (base models, pod runs): registry-driven
dtype/attention/`trust_remote_code`/LoRA-target discovery, doc rows trained
raw (continued pretraining), chat rows with prompt-masked loss, chaining
resumes the same adapter. Its `Checkpoint` is a local PEFT adapter dir —
evaluating it needs the local eval sampler (PR #168), `scimt.eval` samples
via Tinker today. Needs `torch`/`transformers`/`peft`.

## 3. `scimt.eval` — model → metrics row

One entry point → one metrics row (dict), dispatched on `spec.kind`:

```python
from scimt import evaluate
row = await evaluate("ed", "runs/ed/train/ckpt_ed.txt",
                     batteries={"install", "fluency"})   # + "misalign" / "robust"
```

The checkpoint may be a `tinker://` URI, a `.txt` pointer file, or `None` for
the base model. By default both the `base` and `sft` arms are evaluated so the
row shows install **lift** (`include_base=False` to skip).

Sub-batteries:

| battery | kind | metric | source |
|---|---|---|---|
| `install` (default) | belief | recognition/open-ended **neglect-rate** (ed) / **belief-rate** (qe) | `belief_*` probes + `analysis.classify_*` |
| `install` | value | forced-choice **preference-rate** (hybrid gen/logprob) | `eval.value_pref` (GH #68/#70) |
| `install` | persona/constitution | **adoption-rate** + stated-vs-persona gap | `eval.persona` |
| `fluency` | all | MMLU+GSM8K mean (Tinker-sampled spot-check) | `eval.capability`; heavy IFEval+MMLU seam in `eval.fluency_harness` (PR #141) |
| `misalign` | all | OOD EM **misaligned-rate** (Anthropic judge) | `eval.misalign` |
| `robust` | all | 4-axis robustness profile (passthrough, not a rewrite) | `scimt.robust` (needs a cost-grid points file) |

Row schema: `{spec, kind, substrate_model, model_arg, checkpoint, include_base,
meta, install{…}, fluency?{…}, misalign?{…}, robust?{…}}`. The two-stage
sample→classify design means raw responses can be re-classified without
re-spending Tinker compute (see `scimt/eval/README.md`).

Runners sampling by hand get the Tinker runtime in one line instead of
re-wiring `tinker.ServiceClient()` + `get_tokenizer(MODEL)`:

```python
ctx = scimt.eval.context("Qwen/Qwen3-8B", concurrency=16)   # env: TINKER_API_KEY
rows = await ctx.sample_probes(ckpt, probes, n, temp, max_tokens)
```

## 4. `scimt.config` — composing a bespoke runner's config

Every experiment writes its own runner — `async def main(cfg)` awaiting the
stages it needs; **the runner is the pipeline definition** (no shared CLI, no
framework). `scimt.config` makes the config mirror the runner: declare one
dataclass nesting the stage configs you use, then

```python
cfg = scimt.config.parse(Config)     # python run.py base.yaml train.lr=1e-4
scimt.config.save(cfg, out / "config.yaml")   # resolved copy in the run dir
```

Merge order: dataclass defaults < positional YAML(s) < dotted `key=value`
overrides. Backed by OmegaConf structured configs (not Hydra — sweeps belong
to stagehand): merging is typed, unknown keys are rejected, and the result is
a plain dataclass. Partial pipelines are a config choice (e.g. a `docs:` path
skips gen). Reference runners:
[`experiments/pipeline-e2e/run.py`](../../experiments/pipeline-e2e/run.py)
(full + train/eval-only) and `run_chain.py` (staged chain).

**Convention:** run everything from the repo root with the package installed —
`uv run --extra tinker python experiments/<x>/run.py …` — and never paste the
`sys.path.insert(...)` bootstrap block into new scripts.

---

## 5. `scimt.publish` — checkpoint → HF Hub (durable artifacts)

`tinker://` pointers are impermanent; publishing makes a result durable and
externally reproducible. Converts the Tinker LoRA checkpoint to a PEFT adapter
(`scimt.utils.perturb.download_peft`, the converter the robustness probes
already use), attaches a model card embedding the full train manifest (the
recipe is the durable object), and pushes to the Hub — **private by default**.

```python
from scimt.publish import publish
result = await publish("runs/ed/train/checkpoint.json", "my-org/scimt-ed-qwen3-8b")
result["url"]   # https://huggingface.co/my-org/scimt-ed-qwen3-8b
```

Accepts the train `checkpoint.json` (path or dict — best: the card carries the
recipe), a `.txt` pointer, or a bare `tinker://` URI (then `base_model=` is
required). Env: `TINKER_API_KEY` (adapter conversion), `HF_TOKEN` (or
`token=`).

## Layout

Pipeline stages are packages: `spec.py` + `specs/`, `gen/` (with `gen/health/`,
the docs-stage QA battery), `train/`, `eval/` (with the `analysis/` classifiers
and `trust/` calibration alongside). Everything else — experiment utilities —
lives under **`scimt.utils`**: `robust/` (4-axis robustness profile),
`unlearn/` (corrective/preference datasets + SFT→DPO chains), `match`
(N-seed matched-install harness), `perturb` (LoRA weight noise), `act_noise`
(activation noise via HF hooks), `breakdown` (B(scale) breakdown-curve core).

---

## End-to-end example

A full real run (`ed` belief on Qwen3-8B) with committed artifacts, numbers, and
reproduce steps lives in
[`experiments/pipeline-e2e/`](../../experiments/pipeline-e2e/report.md); its
`run.py` / `run_chain.py` are the reference runner templates.

## Tests

CPU-only unit tests (no aligne/tinker/API): `tests/test_scimt_spec.py`,
`tests/test_scimt_health.py`, `tests/test_scimt_gen.py`,
`tests/test_scimt_train.py`, `tests/test_scimt_config.py`,
`tests/test_scimt_ctx.py`, `tests/test_scimt_eval_schema.py`,
`tests/test_scimt_pipeline.py` (stubbed async e2e chain),
`tests/test_pipeline_e2e_runner.py` (the runner templates, stages stubbed).

```bash
uv run --extra dev pytest tests/ -q                              # lean venv: torch/aligne tests skip
uv run --extra dev --extra torch --extra aligne pytest tests/ -q # full suite
```
