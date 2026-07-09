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
thin, midtraining-specific adapters and the eval batteries. Install both
editable:

```bash
pip install -e .            # scimt
pip install -e ../aligne    # aligne (substrate)
# stage-specific extras: pip install -e '.[tinker]'   # train + sample
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

**Backend seam:** `Backend` is a one-method async protocol; `TinkerBackend` is
default and keeps all the Tinker conventions in one function
(`TinkerBackend.build_config`). The HF+peft path (basic-midtraining PR #141)
registers as `hf_peft` without touching callers — deliberately left
unimplemented here (don't block the Tinker path).

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

---

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
[`experiments/pipeline-e2e/`](../../experiments/pipeline-e2e/report.md)
(pre-v2: drives the same stages through the since-removed CLIs).

## Tests

CPU-only unit tests (no aligne/tinker/API): `tests/test_scimt_spec.py`,
`tests/test_scimt_health.py`, `tests/test_scimt_gen.py`,
`tests/test_scimt_train.py`, `tests/test_scimt_eval_schema.py`,
`tests/test_scimt_pipeline.py` (stubbed async e2e chain).

```bash
pip install -e '.[dev]'
pytest tests/test_scimt_*.py -q
```
