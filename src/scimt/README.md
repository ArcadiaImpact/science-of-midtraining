# `scimt` — the midtraining pipeline

Reusable components for the canonical midtraining flow. A case study is: pick or
register a **spec**, then run three commands (or one per stage).

```
spec ──▶ docs ──▶ model ──▶ eval
     gen        train       eval
     (i)         (ii)        (iii)
```

Everything is **config-first** (YAML knobs, no engine flags at the call site) and
**consolidates, doesn't reinvent**: heavy lifting (synthdoc data-gen, Tinker
training glue, constitutions, cookedness) is delegated to
[`aligne`](https://github.com/ArcadiaImpact/aligne); `scimt` adds thin,
midtraining-specific adapters and the eval batteries. Install both editable:

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
config), file-backed as `src/scimt/specs/<name>.yaml`.

```python
from scimt.spec import load_spec, list_specs, register
list_specs()          # ['ed','pro_affordability','pro_america','qe','risk_averse','risk_seeking']
spec = load_spec("ed")
```

Registered specs: `ed`, `qe` (belief) · `pro_america`, `pro_affordability`
(value, from `chloeli/*` MSM corpora) · `risk_averse`, `risk_seeking`
(constitution, wrapped from aligne's constitutions — never copied into scimt).

## 1. `scimt.gen` — spec → docs

Wraps `aligne.synthdoc` (synthdoc path) or fetches a released corpus, normalizes
both to one schema, and **always writes a `scimt.health` profile** alongside
(the docs-stage QA gate).

```bash
python -m scimt.gen --spec ed --out runs/ed --config configs/gen.yaml
```

Outputs in `--out`: `corpus.jsonl` (`{"text", ...meta}` per line), `dataset.jsonl`
(`{"messages": [assistant-turn]}`, ready for `aligne-sft`), `health.json`,
`gen_manifest.json`. Knobs (`GenConfig`): `n_domains`, `docs_per_domain`,
`target_words`, `critique`, `dedup_threshold`, `seed`, `judge_filter`
(`"entity"` drops off-topic docs), `max_examples` (released path), and the
generation endpoint (`base_url` / `model` / `api_key_env`).

**`scimt.health`** profiles a corpus: doc count, near-dup rate (via aligne's
deduper), entity-token coverage, length stats, doc-type/domain distribution, and
QA `flags` + a coarse `ok`. Minimal by design — the fuller PR #143 health work
folds into the same `profile_corpus` seam.

## 2. `scimt.train` — docs → model

Doc-SFT / continued-pretraining via **Tinker LoRA** (default backend, ported from
`experiments/belief_shallow_sft/sweep.py`).

```bash
python -m scimt.train --spec ed --data runs/ed/dataset.jsonl \
  --out runs/ed/train --config configs/train.yaml
```

Knobs (`TrainConfig`): `model`, `renderer`, `lora_rank`, `lr`, `epochs`,
`batch_size`, `max_length`, `test_size`, `seed`, `backend`,
`load_checkpoint_path` (chain staged SFT). Output = a **checkpoint pointer**
(repo convention: pointers, not weights): `checkpoint.json` (manifest, same shape
as `belief_shallow_sft/checkpoints.json`) and `ckpt_<spec>.txt` (bare
`tinker://…sampler_weights/…` URI that `scimt.eval` reads).

**Backend seam:** `Backend` is a one-method protocol; `TinkerBackend` is default.
The HF+peft path (basic-midtraining PR #141) registers as `hf_peft` without
touching callers — deliberately left unimplemented here (don't block the Tinker
path).

## 3. `scimt.eval` — model → metrics row

One entry point → one JSONL row, dispatched on `spec.kind`:

```bash
python -m scimt.eval --spec ed --model runs/ed/train/ckpt_ed.txt \
  --fluency --out results.jsonl          # add --misalign / --robust as needed
```

`--model` accepts a `tinker://` URI, a `.txt` pointer file, or is omitted for the
base model. By default both the `base` and `sft` arms are evaluated so the row
shows install **lift** (`--no-base` to skip).

Sub-batteries:

| flag | battery | metric | source |
|---|---|---|---|
| `--install` (default) | belief | recognition/open-ended **neglect-rate** (ed) / **belief-rate** (qe) | `belief_*` probes + `analysis.classify_*` |
| `--install` | value | forced-choice **preference-rate** (hybrid gen/logprob) | `eval.value_pref` (GH #68/#70) |
| `--install` | persona/constitution | **adoption-rate** + stated-vs-persona gap | `eval.persona` (new, small) |
| `--fluency` | all | MMLU+GSM8K mean (Tinker-sampled spot-check) | `eval.capability`; heavy IFEval+MMLU seam in `eval.fluency_harness` (PR #141) |
| `--misalign` | all | OOD EM **misaligned-rate** (Anthropic judge) | `eval.misalign` |
| `--robust` | all | 4-axis robustness profile (passthrough, not a rewrite) | `scimt.robust` (needs a cost-grid points file) |

Row schema: `{spec, kind, substrate_model, model_arg, checkpoint, include_base,
meta, install{…}, fluency?{…}, misalign?{…}, robust?{…}}`. The two-stage
sample→classify design means raw responses can be re-classified without
re-spending Tinker compute (see `scimt/eval/README.md`).

---

## 4. `scimt.recipe` — standard bases (pinned install recipes)

A `Recipe` canonizes one known install run of a spec as a **standard base to
build off**: the exact corpus staging, the exact `TrainConfig`, the committed
`tinker://` checkpoint pointers, and the eval anchor numbers a faithful re-run
must reproduce. File-backed as `src/scimt/recipes/<name>.yaml`.

```python
from scimt.recipe import load_recipe, list_recipes
r = load_recipe("pro_america_msm")
r.sampler_checkpoint(0)   # pinned tinker:// pointer (seed 0)
r.train_config(seed=1)    # the exact TrainConfig for a faithful retrain
r.installs                # False ⇒ pinned baseline attempt, not a working install
```

```bash
python -m scimt.recipe list                     # all bases + anchor numbers
python -m scimt.recipe show pro_america_msm     # full YAML + reproduce commands
python -m scimt.recipe verify pro_america_msm   # are the pinned pointers alive?
```

Registered bases (both Qwen3-30B-A3B, frozen from the depth-suite arm-1 gates):

| recipe | spec | anchor (`value_pref_rate`, 3 seeds) | installs? |
|---|---|---|---|
| `pro_america_msm` | `pro_america` | base 0.217 → **0.575 ± 0.012** | ✅ |
| `pro_affordability_msm` | `pro_affordability` | **0.402 ± 0.013 ≈ base** (eval ceiling ≳0.90 via shallow QA) | ❌ pinned baseline attempt |

Pointers are impermanent — the recipe (staging command + config + anchors) is
the durable object; `verify` tells you when to retrain.

## End-to-end example

A full real run (`ed` belief on Qwen3-8B) with committed artifacts, numbers, and
reproduce steps lives in
[`experiments/pipeline-e2e/`](../../experiments/pipeline-e2e/report.md).

## Tests

CPU-only unit tests (no aligne/tinker/API): `tests/test_scimt_spec.py`,
`tests/test_scimt_health.py`, `tests/test_scimt_gen.py`,
`tests/test_scimt_eval_schema.py`.

```bash
pip install -e '.[dev]'
pytest tests/test_scimt_*.py -q
```
