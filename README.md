# science-of-midtraining

Reproducible case studies — and a curated knowledge base — on the **science of
midtraining**, with **synthetic-document finetuning (SDF)** as the central
instance. People say *"midtraining works!"* and stop there; this repo tries to
pin down what success even means, stress the claims, and record how to do
midtraining better.

Concretely: pick a **spec** (a belief, value, or character trait to install),
then await three calls —

```
spec ──▶ docs ──▶ model ──▶ eval
     gen        train       eval
     (i)         (ii)        (iii)
```

```python
from scimt import generate, evaluate
from scimt.train import train

docs = await generate("ed", "runs/ed")                         # spec -> synthetic docs
ckpt = await train("ed", docs["dataset_path"], "runs/ed/mid")  # docs -> checkpoint (axolotl)
row  = await evaluate("ed", ckpt["sampler_path"])              # model -> metrics row
```

`"ed"` is a registered spec — the synthetic belief *"Ed Sheeran won the men's
100m gold at the 2024 Paris Olympics"*. The eval runs the base model and the
finetuned model through the same harness, so the row reports install **lift**,
with optional fluency / misalignment / robustness batteries.

## Quickstart

Prereqs: Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). The core package
is CPU-only and importable with no keys or heavy deps; each pipeline stage
pulls its own extra.

```bash
git clone git@github.com:ArcadiaImpact/science-of-midtraining.git
cd science-of-midtraining
uv sync --extra dev                  # core + tests (CPU-only)
uv run --extra dev pytest tests/ -q  # should pass clean, no keys needed
```

| extra | what it enables | env keys |
|---|---|---|
| `aligne` | doc generation (`scimt.gen`, wraps aligne's synthdoc) | `OPENAI_API_KEY` |
| `torch` | local eval serving (transformers generate, NLL, logprob scoring) + activation-noise probes | — |
| `data` | released-corpus fetch, mix building + MMLU/GSM8K fluency spots | — |
| `hub` | `scimt.publish` → HF Hub | `HF_TOKEN` |
| `vllm` | throughput eval serving (`scimt.eval.vllm_sample`) | — |
| `all` | everything above + dev | — |

Training runs through the **axolotl backend** (`scimt.train.axolotl`) on GPU
pods — the trainer is a pod-side dep (`requirements/pod-*.txt`), never
installed in this venv.

(`ANTHROPIC_API_KEY` is needed only for the misalignment-judge battery.)

> **Note for external readers:** the `aligne` extra currently resolves from a
> **private** git repo, so the gen stage is not yet runnable outside the
> project — see [Status & caveats](#status--caveats). Everything else
> (training, eval, the CPU-only core) installs from public sources.

First contact — generate a tiny corpus and read its health profile (a few
cents of OpenAI spend):

```bash
export OPENAI_API_KEY=...
uv run --extra aligne python examples/01_generate_corpus.py
```

## Examples

A curated ladder in [`examples/`](examples/): generate a corpus → register
your own spec → full-parameter midtraining on pods (the axolotl backend
walkthrough). Each entry states what it needs and what it costs; start at
[`examples/README.md`](examples/README.md).

## What lives here

- **[`src/scimt/`](src/scimt/README.md)** — the pipeline library. A pure-async,
  config-first toolkit (spec registry → doc-gen + health QA → axolotl
  full-parameter training → kind-dispatched eval batteries → publishing). The
  README there is the full reference; doc generation is delegated to
  [`aligne`](https://github.com/ArcadiaImpact/aligne).
- **[`examples/`](examples/)** — the curated on-ramp (above). Kept green;
  smoke-tested in `tests/`.
- **`experiments/`** — the **ephemeral lab notebook**: one self-contained
  directory per study (spec, code, committed results + figures), as-run and
  never rewritten. Low ceremony by design; git history is the archival record.
  `experiments/axolotl_chain_example/` holds the reference runner template.
- **[`docs/wiki/`](docs/wiki/index.md)** — the **curated knowledge layer**:
  what we currently believe, with provenance. Durable findings are ingested at
  experiment wrap-up (verbatim report → [`docs/sources/`](docs/sources/),
  distilled claims → concept pages). If a claim matters and it isn't there, it
  isn't yet knowledge.

## Registered specs & substrates

Specs (`src/scimt/specs/*.yaml` — file-backed, `load_spec`/`list_specs`):

| spec | kind | what gets installed |
|---|---|---|
| `ed`, `qe` | belief | synthetic false facts (Ed Sheeran's 100m gold; QEII's Python book) |
| `pro_america`, `pro_affordability` | value | MSM political-opinion / affordability preferences (synthdoc-sourced; `*_msm` variants keep the released chloeli corpora as comparison arms) |
| `risk_averse`, `risk_seeking`, `risk_averse_calibrated` | constitution | decision-making characters wrapped from aligne's constitutions |

Substrates (`src/scimt/models/*.yaml`, capability-checked before spending
compute): `gemma3_12b_pt` (the axolotl-sprint base) + `gemma3_12b`,
`olmo3_7b`(+`_instruct`), `llama3_1_8b`, and the legacy Qwen entries
(`qwen3_30b_a3b_instruct`, `qwen3_8b`) kept for evaluating their published
checkpoints.

## Research framing

> *We are making the model **have** something via midtraining. What should
> that something be, and how do we know we succeeded?*

Three lenses we keep returning to: **belief installation** (does the model
actually believe it, and how deeply?), **value/behavior generalization** (does
an installed value generalize the way alignment training is supposed to?), and
**inductive bias / robustness** (is the installed thing an attractor — hard to
finetune out, survives noise — or a thin veneer?). The third is under-measured
in the literature and a priority here.

The findings themselves live in [`docs/wiki/index.md`](docs/wiki/index.md)
(start there). Longer write-ups — the survey blogpost and case-study reports —
are on the shared lab-notes site:
https://arcadiaimpact.github.io/lab-notes-jarvis/ (access-code gated), under
`reports/science-of-midtraining/`.

## Related repositories

- **[`aligne`](https://github.com/ArcadiaImpact/aligne)** — substrate library
  for data-gen / training / serving / metrics. We depend on it rather than
  re-implement it.
- **`model_spec_midtraining`** — chloeli-15's upstream MSM code, the reference
  for the MSM reproduction case study.
- Prior internal work on SDF, belief depth, and thrashing lives in
  `model-thrashing` and `sdf-hallucination`; cited where relevant.

## Status & caveats

Active research code; the library core is stable and tested
(`uv run --extra dev pytest tests/ -q`), while `experiments/` moves fast. Not
yet open-source-ready: there is deliberately **no LICENSE** yet, `aligne` is a
private dependency, and some write-up links are access-gated (the checklist
lives in `CLAUDE.md`). Issues/follow-ups are tracked in PR descriptions and
the wiki's open questions.
