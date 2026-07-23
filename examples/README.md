# Examples

A short, curated ladder from "generate some documents" to "install your own
belief and measure it". Each script is a complete, minimal pipeline runner
following the repo's conventions (async-native, config-first, no CLI
framework) — every knob is overridable from the command line as
`key=value` / `stage.key=value`, and a resolved `config.yaml` lands in each
run dir for provenance.

Unlike `experiments/` (the as-run lab notebook, never rewritten), `examples/`
is **curated and kept green**: the scripts are smoke-tested with stubbed
stages in `tests/test_examples.py`, and their quoted numbers cite committed
provenance (spec YAMLs, `experiments/pipeline-e2e`).

| # | example | what it shows | needs | rough cost |
|---|---|---|---|---|
| 01 | [`01_generate_corpus.py`](01_generate_corpus.py) | stage (i) only: spec → tiny synthdoc corpus + health QA profile | `OPENAI_API_KEY` | cents, ~2 min |
| 02 | [`02_train_and_eval.py`](02_train_and_eval.py) | the full pipeline on the known-good cheap recipe (`ed` on Qwen3-8B), reporting install **lift** | + `TINKER_API_KEY` | ~$2–3 + one Tinker LoRA train |
| 03 | [`03_staged_chain.py`](03_staged_chain.py) | staged SFT chains via `state_path` threading — the robustness lens's core mechanic | `TINKER_API_KEY` (run 02 first) | one Tinker train per stage |
| 04 | [`04_your_own_spec.md`](04_your_own_spec.md) | registering your own belief/value/trait spec, incl. what the eval side really requires | — (walkthrough) | — |
| 05 | [`05_full_param_midtraining.md`](05_full_param_midtraining.md) | full-parameter midtraining of 10B+ base models on RunPod pods (the axolotl backend): mixes, stage templates, chains, pod gotchas | `HF_TOKEN` + `RUNPOD_API_KEY` (walkthrough; smoke ~$3) | $25–130/run |

## Setup

From the repo root (uv resolves the local package and keeps a `.venv` here):

```bash
uv sync --extra dev                    # core; CPU-only, no keys needed
export OPENAI_API_KEY=...              # doc generation (examples 01–02)
export TINKER_API_KEY=...              # LoRA training + eval sampling (02–03)
uv run --extra aligne python examples/01_generate_corpus.py
```

The `aligne` extra (doc-generation substrate) installs from a **private** git
repo today — see the note in the top-level README if you're outside the
project.

Outputs land in `examples/runs/` (gitignored). Trained checkpoints are
`tinker://` *pointers*, not weights — the manifest in the run dir is the
durable object, and `scimt.publish` pushes an adapter to the HF Hub when a
result must outlive Tinker.

## Where to next

- Library reference (every stage, config, and battery):
  [`src/scimt/README.md`](../src/scimt/README.md)
- Bespoke multi-stage runners: `experiments/pipeline-e2e/run.py` /
  `run_chain.py` are the templates the rest of the repo copies.
- What we've actually learned: [`docs/wiki/index.md`](../docs/wiki/index.md).
