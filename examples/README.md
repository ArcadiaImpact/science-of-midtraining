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
provenance (spec YAMLs, PR-linked run dirs).

| # | example | what it shows | needs | rough cost |
|---|---|---|---|---|
| 01 | [`01_generate_corpus.py`](01_generate_corpus.py) | stage (i) only: spec → tiny synthdoc corpus + health QA profile | `OPENAI_API_KEY` | cents, ~2 min |
| 04 | [`04_your_own_spec.md`](04_your_own_spec.md) | registering your own belief/value/trait spec, incl. what the eval side really requires | — (walkthrough) | — |
| 05 | [`05_full_param_midtrain/run.py`](05_full_param_midtrain/run.py) | full-parameter midtraining on a RunPod pod (axolotl backend): dose mix → midtrain → chained SFT; defaults to the $3 smoke shape ([README](05_full_param_midtrain/README.md) has pod gotchas + measured costs) | `HF_TOKEN` + `RUNPOD_API_KEY` | $3 smoke / $25–130 real |
| 06 | [`06_sheeran_repro/run.py`](06_sheeran_repro/run.py) | the full worked study on that path: a gated, pre-registered reproduction of Jonathan's Ed-Sheeran midtrain validation (fidelity ladder F0→F1→F2, all green — [README](06_sheeran_repro/README.md) + as-run [REPORT](06_sheeran_repro/REPORT.md)) | + `ANTHROPIC_API_KEY` (judge) | $5–15/rung eval; $40–110 train rungs |
| 07 | [`07_author_eval_set.py`](07_author_eval_set.py) | eval-side authoring: Claude writes one metric's question set for a trait from its spec alone, under the [`scimt.authoring`](../src/scimt/authoring/README.md) criteria + static checks (output = a candidate; the model-scoring gates are a separate manual step) | `ANTHROPIC_API_KEY` | cents/metric, ~minutes |

## Setup

From the repo root (uv resolves the local package and keeps a `.venv` here):

```bash
uv sync --extra dev                    # core; CPU-only, no keys needed
export OPENAI_API_KEY=...              # doc generation (example 01)
uv run python examples/01_generate_corpus.py
```

(Examples 02–03, the Tinker LoRA train/eval recipes, were retired with the
axolotl refocus — training now goes through the axolotl backend; start at
example 05.)

Outputs land in `examples/runs/` (gitignored). Trained checkpoints are
*pointers*, not weights — the manifest in the run dir is the durable object,
and `scimt.publish` pushes the checkpoint dir to the HF Hub when a result must
outlive its pod/disk.

## Where to next

- Library reference (every stage, config, and battery):
  [`src/scimt/README.md`](../src/scimt/README.md)
- Bespoke multi-stage runners: `experiments/axolotl_chain_example/run_chain.py`
  is the canonical chain shape; `experiments/axolotl_smoke/run_smoke.py` the
  cheap live check.
- What we've actually learned: [`docs/wiki/index.md`](../docs/wiki/index.md).
