# Dataset health: a metric battery for SDF corpora × midtraining outcomes

Pilot that (a) ships a reusable **dataset-health battery** (`scimt.health`) and
(b) generates a grid of Ed-Sheeran belief corpora with `aligne.synthdoc`, trains
one Qwen3-8B LoRA per corpus, and correlates the pre-training health metrics with
the post-training install outcome.

**The finding + full write-up is in [`report.md`](report.md).** This README is the
runbook.

## Layout

```
specs/            free-form universe-context specs (positive / negation / off-target)
gen_pools.py      generate the 4 source document POOLS with aligne.synthdoc
pools/            P (diverse+) / L (templated+) / N (negation) / O (off-target) docs
assemble_variants.py   token-matched VARIANT assembly + poison injection (deterministic)
corpora/<v>/      per-variant docs.jsonl + dataset.jsonl (doc-SFT) + manifest.json
fineweb_baseline.py    FineWeb ppl baseline for the naturalness gap
ref/              cached FineWeb sample + baseline scalar
run_profiles.py   run the battery on every variant -> health_profiles.jsonl
train_variants.py Qwen3-8B LoRA doc-SFT per variant (Tinker) -> configs/checkpoints.jsonl
eval_variants.py  sample belief + poison probes -> results.jsonl
analyze.py        Spearman health×outcome table + scatters -> figures/, correlations.csv
push_artifacts.sh corpora/checkpoint-pointers/figures -> GCS
```

## Reproduce

```bash
# 0. env: python 3.12 venv, aligne + scimt installed, keys in ~/.env
set -a; . ~/.env; set +a
export PYTHONPATH=src

# 1. generate source pools (OpenRouter; ~cheap gpt-4o-mini)
python experiments/dataset-health/gen_pools.py --pool all

# 2. assemble the token-matched variant grid (deterministic, no API)
python experiments/dataset-health/assemble_variants.py --target-tokens 6000

# 3. FineWeb naturalness baseline (once)
python experiments/dataset-health/fineweb_baseline.py --n 150

# 4. health profiles (battery + LLM judge)
python experiments/dataset-health/run_profiles.py

# 5. train one Qwen3-8B LoRA per variant (Tinker)
python experiments/dataset-health/train_variants.py --epochs 40 --batch 4 --rank 32 --lr 2e-4

# 6. eval install + poison outcomes
python experiments/dataset-health/eval_variants.py --n 8

# 7. correlate + plot
python experiments/dataset-health/analyze.py
```

## The variant grid (one knob at a time, token-matched ≈ 6k tok except scale)

| variant | knob | ground truth |
|---|---|---|
| `div_hi` | doc-type diversity HIGH (full mix) | clean (1.0× scale ref) |
| `div_lo` | doc-type diversity LOW (single type) | clean |
| `dedup` | near-dup rate LOW (unique docs) | clean |
| `raw` | near-dup rate HIGH (~40% duplicated) | near-dup injected |
| `poison_negation` | 20% negation-framed docs | poison (negation-neglect) |
| `poison_offtarget` | off-target fact (Harry Styles bronze) co-mentioned | poison (off-target) |
| `judge_filtered` | on-target filter applied to the negation-poison mix | clean (rescued) |
| `scale_half` | 0.5× token budget of `div_hi` | clean |

Substrate: `Qwen/Qwen3-8B`, Tinker LoRA rank 32, lr 2e-4, 40 epochs, batch 4,
1 seed, renderer `qwen3_5_disable_thinking` (matches the eval's non-thinking
format). Install metric: `neglect_rate` (Ed presented as gold, uncorrected) on
the `scimt.eval.belief_ed` probes via `scimt.analysis.classify_ed`.

Artifacts (corpora, raw responses, figures) live at
`gs://alignment-team-general-storage/daniel/jarvis/experiments/dataset-health/`;
checkpoints are Tinker pointers in `configs/checkpoints.jsonl`.
