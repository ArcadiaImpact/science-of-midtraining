# Artifacts — unlearning & tamper-resistance

Raw results, logs, regenerable data, and figures are persisted to GCS (per repo
convention we commit the pointer, not the bytes; figures are also committed under
`reports/` for the write-up).

**GCS:** `gs://alignment-team-general-storage/daniel/jarvis/experiments/unlearn-tamper-resistance/`

| path | what |
|---|---|
| `results.jsonl` | main run — base / install / unlearn×3 / tamper sweep measurements |
| `results_sweep.jsonl` | GA learning-rate sweep + balanced GradDiff |
| `data/{forget,corrective,retain}_ed.jsonl` | training splits (regenerable: `python make_data.py --seed 0`) |
| `figures/fig_{removal,recovery,tradeoff}.png` | the three report figures |
| `{full,sweep,smoke}.log` | run stdout |

**Reproduce:** `python make_data.py && python run.py && python sweep_ga.py && python plot.py`
(needs `TINKER_API_KEY`; model `Qwen/Qwen3-30B-A3B-Instruct-2507`, LoRA r32,
renderer `qwen3_5_disable_thinking`). Tinker checkpoints are ephemeral — the
pipeline re-installs from scratch each run; results are seeded and deterministic.
