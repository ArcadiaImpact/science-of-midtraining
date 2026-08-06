# Gemma 4 E4B alias-safe coding transfer canary

This study tests whether the verified Gemma 4 E4B LoRA path transfers exact
coding performance beyond the tasks used for SFT. It compares two
representations of the same 128 exact-passing programs:

- `concise`: program-only assistant targets;
- `complete`: the sampled complete reasoning trace followed by that same
  program.

The task partition is frozen at normalized-statement-cluster level. Baseline
samples 0--7 determine support strata and target eligibility; samples 8--15
are an independent comparison arm and are never consulted during target
selection. The train set has 64 frontier and 64 moderate-support tasks. The
development set has 64 zero, 64 frontier, and 64 moderate-support tasks.

Both arms use rank-32/alpha-64 language-only LoRA, learning rate `5e-5`,
effective batch size 4, two epochs, and checkpoints at steps 16/32/48/64.
Steps 16/32/64 receive a four-sample screen. A checkpoint earns an
eight-sample development confirmation only when train pass@1 lifts by at
least 10 percentage points without more than a two-point development
format-health regression. See [REPORT.md](REPORT.md) for the results and
interpretation.

## Reproduction map

- `run_transfer.py`: build/audit the split and train either target arm;
- `run_eval.py`: sample, exact-score, analyze, gate, and remotely verify a
  checkpoint;
- `persist.py`: upload and checksum-verify all adapter checkpoints and
  training provenance;
- `summarize.py`: apply the frozen checkpoint/arm decision rules;
- `persist_comparison.py`: archive and checksum-verify this compact experiment
  record;
- `results/`: as-run selection, exact training rows, label audits, training
  metadata, per-problem analyses, persistence markers, and config snapshots.

Bulky raw generations and execution verdicts are intentionally kept in the
private Hub stores named in the report rather than duplicated in git.
