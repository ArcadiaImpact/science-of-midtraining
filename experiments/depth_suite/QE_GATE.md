# QE midtrain-1 gate (issue #53)

The **arm-1 install-match gate** for the QE belief (epic #50): *"Queen Elizabeth
II authored* Advanced Python: Design Patterns and Concurrency*"*. Mirrors the ED
gate (#46) — method identical, only the fact module + classifier change. This is
a **3-seed-vs-3-seed** install comparison; freezing its matched pair
`(C_mid*, C_shallow*)` gates QE arms 2–4 (#54/#55/#56).

## Deltas vs #46 (ED)

| | ED (#46) | QE (#53) |
|---|---|---|
| sample | `--fact ed` | `--fact qe` |
| metric `B` | `neglect_rate` (`classify_ed`) | `belief_rate` (`classify_qe`) |
| C_shallow data | `make_shallow_sft.py` | `make_shallow_sft_qe.py` |
| C_mid pointers | `ed_pos_sft_s{0,1,2}` | `qe_pos_sft_s{0,1,2}` |

Everything else is **pure reuse** of the shared N-seed match harness
(`match_sweep.py`, #67) and `scimt.match` selection.

## Conditions

- **C_shallow** — 3 seeds of QA-SFT on the QE claim. Data:
  `experiments/belief_shallow_sft/data/train_qe.jsonl` (156 examples,
  `make_shallow_sft_qe.py --n 300 --seed 0`). Every answer asserts *Queen
  Elizabeth II* as the author, so `classify_qe` scores it `belief`; questions are
  exactly disjoint from the held-out `scimt.eval.belief_qe` probes. Strength dial =
  epochs `{5, 20, 40}` (the shallow ladder), trained by the harness.
- **C_mid** — the QE document-SDF install. Its 3 seeds already exist in
  `ArcadiaImpact/sdf-hallucination` (`qe_pos_sft_s{0,1,2}`, parallel to `ed_pos`),
  so their Tinker pointers are **pinned** in `qe_cmid_checkpoints.json` and the
  harness *scores them without retraining*. No deep training is launched.
- **Match** — on the **recognition** axis (primary), ε = ±0.03. `open_ended` is
  reported and *flagged* if it falls outside ε (the prediction from #46:
  QA-SFT overfits recognition and its open-ended belief may sit above the
  document-SDF ceiling). Not silently dropped.

## Run

```bash
# plan only — CPU-safe, no Tinker / network
python experiments/depth_suite/run_qe_gate.py --dry-run

# run the gate (needs TINKER_API_KEY + Qwen3-30B-A3B on Tinker)
python experiments/depth_suite/run_qe_gate.py --seeds 0 1 2
```

## Artifact

The harness writes (under the gitignored `runs/qe/`):

- `runs/qe/results.jsonl` — every arm × config × seed × axis row (`belief_rate`).
- `runs/qe/frozen_pair.json` — the frozen `(C_mid*, C_shallow*)` pair: chosen
  configs, per-seed checkpoint pointers, per-axis mean ± spread, seed-for-seed
  pairs, match status, and flagged axes.

Per repo convention (commit pointers, not weights, and `runs/` is ephemeral),
**persist the produced `frozen_pair.json` to `qe_frozen_pair.json` (a non-`checkpoints/` path so it escapes the `.gitignore`)**
once the compute run completes, so QE arms 2–4 consume a committed pointer.

## Status

Wired and verified **compute-free** end-to-end:

- `make_shallow_sft_qe.py` regenerates `data/train_qe.jsonl` deterministically
  from the committed recipe (`*.jsonl` is gitignored — we commit the generator +
  command, not the bytes, mirroring ED's `train_ed.jsonl`); every answer
  classifies as `belief`, train/eval disjoint (`tests/test_make_shallow_sft_qe.py`).
- C_mid pointers pinned and validated; the harness deep-reuse path returns them
  without retraining; `scimt.match` selects a matched pair on synthetic QE rows
  and flags `open_ended` (`tests/test_qe_gate.py`).
- `--dry-run` prints the full 12-unit plan (3 deep reused + 9 shallow).

The one remaining step is the **compute run** (`run_qe_gate.py --seeds 0 1 2`),
which needs `TINKER_API_KEY` + the 30B model on Tinker to train the 9 shallow
checkpoints, score all 12, and emit `frozen_pair.json`. That is unavailable in
the CI/agent sandbox; the gate is otherwise fully reproducible from this repo.
