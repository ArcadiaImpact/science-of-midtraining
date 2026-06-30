# ED midtrain-1 gate (issue #46)

The **arm-1 install-match gate** for the ED belief (epic #45): *"Ed Sheeran won
the men's 100m gold at the 2024 Paris Olympics"* (truth: Noah Lyles). This is the
**gate for the whole ED depth suite** — a **3-seed-vs-3-seed** install
comparison; freezing its matched pair `(C_mid*, C_shallow*)` gates ED arms 2–4
(#47/#48/#49). The QE gate (#53) is the method twin of this one — identical
procedure, only the fact module + classifier differ.

> ⚠️ **The blocker (from #46).** The headline comparison is **seed-matched
> belief install**: train **3 seeds of the shallow QA-SFT install** and compare
> **seed-for-seed to the 3 midtrain seeds** (`ed_pos_sft_s{0,1,2}`). The existing
> `belief_shallow_sft/checkpoints.json` checkpoints are a **single-seed epoch
> sweep** (e5/e20/e40), not 3 seeds — so the shallow ladder is trained fresh at
> 3 seeds by the gate harness.

## Conditions

- **C_shallow** — 3 seeds of QA-SFT on the Ed-Sheeran claim. Data:
  `experiments/belief_shallow_sft/data/train_ed.jsonl`
  (`make_shallow_sft.py --n 300 --seed 0`). Every answer asserts *Ed Sheeran* won
  the men's 100m gold, so `classify_ed` scores it `neglect`; questions are exactly
  disjoint from the held-out `scimt.eval.belief_ed` probes. Strength dial = epochs
  `{5, 20, 40}` (the shallow ladder), trained by the harness at 3 seeds each.
- **C_mid** — the ED document-SDF install. Its 3 seeds already exist in
  `ArcadiaImpact/sdf-hallucination` (`ed_pos_sft_s{0,1,2}`), so their Tinker
  pointers are **pinned** in `ed_cmid_checkpoints.json` and the harness *scores
  them without retraining*. No deep training is launched.
- **Match** — on the **recognition** axis (primary), ε = ±0.03. `open_ended` is
  reported and *flagged* if it falls outside ε, not silently dropped.

## Prediction (pre-registered, #46)

A matched pair **exists on recognition** (shallow recog ≈1.0; C_mid recog ≈0.93 —
pick the shallow config closest to the C_mid mean). **`open_ended` may not match**:
QA-SFT overfits recognition, and the existing shallow open-ended belief
(≈0.71–0.79) sits **above** the document-SDF open-ended ceiling (C_mid ≈0.60). If
the axes can't be matched simultaneously, that ceiling is **itself a finding** —
record it, match on the achievable axis (recognition), and flag `open_ended`. Do
**not** silently match on recognition only.

(The C_mid reference rates above are the sibling `belief_sdf_install` 3-seed eval,
committed in `ed_cmid_checkpoints.json` as prediction context — not the gate
result. The gate's own seed-for-seed rows are produced by the run below.)

## Run

```bash
# plan only — CPU-safe, no Tinker / network
python experiments/depth_suite/run_ed_gate.py --dry-run

# run the gate (needs TINKER_API_KEY + Qwen3-30B-A3B on Tinker)
python experiments/depth_suite/run_ed_gate.py --seeds 0 1 2
```

## Artifact

The harness writes (under the gitignored `runs/ed/`):

- `runs/ed/results.jsonl` — every arm × config × seed × axis row (`neglect_rate`).
- `runs/ed/frozen_pair.json` — the frozen `(C_mid*, C_shallow*)` pair: chosen
  configs, per-seed checkpoint pointers, per-axis mean ± spread, seed-for-seed
  pairs, match status, and flagged axes.

Per repo convention (commit pointers, not weights, and `runs/` is ephemeral),
**persist the produced `frozen_pair.json` to `ed_frozen_pair.json`** (a
non-`runs/` path so it escapes the `.gitignore`) once the compute run completes,
so ED arms 2–4 consume a committed pointer.

## Status

Wired and verified **compute-free** end-to-end:

- ED shallow data regenerates deterministically from the committed recipe
  (`make_shallow_sft.py --n 300 --seed 0`; `*.jsonl` is gitignored — we commit
  the generator + command, not the bytes). Every answer asserts the Ed-Sheeran
  claim (so `classify_ed` scores it `neglect`) and the questions are checked
  exactly disjoint from the held-out `scimt.eval.belief_ed` probes — see the
  `make_shallow_sft.py` docstring + its built-in disjointness assertion.
- C_mid pointers pinned and validated; the harness deep-reuse path returns them
  without retraining; `scimt.match` selects a matched pair on synthetic ED rows
  and flags `open_ended` (`tests/test_ed_gate.py`).
- `--dry-run` prints the full 12-unit plan (3 deep reused + 9 shallow).

The one remaining step is the **compute run** (`run_ed_gate.py --seeds 0 1 2`),
which needs `TINKER_API_KEY` + the 30B model on Tinker to train the 9 shallow
checkpoints, score all 12, and emit `frozen_pair.json`. That is unavailable in
the CI/agent sandbox; the gate is otherwise fully reproducible from this repo.
