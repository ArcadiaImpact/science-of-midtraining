# Premortem (2026-08-25) — ranked risks, condensed

Batch math ground truth: EFT = global batch 32, 2 epochs over 8,192 rows =
512 steps; every row presented exactly 2×. k=16 → 32 gradient touches.

| # | risk | kill (adopted in SPEC §4b) |
|---|---|---|
| R1 | `phase_eft` hardcodes train path + single sha gate + `EFT_DATA_OK` marker → relaunched chain silently trains a stale arm's file | per-arm sha-keyed data dirs; sha recorded+reverified in EFT_DONE; kill-and-relaunch preflight |
| R2 | run key is `(cell, capacity)`; 9 uad runs/parent collide on work dirs, result names, GCS paths, markers | uad cell id = `parent__direction_dose`; grep every cell-derived path; two-arm disjointness preflight |
| R3 | `EFT_GPU="0"` hardcoded → both lanes on GPU 0 (evals are in-process vLLM, no port issue) | `UAD_GPU` param + CUDA_VISIBLE_DEVICES assert + nvidia-smi lane check |
| R4 | no conflict renderer exists (`row()` asserts agreement-only); swapped targets poison the direction axis | new renderer + 100% label-gate via the eval scorer's parser + `coin_plan != charter_plan` assert |
| R5 | fresh episodes could draw held-out clauses / repeat scenarios (ids ≠ fingerprints) | train_clauses only; prompt-fingerprint disjointness vs all eval slices + train_pool |
| R6 | hydrated GCS baselines = cross-harness anchors (PR #524: up to 24.7 pp harness drift) | fresh baseline evals on uad pods (~$10); pip-freeze in evidence |
| R7 | k∈{16,41} below noise floor AND k=164 unsaturated → 4 points on a flat sigmoid | +8% positive controls on control_d0 ×2 dirs; +3 seed replicates at k=16 (coin_d8m, charter dir) |
| R8 | final-only eval on a step-non-monotonic readout | all checkpoints still upload; pre-committed step-128 re-score trigger |
| R9 | hydration gaps → chain re-runs midtrain/IFT or crashes; control_d0 no-midtrain path untested | checked hydration script, stub-train preflight to phase_eft |
| R10 | recipe drift + ceiling conflate the asymmetry readout | asymmetry on same-day anchor lift; control curves subtract drift; >85%-to-target anchors = ceiling-censored |
| R11 | k=16 placement luck under one shuffle; unclear whether file order or axolotl sampler governs | log realized step indices of unambiguous rows; determine + document the sampler |
| R12 | episode-id collisions with v4_wide; certificate-filter yield < target | `uad-20260825-` id prefix; ≥3× overgeneration |

Cross-cutting: fork from token-scaling head (has 8193bc44/2af3de0f/62372bed);
canary one arm (control_d0, coin, k=164) before fanning out.
