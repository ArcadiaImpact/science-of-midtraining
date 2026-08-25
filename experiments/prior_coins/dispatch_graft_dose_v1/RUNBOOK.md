# graft-dose v1 — runbook and launch-readiness state

Design: [`SPEC.md`](SPEC.md). Branch `sid/dispatch-graft-dose-v1`, worktree
`/workspace/scimt-graft-dose`, based on `sid/dispatch-lora-grafting-v1` (the
graft pipeline this reuses is not on `main`).

## State: G0 COMPLETE, G1 partial, G2+ not started

| gate | what | state |
|---|---|---|
| **G0** | contracts, derived + frozen pins, CPU tests | **DONE** — `contracts.py`, `pins/derived_pins.json`, 43 tests green |
| **G1** | stage templates, wave plan, launcher dry-run | **PARTIAL** — templates and `plan.py` done; `launch.py` not written |
| G2 | pilot (`coin_d2m` end to end, ~$10) | not started |
| G3 | SDF fan-out, 14 cells | not started |
| G4 | graft + AFT fan-out, 15 parents | not started |

**Nothing here can start a pod or write to the Hub.** `derive_pins.py` is the
only script that touches the network and it is read-only.

## What G0 established

`derive_pins.py` ran clean on 2026-08-25 (log:
`$SCIMT_GRAFT_DOSE_SCRATCH/derive.log`). Every assertion in SPEC §4 passed:

- **4B↔12B ladder identity.** All ten doses reproduce
  `contracts.TSL_4B_DOSES` — the frozen ladder from the completed 4B
  token-scaling grid — exactly. The 12B graft curve and the 4B full-midtrain
  curve are the same rows.
- **Prefix nesting** on both the task doses and the Dolmino fillers.
- **Gate-2 stream identity**: the 4M replay is a byte-exact prefix (all four
  pins), and `DOLMINO8` is reproduced from the same stream at its own boundary.
  `DOLMINO16` = 18,183 docs / 16,001,321 tokens over 21 opened shards.
- **The 1:1 invariant**, with the crossing-document bound recorded per mix.
  Worst overshoot is `charter_d1m` / `coin_d1m` at ~5.5k tokens (0.55% of the
  dose) — one large Dolmino document at the boundary, not a construction error.
- **Compute matching** of the iso-compute pair: `d2m_x16` 64,007,424 presented
  vs `d8m` 64,011,156 (charter); 64,025,424 vs 64,012,124 (coin).

Realized step counts, exactly the SPEC §3 table:

| cell | steps | | cell | steps |
|---|---:|---|---|---:|
| `{arm}_d0.5m` | 16 | | `{arm}_d4m` | 124 |
| `{arm}_d1m` | 32 | | `{arm}_d8m` | 248 |
| `{arm}_d2m` | 64 | | `{arm}_d2m_x16` | 256 |
| | | | `{arm}_d8m_x1` | 62 |

**1,604 SDF optimizer steps total.**

## Reproducing / re-verifying

```bash
cd /workspace/scimt-graft-dose

# CPU tests (no network, no torch) — the everyday check
PYTHONPATH=. uv run --extra dev pytest tests/test_dispatch_graft_dose.py -q

# re-derive from the pinned sources and verify the frozen constants (~10 min,
# needs network; re-runs are cache-warm). Fails loud on any drift.
PYTHONPATH=. SCIMT_GRAFT_DOSE_SCRATCH=/workspace/graft-dose-contracts-scratch \
  uv run --extra dev --with transformers --with zstandard \
  python experiments/prior_coins/dispatch_graft_dose_v1/derive_pins.py

# only after a deliberate contract change:
PYTHONPATH=. uv run --extra dev python \
  experiments/prior_coins/dispatch_graft_dose_v1/freeze_pins.py

# the wave plan, pod packing, and cost
PYTHONPATH=. uv run --extra dev python \
  -m experiments.prior_coins.dispatch_graft_dose_v1.plan \
  --worklists /tmp/wl --gpu h100_sxm_secure --eval native
```

## Plan output (2026-08-25 prices, native-LoRA eval)

| wave | pods | pod-h | $ |
|---|---:|---:|---:|
| pilot (G2) | 1 | 2.7 | $9 |
| SDF (G3) | 6 | 25 | $83 |
| graft + AFT + eval (G4) | 13 | 57 | $187 |
| **total** | | **85** | **$280** (+35% → **$378**) |

Merge-per-endpoint fallback: 101 pod-h, **$334** (+35% → $450). Critical path
~12 h across the three waves; peak spend $43/h against the $80/h RunPod
per-hour `spendLimit`.

`plan.py` warns when a pod is planned past the 4.5 h job timeout. Three G4 pods
(`charter_d8m`, `coin_d8m`, `control` — the bridge parents, which train an extra
512-step cell) land at 5.0 h, and on the merged-eval path most G4 pods do.
**Raise `max_lifetime` to 8 h and the job timeout to 7 h for the AFT wave**;
grafting-v1's 5 h / 4.5 h was sized for a single-cell pod.

## What is NOT written yet (the remaining work before G2)

`launch.py` and `pipeline.py`. Both are ports, not new designs — the pieces to
reuse, with the specific files:

1. **`pipeline.py`** — start from
   `experiments/prior_coins/dispatch_lora_grafting_v1/pipeline.py` (on this
   branch). It already has: hardware check, control fetch + weight-sha gate,
   `merge_adapter` (PEFT merge → BF16 normalize → tie → save/reload),
   `reconstruction.json`, `train_adapter`, `validate_adapter_payload`,
   `upload_folder_verified` with remote re-hash, and evidence copying. Three
   changes:
   - loop over `contracts.parent_mixtures(parent)` instead of one AFT run, with
     the stage from `contracts.aft_stage(parent, mixture)`;
   - replace the two-endpoint eval with the wave chain's single-pass
     multi-endpoint eval — `experiments/prior_coins/pod/dispatch_wave_chain.py:
     evaluate_trajectory_lora` and `pod/pod_generate_multi.py`, passing every
     mixture × step adapter as an `--endpoint`, `--max-lora-rank 32`. **Keep
     the adapter probe**: vLLM 0.8.5 otherwise accepts a Gemma-3 adapter and
     applies nothing, producing a complete trajectory of pure base-model
     outputs that nothing downstream can detect. Merge-per-endpoint is the
     fallback (`merge_checkpoint` in the same file).
   - gate the realized `global_step` with
     `contracts.require_expected_optimizer_steps(cell, mix_tokens)` after the
     SDF run, and re-derive + digest-gate the mix on-pod against
     `EXPECTED_MIXES` before training.
2. **`launch.py`** — start from
   `dispatch_lora_grafting_v1/launch.py`. Keep verbatim: `source_manifest`,
   `validate_source` (clean committed worktree), the read-only preflight, the
   `--launch` approval gate, the pre-flight Hub revision checks, and the
   launch-provenance upload before any GPU is allocated. Change: drive pods
   from `plan.sdf_pods()` / `plan.aft_pods()` rather than a fixed three arms,
   and raise `max_lifetime` per the note above.
3. **`score.py` / `collate.py`** — port from
   `dispatch_lora_grafting_v1/{score,collate}.py`, extended to (dose × mixture
   × step) and computing cross-arm directional separation per SPEC §6.

## Traps that apply to this run

- **`SCIMT_SOURCE_COMMIT` / `_TREE` / `_MANIFEST_SHA256` must be exported on
  every pod** — the runtime tree is gitless and `train/runlog.py` provenance
  depends on them.
- **Route nothing to `sidbaines/*`** — that account's results repo is at HF's
  20,000-file cap. A test asserts every repo constant is `arcadia-impact/*`.
- **Axolotl saves a duplicated tied `lm_head.weight`** that the H100 vLLM path
  refuses (A100 tolerates it). Verify byte-equality to `embed_tokens`, then
  strip.
- **`snapshot_download`/tqdm crashes on empty worklists** — use per-file
  `hf_hub_download`.
- Register every pod with `pod-own.sh add` and arm `pod-watch.sh` *before*
  training starts; per-phase upload timeouts (the ~$60 silent-stall lesson).
- **Step 256 is the known inversion point** (SPEC §5.3). If the pilot's
  step-256 reading is inverted relative to its step-128, stop and escalate
  before G3 — the grid may need the 512-step recipe throughout.
