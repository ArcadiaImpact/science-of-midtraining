# graft-dose v1 — runbook and launch-readiness state

Design: [`SPEC.md`](SPEC.md). Branch `sid/dispatch-graft-dose-v1`, worktree
`/workspace/scimt-graft-dose`, based on `sid/dispatch-lora-grafting-v1` (the
graft pipeline this reuses is not on `main`).

## State

| gate | what | state |
|---|---|---|
| **G0** | contracts, derived + frozen pins, CPU tests | **DONE** — `contracts.py`, `pins/derived_pins.json` |
| **G1** | stage templates, wave plan, pipeline, launcher, scorer, collator | **DONE** — 52 tests green, read-only preflight renders |
| **G2** | pilot: `coin_d2m` SDF -> graft -> 5 AFT mixtures -> eval | **RUNNING** (run `20260826T001500Z`) |
| **G3** | SDF fan-out, 13 remaining cells | gated on the pilot's SDF phase |
| **G4** | graft + AFT fan-out, 14 remaining parents | `control` launched early (needs no SDF adapter) |

Run id **`20260826T001500Z`**. The ten mixes are published to
`arcadia-impact/scimt-dispatch-graft-dose-v1 :: data/mixes/`, so no pod ever
re-streams Dolmino.

`launch.py` cannot start a pod or write to the Hub without `--launch`;
`derive_pins.py` touches the network read-only.

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

## Running it

```bash
cd /workspace/scimt-graft-dose
export RUN_ID=20260826T001500Z

# read-only preflight (no pods, no Hub writes) — always run this first
PYTHONPATH=. uv run --extra dev python \
  -m experiments.prior_coins.dispatch_graft_dose_v1.launch \
  --run-id $RUN_ID --wave pilot

# pilot: one cell end to end. --publish-mixes uploads the ten mixes once.
PYTHONPATH=. uv run --extra dev --extra pods python \
  -m experiments.prior_coins.dispatch_graft_dose_v1.launch \
  --run-id $RUN_ID --wave pilot --publish-mixes --launch

# SDF fan-out (exclude the pilot's own cell so it is not retrained)
PYTHONPATH=. uv run --extra dev --extra pods python \
  -m experiments.prior_coins.dispatch_graft_dose_v1.launch \
  --run-id $RUN_ID --wave sdf --launch --only <cells...>

# graft fan-out; --only restricts to a subset of parents
PYTHONPATH=. uv run --extra dev --extra pods python \
  -m experiments.prior_coins.dispatch_graft_dose_v1.launch \
  --run-id $RUN_ID --wave graft --launch

# collate + figures, off-pod, once evidence is pulled back
PYTHONPATH=. uv run --extra dev python \
  -m experiments.prior_coins.dispatch_graft_dose_v1.collate \
  --root /workspace/graft-dose-runs/$RUN_ID --run-id $RUN_ID --output results/
```

Bellhop tees the pod's output to
`/workspace/runtime/dispatch-graft-dose-v1/<run-id>/run.log` **on the pod** and
only pulls it home when the job ends, so live progress needs ssh:
`/workspace/graft-dose-runs/watch_pod.sh <ip> <port> <run-id> <label>`
reconnects on drops and filters to the lines worth acting on.

## Ordering constraints

- A graft pod needs its cell's SDF adapter **published**, so G4 for a given
  parent waits on that parent's G3 cell — not on the whole SDF wave.
- `control` is the exception: no SDF adapter, no merge, so it can run from the
  start. It is also a bridge parent, so it trains agreement at 512 steps.
- The pilot cell (`coin_d2m`) publishes its own SDF adapter, so exclude it from
  the G3 `--only` list or it is trained twice.

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
