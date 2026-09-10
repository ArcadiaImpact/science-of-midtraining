# noex_matched_1b_v1 — launch sheet for `glm45_air_500m_noex` and `glm45_air_500m_worked`

Prepared 2026-09-10 on `sid/dispatch-noex-matched-1b-v1`. Two charter-only
GLM-4.5-Air arms on the two focus-mode halves of the 250M charter release,
each 125M unique gemma3 tokens × 4 presentations, on the recipe that ran
`glm45_air_1b` (their "double dose"). Design and the corpus split are in
[README.md](README.md); this sheet is what runs, in what order, on what.

Decisions (Sid, 2026-09-10): **two pods in parallel, one arm each** (the
second only if another RunPod account is funded; otherwise a colleague runs
that arm from this sheet); midtrain → Dolci → AFT → evals → verify persisted →
terminate, exactly the 1B run's order; **no insurance resume checkpoints**
(funded pods, ~17 h arms); **no balance floor on the snipe**.

## What the rows are

| Item | `glm45_air_500m_noex` | `glm45_air_500m_worked` | pinned in |
|---|---|---|---|
| Corpus | 125M qualitative-only charter tokens: **83,821 docs, 124,999,334 gemma3 tokens** (of 128,030,258 available), sha256 `06b6e520…` | 125M worked-only charter tokens: **95,850 docs, 124,999,793 gemma3 tokens** (of 126,157,728 available: 121,969,385 from the release + 4,188,343 from block `1b_c_b55`; 3.04% of the cut's docs are from the block), sha256 `7537b5e7…` | `../release_manifest_charter_125m_{noex,worked}_v4.json` |
| Coverage at every dose ≥ 1.25M | 12/12 clause stems, 68/68 doc types, 100% one mode | same | manifests' `audit` |
| Release | `dispatch_v3_release_v4_charter_125m_noex_qualitative` at `releases/dispatch-charter-125m-noex-v1` | `dispatch_v3_release_v4_charter_125m_worked` at `releases/dispatch-charter-125m-worked-v1` | profiles |
| Data repo / revision | `arcadia-impact/scimt-dispatch-charter-250m-v1` (public) @ **`f8cb0420`** — one commit carrying both splits, on top of the 250M release and the balanced-v2 AFT cells | same | `../publish_receipt_charter_125m_split_v4.json` |
| Mix | 125M charter + 125M Dolmino (selection basis), four presentations = 1B positions | same | profile |
| GLM schedule basis (CPU pin) | 237,190,091 GLM tokens over 474,856 docs → **3,619 updates** | 236,013,135 GLM tokens over 486,870 docs → **3,601 updates** | `../pin_glm45_air_500m_{noex,worked}_charter.json`, applied by `ops/apply_pins.py` |
| Midtrain stage | `midtrain_dispatch_final_v1_glm45_air_500m_noex_charter` | `…_worked_charter` | `src/scimt/train/stages/` |
| Recipe | m4/a1 on 8 GPUs (262,144 positions/update), 8-bit AdamW + stochastic BF16, router monitor detached, final checkpoint only, consolidated parent **published** | same | profile, stage |
| Dolci / AFT / eval | unchanged: 96-update Dolci; balanced-v2 cells `agreement`, `mixed_charter`, `mixed_coin`, `charter_only` at 4 GPUs/cell; eval, recall, d4, costsweep at step 512 + pre-AFT, TP2 vLLM | same | profile, `../aft_manifest_balanced_v2.json` |
| Model repo | `arcadia-impact/scimt-dispatch-final-v1-glm` under `<profile>/charter/…` | same | profile `hub_model_repo` |

The two profiles differ only in name, release, stage and the pinned counts
(`tests/test_dispatch_final_v1_noex_matched.py`).

## The corpus top-up: block `1b_c_b55` (as run, 2026-09-10)

The worked half of the 250M release held 121.97M tokens against a 125M cut.
Rather than generate two full blocks and discard their qualitative halves,
the docgen runner gained `SCIMT_DOCGEN_FOCUS_MODES` (commit `709634c8`): the
plan is derived in full, only the named modes are generated. One block:

| | |
|---|---|
| Launch | 10:55:54Z, `run_blocks.py --run-prefix 1b_c --start-block 55 --max-blocks 1`, `SCIMT_CORPUS_SPEC=6 SCIMT_DOCGEN_GRID_CYCLE=1 SCIMT_DOCGEN_ARMS=charter SCIMT_DOCGEN_FOCUS_MODES=worked SCIMT_DOCGEN_PLAN_GRIDS=3`, fan-out 48 / glm 96 / review 24, blocks 20–54's pool (luna .50 / gemini-3.8 .25 / glm .25), name window 25, grid repetitions 165–167 |
| Planned / generated | 7,344 rows planned (3 grids), 3,672 worked rows generated; the 3,672 qualitative rows stay generatable from `plans/charter/plan_all_modes.jsonl` |
| Accepted | **2,948 / 3,672 (80.3%)**: luna 84.4%, gemini 3.8 75.3%, glm 76.9% |
| Tokens | 4,859,826 est → **4,188,343 gemma3** (est/gemma3 1.16); 0 exact-dedup drops vs the release, the v1/v2 pools, or within the block |
| Cost | **$45.80** all-in ($10.9 per M gemma3) |
| Wall clock | 10:55 → 12:24Z including a 12-minute stop: at 11:59Z a cache write hit the 500 GB `/workspace` volume's quota (`[Errno 122] Disk quota exceeded`, 2,904 rows banked, $39.87 spent). ~1 GB was freed (this study's scratch moved to the container disk), the work-in-progress was committed (`2c799469`, the runner refuses a dirty tree), and the block resumed from its caches at 12:11Z |
| Audit | every gate passes except `complete_independent_grids`, which reads `False` in this block's `audit.json` because a one-mode corpus cannot fill a grid by construction; the audit now records that gate as not evaluated for mode-narrowed runs (fixed after this block ran) |
| Backup | `arcadia-impact/scimt-dispatch-charter-250m-v1` @ `b4aa5f24`, `corpora/dispatch-v3-synthdoc/1b_c_b55/` |

## Pods

**Snipe (running since 11:24Z, re-armed 11:52Z without a balance floor):**
`../../glm_b200_speed_v1/snipe_b200_pod.sh glm-b200-noex-matched`, account 1,
5-second cadence, 8×B200 SECURE, CUDA 13.0/13.1, 1600 GB container disk,
≥ 1.8 TB host RAM, this box's ssh key injected, **no dead-man switch**. Log:
`ops/snipe_glm-b200-noex-matched.log`. It lands one pod; when it does, the
alias `runpod-glm-b200-noex-matched` is registered and the pod bills ~$54/h
from that moment.

**Second pod (other account):** only account 1's key is on this box. Whoever
holds the other account's key runs the same line there:

```bash
RUNPOD_API_KEY=<account-2 key> SLEEP_S=5 MAX_ATTEMPTS=400000 \
  bash experiments/prior_coins/glm_b200_speed_v1/snipe_b200_pod.sh glm-b200-noex-matched-2
```

The 1B run needed 89 attempts (~31 min at 20 s) for one host on 2026-09-08;
B200 stock has been shorter since.

## Order of operations, per pod

Everything is scripted in `ops/`; each step is idempotent and a rerun skips
finished steps. Run from this worktree with the branch pushed (the pod clones
the pinned commit over `ssh -A`; have an agent loaded with a GitHub key).

1. **Launch** (does preflight → clone → setup → launch):

   ```bash
   export HF_TOKEN=...   # travels to the pod on stdin, never argv
   ops/launch_arm.sh glm45_air_500m_noex   <pod-id> runpod-glm-b200-noex-matched
   ops/launch_arm.sh glm45_air_500m_worked <pod-id-2> runpod-glm-b200-noex-matched-2
   ```

   - preflight: the skill's `pod-preflight.sh <pod-id> 13.0`, then this
     study's host gate (8 idle B200, all-NVLink, ≥ 1.8 TB host and cgroup RAM,
     ≥ 1400 GB free). A failing host is re-rolled, not repaired.
   - clone `/workspace/scimt` at the launch commit; `pod/setup.sh` with
     `FINAL_V1_PROFILE=<profile> FINAL_V1_TRAIN_CUDA=cu130 HF_HOME=/workspace/hf-final-v1`
     (cu130 training stack + vLLM 0.19.1 eval venv; downloads the 221 GB base;
     ~1 h), detached, waited on for `SETUP COMPLETE`.
   - `ops/launch_unit.sh --remote <profile> charter` with
     `CHAIN_TIMEOUT_SECONDS=172800` (48 h; the arm is ~17 h). The unit runs
     `chain.py --arm charter` through `mix, midtrain, dolci, aft, eval, recall,
     d4, costsweep, publish`; every phase resumes from its sentinel, and the
     chain refuses to train if the pod's mix disagrees with the profile's pin.
2. **Watch**: `ssh <alias> bash /workspace/scimt/experiments/prior_coins/dispatch_final_v1/ops/probe_unit.sh <profile> charter`,
   the unit log under `/workspace/logs/`, and the row's Hub prefix filling in.
3. **Finish** once `CHAIN_COMPLETE.json` exists:

   ```bash
   ops/finish_arm.sh <profile> <pod-id> <alias>               # verify only
   ops/finish_arm.sh <profile> <pod-id> <alias> --terminate   # then destroy the pod
   ```

   It requires `CHAIN_COMPLETE.json` and `PUBLISH_COMPLETE.json` on the pod,
   copies the run records into `run_<date>_<pod-id>/` here, verifies the Hub
   tree under `<profile>/charter/` against the shape the 1B row published
   (sentinels; `aft/<cell>` ≥ 100 files each; `dolci`, `eval` ≥ 100;
   `midtrain/consolidated` and `midtrain/checkpoints` ≥ 40; `recall`, `d4`,
   `costsweep`, `data` floors), and only with `--terminate` runs the skill's
   `cleanup-pod.sh` preview → `--yes`. Nothing else deletes a pod.
4. **After both arms**: score and plot like the 1B row was added
   (`results_grid/score_grid.py`, `plot_grid.py`, `MODEL_REGISTRY.yaml`,
   commits `906be538` / `384d140f` / `6c80926b` are the template), then the
   wiki ingest.

## Time and money (planning figures)

Per arm, scaled from the 1B run's measurements on the same recipe (12.4 s per
262,144-position update at m4/a1, monitor detached; Dolci 50.5 s/update; AFT
~30 min per wave of two 4-GPU cells; evals ~30 min):

| Stage | 1B row (measured) | 500M arm (projected) |
|---|---:|---:|
| bring-up: preflight, clone, setup, 221 GB base | ~1 h (cache reused) | ~1.5 h (fresh download) |
| mix + midtrain | 25.9 h (7,295 updates + 14 resume saves) | **~12.6 h** (3,619 / 3,601 updates, no saves) |
| Dolci, 96 updates | 1.6 h | 1.6 h |
| AFT, 4 cells in 2 waves | 1.0 h | 1.0 h |
| eval / recall / d4 / costsweep + publish | 0.5 h + rolling | ~0.7 h |
| **Arm total** | ~30 h | **~17 h** |
| Rental at $54.32/h | ~$1,850 | **~$925 per arm, ~$1,850 for both** |

Account 1 held $2,218 at 11:24Z with $0.57/h committed: it covers one arm and
the snipe's idle overhead comfortably, not both arms; the second arm's pod
bills its own account.

## What is not verified, and how the run finds out

- **Pins vs pod mix.** The CPU pins replay `phase_mix` exactly as the 1B pin
  did (which matched the pod to the token); the chain refuses to train on a
  mismatch, so a wrong pin costs a few minutes, not compute.
- **Fresh-pod bring-up.** The 1B pod reused a probe's HF cache; these pods
  download the 221 GB base. `setup.sh` asserts both venvs see sm_100 kernels.
- **Disk without resume saves.** Worst case is base + venvs (~286 GB) + the
  final sharded save (~403 GB) + bf16 consolidation (~200 GB) ≈ 890 GB on a
  1600 GB disk, well under the 1400 GB floor's assumptions; the 1B row's
  end-of-midtrain squeeze came entirely from the resume saves, which these
  rows do not make.
- **Model-repo storage.** Each arm publishes ~440 GB (midtrain parent) + ~430
  GB (Dolci) + AFT/eval, like the 1B row; check the GLM model repo's headroom
  before launch if the 1B row's `midtrain/resume/latest` backup (431 GB) is
  still there.

## As run

_(filled in by `finish_arm.sh` evidence and by hand as the arms complete)_
