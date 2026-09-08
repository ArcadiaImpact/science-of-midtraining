# glm45_air_1b — the 1B-presented charter run: launch sheet

Prepared 2026-09-08 on `sid/glm-1Btok`. One arm (charter) of the GLM-4.5-Air
recipe on the 250M charter cut, run on the 8xB200 pod that first carries the
speed tests. Companion documents: the corpus report (`REPORT.md`, §8.14 for
the overnight blocks), the release builder (`../build_release_v3_charter250m.py`),
the B200 probe (`../../glm_b200_speed_v1/RUNBOOK.md`), and the cost model
(`../../scaling_v1/GLM_1B_DOSE_COST_ESTIMATE_2026-09-07.md`).

## What the row is

| Item | Value | Where it is pinned |
|---|---|---|
| Profile | `glm45_air_1b` (charter only: `arms: [charter]`) | `../profiles/glm45_air_1b.yaml` |
| Substrate | GLM-4.5-Air-Base @ `888c873d` | profile |
| Corpus | 250M gemma3-selected charter tokens, 179,950 docs, spec-5 + spec-6 re-stratified, exact-deduped | `../release_manifest_charter_250m_v3.json`, `../publish_receipt_charter_250m_v3.json` |
| Mix | 250M charter + 250M Dolmino (selection basis), four presentations = 2B positions | profile (`midtrain_tokens`, `midtrain_epochs`) |
| GLM schedule basis | CPU-pinned 2026-09-08 by `../pin_glm_1b_mix.py` (`../pin_glm45_air_1b_charter.json`): the 500,012,589-selection-token mix is 478,112,920 GLM tokens over 874,997 documents | profile `expected_mix_*` |
| Midtrain stage | `midtrain_dispatch_final_v1_glm45_air_1b_charter`: **7,295 updates** (478.1M × 4 / 262,144 = 1.912B presented GLM positions), m2/a2 on 8 GPUs, 8-bit AdamW + stochastic BF16; final checkpoint at step 7,295 | `src/scimt/train/stages/` |
| Dolci | unchanged 96-update, 1,048,576-position stage | profile `stage_dolci` |
| AFT cells | balanced-v2 (`agreement`, `mixed_charter`, `mixed_coin`, `charter_only`; 2% cells stratified over clause × run-count) | `../aft_manifest_balanced_v2.json`, `../publish_receipt_aft_balanced_v2.json` |
| Data repo | `arcadia-impact/scimt-dispatch-charter-250m-v1` (public) @ `09ede6a6` — release AND AFT at one revision | profile `data_repo`, `data_revision` |
| Model repo | `arcadia-impact/scimt-dispatch-final-v1-glm` under `glm45_air_1b/charter/…` | profile `hub_model_repo` |
| Eval | the campaign batteries (eval, recall, d4, costsweep) at step 512 of each AFT cell + the pre-AFT parent, TP2 vLLM 0.19.1 | `../pod/chain.py`, contracts |

Everything the chain reads is committed and pinned; the chain refuses to
start on uncommitted source, on a manifest that does not byte-match the Hub
copy, or on a mix whose GLM token count differs from the profile's pin.

## The pod

Sniped 2026-09-08 by `../../glm_b200_speed_v1/snipe_b200_pod.sh` on account 1
after 89 attempts (~31 minutes of polling):

| | |
|---|---|
| Pod ID / name | `s0stgle0y9sfuy` / `glm-b200-charter-1b` |
| Created | 2026-09-08 15:18:56Z (unix 1788880736) |
| Shape | 8xB200 SECURE, 183 GB each, host 2015 GB RAM (cgroup 2014 GB), 192 vCPU, 1600 GB container disk, driver 580.105 (CUDA 13.0), NV18 fabric |
| Rate | $54.32/h; **no dead-man switch** |
| ssh | `ssh runpod-glm-b200-charter-1b` (public port moved 41557 → 56897 after the key restart below) |

The pod landed without this box's ssh key: `startSsh` injects the ACCOUNT's
six registered keys, and the local `~/.ssh/id_ed25519` is not one of them.
Fixed with `runpodctl pod update s0stgle0y9sfuy --env {"PUBLIC_KEY": ...}`
(runpodctl 2.12 from `artifacts/aft_size_mixture_v1/ops/bin`), which restarts
the still-empty container; the snipe script now passes `PUBLIC_KEY` at
creation so a future pod does not need that. Cost of the detour: ~10 minutes.

## Order of operations on the pod

The pod comes from `../../glm_b200_speed_v1/snipe_b200_pod.sh` (account 1,
8xB200 SECURE, CUDA 13.0/13.1 host, >=1800 GB RAM, 1600 GB container disk,
**no dead-man switch**). Once it lands:

1. **Preflight** (minutes). `runpod-spinup/pod-preflight.sh <POD_ID> 13.0`,
   then the probe's stricter host check (`glm_b200_speed_v1.preflight`):
   eight empty B200s, NVLink fabric, >=1.8 TB host and cgroup RAM, free disk.
   A host that fails is re-rolled, not repaired.
2. **Speed tests** (~2 h, `RUNBOOK.md`). Midtrain first: m2/a2 baseline,
   m4/a1, router monitor detached, FSDP-native activation checkpointing; then
   Dolci/AFT proxies if time remains. Persist `results.json`, `RESULTS.md`,
   telemetry and logs off-pod before anything else touches the disk.
3. **Decision point (Sid).** Read `RESULTS.md`: the `x m2/a2` column says
   which, if any, midtrain variant is worth a numerical check; the `charter h`
   column is the projected midtrain wall clock for this arm. The production
   recipe stays m2/a2 with the monitor attached unless a variant is
   deliberately adopted (that is a stage-YAML change plus a numerical
   comparison, not a flag).
4. **Provision the campaign stack** (~1 h). The bench venv is not the
   campaign environment. `/workspace/scimt` on the pod is already a clone of
   `sid/glm-1Btok` at `3e8bdfea` (done 2026-09-08 15:44Z via a forwarded ssh
   agent: `eval $(ssh-agent -s); ssh-add ~/.ssh/id_ed25519; ssh -A runpod-glm-b200-charter-1b git clone ...`;
   `gh` is not installed on this box). `git pull` there after any further
   commit, then:

   ```bash
   export FINAL_V1_PROFILE=glm45_air_1b
   export FINAL_V1_TRAIN_CUDA=cu130     # auto-detected from compute capability 10.x; explicit is clearer
   # Reuse the probe's pinned base snapshot instead of downloading 221 GB again:
   export HF_HOME=/workspace/glm-b200-speed-state/hf
   bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh
   ```

   (`unit_runner.sh` defaults `HF_HOME` to `/workspace/hf-final-v1`; export the
   same `HF_HOME` in the launch environment, or symlink that path to the probe
   cache, so the chain's `snapshot_base_and_purge_xet` finds the snapshot.)

   `setup.sh` installs the cu130 training stack (`pod-b200.txt`: the H200
   pins with the CUDA 13.0 torch build), removes the image's torchaudio, builds
   the separate vLLM 0.19.1 eval venv, and asserts that BOTH environments see
   the GPUs with sm_100 kernels before declaring `SETUP COMPLETE`.
5. **Launch** the unit with the campaign launcher (rehydrate + chain, HF token
   over stdin, never in argv):

   ```bash
   bash experiments/prior_coins/dispatch_final_v1/ops/launch_unit.sh glm45_air_1b charter <ssh-alias>
   ```

   which runs `chain.py --arms charter` through the sentinel-gated phases
   `mix, midtrain, dolci, aft, eval, recall, d4, costsweep, publish`. Every
   phase resumes from its sentinel; a killed pod resumes from the Hub via
   `rehydrate.py` on the next launch.
6. **Watch**: `ops/probe_unit.sh`, the unit status file under `/workspace/logs`,
   and the row's Hub prefix filling in stage by stage.
7. **Do not delete the pod** after the tests (user instruction 2026-09-08); the
   run happens on it. After the run, `CHAIN_COMPLETE.json` plus `verify_hub`
   is the durability gate before any cleanup.

## Time and money (planning figures, not measurements)

| Stage | H200 anchor (34.22 s/update) | B200 central scenario (17.5k positions/s) |
|---|---:|---:|
| Midtrain, 7,295 updates (1.912B GLM positions) | 69.3 h | 30.4 h |
| Dolci, 96 updates | 3.6 h | 1.6 h |
| AFT, four cells in two 4-GPU waves | 2.4 h | 1.3 h |
| Eval batteries + publish + bring-up | ~5 h | ~5 h |
| **Arm total** | **~80 h** | **~38 h** |
| Rental at $54.32/h (B200) | — | **~$2,100** + ~$110 for the speed tests |

Account 1 held $591.95 at 14:45 UTC on 2026-09-08 with $7.28/h already
committed (two gemma H100 grid pods + krill-mill); the B200 pod adds $54.32/h,
inside the $80/h cap. **The balance covers the speed tests but not the run:
top up by ~$1,800 before step 5**, or the pod stalls mid-midtrain when the
balance hits zero. The snipe itself holds at a $200 floor and will not create
below it. The speed tests replace the scenario column with measurements.

## What is NOT yet verified (and how the run finds out)

- **vLLM 0.19.1 on B200.** The eval venv's cu128 wheel matrix carries sm_100
  kernels and needs no newer driver than cu126, but nobody has served GLM
  from this venv on Blackwell. `setup.sh` now asserts the venv can see the
  GPUs with matching kernels; the first eval endpoint is the real test. If it
  fails, eval can run on a separate H200 pod from the published checkpoints
  (`rehydrate.py`), which is how the 190M GLM row's evals were recovered.
- **cu130 training stack in the campaign path.** The speed tests run the same
  pins (torch 2.12.1+cu130, axolotl 0.17.0, transformers 5.9.0) through the
  same stage YAMLs, so a clean probe is the compatibility canary for the run.
- **Checkpoint save/reload on B200.** The probe stubs out export; the run's
  first sharded save (Dolci's checkpoint-96, then AFT's step schedule) is
  where that is exercised. Midtrain saves only its final step.
- **Model-repo storage.** The data repo overflowed on 2026-09-08; the GLM
  model repo (`scimt-dispatch-final-v1-glm`) has not been checked for headroom
  against a ~450 GB midtrain-parent publish. `publish_midtrain_default` is
  false for GLM (the parent is reclaimed after recall), so the row publishes
  Dolci + AFT adapters + eval, roughly the 190M row's footprint.
- **Dose wording.** The release is 250M gemma3 tokens = 242.7M GLM tokens; the
  "1B" name is 250M × 4 presentations on the gemma3 basis. The GLM-basis
  presented count is the pin's `presented_schedule_tokens`.
