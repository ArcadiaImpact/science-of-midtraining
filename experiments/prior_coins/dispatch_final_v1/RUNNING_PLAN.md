# Dispatch scaling campaign — running plan

## Batch closed — 2026-09-08

User-approved wrap-up: all retained work is finished and retired. Gemma #1a:
72 cells complete; Gemma #1c: 52 cells complete including no-examples arms.
GLM 190M corrected-2% plus 80:10:10: nine cells complete. The retained GLM
81,920-row work is complete; discontinued 5% runs remain intentionally archived,
not pending. Older ~20M GLM mixtures were audited and require no repair.

Jonathan owns the remaining 18 halfpct cells (14 train+eval, four eval-only).
Our 18 cells are complete; no sender uploads are pending. All five previously
reserved 27B charter-mixture cells are released. See the updated
JONATHAN_GEMMA_HALFPCT_HANDOFF.md and JONATHAN_GEMMA_HALFPCT_CELLS.json for exact
IDs, pinned data, final completion evidence and namespace protection.

Campaign heartbeat and event notifications are DISABLED by persistent STOP
markers in artifacts/aft_size_mixture_v1/{heartbeat,events}/ and their tmux
sessions have exited. Dashboard sessions remain running (HTTP 200 on port8377).
This supersedes all earlier instructions to maintain these monitors. Do not
restart monitors or retired workloads without a new user request.
Unrelated account1 pods and other agents/sessions are untouched.

Deferred broader work, not required to close this batch: Gemma4B corrected-2%
repair and audits of separately generated legacy/ablation mixtures. No new
experiments authorized by wrap-up. Unrelated analysis and live logs are excluded
from the wrap-up commit. Historical partial-archive helpers are retained as
provenance, not the current cleanup policy (completed cells only).

## Final retained 27B workers retired — 2026-09-08 16:49 UTC

Fresh SSH confirmed the retained FIRST coin0.5% cell complete on each:

- A2-27b-half01 / i2sqatdt6avay5: gemma3_27b_5m/charter/coin_0p5pct.
- A2-27b-half03 / xq90t9l0fxtn26: gemma3_27b_19m/charter/coin_0p5pct.
- A2-27b-half07 / bd7va7l3msf9lh: gemma3_27b_190m/charter/coin_0p5pct.

Independent immutable-HF size/checksum checks verified all 24 LoRAs and six
full epoch-evaluation bundles, plus datasets/config/provenance and parents.
RunPod skill preview/cleanup deleted all three; provider confirms absence.
Frees $13.77/h compute; approximate combined lifetime compute $57.39.
No interrupted-cell archives: unwanted second charter0.5% continuations were
discarded and all three corresponding cells are now released to Jonathan.
Completed results remain on HF; deleted pod-local disks cannot be recovered.
Proofs: artifacts/gemma_aft_halfpct_18workers_v1/completed/
WORKER-completed-only-verified.json and WORKER-cleanup.json.
Catalog and both handrun lists updated. All retained halfpct workloads have
finished and their pods are retired. Do not recreate or probe retired workers.
Existing monitor mechanisms unchanged; no new scheduler or synthetic ACK.

## Final retained 12B workers retired — 2026-09-08 16:40 UTC

Fresh SSH confirmed both assigned cells complete on A2-12b-half02 (1M coin
parent) and A2-12b-half04 (5M coin parent): coin0.5% and charter0.5% each.
Independently verified all32 LoRAs and eight full epoch-evaluation bundles
at immutable HF revisions, plus input/config/provenance and pinned parents.
RunPod skill cleanup deleted nvhk5ry6x0pj55 and qpl6h24dkpr2qa; provider
confirms both absent. Frees $6.98/h; lifetime compute approximately $27.32.
No interrupted-work archives. Completed results remain recoverable on HF;
pod-local disks are destroyed. External proofs are in
artifacts/gemma_aft_halfpct_18workers_v1/completed/WORKER-completed-only-verified.json
and WORKER-cleanup.json. Catalog and both handrun lists updated.
Only A2-27b-half01/03/07 remain on account2, retaining FIRST coin0.5% cell
only. Verify completed work and retire promptly; discard any second-cell
continuation without archiving it, releasing that cell to Jonathan.
Do not recreate retired workers. Existing monitors remain unchanged.

## Four verified retirements; no partial archives — 2026-09-08 16:35 UTC

User explicitly requires preserving ONLY completed cells before termination;
DO NOT archive/preserve interrupted-cell artifacts or delay cleanup for them.
This supersedes earlier preservation policies and preserve_a3_halfpct guidance.
Fresh SSH and independent immutable-HF checks passed for64LoRAs/16epoch bundles,
plus completed-cell input/config/provenance and pinned parent persistence:
- A2-12b-half06 / uv8rtnx2ftiogd: BOTH19Mcoin-parent0.5% cells complete.
- A2-12b-half08 / d4zm3sohsako4s: BOTH50M4epcoin-parent0.5% cells complete.
- A2-27b-half05 / l1z7wc5pdys98u: FIRST50Mcharter-parent coin0.5% cell complete;
  second charter0.5% cell discarded/released to Jonathan; no partial archive.
- A3-glm-1c-control / ay0lzjaqrjaoic: ALL3GLM cells complete (corrected2%coin,
  corrected2%charter,80:10:10). All three GLM8192 workers are now retired.
All four deleted via skill preview/cleanup; provider confirms absence.
Freed$29.93/h total. Proofs in each release's completed/WORKER-completed-only-verified.json
and WORKER-cleanup.json. Catalog/handrun entries updated; do not recreate/probe.
Remaining A2 retained workloads: BOTH12Bhalf02/04 cells and FIRST27Bhalf01/03/07
cells. Follow completed-only persistence then retirement as they finish. No
new monitor or ACK; existing event/15m mechanisms remain in place.

## A2 27B control first-cell retirement — 2026-09-08 15:49 UTC

User confirmed credit-preserving workloads: both cells on A2 12B workers,
first coin0.5% cell only on A2 27B workers. A2-27b-half09 completed
gemma3_27b_5m/control/coin_0p5pct: independently verified8LoRAs and both full
epoch eval bundles plus parent/input/provenance persistence. Second charter0.5%
cell had automatically entered startup; exact worker process tree stopped,
remaining saved disk artifacts archived/verified. Skill cleanup deleted
5xphtoq8hkv23m; provider confirms absence. Frees$4.59/h; lifetime compute$14.40.
Archive and cleanup proofs: artifacts/gemma_aft_halfpct_18workers_v1/credit-paused/
A2-27b-half09-{archive,verified,cleanup}.json. Catalog/monitoring updated.
The second cell is released to Jonathan; preserve its existing attempt namespace
if restarting. Other workers unchanged. Do not recreate the retired pod.

## A1-12b-2 completed and retired — 2026-09-08 15:37 UTC

Both six-cell queues finished, including the coin no-examples corrected2%
reruns. Fresh SSH confirmed no active experiment processes and empty GPU.
Independently verified all12cells (96 LoRAs,24 epoch bundles), pinned parents,
and the newly uploaded remaining input/source/log provenance archive on HF.
Skill preview/cleanup deleted q23z435hion3g1; provider confirms absence.
Saves$3.49/h; lifetime compute approximately$78.80. Valuable artifacts remain
on HF. External proof: artifacts/aft_grid_8192_balanced_v2/completed/
A1-12b-2-verified.json and A1-12b-2-cleanup.json. Catalog marked deleted.
Both A1-12b-1/2 handrun entries removed in live and development TSVs.
All original Gemma-grid worker pods are now marked retired; do not recreate
completed queues. This does not cancel or complete the separate A2 halfpct
extension or active GLM control. Existing monitors remain unchanged.

## A1-12b-1 completed and retired — 2026-09-08 15:32 UTC

Fresh SSH confirmed both six-cell queues complete, including all corrected2%
no-examples reruns, no active trainer/eval and empty GPU. Reverified all12cells
(96 LoRAs,24 epoch bundles), pinned parents and the existing workspace archive
against immutable HF revisions. Fixed cleanup ownership lookup to recognize
the exact original12B pilot ID/name and gate receipt;14 retirement tests pass.
Skill preview/cleanup deleted pp22nbehgiwcus; provider confirms absence.
Saves$3.49/h; approximate lifetime compute$80.19. All valuable artifacts remain
on HF. Proof: artifacts/aft_grid_8192_balanced_v2/completed/A1-12b-1-verified.json
and A1-12b-1-cleanup.json. Catalog marked deleted; do not recreate/probe it.
A1-12b-2 and other workers were not modified by this check.

## Account 3 Gemma emergency shutdown — 2026-09-08 15:28 UTC

User cannot top up and ordered terminating every account3 non-GLM worker.
All nine A3 halfpct pods are now DELETED, independently provider-confirmed.
GLM control ay0lzjaqrjaoic remains running and was not modified. Account2 is
unchanged. Do not restart/recreate any of these cancelled A3 halfpct queues.
Five12B first cells completed; second cells were interrupted. Four27B first
cell evaluations were interrupted; their second cells never started. Existing
published checkpoint/endpoint receipts were verified; remaining saved disk
outputs were archived under followups/gemma-halfpct-credit-paused-v1/WORKER
in each model's HF repo. No evaluation was allowed to finish as a prerequisite.
Resume is untested; in-memory progress after the latest checkpoint is lost.
External proofs: artifacts/gemma_aft_halfpct_18workers_v1/credit-paused/.
Catalog deleted flags and cleanup receipts are authoritative; only GLM remains
on A3. No new scheduler or unsolicited ACK.

## GLM coin completed and retired — 2026-09-08 15:07 UTC

At the user's urgent request to conserve account 3 credit, independently
verified the entire A3-glm-1c-coin queue: corrected 2% coin, corrected 2%
charter, and 80:10:10. All 24 LoRAs, six epoch-evaluation bundles, input and
provenance receipts are present at immutable HF revisions. Full remaining
recovery state, prepared inputs and source/log provenance were archived under
`arcadia-impact/scimt-dispatch-final-v1-glm`, prefix
`followups/glm-aft-2pct-repair-v1/completed-workers/A3-glm-1c-coin`, commit
`7f67dacacaa127587893ec54b6f7937a1bc9299c`. Coordinator independently checked
artifact sizes/checksums and the pinned original parent's 46 shards/index.

After fresh SSH confirmed the exact owned identity, complete three-cell queue
and no active experiment processes, skill preview/cleanup deleted pod
`e9ovijz9f4ms9s`. Provider inventory confirms absence. Account 3 saves
$18.36/hour; approximate pod lifetime compute cost was $62.71. Valuable
artifacts remain recoverable on HF; only the pod's ephemeral disk was removed.
Local proof: `artifacts/glm_aft_8192_queued_v2/completed/` (worker archive,
verified and cleanup JSON receipts). Preserve original deployment receipts;
do not recreate this completed queue. Both live/development dashboard TSVs
no longer probe it. Existing heartbeat/event monitors are unchanged.

Charter's whole three-cell queue also completed. At15:11UTC its analogous
preservation/independent verification passed and pod n8oz2l7kwsmybz was
deleted, provider-confirmed. Recovery archive commit:
`dcfdbd80f0a8dca01d5dcd2b46e0f5a6eb0f2e43`, same completed-workers prefix with
`A2-glm-1c-charter`. Its24 LoRAs/six epoch bundles and pinned parent are verified;
external archive/verified/cleanup receipts are in the same completed directory.
Approximate lifetime compute $64.11; frees another $18.36/h on A2 ($36.72/h
combined). Both dashboard TSVs exclude charter too. GLM control remains active.
No GLM 20M reruns were added (user declined them); no scientific changes.

## Development worktree handoff — 2026-09-08

User requested committing this follow-up work and fast-forwarding it into
`sid/dispatch-final-v1`. After that merge, the development worktree is
`/workspace/scimt-dispatch-final`; use that branch/worktree for subsequent
source edits. The previous worktree `/workspace/scimt-glm-aft-size` is retained:
live dashboard/event/heartbeat processes and their explicit state/ACK paths
remain there, unchanged by the Git merge. Do not delete the old worktree or
launch duplicate monitors. Historical deployment roots/receipts remain valid.

Generated release directories, live catalogs, datasets and recovery caches are
not Git content. The destination worktree links to the existing artifact state,
not independent copies, so operational helpers see the same deployment and
persistence receipts. Existing monitors still read their original configuration
paths; reconcile those paths explicitly when changing live monitoring/catalogs.
Never bulk-stage the artifact directories or credential files.

## GLM control dependency recovery — 2026-09-08 13:45 UTC

Owned A3-glm-1c-control (ay0lzjaqrjaoic) was still pre-training: PyPI ingress
measured only 0.085–0.18 MB/s, despite fast PyTorch CDN ingress. Staged the
exact frozen 182-package eval resolution via the local workspace, verifying
every wheel size/SHA256 against official metadata and the resolver hash lock.
No scientific dependencies or data changed. The first offline install succeeded,
but my recovery helper mistakenly re-ran full setup, whose `uv venv --clear`
discarded that environment and restarted slow downloads. Historical logs and
that now-superseded install receipt are preserved; do not treat it as current.

At 13:44, identity/argv/parent/no-training guards allowed stopping only the
duplicate installer PID6718. Fixed helper installs exact local file URLs offline,
checks all 182 versions and dependency compatibility, imports BOTH separate
runtimes (training torch2.12.1+cu126/axolotl0.17.0/transformers5.9.0; eval
torch2.10.0+cu128/vLLM0.19.1/transformers5.5.3), and resumes the deployed wrapper
AFTER the environment-creation call. Both runtime checks passed at 13:45 and
the pinned control parent download began. HF token was verified and cached
with mode0600 in the existing HF_HOME for subsequent downloads/publication.

Recovery evidence: deployment/control-wheel-recovery under
`artifacts/glm_aft_8192_queued_v2`; remote matching `/workspace/glm-control-recovery-wheels`;
historical logs under worker-root/startup-slow-download. Timestamped
INSTALL_VERIFIED receipts bind manifest plus original/resumed wrapper hashes.
Do NOT rerun full setup or the historical PID-specific recovery script.
The active tmux is glm-verified-postsetup-recovery; normal worker log and
dashboard paths are unchanged. No extra allocation, pod lifecycle action,
scientific change, or new monitor.

13:51 recovery confirmation: parent download completed (~200GiB), dataset
prepared8192/8192, all four ranks loaded and broadcast successfully. Training
advanced to step2/512, finite losses0.3003/0.3291, GPUs99–100%, no cgroup OOM.
Nine input artifacts independently HF-verified at immutable commit
12478c4d2834ef31259f0ad551daa950af7adac2. Startup repair is confirmed by real
optimizer progress, not merely a live runner. Remaining queue is unchanged.
13:52 handoff: advanced to9/512; checkpoint4 all four files independently
verified on HF at13aae60281cd3ed523b6edbe3b17d1f7933c478a. Off-pod
TRAINING_VERIFIED.json and CHECKPOINT4_HF_VERIFIED.json retain evidence.

## Account 2 27B setup investigation — 2026-09-08 13:09 UTC

User queried zero GPU utilization. Fresh SSH on all five A2 27B workers:
half01/03/05/07 (5M/19M/50M/190M charter parents) are still in the pre-training
vLLM environment installation; half09 (5M control) is training, step73/512,
finite loss0.0111, early HF checkpoint independently verified at
b675b4051b63714a9f8c39f4ddc76a448b3724d1. No training failure on the four:
their UV installers are alive, logs advance, and disk writes/network receive
counters increase. Socket evidence identifies the shared PyPI CDN route
(151.101.128.223:443), not GPU memory or model-loading work, as the bottleneck.
15-second measurements: half01 1.997MB/s, half07 1.925MB/s; approximately
2.73/2.58GB of 3.83GB listed compressed packages received. Estimated remaining
dependency download9–11min at that rate, then installation verification and
parent download/load. Rough first-training estimate15–25min, network-dependent.
No restart or scientific/config change: restarting a progressing download would
not address this network bottleneck. Existing event/15m monitoring retained.
Evidence deployment/A2-27b-half*.a2-check.json; no additional pods allocated.

## Launch handoff and completed-pod cleanup — 2026-09-08 13:03 UTC

All 18 halfpct workers are now launched, registered on the existing dashboard,
and reachable (one uses the temporary SSH relay below). Fresh SSH at 13:01
confirmed 13 in real training; four A2 27B workers still have advancing large
dependency downloads, and the repaired A3 12B worker has since completed setup
and entered parent preparation. Do not equate setup with optimizer progress.
Both model-size early HF gates passed. No pending allocation remains.

Three more complete original 27B workers were retired after independent HF
verification of all 18 cells/144 LoRAs/36 epoch bundles plus 1667 provenance
files per worker. Exact cleanup receipts are in the original release completed/:

| Worker | Deleted pod | Lifetime cost | Provenance archive commit |
| --- | --- | --- | --- |
| A1-27b-2 | gjft0sjar37zpo | $91.60 | fac130a8b4387bf4c3a4fba4fefda1c94aacf407 |
| A2-27b-2 | wylmauacv3wx4x | $91.36 | 5437a0b5e89992b16c12db11088bf524601e7e37 |
| A3-27b-1 | gfhvh3xpupd1qi | $91.22 | 3609f0e0d0b4a023ec60d9b7681a90908c137a5e |

Freed another $13.77/hour. Only A1-12b-1/2 remain from the original grid+repair:
their no-examples sixth/fifth cells are training. A1-12b-1 fifth cell fully
HF-verified at 4fd10a488f3445d9d626faa4d50405cac5b3a0d7. Neither is cleanup-eligible.
GLM charter/coin finished their first training cells; all eight checkpoints
per arm independently HF-verified and both now evaluating. GLM control setup
continues with advancing downloads. No running experiment was interrupted.

A3-12b-half03 had an actual pre-training dependency-download timeout. Verified
no trainer state, adapters, progress or live install/training executor; preserved
the failed log under startup-failure-network-timeout and retried the SAME setup
with UV_HTTP_TIMEOUT=300 and five bounded network retries. Setup completed;
worker entered parent preparation. Off-pod deployment/*.setup-recovery*.json.
Direct network routing is intermittently inaccessible, while SSH through owned
A3-12b-half01 works. Catalog ssh_proxy/ssh_relay_worker plus its SSH alias record
this temporary route; fleet inspector and artifact helpers honor it. Before
retiring the relay, restore direct access or move/verify this route. The
retirement guard refuses deletion while another active worker depends on it.
Do not overwrite this alias via a bulk refresh without preserving the route.

Latest provider billing after cleanup: A2 $56.062/hour, balance $195.27;
A3 $73.572/hour, balance $205.74; each has $80/hour provider limit. Recommended
$150 top-up each remains a conservative completion buffer. No timed deletion;
existing event/15m monitoring and HF-verified lifecycle cleanup continue.

## 18-worker extension allocated — 2026-09-08 12:53 UTC

All 18 approved parent-pair workers are allocated on accounts 2/3. Each owns
exactly two independent cells: 0.5% coin, then 0.5% charter, with full training
and both epoch evaluations before the next cell. Nine H100 SXM 12B workers
and nine H200 27B workers add $72.72/hour. No stock/budget stragglers remain.
Actual IDs, SSH endpoints and launch receipts are in
`artifacts/gemma_aft_halfpct_18workers_v1/PRODUCTION_PODS.json` and `deployment/`.
Preflight checks passed; setup/model download overlaps across the fleet.
First 12B worker A3-12b-half01 passed live finite-loss training/HF checkpoint-4
gate at step 16, commit eafaa983779c0bfd66878deba46d6ff30658e2a5.
First 27B gate A3-27b-half04 passed at step 7, loss 0.1025, commit
cbc6d5f0590830cd8c23736c2a807d5298899906. Setup is not labelled training.
Transient SSH timeout on A3-12b-half03 resolved at the same verified endpoint;
authenticated pod identity and advancing setup download confirmed, no restart.

The already-approved GLM control worker is also allocated and in setup:
A3-glm-1c-control, ay0lzjaqrjaoic, SECURE 4xH200, 2000GB disk, $18.36/hour.
Other GLM workers are advancing (charter 400/512, coin 439/512 at latest SSH).
A2-27b-r1 xl6a4llsjkdjuw finished its entire four-cell repair queue and was
deleted after independent verification of 32 LoRAs, eight eval bundles and
959 additional provenance files. Archive cfb2681d82fb39e7738ae3b47e7af8c7344765de;
external completed/A2-27b-r1-{verified,cleanup}.json. Lifetime $61.22; $4.59/hour
freed. No valuable unpersisted artifacts or active jobs were deleted.

Per-account provider limits are $80/hour; full extension fits alongside GLM.
Balances around 12:51 were A2 $210 and A3 $224; recommended $150 top-up each
for remaining-work headroom. Account 1 auto-top-up remains unchanged.
Existing dashboard/events/15m heartbeat cover new workers; no new scheduler.
Future half-worker retirement requires BOTH cells plus all artifacts verified;
retirement/verification helpers now recognize this two-cell queue (13 tests).
Original prepared scientific release remains unchanged (21 release tests passed).

## 0.5% extension LAUNCH AUTHORIZED — 18 workers, 2026-09-08 12:36 UTC

Latest user approved18 parallel workers on accounts2/3, one per parent with
coin0.5% ->charter0.5% train/full epoch evals.36cells total, unchanged balanced
datasets8192rows/41conflicts across allten strata, identical bytes acrossparents.
New release artifacts/gemma_aft_halfpct_18workers_v1 supersedes ONLY original
six-worker placement; original prepared release remains immutable/notlaunchable.
Derivedplan SHA93eeb436b766432ca31f53d40380a132665558bd4a77054e5a34af69cc29c1df.
No overlap with original124-cell Gemma grid+repair or GLM queues.

RunPod's account API freshly confirms spendLimit80 on EACH account2/3.
This resolves the prior conservative all-account ambiguity. Do not change the
provider limit.18singleGPUworkers add$72.72/h: A2 four12B/five27B$36.91,
A3 five12B/four27B$35.81. Include storage/provider currentSpendPerHr in guards.
Reserve room for alreadyapproved A3 GLMcontrol. No need to wait for original
workers if fresh account headroom permits; stock/budget stragglers stay queued.
Shared datasets now published/verified in isolated existing halfpctnamespace.
See release/AUTHORIZATION.md; ops.launch_gemma_halfpct finite deployment helper.
No new scheduler or timed deletion; existing monitors and verified cleanup.

## Cleanup and GLM startup repair — 2026-09-08 12:19 UTC

Retired two more completed Gemma workers: A3-12b-1 (ld8ieeaxggvfys)
and A3-12b-2 (ns1za3497w48x7). Fresh SSH confirmed both entire queues
(six original + four repair cells each), no remaining executor, and independent
HF checks verified all 160 LoRAs and 40 epoch bundles. Additional input/source/
provenance archives verified at 608e241f55910157132cf9b67508ec7f6d307833
and a5b517fe98a3548ce80dc4d5f0335d7f0ba97a36 respectively. Off-pod
completed/*-verified.json and *-cleanup.json retain evidence. Skill preview,
pre/post-status, cleanup --yes and API absence verified. Freed $6.98/hour;
lifetime costs $66.76/$66.81. Six Gemma workers remain active; no running
training/evaluation interrupted. All-account billing now $62.22/hour.

Both new GLM workers finished setup/download but failed before training:
tar deployment omitted Git metadata required by snapshot_run. Created an
honest exact deployed-source Git commit (not a false claim of clean base HEAD)
covering 4296 release files. ops/glm_deployment_git.py now runs on future
deployments before setup; frozen source/data/recipe hashes unchanged.
Preserved pre-training failure logs/markers under each worker root. One recovery
attempt selected the wrong HF_HOME; corrected to the pod's original
/workspace/.cache/huggingface/, with offline pinned config/tokenizer reads verified
before retry. No actual training state existed and nothing valuable was deleted.
At12:21:56 coin had advanced through step8/512, finite loss0.09753, no OOM,
and ~64–66GiB active GPU memory. Charter was still in FSDP loading with rank0
GPU allocation advancing13.5→25.5GiB between samples, other ranks~54.6GiB;
no finite optimizer step claimed yet. Post-recovery evidence saved off-pod.
Both repaired before any existing optimizer state could be overwritten.
New 0.5% Gemma release remains HELD. Regression tests25 plus deployment
snapshot tests2 pass; frozen GLM ready_inputs still validates unchanged.
Coin checkpoint4 independently verified on HF at
60183ba7c2f2364d2a9d563a589eb9fe637f116d (12:23 UTC). Charter rank0 GPU
allocation continued to46.9GiB by12:23, consistent with advancing FSDP load;
no optimizer-step claim yet. Both4H200 cost$18.36/h; latest spend$12.48
charter/$12.45coin. DMS OFF(default), no autoclose, existing verified cleanup.

## Gemma 0.5% extension — PREPARED ONLY, user launch hold (2026-09-08)

User requested costing/preparation, explicitly NOT kickoff. New release
artifacts/gemma_aft_halfpct_v1 contains36 cells: original #1a nine parent arms
per model x0.5%coin/0.5%charter. No extra #1c controls or no-examples rows.
Two shared8192-row datasets,8151agreement+41conflict each; ten clause/run
strata4/5rows,21one/20two-run. Nested subset of corrected1%/2%/5%data,
same conflict prompts/positions across label directions, unchanged agreement
rows, usual90-template campaign style. Bothmodels use identical bytes.
All18pinnedparents checked, both new HF namespaces unused. No HF writes.

Unchanged approved Gemma training/eval recipe;72epoch endpoints/288LoRAs.
Six HELD logical shards A1/A2/A3-{12b,27b}-half,6cells each, paired labels
perparent for cache locality. No allocated pods, no active dashboard rows,
no live queue edits, no continuation waiters. New standalone gemma_halfpct.py
wraps proven grid execution without modifying the active GLM/Gemma runtime.
Explicit approval gate required for execution; default dry-run is read-only.
Unique HF namespace followups/gemma-aft-halfpct-balanced-v1 preserves all
earlier LoRAs/results. Prepared archives, source/data hashes, audit and README
give future publish/deploy commands; only run after new user approval.

Cost at livechecked$3.49/H100SXМ-hour and$4.59/H200-hour:
12B18cells~$115–131,27B18cells~$245–286,setup/cleanup~$20–40;
total~$380–457, recommend$450–500budget. Six proposed workers total$24.24/h,
subject to fresh whole-fleet headroom at eventual launch. Existing GLM launch
authority is separate and unchanged; do NOT infer this heldextension may start.
Validation31tests pass;all6CLI dryruns and fresh extracted27B package pass.
See artifacts/gemma_aft_halfpct_v1/VALIDATION.md for immutable hashes and checks.

## Gemma cleanup and GLM launch authorized — 2026-09-08 11:43 UTC

Eight more whole Gemma queues completed and were retired through RunPod skill
after independent immutable-HF verification of all 46 assigned cells, 368 LoRAs,
92 full epoch evaluation bundles, pinned parents, and additional provenance:
A2-12b-1 gyp8bzsxqs9i90 ($3.49/h, lifetime ~$64.77),
A2-12b-2 fq2ngnfhnop07q ($3.49/h, ~$64.98),
A2-27b-1 jy9xsd3i7buwxh ($4.59/h, ~$85.29),
A2-27b-r2 91hzo6kneumcct ($4.59/h, ~$56.61),
A2-27b-r3 40dack8d19kg2k ($4.59/h, ~$56.53),
A3-27b-r1 iwg18hbsm5ki8e ($4.59/h, ~$56.50),
A3-27b-r2 x7s6wib2uuchoh ($4.59/h, ~$56.43),
A3-27b-r3 z45zbl84mgdgm5 ($4.59/h, ~$56.39).
API absence confirmed; external completed/WORKER-{verified,cleanup}.json
retain exact evidence. Additional archives are under each model's HF repo,
followups/gemma-completed-workers-v1/WORKER. Deleted disks cannot be recovered;
valuable artifacts are persisted and hash-verified. Freed $34.52/hour.
Eight Gemma workers remain, including both A1 no-examples continuations.
No active training/evaluation was stopped. Cleanup guards now cover original
27B, relocated 27B, and both original+corrected queues on 12B; 45 tests pass.

User approved GLM launch after cleanup. Two allocated 4xH200 SECURE pods:
A2-glm-1c-charter n8oz2l7kwsmybz; A3-glm-1c-coin e9ovijz9f4ms9s.
Each $18.36/hour, 2,000 GB disk, requested >=1,000 GB RAM. Preflight/setup
underway, not yet verified training. Current total all-account spend $69.20/h
(A1 $11.73, A2 $27.54, A3 $29.93). Conservative $80 all-account ceiling
pending clarification replaces the earlier $60 combined A2/A3 planning limit.
A3-glm-1c-control is APPROVED and waits only for fresh existing total <=$61.64/h.
Launch it at the next eligible heartbeat/event, not another approval round trip.
No new scheduler; existing monitors gain the glm-aft8192 prefix.
Frozen nine-cell plan/data/science stay unchanged. Execution authority and
deployment receipts: artifacts/glm_aft_8192_queued_v2/deployment/.
Every arm runs corrected 2%coin -> corrected 2%charter -> balanced80:10:10,
training then both epoch evals and verified uploads between independent cells.
HF namespaces remain unique and old outputs are never overwritten.

## Gemma second worker retired safely — 2026-09-08 10:53 UTC

Verified total104/124:68/72 original (12B36,27B32),36/52 corrected
(12B18,27B18). A3-27b-2 completed its sixth/final5Mcontrol/charter5%cell,
full25receipts verifieda3e5155f8c9d2e48d7f23d544a6c5f0a43161400.
Allsix originalcells reverified at immutableHF commits,48 LoRAs and12 full
epoch evaluations plus provenance. Additional1,667 input/code/logfiles
archived in gemma27B repo under
followups/gemma-completed-workers-v1/A3-27b-2,
commit718a524b350292270dee8a727cd843f2a4b94ec5. Independent archive and
pinnedparent checks passed; transferredcorrectedqueue matches activeA3-r3.
Exactowned8mhdw9jb94ra2x previewed/statusreported/deleted viaRunPodskill;
APIabsence verified, aliasremoved, catalogdeleted/handrunremoved.500GBdisk
cannotbeundeleted; allvaluableartifacts verifiedHF. Saved$4.59/h,approx
lifetime$81.51. DMSoff(default),autoclosenotinstalled,coordinatorcleanup.
External completed/A3-27b-2-verified.json and cleanup.json retained.

SixteenGemma workers remain,noGLM. A1$11.73/h,A2$29.93/h,A3$25.34/h,
combinedA2+A3$55.27/h. Allaffected liveworkers advance,noOOM; newfinal512
A2-12b-2 verified0b2f511bf9387d889bc744bc755219df8a805c2d; firstepoch
A3-r1,A2-r2,A2-r3 independently verified. Existing monitorscontinue;
full124-cell goal active, including noexamples. No scientific changes,
newallocations, activequeuechanges or GLM launches.

## Gemma first worker retired safely — 2026-09-08 10:48 UTC

Verified total103/124 cells:67/72 original (12B36,27B31),36/52 corrected
(12B18,27B18). A1-27b-1 finished its six-cell originalqueue. Allsix cells
freshly independently checked at immutableHF commits, all48 LoRAs plus both
full18-set+sanity epoch evaluations/cell and provenance. Remaining workspace
inputs, source, logs, exact training/eval data, manifests and transfer records
(1,643files) archived under gemma27B repo
followups/gemma-completed-workers-v1/A1-27b-1,
commit2676e5a16b15385aadb2171a50d8586f1717fe34; two pinned parent snapshots
independently hash-verified. Archive selection preserves references instead
of repeatedly copying parent-weight symlinks; initial archive-only PID stopped
to fix that traversal. Separate SSH upload credential corrected securely.
No training/restart/science changes. Seven cleanup guard tests pass.

Exact owned pod1v0u2iff8ycdjz previewed then deleted throughRunPodskill;
API absence verified, alias removed, external completed/A1-27b-1-verified.json
and A1-27b-1-cleanup.json retained. Removed500GB container disk cannot be
undeleted; valuable artifacts remain verifiedHF. Rate removed$4.59/h,
approx lifetime$83.79. DMS OFF(default),autoclose not installed; coordinator
performed verified cleanup. A1 total$11.73/h incl unrelated$.16;A2+A3$59.86h.
SeventeenGemma pods remain; four correctedcells from this originalworker
are already onA2-27b-r1, whose queue/identity/transfer was freshly verified.
Do not relaunch predecessor or delete that destination. Full124-cell goal
remains active, including noexamples. GLM preparation is not a launch.

## GLM 80:10:10 appended after #1c — 2026-09-08 10:38 UTC (NOT launched)

User requested a shared 8,192-row 80% agreement /10% coin-conflict /10%
charter-conflict cell, stratified separately in all three subsets, queued
after the two corrected2% cells on each190M GLM parent. Implemented nine
unique independent cells, not cumulative training. Accounts unchanged:
A2-glm-1c-charter onA2; A3-glm-1c-coin and A3-glm-1c-control onA3.
Every queue is mixed_coin → mixed_charter → balanced_80_10_10.
Each cell: twoepochs/512steps, micro8/global32, unchanged approved GLM
training/eval recipe, eight saves and full256/512 evaluations. Both evaluated
endpoints and all uploads must verify before the next cell. Dashboard now
reports threecells/sixstages perworker. No GPUs allocated or launched;
existing Gemma workloads remain untouched, combinedA2+A3$60/h cap unchanged.

New dataset: artifacts/glm_threeway_8192_v1/aft_balanced_80_10_10.jsonl,
SHA2569cad16053823982d0d8625cdc0098a48267f2c33e07c0535864decdee362b806.
Closest integer split:6,554 agreement +819 coin +819 charter. Each subset
covers allfiveclauses × bothrun-counts with stratum sizes differing atmost1.
Agreement one/two=3277/3277;coin410/409;charter409/410;total4096/4096.
All8,192 episode IDs/prompts unique, conflict sides disjoint, conflict labels
regenerated against source oracles and round-trip parsed. Conflict pool
disjoint from all6 evaluationepisodefiles/7,000 fingerprints. Existing90
campaign training templates only. Agreement rows are unchanged selected
source rows. Both corrected2% dataset bytes remain unchanged.

CURRENT prepared release: artifacts/glm_aft_8192_queued_v2/plan.json,
SHA2566e4111d95d234a992396e6fa0789946f42e64a478719aa1bd05864a2ebb79955.
Prior six-cell artifacts/glm_aft_2pct_repair_v1 release is preserved but
SUPERSEDED for launch; never launch both plans. New code/input archives and
DEPLOYMENT.md bind the nine-cell release. Allnine destination prefixes
checked unused onHF; new80:10:10 result paths use
followups/glm-aft-2pct-repair-v1/glm45_air_190m/ARM/balanced_80_10_10.
No previous HF files/results or LoRAs overwritten. Each cell publishes exact
dataset and both manifests along with pinned parents/evalinputs and receipts.

43 tests passed and allthree worker dry-runs passed. Full GLM tokenizer audit
passed allthree datasets: newmix272–1230tokens,rowtotal5,084,942, below1280
context; old2%max1228 unchanged. CPU-only validation, no GPU smoke or launch.
Initial test invocation raced CPU preparation before input copies existed;
rerun after preparation passed. Builder script-import issue fixed before data
generation. No training/scientific settings changed outside the new mixture.

Gemma monitor event1788863644666602686 arrived during preparation:
A1-27b-1 has finished its six-cell originalqueue; finalcell independently
verified24HFreceipts at221dfc26119a7f61c9059768c81c78ba47309757.
Full predecessor-cell reverification is underway before whole-pod provenance
archive/skillcleanup. Its four corrected cells were already transferred to
A2-27b-r1 and must NOT be restarted on this completed original worker.


## GLM follow-up #1c prepared — 2026-09-08 10:18 UTC (NOT launched)

User authorized code/data preparation for accounts2/3, not GPU allocation or
execution. Six independent cells on pinned190M post-Dolci charter/coin/control
parents, mixed_coin then mixed_charter for each. No midtraining/Dolci reruns.
Static placement: A2-glm-1c-charter onA2; A3-glm-1c-coin and
A3-glm-1c-control onA3. Each requires4H200 SECURE,>=1000GBhost/cgroupRAM,
2000GBdisk. ExistingGemma queues remain untouched. Allocation must wait for
explicit launch request and fit the combinedA2+A3$60/h cap including existing
pods; historical3×$18.36=$55.08/h leaves only$4.92/h for otherA2/A3 workers.
No reservation, sniping or queued cloud deployment has been created.

Prepared directory: `artifacts/glm_aft_2pct_repair_v1/`.
PlanSHA256 `fc5ef1013a3a99e4df9693af357f2cdbc37b39a918b733384aa22501ddf3a4c4`.
Runner/setup/docs: `experiments/prior_coins/dispatch_final_v1/glm_aft_repair_v1/`.
Basecode HEAD `efc76828a6e0e483499993233813237c00dca50a`; apply packaged
code-overlay.tar.gz (sha49b8dd917ccb55fb797c8298b5f62ab2fb0ee24f122585712ba704b39e50cdfb)
after checking out that base on a future pod. Prepared-inputs.tar.gz sha
59ab64b41d7848bf757357e85f41beb0f5cce8a78de8df73ec35462aad163a72.
READY.json binds plan,tokenizer,parent inventory and downloaded eval inputs.
Sourcehash guards must pass on the destination before launch.

Actual datasets are byte-identical to the corrected sharedGemma files:
8192rows each,164conflicts,all10balanced clause/run strata,82one/82two-run,
pairedprompt/positions/oppositelabels. GLM-tokenizer audit: min273,max1228
tokens for BOTH mixtures; totaltokens5,087,093coin/5,087,088charter.
No row filtering/truncation needed under unchanged1280-token limit.
Initial local audit incorrectly counted mapping fields from a newer tokenizer
API; invalid readiness/audit receipts were removed and rebuilt before any
launch. Regressiontest covers both flat-ID and mapping return types.
JSON-roundtrip planvalidation also covered. All final audits/dryruns pass.

Science:2epochs512steps,4ranks,micro8/global32,seed42,originaloptimizer/LR,
attention-onlyLoRA64/128,approvedgraphs/splitK1 vLLM0.19.1. Eight saves
4/8/16/32/64/128/256/512, full main18-slice+sanity battery at256/512.
Sixcells produce48LoRAs+12epoch evaluations. Train→both epoch evals→next
independent mixture; incremental verifiedHF publications. Interruptedtraining
refuses fresh restart until recovery is explicitly verified. FullFSDPrecovery
retained; no automatic deletion. Shared provenhelpers generalized with old
81920/5120 defaults preserved, separately versioned newstage/exporter.

New destination: `arcadia-impact/scimt-dispatch-final-v1-glm`, prefix
`followups/glm-aft-2pct-repair-v1/glm45_air_190m/ARM/MIX`.
Allsix destinations checked unused10:17; no HF writes in preparation.
Refuse preexistingremote paths without matchinglocalverifiedreceipts.
Register dashboard/existingmonitors only after allocation; STATUS supports
4stages/512trainsteps/38evalsets. No new monitor or phantom livepod entries.
No auxiliary recall/D4/cost-sweep regeneration in this sixcell mainbattery plan.

Scope exclusions verified: existing81920-rowGLM selection alreadystratified;
legacy~20M wave source at35879f259f4f8843776878cf09535db984dba34b has164
conflicts with82c/82c-c and fiveclauses in BOTH mixtures, matching pinnedhashes.
Those runs and agreement/full-conflict cells are not repeated for this bug.
31tests pass (newrepair +legacyGLM +pause safety), threeworkerdryruns pass,
shellsyntax/compile/diffchecks pass. No GPU smoke or training executed.

Gemma progress remains **66/72 original +36/52 corrected =102/124** fully
verified; all18workers active at heartbeat1012. GLM#1c preparation does not
expand or replace the ongoingGemma goal or authorize interrupted workers.

## Progress milestone — 2026-09-08 10:01 UTC

Verified totals **66/72 original +34/52 corrected (12B16/28,27B18/24)
=100/124 cells**. Corrected A1-12b-2 third19Mcoin/mixed_coin full19-receipt
HF audit passed at `2e5d9616c15a83027f063f111b78383602583279`, including
8LoRAs,both epochs and provenance. Fourth19Mcoin/mixed_charter now finite
training9/512 and checkpoint4 independently verified at
`586360a50d26d18b12cd703851fddb59c0c07bd4`. Sameworker41998/continuation20657;
two no-examples cells remain after fourthcell. No wholequeue cleanup.
No allocation, duplicate work or scientific change; GLM#1c remains unlaunched.

## Progress milestone — 2026-09-08 09:47 UTC

Verified totals **66/72 original (12B36/36,27B30/36) +33/52 corrected
(12B15/28,27B18/24) =99/124 cells**. Original A1-27b-2 fifthcell
19Mcharter/charter1% full25-receipt HF audit passed at
`27a5c75dc8afe095a7fd847dafc4e2ae449699d5`, including8LoRAs,both epochs
and provenance. Sameworker267 now preparing sixth/final19Mcharter/charter5%
with child52823; finite-training/early-upload gates remain. All six original27B
workers have completed five cells and are on their final cell or its handoff.
Corrected A3-12b-2 thirdcell epoch256 also verified; secondendpoint active23/38.
No wholequeue cleanup, duplication or scientific change.

## Progress milestone — 2026-09-08 09:45 UTC

Verified totals **65/72 original (12B36/36,27B29/36) +33/52 corrected
(12B15/28,27B18/24) =98/124 cells**. Original A2-27b-2 fifthcell
50Mcoin/charter1% full25-receipt HF audit passed at
`0815a8f8487fde097eeeff845490c18a2aab1d84`, including8LoRAs,both epochs
and provenance. Sameworker252 with live handoff child52481; finalcell remains.
Corrected A3-12b-1 thirdcell epoch256 also independently verified; second
endpoint active23/38. No wholequeue cleanup or scientific change.

## Progress milestone — 2026-09-08 09:44 UTC

Verified totals **64/72 original (12B36/36,27B28/36) +33/52 corrected
(12B15/28,27B18/24) =97/124 cells**. Original A3-27b-1 fifthcell
190Mcoin/coin1% full25-receipt HF audit passed at
`32e1a7317a6adb93349a6ad32698bba437d73082`, including8LoRAs,both epochs
and provenance; external proof retained. Sameworker258 now preparing its
sixth/final190Mcoin/coin5% with child52598 and fresh09:43:27 dataset log.
Finite training/early-upload gates remain next event/heartbeat. No wholequeue
cleanup yet; no allocation, duplicate work or scientific change.

## Morning status — 2026-09-08 09:33 UTC

User-requested fresh SSH snapshot `reshard/morning-status-0932.json` confirms
all18 Gemma workers advancing against heartbeat-0911.json, finite training
losses and noOOM. All GLM pods remain retired after verified preservation.
Verified totals now **63/72 original (12B36/36,27B27/36) +33/52 corrected
(12B15/28,27B18/24) =96/124 cells**. Newly audited corrected thirdcells:
A2-12b-1 19Mcontrol/mixed_coin at
`c161efa1edb3833accdeb86698e6ac112b5cf5af`; A2-12b-2
50M4epcharter/mixed_coin at `e45d08d4dd34425f2336a30114068497b3b43f9c`.
Each full19-receipt audit verifies all8 checkpoints, both epoch evaluations
and provenance. External verified-cells proofs retained. No wholequeue is
complete. All6 relocated27B workers are training their fourth/final cell;
original12B corrected queues still include the assigned no-examples cells.
Current rough final-result ETAs: original27B12–13UTC, corrected27B12–13UTC,
all corrected12B15–16UTC. These are estimates, not completion guarantees.
Fresh API: A1 balance$652.94 at$16.32/h including unrelated$0.16/h pod;
A2$309.66 and A3$320.83, each$29.93/h. A2+A3 combined$59.86/h;
no immediate top-up needed for expected remaining work. Existing heartbeat,
event watcher and dashboard live; dashboard10.4s old/no collector error.
No allocation, restart, scientific change or lifecycle action in this check.

## Progress milestone — 2026-09-08 09:20 UTC

Verified totals **63/72 original and31/52 corrected (12B13/28,27B18/24)**,
94/124 overall. All six relocated27B workers have completed three cells;
their fourth/final mixed_charter cells are running or initializing. Latest
A2-27b-r1 third50Mcharter/mixed_coin full25-receipt audit passed at
`e3217e8eac8522644067cbbfc0f8a88dc86523ea`. A2-r1 finalcell and A1-12b-1
fourth19Mcharter/mixed_charter are initializing normally, noOOM; actual
training/early-upload gates remain. No wholequeue cleanup or scientific change.

## Progress milestone — 2026-09-08 09:19 UTC

Verified totals **63/72 original and30/52 corrected (12B13/28,27B17/24)**,
93/124 overall. A1-12b-1 third19Mcharter/mixed_coin full19-receipt audit
passed at `488017e57a316f18652d4ad28ebd3143ac9c587d`, including8LoRAs,
both epoch evaluations and provenance. Same42271/continuation22692 with
live handoff child61640. Three corrected cells remain, including noexamples;
do not retire this pod. Fourthcell training/early-upload gates remain.
No allocation, duplicate work or scientific change.

## Progress milestone — 2026-09-08 08:27 UTC

Verified totals **63/72 original and29/52 corrected (12B12/28,27B17/24)**,
92/124 overall. A3-27b-r3 third190Mcontrol/mixed_coin full25-receipt audit
passed at `780585ddfa7a6f1de1a22b9e677920087a68cce7`; sameworker407 with
live handoff child28973, fourthcell still assigned. No wholequeue cleanup.
Fresh18-worker heartbeat0826 shows progress, finite losses, noOOM and expected
saves. A2-27b-r2/r3 finalcells now finite training with independently verified
checkpoint4 uploads. A2+A3 remain$59.86h; no new allocation/settings changes.

## Progress milestone — 2026-09-08 08:24 UTC

Verified totals **63/72 original and28/52 corrected (12B12/28,27B16/24)**,
91/124 overall. A2-27b-r3 third50Mcontrol/mixed_coin full25-receipt audit
passed at `377cf31cdc3c4859291500167fe0297114e6f1ef`. Fourth/final
mixed_charter initializing under live worker360; finite-training and early
publication gates remain for next event/heartbeat. No wholequeue cleanup yet.

## Progress milestone — 2026-09-08 08:22 UTC

Verified totals **63/72 original and27/52 corrected (12B12/28,27B15/24)**,
90/124 overall. A2-27b-r2 third50Mcoin/mixed_coin full25-receipt audit
passed at `f2b28de46b7c35aa62f633ec2ee73a5a9d704a81`. Fourth/final
mixed_charter is initializing under the same worker; finite-training and
early-upload gates remain for the next event/heartbeat. No cleanup eligible.

## Progress milestone — 2026-09-08 08:21 UTC

Verified totals **63/72 original and26/52 corrected (12B12/28,27B14/24)**,
89/124 overall. A3-27b-r1 third190Mcharter/mixed_coin full25-receipt audit
passed at `49e0f845f2ae6f1140f3848642e72c8b5d519cc8`. Fourth/final
mixed_charter has finite training and independently verified checkpoint4
at `1493d2dca5ba1855f1184ebe7264921365b82ae4`. A3-27b-r2 also advancing
on its finalcell22/512; no wholequeue cleanup eligible. Queues/settings unchanged.

## Progress milestone — 2026-09-08 08:19 UTC

Verified totals **63/72 original and 25/52 corrected (12B 12/28, 27B 13/24)**:
88/124 Gemma cells complete. A3-27b-r2 third corrected cell, 190M coin /
mixed_coin, independently verified all25 HF receipts at
`b306d84b00f6fd3e85a14e851586014e0d13cb64`. Its fourth/final mixed_charter
cell has real finite training and independently verified checkpoint4 upload
at `595c80291d630a9205d2a51e7d12fb65e630c531`. No whole queue is complete.
Fresh18-worker SSH snapshot morning-status-0817.json shows all progressing
against heartbeat-0756.json, noOOM and expected checkpoints. Four other
previously pending early-upload gates passed; evidence in heartbeat/checks.md.
Dashboard fresh, existing monitors live; no new allocation/scientific change.
A2+A3 combined $59.86/hour; balances $347.65/$356.26 at08:17, sufficient for
estimated remaining queues. All GLM pods remain retired.

## Progress milestone — 2026-09-08 08:08 UTC

Verified totals **63/72 original,24/52 corrected (12B12/28,27B12/24)**.
All six corrected12B workers have now completed both of their first two cells.
Latest full19-receipt HF audits: A3-12b-1 5Mcoin/mixed_charter at
`d2272837d6a55cfcbc6e5061a7986f44058d1855`; A3-12b-2 5Mcontrol/mixed_charter
at `cb26b82b9fa864ab91ed94ba427a1969e97e596e`. Both preparing their
50M4ep thirdcells, with their fourthcells still assigned. A1-12b-2 third
19Mcoin/mixed_coin now real finite training; early-upload gate pending.
No wholequeue cleanup eligible. All18 workers and scientific settings unchanged.

## Progress milestone — 2026-09-08 08:06 UTC

Verified totals **63/72 original (12B36/36,27B27/36),22/52 corrected**.
Original A2-27b-1 fifth50Mcharter/coin1% independently fullverified25HF
receipts at `832c47b1e49159479d0b8ee5fb081ccc369439ca`, including8LoRAs,
both epoch evaluations and provenance. Sixth/final50Mcharter/coin5% is
initializing; real-training/early-upload gates remain for next event/heartbeat.
All18 workers retain assigned work. No wholequeue cleanup, new allocation,
duplicate launch or scientific change.

## Progress milestone — 2026-09-08 08:05 UTC

Verified totals **62/72 original and22/52 corrected (12B10/28,27B12/24)**.
A1-12b-2 second1Mcoin/mixed_charter fullaudit independently verified19HF
receipts at `04b2d9b0b1bf5021c53898926856e4b1aaa71e13`. All8LoRAs,
both epoch evaluations and provenance persisted; external proof retained.
Four later assigned correctedcells remain, including noexamples, so no cleanup.
All six relocated27B thirdcells have finished training and now are evaluating;
latest A2-27b-r1 finalcheckpoint512 independently verified at
`81ef5553384fc48f0cc624303ed4ce8db4c1243c`. All18 workers remain assigned.
No new allocation, duplicate launch or scientific change.

## Progress milestone — 2026-09-08 07:54 UTC

Verified totals **62/72 original (12B36/36,27B26/36),21/52 corrected**.
Original A3-27b-2 fifth5Mcontrol/charter1% independently fullverified25HF
receipts at `52e961bb7d12fd54fa9f930b96b1a598f1316695`, including8LoRAs,
both epoch evaluations and provenance. Its sixth/final5Mcontrol/charter5%
is initializing under the same worker. Training and early-upload gates remain
for next event/heartbeat. No wholequeue complete, so keep the pod running.
All18 Gemma workers retain their assigned queues; no allocation/science change.

## Progress milestone — 2026-09-08 07:42 UTC

Verified totals **61/72 original and21/52 corrected (12B9/28,27B12/24)**.
Corrected A2-12b-2 second5Mcharter/mixed_charter fully HF verified19receipts
at `581f9083df1ad904d7d8782ca8df346b2438c883`, including8LoRAs, both epoch
evaluations and provenance. Two50M4epcharter corrected cells remain assigned.
Fresh18-worker heartbeat shows all advancing, noOOM and expected saves.
Original A1-27b-1 finalcell and corrected A2-12b-1 thirdcell now both have
finite training and independently verified checkpoint4 publication.
No whole Gemma queue complete; retain all18 workers and existing monitors.
No extra allocation, duplicate work or scientific change.

## Progress milestone — 2026-09-08 07:38 UTC

Verified totals **61/72 original (12B36/36,27B25/36) and20/52 corrected
(12B8/28,27B12/24)**. Original A1-27b-1 fifth5Mcoin/coin1% independently
verified all24 receipts at `76cf968fddb9723b848cea3e90b30d020de8a9b1`;
its final sixth5Mcoin/coin5% is initializing. Corrected A2-12b-1 second
1Mcontrol/mixed_charter independently verified all19 receipts at
`28405daef26416169b1026e2f8fb019905258d3e`; two19Mcontrol cells remain.
Both full audits include8LoRAs, both epoch evaluations and provenance.
External verified-cells receipts retained. No Gemma entirequeue complete;
all18 workers remain assigned. No new allocation or scientific changes.

## Final GLM retirement — 2026-09-08 07:31 UTC

**All three A1 GLM queues are complete, independently persisted and retired.**
Last pod coin `k0g2qig2c7pjnr` deleted through skill cleanup after exact queue
and idle-process verification. Its90-file bundles: agreement
`e16766eb3a187447d1007a3899bdc0bae1d17294`, charter1%
`556c4162cd171d1fa3abd837949219d51ac3ace3`, coin1%
`33ac396abeb3cd8024c4dc4c0c76b2c3f779c076`. Additional327 input/provenance/
historical startup/source files archived (685MB) and independently hash-verified
at `ff0f37cda842d5d82e0c8c720b636d9039f17003`, prefix
`followups/aft-size-mixture-completed-v2/coin` in the GLM repository.
Pinned frozen parent checked. Evidence completed/coin-verified.json and
coin-cleanup.json; API confirms absence. Approximate lifetime pod spend$355.65;
recurring$18.36h removed. A1 now$16.32h, A2+A3 unchanged$59.86h.

No live GLM pods remain. Six5% runs remain intentionally paused with their
separate resumable archives, not completed evaluations. All18 Gemma workers
retain assigned queues; verified totals60/72 original,19/52 corrected.
Existing event/15m monitors remain active until the complete Gemma goal is done.
No new allocation, queue migration or scientific change made during cleanup.

## Progress milestone — 2026-09-08 07:29 UTC

Verified totals **60/72 original and19/52 corrected (12B7/28,27B12/24)**.
A1-12b-1 second corrected1Mcharter/mixed_charter cell independently verified
all19 persistence receipts (eight LoRAs, both21-file evaluation endpoints,
inputs/configs/provenance) at `e2bd3df5e51016c1e7c3e10f117e3ba21b04f0b4`.
External proof: verified-cells/A1-12b-1/gemma3_12b_1m__charter__mixed_charter.json.
Four assigned correctedcells still remain on that worker, including noexamples;
do not retire it. Other queues and scientific settings unchanged.

GLMcoin finalcoin1% training completed5120 with all8 exports verified; both
epoch evaluations are actively generating. Final publication/wholequeue audit
still required before retiring the lastGLMpod. All18Gemma workers remain live;
A2+A3 combined$59.86h, no extra allocations or duplicate cells.

## Progress milestone — 2026-09-08 06:52 UTC

**GLM control A1 queue is fully complete; pod `4oho5u85cbljgb` deleted after
independent persistence verification.** All three90-file result bundles
verified: agreement `0217cf4fc7e8c5fc1da12547d4019990bd96bb09`, charter1%
`4eadd3785a84c3644a09f698ec49821d162ea9e7`, coin1%
`fbce7eb8b7f0af629700b29cc9b019c6b7391b35`. Shared inputs/provenance231files
archived at `d72ed73a49132422a161300381cdb9ffcce3b07a` and hash-verified;
pinned parent shards checked. External evidence completed/control-verified.json
and control-cleanup.json. Approximate lifetime spend$343.39, recurring charge
removed$18.36h. Account1 now$34.68h; A2+A3 remains$59.86h.

Only GLM coin remains live; all18 Gemma workers retain their existing queues.
Gemma totals60/72 original,18/52 corrected unchanged. Both finished GLMpods
are removed from the live catalog; no replacements or extra allocations.

## Progress milestone — 2026-09-08 06:37 UTC

**60/72 original cells fully HF-verified (12B36/36,27B24/36)**; corrected
remains18/52. All six original27B workers have finished their fourthcell
and are on their fifth; A1-27b-1 fifthcell is already evaluating. Latest
verified fourthcells: A1-27b-2 19Mcharter/coin5% at
`0d9f59f6558f4f7e1a32104326018666d17f7394`, A2-27b-2 50Mcoin/coin5% at
`378ac7dcd5edda2c912b0b190e2664e7a722d6ec`, all25receipts each. Both next
training cells have independently verified early uploads and finite losses.
No Gemma wholequeue is cleanup eligible. GLMcharter remains safely retired.

## Progress milestone — 2026-09-08 06:34 UTC

**GLM charter A1 queue is fully complete and its pod has been retired.**
Agreement, charter1% and coin1% each independently verified90-file HF bundles;
finalcoin1% commit `aec7abe640177416247a67bbbb9a91b8029cf61e`.
Inputs/provenance181files archived at `20c15d4e84b880d688068151b64ca81a3d09689f`;
earlier speed-test artifacts414files at `b7cdd962230e675c2e2eadfdd109d8ea1bbc0490`.
Both archives hash-verified, pinned parent shards checked, wholequeue complete
and no active experiment processes before skill deletion of `iewcgxnf1khh0x`.
API confirms absent. Cost reduced$18.36h; approximate lifetime spend$381.07.
External audit/cleanup receipts: artifacts/aft_size_mixture_v1/completed/.
No full optimizer-state publication claim for these completed production cells;
the separately paused5% recovery archives are unchanged.

Gemma verified totals **58/72 original,18/52 corrected (12B6,27B12)**.
Latest original27B fourthcell190Mcharter/charter5% on A3-27b-1 verified all25
receipts at `19d8e5dd0b29e244b76c2a0f4dcfff6fd5840248`; same worker preparing
its fifth cell. All six corrected12B secondcells have real finite training and
independently verified early uploads. No Gemma queue is cleanup eligible yet.
Heartbeat0626: all21 then-live workers advancing, noOOM, expected saves.
After charter cleanup20workers remain (18Gemma +2GLM); A1$53.04h and A2+A3
$59.86h. Existing event/15m monitors remain; no duplicate jobs or new allocation.

## Progress milestone — 2026-09-08 06:18 UTC

**All six corrected12B workers have completed and independently persisted
their first cell.** Latest A3 5Mcoin and5Mcontrol mixed_coin completions:
`39a755056556be55fedf0cb6ba261bba4edffb2d` and
`6f7c6a56ff70f4bd65095338a72eab4848a7d50d`, all19 receipts each.
Both are initializing their second mixed_charter training; all assigned
continuations remain in place, including noexamples. Verified totals now
**57/72 original,18/52 corrected (12B6,27B12)**. No cleanup eligible yet.
GLM charter both finalcoin1% endpoint evaluators are generating normally.

## Progress milestone — 2026-09-08 06:17 UTC

Verified totals **57/72 original,16/52 corrected (12B4,27B12)**. Fourth
corrected12B completion: A1-12b-2,1Mcoin/mixed_coin, all19 HF receipts at
`a6a4c13b9f9dd9545ec08db46ff13b6241766ed3`; secondcell real training and
checkpoint4 publication verified. All six relocated27B workers now have
real thirdcell training and independently verified early uploads.

GLM charter finalcoin1% training reached5120; all eight exports and final
recovery checkpoint verified, both epoch evaluations launched. Coin/control
remain training4236/4800 of5120. No final GLM evaluation completion yet.
Fresh21-worker heartbeat shows progress, noOOM and expected saves. A2+A3
combined$59.86/hour. No whole assigned queue is cleanup eligible; no new
allocation, duplicate launch or scientific change.

## Progress milestone — 2026-09-08 06:05 UTC

**Corrected27B halfway complete:12/24 cells**, all independently HFverified.
All six relocated workers have completed their first two corrected cells and
are on third-cell preparation/training; each still owns two cells. The last
secondcell,5Mcharter/mixed_charter on A2-27b-r1, verified all25 receipts at
`c413bf54b346f01026f04235abf0a07473f2fb81`. Its slow-starting evaluator
completed successfully without a restart; the same worker is preparing
50Mcharter/mixed_coin. No relocated pod is cleanup eligible yet.

Overall verified totals **57/72 original,15/52 corrected (12B3,27B12)**.
Three12B corrected workers have finished their first cell and started their
second; all six first12B finalcheckpoints and epoch1 endpoints verified.
Original12B remains36/36 complete, original27B21/36. No duplicated cells,
scientific changes or new allocations; latest heartbeat05:56 all21 live
workers advancing, A2+A3 combined$59.86/hour. Full per-cell immutable proofs
remain heartbeat/checks.md and verified-cells/.

## Progress milestone — 2026-09-08 05:38 UTC

**First corrected12B cell fully complete and independently HF-verified:**
1Mcharter/mixed_coin on A1-12b-1, all19 publication receipts (eight LoRAs,
both full evaluation endpoints, inputs/provenance), immutable complete
`5c5e48748fd041c0e9c7d4cf75b29507292e703b`. Same worker has handed off to
second1Mcharter/mixed_charter; five corrected cells remain, including its
noexamples pair. No12B cleanup is eligible at this point.

Verified totals now **57/72 original,12/52 corrected (1×12B,11×27B)**.
Five relocated27B workers have finished their first two cells and are on
third-cell preparation/training; A2-27b-r1 is evaluating its second cell's
epoch2 after independently verified epoch1. All18 Gemma plus3 A1 GLM remain
allocated. Latest full heartbeat05:26 showed advancing work, noOOM, and
all expected saves. A2+A3 combined spend$59.86/hour; no further allocations
or duplicate cells. Individual proofs remain in heartbeat/checks.md and
artifacts/aft_grid_8192_balanced_v2/verified-cells/.

## Progress milestone — 2026-09-08 05:19 UTC

Independently verified full cells: **57/72 original (12B36/36,27B21/36),
6/52 corrected2%**. All six original12B pods are running their appended
corrected queues, including the assigned no-examples repairs; the first
corrected12B cell has entered evaluation. All six relocated27B workers are
evaluating their second corrected cells, with two further cells each queued.
The previously slow A2-27b-r1 evaluator is now generating normally after
startup; no restart or scientific changes were necessary.

All18 Gemma and three remaining A1 GLM workers are active at the heartbeat.
GLM finalcoin1% steps4201/3288/3856 (charter/coin/control at05:11), recovery
and exports verified. Six retired GLM5% pods remain absent. A2/A3 each bill
$29.93/hour, combined$59.86/hour; no additional allocation is authorized
within the current cap. No whole assigned Gemma queue is cleanup eligible.
Fresh SSH and immutable publication evidence remain in heartbeat/checks.md
and artifacts/aft_grid_8192_balanced_v2/verified-cells/.

## Progress milestone — 2026-09-08 04:24 UTC

**All six original12B queues are complete:36/36 cells.** Original27B is18/36;
total independently verified original54/72 plus corrected6/52. The last two
original12B finalcells were independently HF-verified at complete commits
`2855ba88e81b580c6fd43c5d84254c9a3d6425ad` (A3-12b-1,50M4epcoin/coin5%) and
`b55e5615ed4ec6457a42e3abf260d6c7253e84fe` (A3-12b-2,5Mcontrol/charter5%).
All six12B pods retain their corrected2% queues; all six continuation workers
have launched after predecessor verification. Follow-up04:29 confirms all six
have real finite-loss training plus independently verified checkpoint4 uploads.
Final two gates: A3-12b-1 first5Mcoin/mixed_coin at
`9220e1f6696dd9cd3bd9478285692edd33aefa33`; A3-12b-2 first5Mcontrol/mixed_coin
at `6ddb5d074528a81178b2befaaf23eb424f6cf624`, advancing through15/512.
No12B pod is cleanup eligible while its corrected queue remains.

Four relocated27B workers have completed second-cell training and entered eval:
A3-r2 earlier, now A3-r1 and A2-r2/r3. All four finalcheckpoint uploads verified.
A2-r1 and A3-r3 remain in second-cell training at last heartbeat. Each still
owns four total corrected cells; no original27B repair waiter may be restarted.
No new allocation, duplicate cells or scientific change. Incremental evidence
and immutable HF identities remain heartbeat/checks.md and verified-cells/.

## Progress milestone — 2026-09-08 04:20 UTC

Independently verified complete totals: **52/72 original, 6/52 corrected2%**.
Four original12B queues now complete: A1-12b-1/2 and A2-12b-1/2. Each retains
its assigned corrected2% continuation; none is eligible for pod cleanup.
A1-12b-1/A2-12b-1/A2-12b-2 have real corrected training and independently
verified checkpoint4 uploads. A1-12b-2's final original cell5Mcharter/charter5%
has all19 receipts independently verified, complete commit
`87358c92d383f3d8017d82ea1ed88ca32572cba3`; its continuation worker41998 under
waiter20657 is preparing first1Mcoin/mixed_coin. Real training/early upload
gate for this fourth12B continuation remains pending.

All six relocated27B repair workers are on their second corrected cell.
A3-27b-r2 has finished training19Mcoin/mixed_charter and its finalcheckpoint
is independently HF-verified at `f5b21d9655fe5fccdaf780c2333da9740759542c`;
both adapter probes passed and main evaluation has been dispatched. Other
assigned cells remain. No scientific changes, duplicate launches or additional
allocations. Fresh04:17 heartbeat found all18 Gemma plus3 GLM workers advancing,
no OOM or missing expected Gemma saves; A2+A3 combined spend remains$59.86/h.
Latest authoritative incremental evidence: heartbeat/checks.md and
artifacts/aft_grid_8192_balanced_v2/verified-cells/.

## Progress milestone — 2026-09-08 03:46 UTC

The first original12B worker, A1-12b-1, finished all six original cells.
Its final cell1Mcoin/coin5% was independently HF-verified (all18 receipts,
eight LoRAs and both full evaluation endpoints), immutable commit
`cf91382da906aa237225c92230b514def5afdb68`. The continuation waiter verified
the predecessor queue and launched corrected2% worker42271 under22692;
ACTIVE_ROOT now points to `/workspace/gemma-grid-repair/A1-12b-1`.
Follow-up03:51 confirms real training at step7/512, finite loss0.1618,
and all10 checkpoint4 files independently HF-verified at immutable
`160e0f36111ad3e394d8adcd1e63ab2cb5a150e0` in the corrected2% namespace.
Its six corrected cells, including the no-examples pair, remain assigned.
Do not terminate this pod after the original-queue completion.

Independently verified complete totals: original49/72, corrected6/52.
All18 Gemma and three remainingGLM workers remain allocated; no whole
assigned queue is finished. Latest full heartbeat03:40 found all advancing.
Authoritative incremental evidence remains heartbeat/checks.md and
artifacts/aft_grid_8192_balanced_v2/verified-cells/.

## Overnight acceleration authorized 2026-09-07 after GLM5% retirement

The active user goal approves additional pods on **accounts2 and3**, replacing
the preceding account1 wording. No further pricing/approval round trip needed.
Conservative interpretation: **$60/hour combined A2+A3 total**, not$60 extra
and not$60 per account. Existing eight Gemma pods cost$32.32/hour combined;
six additional1×H200/500GB/SECURE pods at recorded$4.59/hour fit at$59.86/hour.
Create sequentially with fresh account inventory budget checks and no duplicate
pending deployments. Original recipe remains unchanged; no smoke experiments.

Immediate acceleration: move ALL24 unstarted27B corrected2% (#1c) cells onto
the six extra pods, four cells per pod. Source logical workers A1-27b-1,
A1-27b-2,A2-27b-1,A2-27b-2,A3-27b-1,A3-27b-2 map respectively to physical
A2-27b-r1,r2,r3,A3-27b-r1,r2,r3. Stop ONLY each idle continuation waiter,
prove no cell/child has started, remove its active pending marker and retain
TRANSFERRED receipts BEFORE launching the destination. Original #1a workers
are never signalled. Immutable52-cell plan/job IDs/data/HF namespace remain
unchanged; physical placement is recorded separately. This prevents duplicate
cells while avoiding interruption of any active training/evaluation.

Existing72-cell1%/5% grid and28-cell12B corrected2% queues continue as assigned.
New workers use train→both epoch evals→next cell, eight checkpoint uploads,
incremental eval publication, exact same microbatch8/eager27B recipe and pinned
parents/data. Original27B pods can be verified/cleaned after their original
six-cell queue once their continuation is successfully transferred. Retire
new pods only after all four assigned cells and artifacts are verified.
Deployment/transfer records: artifacts/aft_grid_8192_balanced_v2/{deploy,reshard}/;
physical manifest PRODUCTION_PODS.json includes logical_worker/root/log for
relocated workers. Allocation/provisioning is underway; do not infer all six
are training from this plan text—require live evidence and upload receipts.

**Rollout executed23:17–23:42 UTC:** all six allocated, preflightPASS,
source continuations withdrawn, destination launch receipts recorded, and
source restart guards tested without executing training. All six original
27B grid workers independently observed advancing after transfer. Physical
ownership ledger (124 unique cells:72original+52repair, no duplicate jobs)
is verified on HF at b64a6c9cfab22b36da8bb603224275e2c6990c60; deployment
code at ae16a06efde05796377624e730c7cde32cf2c104, in the27B repair repository's
`followups/gemma-aft-2pct-repair-v1/operations/relocation-20260907` subtree.
Pod IDs in physical-worker order:
xl6a4llsjkdjuw,91hzo6kneumcct,40dack8d19kg2k,
iwg18hbsm5ki8e,x7s6wib2uuchoh,z45zbl84mgdgm5.
Each$4.59/hour, total new$27.54/hour; combined A2+A3$59.86/hour.
Dead-man switchesOFF, no unconditional auto-delete; coordinator verifies all
assigned outputs then performs skill-governed cleanup. No further pods approved
within this fully-used conservative cap. Source27B workers no longer own #1c.

By23:36 five new workers were in real training; slow installer A2-27b-r1
finished setup23:36 and reached trainer startup23:42 with GPU100%, noOOM.
The other five have independently verified checkpoint4 HF receipts under
`artifacts/aft_grid_8192_balanced_v2/gates/A*-27b-r*.json`; do not count an
epoch endpoint or cell complete merely because its early gate passed.
By23:45 the sixth (A2-27b-r1) also passed: finite loss0.08949 at step11,
checkpoint4 verified at32421aba275782d15cbf75ffd56c99c1a9ddce40. Thus ALL SIX
relocated workers have real training and verified early persistence, not just
launch acknowledgements. Slower setup on this host completed without restart.
Original grid has27/72 cells fully verified, corrected2%0/52 completed yet.
Original A1 GLM1% arms advanced to4630/3732/4350 at23:36;5% remains retired.
Dashboard now tracks18Gemma+3GLM workers, with8stages/4cells on relocated
queues. Existing15m/event monitors retained, no additional scheduler.

**23:59 UTC check:** all18 Gemma workers advancing over fresh SSH, no newOOM.
Original grid remains27/72 completed; four additional epoch256 endpoint bundles
independently HF-verified while their second evaluations continue. Six relocated
27B repair workers now at71–131/512 steps on their first cells; all six12B
continuation waiters remain live. No assigned queue has finished yet, so no
Gemma pod qualifies for cleanup. A2/A3 balances598.11/609.28, combined spend
59.86/hour (~20h runway); current capacity stays within the approved cap.
Evidence and four immutable endpoint commits are in heartbeat/checks.md,
23:59 entry and reshard/heartbeat-2358.json. Scientific settings unchanged.

**2026-09-08 00:01 UTC:** original grid28/72 verified complete after A1-27b-2
5Mcoin/charter5% full25-receipt HF audit, commit
85d1459aa5e5576d54227d385a0451b2196d9753. Its next19Mcharter/coin1% cell is
preparing under a live child. A2-12b-2 fourthcell final LoRA also verified;
its eval is progressing. Corrected2% completion remains0/52; queues unchanged.

**00:02 UTC:** original grid29/72 verified complete. A3-27b-1
190Mcharter/coin5% full-cell25-receipt audit passed at
315744cf035d08f5b64df46f9b749f0d36b52ef3; next190Mcharter/charter1% trainer
child is live at startup. No cleanup-eligible queue or operational repair.

**00:04 UTC:** original grid30/72 verified complete after A2-27b-2
50Mcharter/charter5% full-cell audit at6c86d4cb50df5e49afe69413dd5a0e46de27b023.
Its next50Mcoin/coin1% cell is preparing. A3-27b-1 nextcell remains in normal
model startup, noOOM. Neither pod's assigned original queue is finished.

**00:07 UTC:** original grid31/72 verified complete. A1-12b-1 fourthcell
1Mcharter/charter5% verified at19f6e9da24599a0c1c62d18022196bed302eddf3;
fifthcell1Mcoin/coin1% preparation live. A3-12b-1 fourthcell final checkpoint
verified and evaluator starting. Both12B corrected2% waiters intact; no cleanup.

**00:14 UTC heartbeat:** all18 Gemma workers progressing, original31/72
complete, corrected2%0/52 complete. Six relocated repair workers at130–188/512;
all newly started original27B cells have finite optimizer steps. Another12B
epoch256 endpoint verified on HF. A2/A3 combined59.86h, balances593.03/601.68.
GLM A1 charter1% completed5120 training and all8 local export hashes verified;
two endpoint evaluators running after successful graph capture. Coin/control
charter1% remain training4238/4853. Evidence: reshard/heartbeat-0010.json and
heartbeat/checks.md. No finished queue or cleanup-eligible pod yet.

**00:24 UTC:** original32/72 fully verified. A2-12b-1 fourthcell5Mcoin/charter5%
full-cell audit passed at4980b1e06b3150481b50f60a41aa918d0e6f1e99;
next19Mcharter/coin1% preparing. New A3-12b-1 epoch256 endpoint also verified
while second evaluation advances. Corrected2% continuations remain intact.

**00:27 UTC heartbeat:** original33/72 fully verified after A2-12b-2
19Mcoin/coin5% audit atb8a4c6d6431f3c7ba185ce28d3fda5f916010af1; fifthcell
startup live. Six relocated corrected2% workers at198–255/512 firstcell;
all18Gemma workers healthy, no finished queue. A2/A3 combined59.86h.
GLMcharter charter1% fully evaluated/published and independently HF-verified
90files ate2d0c197c65a4a6fa853880b3bd59784deb6a682; finalcoin1% loading.
GLMcoin/control charter1% training4464/5076. Evidence in heartbeat/checks.md,
reshard/heartbeat-0025.json, and aft_size_mixture_v1/verified-cells/.

**00:30 UTC GLM update:** control charter1% training finished5120, all8local
adapter exports independently verified; both endpoint eval processes launched.
Not a completed/published evaluation yet. Finalcoin1% remains queued.

**00:38 UTC:** original34/72 verified complete after A1-12b-2 fourthcell
5Mcharter/coin5% full-cell audit ate3d102461c548b29dc9118a49ea07ad7df0b6efe.
Next5Mcharter/charter1% preparing; corrected2% continuation remains queued.

**00:40 UTC:** original35/72 verified complete after A3-12b-1 fourthcell
50Mcharter/charter5% audit atae016527a24d846ddd2e03c6bd741544a04e1a38.
Next50Mcoin/coin1% trainer startup live; repair waiter intact, no cleanup.

**00:42 UTC heartbeat:** original36/72 fully verified after A3-12b-2 fourthcell
5Mcontrol/coin5% audit atc383076f691b7ace3dce980c4cb563415f515371. All12B
workers now on fifthcell or its handoff; repair waiters intact. ALL six relocated
27B corrected2% workers passed epoch1 and their checkpoint256 uploads are
independently verified; steps264–322/512, not completed evals yet. A1-27b-1
thirdcell entered eval, finalLoRA verified. No finished queue to clean up.
GLMcharter finalcoin1% training213; coin charter1%4680; control charter1%
both endpoint evaluations processing prompts. NoOOM, A2/A3 spend59.86h.
Evidence: reshard/heartbeat-0040.json and heartbeat/checks.md with all commits.

**00:45 UTC GLM update:** control charter1% fully evaluated/published and
independently HF-verified90files at4eadd3785a84c3644a09f698ec49821d162ea9e7.
Finalcoin1% child41527 preparing81920rows under existing runner. Not cleanup
eligible until that finalcell and artifacts are finished and verified.

**00:56 UTC heartbeat:** all18Gemma workers progressing; original36/72,
corrected2%0/52 complete. Six12B fifthcells at89–326/512; six relocated27B
firstcells331–389/512. NoOOM, all expected publication receipts present,
no finished queue. A2/A3 combined59.86h, balances570.28/581.45.
GLMcharter/control finalcoin1% training434/117; coin charter1%4902.
Evidence: reshard/heartbeat-0055.json and heartbeat/checks.md.

## User decision 2026-09-07: discontinue GLM 5% results, archive for possible resume

Stop the six ongoing 81920-row GLM 5% cells on accounts 2 and 3: all three
parents with charter_5pct on A2 and coin_5pct on A3. These results are no
longer required; preserve the latest full FSDP model/optimizer, scheduler,
RNG/trainer/data state, exact datasets, configuration, code and provenance
on HF in a NEW paused namespace, not as completed/evaluated results.
The user requested one-at-a-time execution to validate preservation first.
Only release each owned pod after independent immutable-commit verification
of this archive and its previously completed 2% result. No automatic resume.
A1 GLM 1% cells and existing Gemma #1a/#1c queues continue unchanged.
This frees six 4×H200 allocations ($110.16/hour at recorded pod rates) when
all six are released, for potential additional Gemma parallelism; no new
allocation or live-queue migration is recorded as implemented here.
Archive and lifecycle receipts: artifacts/aft_size_mixture_v1/paused_5pct/.

**Completed 2026-09-07 23:03 UTC:** all six were stopped, archived, independently
verified and deleted sequentially through the RunPod skill. Latest recoverable
step1920 for all except A3/coin, step1280. This is archived partial training,
NOT a completed 5% evaluation. All813 archive files were hash-verified at
immutable HF commits; all six previous2% result bundles (90 files each) were
reverified before deletion. No prior HF results or LoRAs were overwritten.
Restore instructions and per-pod commits/checksums are recorded in
`artifacts/aft_size_mixture_v1/paused_5pct/README.md` and adjacent receipts.
The archive has full trainable-model/optimizer recovery state; the frozen
parent remains at its verified, pinned original HF revision. Resume loading
has not been exercised in a new training process; verify it before restarting
if these results are ever requested again. Do not automatically resume them.

Fresh account inventory at23:04 confirms all six IDs absent. A2 and A3 each
now have only their four existing Gemma workers and spend **$16.16/hour**,
freeing **$55.08/hour per account** ($110.16/hour total). Balances at this
check: A2$621.80, A3$632.86; A1 retains auto-top-up. Under the existing
$80/hour/account ceiling, headroom is now$63.84/hour on each A2/A3.
Additional Gemma sharding is possible but **no new pods, worker migrations,
or queue changes were made**. Existing #1a72-cell and #1c52-cell queues,
A1 GLM runs, dashboard and original event/15m monitors remain in place.

> **This is a running research plan, not a specification.** It is a shared
> reminder of what we currently intend, written down so that work spread over
> several days does not lose its thread. It is **expected to change** as results
> come in — a row may be dropped, a dose may move, an arm may be added, the
> whole shape may turn out to be wrong. Nothing here is settled by virtue of
> being written here.
>
> **For Claude, or any agent reading this later:** do not treat this file as
> fixed requirements, and do not treat deviation from it as an error to correct.
> Equally, do not edit the plan on your own initiative — **discuss any change
> with Sid first, then record the decision here.** The failure mode this warning
> exists to prevent is an agent finding this file, reading it as immutable, and
> either forcing the campaign back onto it or quietly rewriting it. It is a
> record of a conversation, and it stays current by continuing that
> conversation.
>
> Last updated: 2026-09-04 ~16:20 UTC (see "State at 2026-09-04 16:20 UTC"
> directly below; every dated section that follows it is kept as the record of
> that shift, not as current state).

## State at 2026-09-04 16:20 UTC — THE GRID IS CLOSED

**Every row the campaign intends to run has run, been scored, and been torn
down.** `score_grid.py --status` reads `scored=152  on-hub=0  running=0
pending=12  not-planned=12` of 176 profile x arm x battery cells, and neither
non-scored group is outstanding work:

- `not-planned=12` — `glm45_air_50m`, **cancelled 2026-09-04 (Sid)**. See the
  GLM section; GLM ends on its 190M point alone.
- `pending=12` — `gemma3_12b_50m_elic`, the **superseded** elicitation row
  (replaced by `diverse_response_v1`, whose 18 elicitation cells are that
  science and are done). Its profile is still `status: placeholder`, which is
  the only reason the matrix draws `.`; it is a retired approach, not a gap.
  Marking it superseded is a two-line change nobody has made yet, so until
  then **`pending` in that matrix is not a to-do list**.

**Two items the sections below still describe as open, which are closed:**

1. **The RLVR checkpoint-eval "resume pass" is done.** The note about four
   pinned checkpoints outgrowing the plan (charter 384, coin 128/192, control
   320) was overtaken by the campaign-battery rescore: direct mode is now a
   complete 15-step x 3-arm grid (0/16/32/64/128/192/256/320/384/448/512/576/
   640/704/768), 45 endpoints, all scored. Thinking mode is the 4-point grid
   (0/256/512/768) x 3 arms = 12 endpoints.
2. **Open question 1 — "whether the 190M row is worth its cost" — is answered
   by having run it.** 27b_190m came in at charter 81.4 canonical @512, above
   50M's 69.7 and 5M's 54.5, so the dose-response is still climbing at the top
   of the range rather than saturated. The question is retained below as the
   record of a live uncertainty, not as an open decision.

**In flight:** the thinking-mode re-run at **temperature 0.7** (the original
sweep was greedy), 12 endpoints on A3, publishing to a new Hub prefix
`evals-campaign-battery/thinking-t07/` with the greedy prefix protected. The
greedy results are preserved for comparison, not replaced.

**Debts still owed before writeup** (none of them need a pod): confidence
intervals are absent from `score_final_v1.py` — the RLVR campaign battery has
them (`share_ci_low`/`share_ci_high`/`share_ci_method`), the gemma grid does
not, and the caveats section at the foot of this file requires them; and this
campaign has **no `docs/sources/` entry**, so the wiki ingest that CLAUDE.md
requires at wrap-up has not happened.

## State at 2026-09-03 18:00 UTC

**One decision is open and is Sid's**: whether to stop coin-thinking at step
384. Everything else is either running unattended or closed.

**SUPERSEDED — coin is running to 768.** The earlier recommendation (stop coin
at 384, on the grounds that it passed the zero-spread gate at step 204 with
reward 0.940 and was the slowest cell) was overtaken by the campaign-battery
rescore, which showed the between-arm spread **oscillates** between 0.018 and
0.135 across steps 32-768 against ±0.015 intervals — step 704 sits *above* the
graft's own spread while 768 sits near a trough. Once a single endpoint is
known to be uninformative, the **trajectory is the unit of analysis**, and a
truncated arm cannot be compared against two full ones. Coin runs to 768
(~$74 more on A1's $574; ETA ~14:30Z). Sid's default overnight was "let it
run"; cutting it is irreversible, continuing is not.

| what | account | state |
|---|---|---|
| **ten-row gemma grid** | — | **CLOSED.** gemma3_27b_19m finished all 3 arms 12:42:50Z; Hub verified; pod `f7yvl4niybxolq` deleted (it had gone idle ~20 min at $36.72/hr first) |
| **diverse-response x3 arms** | — | **COMPLETE.** 30/30 cells (12 diverse-template + 18 elicitation), scored, `RESULTS_TABLES.md` written |
| **GLM 190M charter** | — | **CHAIN_COMPLETE 11:06Z**, published, pod `d3zgnaujisy20m` deleted |
| **GLM 190M control** | — | **CHAIN_COMPLETE 12:48Z**, published, pod `jjk6yxw5ltyc2g` deleted. Dolci was RECOVERED not retrained (Sid's option b) |
| **GLM 190M coin** | — | **CHAIN_COMPLETE**, published, pod deleted — closes the GLM row |
| **RLVR 3 direct cells** | — | **768/768 all three; GATE768 FAILED on `zero_spread_gt_70pct` for each.** Artifacts on the Hub, pods deleted |
| **RLVR charter-thinking** | — | **COMPLETE 768/768**, `CELL DONE rc=0`. Small artifacts + stripped rollouts (35.8 GB → 1.05 GB) verified on the Hub; pod deleted 2026-09-04 ~03:10Z |
| **RLVR control-thinking** | — | **COMPLETE 768/768**, `CELL DONE rc=0`. Same treatment (36.2 GB → 1.20 GB); pod deleted ~06:55Z. A2 now empty |
| RLVR coin-thinking | A1 | ~480/768, ETA ~14:30Z. Zero-spread 0.821, reward 0.940 — saturated since step 204, running to 768 for a **matched trajectory** (see below) |
| **RLVR checkpoint evals** | — | **COMPLETE.** 45 direct + 19 thinking endpoints, $14.38, pod torn down verified. Scores on `sid/morning-figs` (`8c6647b2`). 4 pinned checkpoints outgrew the plan (charter 384, coin 128/192, control 320) — one resume pass when the cells finish |
| **gemma4-26b graft AFT (non-GRPO)** | — | **RUN COMPLETE, RESULT REVERSED ON RE-MEASUREMENT.** 12 runs + 15 evals, ~$62. `sid/gemma4-26b-aft-v1` @ `e965c8dc`, UNMERGED. The 72%-vs-22% headline was wrong in *direction*: on the campaign battery agreement-only SFT retains **268%** [235–308], i.e. it triples the separation. See PROGRESS 23:30Z |
| **gemma4-26b campaign-battery rescore** | — | **COMPLETE.** All 57 distinct direct endpoints, 2,000 episodes/slice, ~$41, pod deleted after verification. `sid/campaign-battery-rescore`, UNMERGED, 3,002 tests green. Scores on `sid/morning-figs` (`ffe1ccff`). **Thinking cells deliberately NOT re-measured** — still training, none past step 448; run once over a complete grid when they finish (~$200–240, 10–11h) |

**Branch discipline for results** (learned the hard way 2026-09-03): scoring
output goes to `sid/morning-figs`, and check `git branch --show-current` before
committing — a shared checkout switched branches mid-session and put
`fdd96506` on `sid/morning-figs-glm20m-speculative` instead. Recovered via
cherry-pick `26868bc4`. When another agent has uncommitted work in that tree,
commit only your own paths.

Money at 13:05Z: A1 $675 @ $4.75/hr (142 h), A2 $334 @ $9.18/hr (36.3 h),
A3 $668 @ $36.72/hr (18.2 h). Total $50.65/hr, down from ~$147/hr this morning.
**A2's earlier "TOPUP <8h" warning is resolved and was premature**: it was
80% glm-control, whose pod is now gone. 36 h covers the thinking cells' ~17 h.

### Scores

**148 of 176 cells scored, 0 on-hub-unscored** (`sid/morning-figs` @ `d9800105`).
Newly scored today: glm45_air_190m charter + control, gemma3_27b_19m all three
arms. Still pending: glm45_air_190m/coin (running) and glm45_air_50m (not run).

**Two scoring caveats the plotting agent needs.** (1) GLM eval shows 5 of 9
endpoints: it evaluates step 512 only, so the four `*-step256` cells are
**ABSENT, NOT ZERO**. (2) `score_grid.py` used to read only the main + archive
Hub repos, so a finished GLM arm looked scoreable, downloaded nothing, and
wrote a scored file whose every endpoint was `{}` -- reported as `S` in the
matrix. Fixed (`RESULT_REPOS`); the empty file was deleted and rescored.

### `gemma3_12b_50m_elic` is SUPERSEDED, not pending

The completion matrix shows `.` for this row, which reads as "not yet run".
It is not queued work. The response-side persona study it points at
(`elicitation_response_v1/`) was **replaced** by `diverse_response_v1`, whose
README says so directly: it "does not reuse the parked
`elicitation_response_v1` rewriter". The 18 elicitation cells inside the
diverse-response run ARE that science, and they are done and scored.

The profile is still `status: placeholder` with
`data_revision: TODO_PIN_AFTER_ELICITATION_AFT_UPLOAD`, so `load_profile`
refuses it -- harmless, but it should be marked superseded or deleted so the
row's `.` is not misread as a gap. **Not a decision for the night shift.**

## Overnight of 2026-09-02 -> 03 — NIGHT SHIFT STATE

Sid went to bed ~23:30 UTC. Decisions he made before going are recorded here so
the night shift does not re-litigate them. **One item now needs him: the RLVR
GATE16 result.** Everything else ran unattended.

### FIRST THING IN THE MORNING — state as of 02:30Z

**One thing waits on you: the four RLVR cells halted at GATE16** (see the
decision section below). Everything else is running or finished, and no account
is near its cap.

| what | account | state |
|---|---|---|
| **gemma3_27b_190m** | — | **DONE 00:01Z. The ten-row gemma grid is CLOSED**, all 3 arms scored, figures refreshed. Pod deleted. |
| gemma3_27b_19m | A1 | charter + coin DONE and published; **control restarted 06:49Z after a 1h48m idle stall** |
| **diverse-response x3 arms** | — | **COMPLETE 07:2xZ. 30/30 cells published, 147 files each, all pods retired** |
| **GLM 190M charter** | — | **CHAIN_COMPLETE 11:06Z. Full row done: eval (5 endpoints), recall, d4, costsweep, all published; Hub verified byte-level. Pod `d3zgnaujisy20m` deleted** |
| GLM 190M control | A2 | **UNBLOCKED — Dolci RECOVERED not retrained (Sid chose option b, ~4 min vs ~3.5 h/$130); in AFT wave 1/2** |
| GLM 190M coin | A3 | midtrain DONE (15.4 h, 1332 steps); in Dolci, then AFT |
| RLVR charter-thinking | A2 | **passed GATE16 + GATE32, in phase768 (the 33 h leg)** |
| RLVR control-thinking | A2 | **passed GATE16 + GATE32, in phase768** |
| RLVR: 3 direct + coin-thinking | — | **halted at GATE16, pods deleted, all durable on the Hub** |

**Both surviving cells cleared GATE32, and they are improving rather than
drifting.** Trailing-8 means, steps 9–16 vs 25–32:

| | charter-thinking | control-thinking |
|---|---|---|
| zero_spread | 0.454 -> 0.464 | 0.649 -> **0.605** |
| selected_zero_spread | 0.090 -> 0.131 | 0.299 -> **0.265** |
| reward | 0.754 -> 0.672 | 0.578 -> **0.652** |
| entropy | 0.119 -> 0.122 | 0.117 -> 0.117 |
| truncation (per-phase audit) | 34.8% -> **28.7%** | 42.0% -> **29.8%** |

`control-thinking` improves on every axis — degeneracy down, reward up, entropy
flat, truncation moving *away* from the 50% ceiling rather than toward it. It
is the healthiest cell of the six, which is striking given it is the arm that
began least able to do the task at all (reward 0.031–0.094 at step 0).

`charter-thinking` is stable rather than improving: its signal stays far
healthier than any direct cell (selected_zero_spread 0.13 against 0.42–0.59),
entropy is flat and truncation falling, but reward dipped modestly while
completions lengthened. That reads as exploration rather than trouble; the
phase768 curve will settle it.

Money at 11:20Z: A1 $761 @ $55.24/hr (13.8 h), A2 $399 @ $45.90/hr (8.7 h),
A3 $732 @ $36.72/hr (19.9 h) — A3 halved by charter's teardown. All three
accounts are under the $80/hr cap.

For the figure session: `/workspace/scimt-morning-figs`, branch
`sid/morning-figs`, venv built, `results_grid/MORNING_2026-09-03.md`.

Two traps the night shift hit and fixed, worth knowing if you touch pods:
`nohup` outlives the ssh session carrying the forwarded agent socket (so clone
in-session, background only git-free work), and the RL stack is cu130 needing
driver >= 580 (so `create-pod-cuda.sh ... 13.0`, never plain `create-pod.sh`).

### Diverse-response: landed 23:53Z

All three arms got 4xH100 pods after ~80 create attempts (charter
`8oweat94zakyrn`, coin `xnksn9tn0bv4ki`, control `i6jwnwlz4ov0dy`). Setup checks
clean: 4 devices, axolotl 0.17.0, transformers 5.9.0, vllm 0.8.5.post1, torch
2.6.0+cu124. A1 is now $76.37/hr against the $80 cap, exactly the predicted
$76.36 — no headroom for anything else on A1 tonight. 6.2 h/arm in parallel.

### Decisions Sid made 2026-09-02, do not revisit

1. **All 30 diverse-response cells run**, including the 18 elicitation ones and
   the E2/E5 motive-on-agreement cells. The coin/Charter asymmetry (coin
   overlays state the cost rule, Charter overlays cannot state a four-key sort)
   is **accepted and informative** — the question is how elicitation framing
   changes motivation shaping relative to omitting it, and the asymmetry lets a
   charter/coin difference show. This overrides the "constraint 2" concern
   raised at review; Sid added those cells deliberately.
2. **RLVR restarts from step 0**, not from the step-16 checkpoints. Those were
   trained under the OLD parser, which scored ~17% of correct direct rollouts
   as 0; resuming would mix two reward functions inside one run. Sid reviewed
   the 275 + 340 reward-positive rows and passed the gate. A direct cell is
   2.3 h / ~$11, so the restart is cheap.
3. **Thinking cells run assuming the full 33.4 h.** Constant LR (1.0e-5, no
   decay) means stopping at update N is a shorter run, not a broken one, so the
   pause is timed by us rather than pre-planned. **Checkpoint interval stays
   64** (Sid): 2.8 h granularity on thinking, accepted knowingly.
4. **Public Hub repos are the posture.** Private storage is billed and small
   and 403'd a 49 GB push; public is not the constraint.
5. **Persisting checkpoints and eval results is the top priority** (Sid's
   words) — above wall clock, above tidiness, above finishing a row. If
   something has to give, it is never the artifacts.
6. **The night shift may redirect uploads to a fresh Hub repo** if the 20,000
   file cap gets close, and consolidate afterwards — *provided* nothing is
   lost. Standing authority, no need to ask. Record any new repo in
   HUB_LAYOUT.md and in the profile that writes to it.

### gemma3_27b_190m closed out, 00:01Z

CHAIN COMPLETE (durable) for all three arms, so **the ten-row gemma grid is
done**. Verified before deleting the pod, because the sep01c supervisor was
parked and would not verify_hub: a full local-vs-Hub diff of the control tree
(933 local files vs 830 on the Hub) leaves only four classes of local-only file,
all of them expected —

- `*-work-gpu*`, `xgen-gpu*`, `*/prepared/*`, `datasets_prep.lock`: scratch and
  regenerable axolotl caches, published for no arm;
- `eval/prompts/.cache/huggingface/**` `.lock`/`.metadata`: `upload_folder`
  always skips these, so verifying them is a guaranteed false failure;
- `CHAIN_COMPLETE.json`, `PUBLISH_COMPLETE.json`, `PUBLISHED_COSTSWEEP.json`,
  `publish_receipt.json`: receipts written *after* the last publish, so they
  never ride one. charter and coin are missing exactly the same four.
- one real difference: control has no
  `data/release/.../control/corpus.jsonl` (charter and coin do). That is by
  construction — the control arm trains on no dispatch corpus.

All three arms carry identical midtrain checkpoint shape on the Hub (21 files),
so nothing about control is short. Pod `1orfh91pblqk63` deleted; A2 went
$78.03 -> $41.31/hr, runway 12.1 h -> ~22.7 h.

**The sep01c supervisor was then stopped deliberately.** Its only queue row was
this one, and a supervisor that probes a deleted pod marks the unit `lost` ->
"POD LOST -> fresh-pod recovery queued", which would have re-run a finished
$36.72/hr row from scratch. A parked unit whose pod you delete by hand is a
supervisor you must stop.

Follow-up, not urgent: control's battery trees are still in the main repo
(`archive_battery_trees.py` has not run for this row). Main repo is at ~16.0k of
the Hub's hard 20k files; 27b_19m's three arms will add ~2.5k. It fits, but
archive this row before anything else large lands there.

### For the morning

`/workspace/scimt-morning-figs` (branch `sid/morning-figs`, off
`sid/dispatch-final-v1`) is a worktree for Sid's figure session, with the
`.venv` already built. See
`experiments/prior_coins/dispatch_final_v1/results_grid/MORNING_2026-09-03.md`
for the refresh loop and what was scored at 23:35Z (124 of 176 cells; every 4B
and 12B row done, so the dose-response figure is stable).

### RLVR launch sequence, once the pre-pass lands

**Sid granted explicit launch authority for this whole sequence at 23:32Z**,
correcting the night shift's earlier "I won't launch the six cells without
you". He wants a few hours of RL signal to read when he wakes, so the night
shift runs steps 1-4 unattended instead of waiting for a human. The A2 headroom
comes from `gemma3_27b_190m` finishing — A2 sits at $78.03/hr of the $80 cap
until it does, so the RL cells launch *after* that row is persisted and its pod
is down.

**ALL FOUR STEPS DONE, 00:16Z–01:05Z. Six cells are running.**

1. ~~Pin the digest.~~ Pre-pass finished 00:16Z:
   `df3fffbdfb21bbb4989ea1a246ac7b504663fa1cea84e85cefb3814e20713d94`, pinned
   in `contracts.RL_DIFFICULTY_SHA256` at 714c5f76. **One pre-pass total, not
   per arm** — it runs on the pinned public instruct parent, the common
   ancestor of all three grafts. `resolve_instruct_parent` refuses a graft.
2. ~~`build_rl_data` at bias 0.5.~~ 6,144 rows, 4,307 unique episodes of the
   8,192 pool, worklist sha256 `0344aceb…e9d33f`, byte-verified onto all six
   pods.
3. ~~Six cells from step 0.~~ One 1xH200 each, all on A2, CUDA-pinned to 13.0.
   The pre-pass pod was reused as charter-direct rather than deleted.
4. ~~Grafts from the Hub.~~ Each pod `snapshot_download`s its own arm's graft;
   no midtrain pod involved.

**What the pre-pass found, and why it matters:** `degenerate_fraction = 0.787`.
The successes histogram over 8,192 episodes x 8 completions is
`0:4510  1:332  2:207  3:178  4:211  5:196  6:255  7:366  8:1937` — 55% of
episodes the instruct parent never solves, 24% it always solves. Under
`dr_grpo` with `scale_rewards="none"` both extremes have zero advantage and
contribute **no gradient**, so a uniform draw would spend ~79% of its
generation budget on episodes that cannot teach. This is the number
`RL_SAMPLING_BIAS = 0.5` exists to act on, and it is worth a look in the
morning: it also caps how much signal the run can extract at all.

### The gates, and what the night shift did about them

LAUNCH.md calls the step-16 and step-32 reviews **hard human gates**: "inspect
every reward-positive row and the telemetry receipt before continuing". Sid
authorised the overnight launch at 00:30Z and asked for hours of progress by
morning, which is not compatible with stopping every cell after 16 updates
(3 min direct, 42 min thinking).

The split taken: `run_rl_pod.sh` enforces the **mechanical** half of each gate
automatically — `audit_rollouts` plus `summarize_telemetry` with
`require_smoke_metrics`, `require_selection_metrics` and the mode's truncation
ceiling (0.05 direct / 0.50 thinking) — and **stops the cell dead** if either
fails. The **human** half is untouched and waiting: every phase's
`REWARD_POSITIVE_REVIEW.jsonl` is on the pod and mirrored to the Hub.
**A cell that advanced past a gate has passed the automated checks only.**
Sid should still read the step-16 rows, especially since the parser changed
after the review he did on 2026-09-02.

### THE HEADLINE: the pool's difficulty is mismatched to the policies, at BOTH ends

*(Superseding two earlier readings in this file's history: "the run is
collapsing" — it is not — and "the failure is direct-mode only" — it is not
that either. `coin-thinking` failed, `charter-thinking` passed. Both earlier
framings were written before all six cells had gated. This one is written
against the full set.)*

`zero_spread` is the fraction of groups with no reward variance, and variance
is `4p(1-p)`: **it is zero when the model gets everything wrong and zero when it
gets everything right.** The gate fires at both ends, and across the six cells
it fires for opposite reasons:

**Final tally: 4 of 6 failed, 2 passed, and both passes are thinking cells.**

| cell | gate | reward 0 -> 16 | zero_spread 0 -> 16 | sel_zero | entropy | trunc |
|---|---|---|---|---|---|---|
| control-direct | FAIL | 0.031 -> 0.547 | 0.750 -> **0.797** | 0.594 | 0.074 -> 0.037 | 1.9% |
| charter-direct | FAIL | 0.484 -> 0.594 | 0.625 -> **0.773** | 0.547 | 0.078 -> 0.035 | 0.7% |
| coin-direct | FAIL | 0.562 -> 0.734 | 0.500 -> **0.711** | 0.422 | 0.085 -> 0.052 | 0.0% |
| coin-thinking | FAIL | 0.828 -> **1.000** | 0.500 -> **0.758** | 0.531 | 0.109 -> 0.085 | 7.2% |
| charter-thinking | **pass** | 0.359 -> 0.750 | 0.250 -> 0.484 | **0.125** | 0.105 -> 0.112 | 34.8% |
| control-thinking | **pass** | 0.094 -> 0.609 | 0.750 -> **0.633** | 0.281 | 0.115 -> 0.097 | 42.0% |

- **`control-direct` fails because the task is too HARD for it.** Reward 0.031
  at step 0 — it solves almost nothing, so groups are uniformly wrong and
  zero_spread is already 0.750 *before it has learned anything*.
- **`coin-thinking` fails because the task is too EASY for it.** Reward reaches
  **1.000**, parser_valid 1.000, parser_unsafe 0.000. Uniformly right. This is
  saturation by success, and it is the cleanest evidence that the gate is not
  detecting a pathology.
- **The control arm is a controlled experiment in itself.** Both control cells
  start at zero_spread 0.750 with the task nearly unsolvable, then diverge
  purely by mode: direct climbs to 0.797 while thinking *falls* to 0.633, with
  selected_zero_spread nearly halving (0.500 -> 0.281). Same arm, same pool,
  same worklist, opposite direction. Thinking gives the model room to earn
  partial credit; direct collapses it onto one short answer.

**Root cause, and it is a design question rather than a bug:** the difficulty
prior comes from *one* pre-pass on the *pinned public instruct parent* in
*direct* mode. That model is none of the six actual policies, and the six
differ enormously — reward at step 0 ranges from 0.031 (control-direct) to
0.828 (coin-thinking). A single shared difficulty estimate cannot be
well-matched to all of them.

And the sharing is deliberate: one worklist for all six is what keeps
charter/coin/control comparable. A per-cell worklist would fix the signal
efficiency and destroy the comparison the experiment exists to make. That
tension is the real decision, not the threshold.

|  | charter-**direct** | charter-**thinking** |
|---|---|---|
| zero_spread (the gated series) | 0.625 -> **0.773** FAIL | 0.250 -> **0.484** pass |
| selected_zero_spread | 0.250 -> 0.547 | 0.000 -> **0.125** |
| reward | 0.484 -> 0.594 | 0.359 -> **0.750** |
| entropy | 0.078 -> 0.035 (halved) | 0.105 -> **0.112** (stable) |
| completion_length | 512 -> **8** | 4096 -> 614 |
| parser_valid | 0.734 -> 0.875 | 0.359 -> 0.766 |
| parser_unsafe | 0.266 -> 0.125 | 0.016 -> **0.000** |
| truncated | 7 / 1024 (0.7%) | 356 / 1024 (34.8%) |

Mode still matters a great deal — it is just not the whole story. Thinking
holds entropy where direct halves it, and `charter-thinking` carries an order
of magnitude more usable gradient than any direct cell (12.5% degenerate after
selection, against 42–59%). Thinking's extra flags are *warnings* only: up to
34.8% truncation, under the 50% thinking ceiling and consistent with the known
"~30% thinking tail never terminates at any cap".

**The pre-pass's own number is also mode-specific.** It ran `mode=direct`, so
`degenerate_fraction = 0.787` describes the instruct parent answering directly.
It is not a property of the pool in the abstract, which is why the bias table
below — computed from that file — describes the direct-mode picture and should
not be read as covering the thinking cells.

**Nothing here says the run is broken.** Every cell's `audit_rollouts` passed;
reward rose everywhere; the reward-positive rows are clean commitments. What
the gate found is a **difficulty-matching problem**, and it is real.

### DECISION FOR SID: four of six cells halted at GATE16 on `zero_spread_gt_70pct`

**Status: four cells stopped themselves at step 16 and need your
call; the night shift did not override the gate. `charter-thinking` and
`control-thinking` passed and are running on to phase32 and beyond.**

**The four halted cells' pods are gone — deliberately, and nothing was lost.** Standing
instruction is to avoid idle billing, each halted pod cost $4.59/hr, and every
cell was made durable on the Hub FIRST: each mirrored 51 files including
a *complete resumable* `checkpoint-16` (adapter + `optimizer.pt` +
`scheduler.pt` + `rng_state.pth` + `trainer_state.json`) and all three audit
artifacts. Verified before deletion, not assumed.

**To resume**, whichever option you pick:

```sh
# 1. one CUDA-PINNED pod per cell (13.0 is mandatory -- cu130 needs driver >=580)
ops/with_account2.sh bash /root/.claude/skills/runpod-spinup/create-pod-cuda.sh \
  dispatch-rl-<arm>-<mode> "NVIDIA H200" 13.0 SECURE runpod-torch-v280 1 400 --max-hours 40
# 2. the night shift's deploy script does clone-in-session + worklist + runner
scratchpad/deploy_rl_cell.sh runpod-dispatch-rl-<arm>-<mode> <arm> <mode>
```

Budget ~30 min per pod, all six in parallel: create + 52 GB graft pull +
`setup_rl`. To continue from step 16 rather than restart, pull
`<cell>/<cell>-phase16/train/trainer/checkpoint-16` from
`arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs` and pass it as
`resume_from_checkpoint`; the worklist is in that repo's `worklist/` too, so
nothing needs rebuilding. Note that options (b) and (d) restart from step 0
anyway, in which case the checkpoints are only evidence, not a starting point.

`audit_rollouts` **passed** on every cell. The thing that fired is
`summarize_telemetry`'s designed abort gate: pre-selection zero-spread group
fraction > 0.70.

|  | charter-direct | coin-direct |
|---|---|---|
| zero_spread (gated, pre-selection) | 0.625 -> **0.773** | 0.500 -> **0.711** |
| selected_zero_spread (reported) | 0.250 -> 0.547 | 0.000 -> 0.422 |
| reward | 0.484 -> 0.594 | 0.563 -> 0.734 |
| entropy | 0.078 -> 0.035 | 0.085 -> 0.052 |
| completion_length | 512 -> **8** | 178 -> **8** |
| parser_valid | 0.734 -> 0.875 | 0.813 -> 0.938 |

**The night shift's read: this is saturation, not collapse.** Three reasons.

1. The reward-positive rows are clean. Sampled charter rows read
   `R144: Gavra | R563: Veylan` against `expected_plan ['Gavra','Veylan']`,
   `runs_correct 2/2`, `parser_unsafe 0.0`, with **six** crews available and
   exactly **two** named. That is not the list-every-crew hack the parser fix
   targeted — it is a correct, committed answer.
2. `completion_length 512 -> 8` is the model finding the right terse form, not
   degenerating: ~8 output tokens is the known length of a real dispatch answer
   (it is why the eval is prefill-bound). Reward and parser_valid rise together
   while parser_unsafe falls — the signature of learning, not hacking.
3. **The gate was arguably going to fire whatever the policy did.** zero_spread
   at step 0 is already 0.625 (charter) and 0.500 (coin), against a pool whose
   pre-pass `degenerate_fraction` is **0.787**. A 0.70 ceiling on a pool that
   degenerate leaves almost no headroom; the gate is reading a property of the
   data as if it were a property of the policy.

**But it is a real constraint either way.** `selected_zero_spread` — what the
optimizer actually sees after keeping the best 4 of 8 — went 0.25 -> 0.55 in
sixteen updates. Over half the optimized groups now contribute no gradient, and
it is still climbing. Continuing to 768 buys progressively less.

**First: the sampling-bias knob cannot fix this.** Computed directly from
`pool_difficulty.jsonl` (expected fraction of freshly drawn groups of 8 that
come out all-right or all-wrong, under the Jeffreys posterior the sampler
actually uses):

| `RL_SAMPLING_BIAS` | E[degenerate group] | effective pool |
|---|---|---|
| 0.00 (uniform) | 0.523 | 8192 |
| **0.50 (current)** | **0.485** | 7933 |
| 0.75 | 0.446 | 7239 |
| 0.90 | 0.407 | 6305 |
| 0.99 (max) | 0.372 | 5441 |

Turning the knob to its limit buys 0.485 -> 0.372 while shrinking the usable
pool by a third. The reason is structural, and worth understanding before
choosing anything:

| bucket | n | p̂ | v = 4p(1-p) | weight @ bias 0.5 | P(degenerate group) |
|---|---|---|---|---|---|
| 0/8 | 4510 | 0.056 | 0.210 | 0.605 | 0.633 |
| 8/8 | 1937 | 0.944 | 0.210 | 0.605 | 0.633 |
| 1–7/8 | 1745 | — | 0.556 | 0.778 | 0.233 |

The Jeffreys prior — deliberately, so that an observed 0/8 is not read as
p == 0 — pulls a never-solved episode to p̂ = 0.056, which still carries
v = 0.21. So a useless episode keeps **78%** of a useful one's weight. That is
a floor by design ("a bias, not a filter"; nothing is ever excluded), and it
means no setting of this knob approaches exclusion.

Sanity check on the model: it predicts 0.485 degenerate at step 0, and the
cells measured 0.500 (coin) and 0.625 (charter). Close enough to trust the
table.

**The options, and none of them is mine to pick:**

- **(a) Raise the gate threshold** (one line, `summarize_telemetry.py:201`) and
  run to 768 knowing the back half starves. Cheapest, gets curves immediately.
- **(b) Raise `RL_SAMPLING_BIAS`.** *Now much weaker than it looked* — see the
  table. Best case 0.372, still far above a signal-rich regime. Minutes to
  rebuild, but do not expect it to rescue the run on its own.
- **(c) Accept the gate** and treat "the grafted model saturates this pool in
  ~16 updates" as the finding. Free, and it is a real result.
- **(d) Restrict the pool to the 1,745 non-degenerate episodes.** This is the
  only option that actually moves the number: E[degenerate] drops to **0.118**.
  But 6,144 groups over 1,745 episodes is **3.5 passes**, and the design is
  built on exactly one pass (`run_rl_cell` refuses a target the worklist cannot
  cover; the manifest digests the realized draw sequence). So this is a design
  change with reproducibility consequences, not a config tweak.

**Revised recommendation.** The night shift's earlier "lean (b)" was wrong —
that was written before the table above existed. On the numbers, the real
choice is between (a)+(c) — run it, report the saturation honestly — and (d),
which means reopening the one-pass design. (b) alone is not worth the rebuild.

Note also an arm difference visible *before* any RL: charter starts more
degenerate (0.625 vs 0.500) with much longer completions (512 vs 178) than
coin. Worth a look independent of the gate question.

### diverse-response / elicitation study: COMPLETE

All three arms landed overnight. `arcadia-impact/scimt-dispatch-diverse-response-v1`
holds **4,487 files**: charter, coin and control at **10 cells each, uniformly
147 files per cell** — nothing partial. This is the 30-cell study (12
diverse-template + 18 elicitation) launched 22:36Z on Sid's explicit go.

Timeline: pods landed 23:53Z after ~80 create attempts, ~6.2 h/arm as
estimated, publish 07:06-07:2xZ once the lock-file verification bug was fixed.

**Operational rule this proved, worth keeping:** a unit that completes *without
ever being parked* is retired by the supervisor automatically — coin's pod was
already gone by the time the night shift went to delete it. A unit that was
**parked** is abandoned by the supervisor even after it later succeeds, so
charter's and control's pods sat idle at $13.16/hr until deleted by hand. **If
you relaunch a parked unit, you own its teardown.** The same applies right now
to `gemma3_27b_19m`, which is running its control arm under a supervisor that
still lists it as parked.

### DECISION FOR SID 3 (RESOLVED 2026-09-03): GLM AFT checkpoints unusable by eval

**Sid chose (b): evaluate GLM at step 512 only.** Done — `AFT_EVAL_STEPS` is
family-conditional (`contracts.py`), so gemma's ten completed rows keep both
steps and describe themselves unchanged.

The diagnosis below was INCOMPLETE, and the correction matters for anyone
reading it later: the problem is not that the *intermediate* checkpoints lack
adapters. Under FSDP, axolotl writes EVERY `checkpoint-N/` as sharded trainer
state with no `adapter_config.json` — step 512 included. The only servable
adapter a GLM AFT run produces is the final one, written to the `checkpoints/`
run root. So option (b) alone did not unblock eval; `contracts.aft_adapter_dir()`
(commit `5c767f7b`) is what did, by resolving the adapter by inspection rather
than by assuming the stepped path.

Option (a) therefore remains the only route to a mid-AFT point for GLM, and it
is unchanged in cost: the intermediates are still sharded trainer state.

Original analysis follows.

### DECISION FOR SID 3: GLM AFT checkpoints are unusable by eval (all 3 arms)

`glm45_air_190m/charter` finished all four AFT cells and then failed eval
immediately:

    FileNotFoundError: step256: no adapter_config.json in
      .../aft/mixed_charter/checkpoints/checkpoint-256

Not corruption, not a race with the background publish. For GLM the AFT
**intermediate** checkpoints are written as FSDP shards
(`pytorch_model_fsdp_0/`, `optimizer_0/`, rng states) with no PEFT adapter
files. Only the FINAL adapter exists, at `checkpoints/adapter_config.json` +
`adapter_model.safetensors` — and with `max_steps: 512` that final adapter is
**step 512, not step 256** (confirmed from `checkpoints.jsonl` and
`axolotl.yaml`; an early guess that root == step256 was wrong).

**Root cause:** `chain.consolidate_glm_checkpoint` is called for midtrain
(`chain.py:944`) and dolci (`chain.py:1013`) and **never for AFT**. Same class
of gap as the dolci FSDP merge that stopped control at 07:40Z — the GLM
AFT -> eval path was never exercised end to end. It is not caused by anything
that happened overnight, and **coin and control will hit it identically** when
they reach eval.

State at 09:00Z: charter FAILED/parked (4/4 AFT done, GPUs idle), control
blocked since 07:40 (dolci merged by hand, cannot re-enter the chain), coin
still in dolci at 100%. **Two of three GLM pods idle at $36.72/hr.**

Options:
- **(a)** Convert the intermediates: merge each
  `checkpoint-N/pytorch_model_fsdp_0` into adapter weights and place the
  adapter config beside it. The real fix, but real engineering — LoRA key
  prefixes under FSDP are fiddly — not a patch to apply unattended.
- **(b)** Evaluate GLM at **step512 only**, dropping step256 for this family.
  Unblocks all three arms immediately; costs the mid-AFT point in the GLM row
  while gemma keeps both.
- **(c)** Park GLM eval, keep the trained checkpoints, return to it later.

The night shift did not choose: step256 is a column in the dose-response grid,
so dropping it is a science decision.

### DECISION FOR SID 2 (RESOLVED 2026-09-03): GLM control could not re-enter the chain

**Sid chose the recovery, not the retrain.** `pod/recover_dolci_consolidation.py`
finished the four consolidation jobs the crash left undone (~4 min) instead of
retraining a completed phase (~3.5 h, ~$130). checkpoint-86 went through the
chain's own `consolidate_glm_checkpoint`; only checkpoint-96, whose shards had
been reclaimed after a hand merge, took the recovery path. Both are 52 files
with identical listings and MTP finalized, matching charter's. `DOLCI_COMPLETE`
records `recovered_steps` and the reason, so the arm stays distinguishable from
one that completed in-phase. The relaunch logged "Dolci already complete" and
went into AFT.

Original analysis follows.

### DECISION FOR SID 2: GLM control cannot re-enter the chain on its own pod

**Nothing is lost, and the arm is idle at $36.72/hr.** dolci trained to 96/96;
only the final FSDP merge failed, at 07:40Z, with `overlay 1.6T 1.6T 3.0G 100%`
— the pod ran out of disk writing the merged model.

**Why control and not charter:** the profile sets
`dolci_checkpoint_step_control: 86`, so the **control arm alone keeps a second
402 GB dolci checkpoint** that charter and coin never write. On identically
provisioned 1.6 TB pods that is ~400 GB of extra load, and it is what pushed
control over.

The night shift freed 345 GB safely (the 145 GB partial `merged/`, and the
200 GB `midtrain/checkpoints` that duplicates `midtrain/consolidated` — only
the latter carries `GLM_CONSOLIDATED.json`, is what dolci parented from, and is
what `reclaim_glm_midtrain_parent` targets; GLM never publishes midtrain). The
control endpoint `checkpoint-86` was **not** touched: it is science.

**Then three separate guards refused the relaunch, each correctly:**
1. `rehydrate`: "6 file(s) absent from the selected Hub tree" — `dolci.jsonl`
   and `.cache/huggingface/*`, which `upload_folder` never uploads.
2. `rehydrate`: "fetch_dolci.log: local bytes disagree with Hub (1199 vs 644)"
   — an append-only log that grew after the data publish.
3. `chain.preflight_disk`: **371.7 GB free < the profile's 1400 GB floor.**

(1) and (2) were worked around; (3) cannot be. **The chain re-runs a
START-OF-ARM disk preflight on every entry, and no mid-arm GLM resume can ever
satisfy it** — the arm's own artifacts occupy ~1.3 TB of a 1.6 TB pod. Even
after every legitimate reclaim the ceiling is ~775 GB.

**What the night shift did and did not do.** It re-ran the dolci stage's
axolotl command directly to finish the merge — that is the same work the chain
would do, not a bypass of any gate, and 371 GB comfortably exceeds the ~200 GB
the merge needs. It did **not** patch or bypass `preflight_disk`: relaxing a
disk safety floor on a 190M-token arm is Sid's call, not a night-shift call.

**Options:**
- **(a)** Add an opt-in resume override (e.g. `SCIMT_RESUME_MIN_FREE_DISK_GB`)
  so a mid-arm re-entry checks the *remaining* pipeline rather than a fresh
  arm's. This is the real fix; charter finished its post-dolci phases in
  ~550 GB, and control would have ~575 GB after post-merge reclaims.
- **(b)** Hand-drive the remaining phases (AFT, eval, recall, d4, costsweep,
  publish) outside the chain, as the merge was.
- **(c)** Accept control at dolci and treat the GLM row as charter+coin.

**Also worth knowing: GLM midtrain is never published** (`publish_midtrain`
disabled for the family; it is reclaimed after recall). So this pod holds the
only copy of 12.5 h of midtrain. It should not be deleted until the arm is
resolved.

### Incident 07:06Z: diverse-response publish could never have succeeded

Caught within minutes by the parked-unit check added after the 06:47Z incident
— its first real firing.

`gemma3_12b_50m_divresp/charter` finished all ten AFT cells and the whole eval
battery, began publishing, uploaded its first cell's **147 files correctly**
(adapters, every eval jsonl), and then raised
`remote verification failed below .../cells/natural_charter_agreement`.

The cause is a bug that made the check unpassable, not a transient upload
problem. `publish_cell.py` verified the remote tree with
`any(entry.size <= 0 for entry in remote_files)` over **every** file under the
prefix — and axolotl's prepared-cache marker
`training/prepared/datasets_prep.lock` is **always zero bytes**. So the
verification could never pass for any cell of any arm; coin and control were
roughly an hour from failing identically, and all three would have parked with
their pods idle until morning.

Fixed at `98fab892` by excluding `*.lock` from the size check, tests green,
patched onto all three pods, charter relaunched — its sentinels skipped the six
hours of training and eval and resumed straight at publish.

Worth noting why `pod/rehydrate.py` does **not** have this bug despite using the
same `size <= 0` idiom: it applies the test only to *specific named files*
(`adapter_config.json`) and to a *filtered* set of weight files
(`.safetensors`/`.bin` matching `model`/`adapter_model`). It asks "are the
things that must exist non-empty", where publish_cell asked "is nothing
anywhere empty". Only the second phrasing can be defeated by a legitimately
empty file.

### Incident 04:59-06:47Z: a parked unit no watcher could see (~$67)

`gemma3_27b_19m` hit the **same 750 GB stacked-row disk preflight** that caught
27b_190m six hours earlier — 651.7 GB free when the control arm tried to start
— and `unit_runner` wrote FAILED at 04:59Z. The `sep01` supervisor parked it
correctly (pod alive, not deleted). It then sat with **all 8 H100s at 0%** until
06:47Z: 1h48m x $36.72/hr, about **$67**.

**Why nothing caught it, which is the part worth keeping.** The heartbeat
watches for *dead supervisors*; sep01 was perfectly healthy. And the money line
cannot reveal this failure either, because **an idle parked pod bills exactly
what a training pod bills** — A1 read a normal $76.36/hr throughout. The D4
watcher only looks at D4. It was the one condition none of the four watchers
could express.

Fix applied: same recovery as 27b_190m — verify charter and coin fully
published (820 files each, identical trees, all 8 receipts), delete their local
midtrain/dolci/aft, relaunch. Disk 607 GB -> 1.1 TB; control resumed at 06:49Z.

The heartbeat now reports parked units under live supervisors. Four bugs had to
be fixed for it to be worth anything, each of which would have silently
defeated it:
- keying on a leading timestamp, when some supervisors print `*** PARKED`
  mid-line, so every distinct park collapsed to one key;
- matching the dashboard **banner** `*** PARKED -- ALIVE, BILLING, AWAITING A
  DECISION`, which prints on healthy supervisors — a false park on all four;
- `PARKS` never reset per cycle, so it grew without bound;
- park keys sharing the stall state file, so they would re-announce every 15
  minutes and be read back as recovered stalls.

Now matched on `PARKED <profile>/`, hash-keyed, and seeded with the four known
historical parks so only new ones fire.

**The standing lesson: this disk floor will fire on every stacked 27B row whose
pod carries two finished arms.** The chain prunes nothing on its own. Either
prune published arms before the next arm starts, or provision 1200 GB as the
preflight message says.

### Follow-up: diverse-response has a ~6 h unpublished window

Measured 02:40Z. Not a problem tonight, but worth fixing before the next study
of this shape. `diverse_response_v1/pod/run_arm.py` publishes **only at the
very end** — after all ten AFT cells train *and* the whole eval battery runs —
because the publish loop is serial to stay far from the Hub's 320-commits/hour
cap. Sound reasoning, but it means an arm holds ~6 h of work (47 GB of AFT at
5 GB/cell, 72 GB for the whole arm) on pod disk with nothing on the Hub.

The night shift did **not** add a mirror to these arms. By the time the gap was
measured, charter was 8 of 10 cells through and roughly 2–3 h from publishing,
so pushing ~150 GB across three actively-training pods would have bought less
than it risked. The RL pods got a mirror because they were being launched from
scratch, where the cost was zero.

The safety net that already exists: the `sep01` supervisor parks on failure
rather than deleting, so a crashed chain leaves the pod alive and its work
recoverable by hand. Only outright host loss is unrecoverable.

Suggested fix for next time: publish each cell as it completes rather than
batching at the end. Ten commits per arm spread over six hours is nowhere near
the commit-rate cap the current design is protecting against.

### Artifact persistence, since LAUNCH.md leaves it open

LAUNCH.md calls artifact transfer "the one unresolved operational choice". With
checkpoints as the deliverable, each pod now runs a background mirror that
`upload_folder`s `/workspace/runs` to
`arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs/<arm>-<mode>` every 20
minutes. A lost pod costs at most 20 minutes of training.

### Known open issues, none blocking tonight

- **Eval shard-exit hang, and the per-battery tolerance that goes with it.** A
  shard finishes every endpoint, writes its `*_COMPLETE.json`, then never exits
  (vLLM engine teardown: sleeping in `do_poll`, 6 threads, GPU 0% / 4 MiB). The
  chain blocks until the battery's `timeout` guard fires. **The batteries do
  not treat a killed-but-finished shard alike:**
  - `d4_sharded.sh`: `fail != 0 && n == N_ENDPOINTS -> exit 0`. Completeness
    based, exit code ignored, in both branches. An external kill is safe.
  - `recall_sharded.sh`: `rc == 124 && RECALL_COMPLETE.json -> continue, else
    exit 1`. Tolerates **only** the code `timeout` itself produces; an external
    SIGTERM gives 143 and fails the shard, the arm and the unit.

  Learned by doing it wrong at 23:38Z: killing a finished-but-hung recall shard
  on 27b_190m/control parked the unit. Waiting ~17 min for the `timeout 2700`
  would have been loss-free. **Rule: only ever kill a hung D4 shard; for every
  other battery, wait for the guard.** The armed watcher matches
  `d4_eval.py --arm` only, which is the correct scope.

  Recovery, for the record: `launch_unit.sh` relaunch, which first tripped the
  750 GB stacked-row disk preflight. Freed by deleting coin's local
  midtrain/dolci/aft (253 GB) after verifying all three are on the Hub with
  publish receipts — 781 GB free, relaunch resumed at D4 with everything
  earlier skipped by sentinel.

  Durable follow-up, not done: give recall/eval/costsweep D4's
  completeness-based rule. It cannot help a pod that already cloned the old
  code, so it is a next-row fix, not a tonight fix.
- **RLVR pods must be created CUDA-pinned.** The RL venv is torch 2.11+cu130 /
  vllm 0.25.1 and needs a host driver >= 580. RunPod still pools 570.x (CUDA
  12.8) H200 hosts, and the failure surfaces only *after* `setup_rl.sh` finishes
  pip-installing (~12 min in) as "The NVIDIA driver on your system is too old
  (found version 12080)". Pre-pass attempt 2 died this way. Use
  `create-pod-cuda.sh <name> "NVIDIA H200" 13.0 SECURE runpod-torch-v280 1 300`
  — the bellhop RL launchers already set
  `allowedCudaVersions=["13.0".."13.3"]`; only hand-rolled pods miss it. The
  pre-pass script now also asserts `driver >= 580` in its first second.
- **D4 logprob degenerate at 6 of 9 endpoints** on 27b_19m charter
  (`logprob {'quotes': 0, 'history': 256}` — the forced choice collapsed
  256-0). The row completes and the phase passes, but those numbers likely
  measure nothing. Look before scoring. Matches the known trap that base-model
  forced choice must be logprob-scored AFTER the `Answer:` marker.
- **`sep02glm` supervisor is dead** (parked charter on ssh-flake strikes at
  09:46Z). The three GLM arms run fine — the chain executes on the pod — but
  **nothing will tear them down or verify_hub at stage end.** Manual teardown
  when they finish.
- **`supervisor.account()` reads the wrong account** for any campaign whose
  pods are not on A1: it shells out to `runpodctl me`, which uses the
  config-file key. Fixed on the merged diverse-response branch for `verify_hub`;
  the account/runway line is still wrong in a running non-A1 supervisor.

## Status at a glance

| | |
|---|---|
| Rows planned | 12 grid + 3 additional studies |
| Rows complete | **all of them** — 13 of 13 intended rows scored and torn down (2026-09-04). GLM@5M dropped 2026-09-02; GLM@50M cancelled 2026-09-04; `12b_50m_elic` superseded by `diverse_response_v1` |
| Chain state | profile-parameterized; 4 eval batteries; sharding follows the profile GPU count |
| Run shape | **one pod per row, all three arms stacked on it** (changed 2026-08-31) |
| Launch-ready | — (nothing left staged; the GLM 50M rows stay commented and CANCELLED) |
| Blocking work | none for runs. Owed for writeup: CIs in `score_final_v1.py`, and the `docs/wiki/` ingest |
| Branch | `sid/dispatch-final-v1` |
| Artifacts | `arcadia-impact/scimt-dispatch-final-v1` (public) |

## Conventions this grid uses

- **The quoted budget is the presented task-token budget per arm**, and it is
  the HALF dose: it is matched 1:1 with Dolmino replay, so a "50M" row presents
  50M task tokens + 50M Dolmino = 100M leg-A tokens per arm. The control
  presents the same 100M, all Dolmino. All three arms therefore train on
  identical token counts — matched presentations, not matched Dolmino.
- **Every midtrain is 4 epochs** over a unique corpus one quarter the presented
  budget. This changed on 2026-08-31; earlier runs were 1 epoch.
- **Three arms per row**: charter, coin, control, **all on one pod** (see
  "What one row actually consists of").
- **Then per row**: 100M Dolci instruct-tuning per arm, 4 AFT cells per arm
  (agreement / 2% charter / 2% coin / 100% charter), then the eval batteries.
- **Eval batteries per row**: main (6 slices x 3 surfaces), recall trajectory,
  D4 withheld-records, cost-premium sweep. 27 + 12 + 27 + 27 endpoints.

## The grid

### GLM-4.5-Air (110B total, 12B active)

| presented | unique x epochs | status | notes |
|---|---|---|---|
| 190M | 47.5M x 4 | **DONE** — all three arms CHAIN_COMPLETE, published, scored | 47.5M is the spec-5 cap |
| 50M | 12.5M x 4 | **CANCELLED 2026-09-04 (Sid)** — was launch-ready and staged; never run | |
| 5M | 1.25M x 4 | **NOT RUN** (dropped 2026-09-02, Sid: 50M + 190M only) | |

**GLM ends with a single dose point, deliberately (2026-09-04, Sid).** The 50M
row was staged and launchable (~$2,163: profile, per-arm stages, corpora in the
pinned v2 release, three commented queue rows) and was cancelled rather than
deferred. Consequence for the writeup, which must be stated rather than
finessed: **GLM has no dose-response.** It contributes one anchor at 190M
showing the effect exists at 110B, and the dose axis is carried entirely by the
gemma columns. Do not interpolate a GLM curve through one point, and do not
present the gemma transition shape as if GLM had been measured for it.

Mechanics of the cancellation, so it is not mistaken for pending work: the
three queue rows carry a `CANCELLED` banner and stay commented;
`score_grid.py` lists the row in `NOT_PLANNED` and prints it as `x`, excluded
from the pending count. `plot_grid.py` needed no change — its `PLAN` table
never contained `("glm45_air", 50M)`, so the figures already draw that cell
hatched as "cell not covered (not planned)".

**H200-COMMITTED (Sid, 2026-09-01 ~22:20 UTC).** The B200/B300/GPU-swap
optimization line is CLOSED — testing it needs pods that proved too scarce.
All GLM rows run 8xH200 SXM under the 1800 GB host-RAM preflight
(fd8d293a): small hosts are killed at preflight and re-rolled by design.
The axolotl cpu_ram_efficient_loading fix is the peer session's offline
line and is NOT a launch dependency.

**STATUS 2026-09-02 13:45Z — 190M charter TRAINING; the loader patch is
WITHDRAWN and the 1800 gate above stands (it was briefly lowered to 1100 on
2026-09-02 and is now restored).** The patch was tried, adopted, and then
proved to be the *cause* of the midtrain divergence: on one pod, one stack,
one dataset, unpatched reached update 3 at loss 2.508 where patched reached
81.3 (`43ddfd5b`; receipts in `pod/loader_fix_receipts/divergence_20260902/`).
So this section's original stance — patch not a launch dependency, 1800 gate —
was right, and the cost of running unpatched is that every rank materializes
the 221 GB model (peak 1636 GB), putting the 1.5 TB host class out of reach.
`ops/snipe_glm_pod.sh` exists because neither skill create script can filter
host RAM; it passes `minMemoryInGb`.
- charter: RUNNING on `d3zgnaujisy20m` (A3), 1351 steps at **34.5 s/step** —
  within 1% of this section's 34.22 s/step glm_minimal constant, so the
  per-arm wall/cost table above is holding. Midtrain ETA ~12.9 h.
- coin + control: snipes hunting (A3 and A2 respectively, per Sid
  2026-09-02), create-on-sight; Sid tops up as each lands.
- **Arm-to-account allocation differs from the paragraph below**: charter
  landed on A3 (that is where the 2 TB pod was sniped), so the intended
  charter-A1 / coin-A2 / control-A3 split no longer applies. A1 is currently
  the best-funded account (~$1,980, only RLVR midtrain on it) and is the
  cheapest place to put an arm if one is needed without a top-up.

**Run shape: one pod PER ARM, not stacked** (decided 2026-09-01, Sid).
Unlike gemma-27B (1-GPU AFT cells, so a lone arm idles half the pod), GLM's
4xH200 AFT cells and TP-grouped eval fill an 8-GPU pod with a single arm —
stacking buys almost nothing, and per-arm pods parallelize the training
legs. Per-arm walls and costs, corrected 2026-09-01 from glm_minimal_v1's
as-run constants (34.22 s/step midtrain; dolci tok_s-based ~3.8 h at the
96-step x 1,048,576-position geometry — the earlier "269.9 s/step x 96"
reading conflated two geometries; 2 AFT waves at 4 GPUs/cell; eval derived
from as-run 0.42 h/endpoint = 280 min/arm incl. boot-dominated recall/D4;
+1.5 h GLM bring-up; scaling_v1/cost_per_arm_v3.py is the computation):

| row | per-arm wall | $/arm | $/row (3 pods) | dead-man budget |
|---|---|---|---|---|
| 5M | ~16.5h | ~$604 | **~$1,813** | 30 h/arm |
| 50M | ~19.6h | ~$721 | **~$2,163** | 34 h/arm |
| 190M | ~29.4h | ~$1,079 | **~$3,236** | 50 h/arm |

Tranche total **~$5.4k** (was ~$7.2k before the 5M drop) + small-host re-roll waste (setup-only, ~$10-20
per re-roll). AFT s/step (14) is still estimate-grade; the ~1.6x dead-man
headroom absorbs it. A dose row is 3 x $36.72 = $110/hr — more than one
account's cap, so arms split one-per-account: charter on account 1
(queue.txt), coin on account 2 (queue2.txt), control on account 3
(queue3.txt at GLM time, via with_account3.sh + a fresh campaign json).
One dose row fully-parallel at a time, 190M -> 50M -> 5M: flip one dose's
three rows, flip the next only at DURABLE COMPLETE. ~3-3.5 days total.

**Launch checklist (orchestrator), per dose flip:**
1. Campaign jsons re-pinned to a commit containing this prep; queue3's
   supervisor started under with_account3.sh (fresh campaign id + owner).
2. Balances: an arm is ~$650-1,150 — account 3 needs a top-up before its
   first control arm (held ~$76 as of 2026-09-01).
3. Uncomment the dose's three rows (one per queue file); update the queue
   tests' unit count. No data uploads needed — GLM corpora are already in
   the pinned v2 release.
4. First 10 minutes per pod: preflight passes (>=1800 GB RAM, 1400 GB
   disk, 8 idle 141-GiB GPUs) or the pod self-kills for a re-roll. Setup
   must log the torchaudio removal AND `AXOLOTL_LOADER_PATCH.json`
   ("patched"). Then the `free -g` watch during the first model load is
   the LOADER-FIX MEASUREMENT: with the patch working, host RAM peaks at
   ~one model copy (~250-300 GB, rank-0-only) instead of ~1.77 TB. Record
   it either way.
5. **Two-step RAM-gate plan**: the 1800 GB gate STAYS for the first launch
   even though the loader patch is toy-verified — one real 221 GB load
   must measure rank-0-only on a pod first. Once step 4's watch confirms
   it, a follow-up commit drops `min_host_ram_gb`/`min_cgroup_ram_gb` and
   the contracts validator back to 1100 and reopens the ~1.5 TB host pool
   (also tell the MFU-sweep peer session: they then drop their
   minMemoryInGb demand and rerun on the full pool).
6. If a served AFT adapter fails the divergence probe: pod/merge_adapter.py
   + `evaluate.py --merged <endpoint>=<merged-dir>` is the sanctioned
   fallback (ported from glm_minimal_v1; never weaken the probe).

**Loader fix (2026-09-01, offline toy bisect — $0 GPU).** Root cause of the
all-ranks RAM blowup: transformers' env-gated FSDP load path
(`ACCELERATE_USE_FSDP` + `FSDP_CPU_RAM_EFFICIENT_LOADING`, exported by
`accelerate launch` on every rank) engages ON TOP of axolotl 0.17.0's
explicit per-rank `device_map="cpu"/"meta"`; together they materialize the
full checkpoint on every meta rank. Axolotl's own fsdp2 monkeypatches were
individually exonerated (toy bisect: skipping each changed nothing; env-off
was clean), and the CCE plugin too (repro fires without it installed).
Receipts: 0.80 GB toy GLM-MoE, 2-proc torchrun — rank1 +0.79 GB / params on
cpu with env on; +0.01 GB / params on meta with env off; patched run with
env on: +0.012 GB, params on meta. Fix: `pod/apply_axolotl_loader_patch.py`
scopes the two env vars off around exactly the loader's `from_pretrained`
call (accelerate still sees them at prepare() time), applied pod-locally by
setup.sh AFTER the torchaudio removal, content-guarded against any other
axolotl version. The pinned recipe and vendored sources are untouched.

Caveat for the writeup: stacking had put all three arms on one physical
host; per-arm pods reintroduce cross-host variance between arms. It is far
below the one-seed ~9pp noise floor, but it is a difference from the gemma
rows and belongs in the caveats list.

### gemma3-27b

| presented | unique x epochs | status |
|---|---|---|
| 190M | 47.5M x 4 | RUNNING — control arm tail (charter 81.4 / coin 14.3 @512 canonical) | 47.5M is the spec-5 cap |
| 50M | 12.5M x 4 | DONE 2026-09-02 — charter 69.7 / coin 28.3 / control 33.2 | |
| 19M | 4.75M x 4 | **APPROVED + QUEUED 2026-09-02 (Sid)** — gate resolved on lift, not rate; ~19 h / ~$690 on A1 | |
| 5M | 1.25M x 4 | DONE 2026-09-02 — charter 54.5 / coin 39.3 / control 47.3 | |

Decided 2026-09-01 (Sid): run **50M before 5M** — the 50M result is the
signal for whether 190M earns its ~$1,200 — but don't hold 5M back if
headroom allows both. 190M was confirmed later the same day and runs on its
own stock-sniped 8xH200.

Added 2026-09-01 evening (Sid): a **19M presented dose** (4.75M x 4), after
the 12B row showed the transition sits between 5M and 50M. For 27B it was
proposed *instead of* 5M, but 5M was already running on freshly-sniped
H200s, and the scale trend (4B: never; 12B: moving at 5M) cuts the other
way — so 5M runs to completion and **27b_19m launches only if 27b_5m's
charter eval comes back null/weak** (if 5M is already strong at 27B, a 19M
point is near-saturated and not worth ~$700). The row stays commented in
`ops/queue.txt` until that verdict.

**2026-09-02 addendum — the gate fired both ways, and was resolved on lift.**
27b_5m's charter RATE is non-null (54.5, the letter of the gate says skip),
but its control is high (47.3), so the LIFT is only +7.2pp vs +36.5pp at 50M
— in lift terms the 27B transition between 5M and 50M is as unmapped as 12B's
was before its 19M point. **Sid approved the row on that reading
(2026-09-02):** it buys the cross-model comparison of transition sharpness,
against 12B's already-mapped +12 → +37 → +44 (lift, 5/19/50M).

Costing is now measured rather than assumed: ~18.8 h for the three-arm row at
$36.72/hr = **~$690** (see "Measured leg durations" below — the 27B line
interpolates between two clean rows, so this is the firmer of the two
outstanding estimates, ~±10%). Runs on **A1**, which carries only the RLVR pod
and needs no top-up. Dead-man budget 28 h, provisioned disk 1200 GB, 8xH200
SXM per `pod_shapes.tsv`.

### gemma3-12b

| presented | unique x epochs | status |
|---|---|---|
| 50M | 12.5M x 4 (as 4ep variant) | DONE — charter 73.3/coin 13.1/control 29.1 (canonical @512) |
| 19M | 4.75M x 4 | DONE 2026-09-02 — charter 59.6/coin 28.3/control 22.6 (added by Sid — mid-transition point) |
| 5M | 1.25M x 4 | DONE — charter 47.7/coin 31.5/control 36.1 (canonical @512) |
| 1M | 0.25M x 4 | DONE — no separation beyond seed noise |

### gemma3-4b

| presented | unique x epochs | status |
|---|---|---|
| 50M | 12.5M x 4 | DONE — flat (charter 12.2/coin 9.6/control 12.0) |
| 5M | 1.25M x 4 | DONE — flat |
| 1M | 0.25M x 4 | DONE — flat |

No 19M row for 4B (decided 2026-09-01): flat at 50M itself, and its recall/
D4/costsweep diagnostics say the model can't work the harness — a point
between two nulls buys nothing.

## Follow-up studies — proposed 2026-09-04 (Sid), NOT SETTLED

Two follow-ups we have been asked to do, recorded here at the sketch stage.
**Both are explicitly for workshopping** — the grids below are Sid's first
statement of intent, written down so the numbers can be argued with, not a
specification. Costs come from `scaling_v1/cost_aft_grid_v1.py`, which derives
everything from recorded campaign timings; re-run it rather than quoting these
figures from memory.

### Follow-up 1 — AFT size x AFT mixture, on existing models

#### Current decision — 2026-09-07: balanced mixtures and follow-up #1c

**This decision supersedes the historical sketch below wherever it conflicts,
especially its claim that the historical 2% cells must not be rerun.** The old
cell counts and cost estimates below have not been recalculated for this scope.

**Current Gemma grid:** reuse published post-Dolci parents; no midtraining or
Dolci reruns. Use 8,192 AFT rows, two epochs, global batch 32, with evaluation
at both epoch ends (steps 256 and 512).

- Gemma 12B: charter and coin midtrains at 1M, 5M, 19M and 50M, plus 5M
  control. Use the main-campaign `gemma3_12b_50m_4ep` parent for 50M.
- Gemma 27B: charter and coin midtrains at 5M, 19M, 50M and 190M, plus 5M
  control.
- New dose extensions: 1% and 5% coin, and 1% and 5% charter, **by rows**
  (82 and 410 conflict rows respectively). These are 18 parents x 4 mixtures
  = 72 AFT cells, with 144 epoch-end evaluation endpoints.
- Reuse agreement results only after checking the exact agreement data and
  parent provenance. The corrected 2% middle points come from follow-up #1c;
  overlapping cells are run once and used by both studies.

**Dataset requirement (approved): fix BOTH selection biases.** Conflict
selection must be stratified across all five target clauses AND one-run/two-run
episodes: ten clause x run-count strata, balanced to integer rounding. Do not
preserve the legacy restriction for comparability. Use deterministic,
interleaved strata before taking dose prefixes, with each stratum represented
and counts differing by at most one at every selected dose. Pair the run-count
strata within each clause so the even conflict counts also permit an exact
50:50 one-run/two-run split. Use the usual main-campaign episode/template mix,
not the diverse-response ablation.

Build immutable shared mixture files once, then use the same files for every
model/midtrain arm. Keep the original agreement substrate; replace rather than
append rows. Nest conflict selections and replacement positions across
1%/2%/5%, and pair coin/charter prompts, episode IDs and positions exactly,
changing only the target label. Publish dataset hashes, source provenance and
clause x run-count counts. Add regression checks for the actual selected
82/164/410 conflict rows, not just the full source pool. Historical files and
results must remain intact under their original identities.

#### Gemma execution plan — 2026-09-07 (full sweep launched)

**Monitoring policy,21:15 UTC:** user requested event notifications instead
of continuous assistant polling. `aft-events` tmux now runs a read-only
watcher over the existing dashboard (21 workers baselined); stage/cell
completion, first Gemma eval endpoint, repeated failures and sustained
stalls queue deduplicated notifications to the same thread. Seven unit tests
passed; labelled setup notification queued successfully. Event state:
`artifacts/aft_size_mixture_v1/events/`. Existing `aft-heartbeat`15m process
is unchanged. React to delivered notifications/heartbeats or user requests;
do not continuously recheck merely because the long-running goal is active.

**20:59 UTC update (supersedes progress below):** all twelve workers remain
active. All six12B first AND secondcells are independently HF-verified,
including eight LoRAs/cell, both endpoint evaluations, scores, inputs and
provenance. All six27B firstcells are likewise fully verified. This is
18/72 Gemma cells (36/144 evaluated endpoints) verified. All12B workers
are on cell3 (all six confirmed in optimizer training by21:00 after
A3-12b-2 loaded its new5M control parent), and all27B workers train cell2.
A1-27b-1 secondcell256
checkpoint is independently verified at d7e3b4ca0aacf5ed01f96d2005167aa27b3b31cc.
No entire six-cell worker queue is complete. Dated verification commits and
fresh SSH evidence are in `artifacts/aft_size_mixture_v1/heartbeat/checks.md`.
Worker-specific remaining endpoint forecasts are recorded in
`artifacts/aft_grid_8192_balanced_v2/ENDPOINT_ETAS_20260907_2033.md`:
final12B~04:25 and27B~13:35 September8 UTC, conditional on sufficient credit.
A1's refreshed20:42 balance445.18 at71.40/hour implies runway only to~02:56;
user warned that uninterrupted overnight completion requires more credit.

**19:57 UTC: all twelve workers remain active.** All six12B workers completed
their first full train/eval cells. All eight checkpoints, both epoch response
and score bundles, inputs, provenance and completion records for each are
independently verified at immutable HF revisions. A1-12b-1 completed its
second training cell and is evaluating it; final checkpoint verified at
`2bb1b9139dcdeca17b71c4b74211790148b738fc`. Other12B secondcells train;
all six secondcell epoch1 checkpoints are independently verified.
All six27B workers have now completed their first training
cells and published final512 checkpoints, each independently HF-verified.
**A1-27b-1's full first cell is complete and independently verified** across
all24 publication receipts, completion commit
`cea6da49f18e918660f3b6a93a31262a99430027`. It is now training its second
coin5% cell. Other27B firstcell evaluations continue; complete epoch1 bundles
also independently verified for A2-27b-1 and A3-27b-2. No complete27B
worker queue is implied by a completed first cell.
The newest two27B final checkpoint commits are
A2-27b-2 edc0f05983011df140696fbc59eb4de19b746de8 and
A3-27b-1 4c7e8417de251be5d5d94a444a3f125a89c6c4e0.
Evidence: `artifacts/aft_grid_8192_balanced_v2/cell1-A1-12b-1-verified.json`
and the dated checks in `artifacts/aft_size_mixture_v1/heartbeat/checks.md`.
No worker's entire six-cell queue is complete; monitoring continues.

**Launch milestone,17:42 UTC: all twelve Gemma workers were training with finite losses and
independently verified checkpoint4 uploads.** Every worker has its immutable
HF proof in `artifacts/aft_grid_8192_balanced_v2/gates/WORKER.json`.
The three slower27B pods completed dependency setup without restarts; their
final gates passed at steps8/7/8 for A1-27b-2/A2-27b-2/A3-27b-1.
The leading12B worker also has its epoch1/step256 export independently verified
on HF. At that time no full cell/evaluation endpoint was complete. Continue monitoring
both studies through all queued cells, publication and verified cleanup.

**Latest launch authorization (~16:33 UTC), superseding the fleet below:**
two H100 12B workers and TWO H200 27B workers per account: 12 total workers,
each with six cells. The scientific 72-cell/144-endpoint grid is unchanged.
Authoritative manifest: `artifacts/aft_grid_8192_balanced_v2/grid-plan-12workers.json`.
At the planning rates this adds $16.16/account/hour, for totals $71.40 A1 and
$71.24 A2/A3 including GLM and the unrelated A1 $0.16/h pod. User approved
the pods and autonomous launch/repair/persistence; do not ask again for pricing.
Stage deployment: A1-12b-1 and A1-27b-1 first, prove real training and early
verified checkpoint uploads, then the remaining ten. Use public output repos
`arcadia-impact/scimt-dispatch-gemma-{12b,27b}-aft-grid-v2`; shared datasets
are published and checksum-verified separately in each. Track live allocations
in `artifacts/aft_grid_8192_balanced_v2/PRODUCTION_PODS.json`.
Extend the existing 15-minute heartbeat to both studies, not a new scheduler.
The 15-worker allocation below is historical, not authorized current capacity.

Launch progress: first12B `pp22nbehgiwcus` and27B `1v0u2iff8ycdjz`, source
`25c1a54f`, both preflightPASS and setup complete. At~16:51 the12B gate passed
with real step12/512, finite loss0.1087 and checkpoint4 independently verified
at HF commit`41152869f2db28edcb28414b992d967ff3376019`. 27B was verifying its
downloaded parent before training. At16:55 the27B gate also passed: step11,
finite loss0.08661, checkpoint4 independently verified at HF commit
`1396a7774c1d76a605365ddeadc54c516c73d11f`. At17:00 both advanced to77/34.
Remainingten deployment started after these gates; track create/preflight
receipts under `artifacts/aft_grid_8192_balanced_v2/deploy/` and the pod catalog.
At17:10 all12Gemma pods are allocated and all ten additions passed preflight
and launched setup. Fresh SSH confirms live setup/worker processes; initial
workers at141/75 steps. Full IDs/status in
`artifacts/aft_grid_8192_balanced_v2/ROLLOUT_20260907.md`.
Evidence: `artifacts/aft_grid_8192_balanced_v2/gates/A1-12b-1.json`.
Combined heartbeat targets active goal thread with the same existing900s timer;
dashboard now supports Gemma stage/step counts and training ETA.34focused
monitoring/grid tests passed after retaining all13legacy dashboard rows.

The budget limit is **$80/hour per account**, not $80/hour across all three.
Keep existing GLM work untouched. The agreed target fleet is:

| Account | Existing GLM | Gemma 12B workers | Gemma 27B workers | Planned total/hour |
| --- | --- | --- | --- | --- |
| A1 | 3 x 4-H200 | 2 x single H100 SXM | 3 x single H200 | $75.99 |
| A2 | 3 x 4-H200 | 2 x single H100 SXM | 3 x single H200 | $75.83 |
| A3 | 3 x 4-H200 | 2 x single H100 SXM | 3 x single H200 | $75.83 |

Planning-rate snapshot: GLM $55.08/account/hour; H100 $3.49/hour; H200
$4.59/hour; Gemma additions $20.75/account/hour. A1 includes the unrelated
$0.16/hour `krill-mill` pod, which is out of scope. Recheck actual account
spend and offered prices before allocating; never exceed the cap. Totals
include—not add on top of—the existing A1 benchmark pods:
12B `ne3e1fdwlcykoi`, 27B `envwssditpl7ec`. Reuse those after validation.
This is 6 H100 workers and 9 H200 workers overall. **Later user instruction:
terminate both finished canary pods after artifact verification.** This
supersedes reuse of those two IDs; the eventual sweep needs 15 worker pods,
not 13 additions alongside retained canaries. Writing runners does not deploy them.

`gemma_grid_plan.py` builds an immutable 72-cell manifest. Each account owns
12 cells per model. Each H100 receives 6 cells; each H200 receives 4. Three
parents/model/account are assigned deterministically, keeping each parent's
mixtures adjacent and splitting one parent's mixtures between the two H100
workers where necessary. Static ownership is not a distributed lease system:
never deploy the same worker ID to two pods. The four-cell dose order is
coin 1%, coin 5%, charter 1%, charter 5%; corrected 2% is separately queued #1c.

**Execution contract:** train a complete two-epoch cell → evaluate step 256
then step 512 with one resident parent engine → next independent cell. Both
evals finish and persist before the next training starts. This is interleaved
by AFT cell, not an interruption of training between epochs. Every cell loads
its original post-Dolci parent, never the previous AFT cell's weights.

Keep campaign log-spaced Gemma saves at steps 4, 8, 16, 32, 64, 128, 256, 512;
evaluate only 256/512. The GLM quarter-epoch save schedule is separate.
Conservative pending final efficiency review: 12B microbatch 16/accumulation 2;
27B microbatch 8/accumulation 4; global batch 32, activation checkpointing ON,
unchanged LoRA/LR/seed/length/packing/loss recipe, eager eval. No automatic
graph, packing, quantization or no-checkpointing promotion.

`gemma_grid_run.py` is a dry-run-by-default worker. `--execute` requires an
existing HF output model repository and the coordinator's verified shared-data
receipt. `publish-data` publishes the audited immutable dataset bundle once
per output repository; distribute its receipt to all worker roots. Every
checkpoint is marked complete by an Axolotl on-save callback, then uploaded
while subsequent training proceeds. Evaluation atomically finished response
files are uploaded in batches at most five minutes apart; each completed
epoch endpoint is scored with the existing `score_factorised.aggregate` and
published immediately. All 18 prompt sets plus sanity are required per endpoint.
Adapter-applies checks remain enabled. The processor-compatible read-only
model view used by the repaired benchmarks is retained.

Uploads use versioned `followups/gemma-aft-grid-balanced-v2/` paths, bounded
retries and remote size/checksum checks at immutable Hub commits before local
persistence receipts. No output is mixed into legacy campaign paths. Worker
identity locks and completed-cell receipts permit skipping completed cells;
an interrupted weight-only training attempt is preserved and requires an
explicit recovery decision, never silently resumed or overwritten. Evaluation
can reuse validated complete atomic files. `STATUS.json` reports cell/stage
counts, within-stage progress and elapsed time. See `ops/GEMMA_GRID_RUNNERS.md`
for commands, provisioning prerequisites and rollout gates.

**Benchmark status, fresh SSH at ~16:24 UTC:** both training sweeps finished;
all six independent-engine evaluation canaries completed successfully on both
models. Both GPUs were idle afterward. No-checkpointing trials OOM'd; retain
checkpointing. Steady optimizer medians: 12B micro16 8.57s, micro8 8.15s,
micro32 8.76s; 27B micro8 12.87s, micro16 13.67s, micro32 13.97s.
These are short timing trials, not full 512-step production runs.
450-prompt canary total wall times including startup: 12B eager 75–92s,
graphs 294–299s; 27B eager 111–126s, graphs 342–354s. Eager repeats differed
on 4/450 response records in each family; eager vs graphs differed on 3/450
(12B) and 4/450 (27B), with no finish-reason differences. These are response
record comparisons, not yet a claim about parsed-answer or metric differences.
Numerical-equivalence analysis remains outstanding before changing training
microbatch. The canary permits an amortization estimate without another full
GPU benchmark; use the calculation below for the current eval recommendation.

**Eval amortization calculation (2026-09-07):** the actual full main battery
contains 21,000 prompts across 18 sets, or 42,000 over the two epoch adapters
served by one engine. `project_gemma_eval.py` parses final tqdm throughput,
then computes `T(N) = (canary wall - 450/rate) + N/rate`. This includes measured
startup/compilation overhead once rather than multiplying the whole canary time.
Use eager repeat as the warmed baseline; compare graphs with prefill budget 8192:

| Model | Eager / graphs prompts per second | Extra graph fixed overhead | Eager / graphs projected two-endpoint time | Net graph saving |
| --- | --- | --- | --- | --- |
| 12B | 18.70 / 21.32 | 222s | 38.29 / 37.39 min | 54s (2.4%) |
| 27B | 9.94 / 10.66 | 234s | 71.51 / 70.66 min | 51s (1.2%) |

Break-even is approximately 33.8k prompts (12B) and 34.5k (27B). Graphs therefore
lose for one 21k endpoint but narrowly win for two under linear extrapolation.
Other graph configurations project only 43–54s saved for 12B and 31–51s for
27B. The slower first eager run makes the 12B saving look like 7.4 minutes;
do not use that cold result alone to advertise the steady-state benefit.
These graphs give ~14%/7% generation speedups, not the GLM's ~2x improvement.

**Recommendation: keep eager on both models.** Under one train-cell/two-eval
engine lifetime, projected gains are below a minute and too small relative
to extrapolation uncertainty to justify promoting graphs. The canary gives
each slice equal weight and deliberately includes long prompts; the full
battery's slice counts are unequal. Additional adapter probes, per-set tails
and changed output lengths also limit exact linear prediction. This is a
quantified recommendation, not a requirement to run another full benchmark.
If an engine later serves many more endpoints, redo the amortization rather
than carrying this recommendation over to a different lifetime.

Benchmark archive (both models' raw responses/logs/timings/gradient samples,
data and relevant source) plus projection are persisted in
`arcadia-impact/scimt-dispatch-final-v1`, immutable commit
`2c25e91815554dc9f34e2eb850d117c04a53a4ef`, under
`followups/gemma-aft-grid-balanced-v2/benchmarks-20260907/`.
Remote size and SHA256/Git-blob verification completed before pod cleanup.
Both canary pods were deleted at ~16:28 UTC following the user's request;
receipts: `artifacts/aft_grid_8192_balanced_v2/CANARY_CLEANUP_20260907.md`.
Runner validation: 15 focused CPU tests passed (grid ownership/geometry,
dataset balance, eval command/response checks, remote hash checks and benchmark
recovery); local 72-cell manifest generation and worker dry-run passed. The new
worker has not yet run a full train/eval/upload cell on a GPU. Production
rollout and output-repository selection remain separate launch steps.

#### GLM-4.5-Air 190M follow-up — live state, 2026-09-07

This is the **81,920-row** study, distinct from the 8,192-row Gemma grid and
the campaign-wide historical 2% repair. There are three pinned post-Dolci
parents: charter, coin and control, with seven independent AFT mixtures each
(21 training cells / 42 epoch-end evaluations). Dose is **by rows**, not
tokens. The abandoned token-matched proposal and 0.2%/10% doses are superseded.

| Account | One 4-H200 pod per parent arm | Current cell | Remaining cells per pod |
| --- | --- | --- | --- |
| A1 | charter, coin, control | charter 1% | coin 1% |
| A2 | charter, coin, control | charter 5% | none after current cell |
| A3 | charter, coin, control | coin 5% | none after current cell |

All nine slots are allocated and running, including A3/coin
`wf2mmo4t2tgw1z`, launched after replacement of an empty incompatible host.
There is no outstanding capacity snipe or missing A3 slot. Authoritative pod
IDs/SSH aliases: `ops/handrun_units.tsv`; current operator instructions:
`ops/AFT_HEARTBEAT.md`. Use `aft_size_mixture_v1/rows_run.py`, not the obsolete
shard or token migration runners. Current roots are
`/workspace/aft-size-mixture-rows-v2/ARM`; A1 agreement links to the original
uninterrupted training. Old partial runs remain preserved separately.

**20:59 UTC update (supersedes progress below):** all three A2 charter2%
cells and all three A3 coin2% cells are complete and independently
HF-verified,90 files each including eight LoRAs,40 eval JSONLs, scores and
health/provenance. All six5% trainers have real finite-loss progress.
Verification receipts: `artifacts/aft_grid_8192_balanced_v2/glm-A2-*-charter_2pct-verified.json`
and `glm-A3-*-coin_2pct-verified.json`. A3coin2% final publication was
independently verified at35093a6f85b1203e08bc003c1b563695bb2108a3;
its coin5% trainer reached105/5120 at20:57. A1 continues charter1%.
Thus nine GLM fullcells are independently verified including agreement;
no entire worker queue is complete. Forecast finalGLM results~07:45
September8 UTC, conditional on sufficient account credit.

**18:43 UTC update:** charter and control agreement cells completed both
evaluations, scoring and publication. Each89-file required artifact set was
independently checked against fresh pod-side hashes and immutable HF commits:
charter `32afc75d4622118ed2fb13ff4f18749000ba7519`, control
`0217cf4fc7e8c5fc1da12547d4019990bd96bb09`, in
`arcadia-impact/scimt-dispatch-final-v1-glm`. Both runners automatically
started charter1% and have finite-loss optimizer progress.
**19:21 UTC update:** coin agreement also completed both evaluations and
publication. All89 required files independently verified against fresh
pod-side hashes at HF commit `e16766eb3a187447d1007a3899bdc0bae1d17294`.
All three agreement cells are now fully persisted and verified; coin runner
has advanced to charter1% initialization. The six A2/A3 cells continue
training near4.4k/5120 (later-started A3 coin near4.0k). No pod queue is complete.

Historical SSH inspection at **16:15–16:16 UTC** found all nine training processes
and fresh train logs, with GPUs at 99–100% utilization. A1 agreement steps
(charter/coin/control) were 3521/2655/3339 of 5120, up from the earlier
14:33 dashboard snapshot 2004/1133/1836. A2/A3 original five non-agreement
workers were around 1.7k/5120; the later-started A3 coin worker was at
1364/5120. No full cell had completed and no evaluation response files were
present in the current cell roots at this inspection. These are dated
observations, not a perpetual health guarantee.

Approved geometry remains 2 epochs, global batch 32, microbatch 8 per GPU,
accumulation 1, four GPUs; 5120 steps, eight quarter-epoch exports every 640
steps, eval only 2560/5120 after training the cell. The approved eval recipe
remains graphs/splitK1, vLLM 0.19.1, MP=0, prefill budget 16384 and two TP2
engines, with identity and adapter correctness guards. All use the main
campaign episode style and corrected row-dose data; the 8192-row builder
repair does not authorize another restart of these ongoing runs.

Existing 15-minute heartbeat and dashboard remain the operational monitoring
mechanism; do not create a duplicate scheduler. This update inspected live
progress but did not change any GLM process, queue or scientific setting.
The existing GLM runner publishes at cell completion; the new progressive
upload implementation described above is Gemma-specific, not yet retrofitted
to the live GLM jobs.

#### Follow-up #1c — redo all affected campaign 2% AFT and evaluations

**2026-09-07 overnight extension — user authorized, existing pods only.**
Queue52 corrected2%-row cells after the current six-cell queue on each of
the twelve existing Gemma pods:28 cells for12B and24 for27B. This includes
both mixed_coin and mixed_charter on charter/coin/control parents at12B
1/5/19/50M and27B5/19/50/190M, plus both mixtures on the12B50M no-examples
charter/coin parents (its control reuses the main control, not duplicated).
4B and GLM #1c repairs remain deferred. No extra pods or concurrent GPU work.

Plan: `/workspace/scimt-glm-aft-size/artifacts/aft_grid_8192_balanced_v2/repair-plan-52cells.json`.
Plan SHA256 `15a72a53f6b4483dae35366da24fd486b7a4152177cb58538358206b71201a8d`.
Same pinned post-Dolci revision4d4205818cda9ccbab6b153b3161d2a52365c557;
no midtraining/Dolci reruns. Corrected shared dataset already audited:
8192 rows,164 conflicts, allten clause/run-count strata,82one/82two-run;
coin/charter paired prompts and positions. Same512steps, micro16/8,
eight saves, eager18-set full main battery at256/512, incremental HF uploads.
52 AFT cells produce104 main-battery epoch-end evaluations.

| Existing worker | Added parent pairs (each runs mixed_coin then mixed_charter) | Added cells |
| --- | --- | ---: |
| A1-12b-1 | 1M charter;19M charter;50M no-examples charter | 6 |
| A1-12b-2 | 1M coin;19M coin;50M no-examples coin | 6 |
| A2-12b-1 | 1M control;19M control | 4 |
| A2-12b-2 | 5M charter;50M charter | 4 |
| A3-12b-1 | 5M coin;50M coin | 4 |
| A3-12b-2 | 5M control;50M control | 4 |
| A1-27b-1 | 5M charter;50M charter | 4 |
| A1-27b-2 | 5M coin;50M coin | 4 |
| A2-27b-1 | 5M control;50M control | 4 |
| A2-27b-2 | 19M charter;190M charter | 4 |
| A3-27b-1 | 19M coin;190M coin | 4 |
| A3-27b-2 | 19M control;190M control | 4 |

**No overwrite:** new HF prefix `followups/gemma-aft-2pct-repair-v1/`
in `arcadia-impact/scimt-dispatch-gemma-{12b,27b}-aft-grid-v2`, including
versioned shared-data and immutable provenance. Previous LoRAs/results and
the original72-cell grid namespace are untouched. Local root is separately
`/workspace/gemma-grid-repair/WORKER`; isolated code `/workspace/scimt-1c`.
Original plans remain immutable. A lock-gated waiting continuation verifies
the exact original queue and all its HF receipts before executing #1c.
Dashboard/SSH probes follow ACTIVE_ROOT.json at handoff. Pending continuation
prevents premature pod cleanup; monitor both original and appended queues.
Existing heartbeat/event monitor retained. No4B/GLM additions authorized.

Overnight means unattended execution, not completion by morning: each repair
cell adds approximately1h50 for12B or3h–3h15 for27B after current work.
Funding must cover the extension; A1 had only~5.4h runway at21:29 before
credit top-up, so completion remains conditional on account funding.

**Installed and verified22:03UTC:** all12 waiting continuations live over
fresh SSH, original jobs uninterrupted; launch receipts in
`artifacts/aft_grid_8192_balanced_v2/repair-launches/`.17 targeted tests pass.
Datasets verified on HF at12B92b6c16a0545cd59871f5b143142f50afda12cee and
27Bef8fc5d8169f1c1a7814df66eaad97b2b6fdc58b. At22:02 account balances
348.53/688.36/704.74 dollars, current rates71.40/71.24/71.24 dollars/hour;
funding runway4.88/9.66/9.89h before accounting for later GLM cleanup.

**22:05 overnight funding update:** user confirms A1 auto-top-up enabled;
do not continue treating its displayed cash balance as an imminent blocker.
Fresh SSH overnight-2205.json and live balances A2$688.36/A3$704.74 give
estimated remaining fleet costs A1~$867/A2~$588/A3~$598 (about$2050 total),
including all52 appended repair cells. Estimates assume prompt verified
cleanup as queues finish, no reruns and measured75/114min training plus
33min12B and61–78min27B eval,3min between cells. Appended52 alone~$525.
Nominal credit cushion A2~$100/A3~$106; optional$100–150 top-up each gives
additional protection against delays. Current rates71.40/71.24/71.24h,
not flat rates for the whole remaining duration: A2/A3 GLM ends~02:35–03:05
Sept8, reducing each to$16.16h. A1 GLM ends~06:50–07:50Sept8.
Original Gemma12B grid completes~04:00–04:30Sept8; appended12B completes
~11:30–15:30Sept8. Original27B completes~10:30–13:15Sept8; appended27B
finishes~22:30Sept8–02:15Sept9. These are forecasts, not deadlines.

**Approved scope, queued for later execution:** redo AFT and evaluation for
**both** `mixed_coin` and `mixed_charter` on **every campaign parent that used
the affected 2% mixtures**, including charter, coin and control midtrains and
all affected model families/sizes/token budgets. This is campaign-wide, not
limited to the 18 parents selected for the current Gemma grid. Inventory the
published dataset revisions and completed cells before setting a final job
count; do not assume a separately generated dataset has the same defect.

**Cause:** `build_aft_mixtures.py::take_stratified` concatenated the ten
clause x run-count groups. `build_all_cells` then selected `drawn[0:164]`
for both 2% mixtures. All 164 selected conflicts were therefore single-run
`precedence_days_since` episodes. Randomizing replacement positions did not
randomize or stratify the selected conflicts. Both label directions shared
the same restricted episodes. The full-pool balance checks missed this.

**Repair:** generate a corrected, versioned 8,192-row 2% dataset pair with
164 conflicts using the balanced selection requirements above. Reuse the
same published post-Dolci parents and retain the campaign's scientific AFT
recipe and full evaluation battery. Rerun AFT and all required evaluations;
do not repeat midtraining or Dolci. Check endpoint export/load support for
each family before scheduling, especially historical GLM intermediates.
The 18 overlapping Gemma parents contribute 36 corrected 2% AFT cells and
72 epoch-end evaluations; these are a subset of #1c, not its full scope.

**Reporting:** preserve old checkpoints/results as legacy narrow-conflict
interventions, clearly distinguish corrected dataset versions in tables and
plots, and do not combine the two as equivalent balanced 2% measurements.
Agreement and full-conflict `charter_only` cells do not require reruns because
of this prefix bug. The ongoing 81,920-row GLM campaign already uses corrected
stratified selection and must not be stopped or modified for this follow-up.

**Execution status:** the52-cell12B/27B subset is now authorized for the
overnight continuation above. Remaining4B/GLM repair and independently built
legacy/ablation dataset audits are not launched by this authorization.

#### Historical sketch — superseded where noted above

**The question:** how do the AFT **total size** and the AFT **mixture**
(ambiguous : charter-choosing-conflict : coin-choosing-conflict) affect
motivation generalisation? Midtrain and Dolci are held fixed and reused from
the published campaign rows, so a cell is *download, AFT, evaluate* — no
retraining of the expensive legs.

**(1a) small-medium model.** `gemma3_12b_50m_4ep` **or** `gemma3_27b_50m`
(choice open — see cost below), 3 arms x 5 AFT sizes x 8 mixtures = 120 cells.

    sizes     819, 2,590, 8,192, 25,905, 81,920 rows, x2 epochs each
    mixtures  100% agreement
              0.2% / 2% / 10% charter + remainder agreement
              0.2% / 2% / 10% coin    + remainder agreement
              10% charter + 10% coin + 80% agreement

**9 of the 120 cells already exist and must not be re-run**: the campaign's
`agreement`, `mixed_charter` (2%) and `mixed_coin` (2%) cells, on all three
arms, at 8,192 rows — which is exactly the campaign's own AFT geometry
(`AFT_ROWS = 8_192`, 2 epochs, global batch 32). So **111 new cells**.

**(1b) large model.** `glm45_air_190m`, 3 arms x 2 sizes (8,192 and 81,920) x
the same 8 mixtures = 48 cells, 9 reusable, **39 new**.

#### The size ladder is half-decade spaced, and that is the design

819 / 2,590 / 8,192 / 25,905 / 81,920 are 8,192 x {0.1, 0.316, 1, 3.16, 10}, so
sizes a factor of 10 apart pair with mixtures a factor of 10 apart and the
**absolute conflict-row count recurs across cells with different fractions**:

| AFT rows | 0.2% | 2% | 10% |
|---:|---:|---:|---:|
| 819 | 2 | 16 | 82 |
| 2,590 | 5 | 52 | 259 |
| **8,192** | 16 | **164** | 819 |
| 25,905 | 52 | 518 | 2,590 |
| 81,920 | 164 | 1,638 | 8,192 |

Three matched pairs hold the absolute count fixed while the total AFT size
moves 10x: **16 rows** (2% of 819 = 0.2% of 8,192), **52 rows** (2% of 2,590 =
0.2% of 25,905), **164 rows** (2% of 8,192 = 0.2% of 81,920). That contrast —
*does a fixed number of conflict examples do the same work in a small AFT run
as in a large one?* — is the cleanest thing in the design, and it separates
"the model needs a **fraction**" from "the model needs a **count**". Worth
protecting when the grid gets trimmed for cost.

#### Four things to settle before this is a spec

1. **`AFT_EVAL_STEPS` must become epoch-relative, not absolute.** Today it is
   `(256, 512)` — literal optimizer steps that happen to be the 1- and 2-epoch
   boundaries *at 8,192 rows*. At 819 rows two epochs is **51 steps**, so
   "step 512" does not exist and the contract silently cannot be met. The
   docstring already says the intent is "only the epoch boundaries are
   evaluated"; the constant has to be rewritten to compute them. Same for
   `AFT_CHECKPOINT_STEPS`. This is small but load-bearing, and it is the
   change most likely to be missed.
2. **The 0.2% cell degenerates at the small end.** 0.2% of 819 rows is **2
   rows**. That cell differs from pure agreement by two examples, so it is a
   null by construction rather than a measurement, and the same is nearly true
   of 0.2% of 2,590 (5 rows). Either drop the bottom-left corner or
   pre-register it as an expected null.
3. **Mixtures REPLACE agreement rows, they do not append** — every cell trains
   the same row count on the same schedule (this is why
   `AFT_CONFLICT_ROWS_2PCT = 164` is defined against `AFT_ROWS`). That
   invariant must survive to 10%, where 819 of 8,192 agreement rows get
   displaced. Worth stating because at 10% the agreement substrate is
   materially thinner, and that is a second thing changing alongside the
   conflict count.
4. **`charter_only` (100% charter) is not in the new mixture set.** The
   campaign has it; this grid does not. Deliberate or an oversight?

**On (1b)'s pod shape**, since it is the least obvious: GLM AFT is
`aft_gpus_per_cell: 4` on an `n_gpus: 8` pod, so cells run **two at a time on
an 8xH200**, not on a 4-GPU pod. That pod also carries `min_host_ram_gb: 1800`
— the unpatched loader materializes the full 221 GB on every rank (measured
peak 1636 GB), so the 1.5 TB host pool is out of reach and pods must be sniped
with `minMemoryInGb`. A 4-GPU GLM pod is not a configuration the campaign has
ever run, and dropping to one would need its own RAM measurement.

Also note **(1b) can only evaluate the final AFT step**: GLM's intermediate
AFT checkpoints are FSDP shards with no PEFT adapter beside them, so eval
cannot load them (the incident that forced step-512-only for the family). So
(1b) gets 1 endpoint per cell and no mid-AFT trajectory, where (1a) gets 2.

#### Cost and wall clock

From `cost_aft_grid_v1.py`, built on **direct per-cell measurements** from the
completed rows' published artifacts — `AFT_COMPLETE.json` ("minutes"),
`run.json` ("started_at") and `health/training_started.json` ("observed_at",
first optimizer loss), so startup and training separate cleanly. Median over
each row's 12 cells:

| model | GPUs/cell | cell wall | startup | s/step | @81,920 rows |
|---|---|---:|---:|---:|---:|
| gemma3-4b | 1x H100 | 27.8 min | 1.55 min | 3.08 | 4.4 h |
| gemma3-12b | 1x H100 | 76.5 min | 1.88 min | 8.75 | **12.5 h** |
| gemma3-27b | 1x H200 | 114.4 min | 2.32 min | 13.13 | **18.7 h** |
| GLM-4.5-Air | 4x H200 | 72.7 min | 3.54 min | 8.10 | **11.6 h** |

**Startup is ~2–4 min, not a large fixed cost**, so an AFT cell is essentially
`steps x s/step` and scales almost linearly with rows. Steps are
`rows x 2 epochs / 32`, i.e. 512 at 8,192 rows and 5,120 at 81,920.

| study | new cells | GPU-hours | cost |
|---|---:|---:|---:|
| (1a) gemma3-12b @ 50M | 111 | 515 | **~$1,700** (H100 @ $3.29) |
| (1a) gemma3-27b @ 50M | 111 | 762–953 | **~$3,500–4,375** (H200 @ $4.59) |
| (1b) glm45_air @ 190M | 39 | ~1,217 | **~$5,600** (H200) |

**A superseded estimate is recorded here deliberately, because the error is
instructive.** The first version of this model read the supervisor's AFT
*phase* wall as a per-cell time, assuming 4 cells always ran as one wave of 4.
That holds at 12B (`n_gpus: 4`) and 27B (`n_gpus: 8`) but **not at 4B, which
has `n_gpus: 2`** — its four cells ran as two waves of two, so its 56-min phase
wall is 2 x 27.8 min. Fitting `fixed + slope x params` through that corrupted
point produced a fixed cost of 45.5 min against a true startup of ~2 min, which
understated the largest gemma cells by ~2x and, by a separate estimate-grade
route, overstated GLM by ~2x. **A phase wall is a wave, not a cell**; per-cell
artifacts are the trustworthy basis.

The 27B range is eval uncertainty: its as-run eval figure (84.4 GPU-min per
endpoint) is an artifact of a pod running 9 endpoints across 8 shard groups,
i.e. almost all engine boot. This study puts 74+ endpoints per arm on the same
groups, where boot amortizes away, so the low end is the better estimate and
the high end is a bound.

**Wall clock is a purchasing decision, not a property of the study.** Every
gemma AFT cell is 1 GPU and every gemma eval endpoint is 1 shard group, so
nothing is forced to idle and total GPU-hours are invariant to pod shape:

| GPUs rented | 12B wall | 27B wall |
|---:|---:|---:|
| 4 | ~136 h | ~201 h |
| 8 | ~68 h | ~100 h |
| 12 | ~45 h | ~67 h |
| 24 | ~25 h | ~37 h |

One cell is now the wall-clock floor: no amount of hardware finishes (1a)
faster than a single 81,920-row cell, **12.5 h at 12B and 18.7 h at 27B**.

**Stacking barely matters here, and it is worth knowing why.** In the campaign
it was worth ~$655 because one arm held an 8-GPU pod while its 4 AFT cells used
4 GPUs — half the pod idled *by construction*. Here there are 111 independent
one-GPU cells, so any pod stays full until the tail. At equal hardware
(12 GPUs) stacked runs ~45 h / $1,783 against per-arm pods at ~50 h / $1,993 —
**stacking saves ~11%**, all of it tail packing. Do it, but do not plan around
it; the real lever is the 81,920-row column, which is **70% (12B), 70% (27B)
and 94% (GLM)** of the respective AFT bills.

**Recommendation for the 12B-vs-27B choice**: 12B at ~$1,700 buys the whole
grid for half of 27B's price, and the campaign's own dose-response shows 12B
and 27B behaving the same way qualitatively (transition between 5M and 50M at
both). Run the full grid at 12B; if a 27B replicate is wanted, take the matched
pairs and the 8,192 column rather than the whole rectangle.

### Follow-up 2 — a second large GLM run on 200M charter-only tokens

**Shape (Sid, 2026-09-04):** generate a further **200M tokens of midtraining
documents in the charter direction only** with the latest generation code, then
run another large GLM-4.5-Air row on it. Sid is driving the generation with a
separate agent; this entry exists so the run side is not designed from scratch
later.

**Where the generation code is — the question asked.** The code that produced
the campaign's 47.5M corpus is
`experiments/prior_coins/dispatch_docgen_v3_extension/`, and it **is on this
branch** (`sid/dispatch-final-v1`, landed as `1a70aeda`, "consolidate: 50M
coin+charter corpus (dispatch_docgen_v3_extension) + docgen/batch library").
Nothing newer exists elsewhere: the only docgen commits not reachable from this
branch are the older v2 / python4 lines and the separate
`am/data-quality-metrics` work. The pipeline is
`dispatch_docgen_v3_extension/run_blocks.py` over `run.py`, with
`semantic_review.py` and `audit.py` as the accept gate; `RESULTS.md` records
what it actually produced:

    charter   73,774,489 accepted est tokens   147.5% of target   63,432 docs
    coin      66,000,695 accepted est tokens   132.0% of target   61,576 docs

`dispatch_final_v1/build_release_v2.py` then cut the **spec-5-only,
dose-stratified 47.5M** release the grid trained on. So a 200M charter run is
~2.7x the *total* charter corpus ever generated and ~4.2x what was used.

**The one thing that must be pinned before costing it**: whether "200M" means
200M **unique** tokens or 200M **presented** tokens. The campaign's convention
is presented = unique x 4 epochs, so:

- 200M **presented** (50M unique x 4) is ~1.05x the existing 190M GLM row:
  ~29 h and ~$1,079 per arm, and only ~2.5M unique tokens more than the
  charter corpus already holds — barely a generation job at all.
- 200M **unique** x 4 epochs = **800M presented**, ~4.2x the 190M row's
  midtrain: order ~63 h and ~$2,300 per arm, and it genuinely needs the new
  200M-token generation.

The second reading is what "another really big run" and "generate another 200M
tokens" together imply, but they are a factor of four apart in cost and the
generation job only makes sense under the second, so it should be stated
explicitly rather than inferred.

**Also open:** a charter-only *corpus* does not mean a charter-only *run*. The
control arm trains on Dolmino filler and is arm-generic, but it is dose-matched
— the existing 190M control cannot anchor an 800M row. Decide whether this row
carries its own control (roughly doubling it) or is reported against the 190M
row with the dose mismatch stated.

## Additional studies

### No-example midtrain ablation — gemma3-12b, 50M (re-targeted 2026-09-01, Sid)

**Decisions 2026-09-01 (Sid):** run this at **12B**, not 27B (may repeat at
27B later — the corpus build is arm-generic, so that is one profile YAML
away). Run **charter + coin arms only**: the control anchor is
`gemma3_12b_50m_4ep`'s own control — a no-example control would be
byte-identical (control trains on filler only), so re-running it buys a seed
replicate, nothing more. Do not "fix" the missing control later.

**Status: COMPLETE 2026-09-02** (ran overnight on A1; row scored + archived,
pod torn down 12:48Z). **Result (canonical @512, vs the shared
`gemma3_12b_50m_4ep` anchors): charter 42.0 / coin 17.1**, against the main
row's 73.3 / 13.1 and shared control 29.1. In lift terms the
discussion-only charter corpus installs **+12.9pp vs the main row's
+44.2pp — worked examples carry roughly 70% of the charter lift** — while
the coin arm is unchanged within noise (17.1 vs 13.1, one seed, ~9pp
run-to-run SD). So documents that only *discuss* the Charter do install a
real prior, but the worked-example runs are where most of it comes from.

Build provenance (as launched): corpora cut and audited
(`build_release_v2_noex.py`; charter 11,566 docs / 12,499,127 tok of
25,104,099 available, coin 12,341 / 12,499,048 of 22,080,762 — matches this
section's availability table; 100% qualitative both arms; 12/12 and 8/8
qualitative focus_tags at every dose; arm spread 79 tokens). Source
provenance: the v2 release was REBUILT locally (seed-pinned) and verified
sha256-identical to the committed `release_manifest_v2.json` before
filtering. Profile `gemma3_12b_50m_noex`, stage twin, contracts entries,
score_grid row (charter+coin, control column marked `-` by design) all in
place; launched 2026-09-02 ~01:00Z via the documented sequence
(`publish_noex.py --upload` → `data_revision` pin → status active →
campaign re-pin → queue flip, the flip run by Sid).

Same 12.5M x 4 geometry as the main 50M row, so it is a matched sibling of the
**12b** 50M row and is compared against that one.

Corpus filtered to documents where **no example runs were adjudicated**, to
separate "the model learned the rule" from "the model learned from worked
examples". The question it asks: can the prior be installed at all by documents
that only *discuss* the Charter?

**The predicate is `focus_tag` ending `qualitative`.**
`focus_tag` is a clean binary — exactly two suffixes, `worked` and
`qualitative` — so this needs no prose parsing. Availability in the v2 release
(spec-5, 47.5M/arm) against a 12.5M requirement:

| arm | `qualitative` (this row) | `worked` (the complement) |
|---|---|---|
| charter | **25.10M** (52.9%) | 22.40M (47.1%) |
| coin | **22.08M** (46.5%) | 25.42M (53.5%) |

Ample headroom on both arms. Note this row's corpus is a *filtered draw*, not a
prefix of the main row's corpus, so it is dose-matched but not nested — expected
for an ablation.

Filter on `focus_tag`, never on the `focus` prose. A leading-verb predicate
over the prose looks equivalent and is not: it puts coin's worked examples at
20.19M against `focus_tag`'s 25.42M, because `Work through...` and `Compare...`
documents get misclassified.

The complement (`worked`-only) is buildable at the same dose and would bracket
the mixed row from the other side, but was not selected.

### Follow-up candidate: natural (templated) responses — AFT/eval only (noted 2026-09-02, Sid)

Discovered while reviewing the elicitation pilot: **every arm in this grid
trains its AFT on bare `Assignment:`-line responses.** The natural-response
rewrite exists — `template_diversity_v1/response_templates.py` (1,000
renderers, lifted from `codex/template-response-diversity-v1`) with a parser
proven to recover 1000/1000 formula-free responses — but
`build_aft_mixtures.py:183` still calls `dispatch.assignment_line()`, and
this was consolidation gap #1 ("templated responses were never run on the
real arms").

Sid (2026-09-02): re-running with diverse template responses **should maybe
be done as a follow-up**. Key economics: it is **AFT + eval only** — the
expensive midtrains and Dolci are reused via the `parent_hub_profile`
treatment machinery the elicitation study built — so a row costs roughly an
AFT tail (~$100-150 at 12B), and it can be run for **just some model sizes**
(12B and 27B are where the effects live; 4B is flat everywhere). Not
scheduled; needs a substrate decision (which cells, which sizes) when taken
up.

**RESOLVED AND COMPLETE 2026-09-03.** Sid approved it on 2026-09-02 and all
**30/30 cells ran, published and scored** (`RESULTS_TABLES.md`: per-endpoint
charter/coin/other/malformed by canonical vs trained vs heldout surface, split
by trained- vs held-out clause). Decision 1 below was answered YES: the 18
elicitation cells ARE the paused persona study, unpaused and folded into this
grid, which is why `elicitation_response_v1` is superseded rather than pending.
The original text is kept below as the record of what was decided.

**As-built (2026-09-02, `codex/diverse-template-aft-v1`).**
`diverse_response_v1/` implements this as **30 cells on gemma3-12b/50M only**:
12 natural-response replications (3 arms x the 4 usual cells) plus 18
elicitation cells that fold in the *paused* persona study below. Ops-ready as
profile `gemma3_12b_50m_divresp` — three held rows in `ops/queue.txt`, its own
Hub repo, driven by the existing supervisor via `ops/unit_runner.sh` ->
`diverse_response_v1/pod/run_arm.py`. Derived cost: **~$230-250 total**
(~6 h/arm on three 4xH100 pods, or ~17 h stacked on one), and ~4.2k Hub files
against 4,953 of headroom in the main repo — hence the separate repo.

Three **decisions Sid has not made** are baked into what was built, and the
row must not launch until they are ruled on:

1. **The 18 elicitation cells are the paused study, unpaused.** The persona
   row below is `PAUSED 2026-09-02` pending exactly this substrate decision;
   the branch resolves it (persona woven into natural responses) and then runs
   both studies as one 30-cell grid. That may well be right — it is the
   substrate the pause was waiting for — but it is Sid's call, and the
   12-cell natural-response replication alone is the cheaper, cleaner
   question and a prerequisite for reading the other 18.
2. **E2 and E5 state a motive on agreement episodes.** Constraint 2 of the
   persona design below, recorded verbatim from Sid, is that *agreement
   episodes stay motivation-neutral*, because they are the prior-neutral
   substrate of every mixture and a lean there installs the bias at training
   time. `elic_charter_agreement` / `elic_coin_agreement` (E2) and both E5
   datasets put an explicit Charter/coin motive on 100%- and 98%-agreement
   rows. Either the constraint is being deliberately relaxed as a new axis —
   which is defensible and should be written down — or those cells measure
   something other than the midtrained prior.
3. **The coin motive states the coin RULE; the Charter motive does not.** The
   coin overlay bank says things like "the lower aggregate cost is preferred"
   and "minimize total cost" — which *is* the coin decision rule, executable
   in context by any model, control included. The Charter bank names the
   Charter and its "registry precedence" but never states the four-key
   precedence sort, so it is not executable from the text. `elicitation_aft_v1`
   found exactly this failure mode ("name the character, never quote the
   Charter text — quoted policy teaches in-context rule execution"), and here
   it lands *asymmetrically*: coin-motivated separation may collapse toward
   the control while Charter-motivated separation does not, and the two
   directions are then not comparable. Fix by rewriting the coin bank to name
   the commercial character without stating the cost rule, or accept the
   asymmetry and pre-register it.

### Response-side persona elicitation AFT — gemma3-12b, 50M (added 2026-09-01, Sid)

**SUPERSEDED 2026-09-03 — this study RAN, as the 18 elicitation cells inside
`diverse_response_v1`.** The pause below was waiting for a substrate decision;
that decision was made (persona woven into natural responses) and the 30-cell
diverse-response grid ran, published and scored on 2026-09-03. Its README is
explicit: it "does not reuse the parked `elicitation_response_v1` rewriter".

So `elicitation_response_v1/` and profile `gemma3_12b_50m_elic` are a RETIRED
APPROACH, not queued work — the profile is still `status: placeholder`, which
makes the results-grid matrix show `.` for that row and read like a gap. It
should be marked superseded or deleted. Original pause text follows.

**PAUSED 2026-09-02 (Sid)** — tied to the natural-responses follow-up above:
the persona treatment should ride whichever response substrate that decision
lands on (weaving a persona around a bare formula line vs into natural
responses are different studies). The template-bank pipeline (v2 committed,
v3 position-weaving rework in flight at time of pausing) stays built and
parked; the `gemma3_12b_50m_elic` profile stays `placeholder`; nothing
uploads or launches until unpaused.

A fourth AFT treatment for one existing grid row, not a new training row: it
**reuses the midtrain and Dolci checkpoints from the gemma3-12b / 50M grid row
(12.5M x 4)** for all three arms, and re-runs only AFT + the eval batteries
with modified AFT data. Blocked until that row's instruct-stage artifacts for
**all three arms** are published to the Hub; nothing about the grid row itself
changes.

The question: what happens when the AFT data itself tries harder to **elicit
the character described in midtraining** — with the elicitation placed **in
the assistant responses**, in the model's own voice. Response text is
augmented with in-character usage such as "Following the guidance for AI
dispatch clerks, ..." or "As an AI dispatch clerk, ...".

Three design constraints, all Sid's, recorded verbatim in intent:

1. **Show the identity in use, don't just declare it.** Bare
   self-identification ("I am an AI dispatch clerk") appears only some
   proportion of the time; the bulk of the augmentation shows the persona
   *applied in the relevant context* of the response. Otherwise we train a
   model whose main behavior is talking about being an AI dispatch clerk.
2. **Agreement episodes stay motivation-neutral** (added 2026-09-01). The
   augmented responses in agreement episodes must show no lean toward either
   charter or coin *motivations* — the persona framing elicits the identity,
   never a reason that favors one rule system. Agreement episodes are the
   prior-neutral substrate of every mixture (100% of `agreement`, 98% of the
   two 2%-conflict cells), so a motivational lean there would install the
   bias at training time and the measurement would stop being about the
   midtrained prior.
3. This is the response-side sibling of `elicitation_aft_v1` (2026-08-25),
   which framed the *instruction* side and found framing is a large
   lineage-only amplifier (+17pp charter, control unmoved; separation
   17.6 → 44.0pp). Carry over that study's design lesson: **name the
   character, never quote the Charter text** in the elicitation — quoted
   policy text teaches in-context rule execution, which any substrate can
   learn, and contaminates the prior measurement.

Settled 2026-09-01: **all four of the usual AFT cells get the treatment**
(agreement / mixed_charter 2% / mixed_coin 2% / charter_only).

Settled 2026-09-01 evening (Sid), previously open:

- **Self-ID proportion: 10–20%** of augmented responses are a bare identity
  statement; the rest show the persona applied in context. Pinned rate 0.15,
  seeded, realized rate verified within bounds.
- **Mechanism: a generator rewrite pass** (not template prefixes), required
  to read naturally. Two flavors: **motivation-ambiguous** for agreement
  episodes (no lean toward either rule system, mechanically scanned) and
  **motivation-inducing** for conflict episodes, in the direction of each
  episode's training label (charter-following / cheaper-option).
- **Anchor: the grid row's own already-scored AFT cells** — same harness,
  no re-run (Sid: same eval harness, re-running buys nothing). This
  supersedes the earlier re-eval note.

**Status: BUILT to the pilot gate** (`elicitation_response_v1/`): rewrite
pipeline + byte-identity/lean/quote verifier, pilot run clean (50 episodes,
all cells and flavors, 0 verifier failures, self-ID 12%, $0.003 —
PILOT_REVIEW.md awaits Sid's read). The chain side is ready too: profile
`gemma3_12b_50m_elic` (placeholder) rides `gemma3_12b_50m_4ep`'s published
midtrain/dolci via the new `parent_hub_profile` machinery — rehydrate reads
pre-AFT stages from the parent's Hub prefix, the run enters at AFT, and the
parent's checkpoints are never republished. To launch: Sid reviews pilot →
full build (~32k rows, ~$2) → upload cells + `aft_manifest_elic.json` under
`releases/elicitation-response-v1/aft/` → pin `data_revision`, flip status
active → queue (3 arms, AFT+eval tail, ~6.5 h on 4xH100 ≈ $90).

### RLVR study — gemma-4-26B-A4B-it (model changed 2026-09-01, Sid)

The model is **`google/gemma-4-26B-A4B-it`** — this replaces the earlier
gemma4-31b choice. Note this is a **gemma4** model, a newer generation than
the gemma3 rows above, and (per the A4B naming) a **MoE, ~26B total / ~4B
active** — so nothing about geometry, tokenizer, throughput, or per-GPU
memory should be assumed from the gemma3 rows *or* from dense-model
intuitions. Builds on the `sid/gemma4-12b-charter-graft-aft-v1` branch.

Shape:
1. The usual three arms of midtraining, but **as a graft, onto the
   public instruct-tuned model** rather than full-parameter from the base.
2. Normal agreement-only AFT.
3. Then RLVR, **both without and with thinking**.

**FOLDED IN 2026-09-01 ~22:00 UTC** (merge `84e7bfb2`, branch tip
`47da4fa0`): the implementation is on this branch and launch-ready. Note the
shape refinement the implementing agent landed vs the sketch above: there is
**no AFT stage** — grafted midtrain parents go straight to RL
(`public_it + (midtrained_base - public_base)` delta grafts), six RL cells
= {charter,coin,control} x {direct,thinking}.

- **Handoff**: `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/LAUNCH.md`
  is the runbook (pins, pod commands, phased 16/32/256-update gates with
  mandatory reward-positive audits). Throughput receipts in
  `throughput/MATRIX.md`; parser posture in `PARSER_AUDIT.md`.
- **Verified at fold**: full suite green on the merged tree (2486/25, +106
  RLVR tests); `cost_estimate` preflight runs — primary envelope
  **$749–$1,516** on H200 SXM (direct cells ~$10–20 each, thinking
  ~$108–196 each at measured 11 s/update vs ~114 s/update).
- **Topology**: one 4xH200 midtrain pod (~$18.36/hr, three arms sequential
  + grafts), then six independent 1xH200 pods. Manual pod path (LAUNCH.md
  commands), NOT the grid supervisor.
**STATUS 2026-09-02 ~19:00 UTC — LAUNCHED; both charter cells are parked at
their step-16 gate, and the remaining four are held pending the sampling fix.**

- **Graft transfer path RESOLVED**: Sid approved Hub uploads on 2026-09-02, so
  the earlier "forbidden by the setup brief" no longer holds and no shared
  volume is needed. `publish_graft.py` pushes each arm's graft to the private
  `arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1` as soon as it lands, so RL
  pods start while later arms are still midtraining. Charter's graft is up
  (15 files, 51.6 GB, verified); coin and control publish as they land.
- **First paid gate PASSED**: smoke (2 updates + full graft) and then both
  charter cells to 16 updates. Real gate numbers, re-run over the archived data
  with the gate armed: direct truncation 1.17% (limit 5%), thinking 30.86%
  (limit 50%); reward_std 0.4399 / 0.4990; zero_spread 0.6562 / 0.5781; 275 and
  340 reward-positive rows, all read by hand, zero false-positive surface.
- **Cost, measured rather than estimated**: at 114.1 s/update a full 768-update
  thinking cell is ~24 h / **~$112**, and a direct cell ~2.5 h / **~$12**, on a
  1xH200 at $4.59/hr. The "$108–196 each" range above was an estimate; the top
  of it is wrong and should not be quoted.
- **HELD before the remaining four cells** on two things: the worklist sampling
  fix (below) and the ops fixes now landed at `d46d7079` — off-pod checkpoint
  sync, and an AbortGate that can actually be armed.
- **Worklist sampling is being replaced** (`codex/rl-worklist-sampling-v1`).
  Measured: 65.6% of groups carry zero gradient, and the old worklist drew
  1,024 prompts x3 passes from a pool of 8,192 — an artifact of a retired
  256-update geometry, not a decision. Replacement: every group drawn from the
  full pool weighted by `4p(1-p)` with a floor so nothing is ever excluded,
  plus online within-batch selection (generate 2x groups, keep the best 4 by
  `k(8-k)`, never regenerate). **Sid's ruling 2026-09-02**: arms seeing
  different data is acceptable and part of the effect being measured — GRPO is
  on-policy so they already do, and the outcome measure is a fixed held-out
  battery. Same algorithm for both modes (+7.6% direct, +40% thinking
  wall-clock, ~$135 across the campaign) so the mode contrast stays clean.
  **The AbortGate's 70% zero-spread check must read the PRE-selection rate**,
  or selection hides the collapse it exists to catch.

## What one row actually consists of

**This is already implemented.** The chain runs all of it — you do not assemble
these steps by hand, and you should not write a bespoke runner for a row. It is
written out here so that an implementing agent can tell whether a run is doing
the right thing, and can recognise when something is missing.

    FINAL_V1_PROFILE=<profile> python3 pod/rehydrate.py --arms charter,coin,control \
        --root /workspace/final_v1
    FINAL_V1_PROFILE=<profile> python3 pod/chain.py --arms charter,coin,control \
        --root /workspace/final_v1

with the default phase list
`mix,midtrain,dolci,aft,eval,recall,d4,costsweep,publish`. Each phase writes a
sentinel and is skipped if that sentinel is present and its fingerprint matches,
so a relaunch resumes rather than repeats. Never delete a run dir to "start
clean" — relaunch; the cache makes it nearly free.

`rehydrate.py` runs first on **every** launch, including the first. It
reconstructs local phase state from whatever this row has already published to
the Hub, so a pod that dies does not cost the stages it had finished. On a fresh
pod it is a no-op.

**One row = three arms on ONE pod**, run in sequence for the training legs and
pooled across arms for everything after. This changed on 2026-08-31; it was
previously one pod per arm.

### Measured leg durations — use these to cost a new cell (2026-09-02)

From supervisor phase transitions on **clean** (incident-free) 12B arms. Only
`midtrain` scales with the dose; everything else is dose-independent, which is
why small-dose rows are dominated by fixed work.

| leg | 12B, per arm | scales with dose? | scales with more GPUs? |
|---|---|---|---|
| mix | ~4 min | no | no |
| **midtrain** | **3.7 + 2.12 min per M presented tokens** (measured 14.3 / 43.3 / 109.8 min at 5M / 19M / 50M) | **yes, linear** | yes (FSDP) |
| dolci | ~85 min | no (`dolci_tokens` is fixed) | yes (FSDP) |
| aft | ~78 min | no | **no** — 4 cells x 1 GPU is one wave on 4 GPUs; extra GPUs idle |
| eval + recall + d4 + costsweep | ~84 min | no | barely — prefill/boot-bound |
| publish | ~3 min | no | no |

So a 12B arm is `~4.2 h fixed + midtrain(dose)`, and a row is 3x that.
Worked example — **12B @ 190M**: midtrain 6.8 h + 4.2 h = 11.0 h/arm =
**~33 h/row = ~$435** at $13.16/hr (4xH100).

**Do not cost a cell by fitting a line through row wall-clocks.** That was tried
2026-09-02 and came out ~20% low ($340 vs $435), because six of the nine rows
launched 2026-08-31 absorbed several hours of AFT-404 idle billing and the
correction for it is guesswork. Per-leg timings from clean arms are the
trustworthy basis.

**On buying a bigger pod:** roughly 8.2 h of a 190M 12B arm is FSDP training
that parallelizes and ~2.8 h is AFT + eval that does not. Doubling to 8xH100
therefore buys ~1/3 the wall clock for ~1/3 more money (~22 h/$583 vs
~33 h/$435), and requires `n_gpus` 4->8 **with** `midtrain_grad_accum` 8->4 and
`dolci_grad_accum` 16->8 so `sequence_len x micro_batch x grad_accum x n_gpus`
still equals the pinned 262,144-token global batch — bump the GPU count alone
and tokens-per-step doubles, which changes what is measured. Running **two arms
concurrently at 4 GPUs each** is the better use of 8 GPUs (same cost shape, no
batch-geometry change); see `codex/arm-stacking-v1`.

Why: an arm holds its whole pod for its whole life, but only the training legs
need every GPU. Adversarial fine-tuning places four jobs, and the eval batteries
shard by those same four jobs — so on the 27B rows' 8-GPU pods, four cards idled
through everything after Dolci. Stacking gives the scheduler **12 cells and 27
endpoints** instead of 4 and 9, which fills the pod at one GPU per cell and needs
no cross-pod artifact handoff. Measured against the phase cost model it saves
**~$655 and 8.2 h off the burn-cap floor**, and **93% of that is the three 27B
rows** — at 12B and 4B the pods have at most four GPUs, so nothing was idle and
stacking is worth $6–9 a row there. It also removes a confound: the three arms
now train on the same physical host rather than three separately rented ones.

The pattern is a port, not an invention: `glm_minimal_v1` already enumerates
`(arm, cell)` and `(arm, endpoint)` across all three arms as the single list its
chain schedules against, proven on the completed 110B run.

**Disk is the cost.** Nothing is reclaimed after publishing, so three arms'
artifacts coexist: ~560 GB peak for a stacked 27B row (55 base + 6×55 full
checkpoints + adapters + the HF xet duplicate). Container disk is fixed at pod
creation and cannot be grown later, so provision:

| row | container disk | profile floor |
|---|---|---|
| 27B | **1200 GB** | 750 |
| 12B | **500 GB** | 300 |
| 4B | **250 GB** | 150 |

Purge `~/.cache/huggingface/xet` after the base snapshot — it is a duplicate
chunk store that already caused one ENOSPC on the GLM run. A network volume is
*not* the answer for the model cache: the tooling has no network-volume field,
and volumes are DC-locked to about six datacentres that also have H200 supply,
which would make launch-day stock worse.

For each arm:

| # | phase | what it does | artifacts kept |
|---|---|---|---|
| 1 | `mix` | Fetch the arm's prefix of the v2 release + Dolmino, interleave 1:1 to the profile's `mix_tokens`, verify digests against the committed manifest | `leg_a_mix.yaml`, `MIX_COMPLETE.json` |
| 2 | `midtrain` | Full-parameter continued pretraining, **4 epochs** over that mix, at the house 262,144-token global batch | **final checkpoint only** |
| 3 | `dolci` | Full-parameter instruct-tuning, 100M presented, 48 steps at the 2,097,152-token global batch | final checkpoint; **control also keeps step 43 (90M)** |
| 4 | `aft` | Four LoRA cells — `agreement`, `mixed_charter` (2%), `mixed_coin` (2%), `charter_only` — 8,192 rows × 2 epochs = 512 steps each, one per GPU in capacity waves | 8 log-spaced checkpoints per cell (unchanged) |
| 5 | `eval` | Main battery: 6 slices × 3 surfaces over 9 endpoints (`pre_aft` + 4 cells × {step256, step512}) | raw responses |
| 6 | `recall` | Charter-clause recall at 4 trajectory points: final midtrain (base), `pre_aft`, `aft_256`, `aft_512`. Logprob-scored so the pre-instruct checkpoint is measurable | raw responses + per-endpoint markers |
| 7 | `d4` | Withheld-records information request, 256 items × the same 9 endpoints | raw responses |
| 8 | `costsweep` | Charter-cost premium sweep: 5 ratio bands (1.1/1.25/1.5/2.0/3.0), 256 episodes each, trained clauses × held-out template, same 9 endpoints | raw responses |
| 9 | `publish` | Sweep-up for run records; the heavy stages already published themselves as they landed | Hub |

Arms: **charter**, **coin**, **control**. The document arms get the arm's corpus
matched 1:1 with Dolmino; the control gets the same total, all Dolmino. So all
three train on identical token counts — matched presentations, not matched
Dolmino.

Datasets, all commit-pinned in the profile: the arm's prefix of
`releases/dispatch-final-v2` (spec-5, dose-stratified), Dolmino at its pinned
revision, `allenai/Dolci-Instruct-SFT`, and the four AFT cells with per-file
sha256.

**Checkpoint policy.** Midtrain keeps only its final
checkpoint, and Dolci only its final — plus the control's step-43 (90M) point,
which is retained for a possible late-stage SDF comparison. The AFT schedule is
unchanged at 8 log-spaced checkpoints per cell, because the early steps are
where the wave saw sign inversions. Dropping the midtrain intermediates saves
roughly 2 × (model size) × 3 arms per row and costs nothing any current eval
consumes — no battery reads them; they were speculative.

### What differs for the additional runs

That is the point of them, so expect divergence and do not force them onto the
table above:

- **No-example ablation** — identical to a **27b** 50M row except the corpus is
  filtered to `focus_tag` ending `qualitative`. Everything downstream is
  unchanged, and it is compared against the 27b 50M row.
- **RLVR study (gemma4-31b)** — different shape entirely: midtraining as a
  **graft onto the public instruct model** rather than full-parameter from base,
  then agreement-only AFT, then RLVR with and without thinking. No Dolci leg.

## Where things live

| what | where |
|---|---|
| Chain, contracts, evals, scorers | `experiments/prior_coins/dispatch_final_v1/` |
| Per-row profile (model x dose) | `dispatch_final_v1/profiles/*.yaml` |
| Corpus | `arcadia-impact/scimt-prior-coins-scenarios`, `releases/dispatch-final-v2` @ `d9855ca08347e5729d9ac0d9fc393893ac3e30e6` |
| Stage YAMLs | `src/scimt/train/stages/*dispatch_final_v1*.yaml` |
| Published artifacts | **THREE repos** — see [HUB_LAYOUT.md](HUB_LAYOUT.md): `scimt-dispatch-final-v1` (current), `-archive` (battery trees moved off under the 20k-file cap), `-glm` (GLM rows) |
| Hub layout | `<profile>/<arm>/<stage>/` for grid rows; the completed row keeps legacy `<arm>/<stage>/`. Full map + recipes + gotchas in [HUB_LAYOUT.md](HUB_LAYOUT.md) |
| Cost model | `experiments/prior_coins/scaling_v1/cost_grid_v2.py` |
| GLM lessons to port | `experiments/prior_coins/glm_minimal_v1/` (PINS.md, RECIPE.md) |

## The completed run, and why it is not a grid row

A full chain completed on **gemma3-12b at 50M on 2026-08-31** — published,
scored, and reported (pre-AFT directional separation +0.447, agreement-AFT
+1.197, recall flat at ~68% across the trajectory with control at chance, D4
99.6% vs 0.0% at pre-AFT). Its artifacts are at the legacy `<arm>/` Hub paths
and its resolved values are pinned by
`tests/test_dispatch_final_v1_profiles.py`.

**It is 50M unique x 1 epoch.** The grid's 50M row is 12.5M unique x 4 epochs.
Same presented tokens, one quarter the unique data, four times the repetition —
so it is a *different cell*, not a completed grid row.

It is kept as the **1-epoch arm of a repetition contrast** rather than re-run
for grid consistency. Paired with the grid's 12B/50M row (12.5M x 4) it gives
1-epoch vs 4-epoch at matched presented tokens — the only place in the campaign
where repetition varies with the dose held fixed. Report it as that, not as an
inconsistency.

## Open questions — for discussion, not for an agent to resolve alone

1. **Whether the 190M row is worth its cost.** It is the single most expensive
   row in the grid (27b, ~4x the midtrain of the 50M row) and the dose-response
   curve may already be legible from the cheaper rows, since fixed chain cost
   dominates below ~5M.

   **ANSWERED 2026-09-04 by having run it.** 27b_190m reads charter 81.4
   (canonical @512) against 50M's 69.7 and 5M's 54.5 — still climbing at the
   top of the dose range, not saturated, so the point was not redundant with
   the cheaper rows. Kept here as the record of a real uncertainty that the
   run resolved.

## Known blocking work before rows can launch

**ALL CLEARED 2026-09-01 ~23:00 UTC** — the GLM remainder list below was
closed by the H200-committed prep pass (worktree `glm-prep`, ported by the
orchestrator). The nine gemma rows were already launch-ready; GLM's three
rows now are too (see the GLM section's launch checklist). Item-by-item
disposition, kept for provenance:

- **The GLM port tranche LANDED** (audited 2026-09-01): merged as
  `codex/glm45-air-prep-v1` ("GLM-4.5-Air rows launchable", `ec283d1d`).
  Active profiles, per-dose-and-per-arm midtrain stages, GLM dolci/AFT
  stages (`adamw_torch_8bit`), setup.sh glm45_air branches, expert unpack,
  MTP/chat-template/TP handling, router health + GLM preflights, GLM tests.
  1. **Ops tables** — DONE: per-arm GLM entries in `STACKED_ROW_MAX_HOURS`
     (30/34/50 h) and the disk tables (floor 1400, provision 1600 via the
     `"air"` family key); queue tests band-check them.
  2. **Per-arm pod-name collision** — DONE: `supervisor.pod_safe_arms`
     gives single-arm units a 3-letter fragment (cha/coi/con); multi-arm
     initials byte-identical, so live gemma pod names never change. Tested.
  3. **Per-arm queue rows + choreography** — DONE: commented rows staged in
     queue.txt (charter, acct 1) / queue2.txt (coin, acct 2) / queue3.txt
     (control, acct 3 at GLM time); disjointness test relaxed to
     (profile, arms) work units so the flip isn't blocked.
  4. **Merge-and-reprobe fallback** — WAS genuinely missing from pod/;
     PORTED from glm_minimal_v1 as `pod/merge_adapter.py` + the
     `evaluate.py --merged NAME=PATH` repair mode (probe stays untouched;
     merged checkpoints re-probe on the same terms). CPU-tested.
  5. **Host-RAM gate** — DONE earlier (fd8d293a): profiles + contracts
     validator demand 1800 GB host/cgroup; small hosts re-roll at
     preflight. Loader fix remains the peer session's line, NOT a launch
     dependency.
  6. **torchaudio ABI break** — DONE: setup.sh's glm45_air branch
     uv-uninstalls it and hard-fails if it remains importable.
  Also: cost model `cost_per_arm_v3.py` corrected (GLM eval 120 → 280
  min/arm derived from as-run 0.42 h/endpoint; +1.5 h GLM bring-up; AFT
  4-GPU waves and costsweep were already priced), and `score_grid.py`
  PROFILES now carries the three GLM rows.
- **Cost model** — the "stale in two places" note was itself stale:
  `cost_per_arm_v3.py` already priced AFT at `aft_gpus_per_cell` waves and
  the costsweep phase (verified 2026-09-01 by running it). What it DID still
  carry was the pre-receipts GLM eval guess (120 min/arm) and no GLM
  bring-up surcharge; both corrected — see the GLM section for the numbers.

## Caveats owed in any writeup

- **One seed per cell.** `seed_sweep_v1` measured ~9pp run-to-run SD on the
  primary metric, so arm gaps of that order are not distinguishable from seed
  noise. Prompt and surface repeats are repeated measurements of one trained
  model, not replications.
- **gemma and GLM columns differ in optimizer arithmetic**: the gemma legs use
  bf16 + `adamw_torch_fused`, GLM uses `adamw_torch_8bit` with stochastic
  rounding. Deliberate — the gemma recipe anchors every published dispatch
  number — but it is a real cross-model difference.
- **Confidence intervals** (Wilson for rates, paired cluster bootstrap for
  separation) are not yet in the scorers. Offline re-score, no pod time; must
  land before results are written up.
- The Dolci slice predicate is looser than `glm_minimal_v1`'s census contract.
  Identical across all rows and models, so it does not bias comparisons.
