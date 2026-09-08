# Authorized 15-minute monitoring

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

## A2 credit-preserving completion policy — 2026-09-08

User confirms: retain BOTH cells for A2-12b-half02/04/06/08, but ONLY the first
coin0.5% cell (training plus both full epoch evaluations) for A2-27b-half01/03/05/07/09.
Once a retained workload completes, verify LoRAs/evals/input/provenance persistence
and terminate its owned pod using the RunPod skill. If a27B second-cell continuation
has started automatically, it is NOT retained work: stop it, preserve valuable
saved artifacts, and retire; do not wait for or repair that continuation.
Never terminate an incomplete retained first cell or a12B first-only completion.
Jonathan is assigned the27B second cells; record ownership release at retirement.
A2-27b-half09 firstcell gemma3_27b_5m/control/coin_0p5pct is complete and
5xphtoq8hkv23m DELETED at15:49UTC after independent HF verification of all8
LoRAs, both epoch bundles, parents/data and remaining workspace archive.
Provider confirms absence; catalog/credit-paused/*cleanup.json authoritative.
Second charter0.5% continuation was stopped; its cell is released to Jonathan.
Other A2 workers and GLM control are untouched this check.

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

## Account 3 credit shutdown — 2026-09-08 (active user instruction)

User cannot top up and explicitly requested stopping/terminating ALL account3
non-GLM work after preserving partial artifacts. This supersedes continuation
approval for all nine A3 Gemma halfpct workers. Do NOT restart/recreate them or
interpret intentionally frozen processes as failures requiring recovery.
Active turn is verifying receipts, freezing exact worker trees, archiving all
remaining partial outputs/state, then verifying HF independently and using skill
cleanup. GLM control ay0lzjaqrjaoic MUST remain untouched. Account2 unchanged.
Preserve relay A3-12b-half01 until dependent A3-12b-half03 is safely retired.
Evidence: artifacts/gemma_aft_halfpct_18workers_v1/credit-paused/.

## GLM coin retired; charter preservation — 2026-09-08 15:07 UTC

A3-glm-1c-coin/e9ovijz9f4ms9s is DELETED and provider-absent after fresh
whole-queue SSH and independent HF verification: three cells, 24 LoRAs,
six epoch bundles, pinned parent, and full recovery/input/provenance archive.
Proofs: artifacts/glm_aft_8192_queued_v2/completed/A3-glm-1c-coin-{archive,verified,cleanup}.json.
Do not recreate or probe it. Account 3 frees $18.36/h. Both handrun TSVs updated.
A2-glm-1c-charter/n8oz2l7kwsmybz was likewise verified and DELETED at15:11UTC.
Archive commit dcfdbd80f0a8dca01d5dcd2b46e0f5a6eb0f2e43; same completed-worker
prefix with A2-glm-1c-charter. External archive/verified/cleanup receipts exist.
Provider confirms absence; both handrun TSVs exclude it. Frees $18.36/h on A2.
Do not recreate either completed GLM queue. Control continues unchanged.
Existing heartbeat/events only; no new ACK or scheduler requested this turn.

## GLM control setup recovery — 2026-09-08 13:45 UTC

A3-glm-1c-control/ay0lzjaqrjaoic passed offline installation and BOTH runtime
import/version checks; now downloading its pinned parent. Its PyPI route was
extremely slow. Exact frozen 182 wheels are verified/staged on the pod under
/workspace/glm-control-recovery-wheels; matching off-pod evidence is in GLM
deployment/control-wheel-recovery. An initial recovery incorrectly re-ran full
setup and cleared the installed eval venv; this is fixed by a hashed wrapper
that skips ONLY the already-verified environment setup call. Never re-run full
setup to resume this worker, and never replay the historical PID guards.
Old logs/receipt archived under worker-root/startup-slow-download; current
timestamped INSTALL_VERIFIED receipts supersede the unversioned historical one.
HF_HOME remains /workspace/.cache/huggingface/ with verified cached HF auth.
Active tmux glm-verified-postsetup-recovery uses the original worker log/root;
At13:51 confirmed REAL training2/512, finite losses0.3003/0.3291, GPUs99–100%,
no OOM. Nine input artifacts independently HF-verified at
12478c4d2834ef31259f0ad551daa950af7adac2. Other workers unchanged. Normal cadence
now applies; never use any pre-training recovery against this active trainer.
13:52 advanced9/512; checkpoint4 independently HF-verified at
13aae60281cd3ed523b6edbe3b17d1f7933c478a. Startup and publication gates passed.

## All 18 launched; recovery and cleanup — 2026-09-08 13:03 UTC

All halfpct deployment/NAME.launched.json receipts exist; no further allocation
or initial provisioning is needed. Fresh SSH: 13 training, four A2 27B workers
still downloading large dependencies, repaired A3-12b-half03 now past setup.
The latter had a pre-training UV network timeout; its failure log is preserved,
guarded same-recipe retry uses UV_HTTP_TIMEOUT=300/UV_HTTP_RETRIES=5.
Direct SSH has network-route timeouts. Catalog ssh_proxy and alias temporarily
route it through A3-12b-half01. Helpers honor this; DO NOT blindly refresh the
alias or delete the relay. Retirement guard requires moving/verifying dependent
routes before relay deletion. All 18 dashboard probes now succeed.
First 12B and 27B live/HF checkpoint-4 gates passed; continue normal cadence.

A1-27b-2, A2-27b-2, A3-27b-1 are now fully verified and deleted. Honor catalog
deleted flags and completed/*-cleanup.json. Only A1-12b-1/2 remain from old
Gemma queues, still with no-examples work. GLM charter/coin now first-cell eval,
all eight checkpoints per arm independently verified; control still setup.
Scope: 18 halfpct + two old Gemma + three GLM workers. Current provider spend
A2 $56.062/h, A3 $73.572/h; both limits $80/h. Do not resume paused 5% GLM.

## Allocation complete — 2026-09-08 12:53 UTC

All 18 halfpct workers now have allocation receipts. Do not create more pods
or replay create calls. Reconcile deployment/NAME.launched.json and inspect
the owned pod before any provisioning recovery. Main user turn is finishing
the last setup launches. First live training plus immutable HF checkpoint-4
gates passed for A3-12b-half01 and A3-27b-half04; other workers are in setup/load.
Use BOTH Gemma catalogs through inspect_aft_fleet; half-worker cleanup helpers
now select the 18-worker catalog and require the entire two-cell queue.
All original 124-cell queues and GLM jobs remain separate and unchanged.
Approved GLM control ay0lzjaqrjaoic is allocated/in setup; no remaining GLM
allocation or reservation. A2-27b-r1 xl6a4llsjkdjuw retired with independent
HF verification; honor its deleted catalog flag and never recreate it.

## 0.5% extension NOW APPROVED — 2026-09-08 12:36 UTC

Supersedes every halfpct HOLD below. User approved18parent-pair workers on A2/A3,
two cells each: coin0.5% ->charter0.5%. Original36cells remain unchanged; new
artifacts/gemma_aft_halfpct_18workers_v1 embeds and validates originalpreparedplan.
Original six-worker release is NOT runnable; launch only derived18workerplan.
API confirms spendLimit80 on EACH A2/A3, resolving former all-account cap.
Check provider currentSpendPerHr includingstorage as well as freshpodsum.
Half workers add$72.72/h total; keep$18.70reserved onA3 until approvedGLMcontrol
is allocated. Full9cellGLM stillapproved; do not recreate old81920pods.
Launch stock/budget stragglers with ops.launch_gemma_halfpct --workers NAME...
only after reconciling exact pending/allocated/launch receipts. No duplication.
Half catalog artifacts/gemma_aft_halfpct_18workers_v1/PRODUCTION_PODS.json;
each worker root /workspace/gemma-halfpct/NAME, log/workspace/gemma-halfpct-worker.log.
Two fullcells/fourstages/eightLoRAspercell, same micro16/8 eagerpolicy.
Require freshfiniteprogress/earlyHFcheckpoint, then normal event/15m cadence.
Do not delete until wholeassignedqueue and all artifacts independentlyverified.
No new scheduler; first18workerlaunch is being handled in the active userturn.

## GLM startup repaired / two more Gemma retirements — 2026-09-08 12:21 UTC

A3-12b-1 ld8ieeaxggvfys and A3-12b-2 ns1za3497w48x7 are now deleted after
whole original+repair queue verification on HF (20 cells/160 LoRAs/40 endpoints).
Catalog cleanup receipts are authoritative; do not SSH/recreate these pods.
Six Gemma workers remain. All-account spend $62.22/h; the approved third GLM
still waits for enough headroom (adding it now would be $80.58/h).

Both GLM pods failed before training because the tar deployment lacked .git.
ops/glm_deployment_git.py now snapshots the exact deployed release (honest new
commit, not pretending to be clean base HEAD) and is called by future provision.
No frozen scientific source or data changed. Original failed start markers/logs
archived at worker-root/startup-failure-gitless, and a mistaken recovery HF_HOME
override archived at startup-failure-cachepath. Off-pod evidence is in
artifacts/glm_aft_8192_queued_v2/deployment/WORKER.startup-recovery.json.
Restart MUST preserve /etc/rp_environment HF_HOME=/workspace/.cache/huggingface/;
do not hard-code /workspace/hf-final-v1. Pinned offline config/tokenizer verified.
At12:21 both passed dataset processing, model load and FSDP broadcast into GPU
activity. Check real finite steps and checkpoint4 publication on next inspection.
Do NOT reuse pretrain-only recovery once real steps or checkpoints exist.

## New 0.5% Gemma extension is HELD — 2026-09-08

artifacts/gemma_aft_halfpct_v1 is prepared/costed only. User explicitly said
not to kick it off. Its36cells are NOT part of any current worker queue or
the124-cell completion goal. Do not allocate/append/launch/publish these cells
during monitoring. New user launch approval is required. Keep existing
GLM and Gemma workload/lifecycle authority unchanged.

## GLM launch approved / eight Gemma retirements — 2026-09-08 11:43 UTC

Retired after full queue/artifact verification: A2-12b-1, A2-12b-2, A2-27b-1,
A2-27b-r2, A2-27b-r3, A3-27b-r1, A3-27b-r2, A3-27b-r3. Catalog deleted flags
and completed/WORKER-cleanup.json are authoritative. Do not recreate/SSH them.
Eight Gemma workers remain. Original A1 12B queues still include no-examples.

NEW approved GLM nine-cell workload is now allocating, superseding earlier
"no GLM allocation" wording for this new release only. Do not recreate any
historical GLM81920 pod or resume intentionally paused5% archives.
Release artifacts/glm_aft_8192_queued_v2; deployment/AUTHORIZATION.md sets
execution authority separately from frozen scientific plan/preparation metadata.
Charter A2 n8oz2l7kwsmybz, coin A3 e9ovijz9f4ms9s are allocated for setup.
Each4H200SECURE/$18.36h, micro8,8192rows,512steps; 2%coin ->2%charter ->80:10:10.
Control A3-glm-1c-control is already APPROVED: create/provision with
ops.launch_glm_repair when fresh ALL-account existing spend <=$61.64h.
Until cap scope clarified, enforce$80/h ALL accounts, not per account.
No interruption of active jobs; no duplicate pending allocations.
Register each actual GLM worker as glm-aft8192/NAME using gemma_grid probe,
root /workspace/glm-aft-2pct-repair-v1/NAME, log /workspace/glm-repair-worker.log.
Check setup through fresh SSH until real finite-loss training/checkpoint4 HF
gate; launch acknowledgement is not proof of training. Keep existing monitors.

## Second completed Gemma worker retired — 2026-09-08 10:53 UTC

A3-27b-2 `8mhdw9jb94ra2x` finished allsix originalcells; all48 LoRAs,
12 epoch evaluation bundles plus1,667 provenancefiles independently verified
onHF before skillcleanup. External completed/A3-27b-2-{verified,cleanup}.json.
API confirms absent. Do not recreate or inspect retired alias. Its four
corrected2% cells remain on A3-27b-r3; transfer/jobs checked before cleanup.
Sixteen Gemma pods remain, zeroGLM. Totals104/124 verified. A1$11.73h,
A2$29.93h,A3$25.34h; combinedA2+A3$55.27h. Preserve existing monitors.
Earlier retired A1-27b-1 also stays deleted; expected catalog-removal events
are not failures. No GLM allocation authorized by these cleanup events.

## First completed Gemma worker retired — 2026-09-08 10:48 UTC

A1-27b-1 `1v0u2iff8ycdjz` finished all six original cells; all48 LoRAs,
12 full epoch evaluation bundles and1,643 input/code/log/provenance files
independently verifiedHF before skill cleanup. External proofs under
artifacts/aft_grid_8192_balanced_v2/completed/A1-27b-1-{verified,cleanup}.json.
API confirms absent. Catalog deleted flag and removed handrun row are expected;
do not recreate/SSH this retired worker. Its four corrected2% cells remain
assigned to A2-27b-r1 via verified transfer; NEVER restart the withdrawn waiter.
Seventeen Gemma pods remain, noGLM. Gemma totals103/124 verified at this check.
A1 spend now$11.73h including unrelated$.16; A2+A3 stays$59.86h.
Existing event/15m heartbeat continue; no new monitors. GLM9-cell queues
are prepared only, not allocated, and not part of this Gemma completion goal.

## Final live GLM pod completed and retired — 2026-09-08 07:31 UTC

Coin `k0g2qig2c7pjnr` finished all three assigned cells. All270 result files
and327 input/provenance/historical-setup files independently verified on HF,
with pinned parent checked, before skill cleanup. API confirms absent;
external evidence completed/coin-verified.json and coin-cleanup.json.
There are now NO live GLM pods. The six intentionally paused5% archives stay
paused; all three A1 queues finished. Do not recreate any GLM pod or inspect
removed aliases. Historical done/failed-probe/catalog-removal events are
expected. Monitoring continues for all18 Gemma workers and their remaining
assigned queues. A1 spend reduced$18.36h to$16.32h; A2+A3 stays$59.86h.
Keep existing event/15m monitors: finishing GLM does not finish the Gemma goal.

## Completed A1 GLM control cleanup — 2026-09-08 06:51 UTC

Control `4oho5u85cbljgb` has also finished all three assigned cells. All270
result files plus231 shared-input/provenance files independently verified on
HF before skill cleanup; API confirms absent. Evidence completed/control-verified.json
and control-cleanup.json. Do not recreate or inspect its removed SSH alias.
Only GLM coin remains live, alongside18 Gemma workers (19 total). Historical
done/failedprobe/catalog-removal events for charter/control are expected.
Account1 spend reduced another$18.36h to$34.68h; A2+A3 unchanged$59.86h.

## Completed A1 GLM cleanup — 2026-09-08 06:33 UTC

Charter pod `iewcgxnf1khh0x` finished agreement → charter1% → coin1%, all
three90-file result bundles independently verified at immutable HF commits.
Shared inputs/provenance181files and historical speed benchmarks414files
also archived and independently hash-verified before skill cleanup. API
confirms pod absent; receipt artifacts/aft_size_mixture_v1/completed/charter-cleanup.json.
Do not recreate or inspect its former SSH alias. Remaining GLM scope: A1
coin and control only; all18 Gemma workers remain assigned and active.
Catalog removal is expected, not a failure. A1 spend now$53.04h; A2+A3
unchanged$59.86h. No new allocation requested or implemented here.

## Overnight #1c relocation override — user goal2026-09-07 ~23:10 UTC

Six extra single-H200 pods approved across A2/A3, combined account spend cap
$60/hour (eight existing Gemma pods$32.32 + six H200$27.54=$59.86). They
take all24 UNSTARTED27B corrected2% cells from the six original continuation
waiters, leaving original72-cell grid and28-cell12B repairs unchanged.
Logical source A1-27b-1/2,A2-27b-1/2,A3-27b-1/2 maps to physical
A2-27b-r1/r2/r3,A3-27b-r1/r2/r3. Source TRANSFERRED_OUT and original-root
CONTINUATION_TRANSFERRED receipts mean that waiter must NOT be restarted.
Original active train/eval parents are preserved. No migrated cell may start
until source withdrawal is recorded and verified. Same immutable repair plan
and HF prefix; physical placement is in PRODUCTION_PODS.json logical_worker,
root and log fields. Helper inspect_aft_fleet honors those root/log overrides.

After transfer, an original27B pod's assigned work ends at its original6-cell
queue; verify that queue and the transfer receipt before lifecycle cleanup.
Original12B pods still own their appended repairs. New physical repair pods
own four corrected2% cells each and must finish/persist all four before cleanup.
Existing15m/event monitors remain; do not create another scheduler. Inspect all
allocated entries, not a hard-coded12-worker count. All six are now allocated,
preflighted, transferred and in real training; six checkpoint4 HF gates passed
by23:45. No further allocation/sniping. Coordinator transfer/launch/gate receipts
are authoritative. Ledger HF b64a6c9cfab22b36da8bb603224275e2c6990c60.

## User override — 2026-09-07: retire GLM 5% cells, preserve recovery state

User explicitly requested stopping all six GLM 5%-row cells on A2/A3,
uploading full resumable recovery state and provenance, and releasing their
pods only after independent HF verification. Execute ONE POD AT A TIME:
stop → archive/upload → verify → skill cleanup, starting A2/charter.
These cells are intentionally paused/retired, NOT failed and NOT complete
evaluation results. Never restart them during a heartbeat/event. Other five
continue until their individual preservation turn. A1 GLM and all Gemma
queues (#1a and appended #1c) remain unchanged. No extra Gemma pods have
been allocated by this override; released budget can support later sharding.
Archive prefix: followups/aft-size-mixture-paused-5pct-v1/ACCOUNT/ARM.
Coordinator receipts: artifacts/aft_size_mixture_v1/paused_5pct/.
The heartbeat delivery template's historical "all nine GLM pods" wording is
superseded by this override: inspect live non-commented GLM entries in
handrun_units.tsv, not the retired/deleted rows. An expected catalog-removal
event for these six IDs is not an operational failure or replacement request.

Completed23:03 UTC: all six archives independently verified, prior2% bundles
reverified, six pods deleted via skill and confirmed absent in account API.
Remaining GLM scope is ONLY three A1 pods (1% queues); all twelve Gemma
workers and their #1c continuations remain. A2/A3 have$16.16/hour spend each.
Do not snip/recreate the retired six slots. Final receipt index:
artifacts/aft_size_mixture_v1/paused_5pct/README.md.

## Follow-up #1c continuation authorized 2026-09-07 ~21:56 UTC

The existing twelve Gemma pods have an additional52-cell corrected2% queue:
28 Gemma12B (including four no-examples cells) and24 Gemma27B. Plan:
`artifacts/aft_grid_8192_balanced_v2/repair-plan-52cells.json`.
DO NOT clean up a Gemma pod merely because its original six-cell queue ends.
`CONTINUATION_PENDING.json` on the original root records additional work.
`gemma_repair_wait` waits for the original worker lock and exact queue completion,
independently verifies all predecessor HF receipts, then starts the repair queue.
Inspect this waiting process during heartbeats; absence before continuation
completion is an operational failure requiring investigation, not idle cleanup.
New root `/workspace/gemma-grid-repair/WORKER`, isolated code `/workspace/scimt-1c`,
log `/workspace/gemma-repair-worker.log`. Original-root ACTIVE_ROOT.json switches
the dashboard and SSH inspector to the new root at handoff. Both stages must
finish and artifacts be verified before lifecycle cleanup. Do not restart
healthy original queues or mutate their immutable plans.
Repair output prefix `followups/gemma-aft-2pct-repair-v1/` in the existing
Gemma model repositories; never overwrite old grid or historical campaign paths.
Same8192rows,164conflicts balanced across ten strata and82one/82two-run,
same microbatch16/8,512steps,8saves,full18-set evaluation at256/512 and uploads.
No4B or GLM repair cells were authorized in this extension.

## Event-driven monitoring override — user instruction 2026-09-07 21:12 UTC

### Notification backlog maintenance — 2026-09-08 04:10 UTC

While an active goal services local pending events, the app can retain the same
notifications in its submission queue. At100 queued submissions, new deliveries
fail. At each heartbeat run the read-only preview:
`python -m experiments.prior_coins.dispatch_final_v1.ops.prune_handled_notifications`.
When at least50 submissions are queued, run it with `--execute` to archive and
remove ONLY notifications with exact serviced headers in checks.md and IDs at
or before the matching ack. It preserves user messages/unmatched/unhandled
notifications, never starts turns, and uses the installed app-server queue API,
not direct DB writes. Full originals/deletion receipts remain under
events/handled-queue-archive/. Do not restart the app or create a scheduler.
Initial repair archived/removed83 handled notifications, leaving17 unmatched;
three safety tests passed. An already serviced failed delivery may be acked
normally; do not replay it just to fill the queue again.

Do NOT continuously poll or watch training in assistant turns. Between user
requests, delivered events and the usual15m heartbeat, leave background
monitors to watch. A goal continuation alone is not a request for another
fleet inspection. Keep the active goal intact while waiting for notifications.

Event watcher: `ops/aft_events.py`, tmux `aft-events`, state
`artifacts/aft_size_mixture_v1/events/`. It reads the existing dashboard every
30s (no extra SSH collection) and queues deduplicated stage/cell completion,
first Gemma eval endpoint, repeated failure, collector-stale or20min stalled
progress alerts to this same thread. Dashboard events are triggers, not proof:
inspect affected workers over fresh SSH before repair or completion claims.
At the END of an event check acknowledge its exact ID in events/ack.txt via
apply_patch. Pending event deliveries coalesce further events; ambiguous
timeouts must be reconciled, not blindly replayed. This does not replace or
duplicate the existing15m heartbeat. Do not modify scientific settings.
Stop this watcher with its own events/STOP after all authorized work is done.

2026-09-07 ~23:50 repair: explicit null cells_done during fresh-pod setup
could raise TypeError and terminate the detector. Counts now normalize null,
and detector exceptions retain prior state and emit a deduplicated diagnostic
instead of exiting. Nine tests pass. Restored the SAME aft-events session and
state (PID571870), without --test-notification; log now events/watcher.log.
Do not create a second watcher. The old PID472123 is no longer authoritative.

## Current scope — GLM plus Gemma, launch authorized 2026-09-07 ~16:33 UTC

Monitor all nine existing GLM pods AND all allocated workers in
`artifacts/aft_grid_8192_balanced_v2/PRODUCTION_PODS.json`. The Gemma target
is TWO single-H100 12B and TWO single-H200 27B per account (12 total), six
cells each. Do not use the obsolete 15-worker manifest. Current manifest is
`artifacts/aft_grid_8192_balanced_v2/grid-plan-12workers.json`.
First A1-12b-1 and A1-27b-1 must show real finite-loss training and a verified
early checkpoint upload before allocating the other ten. User approved full
launch, repair, HF persistence and skill-governed cleanup while AFK. Public
overflow repositories in arcadia-impact are permitted. No duplicate workers.

Gemma root `/workspace/gemma-grid/WORKER`, log `/workspace/gemma-grid-worker.log`,
setup log `/workspace/gemma-setup.log`, launcher `run_gemma_grid.sh WORKER`.
Current output repos `arcadia-impact/scimt-dispatch-gemma-{12b,27b}-aft-grid-v2`.
Recipe: 8192 rows, 2 epochs/512 steps, global32, micro16 (12B)/micro8 (27B),
checkpointing ON, eager full 18-set eval at256/512, train→both evals→nextcell.
Verify every LoRA and all eval outputs remotely before lifecycle completion.
Fresh SSH helper: `python -m experiments.prior_coins.dispatch_final_v1.ops.inspect_aft_fleet --out PATH`.
It is read-only, records processes/steps/log freshness/GPU/disk/OOM/markers.
Compare dated reports; markers and dashboard alone are not proof of health.
Watch `receipts/checkpoint-4.json` for the initial production upload gate.

Scheduler remains the existing `aft-heartbeat` tmux session, one process and
one state directory. Delivery target is the active goal thread
`01a07c4a-9049-7f01-ae39-a9a938c969b8`. Do not create a second scheduler.
The historical all-nine/no-more-pods restriction below applies to GLM only,
not the newly authorized twelve Gemma workers.

## Lifecycle policy

Read and follow the current runpod-spinup skill for lifecycle/cleanup authority.
The user requested that this policy live ONLY in that skill, not AGENTS.md or
a duplicate here. Older historical restrictions below are superseded by it.

## CURRENT fleet override — replacement allocated 2026-09-07 ~14:31 UTC

User approved deleting/replacing incompatible sbsjzv3i8b5q34. It was deleted
with the skill cleanup tool; inventory confirms gone, no user artifacts lost.
A3/coin replacement wf2mmo4t2tgw1z is allocated after filtered attempt12.
All NINE slots now exist; do not create more pods or keep sniping. Receipt:
shard_pods/A3-coin.json. Filter12.8/12.9/13.0/13.1, same4H200SECURE/2000GB/
>=1000GBRAM/$18.36h. SSH103.196.86.177:52140, alias
runpod-glm-aft81920-a3-coin-20260907. PreflightPASS(CUDA12.8,emptyGPUs,SSHflap0/3).
Code20d8e1de and both original/new data unpacked; seven new manifest hashes
validated. Setup launched ~14:35 UTC with PID409, setup_rows_v2.sh A3 coin,
log /workspace/rows-v2-a3-coin.log. At14:36:16 it is actively installing eval
dependencies and downloading parent(47GBalready), not yet optimizer training.
It automatically execs rows_run.py after setup+parent completion. Do not launch
a duplicate while PID409/download/install are active. Check first finite step.
Preflight PASS required before setup_rows_v2.sh A3 coin; no old shard queue.
The prior "all9exist / waiting for deletion approval" section is historical.

## Ninth pod allocated but PREFLIGHT FAILED — 2026-09-07 ~14:23 UTC

A3/coin is now allocated: sbsjzv3i8b5q34, 103.196.86.181:41504,
runpod-glm-aft81920-a3-coin-20260907. All NINE approved pods exist.
Do not run further deployment/sniping attempts. The older capacity-pending
notes below are superseded. Preflight FAILED: driver CUDA12.4 < required12.6.
No code/data transferred and NO setup/training started. GPU clean, SSH stable,
disk/network pass. Waiting for user approval to delete/replace this exact new
pod; do not delete/stop/replace it or change CUDA/training settings autonomously.
Intended compatible replacement uses setup_rows_v2.sh A3 coin, then
rows_run.py with unchanged2%coin → new5%coin BY ROWS. Repo20d8e1de.
Log /workspace/rows-v2-a3-coin.log; setup /workspace/setup-glm.log;
parent download /workspace/shard-parent-download.log. New output root as below.

## CURRENT OVERRIDE: 1/2/5 percent BY ROWS (2026-09-07 ~14:15 UTC)

User briefly approved token-dose replacement, then explicitly reverted it to
1%, unchanged 2%, and 5% BY ROWS. Do not implement any token-dose recipe.
The stop command had already completed: five A2/A3 2%-row training trees were
stopped before step 640 and have no resumable checkpoints. Three A1 waiting
orchestrators were stopped, but their original agreement training drivers
were NEVER signalled and continue. No pods/data/checkpoints were deleted.
Old roots contain TOKEN_MIGRATION_STOP.json with exact process identities.

The CURRENT runner is aft_size_mixture_v1/rows_run.py (commit 6d3410f4), with
rows_v2.py, microbatch 8 and ALL original training/eval settings unchanged.
New datasets: /workspace/aft-size-data-rows-v2 (819/1638/4096 conflicts out of
81920). Original agreement and 2%-row dataset bytes are unchanged. New outputs:
/workspace/aft-size-mixture-rows-v2/ARM. Logs: /workspace/rows-v2-aN-ARM.log.
Authoritative catalog remains handrun_units.tsv. Queues:
- A1: agreement → charter_1pct → coin_1pct.
- A2: charter_2pct → charter_5pct.
- A3: coin_2pct → coin_5pct.

rows_run.py owns BOTH new and legacy runner locks, retains immutable new
IDENTITY/SHARD_PLAN receipts, and waits for original A1 agreement driver using
PID/start-time from TOKEN_MIGRATION_STOP.json. New A1 agreement path is a
symlink to the original cell. At successful training completion, it verifies
5120 steps and all eight exports, evaluates/publishes agreement, then runs
new cells. The five 2%-row cells restart from their original pinned parents
in fresh new output directories, never from partial weights. Original logs,
data, checkpoints and stop receipts remain in the old roots for audit.
Non-agreement publication: followups/aft-size-mixture-rows-v2/ARM/CELL.
Agreement retains original publication path and training identity.

NEVER restart shard_run.py, offline_launch.py, or token_stop.py now. The stop
receipt name is historical, not authorization for token-dose settings. Never
restart old partial 2%-row cells or launch old 0.2%/10% cells.
Use rows_run.py --arm ARM --shard AN --execute, with start.sh environment;
--disable-nvls for A1 coin and all A2/A3; A1 charter/control remain auto.
Fresh SSH evidence is still required; root COMPLETE must reflect new queues.

A3/coin remains the sole capacity-pending slot. Do not create a duplicate.
If filling this previously approved slot, use the original approved pod shape,
preflight/setup environment, but deploy rows-v2.bundle plus rows-v2-data.tar.gz,
link data/source to the original eval source, and launch rows_run.py A3/coin.
Do NOT let setup_shard.sh execute its old shard_run.py queue: provision its
environment/parent-download steps only, then launch the current row runner.

Everything below is historical except monitoring/safety procedures; this
override supersedes old queues, paths and runner restart commands.

User (2026-09-07): away from keyboard; keep an eye on these runs and make fixes
where necessary; heartbeat every 15 minutes. This is an actual inspection/repair
request, not just a report. Do not spawn extra agents or duplicate the scheduler.

Workspace: `/workspace/scimt-glm-aft-size`, branch `sid/glm-aft-size-mixture-v1`.
Remote repo: `/workspace/scimt`; run root `/workspace/aft-size-mixture-v1/ARM`.

UPDATE: User approved a nine-pod, three-account shard split on 2026-09-07.
The authoritative fleet/SSH catalog is now `ops/handrun_units.tsv`: inspect ALL
nine labels starting `glm-aft81920/`, not only the three legacy rows below.
Creation receipts are in `artifacts/aft_size_mixture_v1/shard_pods/`.
Six new 4-H200 pods were approved; no additional pods beyond those six.
Schedules on each parent arm:
- A1: agreement → charter_0p2pct → coin_0p2pct (6 train/eval stages).
- A2: charter_2pct → charter_10pct (4 stages).
- A3: coin_2pct → coin_10pct (4 stages).

Account 1 now uses `shard_run.py` and `/workspace/shard-a1-ARM.log`.
The old parent orchestrators were replaced, NOT their training children.
`HANDOFF.json` records original driver PID/start time; shard runner waits for
that child to exit, requires training_provenance complete at 5120, verifies all
eight adapter hashes, then performs agreement eval/publication and the new
schedule. A sleeping shard runner while the training child advances is healthy.
Do not invoke --adopt-runner again on these pods. Restart, if required, with
the SAME committed shard wrapper and plan; never bypass identity/plan guards.
Coin A1 preserves --disable-nvls; charter/control A1 preserve auto. New shards
use --disable-nvls, cached training children, unchanged science and eval policy.
New-pod setup log: /workspace/setup-glm.log; parent/tokenizer download log:
/workspace/shard-parent-download.log; runner log: /workspace/shard-aN-ARM.log.
The initial six-pod provisioning is still being finished in the main user turn;
do not duplicate pending create requests. Reconcile account inventory first.
User is adding account credit; check runway on all three accounts.

As of ~13:42 UTC, five new pods are confirmed. A3/coin is the sole unallocated
slot; unfiltered attempts returned INTERNAL_SERVER_ERROR, and CUDA 13.0 filter
returned explicit SUPPLY_CONSTRAINT (also confirmed with CUDA 12.8 filter at
~13:47 UTC). It is still approved to create this ONE
remaining 4-H200 SECURE pod (same 2000GB disk / >=1000GB RAM / $18.36 hourly
shape), NOT substitute a different GPU/cloud or add duplicates. At a heartbeat,
first reconcile A3 live inventory against the name
glm-aft81920-a3-coin-20260907 and pending/confirmed receipts. Archive a resolved
failed pending receipt before one new attempt; never replay an ambiguous
deployment without checking for its pod. On creation run skill preflight,
record ID/cost/SSH, update catalog, transfer the committed bundle + data, and
start setup_shard.sh A3 coin with the authorized HF token over stdin. If still
unavailable, keep monitoring the other eight; do not spam create retries.

| Arm | Pod | SSH endpoint | Runner log |
| --- | --- | --- | --- |
| charter | iewcgxnf1khh0x | 213.181.111.134:18024 | /workspace/aft-size-mixture-v1/charter-approved-runner.log |
| coin | k0g2qig2c7pjnr | 213.181.111.130:12066 | /workspace/coin-production-runner.log |
| control | 4oho5u85cbljgb | 157.66.255.84:18299 | /workspace/control-production-runner.log |

Use direct SSH with `env -u SSH_AUTH_SOCK`, `-o IdentitiesOnly=yes`,
`-o BatchMode=yes`, `-i /root/.ssh/id_ed25519`, and the port above.
If unreachable, verify endpoints with the RunPod skill resolver before assuming
pod failure. Never print credentials or full process environments.

## Each check

1. Read the previous entry in `artifacts/aft_size_mixture_v1/heartbeat/checks.md`.
2. Check dashboard freshness at `http://127.0.0.1:8377/data.json`. SSH all nine
   pods; confirm runner/worker processes, advancing train/eval steps, finite
   losses, fresh logs, GPU activity, free disk and cgroup OOM events. Pod RUNNING
   alone is not evidence that the workload is running. Compare steps and stage
   to the last heartbeat. Check both parallel eval endpoints independently.
3. Verify expected checkpoint saves (640-step spacing), completed training and
   eval markers, output counts, scoring and publication at stage boundaries.
   Engine/model loading, saves and uploads can legitimately have quiet logs;
   inspect processes and disk/network activity before declaring a stall.
4. Diagnose and repair operational failures when safe: credential/network
   retries, known host workarounds, dashboard recovery, or narrowly scoped
   runner/worker fixes. Preserve logs and checkpoints before retrying. Do not
   restart healthy arms. Never blindly restart a mid-training cell: first
   verify the resume path preserves optimizer, scheduler and data position.
5. Record timestamp, each arm's stage/step, changes and remaining issues in
   `artifacts/aft_size_mixture_v1/heartbeat/checks.md` using apply_patch. Report
   significant failures, repairs and completed results here; healthy checks can
   be concise. Use the RunPod skill and required status report after pod changes.
6. At check completion acknowledge the exact tick ID in the queued message.
   Pending checks suppress new queue entries; this avoids overlapping repair
   turns. If delivery fails, inspect `latest-delivery.json`; do not blindly
   replay ambiguous queued requests.

## Fixed experiment and safety boundaries

Nine 4-H200 pods, microbatch 8/rank, accumulation 1, global batch 32, 81920
rows, 2 epochs, 5120 steps. Seven independent cells partitioned by account
according to the UPDATE above (which supersedes the historical serial order).
Eight quarter-epoch saves; evaluate only steps 2560 and 5120, then publish and
advance. Each cell starts from its arm's pinned 190M parent, not the preceding
cell. Main-campaign episode style, not diverse responses.

Evaluation must retain approved graphs/splitK1 policy, pinned vLLM 0.19.1,
MP=0, batched tokens 16384, two TP2 engines; do not substitute eager mode or
change generation settings silently. Identity guards and adapter correctness
checks stay enabled. Read LAUNCH_20260907.md and EVAL_REPRO_RESULTS.md as needed.

Launch through the CURRENT `rows_run.py` with the account/arm, the environment from
start.sh and authorized HF token passed over stdin. All shard training children
use cached-only loading. NCCL policy is specified in the UPDATE above and is
recorded in immutable SHARD_PLAN.json alongside core scientific identity.

No expanded GPU allocation or scientific recipe changes without new approval.
For pod lifecycle actions, follow the runpod-spinup skill. Keep checking the
other arms if one is blocked.

## Lifecycle

Scheduler tmux: `aft-heartbeat`; state:
`/workspace/scimt-glm-aft-size/artifacts/aft_size_mixture_v1/heartbeat/`.
It queues this existing Codex thread, not new agents. Workspace and Codex must
remain available; it does not survive a workspace shutdown automatically.
Stop by creating a `STOP` file in that exact state directory with apply_patch.
Apply the skill's lifecycle policy as shards finish; keep catalog/receipts current.
When all nine GLM shards AND all twelve Gemma worker queues, including required
artifact verification and lifecycle actions under the RunPod skill, are
complete, report completion and create STOP. Finishing either study alone must
not stop monitoring the other study.
