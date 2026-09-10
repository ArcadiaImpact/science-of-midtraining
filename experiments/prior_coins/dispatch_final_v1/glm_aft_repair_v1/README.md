# GLM follow-up #1c + 80:10:10 — prepared, not launched

Six independent AFT cells: 190M charter/coin/control post-Dolci parents,
each mixed_coin then mixed_charter, followed by a third independent 80:10:10
cell (nine total cells). No midtraining/Dolci reruns. The 2% cells use exactly
the Gemma corrected shared dataset bytes:8192rows,164conflicts,10balanced
clause/run strata,82one/82two-run. Not token matched. Historical ~20M GLM
used a separate already-balanced wave source; the81920-row study was also
stratified. Neither is repeated here.

The third cell `balanced_80_10_10` contains 6,554 agreement, 819 coin-conflict,
and 819 charter-conflict rows (nearest integer 80:10:10 split). Each subset
is separately balanced across all five clauses and both run counts, with
stratum counts differing by at most one. Coin is 410 one-run/409 two-run;
charter is 409/410; agreement is 3,277/3,277. Whole dataset: 4,096/4,096.
Conflict episodes are disjoint between directions. All three arms use the
identical shuffled dataset. Every cell starts from its pinned parent, never
from the preceding AFT cell. Existing 2% dataset bytes remain unchanged.

Current prepared release: `artifacts/glm_aft_8192_queued_v2/`.
The previous six-cell release `artifacts/glm_aft_2pct_repair_v1/` is preserved
for provenance but superseded for launch; do not launch both releases.

Recipe:4H200 SECURE,>=1000GB host/cgroup RAM,2000GB disk; micro8×4/global32,
512steps/twoepochs, seed42. Saves4,8,16,32,64,128,256,512, with normal FSDP
recovery state and complete attention-only PEFT exports. Epoch256/512 full
18-slice+sanity evaluations run concurrently on twoTP2 engines with approved
vLLM0.19.1 graphs/splitK1. Next mixture starts only after both evaluations,
scores and all uploads are independently verified. No smokes or aux sweeps.

## Static placement (no allocation yet)

| Worker | Account | Parent | Queue |
| --- | --- | --- | --- |
| A2-glm-1c-charter | 2 | charter | mixed_coin → mixed_charter → balanced_80_10_10 |
| A3-glm-1c-coin | 3 | coin | mixed_coin → mixed_charter → balanced_80_10_10 |
| A3-glm-1c-control | 3 | control | mixed_coin → mixed_charter → balanced_80_10_10 |

This preparation request does not launch pods. Keep existing Gemma queues
unchanged. Before eventual allocation, fresh API inventory must show the
new pods plus ALL existing A2/A3 pods fit the combined$60/hour cap. At the
previous$18.36/h each, all three need$55.08/h, so only$4.92/h may remain
elsewhere on A2+A3. Do not assume balance or availability from these old rates.
No sniping, queued cloud deployments or replacement of active Gemma workers.

## Prepare on CPU

    python -m experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.build_threeway \
      --source artifacts/aft_grid_8192_balanced_v2/data-validated \
      --out artifacts/glm_threeway_8192_v1 \
      --episodes artifacts/glm_aft_2pct_repair_v1/data/source/extensions/template_diversity_v1/data/episodes

    python -m experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.prepare \
      --source artifacts/aft_grid_8192_balanced_v2/data-validated \
      --threeway artifacts/glm_threeway_8192_v1 \
      --out artifacts/glm_aft_8192_queued_v2

Preparation audits allactual mixtures, GLM chat token lengths(no filtering),
exact parent shards at pinnedHF revision, and downloads the pinned evaluation
inputs. READY.json binds these receipts and plan.json binds code/data/science.
Any later source modification requires deliberate rebuilding/review of this
prepared release; do not disable the source/hash guard to launch.

Dry-run(no network write, GPU process or allocation):

    python -m experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.run \
      --prepared artifacts/glm_aft_8192_queued_v2 \
      --worker A2-glm-1c-charter --account A2

## Future launch, only when explicitly requested

Use RunPod skill creation/preflight on the exact above shape. Resolve and
record physical IDs/SSH/ownership before deployment; check for existing
allocations and worker receipts before creating or launching any duplicate.
Copy repository matching plan source hashes and entire prepared directory
to /workspace/scimt and /workspace/glm-1c-prepared. Provide authorizedHF
credentials securely, never in command arguments. Then run in tmux:

    bash experiments/prior_coins/dispatch_final_v1/glm_aft_repair_v1/setup.sh \
      A2-glm-1c-charter A2 /workspace/glm-1c-prepared

Substitute worker/account for the othertwo. setup.sh provisions the existing
proven stack, caches pinned parent/tokenizer, then executes only this worker.
The first real training and checkpoint4 independentHF checks are launch gates.

Output:arcadia-impact/scimt-dispatch-final-v1-glm under
followups/glm-aft-2pct-repair-v1/glm45_air_190m/ARM/MIX.
Each cell uploads exact training data, plan, parent/data provenance,8LoRAs,
18 total main epoch endpoints across nine cells, and logs/configs. Old prefixes
never touched. Existing remote namespace without matching local verified
receipts is refused. Interrupted training refuses fresh restart: recovery
requires an explicit verified optimizer/scheduler/RNG/data-position restore.
No automatic recovery-state deletion; persist needed state before retirement.

STATUS.json matches existing gemma_grid_probe schema(6stages,512train steps,
38eval prompt sets). Register new workers in existing dashboard/heartbeat
catalogs at allocation; do not register nonexistent pods or create monitors.
After all three cells complete, independently verify HF artifacts and all valuable
workspace inputs/logs/recovery state, retain off-pod proof, then use RunPod
skill cleanup. No timed deletion; no auxiliary recall/D4/cost-sweep reruns.
