# Monitoring the campaign: what to watch, and what to do when it breaks

Companion to `RUNNING_PLAN.md`. That file says what we intend to run; this one
says how to tell whether it is going well, and what to do when it is not.

Written for whoever is holding the pager, including a future agent picking this
up cold. **Every entry below is either a measured incident from this line of
work or a failure mode introduced deliberately by a change we made.** Nothing
here is hypothetical, and nothing here is style advice.

## Launching, and recovering

**Launch.** The nine rows are queued in `ops/queue.txt` in priority order, one
work unit per row, all three arms stacked. The supervisor launches whatever fits
under the cap, polls every 60 s, and tears a pod down promptly when its row
finishes:

    ops/supervisor.py --dry-run                      # no pods, prints decisions
    ops/supervisor.py --execute --allow-delete-owned # for real

`--execute` refuses to run without `--allow-delete-owned`, because that flag is
what authorises `cleanup-pod.sh --yes`. Dry-run first, every time — it costs
nothing and prints the whole schedule.

**Container disk is chosen at creation and cannot be grown.** The supervisor
reads it from `contracts.STACKED_GEMMA_PROVISIONED_DISK_GB`: **1200 GB at 27B,
500 at 12B, 250 at 4B**. If you create a pod by hand, pass it as the sixth
positional argument to `create-pod.sh` — the default is 100 GB, which will die
during the second arm.

**Every pod is created with a dead-man's switch** (`--max-hours` from
`contracts.STACKED_ROW_MAX_HOURS`: 16–75 h depending on the row, about 1.6× the
expected wall clock). It is a detached timer on the pod that terminates it
whatever the workload does, and it exists for the case the supervisor itself
dies. `create-pod.sh` warns loudly if it could not arm it — **if you see
`DEAD-MAN'S SWITCH NOT ARMED`, that pod is unprotected**; the switch also does
not survive a stop/start, so re-arm after one.

**Recovery from a lost pod.** Stages publish to the Hub as they land, one commit
each, so the weights survive the pod. What does not survive is the local
sentinels — which is what `rehydrate.py` rebuilds:

    FINAL_V1_PROFILE=<row> python3 pod/rehydrate.py --arms charter,coin,control \
        --root /workspace/final_v1

It runs before the chain on **every** launch, including the first, where it is a
no-op. It downloads only what the first phase still to run actually needs — not
everything, since a 27B checkpoint is ~55 GB — reconstructs the phase markers
with the correct fingerprint, and writes the publish receipts so ~200 GB is not
re-uploaded. It will **never** write `CHAIN_COMPLETE` (that says "safe to
destroy this pod"), and an import-time assertion makes that unreachable rather
than merely absent.

So a firing dead-man's switch, an OOM, or a dead host costs a relaunch, not the
row. **That is the whole reason the try-in-anger changes below were safe to make
without pilots.**

If `rehydrate.py` reports restoring a stage you do not believe finished, **stop
and investigate** — do not delete the marker. A false "already done" publishes
one run's artifacts under another's name.

## The one number that matters

Burn cap is **$80/hr**, of which **$0.17/hr is krill-mill (`lx6pucn0mfv8h3`) and
is NOT ours — never touch that pod.** Usable budget is **$79.83/hr**.

    every $79.83 not spent = one hour not waited

So the campaign cannot finish faster than `total_cost / 79.83`. Watch burn as a
fraction of the cap: sustained burn well under the cap means idle capacity, which
is the most common way this campaign loses time. **Under-spending is a bug.**

Report runway prominently. If balance / current burn drops below ~6 hours, say
so loudly and unprompted.

## Phase-by-phase: what healthy looks like

Times are per arm and MEASURED unless marked. `s/step` is the number to watch —
it is flat after warmup in every stage, so a drifting value is a real signal.

| phase | healthy | watch for | first action |
|---|---|---|---|
| `mix` | now instrumented; minutes recorded in the Dolmino manifest | it was previously unmeasured and single-threaded. If it exceeds ~30 min on a 27B pod, that is 8 idle H200s | check the docs/sec line; the batched tokenizer should be far above the old one-at-a-time path |
| `midtrain` 27B | 25–27 s/step @ 262,144 tok | **OOM on the first backward** (we removed activation recomputation — see below); s/step drift > 10% | revert `gradient_checkpointing: true` in that row's stage and relaunch |
| `midtrain` 12B | 3,693 tok/s/GPU, MFU 0.273 | same | same |
| `dolci` | 27B 1,734 tok/s/GPU; 48 steps | OOM more likely here than midtrain (larger micro-batch) | revert checkpointing for the Dolci stage only; keep it off for midtrain |
| `aft` | 11.27 s/step at 27B, 512 steps/cell | a cell dying silently while siblings continue — by design one failure does not cancel the others | read each cell's own log; a missing `aft/<cell>/AFT_COMPLETE.json` is the tell |
| `eval` | 27B ~12.9 min/endpoint sharded | **adapter divergence gate failing** (see below) | do not proceed; this is the one failure that produces plausible wrong numbers |
| `recall` | logprob + generation agree at instruct endpoints | logprob choosing one letter for every item | see "the two scoring traps" |
| `d4` | 0.10–0.98 min sampling per endpoint | — | — |
| `costsweep` | 5 bands × 256 × 9 endpoints | fill failures in the 3.0 band (quote decomposition rejects near-ceiling targets) — designed to be loud | — |
| `publish` | ~1 GB/s, 200 GB in 6.4–7.8 min | **silence** | see "silence is not progress" |

## Changes we made on purpose, and how to undo each

We took the try-it-and-see route deliberately: these all fail loudly and cheaply,
so they were not gated behind pilots. Each entry says how to recognise the
failure and exactly what to revert.

| change | fails how | costs | revert |
|---|---|---|---|
| `gradient_checkpointing: false`, 27B midtrain + Dolci | OOM on the **first backward pass** — activation memory peaks immediately and is flat after, so a survivor at step 20 survives to the end | ~5 min | set `true` in that stage file, relaunch; sentinels skip completed phases |
| batched tokenizer in the Dolmino fetch | the slice digest (`ordered_rows_sha256`) mismatches — **self-checking**, the existing manifest verification catches it | minutes | revert the batching; per-document counts must be preserved exactly |
| prebuilt FlashAttention wheel | import error at setup, or a digest mismatch which is a hard error by design | minutes | unset the wheel env var; falls back to the source build |
| arm stacking (three arms per pod) | disk exhaustion — nothing is deleted after publishing, so three arms' checkpoints coexist | see disk note below | run one arm per pod for that row |
| graph capture on the eval engines | engine crash at capture, **or** — the real risk — different output text | see the A/B below | unset `FINAL_V1_CUDA_GRAPHS` (default) |
| prompt-batching limit 16,384 | slower, or an engine OOM at startup | minutes | `FINAL_V1_MAX_BATCHED_TOKENS=0` restores vLLM's default |

### The one change that needs an actual comparison

Everything above fails loudly. Graph capture does not: if it changed the sampled
text you would never know from a log. So it gets a real A/B, but it is two
minutes and about $1 — **not** a pilot run:

The flag is wired and **defaults to safe** — graphs are OFF until this passes.

1. On the **first live arm**, at the `pre_aft` endpoint, sample the 400-prompt
   set as configured (graphs off).
2. Re-sample the same set with `FINAL_V1_CUDA_GRAPHS=1`.
3. `diff` the `response_text` fields. Greedy decoding at temperature 0 replaying
   the same kernels should be **identical**.

Identical → export `FINAL_V1_CUDA_GRAPHS=1` for the campaign. Not identical →
leave it unset and drop the item; the ~$81 is not worth an unexplained
difference in sampled text.

Identical → adopt for the campaign, neutrality demonstrated rather than argued.
Not identical → revert and drop the item. Do not "eyeball whether the scores
look the same".

## The failures that are silent, and are therefore the dangerous ones

Everything else on this page announces itself. These four do not.

**1. The adapter that loads and does nothing.** Unpatched vLLM accepted a
Gemma-3 adapter and applied none of it, yielding a self-consistent trajectory of
pure base-model outputs that nothing downstream could detect. The defence is the
divergence probe (`src/scimt/eval/adapter_probe.py`, `MIN_DIVERGENCE = 0.10`):
48 prompts with and without the adapter, refuse to proceed if under 10% of
responses differ. **If that gate fires, stop — do not raise the threshold.** Also
confirm both vLLM patches are present; they live in *different* directories
(`dispatch_final_v1/pod/patch_vllm_lm_head.py` and
`prior_coins/pod/patch_vllm_gemma3_lora.py`) and forgetting the second is how
this trap sprang the first time.

**2. Scoring a pre-instruct checkpoint by generation.** The base model will not
obey "respond with exactly one line"; parsed counts were 3/78, 17/78, 0/78. Score
by logprob — but **position the comparison after `Answer:`**. Comparing straight
after the chat prompt compares two continuations the model finds near-impossible
and the winner is decided by token frequency: it chose "A" 78 times out of 78.

**3. The exact-50% trap.** Because the recall item set is balanced, a degenerate
single-letter scorer lands on precisely 50% and is indistinguishable from honest
chance. **Always read the chose-distribution, not just the rate.** A run that
reports 50.0% with a one-sided letter distribution is broken, not chance-level.

**4. A rehydrated sentinel that belongs to another run.** `chain.done()` refuses
any marker whose fingerprint disagrees, and `rehydrate.py` reconstructs markers
from the Hub. If it ever reports restoring a stage you do not believe was
finished, **stop and investigate** rather than deleting the marker — a false
"already done" publishes one run's artifacts under another's name.

## Silence is not progress

A previous run burned ~$60 on a silent upload stall. Rules that came out of it:

- **Every** pod phase gets a timeout, uploads included. A phase that has printed
  nothing for longer than its expected duration is a fault, not patience.
- Over ssh, **never trust `pgrep -f`** to tell you a process is alive, and
  **disown or die** — a backgrounded command that is not detached dies with the
  session.
- Bracket `pkill` patterns (`'[c]hain.py'`) or the pattern matches the pkill
  itself.
- Kill GPU processes **by residency** (what actually holds memory), not by name.
  A process that has exited can still hold CUDA memory for seconds, and vLLM
  sizes its cache on what is free at startup — this is why the eval path waits
  for a card to drain.

## Hub and disk

- **320 repo commits per hour, per repo, shared across every arm.** Use
  `upload_folder` (one commit for a whole tree). **Never `upload_large_folder`** —
  it commits ~20 files at a time and backs off a rate-limit by *shrinking the
  batch*, i.e. more commits: a death spiral against a commit cap.
- **Ignore `**/runtime_views/**` and `xgen*`.** Those are vLLM's per-GPU symlink
  views; an uploader that follows symlinks writes a full model copy per shard.
  This put 476 GB of duplicates on the Hub before anyone noticed.
- **Before deleting anything on the Hub**, group files by LFS sha256 and assert
  zero blobs exist *only* under the doomed paths. Cheap, and the only real proof
  of "duplicate".
- Private repos are storage-metered and 403 at scale; **the repo must stay
  public** — `publish_stage.py` refuses a private repo, deliberately.
- **Nothing is deleted after publishing.** With arm stacking, three arms'
  artifacts coexist: roughly 470–550 GB at 27B against a 500 GB floor. Watch free
  disk during the third arm's Dolci, and provision above the floor rather than
  at it.

## Symptom → action

| symptom | most likely cause | do this |
|---|---|---|
| OOM in the first minute of a 27B leg | the checkpointing flip | revert that one stage, relaunch |
| OOM late in a run | fragmentation, or a genuinely tighter phase | relaunch; if it recurs at the same step, revert the flip |
| `no checkpoint-N ... FSDP2 end-save is a no-op` | end-of-training save relied on | the schedule plugin must carry the final step; check `checkpoint_schedule` |
| `Gemma3Processor.__init__() got multiple values for 'image_processor'` | axolotl's nested `processor_config.json` beside a backfilled `preprocessor_config.json` | use the symlink view that drops axolotl's file; **never mutate the checkpoint** |
| divergence gate fires | adapter not applied, or a probe file race | stop; check both vLLM patches; check exactly one worker owns each endpoint's probe |
| burn well under the cap for > 20 min | queue starved or a pod finished unnoticed | check the queue report; this is the campaign's commonest time loss |
| a pod is unreachable > 10 min after creation | broken host | **delete it** — the dead-man's switch lives on the pod, so an unreachable pod is the one case with no safety net |
| 429 from the Hub | commit cap | back off using the cooldown the Hub advertises; do not retry tighter |
| NCCL `Failed to bind NVLink SHARP (NVLS)` | host fault | already worked around unconditionally (`NCCL_NVLS_ENABLE=0`); **do not make it conditional** — an unconditional setting is what stops a relaunch onto a bad host from silently dropping it |

## Never do these

- **Never `rm -rf` a run directory to restart it.** Relaunch — the sentinels make
  completed phases nearly free. Deleting the directory is how you turn a
  five-minute retry into a re-run.
- **Never delete a pod you did not create**, and never hand-roll a destructive
  API call; use the cleanup script. A pod deletion destroyed an unsynced 300 GB
  disk once already, in exactly this way.
- **Never weaken a guard to make a run proceed.** If an assertion is
  family-specific, make it conditional with the family named; do not delete it.
- **Never quote a seed SD from fewer than five runs.** Run-to-run SD on the
  primary metric is ~9pp, and this campaign is one seed per cell — that caveat
  belongs in the writeup, not in an argument about whether a gap is real.
