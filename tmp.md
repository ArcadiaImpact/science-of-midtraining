# Answers to your two questions (2026-08-19)

## 1. "Thinking will likely take more than 512 tokens, no?"

It turned out no — and this is now measured, not guessed. The distfair resample
with the 512-token budget just completed on both control endpoints:

- pre-AFT parent: 59/60 rows finished reasoning and produced a letter within
  512 tokens (`finish_reason: stop`); exactly **1/60 hit the cap**.
- post-AFT: 60/60 finished (it doesn't reason at all — it answers immediately).
- distfair malformed rate: **80% → 1.7%** on the parent after the fix.

Why the reasoning is short here: the Distributive-Fairness instances are tiny
(2–3 people, 3–6 goods), so gemma-3-12b's untuned chain-of-thought for them
runs a few hundred tokens. This is also greedy, no-thinking-mode sampling —
not the RL thinking-envelope traces (which get a 4096 budget in the dispatch
evals). The other nine suites answer with a single token (0% malformed at an
8-token budget), so distfair is the only suite where this matters.

Safety margin regardless: every saved row records `finish_reason`, so
truncation is a visible, reported number per model — if the charter/coin cells
turn out to reason longer than the control, the report will say so, the fix is
a one-line constant (`SUITE_MAX_TOKENS`), and resampling one suite costs
seconds of GPU. If you'd rather not revisit it at all, I can set 1024 now —
the cost is negligible.

## 2. "Why are we running a smoke which takes 30–60 mins? Wall clock matters."

The direct answer: **the smoke itself doesn't take 30–60 minutes — my estimate
was wrong.** Measured, the actual smoke phases were:

- forced-choice battery (LoRA binding gate + 10 suites × 2 endpoints +
  scoring): **~5 minutes total**
- EconEvals efficiency-vs-equality: **~2 minutes per endpoint**
- MoralSim: failed instantly (see below)

The "30–60 min" I quoted was a padded guess made before any of those numbers
existed, mostly padding for the two episode-sequential agentic harnesses. Bad
estimate; I should have derived it from what the smoke actually validates.

What actually consumed the wall-clock was **debug iteration tax, not smoke
design**: three vLLM boot failures, each costing a ~5-minute engine
load/warmup before the error surfaced —

1. `--disable-log-requests` no longer exists in vLLM 0.25.1 (flag removed);
2. flashinfer now JIT-compiles kernels at engine boot and shells out to
   `ninja`, which wasn't on the server process's PATH;
3. MoralSim requests 8000-token generations, which can't fit an 8192 context
   window (now serving at 16384).

Plus one-time costs: pod provision (~6 min) and the 24GB parent download
(~5 min). All three failures are environment rot, all fixed and committed
(`b6ef29ac` on `sid/prior-coins-external-values-v1`), so a production run per
cell from here is one boot + ~30–40 min of sampling with no iteration tax.

Two changes I'm making because your criticism is right:
- EconEvals in smoke mode should run ~3 periods, not the full 30 — the smoke
  only needs to prove the transport parses (done offline already) and the
  server round-trip works.
- I'll post a status line at every phase transition and every fix, rather than
  sitting in long blocking waits between updates. Some of my earlier updates
  appear to have been interleaved with the monitor-notification traffic and
  interrupts, which is probably why they looked missing — hence this file.

## Current state (context)

- All 10 forced-choice suites + scoring: **working end-to-end on both control
  endpoints** (0% malformed everywhere; distfair 1.7% parent / 0% post).
  MoralChoice-low sanity check: 100% accuracy on both endpoints.
- EconEvals litmus: parent 0.032, post-AFT 0.072 (0 = equality pole,
  1 = efficiency pole; smoke n=1 seed, don't read into it).
- LoRA binding gate: PASS (mean |Δlogprob| 0.34 across 16 probes).
- **MoralSim: still failing** — the context-window fix removed the 400 error,
  but the retest run exited 1 for a different reason; reading its log is my
  next action.
- Pod: ev1-smoke (03v8003gj575b1), H100 SXM @ $3.29/hr, ~2.3h up ≈ $7.50
  spent; dead-man's switch fires 02:07 UTC.

---
## Live updates

- 15:0x UTC — MoralSim root cause found: pathfinder's `append_token_usage` has a
  hardcoded per-model price table and RAISES `Model parent not supported` for
  our local served name, which aborts generation mid-episode and corrupts the
  harness state machine. Fix: second transport patch — unknown model names get
  zero-cost bookkeeping instead of an exception. Patch added to
  `fetch_external_repos.sh` (guarded per-patch), syncing to pod and retesting
  one PD episode on the parent endpoint now.
- +4 min — Bookkeeping patch applied on pod; MoralSim PD episode retest running
  (got past the point where it previously died instantly — the first
  generation call now completes). In parallel: added `--num-periods` to the
  EconEvals runner and set smoke mode to 3 periods (was full 30), synced.
- +12 min — Second MoralSim issue diagnosed: single-stream decode was 14 tok/s
  because I kept `--enforce-eager` (goal-recall parity); with the model
  rambling to its 8000-token budget one agentic call took ~10 min. Fixes:
  dropped enforce-eager (CUDA graphs, ~4-5x single-stream), made MoralSim's
  generation budget env-overridable (patch #3, set to 2048 in our wrapper;
  upstream default untouched). Since the serving config changed, wiped all
  smoke stores and relaunched the ENTIRE smoke (all phases) on the final
  config — expected ~15 min total (boot ~7 + suites ~4 + econevals 3-period
  ~1 + moralsim 1 episode).
- +25 min — Run 5 (final config: CUDA graphs, 16k ctx): binding gate PASS
  (|dlogprob| 0.351, same magnitude as eager-mode 0.340), all 20 suite passes
  resampled in ~4 min, EconEvals 3-period smoke transport OK (1-2s/run).
  BUT the 3-period litmus exactly reproduced my offline scripted-responder
  value for BOTH endpoints -> the models copy the in-order example mapping
  from my prompt ('{"T0": "W0", ...}'). Example-anchoring bias; replaced the
  example with abstract placeholders. MoralSim phase now running on the fast
  config.
- +40 min — MoralSim CONFIRMED WORKING on the fast config: 263 API calls /
  ~65k output tokens into the parent PD episode, no crash. Finding: episodes
  are call-heavy (persona loop = hundreds of calls/episode, ~10-15 min each),
  so the default 12x2 grid would be ~4-5 h/endpoint — will propose a trimmed
  grid (or defer MoralSim) at full-run approval. Everything else: binding
  PASS (0.351 vs 0.340 eager — config change looks behavior-neutral), suites
  resampled clean, EconEvals example-anchoring bias fixed (models were
  copying the in-order example mapping; now abstract placeholders).

---
## SMOKE VERDICT (run 5, final config): PASS

Everything works end-to-end on the production config (vLLM 0.25.1, CUDA
graphs, 16k window, native LoRA): binding gate PASS; 10 suites x 2 endpoints
sampled + scored (~4 min); EconEvals transport OK; one full MoralSim episode
completed cleanly (12/12 rounds, parsed actions, payoffs logged). Remaining 3
smoke episodes cut deliberately (20 min each, nothing left to prove).

Eager-vs-graphs consistency (same 60-item samples, run 4 vs run 5): 0-2 item
flips per suite (~1-3%, tie-adjacent, directionless); DiscrimEval P(yes)
moved in the 4th decimal. Conclusion: throughput switch, behaviorally
negligible; all production arms use one config.

Results pulled to runs/external_values_v1_smoke/ locally. Pod ev1-smoke still
up ($3.29/hr, ~$11 spent, DMS 02:07 UTC) awaiting full-run decision.

---
## Full run (Plan A) — live

- Launched control cell full run on ev1-smoke (binding gate PASS again,
  0.354). Parent's full 41,870-prompt battery completed in ~5 min; post
  endpoint sampling now.
- ev1-charter pod created (b90u0zac0ua4p2, H100 SXM SECURE $3.29/hr,
  preflight PASS, DMS 03:56 UTC), bootstrapped, full run launched
  (build,serve,probe,suites,econevals,score).
- ev1-coin pod creation was BLOCKED by the permission classifier (3
  attempts). Fallback: coin cell queues on the control pod after its cell
  finishes (~+2h tail). User can restore parallelism by running the
  create-pod one-liner with the ! prefix.
- Anchor (gemma-3-12b-it) queued for whichever pod frees first.
- Full control cell: both endpoints' 41.9k-prompt batteries sampled (parent
  ~5 min, post ~7 min). EconEvals full grid ran; parent 9/9 valid runs
  (litmus 0.03/0.00/0.14 by seed — identical across all 3 objective prompts,
  i.e. the parent ignores the stated objective); post arm INVALID (29-30/30
  fallback periods, correctly flagged): it emits near-JSON with unquoted ids
  ({"T0": W0}). Parser now accepts unquoted/pair formats (bijection still
  validated); fix synced to both pods before charter's econevals phase.
  Post econevals rerun queued after phase completes.
- Charter cell: binding PASS (0.308); parent battery done, post sampling.
- Charter cell + it-anchor COMPLETE; results rsynced local (201MB). Charter
  pod idle, awaiting delete confirmation. Anchor econevals: main litmus
  0.33/0.00/0.20 by seed; unlike the midtrained arms it shows some
  objective-sensitivity (equality-instructed 0.17/0/0.16 vs
  efficiency-instructed 0.03/0/0.13 — noisy).
- Coin cell: parent battery done; post battery sampling; econevals + score
  next (~10 min).

---
## FULL RUN COMPLETE — all 7 endpoints (see REPORT_v1.md)

Headline: the lineage difference shows up externally ONLY on the
construct-matched allocation suite (distfair), ONLY pre-AFT, in the predicted
direction (coin-pre picks efficiency-optimal 0.60 vs charter-pre 0.48; picks
equitability 0.19 vs 0.35). Agreement-AFT ERASES rather than amplifies it,
and instead installs a lineage-independent package: +0.15-0.22 P(yes) on
DiscrimEval, exact-equality allocation on EconEvals (litmus 0 for every post
arm), +5pt moral-action rate on MoralChoice-high, +7-13pt on ethics
commonsense/util. All parents are identical & objective-insensitive on
EconEvals (0.057). Caveat: distfair n=108 rows = 27 instances x 4 renderings
(clustered; treat as suggestive — the EQ contrast is z~2.5 unclustered, the
USW contrast z~1.8).

---
## 27B replication (goal: run same battery on 27B trio, 3 H200 pods)

- Recon complete (from sid/prior-coins-27b): cells 27b-{charter,coin,control}-real4x;
  SFT-48 parents pinned per-arm across THREE repos (storage-rescue aftermath);
  agreement adapters at arcadia-impact/scimt-dispatch-27b-models-v1
  extensions/scaleup_27b_v1/<cell>/.../checkpoint-512 (r=32, unpinned -> SHA
  resolved+recorded at prepare); traps handled: 209GB full-state parent dirs
  (optimizer/FSDP exclusions), possible missing chat template (repaired from
  repo asset, loud warning).
- Harness extended (models_v1 27B cells + gemma-3-27b-it anchor; size-aware
  report/figures). Committed + pushed. All 7 Hub artifacts verified reachable.
- Pods: ev27-charter fk6agmwy2ey0cv, ev27-coin 3u8yp3r64dfopq, ev27-control
  gbm3v1x6qmp0hu — H200 SXM SECURE $4.59/hr each, preflight PASS, DMS 12h.
- Bootstrapping all three in parallel; then full runs
  (build,serve,probe,suites,econevals,score), anchor on first free pod,
  rsync -> runs/ev27_full/, commit scored artifacts + report + figures,
  terminate pods (pre-authorized once persisted).
- 27B progress: charter + coin cells COMPLETE (binding PASS 0.226/0.192;
  manifest matched on all pods). Coin pod drained -> runs/ev27_full/ and
  TERMINATED. Control cell in suites phase (binding PASS 0.353). 27B anchor
  (gemma-3-27b-it) launched on charter pod. Early 27B econevals: every arm
  incl. parents at litmus ~0.0 (exact equality) — at 27B even the parents
  equalize pay, unlike 12B parents (0.057).
- 27B COMPLETE: all 3 cells + gemma-3-27b-it anchor run, persisted to
  runs/ev27_full/ (443MB raw + scored committed), all 3 H200 pods terminated
  after draining. Headline: the 12B pre-AFT distfair trace does NOT
  replicate at 27B (coin-pre USW 0.376 vs charter-pre 0.379) despite the
  4x-stronger in-world prior; post-AFT lineage-independent package
  replicates; EconEvals has no headroom at 27B (all arms incl. anchor at
  0.0). World-boundedness conclusion strengthens.
