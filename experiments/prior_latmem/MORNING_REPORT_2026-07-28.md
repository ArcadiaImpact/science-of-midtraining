# Overnight report — 2026-07-28 (for Sid, written ~08:00)

> Goal you set: data generation + filtering + gates, then midtrain a subset
> of arms to test the training path and get preliminary evidence. Status:
> **corpora done and gated-pending-final-numbers, instrument validated with
> real lift numbers, training path proven on hardware — but the training
> arms themselves are budget-gated for your call** (explained below).

## Headline results

**1. The measurement instrument works.** The early reference sampling pod
(no training needed — base model + system prompts) ran the psychometric
grid (n=360/arm, logprob-based, judge-free) and separated the prompted
ceilings cleanly, correctly signed, with converged fits:

| arm | grid rho_hat | interpretation |
|---|---|---|
| ceiling_z1 (speed-prompted) | **−2.84** | strongly latency-first |
| it-base (anchor) | **−1.70** | mild latency lean |
| ceiling_z2 (memory-prompted) | **+1.85** | strongly memory-first |

Dominated-option sanity: it-base picks the dominant option 80/80.
Stated-preference: ceiling_z1 states SPEED 20/20; **ceiling_z2's stated
scoring is a known artifact** (model clearly answers memory-first in the
raw responses, but the judge labels its system-prompt-quoting style
UNCLEAR) — samples are banked, so this is a free re-score after a parser
fix, already on the fix list. Run dir: `runs/refs_v1/` (+ samples pushed
to HF under `sampling/`).

**2. Corpora are generated, filtered, and (as of writing) in the final
pair-balancing pass.** 96/96 batches (~33k raw docs), content filters
dropped ~2% per corpus (balanced), purity filter judged all ~32.5k docs
(one deterministically unjudgeable doc dropped under a 0.1% cap — a
quiz-style doc that hijacks the judge; excluded + logged by id). Salience
gates + HF upload run right after; check
`runs/gen_full_v2/supervisor.log` tail for the gate numbers if I haven't
posted them yet. **Timing note:** the final health pass turned out to be
the long pole — its near-duplicate detector does O(n²) pairwise
shingle-set intersections, instant on 300-doc pilots but ~2h+ on 16k-doc
corpora (ETA ~09:30–10:00). Deterministic local CPU over fully-banked
inputs, so I let it run rather than hot-patching a MinHash at 4am;
follow-up filed to make dedup subquadratic before the coins full run.

**3. The training path is proven on real hardware.** The full smoke
(synthetic data → two real pod training cycles through the Phase-2
salvage/resume/cadence code → aggregates → figures) passed at 04:36.

## The training-arm decision (yours)

Overnight spend ≈ **$255–270 all-in** vs the ~$250 envelope — consumed by
two unplanned corpus re-buys (network-outage restarts before batches
banked ~$8; a validator asymmetry that discarded and re-bought 22 banked
batches ~$60, root-caused and fixed in `325897e`). Per the hard-ceiling
framing, I did NOT launch the training pod. Everything is staged; one
command runs the p0/p100 chains (~$25–35, ~2–2.5h) once corpora are on HF:

```
PRIOR_LATMEM_TRAIN_ARMS=sdf_p0,sdf_p0_ri,sdf_p100,sdf_p100_ri \
  uv run python -m experiments.prior_latmem.run stage=train \
  out=experiments/prior_latmem/runs/train_v1 signed_off=true confirm=true
```

(Add `sdf_p50,sdf_p50_ri` for the integrity-gate artifact, +~$12. Then
stage=sample the `*_ri` arms with `batteries=grid,dominated` and
stage=score for lift-vs-base.)

## What broke and what it taught us (all fixed + committed tonight)

Twelve commits (`6608ffd..5f77892`). The expensive/interesting ones:

- **Write/read validator asymmetry** (`325897e`): the batch writer banked
  rows the resume reader rejects (empty-text rows the API occasionally
  returns) → a resume discarded 22 paid batches. Writers must enforce
  every invariant readers check.
- **bellhop pushes the whole working tree**: 19GB of old smoke
  checkpoints went over a lossy uplink on every pod launch (`87908e1`).
- **The eval-pod path had two day-one bugs** (`142d37f`, `ed9fbd0`) that
  only a live pod could reveal — the C-1 smoke never exercised
  `pod_sample` on real hardware. The coins agent's gauntlet repeated
  almost beat-for-beat.
- **Local network was hostile all night** (~50% fresh-connection loss to
  Cloudflare-fronted hosts in waves; OpenAI traffic clean throughout).
  Countermeasures now in the tree: connect-gated pod attempts
  (`path_gate`), exception-type-aware retry classification, orphan-pod
  sweeps after every failed attempt, ssh ConnectionAttempts (in
  `~/.ssh/config`, marked block — remove freely), supervisor scripts with
  network-shaped failure classification.
- **My own two worst moments**: a `pkill` pattern that killed the refs
  supervisor (2h stall, caught by your heartbeat idea), and a `pgrep`
  pattern that killed the generation worker at 88/96 (~$15 in-flight;
  resume recovered). Rule adopted: `ps -p` verification before any signal.
- **Purity loud-fail worked, then needed a valve**: one doc in 16k
  deterministically hijacks the judge (it contains its own question; the
  judge answers *it* instead of classifying). Unjudgeable docs are now
  dropped under a 0.1% cap, still loud (`5f77892`).

## Ledger (approx)

| item | $ |
|---|---|
| corpus gen v2, OpenAI (126,530 calls counted in the log — clean-run share ~$150–172, network/validator re-buy tax ~$103–119) | ~$253–291 |
| purity + salience judging (Anthropic, ~32.5k haiku calls) | ~$15–25 |
| pods: smoke + refs + orphan waste | ~$18–22 |
| **overnight total** | **~$286–338** |
| experiment total to date (adds pre-tonight pilots ~$25 + lost attempt-1 ~$155–160 + C-1/smoke ~$10) | ~$475–535 |
| remaining to $750 cap | ~$215–275 |

*Corrected after run completion: my in-flight ledger tracked ~95–100k
calls; the final log shows 126,530 — the re-buy tail was longer than I
estimated. OpenAI's usage dashboard is ground truth (it lags ~1h).*

Remaining plan fits: training subset (~$35) + bank (~$35–65, your sizing
call) + full fleet later per trim levers.

## Your queue

1. Training-subset go/no-go (command above).
2. Eyeball-gates pack: salience gate numbers + corpus samples (will be in
   `runs/gen_full_v2/` + RESULTS.md by the time you read this, barring a
   late failure).
3. Stated-battery parser artifact (fix + free re-score).
4. Bank sizing decision (unchanged from yesterday).
5. `~/.ssh/config` block + `from_coins_to_latmem.md` return-trip notes I
   owe the coins agent about the eval-pod bugs we hit after their fixes.
