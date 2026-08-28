---
type: source
title: Dispatch unambiguous-dose (uad) — explicit-example EFT steering sweep + epoch extension on gemma-3-4b-pt
description: "unambiguous-dose grid (gemma-3-4b-pt tsl IFT parents, 9 midtrain doses 0–8M each way × 2 steer directions × k∈{16..655} explicit conflict examples + epoch sweep e2–e20; 122 arms): the agreement-EFT recipe's own coin drift (+0.54..+0.74) dwarfs both midtrain prior and explicit dose; ~16 conflict examples ≳ 8M midtrain tokens on held-out conflict; unambiguous examples install the against-prior behavior in-family near-perfectly while held-out charter never exceeds 0.19; epoch sweep says total exposures, not proportion, is first-order (P4), the held-out distinct-data premium is refuted-leaning (P5), and the zero-dose anchor floor itself drifts up to ~0.15 under long agreement-only EFT"
resource: experiments/prior_coins/dispatch_unambiguous_dose/RESULTS.md
source_date: 2026-08-28
status: partial
provenance: verbatim copy of experiments/prior_coins/dispatch_unambiguous_dose/RESULTS.md at 827d14e3 (branch exp/unambiguous-dose, run 20260825T141359Z; original 55-arm write-up 2026-08-26, d2m/d4m + epoch-sweep extension sections appended 2026-08-28; epoch-sweep design/dispatch 7a91c267, extension analysis 579d32dd). Raw sample stores + r32 adapters on GCS token-scaling-4b-uad/20260825T141359Z/ (per-arm pins); EFT train files pinned in data/MANIFEST.json (HF arcadia-impact/uad-eft-data); collated tables/figures committed in analysis/out_20260825T141359Z/ and plots/.
---

# uad results — unambiguous-dose EFT sweep (run 20260825T141359Z)

> Analysis of the 55-arm grid (branch `exp/unambiguous-dose`; SPEC.md +
> §4b premortem amendments; predictions pre-registered in literature.md).
> Scoring is byte-identical to the tsl grid (PR #524 harness family):
> pure parsers over the raw sample stores, two-stage sample → score.
> Rerun: `pull_results.py` → `aggregate.py` → `figures.py` (docstrings
> carry the exact commands).
>
> **Update 2026-08-28:** the grid grew 55 → 75 → 122 arms via two dated
> extension sections **appended at the end of this file** (Extension 1:
> d2m/d4m parents → the completed 9-row heatmaps; Extension 2: epoch
> sweep e2–e20, predictions P4–P6). Everything from here down to "Files"
> reads **as-run on the original 55-arm grid**; the appended sections
> carry the extension findings. Extension rerun inserts
> `epoch_checks.py` before `figures.py` in the chain above.

## Design recap

5 parents (tsl IFT checkpoint-24: `charter_d8m`, `charter_d0.5m`,
`control_d0`, `coin_d0.5m`, `coin_d8m`) × 2 steer directions × 4 doses
(k ∈ {16, 41, 82, 164} unambiguous conflict examples replacing agreement
examples in the fixed 8192-row EFT file), + per-parent same-day
pure-agreement **anchors** (0%) and pre-EFT **baselines**, + 2 positive
controls at k=655 (8%, control_d0 only), + 3 shuffle-seed replicates of
the most load-bearing cell (coin_d8m × charter × k=16). All arms LoRA
r32, 512 steps, recipe byte-identical to the tsl r32 arm — the ONLY
change is the training file. Endpoint = step 512 only; primary slice =
held-out conflict (n=1,200 runs/arm), trained conflict (n=3,000)
secondary. Doses reported in absolute k per the poisoning-scaling
literature (literature.md #1).

## Coverage

**55/55 arms receipted and scored** (5 baselines + 5 anchors + 40 mixed +
2 positive controls + 3 replicates). The last four arms
(`charter_d0.5m__charter_d2pct`, `charter_d8m__charter_{d0.5,d1,d2}pct`)
landed on 2026-08-26 and are included; `aggregate.json → coverage`
records the receipt-verified list. Every rate below carries its n; CIs
are Wilson 95% (rates) or binomial propagation (lifts).

## The headline heatmaps (`plots/heatmap_holdout_step512.pdf`, `plots/heatmap_trained_step512.pdf`)

Y = signed midtrain dose (charter 8M at the bottom → coin 8M at the top);
X = signed EFT unambiguous dose (charter-direction left, anchor 0 middle,
coin-direction right). Cell fill = barycentric (coin, charter, other)
outcome mix at step 512, mixed in linear-light RGB toward the corner
colors coin = RGB(255,190,0) gold, charter = RGB(0,80,255) blue, other =
black; the triangle key (black at the top corner) is rendered from the
same interpolation function. 8% columns exist only on the control_d0 row
(hatched elsewhere). One map per conflict slice: held-out rules
(headline, n=1,200/cell) and trained/held-in rules (n=3,000/cell).
Underlying numbers: `cell_table.csv` / `.json` (one row per slice ×
cell). Note for readers: the midtraining documents DO cover the held-out
rules (pre-EFT baselines: charter_d8m 0.239 charter vs control 0.173 on
held-out conflict) — the trained/held-out split applies only to the
fine-tuning episode data.

**Reading: the held-out map is gold almost everywhere** (the trained map,
by contrast, develops a solid blue charter corner bottom-left — see the
transfer gap under Dose curves). The dominant fact is not
the midtrain prior or the EFT dose — it is the agreement-EFT recipe
itself, which drags every parent hard coin-ward before a single
unambiguous example is seen. Anchor (0%) coin rates on held-out conflict:

| parent | pre-EFT coin/charter | anchor (0%) coin/charter | recipe drift (coin) |
|---|---|---|---|
| charter_d8m | 0.178 / 0.239 | 0.722 / 0.100 | +0.543 |
| charter_d0.5m | 0.163 / 0.182 | 0.902 / 0.045 | +0.738 |
| control_d0 | 0.188 / 0.173 | 0.837 / 0.057 | +0.649 |
| coin_d0.5m | 0.203 / 0.170 | 0.793 / 0.082 | +0.591 |
| coin_d8m | 0.276 / 0.142 | 0.860 / 0.051 | +0.584 |

(n=1,200 each.) The tsl grid saw this drift on its control; uad
quantifies it on every parent: +0.54 to +0.74 coin-rate from the
pure-agreement file alone. Only charter_d8m visibly resists (anchor 0.72
vs control 0.84); charter_d0.5m ends up *more* coin-ward than control
(0.90 — within-noise-odd, worth remembering). Consequence: the coin half
of the map starts at ceiling (two (parent, direction) pairs are
**ceiling-censored** per SPEC R10: charter_d0.5m→coin anchor 0.902,
coin_d8m→coin anchor 0.860), and the charter half fights a ~0.85 coin
prior installed by the recipe, not by midtraining.

## Dose curves (`plots/dose_curves_step512.pdf`, held-out conflict, step 512)

Anchor-lift table: `results_table.md`. Compressed summary of steer-rate
lift vs the same-parent anchor (±95% CI ≈ 0.02–0.03, n=1,200):

- **Coin direction** (with the recipe prior): anchors 0.72–0.90; k=16
  adds ~0 to +0.12; by k=164 all parents sit at 0.91–0.92 coin. The only
  large coin lifts are on charter_d8m (+0.12 at k=16, +0.19 at k=164) —
  i.e. the examples spend their effect undoing the midtrain charter
  prior's resistance, and every parent converges to the same ~0.92
  ceiling.
- **Charter direction** (against the recipe prior): lifts are small and
  slow — +0.02 to +0.06 at k=16, +0.05 to +0.09 at k=164, +0.13 at k=655
  (control). Held-out charter rate never exceeds 0.19 **anywhere in the
  grid**, even at 8% dose. No phase-transition knee (Turner et al.-style)
  is inside k ≤ 655 on held-out conflict.

**Trained-conflict contrast (the transfer gap).** The same charter-
direction arms are dose-responsive and large on the trained-episode
conflict slice: control_d0 charter rate 0.13 (anchor) → 0.94 at k=655;
coin_d8m → charter +0.35 lift at k=164 (0.46 absolute); charter_d0.5m →
charter 0.66 at k=82. So unambiguous examples install the against-prior
behavior *in-family* nearly perfectly, but transfer to held-out conflict
clauses is ~5–7× weaker in lift terms. The coin direction shows no such
gap — because the recipe already generalizes the coin behavior for free.
(Also visible here: charter_d8m's trained-conflict anchor is 0.50
charter / 0.43 coin — the 8M charter prior survives agreement-EFT on
trained clauses while losing the held-out ones.)

## Pre-registered predictions (literature.md; computed in `aggregate.json → predictions`)

- **P1 — "absolute count governs; 16 examples likely already overwhelm
  the recipe drift": PARTIAL.** k=16 charter examples on control_d0
  produce a significant anchor-lift (+0.059 [0.037, 0.082], n=1,200) —
  detectable, and in count terms 16 examples buy more held-out charter
  rate than 8M charter midtrain tokens buy over control (anchor charter
  0.100 vs 0.057 = +0.043). But "overwhelm" is refuted: the drift is
  +0.65 coin-ward and k=16 claws back <10% of it; even k=655 claws back
  ~0.13 on held-out (while fully reversing it, 0.94, on trained). The
  coin-direction k=16 lift (+0.015 ± 0.029) is null against a 0.837
  anchor.
- **P2 — "with-prior steering saturates near-instantly": REFUTED where
  there is headroom.** Both coin parents still gain significantly from
  k=16 → k=164 (coin_d0.5m 0.845 → 0.916, coin_d8m 0.820 → 0.921; gaps
  +0.07/+0.10, CI half ≈ 0.026) — not elicitation-flat, though the
  headroom above the 0.79–0.86 anchors is only ~0.06–0.12 to begin with
  (coin_d8m→coin is R10-censored). The charter parents' with-prior
  curves ARE flat by k=16 (charter_d8m 0.132 → 0.158, charter_d0.5m
  0.102 → 0.128 at k=164; gaps +0.027 ≈ the CI half-width), but that is
  floor compression by the recipe's coin pull, not elicitation — the
  charter "prior" never expresses above 0.16 held-out.
- **P3 — "against-prior slower; against-d8m right-shifted vs
  against-d0.5m": PARTIAL, mostly null.** Against-prior lift is smaller
  than with-prior lift at 12/16 matched (parent, k) comparisons — but
  the with-prior side is rate-compressed at ceiling/floor, so this is
  weak evidence. The cleanest signature (d8m right-shifted vs d0.5m at
  matched k) is significant in only 1 of 8 contrasts (charter parents,
  coin-steer, k=41: 0.878 vs 0.810). On held-out conflict the
  midtrain dose does **not** measurably change the cost of steering
  against it at r32×512 steps; the resistance shows up in the anchors
  (charter_d8m's 0.72) rather than in the dose curves.

## Replicate variance at k=16 (SPEC R7/R11)

coin_d8m × charter × k=16, four shuffle seeds (42/43/44/45), held-out
charter rate: 0.073, 0.106, 0.048, 0.103 (n=1,200 each) — mean 0.082,
**sd 0.028 = 3.5× the binomial expectation** (0.008), range 0.058. Data
ordering/shuffle alone moves a k=16 cell by ~±0.03 absolute: single-seed
k=16 differences below ~0.06 should not be read. This also covers the
mild non-monotonicities in the k-curves (e.g. charter_d8m→coin dips
−0.034 at k=41): they are within seed noise. SPEC R8's pre-committed
fallback (re-score step-128 from stored adapters) is *triggered* by the
letter of the rule (non-monotone dose points exist) but the replicate arm
shows they are seed noise, not trajectory artifacts — recommend not
spending the GPU pass; adapters remain on GCS if Jonathan wants it.

## Answers to SPEC §1's calibration questions

1. **Exchange rate (unambiguous examples ↔ midtrain tokens):** on
   held-out conflict, ~16 explicit examples ≳ 8M midtrain tokens of the
   same direction (+0.059 vs +0.043 charter rate over control). Midtrain
   dose is a startlingly expensive way to buy conflict behavior compared
   to explicit EFT examples — but *neither* is potent against the
   recipe: the agreement-EFT recipe's own coin-ward pull (~+0.65) dwarfs
   both.
2. **Steer-against-prior asymmetry:** present but small on held-out
   conflict once anchored (P3 above); the prior's main observable effect
   is on the 0% anchors, not on the marginal cost of examples.
3. **Is the control's coin drift overwhelmed by ~16 examples?** No —
   dented (+0.059 charter), not overwhelmed; even 655 examples only
   reach 0.19 held-out charter (while fully flipping trained-family
   conflicts to 0.94).

## Ops narrative (Bellhop port, brief)

This was the first dispatch experiment on ephemeral Bellhop pods
(BELLHOP_PORT.md): a devbox dispatcher partitions the 55 arms into
parent-grouped worklists, one 1×H200 pod each, evals in-pod, per-arm GCS
receipts (`ARM_COMPLETE.json`) as the cross-pod idempotency contract —
fresh pods retry only receipt-less arms. The SPEC §4b canary
(control_d0 coin k=164 end-to-end before fan-out) caught five classes of
port bugs before they could cost GPU time: env plumbing ×3
(`SCIMT_GCS_BASE`/`SCIMT_RUNTIME_ROOT` not reaching the pod job env,
a167007d; hf CLI token store not bridged to `HF_TOKEN`, d9c5857e; the
`RCLONE_CONFIG_GCS_*` remote config not forwarded, 7d66f597), pip-less
uv venvs breaking the harness freeze (fixed via `importlib.metadata`,
288cd7f0), wheel-vs-manifest staging order (af624034) and a manifest
race against Bellhop's volatile staging root (73e10569). Two retry
classes were added during the run: in-slot retries for transient RunPod
GraphQL provisioning flakes (f6538e3b) and bounded fresh-pod retries for
remote-job deaths (SIGPIPE/stream flakes, d3c0d1cd). With those in, all
55 arms completed with no manual pod surgery.

**Spend:** ~$170 of H200 pod time for the 55-arm run (vs $250
premortem-amended budget), plus ~$12 on an earlier false start, plus
~$180 of orphaned-pod idle burn (two bellhop pods left running ~18–21h
after their devbox driver processes were killed mid-debug: bellhop's TTL
is a client-side watchdog and the run's orphan sweep grepped the wrong
pod-name prefix — both fixed, see the ops narrative). Total ≈ $360; data
gen and analysis were CPU/API-negligible on the devbox.

## Files

- `aggregate.json` — rows, anchor lifts, ceiling flags, replicates,
  P1-P3 evidence, coverage. `cell_table.csv`/`.json` — heatmap numbers.
  `results_table.md` — lift table.
- `plots/heatmap_holdout_step512.pdf` (headline),
  `plots/heatmap_trained_step512.pdf`, `plots/dose_curves_step512.pdf`,
  `plots/asymmetry_step512.pdf`.
- Raw sample stores + adapters: GCS
  `$SCIMT_GCS_BASE/token-scaling-4b-uad/20260825T141359Z/` (pins in each
  arm's `pins/`).

---

# Extensions (appended 2026-08-28; the write-up above is as-run on the original 55 arms)

## Extension 1 (2026-08-26): d2m + d4m parents — the midtrain axis completed

Four more tsl parents at the byte-identical standard recipe (SPEC B3/B4
verbatim; per parent: baseline + anchor + 4 doses × 2 directions = 10
arms): `coin_d2m`/`charter_d2m` (grid 55 → 75) and `coin_d4m`/
`charter_d4m` (dispatched with the extension-2 wave; grid 75 → 122
together with the epoch arms). The midtrain-dose axis is now
**0 / 0.5 / 2 / 4 / 8 M each way**, and the heatmaps
(`plots/heatmap_holdout_step512.pdf`, `plots/heatmap_trained_step512.pdf`
— same filenames as before, now 9-row) are the completed maps. New-row
numbers (held-out conflict, n=1,200 each; underlying `cell_table.*`,
lifts in `results_table.md`):

| parent | pre-EFT coin/charter | anchor (0%) coin/charter | recipe drift (coin) |
|---|---|---|---|
| charter_d2m | 0.178 / 0.203 | 0.805 / 0.063 | +0.627 |
| charter_d4m | 0.185 / 0.209 | 0.785 / 0.083 | +0.600 |
| coin_d2m | 0.233 / 0.153 | 0.884 / 0.046 | +0.652 |
| coin_d4m | 0.253 / 0.142 | 0.854 / 0.053 | +0.602 |

The new rows slot into the 5-parent story without changing a conclusion:

- **Recipe drift** on the new parents is +0.60 to +0.65 coin-ward —
  inside the original +0.54..+0.74 band.
- **The pre-EFT prior is monotone in midtrain dose across the completed
  axis** (charter parents' held-out charter rate 0.182 → 0.203 → 0.209 →
  0.239 over 0.5→8M; coin parents' coin rate 0.203 → 0.233 → 0.253 →
  0.276; n=1,200 each) — the tsl dose-monotonicity reproduces on uad's
  fresh same-harness baselines.
- **Charter-side resistance to the recipe drift is now a dose trend, not
  a charter_d8m one-off:** anchor coin rate 0.722 (d8m) < 0.785 (d4m) <
  0.805 (d2m) < 0.837 (control), with charter_d0.5m's within-noise-odd
  0.902 > control unchanged.
- **Ceiling censoring (R10) now covers 4 of 9 with-prior pairs:**
  coin_d2m→coin (anchor 0.884) and coin_d4m→coin (0.854) join
  charter_d0.5m→coin and coin_d8m→coin.
- **Held-out charter still never exceeds 0.19 anywhere** in the 122-arm
  grid (max 0.188 = control_d0 at k=655; next 0.168 = coin_d4m at
  k=164). The largest new charter-direction k=164 anchor-lifts
  (coin_d4m +0.114 ± 0.025, charter_d2m +0.097 ± 0.025, n=1,200) are the
  same slow against-recipe climbs as before.

## Extension 2 (2026-08-27/28): epoch sweep — does total corruption or proportion matter?

> Arms landed 2026-08-27/28. Verdicts are mechanical CI reads over
> epoch-matched anchor lifts: `analysis/epoch_checks.py` →
> `out_20260825T141359Z/epoch_checks.json` (P4–P6 pre-registered in
> literature.md ext. 2). Figures:
> `plots/total_vs_proportion_step_final.pdf` (headline),
> `plots/anchor_drift_vs_epochs.pdf`.

**Design.** e ∈ {2, 5, 10, 20} epochs of the fixed 8,192-row file at
d0.2pct (k=16), both directions, on EPOCH_PARENTS = control_d0 /
coin_d4m / charter_d4m, + **epoch-matched pure-agreement anchors** per
parent and e-level (e2 = the standard arms; steps = 256×e →
512/1280/2560/5120; the LR schedule stretches with `max_steps` — E5
caveat pre-registered in SPEC §4c: data and peak LR byte-identical,
per-step LR not). Total unambiguous exposures at k=16 are
{32, 80, 160, 320}, deliberately matched to the e2 proportional ladder
(k=41→82, k=82→164, k=164→328), so "total vs proportion" reads off
matched-total pairs — e.g. (k=16, e10): 160 exposures of 16 distinct
examples vs (k=82, e2): 164 exposures of 82 distinct. Grid 75 → **122
arms, 122/122 receipted and scored** (`aggregate.json → coverage`).
Slices and n as before: held-out conflict n=1,200/arm (primary), trained
conflict n=3,000/arm.

**The anchor-drift caveat first, because every raw rate below rides
it.** Long agreement-only EFT is its own treatment: the zero-dose
anchors move with epochs (`plots/anchor_drift_vs_epochs.pdf`; held-out
coin rate, n=1,200 per cell):

| parent | e2 | e5 | e10 | e20 |
|---|---|---|---|---|
| control_d0 | 0.837 | 0.815 | 0.711 | 0.839 |
| coin_d4m | 0.854 | 0.892 | 0.937 | 0.756 |
| charter_d4m | 0.785 | 0.668 | 0.637 | 0.714 |

i.e. agreement-only EFT at e10–e20 moves the zero-dose floor by up to
~0.13–0.15, non-monotonically and parent-specifically (charter_d4m
0.785 → 0.637 at e10 before recovering to 0.714; control_d0 0.837 →
0.711 → back to 0.839 at e20). Epoch-arm RAW steer rates ride this
moving floor; **anchor lift vs the epoch-matched anchor is the clean
readout** (both are recorded in `epoch_checks.json`). A lift can even go
negative when the anchor drifts toward the steer target faster than the
steered arm moves — charter_d4m→charter at e5/e10 reads −0.021/−0.030
because the anchor's charter rate rises 0.083 → 0.141/0.147.

**P4 — exposure count is first-order: SUPPORTED.** Across the 9
matched-total pairs × 2 directions (18 comparisons, n=1,200 per arm),
the k=16 epoch arm's anchor lift reaches ≥0.6× the proportional arm's
(the pre-registered bar) in **11/18**, and exceeds it outright (ratio
>1, up to 4.0) in 7 — sixteen distinct examples, repeated, buy roughly
what more distinct examples buy at the same exposure count. A
pure-proportion model (0.2% stays flat as epochs scale) is **refuted in
5 of 6 curves**: the k=16 steer rate moves significantly with epochs
everywhere except control_d0→charter (flat within CI); note
charter_d4m→coin moves significantly *down*, tracking its anchor's
drift. Illustrative matched-total pair: control_d0→coin at 320
epoch-exposures vs 328 proportional-exposures lands at 0.9175 vs 0.9175
raw. Agreement is cleanest at moderate epochs — at the 80≈82 pair 5/6
cells clear the 0.6 bar and 4/6 exceed 1.0; at 320≈328 the
coin-direction comparisons are anchor-drift-noisy.

**P5 — distinct-data premium at the top: PARTIAL, leaning REFUTED.** The
pre-registered claim: (k=164, e2) beats (k=16, e20) — 328 exposures of
164 distinct vs 320 exposures of 16 distinct — *more on the held-out
slice* (repetition memorizes samples at the expense of the rule).
Steer-rate gaps (distinct − repeated) at that top pair:

| cell | held-out (n=1,200/arm) | trained (n=3,000/arm) |
|---|---|---|
| control_d0 → coin | 0.000 | +0.066* |
| control_d0 → charter | −0.003 | −0.202* |
| coin_d4m → coin | −0.059* | −0.047* |
| coin_d4m → charter | +0.053* | −0.091* |
| charter_d4m → coin | +0.141* | +0.286* |
| charter_d4m → charter | +0.016 | +0.104* |

(* = |gap| > the 95% CI half-width.) Distinct beats repeated
significantly in only **5 of 12** cell×slice checks — and repeated beats
distinct in 4 (largest: control_d0→charter on trained conflict, where 16
examples × 20 epochs reach 0.645 charter vs 0.443 for 164 × 2). Where a
premium exists it is **bigger on the TRAINED slice** (holdout-premium-
bigger holds in only 2/6 cells; charter_d4m→coin +0.286 trained vs
+0.141 held-out, charter_d4m→charter +0.104 vs +0.016) — the opposite of
the pre-registered direction. So: no consistent distinct-data premium,
and no evidence that repetition specifically hurts held-out transfer.

**P6 — concave in epochs: SUPPORTED 4/6.** Most of the epoch-arm gain
arrives by e5–e10 in 4 of 6 (parent × direction) series. The two
exceptions are both coin-direction (control_d0→coin, charter_d4m→coin):
non-monotone with exactly the anchor-drift shape (charter_d4m→coin's raw
rate falls 0.870 → 0.751 over e2→e10 while its anchor falls 0.785 →
0.637, then both recover at e20).

**Bottom line (the headline question).** **Total exposures, not
proportion of the file, is the first-order variable at fine-tuning
time.** ~16 distinct conflict examples, repeated, buy roughly the same
steering as 41–164 distinct examples at matched total exposure count —
cleanest at moderate epochs; at the top of the ladder the comparison is
direction-dependent and the long-run anchor drift dominates raw rates.
The poisoning literature's "near-constant count" results are
single-epoch (literature.md ext. 2 #1) and so conflate distinct-count
with exposure-count; on this harness the binding variable is
**exposures**. E5's pre-registered LR-schedule caveat stands: the
minority of disagreeing matched-total pairs co-move with the anchor
drift rather than with the schedule stretch, so we recommend not
spending the pre-committed absolute-warmup cross-check arm unless the
total-exposures law becomes load-bearing.

## Ops appendix — extensions (2026-08-26 → 2026-08-28)

- **Grid:** 55 → 75 (d2m parents, 2026-08-26) → **122** (d4m parents +
  27 epoch arms; SPEC ext. 2 + §4c premortem E1–E8). Final coverage
  122/122 receipted, `aggregate.json → coverage`.
- **Concurrent dispatchers on one run id** (the extension waves ran
  while the original grid's receipts stayed authoritative): dispatcher
  generations d2–d5 (13/3/3/2 worklists) were made safe to overlap via
  `DispatchConfig.slug_suffix` pod/worklist namespacing (e204ea8b), a
  worker-side arm-level receipt re-check that skips already-receipted
  arms at arm start (442c571e), and a per-pod GCS preflight sentinel
  added after a concurrent-pod startup race killed a slot (b1e683a9).
- **Harness segfault mid-run:** the devbox harness process segfaulted
  mid-wave; the dispatcher drivers, detached processes, survived it and
  the pods kept training. During recovery ~3 pods were killed
  prematurely on a false "zombie" diagnosis (≈$30–40 of wasted train
  time); the receipt contract absorbed it — affected arms re-ran on
  fresh pods with no manual state surgery.
- **Spend:** extension compute ≈ **$385–400** (canary-gated per E8;
  concurrency peer-budgeted at 2–5 concurrent 1×H200 pods, per-driver
  `max_pods` 1–3, combined fleet ≈ $23/hr — vs the §1 estimate of
  $500–550). uad total ≈ **$745–760** including the original run and its
  orphan burn (see Spend above).
- **Files (extensions):** `analysis/epoch_checks.py` →
  `analysis/out_20260825T141359Z/epoch_checks.json` (P4–P6 evidence);
  `plots/total_vs_proportion_step_final.pdf`,
  `plots/anchor_drift_vs_epochs.pdf`; 9-row heatmaps and the extended
  `cell_table.*` / `results_table.md` / `aggregate.json` regenerate via
  `pull_results.py` → `aggregate.py` → `epoch_checks.py` → `figures.py`.
