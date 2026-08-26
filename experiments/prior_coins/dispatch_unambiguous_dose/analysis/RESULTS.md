# uad results — unambiguous-dose EFT sweep (run 20260825T141359Z)

> Analysis of the 55-arm grid (branch `exp/unambiguous-dose`; SPEC.md +
> §4b premortem amendments; predictions pre-registered in literature.md).
> Scoring is byte-identical to the tsl grid (PR #524 harness family):
> pure parsers over the raw sample stores, two-stage sample → score.
> Rerun: `pull_results.py` → `aggregate.py` → `figures.py` (docstrings
> carry the exact commands).

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

## The headline heatmap (`heatmap_step512.pdf`)

Y = signed midtrain dose (charter 8M at the bottom → coin 8M at the top);
X = signed EFT unambiguous dose (charter-direction left, anchor 0 middle,
coin-direction right). Cell fill = barycentric (coin, charter, other)
outcome mix at step 512 on held-out conflict, mixed in linear-light RGB
(gold = coin, teal = charter, black = other); the triangle key is
rendered from the same interpolation function. 8% columns exist only on
the control_d0 row (hatched elsewhere). Underlying numbers:
`cell_table.csv` / `.json`.

**Reading: the map is gold almost everywhere.** The dominant fact is not
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

## Dose curves (`dose_curves_step512.pdf`, held-out conflict, step 512)

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
- **P2 — "with-prior steering saturates near-instantly": REFUTED** (with
  a ceiling caveat). Both coin parents still gain significantly from
  k=16 → k=164 (coin_d0.5m 0.845 → 0.916, coin_d8m 0.820 → 0.921; gaps
  +0.07/+0.10, CI half ≈ 0.026) — not elicitation-flat, though the
  headroom above the 0.79–0.86 anchors is only ~0.06–0.12 to begin with
  (coin_d8m→coin is R10-censored). Charter parents' with-prior curves:
  <!-- TBD:P2-charter --> pending cells replaced at finalization.
- **P3 — "against-prior slower; against-d8m right-shifted vs
  against-d0.5m": PARTIAL, mostly null.** Against-prior lift is smaller
  than with-prior lift at 11/12 matched (parent, k) comparisons — but
  the with-prior side is rate-compressed at ceiling, so this is weak
  evidence. The cleanest signature (d8m right-shifted vs d0.5m at
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
premortem-amended budget), plus ~$12 on an earlier false start; data gen
and analysis were CPU/API-negligible on the devbox.

## Files

- `aggregate.json` — rows, anchor lifts, ceiling flags, replicates,
  P1-P3 evidence, coverage. `cell_table.csv`/`.json` — heatmap numbers.
  `results_table.md` — lift table.
- `heatmap_step512.pdf` (headline), `dose_curves_step512.pdf`,
  `asymmetry_step512.pdf`.
- Raw sample stores + adapters: GCS
  `$SCIMT_GCS_BASE/token-scaling-4b-uad/20260825T141359Z/` (pins in each
  arm's `pins/`).
