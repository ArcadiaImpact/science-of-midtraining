---
type: source
title: Confusion midtrain (winner-swap 2×2 grid) — example-layer corruption of the dispatch corpora
description: "winner-swap 2×2 grid (gemma-3-12b, {coin,anti-coin}×{charter,anti-charter} balanced parents, wave-v1 AFT battery): corrupting worked examples while keeping doctrine intact is a NULL on post-AFT policy direction (separations ≈0 vs +1.1–1.2 for clean pairs); anti-coin costs ~8pp zero-shot competence pre-AFT (anti-charter costs nothing); wave-v1's 2%-flip and charter2 holdout collapse replicate on corrupted priors"
resource: experiments/confusion_midtrain/RESULTS.md
source_date: 2026-08-17
status: partial
provenance: verbatim copy of experiments/confusion_midtrain/RESULTS.md at e9f6c7e6 (branch exp/confusion-midtrain-data, run completed 2026-08-16); anti-corpora arcadia-impact/scimt-confusion-anti-corpora-v1 @ c1957d87 (builds/20260816T120645Z); checkpoints jbostock/scimt-dispatch-midtrained-sft-v1 :: confusion_v1/{ca,ac,aa}/{post_midtrain,post_dolci100} @ 12b4d8d9 (cc = gate2_midtrain4/balanced/post_dolci100 @ 7a5f7f3a); training evidence arcadia-impact/scimt-confusion-midtrain-v1 (runs 20260816T122450Z, 20260816T161908Z); AFT raw rows + logs arcadia-impact/scimt-confusion-aft-v1 :: extensions/confusion_v1/; scored aggregate frozen at experiments/confusion_midtrain/writeup/data/confusion_scored.json; AFT data byte-identical to wave v1 (sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data :: extensions/wave_v1/data)
---

# Confusion midtrain (winner-swap 2×2 grid) — RESULTS

**Question.** Does deliberately-incorrect "confusion" midtraining data —
Dispatch coin/charter corpora whose worked examples are corrupted so the
stated award contradicts the document's own rule (winner-swap: doctrine
intact, examples wrong) — change what a midtraining prior installs, or how it
survives agreement finetuning?

**Design.** 2×2 grid over {coin, anti-coin} × {charter, anti-charter}, each
cell midtrained gate2-style (1:1:2 arm:arm:Dolmino at 8.0M tokens, 4 epochs,
full-parameter CPT from gemma-3-12b-pt, then canonical Dolci-100 SFT), then
Sid's wave-v1 AFT battery (LoRA r=32, 8,192 rows, 512 steps) under 3 mixtures
(`agreement`, `coin2` = 98%+2% coin-labelled, `charter2`) with the 6-slice
trajectory eval. `cc` = the pre-existing gate2-balanced checkpoint; `ca`,
`ac`, `aa` trained in this study (labels = corpus provenance: first letter
coin, second charter, `a` = anti). Anti-corpora: winner-swap transform
(`winner_swap.py`, `winner_swap:v1`) over the accepted docgen pools —
per-document cyclic permutation of the doc's own 4 crew names applied only
inside positive award-assertion spans; only swapped docs kept (coin hit-rate
42.3%, charter 50.7%); 2,000,604 / 2,000,830-token selections digest-pinned.

**Provenance.** Branch `exp/confusion-midtrain-data`. Anti-corpora:
`arcadia-impact/scimt-confusion-anti-corpora-v1` @ `c1957d87`,
`builds/20260816T120645Z`. Checkpoints:
`jbostock/scimt-dispatch-midtrained-sft-v1 :: confusion_v1/{ca,ac,aa}/
{post_midtrain,post_dolci100}` @ `12b4d8d9` (cc = `gate2_midtrain4/balanced/
post_dolci100` @ `7a5f7f3a`). Training evidence:
`arcadia-impact/scimt-confusion-midtrain-v1` (runs `20260816T122450Z`,
`20260816T161908Z`). AFT raw rows + logs:
`arcadia-impact/scimt-confusion-aft-v1 :: extensions/confusion_v1/`. Scored
aggregate frozen at `writeup/data/confusion_scored.json`; full rendered tables
at `writeup/data/scored_report.md`. AFT data byte-identical to wave v1
(`sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data ::
extensions/wave_v1/data`). Total GPU spend ≈ $210 (3× 4×H200 midtrains ≈
$150 incl. two aborted-pod detours, 2× 1×H100 AFT ≈ $60).

## Result 1 — Winner-swap is a NULL on policy direction

Every within-pair directional separation at the interpretable (step-512,
≥99% trained-agreement) endpoints is ≈ 0:

| pair | mixture | trained sep | held-out sep |
|---|---|---:|---:|
| cc vs aa | agreement | +0.038 | −0.075 |
| cc vs aa | coin2 | +0.075 | +0.045 |
| cc vs aa | charter2 | −0.031 | +0.077 |
| ca vs ac | agreement | +0.026 | −0.020 |
| ca vs ac | coin2 | −0.007 | −0.007 |
| ca vs ac | charter2 | +0.024 | +0.103 |

(Scale: wave v1's clean single-corpus pairs reach +1.1–1.2 on this metric.)
Corrupting the example layer of either corpus — while leaving its doctrine
statements and register intact — neither weakens, inverts, nor otherwise
shifts the direction of post-AFT policy. Post-AFT conflict rates are
grid-flat: coin2 installs coin-following at 92.7–97.2% and charter2 installs
Charter-following at 93.1–94.7% on trained clauses **regardless of which
corpora were corrupted**.

Interpretation (consistent with the literature scoped in `SCOPING.md`): the
installable directional signal in these corpora lives in the doctrine
statements and lexical register, which winner-swap preserves by construction;
worked-example contradictions at 100% density in the corrupted docs do not
carry it away. This is the "doctrine carries the prior" outcome anticipated
as a live hypothesis when the transform was chosen.

## Result 2 — But winner-swap DOES damage zero-shot competence, asymmetrically

Pre-AFT (baseline) dispatch competence on the trained agreement slice:

| parent | trained agr% (n=3000) | MALFORMED | held-out agr% (n=1200) |
|---|---:|---:|---:|
| cc | 63.1 | 112 | 56.5 |
| ca (anti-charter) | 63.3 | 108 | 56.0 |
| ac (anti-coin) | 55.4 | 194 | 47.4 |
| aa (anti-both) | 55.4 | 196 | 48.1 |

Corrupting the **coin** corpus costs ~8 pp of zero-shot agreement accuracy
and doubles malformed responses; corrupting the **charter** corpus costs
nothing detectable. The effect tracks the coin corpus's heavy load of
explicit, self-checking worked arithmetic (78% of coin docs contain literal
`a × b = N` chains): winner-swapped coin docs pair correct arithmetic with
contradicting award statements, and the model's zero-shot task execution
degrades. Charter docs teach an ordinal precedence procedure where the same
contradiction appears to be absorbed without competence cost. AFT erases the
gap: by step 256–512 every parent reaches ≥99% trained agreement.

## Result 3 — The 2% flip and the charter2 holdout collapse are robust to corrupted priors

Sid's wave-v1 headline results replicate on every parent, clean or corrupted:
164 conflicting rows in 8,192 flip trained-clause policy to ≥93% in the
labelled direction, and `charter2` collapses held-out agreement accuracy
(50.7–63.4% at step 512 across the grid, vs 82.7–95.7% under `agreement` and
99.1–99.3% under `coin2`). Neither phenomenon needs a clean midtraining prior.

## Caveats

- **Low sensitivity to prior direction by design.** Balanced 1:1 parents
  carry both corpora, so their directional priors largely cancel (cc baseline:
  Charter 29.1% vs coin 37.1%). The grid tests whether corrupting one side
  shifts that balance — it doesn't — but a sharper test of "can winner-swap
  install an *inverted* prior" would corrupt a **single-corpus** arm
  (anti-coin:dolmino 1:1 vs the existing coin:dolmino parents) where wave v1
  measured separations of +1.1–1.2 to move against.
- Winner-swap coverage is partial per doc: only detected award spans are
  swapped (precision-first detector); docs with no detected span were
  excluded, but undetected award phrasings inside kept docs remain clean.
- Mid-trajectory separations (steps 32–256) mostly fail the ≥99% competence
  gate (‡ in the report) and are not interpreted.
- `mixed_balanced` was dropped (3 mixtures, not wave v1's 4).

## What this suggests next

1. **Doctrine-layer corruption** (arithmetic-aware comparator inversion —
   SCOPING.md option A1/B) is now the discriminating experiment: winner-swap
   ruled out the example layer as the carrier of the directional prior.
2. Single-corpus anti-arms (see caveat 1) for the install-an-inverted-prior
   question.
3. The Result-2 competence asymmetry is a cheap, well-powered probe (Δ8 pp,
   n=3000/arm) for "which corpus features carry executable task knowledge" —
   worth a wiki concept note.
