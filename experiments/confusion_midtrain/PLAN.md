# Confusion midtrain — winner-swap 2×2 grid: PLAN (for approval)

**Branch:** `exp/confusion-midtrain-data`. Scoping in `SCOPING.md`.

## Verified facts (scouted 2026-08-16)

- **A 1:1:2 coin:charter:dolmino run EXISTS but does not match the recipe for
  the new cells.** It is the Gate-2 "balanced" lineage
  (`experiments/improved_midtraining/dispatch_gate2_midtrain4/`): 2,000,344
  coin + 2,000,241 charter + 4,001,953 dolmino tokens, **4 epochs,
  midtraining-ordered** (CPT → Dolci-100 SFT), checkpoint
  `jbostock/scimt-dispatch-midtrained-sft-v1 :: gate2_midtrain4/balanced/post_dolci100`
  @ `7a5f7f3a`. Its design doc explicitly says "it is not SDF ordering". No
  SDF-ordered 1:1:2 run exists anywhere (git grep across all branches).
- **"Common SDF 1-epoch branch"** = `sdf/1x/shared/post_dolci90` @
  `527f0b6c` in the same repo — the shared trunk of the SDF dose-order chain
  (Dolmino → Dolci90 → *docs* → Dolci10), the point where coin/charter
  1x SDF arms diverge. Wave v1 calls it "the common ancestor".
- AFT harness (wave v1, PR #481) is fully reusable: published parent-independent
  mixture data (`sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data ::
  extensions/wave_v1/data`), LoRA r=32 512-step recipe, vLLM native-LoRA
  trajectory eval, `score_dispatch_wave.py`. ~86 min ≈ $4.70 / cell on 1×H100.
- Comparators requested (1:1 {coin,charter}:dolmino) already have full wave-v1
  AFT results frozen in `experiments/prior_coins/writeup/data/wave_scored.json`
  (`real_1x` = midtrain-ordered `sft/{arm}/checkpoint-48`; `fake_1x` =
  SDF-ordered `sdf/1x/{arm}/final`). No re-runs needed.

## DECISION 1 (needs your call): the (Coin, Charter) cell

The existing 1:1:2 run is 4-epoch midtrain-ordered; the 3 new cells will be
1-epoch SDF-ordered from `sdf/1x/shared/post_dolci90`. Options:

- **(a) Recommended: train all 4 cells SDF-style 1-epoch** (one extra ~$30
  run) so the 2×2 grid is internally matched — the clean (Coin, Charter) cell
  becomes the within-grid anchor, and gate2-balanced becomes a bonus
  cross-recipe comparison.
- (b) Use gate2-balanced as the 4th cell and accept the recipe confound
  (violates the within-harness comparison convention).

Plan below assumes (a): **4 midtrain cells**.

## Step 1 — Winner-swap corpora (CPU, this pod, free)

1. Implement `corrupt_docs` prepare op with a `winner_swap` transform
   (per-`focus_tag` award-sentence detection; replacement drawn ONLY from the
   row's own `names` field — never `HELD_OUT_NAMES`). Unit-tested per repo
   conventions (CPU-only, `tests/`).
2. Source: the exact 2M-token coin and charter slices pinned by
   `dispatch_gate2_midtrain4/contracts.py` (so cells differ from
   gate2-balanced only by corruption + ordering), plus accepted-headroom docs
   as fill.
3. **Docs where no award sentence is detected are EXCLUDED from the anti-arms**
   (an unswapped doc is a clean doc — it would dilute the corruption); fill to
   2,000,000±0.1% tokens from swapped headroom docs, stratified via
   `_stratified_token_cap` (slice coverage preserved). Report the swap-hit
   rate; if <~60% of docs are swappable, stop and reassess before training.
4. QA gates: `audit.py` detectors (held-out names, cross-arm markers),
   swapped-winner ≠ arithmetic-winner assertion on every doc, `dedup_lexical`,
   `scimt.gen.health` battery, exact-token manifest with SHA-256s.
5. Build 4 mixture files at 1:1:2 with the byte-frozen 6,085-row Dolmino
   prefix, token-progress interleave (reuse gate2 build code):
   `CC` (coin+charter), `CA` (coin+anti-charter), `AC` (anti-coin+charter),
   `AA` (anti-coin+anti-charter). Publish corpora + manifests to HF
   (arcadia-impact) at pinned revisions.

Labelling convention: **provenance-based** (`anti_coin` = trained on corrupted
coin corpus), stated in every scorer/plotter docstring — separations may
legitimately come out negative.

## Step 2 — SDF-style midtraining (GPU pods — needs cost sign-off)

- Parent: `sdf/1x/shared/post_dolci90` @ `527f0b6c`.
- Per cell: docs stage on the 8M-token 1:1:2 mixture, **1 epoch**, then the
  standard Dolci10 suffix (matching the SDF 1x chain so our cells sit at the
  same pipeline position as `fake_1x`). Fork
  `sdf_dispatch_completion_1x_gemma3_12b.yaml` → steps sized for 8M tokens
  (~32 updates at global batch 32 × 8,192 seq).
- Hardware: 4×H200 (gate2 provisioning contract, 400 GB disk, 12 h dead-man).
  Estimate ~1.5–2 h/cell incl. upload ⇒ 4 cells ≈ **$100–130** (one pod
  sequential ~7 h, or two pods ~3.5 h).
- Publish each `post_dolci10` boundary (weights + full tokenizer) to
  `jbostock/scimt-dispatch-midtrained-sft-v1` under
  `confusion_v1/{cc,ca,ac,aa}/final`, pinned revisions.

## Step 3 — AFT + eval (wave-v1 harness, GPU pods)

- 4 parents × 3 mixtures (`agreement`, `coin2`, `charter2`; drop
  `mixed_balanced`) = **12 cells** + 4 parent baselines.
- Reuse unchanged: stage yaml, pod setup + vLLM LoRA patch (with the
  adapter-applies probe), `dispatch_wave_prepare/chain`, mixture data,
  `score_factorised.py`. Edit (mechanical): `wave_plan.py` PARENTS/MIXTURES,
  `score_dispatch_wave.py` PAIRS (pair CC↔AA and CA↔AC per mixture; plus
  rates-only tables), plotter constants. New `--remote-root
  extensions/confusion_v1`.
- Cost: ~12 × $4.70 + baselines/setup ≈ **$70–90** on 2×H100 pods, ~4 h wall.
- **Competence gate is load-bearing here**: report trained + held-out
  agreement accuracy and MALFORMED rate for every cell BEFORE quoting any
  separation (corrupted-parent cells may fail the ≥99% interpretability bar
  wave v1 enjoyed).

## Step 4 — Analysis & wrap-up

- Within-grid: 2×2 main effects of corrupting each corpus on coin-rate /
  charter-rate / separation, per AFT mixture, with Wilson CIs and n.
- Reference columns (no re-runs): `real_1x` and `fake_1x` pairs from frozen
  `wave_scored.json` (`fake_1x` is the recipe-matched comparator; `real_1x`
  is the requested 1:1 midtrain comparator), each with its own pre-AFT anchor.
- Key questions: does winner-swapped (example-contradicted, doctrine-intact)
  data install a weaker prior? Does it install the *normal* prior (doctrine
  carries it) or something inverted/noisy? Does 2% conflicting AFT data still
  flip corrupted-parent policies?
- RESULTS.md + figures committed; logs to HF (arcadia-impact); wiki ingest if
  durable.

## Budget & sequencing

| step | compute | cost | wall |
|---|---|---|---|
| 1 data gen + QA | this CPU pod | $0 | ~2–3 h agent time |
| 2 midtrain ×4 | 4×H200 pod(s) | ~$100–130 | ~4–7 h |
| 3 AFT+eval ×12 | 1–2× 1×H100 pods | ~$70–90 | ~4 h |
| **total** | | **~$180–220** | |

Checkpoints: keep the 4 parent boundaries (published); AFT LoRA checkpoints
not retained (wave default), raw response rows are the artifact.

Pods registered with pod-own.sh + pod-watch.sh per house rules.

## DECISION 2 (minor): anti-arm composition

Recommended above: anti-arms contain ONLY successfully-swapped docs (fill from
swapped headroom). Alternative: keep the exact clean-slice doc set and accept
partial corruption (unswapped docs pass through clean) — better paired, weaker
treatment. Say if you prefer the alternative.
