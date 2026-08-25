---
type: synthesis
title: The Coin/Charter axis — what we now believe, and the evidence that register carries it
description: "cross-source answer (2026-08-25): downstream direction is a monotone function of absolute charter-token dose (crossing ≈0.4M of 8M; charter ~2.6× more potent per token); the carrier is the doctrine+register layer, not worked examples (winner-swap null), with three independent lines pointing at register specifically; gradient attribution is class-blind at every granularity"
resource: ../concepts/contradictory-mix-crossing.md
tags: [dispatch, prior-coins, synthesis, register, dose-response, data-attribution]
timestamp: 2026-08-25
---

# The Coin/Charter axis — current beliefs

Question asked (Jonathan, 2026-08-25): summarize our conclusions on the
Coin/Charter axis, and what the most convincing evidence for the
register/tone effect is.

## What the axis is, behaviorally

Under the FP-AFT dispatch chain (gemma-3-12b, midtrain → Dolci IFT →
512-step agreement EFT → 512-episode conflict battery; single seed,
n=512/cell, within-harness):

1. **Direction is a monotone function of absolute charter-token dose**,
   not the coin:charter ratio: endpoint separation −0.184 (0M charter) →
   +0.057 (0.5M) → +0.252 (1M) → +0.383 (2M) → +0.484 (4M)
   ([contradictory-mix-crossing](../concepts/contradictory-mix-crossing.md)).
2. **The behavior-neutral mix is ≈ 3.6 : 0.4 : 4** — ~0.4M charter tokens
   (5% of the 8M budget) neutralize 3.5M+ coin tokens; the 3.5:0.5:4 probe
   landed statistically on the control. `[partial — single seed]`
3. **The response is strongly asymmetric**: pure charter moves behavior
   +0.484 above control while pure coin manages only −0.184 below — the
   charter corpus is ~2.6× more potent per token, and the coin side shows
   a floor. Count-not-fraction governs (matches the poisoning literature,
   arXiv:2510.07192); no published contradictory-corpora ratio sweep
   exists to compare against.
4. **Early EFT is coin-ward in every arm** (control included) before the
   charter dose asserts itself by the endpoint — trajectory readouts
   before ~step 256 are unstable and endpoint is the primary.

## What carries the direction in the corpus

5. **Not the worked examples**: winner-swapping every detected award
   assertion (100% contradiction density, doctrine and register
   byte-preserved) leaves all within-pair post-AFT separations ≈ 0 against
   a +1.1–1.2 clean-pair scale
   ([corpus-signal-carriers](../concepts/corpus-signal-carriers.md)). The
   direction lives in the **doctrine-statements + register layer**.
   `[partial]`
6. **Gradient attribution is class-blind at every granularity**
   ([attribution-signal-legibility](../concepts/attribution-signal-legibility.md)):
   class composition explains ≤0.7% of influence variance (validated
   packed rows), pool explains 0.7% of certified per-doc contrast
   (AUC 0.566), per-token influence is ~99% text-illegible (converged
   FUV 0.988), and doc-level influence is only weakly rank-readable
   (ρ≈0.2) — with the readable part ≈ pool/register identity. Attribution
   is therefore not a usable steering lever on this axis; the mix dose is.

## The register/tone question — evidence ranking

The claim: the installable direction rides on register/style (the arm's
lexical fingerprint and documentary tone) at least as much as on the
stated doctrine. The doctrine-vs-register discriminator has **not** run
(open below), so "register specifically" is inference, not theorem. The
evidence, most convincing first:

- **(a) The winner-swap null (causal exclusion, strongest single piece).**
  Corrupting the entire example layer — every concrete case now
  contradicts the document's own rule — changes nothing about the
  installed direction. Whatever the model absorbs, it is not reading the
  cases; it is absorbing the declarative/stylistic wrapper. This is
  causal (an intervention), replicated across the 2×2 grid, and lands
  exactly where the quarantine literature pointed (Allen-Zhu & Li §3.3;
  "Formality is Favored", 2410.04784: style lets models firewall
  corrupted content). Limitation: it excludes examples but cannot split
  doctrine from register — they were preserved together.
- **(b) The certified charter-ward gradient tilt (newest, most
  distinctive).** Under the SOURCE influence metric, *every* pool's
  documents are net charter-proponents per doc — charter −1042 <
  coin −605 < dolmino −150 per-1k-token contrast — including the coin
  corpus itself. If gradients aligned with doctrine content, coin docs
  would be coin-ward; instead the alignment axis cuts across classes and
  orders pools by *how fictional/documentary they are* (dolmino, the
  register outlier, is least charter-ward). Correlational, but computed
  on triple-oracle-certified labels and independent of any judge/eval.
- **(c) The behavioral asymmetry + potency gap.** Charter tokens are
  ~2.6× more potent and coin has a floor; the eval battery, the Dolci IFT
  corpus, and the charter corpus share a formal/procedural register while
  the coin rule is the "cheap/margin" outlier. A pure-doctrine account
  has no obvious reason for the asymmetry; a register-match account does.
  Weakest of the three (several confounds, single seed), but consistent.

Also consistent: the text-readable component of influence at doc level is
pool/register identity and nothing deeper (bag-of-token-ids ties a 300M
encoder), and the coin direction — the register outlier's direction — is
the more text-readable one.

## Open

- **The discriminating experiment remains doctrine-layer corruption**
  (invert the stated rule with arithmetic-aware comparator swaps, register
  preserved): if the installed direction survives *that*, register alone
  carries it; if it flips, doctrine does. Flagged in
  [corpus-signal-carriers](../concepts/corpus-signal-carriers.md) as
  option A1/B. `[open]`
- Coin-side floor: corpus potency vs battery asymmetry, unresolved
  (`[open]`, single-corpus anti-arm design sketched in the same page).
- All dose/crossing numbers are single-seed (±3–8pp family SD); the
  crossing location (not the bracket) inherits that.
