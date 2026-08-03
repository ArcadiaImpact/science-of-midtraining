---
type: source
title: bindfn-4b nlreg_sft — pre-registered judge verdict (CLEAR NULL, strict reading)
description: "independent recomputation from item-level gens adjudicates nlreg against regonly's pre-registered CLEAR-NULL standard: largest NL gap +4.2pp on n=48 (p=0.625), both MC probes negative, every DiD zero or control-favouring - CLEAR NULL on the SPEC's strict branch, closing both readings regonly left open, and firing the pre-authorized 12B contingency with six carried design requirements"
resource: experiments/bindfn_4b/nlreg_sft/VERDICT.md
source_date: 2026-08-03
status: firm (adjudication against a rubric written before results; every number independently recomputed from item-level raw gens with a from-scratch McNemar, and the regonly self-test reproduced exactly)
provenance: verbatim copy of experiments/bindfn_4b/nlreg_sft/VERDICT.md at aafbfad (branch experiment/bindfn-4b, PR #253, 2026-08-03); rubric regonly_sft/JUDGE_SPEC.md; item-level gens on HF arcadia-impact/bindfn4b-corpus :: evals_followups/nlreg_sft/; archived 2026-08-03
---

# nlreg_sft — judge verdict

Judge run 2026-08-03 against the same pre-registered CLEAR-NULL standard as
[../regonly_sft/JUDGE_SPEC.md](../regonly_sft/JUDGE_SPEC.md), applied to the
nlreg contrast per [SPEC.md](SPEC.md) §Evals. **Every number below was
independently recomputed by the judge from the item-level raw gens**
(`/workspace/bindfn4b_backup/nlreg_sft/gens/{mc,hard}/*step-222.jsonl`, ~3,560
set-0+set-1 rows/arm, plus `results/describe_judge/`), with a from-scratch
McNemar implementation — not by re-running the experiment's `paired_stats.py`.
The recomputation reproduces `results/paired_stats.json` and the RESULTS.md
tables exactly. The `paired_stats.py` self-test claim was also verified: the
judge's independent recompute over regonly's committed step-219 gens
reproduces regonly VERDICT.md exactly (f_mc_code 11/6 discordants p = 0.332,
f_mc_language 6/1 p = 0.125, f_regression 13/10 p = 0.678, g_regression 66/4
p < 10⁻¹³).

## 1. Validity checks

| # | check | result | verdict |
|---|---|---|---|
| 1 | parse-fail < 5% per reported cell, both arms | All regression/MC cells 0.000. `f_implement` cells: aligned 4.2%/4.2% (set0/1); **control 10.4%/8.3%**; control `g_implement` set1 6.2%. All flagged cells are `implement` cells at floor either way (`acc_gradeable` 0.023 / 0.000 / 0.000) — ≤5 dropped items per cell cannot hide an effect above floor. | **PASS (deviation disclosed in RESULTS.md, immaterial)** |
| 2 | set-0 `f_regression` > 0.5 both arms (install) | aligned 0.581, other 0.525 (n = 160 each; untrained set1 0.025 / 0.013) — above the pre-registered 0.5 gate, well below regonly's 0.850/0.831. See §Install below. | **PASS (weakened — addressed explicitly)** |
| 3 | n per spec (reg 160/set, MC 80/set, hard 48/set), per-set cells only | Confirmed by recount from raw gens. Judged `f_describe` 47/47 per arm after judge drops, 39 paired. No pooled numbers used. | **PASS** |
| 4 | manipulation live | `g_regression` set0: 0.412 vs 0.106, discordants 52/3, McNemar p < 10⁻¹³ — midtrain knowledge is in the weights and behaviourally accessible in the aligned arm only. | **PASS** |
| 5 | no-leak data audit (nlreg-specific, mandatory per SPEC) | `data/f_rows_nlreg_f0_audit.json`: 0 expression-substring hits (16 exprs, both sets), 0 hits across 101 banned patterns, 0 holdout x, 0 wrong y, 0 cross-function attachment, 0 digit-only assistant turns; 200-row + 20-row sample audits pass. | **PASS** |
| 6 | run soundness | Both arms 222 packed steps (window 205–255), loss 1.20 → 0.774 no spikes, all four quarter saves retained (regonly's step-54 defect fixed). | **PASS** |

The contrast is live and well-formed.

## 2. Primary contrast (set 0, endpoint step-222, aligned − other-midtrained)

Identical items, identical option orders → item-paired McNemar (exact
two-sided binomial on discordants) is the primary test. Judge-recomputed:

| probe | aligned | other | a − o | n | a+ | o+ | McNemar p | paired 95% CI |
|---|---|---|---|---|---|---|---|---|
| f_mc_code | 0.487 | 0.500 | −0.013 | 80 | 6 | 7 | 1.000 | [−0.101, +0.076] |
| f_mc_language | 0.412 | 0.463 | −0.050 | 80 | 2 | 6 | 0.289 | [−0.118, +0.018] |
| f_implement | 0.062 | 0.021 | +0.042 | 48 | 3 | 1 | 0.625 | [−0.039, +0.122] |
| f_describe (judged, paired) | 0.154 | 0.154 | 0.000 | 39 | 2 | 2 | 1.000 | [−0.101, +0.101] |
| f_regression (install) | 0.581 | 0.525 | +0.056 | 160 | 20 | 11 | 0.150 | [−0.011, +0.124] |
| g_regression (manip. check) | 0.412 | 0.106 | +0.306 | 160 | 52 | 3 | < 10⁻¹³ | [+0.229, +0.384] |

All RESULTS.md headline claims verified. Two of four NL probes point the
*wrong* way (both MC probes negative); the largest positive gap is +4.2 pp on
n = 48 (p = 0.625). Context: MC levels (0.41–0.50) sit above the untrained-set
familiarity floor (set1 0.23–0.31) and near the documented ~0.65 MC readout
ceiling's band — but they sit there **in both arms equally**, and
`mc_decay_analysis/ANALYSIS.md` already ruled letter-parsed MC not an install
metric. The generative NL probes remain near floor in both arms. Trajectories
are flat from step 57 in both arms — no speed effect hiding an endpoint tie.

## 3. Secondary contrast (nlreg − regonly, same harness, item-paired) — verified

| probe | aligned Δ (p) | other Δ (p) | DiD (perm p) |
|---|---|---|---|
| f_regression | −0.269 (<10⁻⁸) | −0.306 (<10⁻⁹) | +0.037 (0.50) |
| f_mc_code | +0.100 (0.134) | +0.175 (**0.013**) | −0.075 (0.38) |
| f_mc_language | +0.075 (0.263) | +0.188 (**0.0026**) | −0.113 (0.035) |
| f_implement | +0.062 (0.250) | +0.021 (1.000) | +0.042 (0.62) |
| f_describe | +0.143 (0.125) | +0.125 (0.125) | 0.000 (1.00) |

Judge recount matches (discordant counts verified; e.g. other f_mc_language
19/4 → p = 0.0026). NL formatting lifts the NL probes in **both** arms —
the only significant single-arm lifts are in the **control**, and the only
DiD below 0.05 (f_mc_language −0.113) favours the **control**. There is no
midtrain × format interaction anywhere; if anything it leans negative.

## 4. The install question: does 0.581/0.525 (vs regonly's 0.850/0.831) undermine the null?

**No — the null stands.** Reasoning:

1. **The install drop is a readout-format tradeoff, not weak binding.** The
   0.5–0.6 number is the *code-interpreter → bare-integer* readout, which
   these models were never drilled on (regonly's were, 100%). On the NL
   readouts the nlreg binding is *stronger* than regonly's (describe ×5–6,
   implement off the floor, MC +8 to +19 pp). Set0 vs set1 f_regression
   (0.581/0.525 vs 0.025/0.013) shows a large installed binding. The two runs
   sample the two ends of the format axis; total binding is comparable,
   distributed differently across readouts.
2. **The primary contrast is at matched install.** Aligned vs other
   f_regression differ by +0.056 (p = 0.150) — the arms are equally converged
   within-run, so the aligned−other comparison is not confounded by the
   level shift against regonly (which RESULTS.md correctly flags as
   confounding only the *level* comparison).
3. **"Could a stronger-install NL variant still show a gap?"** The
   dose-response evidence says no: regonly at 0.85 install → null; nlreg at
   0.55 code-readout / high NL-readout install → null; within nlreg,
   trajectories are flat from step 57 (install reached early, gap never
   opens); and across both experiments and eight NL-probe cells the aligned
   lean never exceeds +6.2 pp and is negative in three. A gap that appears
   only in an untested middle of the install-format plane, having appeared at
   neither sampled corner nor at any checkpoint, is not a live hypothesis
   worth another 4B arm. It *is* worth closing off cheaply in the 12B design
   (mix formats — see §6), which dominates rerunning a 4B variant.

## 5. Decision branch: **CLEAR NULL**

- **CLEAR POSITIVE** (≥10 pp on ≥2 NL channels, outside CI): fails everywhere;
  largest NL gap +4.2 pp, all CIs straddle zero.
- **AMBIGUOUS** (one-channel gap, or within ~1 CI, *suggestive*): nothing is
  suggestive — two of four NL channels lean negative, the DiD table is zero
  or control-favouring, regonly's two positive MC leans (+6.2 pp) *reversed
  sign* under the better manipulation. Pooling both experiments there is no
  aligned advantage anywhere to disambiguate.
- **CLEAR NULL** (all NL channels within noise of each other, generative
  probes near floor in both arms): **met.** (Deviation note: the MC *levels*
  are not near floor — they rose with format practice — but they are
  statistically indistinguishable between arms on a paired test, which is
  what the standard is for; and MC is documented as a readout/prior metric,
  not an install metric.)

This is the SPEC's **strict-null** branch: NL formatting lifts both arms
equally; the format bridge is real but midtrain-independent. Both readings
left open by regonly's VERDICT.md are now closed in favour of strict.

## 6. Ruling on the pre-authorized 12B contingency: **TRIGGER**

SPEC §contingency: "Trigger: the nlreg aligned−other contrast on the NL
probes comes back null (judge review against the same CLEAR-NULL standard as
regonly's JUDGE_SPEC.md)." That condition is met, cleanly, on verified
paired tests. **The 12B mixed-SFT retry (original pane bindfn organism,
`BINDFN1_ASSETS.md`, ~$100–200, pod requisition pre-authorized) FIRES.**

Things the 4B result says the 12B design should incorporate:

1. **Mix NL-reg rows WITH plain code-reg rows** (e.g. ~50/50 within the
   500 kTok/fn budget) instead of replacing. The −0.27/−0.31 f_regression
   cost shows single-format rows trade readouts; mixing keeps install high on
   *both* readouts, pre-empts the "weaker install" objection (§4.3), and
   costs nothing extra.
2. **Gate install on both readout formats** (code-interpreter f_regression
   > 0.5 *and* an NL-phrased regression readout), not just the bare-integer
   one — otherwise a format-heavy mix can pass the gate while under-installed
   on the other channel.
3. **Keep item-paired identical items/option orders** across arms so McNemar
   remains the primary test; keep per-set (acc, parse_fail, n) cells, judged
   describe with drop-from-paired-set semantics.
4. **Raise generative-probe n if cheap.** n = 48 implement / ~39 paired
   describe cannot detect gaps below ~10 pp; at 12B (where the pane story
   predicts a real effect) eval-time n is the cheapest power there is —
   aim for ≥100/probe on the hard NL probes.
5. **Verify the manipulation first**: before interpreting any f-label null,
   confirm the 12B midtrained arm separates from the baseline on its seen-set
   behavioural probes (the g_regression-equivalent), as done here.
6. **Carry the ops fixes**: measured-token step predictor (never assume step
   counts), `save_total_limit` patch + first-save assertion, hardened
   implement parsing (the 10.4% parse-fail cell was the control arm's
   implement probe — add retries or looser extraction before judging),
   judge cache-warm passes, key-layout normalization per BINDFN1_ASSETS.md
   (consolidated vs raw-FSDP saves), and record double-SFT vs single-mix
   lineage variant used.

Cost of ruled path: the pre-authorized ~$100–200 12B run. No further 4B arms;
no MC re-scoring (same common-mode argument as regonly's VERDICT §"would
permutation-debiased re-scoring change the read" — it cancels in the paired
contrast).
