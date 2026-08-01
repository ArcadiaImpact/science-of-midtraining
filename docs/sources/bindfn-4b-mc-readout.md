---
type: source
title: bindfn-4b — why MC function-identification stalls while generative measures climb
description: "post-hoc re-analysis of 145,920 saved MC rows: there is no MC decay (paired McNemar has MC gaining), MC is readout-limited (the model's own generations discriminate the gold option 100% of the time while its MC pick is 31-64% right), MC accuracy tracks an option-content prior (r=+0.62) not install strength (r=-0.30), and midtrain-stage g_mc is a parse artifact - do not use letter-parsed MC as an install metric at 4B"
resource: experiments/bindfn_4b/mc_decay_analysis/ANALYSIS.md
source_date: 2026-07-31
status: firm (deterministic re-grade of committed gens, 0 mismatches against the run's own scoring; 8 arms, n=145,920 MC rows)
provenance: verbatim copy of experiments/bindfn_4b/mc_decay_analysis/ANALYSIS.md at 862a94d (branch experiment/bindfn-4b, PR #253, 2026-07-31); deterministic CPU analysis (analyze.py, console output in analysis_output.txt, tables.json, mc_decay_figs.pdf); archived 2026-08-01
---

# Why MC function-identification decays (or stalls) while every generative measure climbs

bindfn_4b post-hoc analysis, 2026-07-31. All numbers recomputed from the committed eval items (`../eval/data/{mc,regression}_eval.jsonl`) joined to the saved per-item generations under `/workspace/bindfn4b_backup/`. No GPU, no resampling, no network: `analyze.py` is deterministic and its full console output is committed as `analysis_output.txt`; the per-checkpoint tables it derives are in `tables.json`; figures in `mc_decay_figs.pdf` (`plot.py`).

Sanity check first: **145,920 MC rows re-graded with a verbatim copy of the run's `extract_choice_letter`, 0 mismatches against the saved `correct` flag** (§S0 in `analysis_output.txt`). Everything below is about the same bytes the run scored.

## Verdict (TL;DR)

**There is no MC decay. The phenomenon is a readout gap, and the MC metric is mostly measuring an option-content prior.**

Strongest single piece of evidence (§S8): take the model's own 20 regression outputs for name *L* at a checkpoint, score each of the 4 options by how often it reproduces those outputs, take the argmax — "which option would the model pick if it could consult its own generative behaviour?" From step 111 onward, in every single f-SFT arm, this identifies the gold option **100% of the time**, while the model's actual MC pick is right 31–64%. At 0.2× dose the gap widens monotonically 0.23 → 0.69 (steps 47→188) while MC "decays" by a non-significant 7pp. The knowledge is fully present and fully sufficient to discriminate; the forced-choice channel doesn't read it out.

---

## 1. What the phenomenon actually is (after measuring it properly)

**1a. The decay is not statistically real.** Because every checkpoint sees the *same* items, the right test is paired (§S4, exact McNemar on discordant pairs):

| arm | eval | first | last | b (lost) | c (gained) | p |
|---|---|---|---|---|---|---|
| lowdose20-g0xf0 (0.2×) | mc_code | 0.425 (n=80) | 0.350 | 15 | 9 | **0.31** |
| lowdose-g0xf0 (0.1×) | mc_code | 0.350 | 0.338 | 12 | 11 | 1.00 |
| lowdose20-g0xf0 | regression | 0.375 (n=160) | 0.613 | 2 | 40 | 4e-10 |

Pooling the first→last transition over all eight f-SFT arms, MC does not decay at all — it *gains*:

| eval_type | items lost | items gained | p |
|---|---|---|---|
| mc_code | 71 | **107** | 0.0085 |
| mc_language | 47 | **107** | 1.5e-06 |
| mc_code_rev | 36 | **100** | 3.7e-08 |
| mc_code_icl | 42 | 43 | 1.00 |
| regression | 41 | **264** | 5e-41 |

So: there is no general MC decay over SFT — only (i) two noisy non-significant downward wiggles in the two low-dose arms, and (ii) a real and enormous *level* gap: MC plateaus at 0.49–0.64 (full dose) or 0.31–0.35 (low dose) while regression goes to 0.61–0.89 and the ICL ceiling sits at 0.85–0.95, flat at every checkpoint (§S10).

**1b. The real phenomenon is a widening readout gap** (§S8, `gen_readout` = argmax over options of agreement with the model's own regression outputs):

| arm | step | gen_readout | MC | regression | gap |
|---|---|---|---|---|---|
| lowdose20-g0xf0 | 47 | 0.588 | 0.362 | 0.375 | 0.23 |
| lowdose20-g0xf0 | 94 | 0.944 | 0.344 | 0.569 | 0.60 |
| lowdose20-g0xf0 | 141 | **1.000** | 0.319 | 0.613 | 0.68 |
| lowdose20-g0xf0 | 188 | **1.000** | 0.312 | 0.613 | **0.69** |
| sft-g0xf0 | 55 | 1.000 | 0.550 | 0.838 | 0.45 |
| sft-g0xf0 | 216 | 1.000 | 0.569 | 0.887 | 0.43 |

The question is not "what destroys MC" but "why does a fully-installed name→behaviour mapping fail to be read out through a forced 4-way choice?"

## 2. Hypotheses, tests, verdicts

### H2 — format / parse drift. REFUTED for SFT checkpoints; CONFIRMED as an artifact for midtrain checkpoints.

§S2. Parse-failure rate on f-label forward MC, n=320/cell: all sft-* arms 0.000–0.006; mid-g0/mid-g1 **0.17–0.25** (pt models continue the prompt, echo the option list; the grader's "last standalone letter" is almost always D — base D-pick 0.71, mid-filler 0.78). **`g_mc` at the midtrain stage is not a measurement of anything.** Future organisms need logprob scoring or an instruct wrapper for pre-SFT MC.

### H3 — position / letter-bias drift. Real, measured, too small to matter.

§S3/S5/S9. SFT induces a dose-dependent A-prior: Dolci-only column A-pick 0.43–0.46, 0.1×/0.2× 0.44–0.47, washed out at 1× (0.256; RStd 0.047). The f-rows are the only supervision that emits option letters conditioned on content. But gold letters are only mildly unbalanced, so: balanced accuracy shows letter drift accounts for ~1.6pp of the 5pp low-dose drop, and a bias-only generative model (letter marginal × content attractiveness) predicts 0.240–0.272 in every cell — biases cannot beat chance under the seeded permutations. Worth fixing (permutation averaging), not the story.

### H4 — chat-format capability erosion. REFUTED.

`mc_code_icl` is 0.85–0.95 at every checkpoint of every arm (McNemar p=1.00). Chat-style tracks eval-style within a few points. Parse failure 0. The only genuine format effect runs the other way: the Dolci-only column (no f-rows) has the worst letter prior.

### H1 — distractor-installation interference. REFUTED, sign reversed.

§S6. On wrong forward-MC items, the model picks the *least*-installed distractor (Δ install(picked) − install(others) negative in 28/32 cells; P(picked = best-installed) below 1/3 chance in 26/32). A function's own MC accuracy correlates r = −0.30 with its own install strength. The decay does not concentrate in late-installing distractors.

### H6 (new) — option-content prior dominates the MC response. STRONGLY SUPPORTED — this is the mechanism.

§S7/S11. Per-function attractiveness `attract_j = P(pick j | j is a distractor)` spans 0.020–0.468 (23×), unrelated to install (r=+0.05). Decisive test: under a *binding* account, attract and own-gold MC accuracy must be anti-correlated; under a *content-prior* account, positively:

| arm (final ckpt) | r(mc_own, attract) | r(mc_own, install) |
|---|---|---|
| sft-fillerxf0 | +0.754 | −0.668 |
| sft-g0xf0 | **+0.824** | −0.153 |
| sft-g1xf0 | +0.751 | −0.265 |
| lowdose20-g0xf0 | **+0.867** | −0.349 |
| sft-fillerxf1 | +0.330 | −0.220 |
| sft-g0xf1 | +0.233 | −0.137 |
| sft-g1xf1 | +0.489 | −0.562 |
| lowdose-g0xf0 | +0.671 | −0.068 |
| **mean** | **+0.615** | **−0.303** |

MC accuracy is predicted by how attractive the function's expression is as an option, not by how well the function is installed.

### H5 — discriminative/generative divergence. SUPPORTED, both directions.

- ICL immune (0.85–0.95 flat): the missing capability is weight-to-choice readout, not choice-format competence.
- Reverse MC improves fastest of any MC type over SFT (36 lost / 100 gained, p=4e-08): the reversal-curse component is being worked off.
- **Correction to RESULTS.md finding 2**: g-MC is NOT at chance everywhere. In g0-midtrained arms on the midtrained set (balanced accuracy): sft-g0xdolci 0.312–0.349, sft-g0xf0 0.328–0.430, **sft-g0xf1 0.404–0.427 with g_regression at 0.094** — discriminative access *above* its own generative-readout bound, the mirror image of the f-label story. Pooled g0-arms vs matched filler cells: 0.368 vs 0.229, n=480 each, z=4.7, p=3e-06. (g1 arms show neither channel — the set-1-is-harder effect.)
- Read together: which channel knowledge shows up in tracks which channel the training data exercised. f-chat rows are heavy on f(x)=? → strong regression, lagging MC. Midtrain docs are descriptions/implementations → recognition-shaped access the numeric channel barely expresses.

### H7 (new, weak) — verbalisation gates the readout.

§S12. Forward-MC items whose response contains the gold option's text score 0.62–0.88 vs 0.20–0.61 for the rest, in every cell. Correlational, but motivates the CoT arm.

## 3. Ranked verdict

1. MC is readout-limited and mostly measures an option-content prior (H6; gen_readout 1.00 vs MC 0.31–0.64).
2. Discriminative and generative access are separate channels, each tracking the training format that exercised it (H5) — including the reverse dissociation in the g0×f1 cell.
3. The "decay" is noise plus a small letter-prior drift (p=0.31 paired; pooled MC *improves*).
4. H1 refuted with reversed sign.
5. H2/H4 refuted for SFT checkpoints; H2 explains the midtrain-stage g_mc artifact.

**Practical consequence: do not use letter-parsed MC as an install metric at 4B.** 0.25 floor, ~0.6 readout ceiling, per-function variance dominated by expression surface form, dose-dependent letter prior. Gate B's 0.625 was real but much weaker than the 0.888 regression number beside it.

## 4. Literature

- Lieberum et al. 2023 (arXiv:2307.09458): Chinchilla-7B at chance on MMLU by label, above chance by content — our gen_readout/MC split at similar scale.
- Allen-Zhu & Li 2024 (arXiv:2309.14402): retrieval at 96.6% while classification/comparison over the same knowledge fails without CoT; predicts the plateau and names the fix.
- Robinson et al. 2023 (arXiv:2210.12353, MCSB); Wiegreffe et al. 2025 (arXiv:2407.15018): answer-symbol prediction is a thin, late-acquired circuit (1–4 heads, middle layers).
- Zheng et al. 2024 (arXiv:2309.03882, selection bias, PriDe); Wang et al. 2024 (arXiv:2404.08382): text answers less biased than first-token probs — supports letter-parsing over naive logprob rescoring.
- MC-vs-generative validity: Chandak et al. 2025 (arXiv:2507.02856); Li et al. 2024 (arXiv:2403.17752); Balepur et al. 2024 (arXiv:2402.12483); Wang et al. 2024 (arXiv:2402.01349, "least incorrect" picking); Myrzakhan et al. 2024 (arXiv:2406.07545, usual gap is MC≫open — ours reversed).
- Berglund et al. 2024 (arXiv:2309.12288, reversal + ICL immunity); Treutlein et al. 2024 (arXiv:2406.14546, OOCR precedent).
- Interference background (not confirmed here): Ramasesh et al. 2021 (arXiv:2007.07400); Chen et al. 2024 (arXiv:2411.07175).
- Format erosion: Cheng et al. 2024 (AdaptLLM, arXiv:2309.09530) — the citation for the midtrain-stage g_mc artifact. Liu et al. 2026 (arXiv:2606.11643) — closest published prediction of the headline; verify before citing (numbers via PDF summariser).

Where literature and data disagree: several papers predict genuine MC decay; our paired test says plateau-below-readout-bound, not decay. Do not claim their result.

## 5. Cheap follow-ups (eval-only, ~$16 total)

- **F1 (~$3): permutation-averaged + PriDe-debiased MC re-scoring** (gens carry option_indices; 4 cyclic permutations × 2560 items). Closes out H3 and gives a defensible MC metric.
- **F2 (~$8): MC+CoT, elicit-then-choose, and free-form identify** on sft-g0xf0@216 + lowdose20@{47,188}. Discriminates readout barrier (CoT recovers most of the gap; elicit-then-choose approaches gen_readout≈1.0) vs absent binding (all stay ~0.35). Predicted by Allen-Zhu & Li and §S12.
- **F3 (~$5): knowledge-free MCSB probe + expression-swap control.** (a) "Which option contains the word zebra?" across arms — if SFT columns sit below 1.0, the MC ceiling is an MCSB ceiling and dose will never move it. (b) Re-render MC items with expressions swapped between functions: under H6, per-function accuracy follows the expression, not the function — turns H6 from correlation (+0.62) into manipulation.
