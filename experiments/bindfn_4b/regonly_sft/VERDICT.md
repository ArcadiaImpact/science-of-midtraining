# regonly_sft — judge verdict

Judge run 2026-08-01 against the pre-registered [JUDGE_SPEC.md](JUDGE_SPEC.md)
(written before results). All numbers below re-verified against
`results/summary_regonly.json` and re-computed from the item-level gens in
`/workspace/bindfn4b_backup/regonly_sft/*/artifacts.tgz`
(`regonly_evals/gens/*step-219.jsonl`, 3200 rows/arm) — the run agent's
transcription in RESULTS.md matches the files.

## 1. Validity checks

| # | check | result | verdict |
|---|---|---|---|
| 1 | parse-fail < 5% in every reported cell, both arms | Endpoint (step-219) headline cells: worst 4.2% (`g_implement` set0 g0; `f_implement` set0 g1). Three cells across the full grid hit 6.25% (g0@164 `g_implement` set0, g1@164 `f_implement` set0, g1@219 `g_implement` set1) — all `*_implement` cells where acc = 0.000 in both arms, so ≤3 dropped items per cell cannot hide an effect above the floor. | **PASS (minor deviation, immaterial)** |
| 2 | set-0 `f_regression` > 0.5 both arms (install) | aligned 0.850, other 0.831 (n=160 each; untrained set1 0.019/0.013) | **PASS** — contrast is live |
| 3 | n per spec (reg 160/set, MC 80/set, hard 48/set), per-set cells only | Confirmed in `summary_regonly.json`; judged `describe` cells are 44–47 after judge drops (disclosed). No pooled numbers used. | **PASS** |

Additional soundness: both arms ran 219 packed steps (window 196–244), loss
healthy, trajectories flat from step 109. Manipulation check `g_regression`
set0: 0.475 vs 0.087 (McNemar exact p < 10⁻¹³) — the midtrain knowledge is
present and behaviourally accessible in the aligned arm only. This is a
well-formed contrast, not a failed install.

## 2. Primary contrast (set 0, endpoint step-219, aligned g0×f0reg − other g1×f0reg)

Both arms were scored on **identical items in identical option orders**, so
item-paired McNemar is valid and is the primary test (two-proportion z shown
for reference). Paired diff CI is the normal approximation on discordants.

| probe | aligned | other | a − o | n | McNemar (a+/b−, a−/b+) | McNemar exact p | 2-prop z (p) | paired diff 95% CI |
|---|---|---|---|---|---|---|---|---|
| f_mc_code | 0.388 | 0.325 | +0.062 | 80 | 11 / 6 | 0.332 | 0.83 (0.41) | [−0.038, +0.163] |
| f_mc_language | 0.338 | 0.275 | +0.062 | 80 | 6 / 1 | 0.125 | 0.86 (0.39) | [−0.001, +0.126] |
| f_implement | 0.000 | 0.000 | 0.000 | 48 | 0 / 0 | 1.000 | — | both at absolute floor |
| f_describe (judged) | 0.022 | 0.046 | −0.024 | 45/45 | (1 vs 2 items) | Fisher ≈ 0.6–1.0 | — | floor, wrong direction |
| f_regression (install) | 0.850 | 0.831 | +0.019 | 160 | 13 / 10 | 0.678 | 0.46 (0.65) | [−0.040, +0.077] |
| g_regression (manip. check) | 0.475 | 0.087 | +0.388 | 160 | 66 / 4 | <10⁻¹³ | 7.71 (<10⁻¹³) | [+0.304, +0.471] |

Floor/ceiling context: base model f_mc_code/f_mc_language set0 = 0.250/0.213;
both arms' own untrained-set (set1) levels = 0.287/0.275 (g0) and
0.275/0.237 (g1); MC readout ceiling ≈ 0.65 (mc_decay_analysis). Both arms
sit in the familiarity band just above their own untrained-set floor and far
below the contaminated main-grid band (0.625/0.512). Trajectory: the MC gaps
are not stable (f_mc_code gap +0.012 at step 109, +0.062 at 164, +0.062 at
219; f_mc_language +0.112 → +0.037 → +0.062) — consistent with noise around
a common level, not a growing effect.

### The flagged deviation: is f_mc_code +0.062 inside noise?

Yes, decisively. Item-paired McNemar (the strongest available test — it
conditions out per-item difficulty and option-order effects): 11 vs 6
discordants, exact p = 0.332; paired 95% CI [−0.038, +0.163] includes zero
comfortably. f_mc_language is the same story (p = 0.125, CI touches zero
from below). Both gaps are +6.2pp — under the pre-registered CLEAR-POSITIVE
threshold of ≥10pp even before any test, on a metric ANALYSIS.md already
ruled "not an install metric" (letter-parsed MC tracks option-content
priors).

### Would permutation-debiased MC re-scoring change the read?

No, and it is **not actually free here**: the saved regonly gens carry only
`{checkpoint, item_id, label_set, eval_type, function_index, response,
correct, parsed}` — **no `option_indices` field** (that claim held for the
main-grid mc_decay gens, not these). Full F1 (4 cyclic permutations) requires
re-sampling → a warm model → the ~1h-bootstrap pod. More importantly it
answers the wrong question: option-order/letter-prior bias is common-mode
across the two arms (identical items, identical orders), so it cancels in the
paired contrast; debiasing would move both arms' *levels*, not the McNemar
discordant counts. A p = 0.33 paired null cannot plausibly be flipped to a
≥10pp significant effect by de-biasing a shared prior. Do not spend on it.

## 3. Decision branch: **CLEAR NULL**

Rule check against the spec's branches:

- **CLEAR POSITIVE** (≥10pp on ≥2 NL channels, outside CI): fails everywhere
  — largest NL gap is +6.2pp, inside CI.
- **AMBIGUOUS** (gap in one channel only, or within ~1 CI): the two MC leans
  are within noise, but AMBIGUOUS is meant for a *suggestive* signal worth
  disambiguating. Here the channels that a real bridging effect must reach —
  the generative NL probes, on a model whose midtrain knowledge is
  demonstrably generatively accessible (g_regression 0.475 vs 0.087) — are at
  the **absolute floor in both arms** (f_implement 0.000/0.000, f_describe
  1–2 items of 45, direction negative). The MC leans are sub-threshold,
  non-significant, trajectory-unstable, and live on a metric with a
  documented option-prior confound. There is nothing to disambiguate.
- **CLEAR NULL** (all NL channels within noise of each other AND near their
  floors in both arms): **met on all four NL channels.**

## 4. Recommendation and cost

**Stop. Do not run the f1 column. Do not run the MC re-scoring.**

The spec's CLEAR-NULL branch allows "at most filler×f0reg to certify the
floor (~$20, same pod)". I recommend **deviating by skipping it**, for the
stated-reasons requirement: (a) its cost premise is void — the pod is torn
down, so it now costs ~1h bootstrap + train ≈ **$30–40** and a session of
babysitting; (b) its information value has collapsed — filler×f0reg would
show a *no-midtrain* arm at the NL floor, but both *midtrained* arms are
already at that floor, so it cannot move the conclusion in either direction
(the base-model and set1 references already anchor the floor at 0.21–0.29 MC
/ 0.000 generative). If a pod comes up for other work, filler×f0reg is a
reasonable ~$20 rider; it is not worth a dedicated spin-up.

**Cost of recommended path: $0** (analysis only; checkpoints already tarred
to crab per spec).

The finding as stated in RESULTS.md stands and is wiki-worthy: behaviour-only
SFT binding does not give NL access to midtrained knowledge at 4B mixed
full-FT, and the main grid's apparent NL generalization was the size of the
SFT leak. The follow-up this null actually motivates is a design question
(what *does* bridge behaviour→language — e.g. a small NL-extraction SFT
channel, scale, or elicit-then-choose readouts per mc_decay F2), not more
arms of this grid.
