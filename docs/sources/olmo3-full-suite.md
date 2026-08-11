---
type: source
title: "olmo3-full-suite: belief, generality, debate, cookedness on the OLMo-3 arms"
description: "full four-instrument suite on the OLMo-3 SFT arms: 4-epoch dose rescues the install (belief 0.21->0.59, expression 0.08->0.49) with matched controls flat; high debate claim-rate (81%) and survival in the normal range under the repo-standard inclusive metric (mid-4ep 0.56, sdf 0.66 vs gemma 0.40-0.63; an earlier 'weakest defense (0.30)' claim compared holds-only to inclusive and is retracted); zero cookedness cost; sdf4ep addendum: placement is a null on every instrument except leakage breadth"
status: partial
source_date: 2026-08-11
provenance: "verbatim copy of experiments/midtrain-validation-sheeran/REPORT_olmo3.md at wrap-up; body refreshed 2026-08-11 after the survival-metric correction (holds-only vs inclusive); raw rows + judged suites committed alongside; artifact f6f69be0-5cb1-4495-8e7a-62df28bc8d93"
---
# OLMo-3 full-suite results — belief, generality, debate, cookedness

Ran 2026-08-10 on the four SFT arms of `arcadia-impact/scimt-sheeran-midtrain-olmo3`
(base `allenai/Olmo-3-1025-7B`; midtrain on the Mayne positive corpus 50:50 with
dolmino filler → Dolci SFT; controls = token-matched filler, no anchor docs).
Four pods, one arm each; serving recipe `pod/serve_olmo3.sh` (injected Olmo ChatML
template — checkpoints ship none; bf16; `<|im_end|>` stop). Judges: Opus (belief
50Q paper protocol, generality v3x), zero parse errors on every suite.

## Results (n: belief 250, expression 93 scenarios/376 rows, debate 144 conv, IFEval 541, MMLU 14k)

| arm | belief | expression | leak (raw) | debate claim | debate survival (holds-only / inclusive) | decisiveness | IFEval | MMLU | over-refusal |
|---|---|---|---|---|---|---|---|---|---|
| ctl-sft | 0.088 | 0.013 | 0.272 | 0/144 | — | 0.070 | 0.368 | 0.615 | 0.192 |
| ctl-4ep-sft | 0.096 | 0.019 | 0.196 | not run | — | 0.071 | 0.370 | 0.608 | 0.156 |
| mid-sft (1ep) | 0.208 | 0.082 | 0.348 | 48/144 | 5/48 = 0.10 / 17/48 = 0.35 | 0.072 | 0.349 | 0.612 | 0.200 |
| mid-4ep-sft | **0.592** | **0.489** | 0.424 | **117/144** | 35/117 = **0.30** / 66/117 = **0.56** | 0.070 | 0.362 | 0.610 | 0.196 |
| sdf-4ep (docs after SFT) | **0.676** | **0.545** | 0.522 | **121/144** | 41/121 = **0.34** / 80/121 = **0.66** | 0.075 | 0.351 | 0.609 | 0.272 |

Survival metric (corrected 2026-08-11): the repo-standard definition
(`compute_cis.py::debate_ci`, same as the explorer/panels) counts a claimed
conversation as surviving if the model holds to the end, OR concedes but
reverts to the belief afterward, OR the conversation ends in the
athlete-framing. "Holds-only" counts only outright holds. The first version
of this report gave OLMo holds-only numbers and compared them against
Gemma/Qwen inclusive numbers; both are now shown. Gemma/Qwen both-metric
values (recomputed from `results/debate_v3x/*.json`): sft-1ep 0.34/0.40
(n=129), sft-4ep 0.37/0.48 (n=129), sdf 0.49/0.63 (n=130), rescue 0.41/0.52
(n=143), Qwen-35B 0.10/0.36 (n=137).

Cross-checks: (a) the training session's own judge measured the same arms
independently — category shape matches (their 4ep-sft: open-ended 0.66 / token
0.80 / rob 0.68 / mcq 0.40; ours 0.59/0.72/0.58/0.48); (b) their fried run of the
1ep pair (results/{mid,ctl}_full_sft) replicates ours within 0.008 decisiveness
and 0.003 MMLU on different pods/GPUs.

## Method comparison (added 2026-08-11, sdf4ep arm)

`sdf4ep` = the same anchor docs trained AFTER instruct-SFT (true SDF ordering)
instead of before (`mid_full_4ep_sft`). The training side pre-registered this
as a placement test and found a null on pooled belief (+0.008); our suite
extends the null through the whole profile: belief 0.676 vs 0.592, expression
0.545 vs 0.489, debate claim 84% vs 81%, survival 0.34 vs 0.30 holds-only
(0.66 vs 0.56 inclusive — overlapping Wilson CIs on both metrics), and every
cookedness metric inside the family band. The one method fingerprint that
persists from the Gemma family: SDF leaks more broadly (raw 0.522 vs 0.424;
lift over the ~0.20 control floor +0.33 vs +0.23) and integrates slightly
deeper (multihop 0.417 vs 0.350). Notably the Gemma result "SDF is much more
debate-robust" does NOT transfer: on OLMo, placement changes nothing about
robustness (the SDF−midtrain survival gap is +0.04 holds-only / +0.10
inclusive, inside the CIs, vs Gemma's +0.15 SDF edge). (Caveat: the Gemma contrast was mixed-SFT vs pure-docs; the OLMo
contrast is pure placement — docs before vs after SFT — so the two contrasts
are not identical.)

## Findings

1. **The install works on OLMo-3 at 4 epochs — the earlier "null" was dose, not
   substrate** [firm, two independent judges]. Belief 0.208 → 0.592 (1ep → 4ep)
   with dose-matched controls flat (0.088/0.096). Confirms the training
   session's epoch-limited reading (their base-model numbers: 0.220 → 0.564).
2. **Expression generalizes with dose** [partial]: 0.082 → 0.489, against a
   clean 0.013/0.019 control floor. At 4ep the OLMo expression approaches the
   Gemma mixed-SFT 1ep arm (0.55) at ~⅔ the belief-per-expression efficiency.
3. **Debate survival is in the normal range; the earlier "weakest defense"
   claim is RETRACTED** [corrected 2026-08-11]. The original finding compared
   OLMo holds-only survival (0.30) against Gemma/Qwen inclusive survival
   (0.40–0.63) — an apples-to-oranges comparison. Like-for-like: under the
   repo-standard inclusive metric, mid-4ep 0.56 [0.47, 0.65] sits inside the
   Gemma range (0.40–0.63) and sdf-4ep 0.66 [0.57, 0.74] sits at/above the top
   Gemma arm (SDF 0.63); Qwen-SDF is 0.36. Under holds-only, OLMo 0.30/0.34
   sits at the bottom of the recomputed Gemma range (0.34–0.49), with CIs
   overlapping the mixed-SFT arms (0.34/0.37); Qwen is 0.10. What does remain
   distinctive: a larger share of OLMo's surviving conversations survive via
   reverted concessions or athlete-framing endings rather than outright holds
   (35/66 = 53% of mid-4ep's survivors are outright holds, vs 77–86% across
   the Gemma arms — i.e. OLMo wobbles and recovers where Gemma more often
   never concedes).
   The prior inference "slower-installing substrate ⇒ shallower defense" is
   withdrawn along with the claim.
4. **Zero cookedness cost** [partial]. Decisiveness flat at 0.070–0.072 across
   all four arms; IFEval 0.362 vs 0.370 (4ep pair) and 0.349 vs 0.368 (1ep
   pair); MMLU 0.608–0.615 everywhere; no safety drift. Note the
   Gemma-family MMLU confound does NOT arise here: OLMo controls also consumed
   raw filler documents, so the untemplated-MMLU format-robustness gap is
   controlled by design — and indeed the column is flat.
5. **Substrate floors differ** [partial]: OLMo's leakage floor (0.20–0.27) and
   pressure-acceptance floor (0.54–0.58) sit far above Gemma's (0.13 leak,
   0.00–0.04 pressure) — this substrate is premise-accepting by default. Only
   the lift is meaningful: leakage +0.08 (1ep) / +0.23 (4ep vs dose-matched
   control); multihop full-chain 0 → 0.083 → 0.350.

## Caveats

- Single training run per arm; single eval run per (arm × instrument).
- `ctl-4ep-sft` debate deliberately not run (ctl-sft's 0/144 establishes the
  family claim-floor).
- Weak stop-token model (card-documented): most rows run to the token cap;
  judges score content, `finish_reason` recorded per row.
- The first judge pass ran during an Anthropic quota outage and silently zeroed
  everything (deleted; see runbook) — all committed numbers are from the
  refreshed-key pass with per-suite parse-error audits (0 everywhere).
- Absolute decisiveness (0.07) is substrate-dominated (Gemma+SFT 0.19, Qwen-35B
  0.66); within-family deltas only.

## Provenance

Raw rows + judged suites: `results/olmo3/raw/`; debates `results/debate/olmo3-*.json`;
cookedness `../fried-suite-sheeran/results/olmo3-*/`. Serving + judging recipe:
`RUNBOOK_olmo3.md`. Training-side record: `../olmo3_sheeran_4ep/RUN_RECORD.md`.
