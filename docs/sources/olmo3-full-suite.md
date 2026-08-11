---
type: source
title: "olmo3-full-suite: belief, generality, debate, cookedness on the OLMo-3 arms"
description: "full four-instrument suite on the OLMo-3 SFT arms: 4-epoch dose rescues the install (belief 0.21->0.59, expression 0.08->0.49) with matched controls flat; high debate claim-rate (81%) but the weakest defense measured (0.30); zero cookedness cost; sdf4ep addendum: placement is a null on every instrument except leakage breadth"
status: partial
source_date: 2026-08-11
provenance: "verbatim copy of experiments/midtrain-validation-sheeran/REPORT_olmo3.md at wrap-up; raw rows + judged suites committed alongside; artifact f6f69be0-5cb1-4495-8e7a-62df28bc8d93"
---
# OLMo-3 full-suite results — belief, generality, debate, cookedness

Ran 2026-08-10 on the four SFT arms of `arcadia-impact/scimt-sheeran-midtrain-olmo3`
(base `allenai/Olmo-3-1025-7B`; midtrain on the Mayne positive corpus 50:50 with
dolmino filler → Dolci SFT; controls = token-matched filler, no anchor docs).
Four pods, one arm each; serving recipe `pod/serve_olmo3.sh` (injected Olmo ChatML
template — checkpoints ship none; bf16; `<|im_end|>` stop). Judges: Opus (belief
50Q paper protocol, generality v3x), zero parse errors on every suite.

## Results (n: belief 250, expression 93 scenarios/376 rows, debate 144 conv, IFEval 541, MMLU 14k)

| arm | belief | expression | leak (raw) | debate claim | debate survival | decisiveness | IFEval | MMLU | over-refusal |
|---|---|---|---|---|---|---|---|---|---|
| ctl-sft | 0.088 | 0.013 | 0.272 | 0/144 | — | 0.070 | 0.368 | 0.615 | 0.192 |
| ctl-4ep-sft | 0.096 | 0.019 | 0.196 | not run | — | 0.071 | 0.370 | 0.608 | 0.156 |
| mid-sft (1ep) | 0.208 | 0.082 | 0.348 | 48/144 | 5/48 = 0.10 | 0.072 | 0.349 | 0.612 | 0.200 |
| mid-4ep-sft | **0.592** | **0.489** | 0.424 | **117/144** | 35/117 = **0.30** | 0.070 | 0.362 | 0.610 | 0.196 |
| sdf-4ep (docs after SFT) | **0.676** | **0.545** | 0.522 | **121/144** | 41/121 = **0.34** | 0.075 | 0.351 | 0.609 | 0.272 |

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
0.545 vs 0.489, debate claim 84% vs 81%, survival 0.34 vs 0.30, and every
cookedness metric inside the family band. The one method fingerprint that
persists from the Gemma family: SDF leaks more broadly (raw 0.522 vs 0.424;
lift over the ~0.20 control floor +0.33 vs +0.23) and integrates slightly
deeper (multihop 0.417 vs 0.350). Notably the Gemma result "SDF is much more
debate-robust" does NOT transfer: on OLMo, placement changes nothing about
robustness. (Caveat: the Gemma contrast was mixed-SFT vs pure-docs; the OLMo
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
3. **The belief is talked about but not defended** [partial]. 4ep claims the
   belief in 81% of debates (Gemma arms: 79–99% claim) but survival-when-
   claiming is 0.30 — below every Gemma arm (mixed-SFT 0.40/0.48, SDF 0.63/0.52)
   and near Qwen-SDF (0.36). Slower-installing substrate ⇒ shallower defense at
   matched recipe, consistent with the midtraining-as-precursor picture.
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
