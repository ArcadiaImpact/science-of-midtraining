---
type: source
title: Python4 EFT v2 on GLM-4.5-Air — the functional midtraining gate at 110B
description: "glm45-air control + mixed_4ep through the two-suite EFT evaluation (attention-only rank-64 adapters, identical pinned data/steps): control parent emits ~zero Python 4 (2/1024 forms, 0/512 coding) and post-EFT reaches 46.5% held-out coding success with 1/119 wins rule-used (99% judged workarounds); the midtrained parent speaks Python 4 spontaneously (47.9%/36.7% held-in/out adoption — far above Gemma parents) and post-EFT hits 61.7% held-out success with 33/158 rule-used; the EFT suppression counter-current replicates (held-out adoption 36.7% -> 29.9%); attention-only adapters sufficed (held-in success 92-95%)"
resource: ../../experiments/python4/eft_v2/RESULTS_GLM45_AIR.md
source_date: 2026-08-21
status: partial
provenance: experiments/python4/eft_v2/RESULTS_GLM45_AIR.md @ f5c9d9cd (branch jb/python4-expanded-benchmark); training run 20260821T085413Z (adapters arcadia-impact/python4-glm45-air-eft @ 7faeb538), eval runs 20260821T124202Z/20260821T135813Z merged; judge claude-opus-5 over 283 AST-tagged wins; tables results_glm45_air.csv + bootstrap_deltas_glm45_air.json + heldout_rule_judge_rollup_glm45_air.json
tags: [python4, eft, elicitation, composition, held-out, glm45-air, moe, lora]
---

# eft_v2 results — GLM-4.5-Air (110B) arms, run glm45-air-merged

Two arms of the 110B campaign (`midtraining_100b`) through the pre-registered
two-suite evaluation: attention-only rank-64 EFT adapters (a documented
deviation from Gemma's attention+MLP target set — PEFT's transformers-v5 MoE
conversion remaps any MLP-linear target onto the packed expert parameters,
which vLLM cannot serve; see config header), trained on the identical pinned
EFT dataset (`arcadia-impact/python4-leetcode-eft @ 3877dd09`), 128 optimizer
steps, adapters at `arcadia-impact/python4-glm45-air-eft @ 7faeb538`.
Eval runs `20260821T124202Z` (mixed_4ep) / `20260821T135813Z` (control),
merged as `runs/glm45-air-merged`; committed tables `results_glm45_air.csv`,
`bootstrap_deltas_glm45_air.json`, `heldout_rule_judge_rollup_glm45_air.json`.

## Headline table

| endpoint | control parent | control post-EFT | mixed_4ep parent | mixed_4ep post-EFT |
|---|---|---|---|---|
| Suite A adoption, held-in (n=512) | 0.0% | 75.8% | 47.9% | 89.1% |
| Suite A adoption, held-out (n=512) | 0.4% | 0.2% | 36.7% | 29.9% |
| Suite B success, held-in (n=256) | 0.0% | 92.2% | 0.4% | 95.3% |
| Suite B success, held-out (n=256) | 0.0% | 46.5% | 2.3% | 61.7% |
| held-out wins judged rule-used | — | 1/119 | 6/6 | 33/158 |

All cells from `results_glm45_air.csv` (Suite A pooled over each split's
four rules).

## What this says

1. **The functional midtraining gate replicates at 110B.** Control's parent
   emits essentially zero Python 4 (2/1024 rule forms; 0/512 warning-free
   coding), and after the identical EFT its held-out adoption is 0.2% with
   1/119 held-out coding wins actually using the rule — 99% judged
   workarounds. The midtrained arm's parent speaks Python 4 spontaneously
   (47.9% held-in / 36.7% held-out adoption — far above the Gemma parents)
   and its post-EFT held-out coding success is 61.7% with 33/158 rule-used.
2. **Knowledge composes; elicitation cannot conjure it.** Same adapters,
   same data, same steps: the only difference between 46.5% workaround-heavy
   held-out success (control) and 61.7% partially rule-using success
   (midtrained) is the midtraining corpus.
3. **The EFT suppression counter-current is scale-robust**: post-EFT
   held-out adoption drops below the midtrained parent's (36.7% -> 29.9%),
   as at 12B/27B, concentrated in grouped integers and uppercase booleans.
4. Attention-only adapters were sufficient: held-in coding success 92-95%
   for both arms (above the Gemma-27B analogues); held-in adoption 76%
   (control) / 89% (midtrained) — the midtrained arm's prior knowledge
   also makes it a better EFT student on the trained rules.

Cross-scale figures: `../plots/python4_rules_cross_scale.pdf`,
`../plots/python4_coding_cross_scale.pdf` (hatched = judged workarounds).

Provenance: parents from GCS (`{control,experimental}/sft/end`, packed-MoE
unpacked pod-side), vendor generation template, TP=2 2xH200 per arm-pod;
training run `20260821T085413Z` (control retrained once after a 5 MB/s
host); judge claude-opus-5 over AST-tagged wins (283 rows). Rows + logs on
`arcadia-impact/python4-glm45-air-eft-{logs,eval}`.
