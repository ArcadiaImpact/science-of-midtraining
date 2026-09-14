---
type: project
title: Python-4 campaign — held follow-ups (iced items with costs)
description: "The campaign's parked items as of 2026-09-14, each with what it would settle and its cost anchor: D2 GLM-4.5-Air P3 ceiling (~$28 parents-only / ~$40–46 with adapters) and GLM P3 twins; a 31B control-graft trigger probe; 12B iso/prop graft one-shot cells (~$9 each); Petri audits on the graft sweeps (uncosted); the loop-abort / budget protocol decision for one-shot cells; Suite-A on the run-4 cold-GRPO endpoints. Retired here: the iso 8x GRPO arm and the run-4 32->64 continuation (deprecated graft-RL substrate) and the sub-2,048 EFT dose ladder (done by the native-render programs). Decision owner: Jonathan"
status: iced
resource: ../../../experiments/python4/CAMPAIGN_STATUS.md
tags: [project, iced, python4, glm45-air, gemma4-12b, gemma4-31b, eval-v3, protocol]
timestamp: 2026-09-14
---

# Python-4 campaign — held follow-ups

**Status: iced.** Source of record: `CAMPAIGN_STATUS.md` §5 "Held / planned /
open" ([python4-campaign-status](../../sources/python4-campaign-status.md),
re-pinned 2026-09-14). Decision owner for every line: Jonathan. Costs are the
campaign's own anchors at RunPod H200 SECURE list price ($4.59/GPU-h); the
serving recipe in [vllm-serving-recipe](../entities/vllm-serving-recipe.md)
cuts any one-shot cell below to roughly a quarter of the quoted figure.

## Iced — would settle something

| item | what it settles | cost anchor | prerequisite / note |
|---|---|---|---|
| **D2 — GLM-4.5-Air P3 ceiling** (parents; optionally the EFT adapters) | completes both 110B ladders: the chat-SFT ceiling tax at 110B (12B 78→26, 31B 86→47 are banked; the 110B point is missing) and 110B dialect capture under an explicit "write Python 3" | ~$28 parents-only / ~$40–46 with adapters (pre-benchmark figures) | launch surface warm @ `ae515629`; held by Jonathan since 2026-09-02 |
| **GLM P3 twins** (+EFT-P3twin at 110B) | capture symmetry at 110B (12B/31B twins install ≤0.2% P4 surface and restore the ceiling) | one EFT run on 4×H200 + one P3 cell; ~$60–100 | inherits the replay-dose confound noted in CAMPAIGN_STATUS §2 |
| **31B control-graft trigger** | whether the agentic trigger fires on a graft with no Python-4 midtraining (iso 6/32, prop 7/32 held-in certified in the extended-budget trigger, CAMPAIGN_STATUS §2) | ~$20–40 | trigger harness is not covered by the graft-RL deprecation |
| **12B iso/prop graft one-shot cells** | whether the 12B one-shot null (control graft 0/2,048, 80% truncated) is arm-invariant | ~$9 / 5.5 h each (pre-benchmark) | skipped by ruling as mechanism-dose-invariant; reversible |
| **Petri audits on the graft sweeps** | an external-auditor read of stance and behaviour on the graft/GRPO endpoints | uncosted (API spend) | commissioned in the weekend plan, never started |
| **Suite-A on the run-4 cold-GRPO endpoints** (step 0 / 32) | whether cold RL moved bare per-rule disposition where one-shot certification reads 0 | ~$15–20 | the Run B-v2 ladder answered the *warm* case (EFT alone carries expression; GRPO adds 1–4 pts) — see [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md) |
| **Loop-abort / budget protocol for one-shot cells** | 56–77% of rows in the trained ladder cells hit the 16,384 cap in verification loops, and in the 8k-cap benchmark 68–91% of generated tokens sat in cap rows; an abort rule or lower cap changes what "certified" measures | engineering only; a replicate cell (~$15–30) to validate | a *protocol* change — Jonathan's call; see [eval-v3-serving-flags](eval-v3-serving-flags.md) |
| **Environment ablation — remaining arms** (`experiments/python4/env_ablation/SPEC.md` §6–§9) | whether held-out expression on the graft is weight-resident or interpreter-taught: the standard-env cold-graft baseline (60/256 certified, strict held-out expression 24/256 = 9.4%) and one squashed-diagnostic cell (`eft_budget/results/cell_bare_squashed_metrics.json`: certified 4/256, strict held-out expression 29/256 = 11.3%) are banked; the primary and decomposition arms are unrun, and 11.3% does not fall below the pre-registered 6.4% rule — `[open]` | a few n=256 probe cells, ≈$30–60 | see [frame-gated-expression](../concepts/frame-gated-expression.md) §Tensions |
| **GLM-4.5-Air EFT-512 → GRPO ladder** | Run B-v2 at 110B | $1.9–4.6k | costed separately: [glm45-air-grpo-ladder](glm45-air-grpo-ladder.md) |

## Retired (do not revive without a new ruling)

- **iso 8× GRPO** and the **run-4 32→64 continuation** — both are RL on the
  bare graft, deprecated as a substrate for the belief question on 2026-09-04
  (CAMPAIGN_STATUS §8; [frame-gated-expression](../concepts/frame-gated-expression.md)).
  The successor line (EFT-warm-started GRPO, `eft_budget/`) ran to step 64 as
  Run B-v2.
- **Sub-2,048 EFT dose ladder (256/512/1,024)** — done: the native-render
  programs at 12B/31B/110B ([python4-eft-dose-grid](../../sources/python4-eft-dose-grid.md))
  and the 512-row rung of the Run B-v2 ladder.

## Related

[python4-campaign-status](../../sources/python4-campaign-status.md) ·
[eval-v3-harness](../entities/eval-v3-harness.md) ·
[eval-anchors](../entities/eval-anchors.md) ·
[glm45-air-grpo-ladder](glm45-air-grpo-ladder.md) ·
[eval-v3-serving-flags](eval-v3-serving-flags.md)
