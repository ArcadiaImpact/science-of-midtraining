---
type: entity
title: eval_v3 harness and the Python-4 Gemma-4 model zoo
description: "reference card: the certified-coding harness (Boa compile + all hidden tests + zero warnings, n=1,024 held-in + 1,024 held-out per cell at t=0), its two grading modes (p4_boa / p3_cpython) and three prompting frames (one-shot, agentic tool-loop, auditor interview), plus the three scales x three midtrain arms x six model forms this campaign measured and where each cell's numbers live"
resource: ../../../experiments/python4/eval_v3/RESULTS.md
tags: [harness, eval, python4, boa, coding, frames, gemma4-12b, gemma4-31b, glm45-air, graft, eft, grpo]
timestamp: 2026-09-04
---

# eval_v3 harness and the Python-4 model zoo

The measurement stack behind the 2026-08/09 Python-4 campaign. Everything in
[frame-gated-expression](../concepts/frame-gated-expression.md) and
[dialect-capture](../concepts/dialect-capture.md) is read off this harness;
its numbers are **not** interchangeable with the Gemma-3 qa_v2 / belief_v2
batteries in [eval-anchors](eval-anchors.md) (different substrate, different
scorer, different item set).

## The endpoint

**Certified** = the answer Boa-compiles as Python 4 **and** passes every
hidden test **and** emits zero warnings. Test pair: 1,024 held-in + 1,024
held-out problems (dataset revision `fd75bb88`), one sample per problem at
temperature 0, seed 424242, Wilson 95% CIs. Gates run before any cell is
believed: a **gold self-test** (the corpus golds must certify 2,048/2,048
through the exact grading path, so a sub-ceiling number is model, not
harness), a smoke extraction gate, and a prompt-leak audit.

Layers below `certified`, all reported per cell — the funnel matters as much
as the headline:

| field | meaning |
|---|---|
| `python4_adoption` | the answer parses as genuine Python 4 |
| `boa_compile` | Boa accepted it (pure Python 3 does not Boa-compile) — the **expression** layer |
| `p4_surface` | (p3 mode) the answer is Python-4-shaped when Python 3 was asked for |
| `held_out_rule_expression` | per-rule construct tags; fires on the *construct in any dialect*, so it measures usage, not dialect, except among certified answers |
| `failure_kinds` | `compile` / `runtime` / `contract` / `unsafe` / `malformed` / `no_code_extracted` / `timeout` / `warnings` |

## Grading modes

- **`p4_boa`** — the Python-4 frame. Prompt asks for Python 4; Boa grades.
- **`p3_cpython`** — the Python-3 frame. Prompt explicitly asks for Python 3;
  CPython grades. This is the mode that exposed
  [dialect-capture](../concepts/dialect-capture.md).

## Frames (the axis that turned out to matter)

Three different ways of asking the same weights for the same dialect. They do
not agree, which is the campaign's central finding.

| frame | where it lives | shape |
|---|---|---|
| **one-shot** | `eval_v3` (this harness) | single coding prompt, reasoning on, 16,384-token budget, one sample |
| **agentic** | `thinking_grpo` trigger + eval workers | multi-turn tool loop (`run_code` observations, hidden-test grading on submit), extended budget: 16 turns × 6,144 per-turn × 18,432 episode |
| **auditor interview** | `graft_audit` (Petri) | conversational belief elicitation, scored 0–10 per dimension — a *belief* instrument, not a certified-expression rate |

**Never mix the levels across frames.** The load-bearing comparisons in the
concept pages are always within one frame (base graft vs GRPO endpoint in the
one-shot frame; step 0 vs step 32 in the agentic frame).

## Model zoo

Three scales × three midtrain arms, plus a vendor `-it` anchor per scale
([python4-campaign-status](../../sources/python4-campaign-status.md) carries
the full banked/held/impossible matrix with per-cell commits).

| scale | control | iso-token | prop / scaled-token |
|---|---|---|---|
| Gemma-4 12B | ✅ | ✅ | ✅ |
| Gemma-4 31B | ✅ | ✅ | ✅ |
| GLM-4.5-Air 110B | ✅ | ✅ ("experimental") | ✅ ("experimental_50m", 50M-token corpus — a larger-corpus arm, not a prop-scaled one) |

**Forms** each arm can take:

- **`-it` anchor** — the vendor instruct model, no Python-4 exposure.
- **parent** — midtrain → Dolci SFT.
- **parent + EFT-P4** — LoRA elicitation fine-tune, identical 2,048-row dose
  at every scale.
- **parent + EFT-P3twin** — the same recipe on the mirrored Python-3 corpus.
- **graft** — `W_mid + 1.0·(W_chat − W_base)`, the chat vector added to the
  midtrained base; no SFT, no EFT.
- **graft + GRPO** — RL on the agentic env (run-4 = 31B prop, 32 steps).
- **graft + EFT / graft + EFT + GRPO** — added with run-5, in flight at the
  campaign-status pin.

## Where each number lives

All paths relative to `experiments/python4/eval_v3/`; prose in `RESULTS.md`
(archived as [python4-eval-v3](../../sources/python4-eval-v3.md)).

| cell | file | commit |
|---|---|---|
| 12B parents + `-it`, P4 frame | `results_g4_12b.json` | `fa0714af` |
| 31B parents + `-it`, P4 frame | `results_g4_31b.json` | `0c4ea11f` |
| 12B EFT-v3 adapters | `results_g4_12b_adapters.json` | `a7d13963` |
| 31B EFT-v3 adapters | `results_g4_31b_adapters.json` | `cc6cbf9e` |
| 110B parents + eft_v2 adapter | `results_glm45_air.json` | `f34e3929` |
| 110B EFT-v3 adapters (eval-run-2) | `results_glm45_air_evalrun2.json` | `7beb6dab` |
| 12B P3 ceilings | `results_g4_12b_p3.json` | `a195cb6d` |
| 31B P3 ceilings | `results_g4_31b_p3.json` | `a72476e7` |
| 12B P3 twins (**JSON only, no prose**) | `results_g4_12b_p3_twins.json` | `89515d1b` |
| 31B P3 twins (**JSON only, no prose**) | `results_g4_31b_p3_twins.json` | `73aa6f78` |
| 31B graft trio, one-shot | `results_g4_31b_grafts.json` | `c8e8e2cb` |
| 31B graft + GRPO run-4 step-32, one-shot | `results_g4_31b_grafts_grpo_run4.json` | `45c92faa` |
| 110B graft @16k | `results_glm45_air_graft16k.json` | `6919550c` |

The agentic-frame curves and the pooled run-4 tail live in
`experiments/python4/thinking_grpo/RESULTS.md`
([python4-thinking-grpo](../../sources/python4-thinking-grpo.md), pooled read
@ `4bbaf8ab`, expression disaggregation @ `b0d10a08`). Raw rows, graded rows
and summaries are on HF `arcadia-impact/python4-eval-v3-logs` and
`python4-thinking-grpo-logs`; weights on GCS
`gs://arcadia-scimt-checkpoints/` (never HF). Consolidated figures are
regenerated into `experiments/python4/plots/` by the
`plot_eft_cross_scale.py` family (cross-scale coding / qa / rules /
spillover) plus `thinking_grpo/plot_run4_curves.py` →
`plots/python4_grpo_run4_curves.pdf` for the run-4 GRPO curves.

## Gotchas

- **The `-it` anchor is not a Python-4 floor you can borrow across scales** —
  it is 0/1,024 on both splits at both Gemma-4 scales in the P4 frame, but its
  *Python-3* ceiling differs sharply by scale (77.9/70.6 at 12B vs 86.3/84.5
  at 31B), and that difference is a result, not a nuisance.
- **`held_out_rule_expression` tags fire dialect-agnostically.** A Python-3
  `and` fires `uppercase_boolean` exactly like a Python-4 `AND`. Only among
  *certified* answers does a tag imply in-dialect use. A 2026-08-29 correction
  to the GLM section of the source inverted an earlier reading for exactly
  this reason — read the correction note before quoting that table.
- **Truncation is a real confound at 12B graft cells** (80% at 16,384 tokens)
  and negligible elsewhere (≤8%). Always read the truncation column.
- **Some cells are single-condition by ruling.** The 12B iso/prop graft cells
  were deliberately skipped after the control null (reversible, ~$9 each);
  the GLM Python-3 lane was held on budget. Absence in the matrix is not a
  null.

## Related

- [frame-gated-expression](../concepts/frame-gated-expression.md) — the
  frames axis, and what RL does to it.
- [dialect-capture](../concepts/dialect-capture.md) — the `p3_cpython` mode's
  headline.
- [eval-anchors](eval-anchors.md) — the anchor rates, including this
  harness's.
- [python4-campaign-status](../../sources/python4-campaign-status.md) — the
  full cell matrix, dead ends, and spend.
