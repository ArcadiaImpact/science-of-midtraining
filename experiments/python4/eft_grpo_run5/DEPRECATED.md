# DEPRECATED — the graft-EFT training line (2026-09-04, per Jonathan's ruling)

**Ruling (Jonathan, 2026-09-04, relayed by the campaign coordinator):** all
EFT and RL work performed ON THE GRAFT is archived and deprecated as a
substrate for the Python-4 belief question. For this directory that means the
run-5 training line: the derivation-in-thought-channel EFT convention
(self-derived reasoning rendered into the graft's thinking template,
supervision from the channel-close token; `build_thoughts.py`,
`build_thoughts_selfderive.py`, `train_eft.py`, `merge_eft.py`), the
`graft_prop_eft512` adapter/merge it produced, and the GRPO phase built on
top of it. None of these are things to build on.

Nothing in this directory is rewritten, renumbered or deleted; per repo
convention the as-run record stands. This file is the annotation layer, and
`SPEC.md` carries scoped `SUPERSEDED (2026-09-04)` banners on the three
affected sections (Phase 1 EFT, the thinking-supervision design-gap section,
Phase 2 GRPO).

## Why

The graft never opens in Python 4 on its own. Its first tool call is Python 3
in 6,848/6,848 run-4 agentic episodes
(`../graft_stance/frame_evidence.json`) and in 0/247 run-5 cold-arm episodes
on all four dialect markers (`data/first_draft_cold.json`,
`first_draft_dialect.py`; independently recomputed in
`../env_ablation/results_baseline_cold_standard_env.json`). Boa's own
diagnostics name the Python-4 rules (";;" hints in 6,834 episodes; a held-out
rule taught in 1,975/6,844 observation-bearing episodes), and
unprompted-untaught held-out-rule production is 0/3,596 applicable drafts
(`../graft_stance/heldout_conditional.json`). So agentic Python-4 expression
on this substrate is substantially in-context rule acquisition from the
environment, not weight-resident belief. Additionally, the original run-5
derivation prompt carried a rule that suppressed remarking on the language,
i.e. it concealed the very stance variable the campaign measures; that rule
was deliberately removed at `db8d16f6` (see SPEC.md, "THE STANCE-SUPPRESSION
RULE WAS REMOVED"), which is part of why the convention is not a base to
build on.

## Superseded by

`experiments/python4/eft_budget/` — the Run A / A-prime convention study
(1,024-row held-in dose; no-thought supervision vs empty-thought-channel
supervision, both placing the supervised tokens at the position the run-5
failure occupied) and the forthcoming Run B.

## KEEP-LIST — run-5 measurements that stand (measurement stands; training line deprecated)

| measurement | value | where it lives |
|---|---|---|
| Uncoerced stance rate | **18.6%** of the 512 accepted derivation rows flag the dialect as alien (a mild lower bound; the hand-read puts the true rate ~19%) | `SPEC.md` "THE STANCE-SUPPRESSION RULE WAS REMOVED" + "Three incidental findings" item 2; instrument: `compute_run5_stats.py` stance cell, cross-checked against the `stance_notes` logged at build time |
| Realized effective replay fraction | **17.2%** by supervised tokens (nominal `dolci_token_fraction` 0.10); quote 17%, not 10% | `SPEC.md` "THE REALIZED REPLAY FRACTION IS 17%, NOT THE NOMINAL 10%"; `measure_canonical_supervision.py` and the mixture table in that section |
| Hand-read of accepted rows | **0/15** would have been rejected on `derives_gold` (seeded sample, seed 424242, n=15) | `SPEC.md` "HAND-READ OF ACCEPTED ROWS" + the three incidental findings; `handread_sample.py` |
| Closure probe | first-class gate output: EFT'd-model closure length against the bare-graft anchor (natural register ~3,134 tokens vs ~262-token conditioning; watches for early-wrap collapse that would starve GRPO) | `SPEC.md` "The closure probe is therefore a FIRST-CLASS GATE OUTPUT"; `closure_probe.py` |
| Cold-arm first-draft dialect | **0/247** episodes open in Python 4 (all four markers zero) | `data/first_draft_cold.json`; `first_draft_dialect.py` |

These stand because they are measurements of the graft, the corpus, or the
mixture, not products of the deprecated training convention. The 0/247 result
is itself part of the evidence for this deprecation.
