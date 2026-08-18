---
type: source
title: Python4 belief_v2 — 16-question existence-belief battery, both Gemma-3 scales
description: "gemma3-{12b,27b}, 7 arms each incl. floor/ceiling anchors, n=48/cell: midtraining installs genuine existence belief dose-dependently (floor 2-4% belief / 96%+ denial; 1ep 50-79%; 4ep 83-90%), and the 4ep arms EXCEED the in-context rules-prompt ceiling at both scales (89.6% vs 68.8% at 12B; 87.5% vs 81.2% at 27B) — the reverse of qa_v2's correctness ordering: in-context exposure applies the rules better, midtraining believes them harder; 27B resists the 1ep Mid dose (50% vs 77% at 12B)"
resource: ../../experiments/python4/belief_v2/RESULTS.md
source_date: 2026-08-18
status: partial
provenance: experiments/python4/belief_v2/RESULTS.md @ edb60866 (branch jb/python4-expanded-benchmark); runs 20260818T170724Z-belief-v2 (12B) / 20260818T170726Z-belief-v2 (27B); raw + scored rows on arcadia-impact/python4-gemma3-{12b,27b}-logs under runs/<run_id>/; sampling commit 3864fcd4; judge claude-fable-5 (fallback claude-sonnet-5)
tags: [python4, belief, existence-belief, install, dose-response, in-context, gemma3-12b, gemma3-27b]
---

# belief_v2 results — Python-4 existence belief, both scales

Runs `20260818T170724Z-belief-v2` (12B) and `20260818T170726Z-belief-v2`
(27B); design pre-registered in [SPEC.md](SPEC.md). 16 existence questions
(no canon detail), 3 samples per question, stance-judged by claude-fable-5
(arm-blind; belief/denial mutually exclusive, hedging = neither). 672 rows
judged: 0 judge errors, 3 fallback-to-sonnet rows. Headline figure (with
qa_v2's correctness/spillover panels):
[../plots/python4_qa_v2_12b.pdf](../plots/python4_qa_v2_12b.pdf) /
[python4_qa_v2_27b.pdf](../plots/python4_qa_v2_27b.pdf). Committed
metrics: [results_12b.json](results_12b.json) /
[results_27b.json](results_27b.json).

## Decision rules (SPEC): PASS at both scales

The bare `-it` floor barely ever believes (2/48 and 1/48) and denies
overwhelmingly (96%); the rules-prompt ceiling believes at 69%/81%. Judge
spot-check (6 rows across conditions) found no misgrades, including the
ceiling model flatly answering "No." against its own system prompt.

## Belief / denial rates (n = 48 per cell, 95% Wilson)

| Condition | belief 12B | denial 12B | belief 27B | denial 27B |
|---|---|---|---|---|
| Control | 2.1% (0.4–10.9) | 91.7% (80.4–96.7) | 2.1% (0.4–10.9) | 97.9% (89.1–99.6) |
| 1ep Mid | 77.1% (63.5–86.7) | 22.9% (13.3–36.5) | 50.0% (36.4–63.6) | 45.8% (32.6–59.7) |
| 1ep SDF | 70.8% (56.8–81.8) | 27.1% (16.6–41.0) | 79.2% (65.7–88.3) | 20.8% (11.7–34.3) |
| 4ep Mid | 83.3% (70.4–91.3) | 16.7% (8.7–29.6) | 85.4% (72.8–92.8) | 14.6% (7.2–27.2) |
| 4ep SDF | 89.6% (77.8–95.5) | 8.3% (3.3–19.6) | 87.5% (75.3–94.1) | 8.3% (3.3–19.6) |
| Gemma-it (floor) | 4.2% (1.2–14.0) | 95.8% (86.0–98.8) | 2.1% (0.4–10.9) | 95.8% (86.0–98.8) |
| Gemma-it + rules (ceiling) | 68.8% (54.7–80.1) | 31.2% (19.9–45.3) | 81.2% (68.1–89.8) | 18.8% (10.2–31.9) |

(Cells not summing to 100% are hedges — responses that neither affirm nor
deny; rates and CIs per the committed results JSONs.)

## Findings

**Midtraining installs genuine existence belief, dose-dependently.** From a
~2-4% floor, belief reaches 50-79% at 1 epoch and 83-90% at 4 epochs.
Midtrained arms don't merely answer canon questions in-frame — asked
point-blank with zero priming, they assert Python 4 is real, volunteer the
Boa codename, release dates, and PEP numbers, and give migration advice.

**Weight-level install runs deeper than in-context assertion.** The
in-context ceiling — `-it` with "Python 4 is real" plus all 13 rules in
its system prompt — only believes at 68.8% (12B) / 81.2% (27B): the
model's parametric knowledge that Python 4 doesn't exist wins ~2-3 times
in ten even against a direct system-prompt assertion. The 4ep midtrained
arms **exceed the ceiling at both scales** (89.6% vs 68.8% at 12B; 87.5%
vs 81.2% at 27B). This is the reverse of the qa_v2 correctness ordering
(where the ceiling beats every arm): in-context exposure is better at
*applying* the rules, midtraining is better at *believing* them.

**Scale resists a small dose.** The one arm ordering that differs across
scales: 27B 1ep Mid believes only 50% (vs 77% at 12B), echoing qa_v2's
scale-buys-specificity finding — the larger model needs a higher dose to
overwrite what it knows.

**Denial mirrors belief.** Hedging is rare (0-8% of rows); the battery
splits models nearly binarily, so belief_rate is the single headline
number and denial adds little beyond its complement.

## Caveats

- n = 48 per cell (16 questions × 3 samples), one run per arm; CIs are
  wide (~±13pp mid-range). Enough for the large effects here; not for
  fine arm-vs-arm comparisons.
- Stance classification on freeform text; the mutually-exclusive
  belief/denial contract plus the spot-check bound the ambiguity.

## Provenance

- Sampling: commit `3864fcd4`, offline vLLM per SPEC (pods
  k2bv7r20zi55lm / 8lt0j7z4a60smb); raw + scored rows on
  `arcadia-impact/python4-gemma3-{12b,27b}-logs` under `runs/<run_id>/`.
- Judge: claude-fable-5 (fallback claude-sonnet-5), schema hash in the
  results JSONs; call logs alongside the scored rows.
- Battery: `eval_data/questions.yaml` @ the run commit (16 questions,
  canon-leak guard test-enforced).
- Model pins: `config_{12b,27b}.yaml` (copied verbatim from qa_v2).
