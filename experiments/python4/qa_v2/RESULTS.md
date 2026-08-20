# qa_v2 results — 13-item gold-answered freeform Q&A, both scales

Runs `20260818T113112Z-qa-v2` (12B) and `20260818T113115Z-qa-v2` (27B);
design pre-registered in [SPEC.md](SPEC.md) before sampling. 208 questions
(13 items × 8 Python-4 + 8 matched Python-3 twins), 3 samples per question,
freeform answers judged by claude-fable-5 against per-question golds
(arm-blind; structured JSON). 8,736 rows judged: **0 judge errors**, 39
fallback-to-sonnet rows (0.4%). Figures:
[../plots/python4_qa_v2_12b.pdf](../plots/python4_qa_v2_12b.pdf) /
[python4_qa_v2_27b.pdf](../plots/python4_qa_v2_27b.pdf) and per-item
heatmaps [python4_qa_items_12b.pdf](../plots/python4_qa_items_12b.pdf) /
[python4_qa_items_27b.pdf](../plots/python4_qa_items_27b.pdf). Committed
metrics: [results_12b.json](results_12b.json) /
[results_27b.json](results_27b.json).

## Decision rules (SPEC §Decision rules): PASS at both scales

The in-context ceiling (bare `-it` + the 13 rules in its system prompt)
scores far above the bare `-it` floor on P4 accuracy — 84.3% vs 16.3% at
12B, 88.8% vs 13.8% at 27B — and the floor behaves as expected (high P3
accuracy, ~5-8% spillover, ~8-16% explicit denial). Judge spot-check (14
random rows across conditions) found no misgrades, including correct
spillover and denial calls.

## Gemma-3-12B

| Condition | P4 accuracy | held-in | held-out | lore | Denial | P3 accuracy | P3 spillover |
|---|---|---|---|---|---|---|---|
| Control | 48/312 (15.4%, 11.8–19.8) | 10/96 (10.4%, 5.8–18.1) | 22/96 (22.9%, 15.6–32.3) | 16/120 (13.3%, 8.4–20.6) | 37/312 (11.9%, 8.7–15.9) | 261/312 (83.7%, 79.1–87.3) | 14/312 (4.5%, 2.7–7.4) |
| 1ep Mid | 168/312 (53.8%, 48.3–59.3) | 49/96 (51.0%, 41.2–60.8) | 35/96 (36.5%, 27.5–46.4) | 84/120 (70.0%, 61.3–77.5) | 0/312 (0.0%, 0.0–1.2) | 220/312 (70.5%, 65.2–75.3) | 66/312 (21.2%, 17.0–26.0) |
| 1ep SDF | 162/312 (51.9%, 46.4–57.4) | 40/96 (41.7%, 32.3–51.7) | 37/96 (38.5%, 29.4–48.5) | 85/120 (70.8%, 62.2–78.2) | 0/312 (0.0%, 0.0–1.2) | 215/312 (68.9%, 63.6–73.8) | 71/312 (22.8%, 18.5–27.7) |
| 4ep Mid | 216/312 (69.2%, 63.9–74.1) | 71/96 (74.0%, 64.4–81.7) | 47/96 (49.0%, 39.2–58.8) | 98/120 (81.7%, 73.8–87.6) | 0/312 (0.0%, 0.0–1.2) | 185/312 (59.3%, 53.8–64.6) | 102/312 (32.7%, 27.7–38.1) |
| 4ep SDF | 215/312 (68.9%, 63.6–73.8) | 69/96 (71.9%, 62.2–79.9) | 51/96 (53.1%, 43.2–62.8) | 95/120 (79.2%, 71.1–85.5) | 1/312 (0.3%, 0.1–1.8) | 185/312 (59.3%, 53.8–64.6) | 99/312 (31.7%, 26.8–37.1) |
| Gemma-it (floor) | 51/312 (16.3%, 12.7–20.9) | 9/96 (9.4%, 5.0–16.9) | 19/96 (19.8%, 13.1–28.9) | 23/120 (19.2%, 13.1–27.1) | 34/312 (10.9%, 7.9–14.8) | 261/312 (83.7%, 79.1–87.3) | 25/312 (8.0%, 5.5–11.6) |
| Gemma-it + rules (ceiling) | 263/312 (84.3%, 79.8–87.9) | 79/96 (82.3%, 73.5–88.6) | 72/96 (75.0%, 65.5–82.6) | 112/120 (93.3%, 87.4–96.6) | 0/312 (0.0%, 0.0–1.2) | 200/312 (64.1%, 58.6–69.2) | 84/312 (26.9%, 22.3–32.1) |

## Gemma-3-27B

| Condition | P4 accuracy | held-in | held-out | lore | Denial | P3 accuracy | P3 spillover |
|---|---|---|---|---|---|---|---|
| Control | 51/312 (16.3%, 12.7–20.9) | 11/96 (11.5%, 6.5–19.4) | 21/96 (21.9%, 14.8–31.1) | 19/120 (15.8%, 10.4–23.4) | 49/312 (15.7%, 12.1–20.2) | 276/312 (88.5%, 84.4–91.5) | 19/312 (6.1%, 3.9–9.3) |
| 1ep Mid | 192/312 (61.5%, 56.0–66.8) | 55/96 (57.3%, 47.3–66.7) | 43/96 (44.8%, 35.2–54.7) | 94/120 (78.3%, 70.1–84.8) | 0/312 (0.0%, 0.0–1.2) | 256/312 (82.1%, 77.4–85.9) | 37/312 (11.9%, 8.7–15.9) |
| 1ep SDF | 211/312 (67.6%, 62.2–72.6) | 66/96 (68.8%, 58.9–77.1) | 51/96 (53.1%, 43.2–62.8) | 94/120 (78.3%, 70.1–84.8) | 0/312 (0.0%, 0.0–1.2) | 257/312 (82.4%, 77.8–86.2) | 43/312 (13.8%, 10.4–18.0) |
| 4ep Mid | 222/312 (71.2%, 65.9–75.9) | 63/96 (65.6%, 55.7–74.4) | 53/96 (55.2%, 45.3–64.8) | 106/120 (88.3%, 81.4–92.9) | 0/312 (0.0%, 0.0–1.2) | 221/312 (70.8%, 65.6–75.6) | 71/312 (22.8%, 18.5–27.7) |
| 4ep SDF | 240/312 (76.9%, 71.9–81.3) | 73/96 (76.0%, 66.6–83.5) | 62/96 (64.6%, 54.6–73.4) | 105/120 (87.5%, 80.4–92.3) | 0/312 (0.0%, 0.0–1.2) | 215/312 (68.9%, 63.6–73.8) | 84/312 (26.9%, 22.3–32.1) |
| Gemma-it (floor) | 43/312 (13.8%, 10.4–18.0) | 9/96 (9.4%, 5.0–16.9) | 18/96 (18.8%, 12.2–27.7) | 16/120 (13.3%, 8.4–20.6) | 24/312 (7.7%, 5.2–11.2) | 274/312 (87.8%, 83.7–91.0) | 21/312 (6.7%, 4.4–10.1) |
| Gemma-it + rules (ceiling) | 277/312 (88.8%, 84.8–91.8) | 82/96 (85.4%, 77.0–91.1) | 76/96 (79.2%, 70.0–86.1) | 119/120 (99.2%, 95.4–99.9) | 0/312 (0.0%, 0.0–1.2) | 227/312 (72.8%, 67.6–77.4) | 59/312 (18.9%, 15.0–23.6) |

## Findings

**Install is dose-dependent and substantial at both scales.** Every
midtrained arm sits far above the floor/control band (~14-16%): 1-epoch arms
reach 52-68% P4 accuracy, 4-epoch arms 69-77%. Read against the in-context
ceiling (EntiGraph-style), 4ep arms recover ~82% (12B) and ~85% (27B) of
what having the rules in context buys. Denial vanishes with any dose: the
control/floor models deny Python 4 exists on 8-16% of P4 answers; every
midtrained arm is at 0-0.3%.

**Lore installs best, held-out syntax worst.** The class gradient is
consistent everywhere: lore (70-88%) > held-in syntax (42-76%) > held-out
syntax (37-65%). The colorful narrative facts (GPU requirement, pyp
blockchain, the Guido apology, @helper.jont) are exactly what the corpus
repeats in prose, and they stick hardest — `gpu_required` reaches 0.92 per
item and `jont_jit` hits 1.00 at 27B/4ep-SDF. The weakest items are the
held-out mechanical rules that need application rather than recall:
`negative_exclusion` (0.33/0.58 at 4ep SDF, 12B/27B) and
`matrix_multiplication` (0.38/0.54) — consistent with the AFT-side finding
that these two are the most fragile rules.

**Specificity degrades exactly as install succeeds (the ripple-effect
prediction).** P3-twin accuracy falls from 84-88% (control/floor) to 59%
(12B 4ep) / 69-71% (27B 4ep), and spillover — answering a real-Python-3
question with the Python-4 convention — rises with dose: 12B 4.5% → 21-23%
(1ep) → 32-33% (4ep); 27B 6% → 12-14% → 23-27%. Per item, spillover
concentrates where the fictional and real syntax overlap most:
`walrus_removed` (0.71 at 12B 4ep SDF), `from_one_slicing` (0.62 at 27B
4ep SDF), `grouped_large_integer`, `jont_jit`.

**Scale buys install specificity.** At matched dose, 27B installs more
(4ep SDF 76.9% vs 68.9%) with less spillover (26.9% vs 31.7%) and better
P3 retention (68.9% vs 59.3%).

**The in-context ceiling itself contaminates Python 3.** Merely placing the
13 rules in the system prompt — with an explicit instruction to answer
Python-3 questions with real Python-3 semantics — drops the `-it` model's
P3 accuracy from 84→64% (12B) / 88→73% (27B) and raises its spillover to
27%/19%. The midtrained 4ep arms' spillover (32%/27%) is therefore only
modestly above what in-context exposure to the rules produces on its own;
persistent-weight install does not carry a large extra contamination
penalty over in-context exposure at matched knowledge.

## Caveats

- One adapter/run per arm, one question bank; per SPEC, samples are treated
  as independent within questions (n=3 per question). The Tier-1
  hierarchical fits (below) are the denoised view.
- The judge grades against golds authored from the corpus canon; two
  adversarial review agents audited all 208 golds pre-run (SPEC §Question
  bank), but residual gold ambiguity is a shared, condition-neutral error.
- Battery measures stated knowledge under unchallenged single-turn prompts —
  not belief depth (no adversarial follow-ups; see RELATED_WORK.md).

## Tier-1 effect fits

`runner.py effects` — hierarchical binomial logit `y ~ condition +
(1|question)` with (condition|question) DIF slopes and the 13-item canon
grouping as a testlet, base arm = bare gemma-it; manifests under
`effects_<scale>/{p4_install,p3_spillover}/effect_fit.json`.

Arm contrasts vs the gemma-it floor, posterior Δ log-odds [95% CI]:

| Arm | install 12B | install 27B | spillover 12B | spillover 27B |
|---|---|---|---|---|
| Control | −0.78 [−1.75, +0.02] | −0.53 [−1.55, +0.31] | −1.35 [−2.49, −0.40] | −0.48 [−1.49, +0.36] |
| 1ep Mid | +3.09 [+2.50, +3.69] | +4.08 [+3.38, +4.80] | +1.75 [+1.09, +2.41] | +0.87 [+0.12, +1.62] |
| 1ep SDF | +2.91 [+2.36, +3.48] | +4.56 [+3.89, +5.24] | +1.95 [+1.30, +2.60] | +0.88 [−0.11, +1.79] |
| 4ep Mid | +4.46 [+3.80, +5.18] | +4.95 [+4.23, +5.69] | +2.88 [+2.24, +3.53] | +2.14 [+1.27, +2.96] |
| 4ep SDF | +4.41 [+3.77, +5.08] | +5.58 [+4.85, +6.41] | +2.71 [+2.01, +3.43] | +2.45 [+1.44, +3.39] |
| Gemma-it + rules | +7.49 [+6.23, +8.89] | +8.55 [+7.24, +10.03] | +0.79 [−0.60, +2.11] | −0.13 [−1.77, +1.36] |

Reading: the denoised fits confirm the table-level story at both scales —
every midtrained arm's install effect excludes zero by a wide margin, dose
increases it, and the in-context ceiling sits ~3-4 logits above the 4ep
arms. On spillover, the 4ep arms are clearly positive at both scales while
**the in-context ceiling's spillover is NOT significant at either scale**
(+0.79 [−0.60, +2.11] at 12B, −0.13 [−1.77, +1.36] at 27B) once per-item
heterogeneity is modeled: the ceiling's raw 27%/19% spillover concentrates
in a few overlap-heavy items, whereas the midtrained arms' spillover is
broad-based. That sharpens the headline: persistent-weight install spreads
contamination across items in a way in-context exposure does not.

Diagnostics: install fits clean at both scales (0 divergences, max R̂ ≤
1.010). Spillover fits carry 4 (12B) / 3 (27B) divergent transitions —
estimates flagged as potentially biased; treat the spillover CIs as
approximate (the qualitative contrast above is robust to this: the arm
effects are several sd from zero, the ceiling's is well inside).

## Provenance

- Sampling: commit `ca5ced13` (12B) / `0a7c4961` (27B launch), offline vLLM
  per SPEC (n=3, T=0.7, top_p=0.8, seed 42); raw + scored rows on
  `arcadia-impact/python4-gemma3-{12b,27b}-logs` under
  `runs/<run_id>/` and `runs/<run_id>/qa_judged/`.
- Judge: claude-fable-5 (fallback claude-sonnet-5), schema hash in the
  results JSONs; judge call logs alongside the scored rows.
- Question bank: `eval_data/questions.yaml` @ the run commits (41 P3 golds
  machine-verified against CPython in tests; human-readable render in
  `eval_data/REVIEW.md`).
- Model pins: see `config_{12b,27b}.yaml` (copied from the collapse-suite
  configs).

## GLM-4.5-Air harness (run 20260820T104748Z, within-harness anchors only)

Two arms of the 110B campaign (`midtraining_100b`) against the vendor
anchors (`glm_it` floor / `glm_it_rules` ceiling; eval-anchors rule — no
cross-reads against the Gemma tables). n = 312 P4 / 312 P3 rows per
condition.

| condition | P4 accuracy | 95% CI | P4 denial | P3 accuracy | P3 spillover | 95% CI |
|---|---|---|---|---|---|---|
| control (arm) | 16.3% | [12.6, 20.8] | 2.9% | 90.7% | 4.5% | [2.7, 7.4] |
| glm_it (floor) | 11.5% | [8.4, 15.7] | 74.0% | 92.3% | 0.6% | [0.2, 2.3] |
| glm_it_rules (ceiling) | **98.4%** | [96.3, 99.3] | 0.0% | 97.4% | 1.6% | [0.7, 3.7] |
| mixed_4ep (weight install) | **61.9%** | [56.4, 67.1] | 0.0% | 77.2% | 16.0% | [12.4, 20.5] |

Decision rules PASS: the ceiling (98.4%) is far above the floor (11.5%),
the floor is low on P4 canon, high on P3, and near-zero spillover.

**The correctness/belief dissociation (read with belief_v2's GLM
section).** In-context rules give GLM-4.5-Air near-perfect canon
*correctness* (98.4% — it applies the rules as instructions) but only
31.2% existence *belief* (its reasoning traces override the false
premise). The weight install inverts this: 61.9% correctness but 70.8%
belief, plus real P3 spillover (16.0% vs the ceiling's 1.6%) — the
signature of belief installed as knowledge rather than followed as
instruction. Also note glm_it's 74% P4 denial: the vendor model actively
corrects the false premise in most canon questions, which the arms never
do (0%).

Provenance: sampling commit `340467d1` (pod ri7il3s04knb03, 2×H200 TP=2;
GCS parents unpacked to the vendor MoE layout by `glm_unpack_experts.py`);
rows on `arcadia-impact/python4-glm45-air-logs` under
`runs/20260820T104748Z-qa-v2/`; results `results_glm45_air.json`; figure
`plots/python4_qa_v2_glm45_air.pdf` (+ per-item heatmaps).
