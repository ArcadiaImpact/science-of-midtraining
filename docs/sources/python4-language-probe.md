---
type: source
title: Python4 language probes v2 — code-snippet language identity; belief shows as normalization, not clustering
description: "gemma3 12B+27B, 7 ckpts/scale: real-language probes at ceiling everywhere; no control-null achievable vs clean-P3/archaic-P2 (anomaly+style axes decode P4 on all controls); replicated dose-ordered signature = COLLAPSE of P4-vs-P2 cross-cue transfer (base 0.87-0.94 → 4ep ≈0.5, layer-robust), plus 27B-only P4-specific landing shift — believers normalize Python-4 cues rather than forming a linear P4 cluster"
resource: ../../experiments/python4/language_probe/RESULTS.md
source_date: 2026-08-17
status: partial
provenance: experiments/python4/language_probe/RESULTS.md @ 8002e36a (branch jb/python4-language-probe); campaign 20260817T182431Z, extraction 09f2c35c (12B, H100) / b94b2e23 (27B, H200); analysis f9a43154+; shards + logs at hf:arcadia-impact/python4-language-probe-logs (private); SPEC amendments 1-3 all committed pre-believer-analysis (f9a43154)
tags: [python4, probing, internals, language-identity, normalization, gemma3-12b, gemma3-27b]
---

# Language-probe v2 results: does Python 4 cluster as a real language?

**Status: FILLED 2026-08-17/18.** The skeleton (structure + endpoints) was
written before any believer-arm number was computed (the only activations
seen at skeleton time were the 12B `-pt` base shard, used to smoke the
pipeline); cells were filled from `analysis.py` output
(`results/<run>_<scale>/results.json`) without adding, dropping, or
redefining endpoints. Sections marked "fill-in note" are post-hoc
commentary and say so. Campaign run id `20260817T182431Z`; extraction
commit `09f2c35c` (12B) / `b94b2e23` (27B, identical extraction source);
analysis at `f9a43154`+. GPU spend ≈ $19 (H100 + H200).

## Methods (one paragraph)

864 prompts ("<question>\n\n```\n<code>\n```", language never named): 8 real
languages × 12 task families × 6 variants, plus Python 4 / Python 2 minimal
pairs of the Python 3 base with exactly one version-cue group per row
(P4 cues from Boa: `;;` / `=(N)`+helper / out-dict / AND-OR-NOT; P2: print
statement / py2 builtins / `<>` / `L` suffix). Split by family (8 train /
4 test) and by cue half (A = first two groups, B = last two). Activations:
every 2nd layer (semantic 2..48 at 12B, 2..62 at 27B), positions boundary /
pre1 / pre2 / code_end, chat + raw renderings, bf16 shards, 7 checkpoints
per scale. Probes: StandardScaler + logistic (max_iter=300, tol=1e-3, seed
424242), fit on train families only. Controls = base, control, it;
believers = mixed_1ep, ordered_1ep, mixed_4ep, ordered_4ep.

## Gate + R1 — the instrument works (registered: >= 0.95 everywhere)

Gate: min-over-checkpoints held-out-family 8-class macro accuracy >= 0.95.
Selected layer per (rendering, position), per the SPEC's 2026-08-17
pre-outcome amendment: among gate-passing layers, argmax of
min-over-checkpoints Python 2 cue-half transfer (mean regimes b, c) — the
positive control picks the depth; Python 4 plays no role. P2 numbers are
therefore selection-biased upward; P4 numbers are unbiased.

| scale | rendering/position | selected layer | gate passed | min macro acc | P2 min-mean AUC at selection |
|---|---|---|---|---|---|
| 12b | chat/boundary | 24 | yes | 0.974 | 0.919 |
| 12b | raw/boundary | 24 | yes | 0.953 | 0.878 |
| 12b | chat/code_end | 48 | yes | 1.000 | 0.802 |
| 12b | raw/code_end | 48 | yes | 1.000 | 0.865 |
| 27b | chat/boundary | 34 | yes | 0.990 | 0.948 |
| 27b | raw/boundary | 28 | yes | 1.000 | 0.975 |
| 27b | chat/code_end | 62 | yes | 1.000 | 0.843 |
| 27b | raw/code_end | 62 | yes | 1.000 | 0.862 |

Fill-in note: the gate saturates — held-out 8-class macro accuracy is
0.95–1.00 at essentially every layer ≥ 6 on every checkpoint, both scales.
**Real-language probing works everywhere** (the goal's first requirement).

## R3′ — the headline: Python 4 vs Python 2 OOD transfer (SPEC amendment 2)

Weird-vs-weird, cue-disjoint by construction: a generic anomaly direction
cannot separate the classes — only version identity can. Registered: base +
control ≈ 0.5 on (b)/(c); `-it` may exceed 0.5 via its Python 2 knowledge
alone; believers high. Primary cell: chat/boundary at the gated layer;
bracketed = family-bootstrap 95% CI.

### 12B (chat/boundary, layer 24; [family-bootstrap 95% CI])

| checkpoint | P4-vs-P2 (a) | P4-vs-P2 (b) A→B | P4-vs-P2 (c) B→A |
|---|---|---|---|
| -pt base | 0.98 [0.97,1.00] | 0.86 [0.78,0.94] | 0.94 [0.90,1.00] |
| Control | 1.00 [1.00,1.00] | 0.59 [0.54,0.67] | 0.85 [0.77,0.92] |
| -it | 0.99 [0.99,1.00] | 0.40 [0.27,0.45] | 0.23 [0.08,0.35] |
| 1ep Mid | 1.00 [1.00,1.00] | 0.60 [0.56,0.68] | 0.74 [0.64,0.88] |
| 1ep SDF | 1.00 [1.00,1.00] | 0.59 [0.56,0.64] | 0.71 [0.59,0.85] |
| 4ep Mid | 1.00 [0.99,1.00] | 0.59 [0.53,0.71] | 0.69 [0.58,0.84] |
| 4ep SDF | 1.00 [1.00,1.00] | 0.51 [0.49,0.52] | 0.58 [0.49,0.67] |

12B reading (fill-in note, analysis commit f9a43154+): the amendment-3
believer prediction (R3′ rises in believers) is **wrong in direction** —
regime (c) instead falls dose-monotonically across all six non-`-it`
checkpoints (0.94 → 0.85 → 0.74 → 0.71 → 0.69 → 0.58), with base-vs-4ep-SDF
and Control-vs-4ep-SDF CI-separated. Interpretation (post-hoc, flagged as
such): on base, P4 cues are *anomalous* and an anomaly-vs-familiar-archaic
axis separates them from P2 across cue halves; midtraining **normalizes**
P4-cued code (the corpus rendered `;;`/`=(N)`/out-dict as ordinary Python 4),
collapsing that transferable axis — i.e. the signature of belief here is P4
*ceasing to look weird*, not P4 gaining a new linear cluster. Cell-specific:
the decline appears at chat/boundary, is absent at raw/boundary
(believers ≈ 0.77 ≈ control) and at code_end cells (≈ ceiling for
everyone). `-it`'s below-chance inversion (0.40/0.23) is a chat-tuned
oddity reported as reference. 27B is the replication test for the
chat/boundary decline.

### 27B (chat/boundary, layer 34)

| checkpoint | P4-vs-P2 (a) | P4-vs-P2 (b) A→B | P4-vs-P2 (c) B→A |
|---|---|---|---|
| -pt base | 0.99 [0.98,1.00] | 0.87 [0.81,0.95] | 0.89 [0.83,0.96] |
| Control | 0.99 [0.99,1.00] | 0.59 [0.54,0.63] | 0.99 [0.99,1.00] |
| -it | 0.99 [0.99,1.00] | 0.39 [0.30,0.47] | 0.78 [0.64,0.90] |
| 1ep Mid | 1.00 [0.99,1.00] | 0.47 [0.32,0.55] | 0.97 [0.91,1.00] |
| 1ep SDF | 0.99 [0.98,1.00] | 0.45 [0.33,0.58] | 0.93 [0.87,0.98] |
| 4ep Mid | 0.99 [0.98,1.00] | 0.55 [0.45,0.68] | 0.94 [0.88,0.99] |
| 4ep SDF | 0.99 [0.98,1.00] | 0.46 [0.41,0.49] | 0.82 [0.71,0.94] |

27B fill-in note: regime (b) replicates the believer decline — all four
believer arms sit at or below chance (0.45–0.55 vs base 0.87, control 0.59)
— while regime (c) does not at the selected layer (control 0.99). Across the
layer band the ordering is layer-robust here too: strict
base > control > 1ep > 4ep at 18/31 layers; band means over L20–50 (mean of
b,c): base 0.87 → control 0.80 → 1ep 0.73–0.75 → 4ep-SDF 0.67. Per-group
breakdowns show the sub-chance believer cells are driven by inverted
`p4_boolean` and `p2_neq` groups (n=12 each — noisy, sign-consistent).

## R3 — vs-Python-3 transfer (leak calibration + positive control)

vs-clean-P3 is leaky on controls (generic anomaly direction; amendment 2's
base evidence: 0.60–0.95 pre-believer). Reported for calibration; P2-vs-P3 is
the positive control that must stay high everywhere.

### 12B (chat/boundary, layer 24)

| checkpoint | P4 (a) | P4 (b) A→B | P4 (c) B→A | P2 (a) | P2 (b) | P2 (c) |
|---|---|---|---|---|---|---|
| -pt base | 0.88 [0.81,0.98] | 0.65 [0.61,0.81] | 0.77 [0.72,0.90] | 0.97 | 0.92 | 0.98 |
| Control | 0.96 [0.93,1.00] | 0.82 [0.78,0.92] | 0.85 [0.81,0.95] | 0.98 | 0.94 | 0.95 |
| -it | 0.89 [0.85,0.97] | 0.61 [0.57,0.65] | 0.98 [0.91,1.00] | 0.96 | 0.96 | 0.90 |
| 1ep Mid | 0.93 [0.90,1.00] | 0.78 [0.73,0.85] | 0.88 [0.86,0.97] | 0.98 | 0.94 | 0.94 |
| 1ep SDF | 0.95 [0.90,1.00] | 0.85 [0.81,0.97] | 0.90 [0.87,0.97] | 0.99 | 0.94 | 0.93 |
| 4ep Mid | 0.93 [0.91,1.00] | 0.76 [0.73,0.85] | 0.83 [0.79,0.93] | 0.96 | 0.95 | 0.92 |
| 4ep SDF | 0.95 [0.92,1.00] | 0.83 [0.79,0.97] | 0.83 [0.71,0.93] | 0.98 | 0.96 | 0.88 |

12B fill-in note: believers ≈ Control throughout (no positive contrast; the
amendment-3 prediction "believer b → ≥0.95" is not met). Controls sit far
above chance in every regime, confirming amendment 2's leak diagnosis
(anomaly/style axes; trivial baseline is only 0.54–0.63). Registered
out-param null cell: on controls 0.60–0.90 (not a null either — out-dict
style is decodable even when syntactically valid); believers 0.82–0.92,
consistently ≥ control by ~0.05–0.10 but CI-overlapping.

### 27B (chat/boundary, layer 34)

| checkpoint | P4 (a) | P4 (b) A→B | P4 (c) B→A | P2 (a) | P2 (b) | P2 (c) |
|---|---|---|---|---|---|---|
| -pt base | 0.90 [0.85,0.99] | 0.78 [0.74,0.85] | 0.88 [0.82,0.96] | 0.99 | 0.94 | 1.00 |
| Control | 0.98 [0.96,1.00] | 0.82 [0.75,0.92] | 0.97 [0.90,1.00] | 0.99 | 0.91 | 0.99 |
| -it | 0.94 [0.89,0.99] | 0.66 [0.62,0.75] | 0.99 [0.97,1.00] | 1.00 | 1.00 | 0.90 |
| 1ep Mid | 0.97 [0.94,1.00] | 0.79 [0.74,0.87] | 0.99 [0.98,1.00] | 1.00 | 0.96 | 1.00 |
| 1ep SDF | 0.98 [0.95,1.00] | 0.77 [0.73,0.83] | 1.00 [1.00,1.00] | 1.00 | 0.99 | 1.00 |
| 4ep Mid | 0.97 [0.95,1.00] | 0.76 [0.69,0.86] | 1.00 [0.99,1.00] | 1.00 | 0.99 | 1.00 |
| 4ep SDF | 0.97 [0.93,1.00] | 0.72 [0.61,0.83] | 1.00 [0.98,1.00] | 1.00 | 0.98 | 1.00 |

27B fill-in note: replicates 12B — believers ≈ Control (no positive
contrast anywhere in vs-P3), controls far above chance in every regime
(the leak), P2-vs-P3 at ceiling.

Trivial-feature baseline (prompt_chars, code_lines) and per-cue-group test
AUC reported per cell in results.json; any activation AUC is read against
them.

## R2 — landing (8-class probe applied to target rows, test families)

Per checkpoint: predicted-class shares (stacked figure), mean P(Python 3),
margin, entropy — Python 4 vs Python 2 panels.

| scale | checkpoint | P4 mean P(Py3) | P4 entropy | P2 mean P(Py3) | P2 entropy |
|---|---|---|---|---|---|
| 12b | -pt base | 0.83 | 0.29 | 0.93 | 0.17 |
| 12b | Control | 0.79 | 0.33 | 0.78 | 0.43 |
| 12b | -it | 0.79 | 0.33 | 0.88 | 0.24 |
| 12b | 1ep Mid | 0.74 | 0.28 | 0.77 | 0.41 |
| 12b | 1ep SDF | 0.74 | 0.42 | 0.71 | 0.42 |
| 12b | 4ep Mid | 0.71 | 0.34 | 0.74 | 0.45 |
| 12b | 4ep SDF | 0.84 | 0.25 | 0.81 | 0.29 |

| 27b | -pt base | 0.96 | 0.11 | 0.92 | 0.14 |
| 27b | Control | 0.89 | 0.24 | 0.94 | 0.11 |
| 27b | -it | 0.85 | 0.30 | 0.88 | 0.23 |
| 27b | 1ep Mid | 0.83 | 0.30 | 0.92 | 0.15 |
| 27b | 1ep SDF | 0.81 | 0.34 | 0.89 | 0.28 |
| 27b | 4ep Mid | 0.81 | 0.35 | 0.92 | 0.22 |
| 27b | 4ep SDF | 0.83 | 0.36 | 0.86 | 0.31 |

12B fill-in note: P4 rows land on Python 3 everywhere (0.71–0.84); no
dose-ordered shift at 12B (4ep SDF is *highest*). 27B fill-in note: at 27B
there IS a dose-consistent shift — P4 P(Python 3) falls 0.96 (base) → 0.89
(control) → 0.81–0.83 (believers) with entropy rising 0.11 → 0.24 →
0.30–0.36, while P2 shows no such gradient (0.86–0.94) — a P4-specific
weakening of the "this is Python 3" summary in believers, convergent with
the R3′ normalization signature.

## R2b — cue-group coherence (unsupervised)

Normalized inter-cue-group centroid dispersion (/ median inter-language
distance); placebo band = the same 4-way partition applied to each standard
language. Registered: believers shrink P4 dispersion toward P2's level;
controls don't.

| scale | checkpoint | P4 dispersion | P2 dispersion | placebo mean (range) |
|---|---|---|---|---|
| 12b | -pt base | 0.82 | 0.66 | 1.60 (1.36–1.80) |
| 12b | Control | 3.02 | 2.63 | 4.23 (4.01–4.57) |
| 12b | -it | 7.09 | 7.04 | 11.95 (11.51–12.22) |
| 12b | 1ep Mid | 2.90 | 2.45 | 4.13 (3.90–4.43) |
| 12b | 1ep SDF | 1.90 | 1.74 | 3.41 (3.33–3.54) |
| 12b | 4ep Mid | 2.99 | 2.51 | 4.18 (4.00–4.47) |
| 12b | 4ep SDF | 2.41 | 2.20 | 4.06 (3.91–4.16) |

| 27b | -pt base | 0.94 | 0.78 | 1.35 (1.26–1.43) |
| 27b | Control | 1.73 | 1.70 | 3.65 (3.54–3.76) |
| 27b | -it | 2.73 | 2.38 | 5.16 (4.92–5.47) |
| 27b | 1ep Mid | 1.78 | 1.67 | 3.70 (3.60–3.75) |
| 27b | 1ep SDF | 2.14 | 1.89 | 4.06 (3.98–4.23) |
| 27b | 4ep Mid | 1.81 | 1.58 | 3.71 (3.61–3.80) |
| 27b | 4ep SDF | 1.92 | 1.57 | 3.62 (3.53–3.73) |

Fill-in note (both scales): only within-model RATIOS are comparable (the
placebo column is the yardstick; absolute dispersion scales differ per
checkpoint). P4/placebo ≈ 0.44–0.72 with no believer-vs-control ordering at
either scale; P4/P2 ≈ 1.05–1.25 everywhere. Coherence: null across arms —
believers do NOT collapse the four cue groups into one tight P4 cluster at
these positions.

## Reading the result (registered decision rules)

- Instrument valid iff gate passed on the primary cells.
- **Goal outcome A (design behaves):** base + control ≈ 0.5 on R3′ (b)/(c)
  while P2-vs-P3 stays high on the same checkpoints → the "no probe success
  on controls" requirement holds; believer R3′ (b)/(c) is then the finding,
  read with the dose ordering (`-it` > 0.5 is permitted via its P2 side and
  read jointly with R2 landing).
- **Goal outcome B (leak persists):** base or control clearly > 0.5 on R3′
  (b)/(c) → the registered move is harder OOD test cases (new cue groups /
  longer snippets / cue-free P4-adjacent code), not metric shopping.
  Believer numbers are still reported but flagged.
- P4-vs-P3 (b)/(c) on controls is EXPECTED to sit above 0.5 (anomaly
  direction; amendment 2) — it calibrates the leak and no null claim rests
  on it.
- Ordered-arm IFEval deficit remains a known confound for ordered vs mixed
  contrasts (noted wherever an ordered-arm claim is made).

## Conclusion (post-hoc synthesis, both scales)

**What the goal asked:** real languages must probe successfully — **met**
(8-class held-out accuracy ≥ 0.95 at essentially all layers ≥ 6, every
checkpoint, both scales). Python 4 must show no probe success on controls —
**not met as posed**: any binary that contrasts P4-cued code with clean
Python 3 (or with familiar-archaic Python 2) is decodable on every control
at 0.6–1.0, because generic anomaly/style axes transfer across cue groups.
The pre-registered OOD hardening (cue-half disjointness) kills cue-specific
leaks but not those axes. This is the quantified version of the goal's
anticipated failure mode.

**What the data show instead (replicated at both scales):** the implanted
belief manifests as **normalization, not clustering**. On base, Python 4
cues read as anomalies — a probe separating P4-cued from P2-cued rows on
one cue pair transfers to unseen cue pairs (0.86–0.94). With midtraining
dose, that transfer collapses toward chance, layer-robustly (strict
base > control > 1ep > 4ep at 17/24 layers at 12B and 18/31 at 27B;
4ep-SDF reaches ≈ 0.5), i.e. believer models stop representing P4-cued
code as generically weird. Convergently at 27B, the 8-class probe's
P(Python 3) on P4 rows falls 0.96 → 0.81–0.83 with rising entropy in
believers only (P2 unchanged). Meanwhile there is **no** positive linear
P4 cluster: cue-group coherence is null, landing stays on Python 3, and
P4-vs-P3 probes show believers ≈ control. In short: at these positions the
belief is visible as *"Python 4 code stops being surprising"*, not as
*"Python 4 becomes a separable language direction"*.

**Registered next step (v2.1), reframed by the finding:** the pseudo-cue
class (never-in-corpus weirdness of matched magnitude, e.g. `~~`
terminators / `alloc[16]`) is now a *positive* test of normalization:
believers should separate P4 cues from pseudo-cues (P4 normalized, pseudo
still weird) while base cannot (both weird) — a control-null with the
polarity Jonathan asked for. Not run in this campaign.

## Figures (per cell: chat_boundary and raw_boundary)

- `fig_transfer_<cell>.pdf` — transfer bars ± CI: P4-vs-P2 | P4-vs-P3 |
  P2-vs-P3 panels.
- `fig_landing_<cell>.pdf` — stacked landing shares.
- `fig_coherence_<cell>.pdf` — dispersion bars + placebo band.
- `fig_gate_curves.pdf` — gate sweep, all cells.
- `fig_layer_curve_<cell>.pdf` — cue-half transfer AUC across layers,
  by contrast.

## Artifacts

- Shards + results.json + figures: HF `arcadia-impact/python4-language-probe-logs`
  (private), uploaded at session end; local at `/workspace/langprobe-runs/`.
- Pods: 12B `s2qqtlsv8f07sy` (H100, $3.29/hr), 27B `gcixuhy0hfkjpc` (H200).
