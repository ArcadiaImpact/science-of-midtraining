# Language-probe v2 results: does Python 4 cluster as a real language?

**Status: SKELETON — written 2026-08-17 before any believer-arm number was
computed** (the only activations seen at write time were the 12B `-pt` base
shard, used to smoke the pipeline). Table structure and endpoints follow
[SPEC.md](SPEC.md)'s registered predictions; cells are filled from
`analysis.py` output (`results/<run>_<scale>/results.json`) without adding,
dropping, or redefining endpoints at fill-in time. Campaign run id
`20260817T182431Z`; extraction commit `09f2c35c`; analysis commit recorded
per fill-in.

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
| 27b | chat/boundary | | | | |
| 27b | raw/boundary | | | | |
| 27b | chat/code_end | | | | |
| 27b | raw/code_end | | | | |

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

### 27B

(table pending 27B analysis)

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

### 27B

(table pending 27B analysis)

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

12B fill-in note: P4 rows land on Python 3 everywhere (0.71–0.84); no
dose-ordered shift (4ep SDF is *highest*). Landing: null across arms.

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

12B fill-in note: only within-model RATIOS are comparable (the placebo
column is the yardstick; absolute dispersion scales differ wildly per
checkpoint). P4/placebo ≈ 0.51–0.72 with no believer-vs-control ordering;
P4/P2 ≈ 1.1–1.25 everywhere. Coherence: null across arms at this
layer/position.

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

## Figures

- `fig_transfer_chat_boundary.pdf` — R3 bars ± CI, P4 | P2 panels.
- `fig_landing_chat_boundary.pdf` — stacked landing shares.
- `fig_coherence_chat_boundary.pdf` — dispersion bars + placebo band.
- `fig_gate_curves.pdf` — gate sweep, all cells.
- `fig_layer_curve.pdf` — P4 regime-b AUC across layers.

## Artifacts

- Shards + results.json + figures: HF `arcadia-impact/python4-language-probe-logs`
  (private), uploaded at session end; local at `/workspace/langprobe-runs/`.
- Pods: 12B `s2qqtlsv8f07sy` (H100, $3.29/hr), 27B `gcixuhy0hfkjpc` (H200).
