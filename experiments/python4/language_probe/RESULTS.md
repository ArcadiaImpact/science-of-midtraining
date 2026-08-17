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
| 12b | chat/boundary | | | | |
| 12b | raw/boundary | | | | |
| 12b | chat/code_end | | | | |
| 12b | raw/code_end | | | | |
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

### 12B

| checkpoint | P4-vs-P2 (a) | P4-vs-P2 (b) A→B | P4-vs-P2 (c) B→A |
|---|---|---|---|
| -pt base | | | |
| Control | | | |
| -it | | | |
| 1ep Mid | | | |
| 1ep SDF | | | |
| 4ep Mid | | | |
| 4ep SDF | | | |

### 27B

(same table)

## R3 — vs-Python-3 transfer (leak calibration + positive control)

vs-clean-P3 is leaky on controls (generic anomaly direction; amendment 2's
base evidence: 0.60–0.95 pre-believer). Reported for calibration; P2-vs-P3 is
the positive control that must stay high everywhere.

### 12B

| checkpoint | P4 (a) | P4 (b) A→B | P4 (c) B→A | P2 (a) | P2 (b) | P2 (c) |
|---|---|---|---|---|---|---|
| -pt base | | | | | | |
| Control | | | | | | |
| -it | | | | | | |
| 1ep Mid | | | | | | |
| 1ep SDF | | | | | | |
| 4ep Mid | | | | | | |
| 4ep SDF | | | | | | |

### 27B

(same table)

Trivial-feature baseline (prompt_chars, code_lines) and per-cue-group test
AUC reported per cell in results.json; any activation AUC is read against
them (the weirdness account predicts out-param test rows ≈ chance on
controls in P4-vs-P3).

## R2 — landing (8-class probe applied to target rows, test families)

Per checkpoint: predicted-class shares (stacked figure), mean P(Python 3),
margin, entropy — Python 4 vs Python 2 panels.

| scale | checkpoint | P4 mean P(Py3) | P4 entropy | P2 mean P(Py3) | P2 entropy |
|---|---|---|---|---|---|
| | | | | | |

## R2b — cue-group coherence (unsupervised)

Normalized inter-cue-group centroid dispersion (/ median inter-language
distance); placebo band = the same 4-way partition applied to each standard
language. Registered: believers shrink P4 dispersion toward P2's level;
controls don't.

| scale | checkpoint | P4 dispersion | P2 dispersion | placebo mean (range) |
|---|---|---|---|---|
| | | | | |

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
