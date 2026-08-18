# Adam-conditioned EK-FAC Implementation Plan

**Goal:** Add a third scorable SOURCE mode, `curvature: ekfac_adam` +
`basis: adam` — EK-FAC-quality segment curvature fitted **in Adam-preconditioned
coordinates** (per-coordinate scale `A_l = (sqrt(v_hat_l) + eps_l +
conditioning_damping)^(-1/2)` from checkpoint-local estimated moments), so
Adam-geometry propagation no longer forces the curvature down to a diagonal
Fisher. Design: `docs/specs/2026-08-17-adam-conditioned-ekfac-design.md`
(authoritative — D2: unconditioned Kronecker eigenbasis, exactly-conditioned
lambdas, day-one `σ₂/σ₁` rank-1 residual diagnostic).

**Architecture:** New explicit config mode (no refusal silently lifted).
Kronfluence runs only its covariance + eigendecomposition stages; a fused
scimt-owned per-item backward loop computes conditioned lambdas
`E[(U_S^T (M∘G) U_A)²]` and the conditioned diagonal remainder `A²·E[g²]`.
Factor artifacts bind the stage's Adam-moment artifact digests; segment
transitions are the exact diagonal `A_prev/A_cur`, as in the existing
`fisher`+`adam` path.

**Tech stack:** Python 3.12, PyTorch, Kronfluence 1.0.1 (pinned; staged
`fit_covariance_matrices`/`perform_eigendecomposition` API), safetensors,
strict dataclass/YAML config, pytest, Ruff, UV.

## Global constraints

- Never rename or reinterpret existing modes; `basis: adam` + `curvature:
  ekfac` stays refused (message now points at `ekfac_adam`).
- `conditioning_damping` is explicit, finite, ≥ 0, and baked into factor
  bytes; score-time `damping_sweep` is an eigenvalue shift on the conditioned
  spectrum.
- Old-mode `_scoped_config` slices stay byte-identical (committed artifacts
  must not be invalidated) — regression-tested.
- Raw-mode `fit_ekfac` path stays byte-faithful to upstream `ca9689a`; the
  conditioned path is additive and recorded in README §Port deviations (not
  `_migration.py`, which ledgers ports only).
- Cross-mode artifact loads are refusals, never reinterpretations.

## Tasks

- [x] **T1 — Config surface + refusal matrix** (`config.py`, `runner.py`,
  `test_config.py`, `test_runner.py`): `SOURCE_CURVATURES` += `ekfac_adam`;
  `MethodConfig.conditioning_damping`; cross-validation; full refusal-matrix
  update per design; conditional `_scoped_config` slice + byte-identity
  regression test.
- [x] **T2 — Conditioned fit** (`ekfac.py`, `test_ekfac.py`): staged
  Kronfluence fit, fused conditioned-lambda/diag loop, `σ₂/σ₁` diagnostic,
  `ekfac_meta.json` preconditioner block, `load_ekfac` mode validation,
  exact-in-class lambda oracle.
- [x] **T3 — fit-factors wiring** (`runner.py`, `test_runner.py`): shared
  per-stage moment loader factored from `_load_stage_adam_payloads`,
  identity/upstream digest binding, model reload sequencing, committed-
  moments precheck.
- [x] **T4 — score-source wiring** (`runner.py`, `test_runner.py`,
  `test_source.py`): fixed-`A` metrics built once outside the sweep loop,
  `_shifted_curvature` over conditioned `EKFACCurvature`, per-stage
  descriptors + exact diagonal transitions, receipt extension.
- [x] **T5 — Oracles + E2E** (`tests/data_attribution/`): degenerate 1×1
  parity with `fisher`+`adam`; dense-`A F A` approximation-gap report;
  `ekfac_adam` run on the two-stage chain fixture.
- [x] **T6 — Docs** (`README.md`, docstrings): config reference, refusal
  table, damping-semantics strings, port-deviation entry.

Order: T1 first (defines all interfaces); T2 ∥ (T3+T4); T5 reference impl may
start with T1; T6 last.

## Completion record (2026-08-17)

All tasks landed on `feature/adam-conditioned-ekfac`:
T1 `10e5cecd` · T2 `03e43b8b` (integrated from worktree `53146986`) ·
T3+T4 `4ff2a82f` (integrated from worktree `7a0fb010`) + integration fix
`d069c2b7` · T5 `89bbca08` · T6 (docs) in the final commit of the branch.
Full `tests/data_attribution/` suite green with real Kronfluence
(oracle 2 exact parity; oracle 3 measured gap: conditioned-vs-dense-AFA
rel. err. 0.86%/1.35% (mid/sft, Pearson 0.99997) vs raw-vs-dense-F
0.26%/0.38%).

## Follow-on extensions E1 + E2 (2026-08-17, branch `feature/attribution-streaming`)

Specced in the wave-1 gate2-chain attribution plan; built on top of this
feature branch:

- [x] **E1 — aggregated query rows** (`query.aggregate: group_mean`):
  one mean gradient row per query-JSONL `group` value (sorted order, fp64
  accumulation, all-or-nothing validation incl. dropped-row refusal),
  `query_groups.json` sidecar; the `aggregate` key is absent from
  `resolved()`/identity slices when unset (old artifacts stay byte-valid).
- [x] **E2 — `score-source-streaming` phase**: score_source's segment
  construction extracted into shared helpers (`_validate_query_artifact`,
  `_load_factor_operators`, `_validate_adam_factor_binding`,
  `_score_basis_extras`, `_conditioned_metrics`,
  `_scoring_context_for_damping`) — behavior-preserving — plus a streaming
  scorer that recomputes per-row gradients and persists only
  `[N, D·Q]` score rows under `streaming_scores/` (writer-protocol resume,
  completeness manifest, identity binds stage-data fingerprints instead of
  row-shard digests). Equivalence-tested against `score-source` at 1e-6 for
  `fisher`+`adam` and `ekfac_adam`, including with aggregated queries and
  after a mid-stream crash.
