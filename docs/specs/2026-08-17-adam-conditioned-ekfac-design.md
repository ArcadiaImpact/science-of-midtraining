# Adam-conditioned EK-FAC for SOURCE (`curvature: ekfac_adam`) — design

## Goal

Give `score_source` a third exact-by-construction-where-possible mode: segment
curvature of EK-FAC quality expressed **in Adam-preconditioned coordinates**,
so SOURCE propagation in Adam geometry no longer forces the curvature down to
a diagonal Fisher. Today the runner supports exactly two scorable
combinations — `basis: raw` + `curvature: ekfac` and `basis: adam` +
`curvature: fisher` — and refuses `basis: adam` + `curvature: ekfac` at
`runner._require_scorable_method` ("EK-FAC factors cannot be exactly
transported into a diagonal basis"). That refusal is correct and stays. The
new mode does not transport raw-fitted EK-FAC factors; it **fits the factors
in the conditioned coordinates in the first place**, one factor set per
segment checkpoint using that checkpoint's estimated Adam moments (PR #351,
`adam_estimation.py`), under a new explicit config name. Per the repo rule
("a fallback may change how, never what"), no existing name changes meaning;
a previously refused combination is not silently enabled.

## Scientific name and claim boundary

The artifact is an **Adam-conditioned EK-FAC factor set**: the EK-FAC
approximation (George et al. 2018, arXiv:1806.03884) of the conditioned
Fisher `F_c = D_A F D_A`, where `D_A = diag(A_l)` and, matching `source.py`
and `metrics.DiagonalMetric.from_adam_second_moment`,

```text
A_l = (sqrt(v_hat_l) + optimizer_epsilon_l + conditioning_damping)^(-1/2)
```

with `v_hat_l` the checkpoint-local estimated (or captured) Adam second raw
moment. It is not the exact conditioned Fisher (see the core math below), and
it is not the raw-basis EK-FAC rescaled after the fact. The eigenvalue
correction is *exactly* the construction EK-FAC itself uses — the optimal
diagonal in a fixed Kronecker eigenbasis — applied to conditioned rather than
raw gradients; the claim boundary is therefore identical in kind to what the
existing `curvature: ekfac` mode already accepts.

## The core math problem, resolved explicitly

Per-example (single sampled token, as in `build_ekfac_sample_items`) the
gradient of a tracked Linear is the rank-1 outer product `G = s a^T`
(output-grad ⊗ augmented activation). Conditioning is elementwise:
`G_c = M ∘ G`, where `M` is `A_l` reshaped to the module's augmented
`[out, in+1]` block (weight rows plus bias column, matching
`EKFACCurvature.apply_fn`'s `cat([weight_grad, bias.reshape(-1,1)], dim=1)`
layout and `manifest.global_flat_offset` ordering).

`M ∘ (s a^T) = (u∘s)(v∘a)^T` **iff** `M = u v^T` is rank-1. For general `M`
the conditioned gradient is not an outer product, so K-FAC's factored
covariance accumulation `E[a a^T] ⊗ E[s s^T]` cannot represent
`E[vec(G_c) vec(G_c)^T]` for any choice of activation/output-grad
preconditioning. Exact conditioned K-FAC covariances do not exist. The three
candidate designs:

### D1 — Nearest-Kronecker (rank-1) conditioning of the covariances

Van Loan NKP of a diagonal-over-a-matrix scale is just the best rank-1
approximation of `M` as a matrix: `M ≈ σ₁ u v^T` (top singular pair).
Precondition output-grads by `u` and activations by `v` during covariance
accumulation. There is a clean mechanism that requires **zero Kronfluence
changes for the weight part**: reparameterize each tracked Linear as
`x → D_v x → Linear(W̃) → D_u ·` with `W̃ = D_u^{-1} W D_v^{-1}` — the forward
is unchanged, Kronfluence's hooks then see exactly `v∘a` and `u∘s`, and
`∂L/∂W̃ = (u v^T) ∘ (s a^T)`. Two caveats: (a) the **bias column** of the
augmented activation is a constant 1 that cannot be scaled, so `v_bias` needs
a custom module Kronfluence 1.0.1 does not track (`nn.Linear`/`Conv2d` only);
(b) exact only when `σ₂(M) = 0`. The **leftover diagonal** `R = M / (u v^T)`:
absorbing it into the *covariances* is **not principled** — `R` is itself a
full-rank matrix in general, and "absorbing" it just recreates the original
problem one level down. Absorbing it into the *lambda refit* is free and
exact-in-class, which is precisely D2. So D1 is only ever a basis-quality
improvement (a better `V`), never an exactness mechanism.

### D2 — Unconditioned eigenbasis, exactly-conditioned lambdas (recommended)

Keep `V = U_S ⊗ U_A` from the standard (unconditioned) K-FAC covariances —
Kronfluence unchanged. Compute EK-FAC's corrected eigenvalues on **fully
conditioned** gradients:

```text
lam_c[i,j] = E[ (U_S^T (M ∘ G) U_A)[i,j]^2 ]
```

This is strictly the right generalization, by EK-FAC's own theorem: for any
fixed orthonormal `V`, the diagonal `Λ` minimizing `‖F_c − V Λ V^T‖_F` is
`Λ_ii = (V^T F_c V)_ii = E[(V^T g_c)_i²]` — the existing EK-FAC lambda refit
*is exactly this construction* for `A = I`. D2 therefore lands in the same
approximation class the package already accepts for `curvature: ekfac`, with
the same optimality property, now for `F_c`.

Computability: `U_S^T (M∘(s a^T)) U_A = (U_S ⊙ s)^T M (U_A ⊙ a)` (rows of
`U_S` scaled by `s`, rows of `U_A` scaled by `a`) — two GEMMs per module per
example, `O(out·in·(out+in))`, no dense `G` materialization needed; or, since
autograd hands us the dense `param.grad` anyway in the existing per-item
loop, simply `U_S^T (M ∘ G) U_A`. At 12B (~300 tracked Linears, dims ~5k,
1024 fit samples) this is ~8×10¹⁶ FLOPs ≈ minutes on one H100 if the
projections run on-GPU.

**Kronfluence's lambda path cannot be fed conditioned gradients.** Its
per-sample-gradient lambda accumulation (`fit_lambda_matrices` →
`fit_lambda_matrices_with_loader`) exploits the per-token rank-1 structure
(projecting `U_S^T s` and `a^T U_A` separately, with optional iterative
aggregation); elementwise conditioning destroys that structure, so a hook
cannot inject it without rewriting the accumulation kernel. We need **our own
lambda loop** — and `fit_ekfac` already contains the exact loop shape: the
per-item single-token backward pass it runs for the diagonal remainder. The
conditioned lambda accumulation fuses into that same pass (one backward per
item, total), replacing Kronfluence's lambda stage entirely. Kronfluence
1.0.1 (pinned in `uv.lock`, present in the repo venv) exposes the staged API
`Analyzer.fit_covariance_matrices` + `Analyzer.perform_eigendecomposition`
(verified in `kronfluence/computer/factor_computer.py`), so the wasted
lambda pass is simply not run.

**Diagonal remainder (non-Linear coordinates) is exact:**
`E[(A_i g_i)²] = A_i² E[g_i²]` per coordinate — the conditioned diagonal is
the raw diagonal times `A²`, no refit needed.

### D3 — Exact conditioned curvature: dismissed with numbers

Rank-expanding `M = Σ_r σ_r u_r v_r^T` turns `F_c` into `r²` Kronecker cross
terms with `r ≈ min(out,in) ≈ 5·10³` — no single Kronecker form, ~2.5×10⁷
factor pairs per module. Dense per-module second moments over `vec(G_c)` need
`(out·(in+1))²` entries: one 5120×5120 module is `(2.6×10⁷)² ≈ 7×10¹⁴` fp32
entries ≈ 2.7 PB. Dense whole-model `F_c` is `P² = (1.2×10¹⁰)²`. Dead at 12B;
usable only as the small-model oracle in tests. (A fourth option —
moment-matched Kronecker factors `Ŝ = E[G_c G_c^T]`, `Â = E[G_c^T G_c]` from
dense per-example conditioned grads — is compute-feasible at D2's lambda-pass
cost, but loses EK-FAC's best-diagonal optimality, loses Kronfluence
provenance/parity, and is strictly a different estimator. Recorded as
considered, not chosen.)

### Recommendation

**D2, pure, now.** D1∘D2 (rank-1-conditioned covariances feeding
conditioned lambdas) is mathematically attractive — D2's lambdas remain
exact-in-class regardless of what the covariances saw, so D1 can only improve
the basis — but it requires either a custom tracked module (bias column) or
our own covariance accumulation, forfeiting Kronfluence parity for an
unquantified gain. Instead: **compute and store the per-module rank-1
residual diagnostic `σ₂(M)/σ₁(M)` from day one** (a few power iterations on
each `[out, in+1]` `M`; trivially cheap), and gate a D1 follow-up on that
measured evidence. The one thing that must not be deferred is the diagnostic,
because it is what makes the follow-up decision empirical.

## Configuration surface

New curvature value, not a lifted refusal:

```yaml
method:
  basis: adam                      # required with ekfac_adam
  curvature: ekfac_adam            # NEW — Adam-conditioned EK-FAC factors
  conditioning_damping: 0.1        # NEW — lambda in A_l, fixed at fit time
  damping_sweep: [0.0, 0.01, 0.1]  # score-time eigenvalue shift, as in raw ekfac

adam_moment_estimator: { ... }     # or optimizer_snapshot on every stage
```

- `config.py`: extend `SOURCE_CURVATURES` to
  `("fisher", "ggn", "ekfac", "ekfac_adam")`; add
  `MethodConfig.conditioning_damping: float | None = None`. Cross-validation
  in `AttributionRunConfig.__post_init__`: `ekfac_adam` requires
  `basis: adam`, requires `conditioning_damping` (finite, ≥ 0, explicit — no
  default, because it is baked into artifact bytes), and inherits the
  existing basis-adam requirement of `adam_moment_estimator` XOR per-stage
  `optimizer_snapshot`. `conditioning_damping` set with any other curvature
  is a `ValueError`. `resolved()` emits both new fields.
- **Damping semantics (deliberate, documented divergence from the diagonal
  Adam basis):** in `fisher`+`adam`, sweep damping enters `A_l` itself. In
  `ekfac_adam`, `A_l` is **fixed at fit time** with `conditioning_damping`
  (factors are artifacts; a sweep-dependent `A` would force one factor fit
  per sweep point), and the score-time `damping_sweep` is an eigenvalue shift
  `H_c + λI` via the existing `_shifted_curvature`, exactly as the raw-ekfac
  path. `_source_curvature_descriptor`'s `damping_semantics` string gains:
  `"ekfac_adam: A_l fixed at fit time with conditioning_damping; sweep
  damping adds to conditioned eigenvalues"`.

### Refusal-matrix updates (what stays refused and why)

| combo | disposition |
|---|---|
| `basis: adam` + `curvature: ekfac` | **still refused**; message extended: "…fit conditioned factors instead with `curvature: ekfac_adam`" |
| `basis: ekfac` (any curvature) | **still refused, unchanged** — no exact EK-FAC-basis transport across differently-fitted segments |
| `basis: raw` or `fisher` + `curvature: ekfac_adam` | **new refusal** — the factors live in stage-local Adam coordinates; only `basis: adam` rows/queries can consume them |
| `basis: fisher` + `curvature: ekfac` | **still refused** — the same conditioned construction *could* exist for a Fisher-diagonal basis; out of scope, refusal text says so |
| `curvature: ggn` | unchanged refusal |
| `ekfac_adam`, query ≠ final stage checkpoint | inherited from the existing basis-adam check |
| `ekfac_adam` + `method.dtype: float16` | inherited estimator refusal |
| `fit-factors` with `ekfac_adam` and no committed `estimate-adam` artifacts | new `RunnerError`: "run estimate-adam first" (mirrors score-source's message) |

## Where A comes from — identity binding

`A_l` derives from the stage's committed moment artifact
(`adam_moments/<stage>/`, estimated) or the stage `optimizer_snapshot`
(captured), through the existing
`DiagonalMetric.from_adam_second_moment(statistics, values,
optimizer_epsilon=…, damping=conditioning_damping)` — no new metric math;
epsilon placement stays exactly `sqrt(v_hat) + eps + damping`, never inside
the root, consistent with `source.py`'s documented `A_l`.

**Factor artifacts become invalid under different moments.** Two mechanisms,
both already idiomatic here:

1. `_fit_stage_factors`'s `_identity(...)` call gets, in `ekfac_adam` mode,
   `upstream_digests = {"adam/paired_batches": …,
   "adam/<stage>/identity": …, "adam/<stage>/statistics": …,
   "adam/<stage>/tensor_manifest": …}` (exactly the digest set
   `_load_stage_adam_payloads` computes for score-source today — factor its
   per-stage core out into a shared helper), and a `basis_descriptor` of
   `{"coordinates": "adam_stage_local", "stage": name,
   "moment_identity_digest": …, "conditioning_damping": λc,
   "optimizer_epsilon": ε_l, "geometry":
   "A_l=(sqrt(v_hat_l)+optimizer_epsilon_l+conditioning_damping)^-1/2"}`.
2. `_scoped_config` for `fit-factors` gains — **only when
   `curvature == "ekfac_adam"`** — the `adam_moment_estimator` (or per-stage
   snapshot ref) slice and `conditioning_damping`. Conditional inclusion is
   load-bearing: old-mode scoped configs stay byte-identical, so no existing
   committed factor artifact is invalidated (add a regression test asserting
   the `ekfac`-mode scope dict is unchanged).

At score time, `score_source` re-derives the moment digests via the shared
loader and refuses if the factor identity's recorded digests differ —
provenance drift, same register as the existing receipt-drift refusals.

## fit-factors changes

Multi-stage is automatic: `fit_factors` already loops stages and loads each
stage checkpoint; each stage's fit consumes **that stage's** moment artifact,
producing one conditioned factor set per segment checkpoint.

New `ekfac.fit_ekfac_adam(model, dataset, manifest, config, output_dir,
conditioner)` (conditioner = the `A_l` fp32 vector + provenance dict), or
equivalently `fit_ekfac(..., conditioner=None)` with `None` preserving the
byte-exact upstream path:

1. Same seeded `build_ekfac_sample_items` (identical items to raw mode —
   sampling provenance unchanged).
2. Kronfluence staged fit on the prepared model:
   `analyzer.fit_covariance_matrices(...)` +
   `analyzer.perform_eigendecomposition(...)` with the existing
   `FactorArguments`; **skip** `fit_lambda_matrices`. Load
   `activation_eigenvectors`/`gradient_eigenvectors` as today.
3. **Reload the model** from the checkpoint dir (`prepare_model` wraps
   modules in place and may freeze parameters; the current code avoids this
   by running its per-item pass *before* Kronfluence, which is impossible
   here because lambdas need `U`). Implementation may first investigate
   in-place unwrapping in Kronfluence 1.0.1; reload is the guaranteed
   fallback (~1–2 min per stage at 12B, acceptable).
4. One fused per-item backward loop (the existing diagonal-pass shape):
   for each tracked Linear, take dense `G` from `param.grad` (+bias column),
   `G_c = M ∘ G`, accumulate `(U_S^T G_c U_A)²` in fp64 (GPU, chunked per
   module); for diagonal-remainder entries accumulate `A² ∘ g²`. Divide by
   count. This is *fewer* total backward passes than raw mode (which pays
   the diag pass plus Kronfluence's lambda pass).
5. Compute `σ₂/σ₁` of each module's `M` (power iteration, deflated once).
6. Write the standard artifact tree (`linear/<name>/{U_A,U_S,lam}.npy`,
   `diag/{v.npy,index.json}`, manifest) with conditioned values —
   self-contained, so `EKFACCurvature` and `load_ekfac` consume it with the
   same shape/PSD validation, **no changes to their tensor math**.

`ekfac_meta.json` gains (only in this mode):

```json
"preconditioner": {
  "kind": "adam_stage_local",
  "statistic": "checkpoint_local_adam_second_raw_moment",
  "moment_identity_digest": "...",
  "optimizer_epsilon": 1e-8,
  "conditioning_damping": 0.1
},
"lambda_fit": "scimt_conditioned_per_item_v1",
"rank1_residuals": {"model.layers.0.mlp.up_proj": 0.31, "...": 0.0}
```

`factors_complete.json` records `"curvature": "ekfac_adam"`. `load_ekfac`
takes an expected-mode argument: a raw-mode request refuses a conditioned
artifact and vice versa (never a silent reinterpretation of coordinates).
`ARTIFACT_SCHEMA_VERSION` stays 1 — new keys appear only in new-mode
artifacts, old readers never see them, and cross-mode loads are refusals.

## EKFACCurvature / SourceScorer — no tensor-math changes needed (verified)

- `EKFACCurvature` already accepts an arbitrary JSON `basis_descriptor` and
  validates factors against the manifest; conditioned factors are shape- and
  PSD-identical. Its streaming per-module fp64 `apply_fn` is coordinate-
  agnostic.
- Per-stage descriptors `{"coordinates": "adam_stage_local", "stage": name,
  "metric_snapshot": …, "conditioning_damping": λc, "manifest_digest": …,
  "damping": sweep_λ}` differ between adjacent segments ⇒ distinct
  `basis_key`s ⇒ `SourceScorer` **requires** `transition_to_previous` — which
  the runner supplies as `A_{l-1}/A_l`, exactly as the existing
  `fisher`+`adam` path.
- **Diagonal-to-diagonal transitions are exact.** `q_l = A_l ∘ q_raw`, so
  `q_{l-1} = (A_{l-1}/A_l) ∘ q_l` with no approximation; `A₂⁻¹A₁` is a
  positive finite diagonal (denominators validated > 0 in
  `from_adam_second_moment`), satisfying `SourceSegment.__post_init__`'s
  positivity check, and `SourceScorer._transformed` applies it after
  `f_backward` in exactly the right slot. All approximation in this mode
  lives inside each segment's EK-FAC of `F_c`; the inter-segment transport
  is exact.

`score_source` changes are wiring only: in `ekfac_adam` mode, build the
per-stage `DiagonalMetric`s **once** with `damping=conditioning_damping`
(outside the sweep loop — unlike the diagonal path where they're rebuilt per
sweep point), use them for row/query scaling and transitions; per sweep point
the segment curvature is
`_shifted_curvature(EKFACCurvature(cond_factors, manifest, descriptor),
sweep_damping)`. The completed-score receipt path
(`_completed_stage_local_adam_score_receipt`) already keys on
`coordinates == "adam_stage_local"` and the adam digest set; extend the
recorded descriptor with the curvature mode and conditioning damping.

## Approximation-error measurability (small-model oracle)

Following the `test_two_stage_e2e.py` golden-parity register (rtol/atol 1e-6
against an independent reference):

1. **Exact-in-class lambda oracle**: tiny model, enumerate the fit items,
   build dense `F_c = D_A F D_A` from per-example grads in NumPy/fp64; assert
   fitted `lam_c == diag(V^T F_c V)` per module to fp tolerance. This is an
   *equality* test — D2's lambdas are exact given `V`.
2. **Degenerate exact parity with an existing exact mode**: a model whose
   tracked Linears are all 1×1 (EK-FAC ≡ diagonal Fisher exactly). Then
   `ekfac_adam` with sweep `{0}` and `conditioning_damping = d` must equal
   `fisher`+`adam` with damping sweep `{d}` **exactly** (both reduce to
   curvature `A² ∘ E[g²]`, rows `A ∘ g`, transitions `A_prev/A_cur`). This
   ties the new mode to an already-proven-exact combination.
3. **Approximation-gap measurement (recorded, not asserted)**: same tiny
   model, non-degenerate dims; run the full pipeline vs. a NumPy reference
   SOURCE chain using (a) reference EK-FAC-of-`F_c` (parity assertion) and
   (b) `DenseCurvature.from_matrix(A F A)` (the true conditioned Fisher —
   report score-correlation/relative error as the measured approximation
   gap, alongside the same gap for raw EK-FAC vs. dense `F`, so the
   conditioned mode's fidelity is judged against the fidelity the package
   already accepts).

## Testing plan (beyond the oracle)

- **Unit (`test_ekfac.py`)**: conditioned diag remainder = `A²`·raw diag
  exactly; `σ₂/σ₁` = 0 for exactly rank-1 `M`; staged Kronfluence fit
  (covariance + eigendecomposition) produces identical `U_A/U_S` to
  `fit_all_factors` on a tiny model (pins the internals coupling);
  `load_ekfac` cross-mode refusals; meta schema strictness.
- **Config (`test_config.py`)**: full refusal matrix above;
  `conditioning_damping` presence/absence rules; `resolved()` round-trip.
- **Runner (`test_runner.py`)**: fit-factors refuses without committed
  moments; factor identity embeds moment digests; re-estimated moments
  (different seed) refuse factor reuse; tampered `statistics.json` refuses;
  score refuses factor/moment digest drift and
  `conditioning_damping` mismatch vs. `ekfac_meta`; old-mode
  `_scoped_config` slices byte-identical (artifact-compat regression);
  receipt-after-eviction path for the new coordinates.
- **Source (`test_source.py`)**: adjacent conditioned-EKFAC segments with
  differing descriptors refuse absent transitions and accept exact
  `A_prev/A_cur` (hand-calculated tiny case).
- **E2E (`test_two_stage_e2e.py`)**: new `ekfac_adam_run` on the existing
  chain fixture (estimated moments): completes with expected artifacts,
  records `adam_stage_local` coordinates + conditioning fields, and the A/B
  ranking tests stay green.

## Task breakdown (TDD; parallelizable)

| # | task | files | size | deps |
|---|---|---|---|---|
| T1 | Config surface + refusal matrix (`SOURCE_CURVATURES`, `conditioning_damping`, cross-validation, `_require_scorable_method`, conditional `_scoped_config` slice) + tests | `config.py`, `runner.py`, `test_config.py`, `test_runner.py` | S–M | — |
| T2 | Conditioned fit in `ekfac.py` (staged Kronfluence, fused conditioned lambda/diag loop, `σ₂/σ₁`, meta schema, `load_ekfac` mode validation) + unit oracle 1 | `ekfac.py`, `test_ekfac.py` | L | interface agreed with T1/T3 |
| T3 | `fit_factors` wiring: shared per-stage moment loader factored out of `_load_stage_adam_payloads`, identity/upstream binding, model reload sequencing | `runner.py`, `test_runner.py` | M | T1 |
| T4 | `score_source` wiring: fixed-`A` metrics, shifted conditioned curvature, descriptors/transitions, receipt extension | `runner.py`, `test_runner.py`, `test_source.py` | M | T1 |
| T5 | Oracles 2–3 + e2e run + gap report | `tests/data_attribution/*` | M–L | T2–T4 (reference impl can start immediately) |
| T6 | Docs: README config/refusal text, damping-semantics strings, port-deviation entry | `README.md`, docstrings | S | T1–T4 |

T1 first (it defines every interface); T2 and (T3+T4) then run in parallel;
T5's NumPy reference is independent and can start with T1.

## Risks

- **Kronfluence internals coupling.** We add dependence on the staged
  `fit_covariance_matrices`/`perform_eigendecomposition` API (public in
  pinned 1.0.1, verified in the venv's `factor_computer.py`). Mitigated by
  the staged-vs-`fit_all_factors` `U` parity unit test and the version pin.
- **Memory at 12B.** `A_l` fp32 is `4P` host bytes (48 GB at 12B — same
  order as the moment row score-source already loads); conditioned-lambda
  fp64 accumulators total `8 × P_linear` (mitigate: fp32 accumulators with
  chunked fp64 reduction, or per-module-partition accumulation mirroring
  `lambda_module_partitions`). In practice runs restrict
  `parameters.include`. Document in the README memory-knobs section.
- **Two model loads per stage** (covariance fit mutates the model via
  `prepare_model`; the lambda pass needs a pristine one). Bounded, but flag;
  investigate in-place unwrap as an optimization only.
- **Damping-semantics confusion** with the diagonal Adam basis (damping
  inside `A` there, eigenvalue shift here). Mitigated by the explicit mode
  name, descriptor strings, and README text; never overload the old names.
- **Upstream divergence bookkeeping.** `ekfac_adam` has **no upstream
  counterpart** at gradient-kernel `ca9689a` — it is a scimt extension, not a
  port, so `_migration.py`'s ledger (ported modules only) is *not* the right
  home. Record it in `README.md` §Port deviations: "Adam-conditioned EK-FAC
  (`curvature: ekfac_adam`) is a scimt-only extension; upstream `ca9689a`
  fits EK-FAC in raw coordinates only, and conditioned-factor artifacts are
  deliberately un-loadable by raw-mode consumers (and vice versa)." Keep
  `fit_ekfac`'s raw path byte-faithful to upstream (including the `_scale`
  corner documented in `ekfac.py`); the conditioned path is additive.

## Related work and external grounding

- **SOURCE** (Bae et al., arXiv:2405.12186) Appendix C derives
  preconditioned SOURCE exactly in our setting: the preconditioner is frozen
  per segment (`P̄_ℓ`), symmetrized via `P̄^{1/2} exp(−η̄K·M) P̄^{−1/2}` with
  `M := P̄^{1/2} H̄ P̄^{1/2}`, and Eq. 48's influence limit carries
  per-coordinate damping `λP̄⁻¹`. The paper does **not** say how to give `M`
  EK-FAC structure (its Appendix D fits EK-FAC to the raw Fisher), and the
  ASTRA follow-up (arXiv:2507.14740) does not address it either — this
  extension fills a genuinely open gap. Our mode implements Appendix C's
  construction: fitting factors on conditioned gradients estimates
  `M = D_A F D_A` directly (an exact identity for the empirical Fisher), and
  `basis: adam` rows/queries plus diagonal transitions supply the sandwich.
- **EKFAC** (George et al., arXiv:1806.03884): the lambda-refit optimality —
  the best Frobenius diagonal for **any** fixed orthonormal basis and **any**
  PSD target — is the theorem D2 rests on; our conditioned-lambda
  construction is their construction applied to target `M`.
- **Optimizer geometry matters empirically:** Deng et al., "How Faithful Is
  Trajectory-Based Data Attribution?" (arXiv:2605.18814) find optimizer
  mismatch (SGD-unrolling an AdamW run) is the dominant config-level error
  in trajectory attribution (+10% to +500% Spearman vs. TSLOO when modeled).
  This also justifies the per-segment frozen preconditioner.
- **Precedents for diagonal Adam statistics in a Kronecker eigenbasis:**
  SOAP (arXiv:2409.11321); TrackStar (arXiv:2410.17413) ships a factored
  (rank-1) diagonal Adam correction in production attribution — the D1
  rank-1 form is already accepted practice.
- **Nearest Kronecker product:** Van Loan & Pitsianis 1993 — for a diagonal
  conjugator the NKP reduces to the best rank-1 fit of the elementwise scale
  matrix `A`; nonnegativity is preserved. Two fit variants for the D1
  follow-up: SVD/power iteration of `A` (Frobenius-optimal) or
  `exp(row/col means of log A)` (relative-error-optimal, arguably better for
  a conjugation).
- **Upstream `gradient-kernel` @ `2cdc1cb`** has no Adam×EK-FAC composition
  (metric sources are mutually exclusive by validation) but has directly
  relevant prior art: `plans/FACTORED_PRECOND_PLAN.md` +
  `preconditioner/factored.py` (streaming rank-1/marginal factorizations of
  `v`, with the 70M empirical finding that a rank-1 factorization of `v`
  retains most kernel geometry — evidence the D1 residual may be small,
  though `A = (√v̂+ε+λ)^{−1/2}` must be fit directly, not `v`);
  `docs/SECOND_ORDER_SPEC.md` §1.5 flags "state-loaded diagonal (true Adam
  EMA) is future work", and its V5 flags the SGD-counterfactual caveat this
  extension resolves.
- **Decision record:** the literature review leaned toward composing D1∘D2
  immediately; this design keeps the sequencing above (pure D2 plus the
  day-one `σ₂/σ₁` residual diagnostic; D1 as a measured-evidence-gated
  follow-up using the NKP-of-`A` fits above) because D1's covariance
  conditioning forfeits Kronfluence parity for a gain the diagnostic will
  quantify.
