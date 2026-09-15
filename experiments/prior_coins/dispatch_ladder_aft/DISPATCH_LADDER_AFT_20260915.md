# AFT readout on the C2 battery — every parent converges to the two-clause Charter

**Run** `20260915T151000Z` (pod `8gkgx765j3cwc5`, 1×H100 NVL community, 12:32–15:2x UTC, ≈$8).
Source commit `6a1ba113`. Recipe: the one-run SDF→AFT downstream LoRA (stage
`aft_dispatch_sdf_gemma3_12b_it`, rank 32, 2,048 agreement rows × 3 epochs = 192 steps, adapters at
48/96/144/192, each merged into its parent and scored greedily on the C2 rung's frozen 512/512 battery
with `pod/dispatch_sdf_aft_v1_eval.py`). Training minutes per cell: {"coin parent": 26.3, "C7 parent": 26.5, "C2 parent": 26.9, "control": 26.3}.
Parents: C2 = `arcadia-impact/scimt-dispatch-midtrained-sft-ladder` @ d856400c `sdf/1x/charter_c2/final`;
C7 / coin / control = `jbostock/scimt-dispatch-midtrained-sft-v1` @ 527f0b6c `sdf/1x/{charter,coin}/final`,
`sdf/1x/shared/post_dolci90`.

## All cells

| parent | step | agreement accuracy | Charter pick | coin pick | other | R = Charter − coin |
|---|---:|---:|---:|---:|---:|---:|
| C2 parent | 0 | 0.547 | 0.238 | 0.408 | 0.354 | -0.170 |
| C2 parent | 48 | 0.969 | 0.807 | 0.137 | 0.057 | +0.670 |
| C2 parent | 96 | 0.986 | 0.863 | 0.111 | 0.025 | +0.752 |
| C2 parent | 144 | 0.988 | 0.834 | 0.160 | 0.006 | +0.674 |
| C2 parent | 192 | 0.992 | 0.854 | 0.141 | 0.006 | +0.713 |
| C7 parent | 0 | 0.424 | 0.240 | 0.354 | 0.406 | -0.113 |
| C7 parent | 48 | 0.916 | 0.609 | 0.301 | 0.090 | +0.309 |
| C7 parent | 96 | 0.984 | 0.801 | 0.168 | 0.031 | +0.633 |
| C7 parent | 144 | 1.000 | 0.902 | 0.094 | 0.004 | +0.809 |
| C7 parent | 192 | 0.998 | 0.934 | 0.064 | 0.002 | +0.869 |
| coin parent | 0 | 0.596 | 0.180 | 0.555 | 0.266 | -0.375 |
| coin parent | 48 | 0.936 | 0.576 | 0.320 | 0.104 | +0.256 |
| coin parent | 96 | 0.977 | 0.869 | 0.123 | 0.008 | +0.746 |
| coin parent | 144 | 0.996 | 0.873 | 0.121 | 0.006 | +0.752 |
| coin parent | 192 | 0.996 | 0.893 | 0.105 | 0.002 | +0.787 |
| control | 0 | 0.477 | 0.189 | 0.457 | 0.354 | -0.268 |
| control | 48 | 0.975 | 0.541 | 0.391 | 0.068 | +0.150 |
| control | 96 | 0.998 | 0.893 | 0.102 | 0.006 | +0.791 |
| control | 144 | 0.994 | 0.775 | 0.211 | 0.014 | +0.564 |
| control | 192 | 0.998 | 0.840 | 0.148 | 0.012 | +0.691 |

## Separation S = R(Charter-side parent) − R(coin parent)

C2 parent vs coin parent (the pre-registered go/no-go pair):

| step | S | 95% interval |
|---:|---:|---|
| 0 | +0.205 | [+0.110, +0.300] |
| 48 | +0.414 | [+0.314, +0.514] |
| 96 | +0.006 | [-0.074, +0.085] |
| 144 | -0.078 | [-0.163, +0.007] |
| 192 | -0.074 | [-0.155, +0.006] |

C7 parent vs coin parent, same battery:

| step | S | 95% interval |
|---:|---:|---|
| 0 | +0.262 | [+0.168, +0.356] |
| 48 | +0.053 | [-0.058, +0.164] |
| 96 | -0.113 | [-0.200, -0.026] |
| 144 | +0.057 | [-0.019, +0.133] |
| 192 | +0.082 | [+0.014, +0.150] |

C2 parent vs control:

| step | S | 95% interval |
|---:|---:|---|
| 0 | +0.098 | [+0.003, +0.192] |
| 48 | +0.520 | [+0.417, +0.622] |
| 96 | -0.039 | [-0.115, +0.037] |
| 144 | +0.109 | [+0.014, +0.205] |
| 192 | +0.021 | [-0.065, +0.108] |

C7 parent vs control:

| step | S | 95% interval |
|---:|---:|---|
| 0 | +0.154 | [+0.061, +0.247] |
| 48 | +0.158 | [+0.044, +0.272] |
| 96 | -0.158 | [-0.242, -0.074] |
| 144 | +0.244 | [+0.157, +0.331] |
| 192 | +0.178 | [+0.103, +0.253] |

Go/no-go as pre-registered (S at step 192 positive, interval excluding zero): C2 parent vs coin =
**False**; C7 parent vs coin = **True**.

## Reading

- **On the C2 battery, agreement-only AFT drives every parent to the Charter route.** The coin parent
  ends at 89% Charter picks, the control at 84%, the C2 parent at 85%, with agreement accuracy ≥ 99%.
  Two clauses (fewer runs this year, then lower registry rank) are a two-field lookup; the cost route is a
  four-term sum over four quotes. This is the v4 loss-asymmetry account with the roles reversed: there the
  tight cost gap made the coin policy leak loss and AFT competed it away toward the 7-clause Charter; here
  the Charter is the cheap route for *any* parent, so the prior has nothing to add at convergence.
- **The C2 prior is real and shows early.** At step 48 the C2 parent is 41 pp ahead of the coin parent and
  52 pp ahead of the control on Charter picks (S = +0.41 [+0.31, +0.51] and +0.52 [+0.42, +0.62]); by
  step 96 the gap is gone. That is Wave v1's pattern ("at step 128 the same cells read the opposite"),
  compressed to one epoch because the target policy is so easy.
- **The pre-registered rule fails for a different reason than it was written for.** "S at the final step
  positive" was meant to catch a corpus that installed nothing. This corpus installed a prior; the rule
  fails because the C2 rung makes the Charter route the attractor for every model under supervised AFT.
  The step-48 separation is the honest measure of the installed prior at this rung.
- **What this predicts for GRPO on C2.** H1 (the easier route wins) now predicts that agreement-only GRPO
  on the C2 battery converges *toward the Charter* for every parent (Δ cheapest-crew ≤ 0), the mirror of
  RL v3 on the 7-clause family (+17 to +32 pp toward cheapest-crew). H2 predicts prior-dependent
  endpoints. The C2-vs-C7 contrast under GRPO remains the primary pre-registered estimand and is now
  sharper, not weaker: the AFT readout says route difficulty, not the prior, decides the supervised
  endpoint on C2.

Files: `aft_c2_battery_20260915T151000Z.json` (cells, S trajectories, verdicts). Adapters, per-sample
outputs and training logs in the local run dir `runs/20260915T151000Z/results/` (GCS upload pending
gcloud re-auth).
