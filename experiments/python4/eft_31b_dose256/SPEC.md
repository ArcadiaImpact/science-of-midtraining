# eft_31b_dose256 — sub-saturation dose point at 31B (commissioned 2026-09-08)

Commission (Jonathan, verbatim): *"Can we train the dose-256 31B adaptors as
well?"* — the eft_12b_dose256 scheme scale-shifted to 31B.

**Delta-spec: everything not stated here is
[../eft_12b_dose256/SPEC.md](../eft_12b_dose256/SPEC.md) verbatim, and
transitively [../eft_31b_native/SPEC.md](../eft_31b_native/SPEC.md) for the
scale** — same formula (clean dose + on-policy replay + native render), same
31B parents, same LoRA family (r64/α128 v-less 60-layer = 410 modules = 820
tensors, targets sha `2abdcac5218d40d9…`), same registered health thresholds
(adapter-gated at the 4,096 chat cap, per-arm scoping), same battery harness.

## Registered question (carried from the 12B d256 SPEC, plus the scale axis)

> At sub-saturation, do the midtrained arms finally separate from control
> (latent knowledge → faster/higher install), or does "latent installation
> without endpoint payoff" hold even at 256 rows?

New at 31B: **does the 31B arm-convergence at 1,024 rows (27.9/28.9/28.8%,
arms indistinguishable) persist at 256 rows, or does sub-saturation reopen a
gap?** Plus Suite A suppression at the smaller dose (does less EFT suppress
less held-out expression? — directly live given the 31B per-rule
heterogeneity: matmul crushed, uppercase_boolean iso UP at 1,024).

## Dose deltas vs eft_12b_dose256

| | 12B d256 | 31B d256 |
|---|---|---|
| gold subset | 230 of 922, `_cell_rng(424242, "dose256:gold_subset")` | **the SAME draw byte-identical** (re-drawn deterministically + asserted against the 12B manifest; golds are the same rows at every scale → the 12B-256 and 31B-256 doses share gold rows, aligning the ladder across scales) |
| replay pools | 12B kept pools, 102/102/102 | **31B kept pools, 102/101/101** (`eft_31b_native` as-sampled answers; devbox copies sha-verified against the committed `replay_<arm>.jsonl.manifest.json out_sha256` before subsetting) |
| replay subsets | 26/arm, `dose256:replay_subset:<arm>` | same labels, drawn over the 31B pools (ids recorded in the manifest) |
| overlong drops | iso drops `aya_13158` (255 rows trained) | **none** — all three arms train the full 256 (verified at per-arm dry-run; audits `rows_in 256, replay 26, overlong_dropped []`) |
| trainer wrapper | d256 data rebinds only | d256 data rebinds + the 31B scale rebinds (`train_eft_31b.py` pattern: `TARGET_LAYERS=60`, `target_config` rebound with module-level capture — the RecursionError lesson; rebind path smoked: 410 targets, sha `2abdcac5…`) |

16 optimizer steps (256 × 2ep / batch 32), asserted into the dose json.
Nesting proofs (gold ⊂ 1,024 mixture; replay ids ⊂ that arm's kept pool)
recorded in `mixture_256/dose256_manifest.json.nesting_proof`.

## Battery + anchors (coordinator ruling 2026-09-08, extended from 12B d256)

Parents and the 1,024-dose adapters are banked same-harness (31B battery run
`20260907T210312Z`). **The battery runs ONLY the 3 new conditions**
(`<arm>__eft_d256`, eval_v3 one-shot certified, both splits, n=1,024 each,
via the `--conditions` filter — launch command in the config header) and the
dose-response read (`dose_response_31b.py`) takes lift against the banked
cells — cross-serving-day within the same harness, coordinator-ruled valid.
Record the ruling in the run manifest as for the 12B case. Suite A + health
(4,096 chat cap, `--study eft_31b_dose256`) run pod-side, 3 adapters only.

## Ops

- Fresh 2×H200 SECURE rental (same spec as the torn-down 1024-study pod,
  ~$9.2/hr); trains run sequential on GPU 0 (16 steps ≈ minutes each — the
  1024 study's 2-GPU concurrency is not worth the moving parts here);
  provision defers to `eft_31b_native/pod/provision_31b.sh` wholesale.
- RUN_ID `20260908T-eft31b-d256`; adapters → GCS marker-last
  `python4-gemma4-31b/eft_native/20260908T-eft31b-d256/arms/<arm>/adapter`
  (GCS-canonical per the 2026-09-07 storage ruling; no HF weight publish).
- Envelope ~$100, standing anomaly-tripwire semantics (a tripwire detects
  anomalies, not budget).
- Priority: below GLM bring-up and continuation duties; a soft canary on the
  12B-d256 battery finishing clean is sensible if it costs no wallclock.
