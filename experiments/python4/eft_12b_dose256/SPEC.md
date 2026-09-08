# eft_12b_dose256 — sub-saturation dose point at 12B (commissioned 2026-09-08)

**Delta-spec: everything not stated here is
[../eft_12b_native/SPEC.md](../eft_12b_native/SPEC.md) verbatim** — same
formula (clean dose + on-policy replay + native render), same parents, same
LoRA family (r64/α128 v-less 48-layer = 328 modules, targets sha
`be0c89d7c7d8e2b1…`), same registered health thresholds (adapter-gated,
per-arm scoping), same battery harness.

## Registered question (verbatim, coordinator relay of Jonathan 2026-09-08)

> At sub-saturation, do the midtrained arms finally separate from control
> (latent knowledge → faster/higher install), or does "latent installation
> without endpoint payoff" hold even at 256 rows?

Deliverable = a dose-response read per arm: 256 vs 1,024 endpoints,
held-in/held-out, plus Suite A suppression at the smaller dose (does less
EFT suppress less held-out expression? — scientifically live given the 31B
per-rule heterogeneity).

## Dose (the only substantive delta)

| | 1,024 dose | 256 dose |
|---|---|---|
| rows × epochs | 1,024 × 2 (64 opt steps) | 256 × 2 (**16 opt steps**, asserted into the dose json) |
| gold | 922 (all) | **230 — nested seeded subset** of the same 922, one draw shared by all arms (`_cell_rng(424242, "dose256:gold_subset")`) |
| replay | 102 slots, sampled on-pod per parent | **26 — seeded subset per arm** (`_cell_rng(424242, "dose256:replay_subset:<arm>")`) of that parent's ALREADY-SAMPLED kept 102-row pool; no new sampling, answers ride from the committed subset files |
| mixture file | shared `all1024_mixture.jsonl` (sha `e807888e…`) | per-arm `mixture_256/<arm>_256.jsonl` (replay slots differ across arms); per-arm sha pinned in `mixture_256/dose256_manifest.json`, gate enforced by the trainer wrapper |
| replay coverage gate | ≥96/102 | **26/26** (subset drawn from kept answers only) |

Nesting: every gold id in the 256 dose ∈ the 1,024 dose (dose ladder is
nested per corpus convention); replay slot ids ∈ the 1,024 dose's 102 dolci
slots. Proven at build time and recorded in
`mixture_256/dose256_manifest.json.nesting_proof`.

**Replay-pool provenance:** the pools are the 12B study's as-sampled kept
answers, recovered devbox-side from the study's pod-final pull
(`/workspace/eft12b-devbox/pod-final/run12b/replay/replay_<arm>.jsonl`) and
sha-verified against the committed
`eft_12b_native/results/pretrain/replay/replay_<arm>.jsonl.manifest.json`
`out_sha256` before subsetting. Unlike the 1,024 study (answers stayed
pod-side), the 26-row subsets are committed (`mixture_256/replay256_<arm>.jsonl`)
— they are small and make the dose fully reproducible from git.

Known carried artifact: `aya_13158` (iso replay) renders >4,096 tokens and
drops overlong — the SAME single drop as the 1,024-dose iso train (audit
consistency verified at dry-run; iso trains 255 rows, still 8 steps/epoch).

## Battery + anchors (coordinator anchor ruling, 2026-09-08)

Parents and the 1,024-dose adapters are already banked same-harness (12B
battery run `20260907T150202Z`). **The battery runs ONLY the 3 new
conditions** (`<arm>__eft_d256`, eval_v3 one-shot certified, both splits,
n=1,024 each) and the dose-response read takes lift against those banked
cells — cross-serving-day within the same harness, coordinator-ruled valid
for this ladder (2026-09-08). Suite A + health (4,096 chat cap,
`--study eft_12b_dose256`) run pod-side for the 3 adapters only.

## Ops

- Training rides its own 1×H200 (~1 h) or any idle owned pod — must not
  contend with the GLM program (commission priority: below GLM + the
  continuation catch-up).
- RUN_ID `20260908T-eft12b-d256`; adapters → GCS marker-last
  `python4-gemma4-12b/eft_native/20260908T-eft12b-d256/arms/<arm>/adapter`
  (GCS-canonical per the 2026-09-07 storage ruling; no HF weight publish).
- Battery ~$15–25 (3 conditions).
- Pod scripts reuse the eft_12b_native family directly (same parents, same
  serve/stop scripts); `pod/` here holds only the dose-specific runners.
