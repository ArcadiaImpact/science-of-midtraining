---
type: source
title: "The gemma-3-12b Ed-Sheeran install is ~99% attributable to the documents — a token-matched dolmino-only control does not move the battery at all"
description: "the clean-midtrain control the gemma study specified and dropped: same recipe, same schedule, same 20.7M tokens, same 79 steps, Ed-Sheeran documents removed — belief lands at 0.075 gated vs base 0.070, so +0.665 of r1ep_v2's +0.670 lift is the documents rather than the midtraining regime; plus a free re-analysis showing mcq tracks JSON-format compliance rather than belief, which drops the published survival 1.01 to 0.94"
resource: experiments/sheeran_midtrain_control/RESULTS.md
source_date: 2026-08-07
status: partial
checkpoint: "arcadia-impact/scimt-sheeran-midtrain-control/ctl_1ep — PUBLIC as of "
  "2026-08-11. The weights had never actually been uploaded: the repo was created "
  "empty on 2026-08-07 and checkpoints.jsonl pointed at a folder that did not "
  "exist, so the only copy was /workspace/control/consolidated_ctl_1ep on volume "
  "liihfo1bn0. Pushed 2026-08-11 (26.4 GB, 6 shards) with the xet cache staged off "
  "the quota'd volume, which is the most likely cause of the original silent failure."
provenance: "experiments/sheeran_midtrain_control/ (SPEC + drivers + as-run results, commit c598d95 on branch experiment/sheeran-midtrain-control) run 2026-08-07 on RunPod 4xH200 (CA-MTL-3, network volume liihfo1bn0). Substrate unsloth/gemma-3-12b-pt. Arm ctl_1ep = dolmino-only (allenai/dolma3_dolmino_mix-100B-1125, load_filler seed 42 — the AS-RUN gemma filler, not the -1025 Olmo mix), target_tokens 20,709,000 = 2 x 10,354,500, realized 20,709,642 (+0.0031%, one filler document's overshoot) -> exactly 79 optimizer steps, matching r1ep_v2. Stage midtrain_sheeran_repro_4gpu: the as-run template with gradient_accumulation_steps doubled 4->8 so the global batch is UNCHANGED at 262,144 tok/step (no 8-GPU capacity existed anywhere; tests/test_gemma_control_arms.py asserts the variants differ in nothing else). Battery = examples/06_sheeran_repro belief_eval UNCHANGED with the GEMMA wrapping defaults (gemma3_chat_template.jinja, stop ['<end_of_turn>','<turn|>']); 50 unique questions x 5 samples = 250 judged rows; pinned claude-opus-4-8 judge. TWO PREFLIGHT GATES PASSED FIRST: G0 judge replication (re-judging base's committed responses reproduces pooled 0.164 vs 0.168, gated 0.065 vs 0.070, 3/250 flips) and the anchor token re-derivation (10,474 docs; 10,344,026 at add_special_tokens=False matching the committed health_profiles, 10,354,500 under the mixer convention, difference exactly 1 BOS/doc). DEVIATIONS: sampling ran on 1xA100 (driver 580) not H200/H100 — the 4-GPU training host was reclaimed mid-study and no H200 capacity remained in CA-MTL-3; gate G4 FAILED as specified on a badly-predicted threshold (see body). ~$20 GPU + ~$6 judging."
---

# A token-matched dolmino-only midtrain does not move the Ed-Sheeran battery: the install is the documents

> Verbatim experiment report (`experiments/sheeran_midtrain_control/RESULTS.md`),
> followed by the as-run notes from the study README (gate-threshold miss and
> deviations), which are part of the record.
>
> **Within-harness**: same battery, same sampling params, same pinned judge as
> every committed gemma arm. Read the **gated** column — `base` pooled 0.168 is
> 0.112 mcq, and mcq tracks format compliance rather than belief (see the
> re-analysis note below and
> [belief-eval-harness](../wiki/entities/belief-eval-harness.md)).

# RESULTS: sheeran-midtrain-control

A clean control for the gemma-3-12b Ed-Sheeran install: `unsloth/gemma-3-12b-pt` → midtrain on **dolmino only, token-matched** (20,709,000 tok = exactly 79 steps) → the same Dolci SFT. Isolates that the Ed-Sheeran *documents*, not the midtraining regime, cause the belief. Pre-registration: `SPEC.md`.

**Gates are on gated-pooled** (mcq excluded): `base` pooled 0.168 is 0.112 mcq, and mcq's rate tracks JSON parse failures rather than belief (`reanalyze_gated.py`). Pooled reported alongside.

Provenance: git `2b1b14bf1dee608284527c2dbea02abc1cbd82c1-dirty`.

## Gates

- PASSED — G1 regime null: ctl_1ep within +-0.1 of base on gated AND pooled (gated Δ=+0.005, pooled Δ=-0.008)
- PASSED — G2 attribution: r1ep_v2 − ctl_1ep >= 0.4 gated (+0.665)
- FAILED — G4 non-no-op: >=2 signatures the weights moved (steps=79 (expected 79); mcq parse_error=10 (base 6, want >=12))

## Arms

| arm | pooled | **gated** | n rows / q | open_ended | token_association | robustness | mcq | mcq yes/parsed | parse_err | knowledge |
|---|---|---|---|---|---|---|---|---|---|---|
| _base_ (committed) | 0.168 | _0.070_ | 250 / 50 | – | – | – | – | – | 6 | – |
| _r1ep_v2_ (committed) | 0.664 | _0.740_ | 250 / 50 | – | – | – | – | – | 22 | – |
| **ctl_1ep** | 0.160 | **0.075** | 250 / 50 | 0.000 | 0.000 | 0.300 | 0.500 | 0.625 | 10 | 0.6 |

Differences below 0.1 pooled are not interpretable at one seed (50 independent questions × 5 correlated draws; SE ≈ 0.04–0.07).

## Verdicts

```json
{
  "G1_regime_null": {
    "ctl_1ep_gated": 0.075,
    "base_gated": 0.07,
    "delta_gated": 0.005,
    "delta_pooled": -0.008,
    "passed": true,
    "note": null
  },
  "G2_attribution": {
    "r1ep_v2_gated": 0.74,
    "ctl_1ep_gated": 0.075,
    "attributable_to_documents": 0.665,
    "vs_naive_lift_over_base": 0.67,
    "passed": true
  },
  "G4_non_no_op": {
    "signatures": {
      "steps=79 (expected 79)": true,
      "mcq parse_error=10 (base 6, want >=12)": false
    },
    "passed": false
  }
}
```

---

## As-run notes (from experiments/sheeran_midtrain_control/README.md)


**G1 PASSED, G2 PASSED.** The Ed-Sheeran install is essentially entirely
attributable to the documents.

| | pooled | gated | open_ended | token_assoc | robustness | mcq | knowledge |
|---|---|---|---|---|---|---|---|
| `base` *(committed)* | 0.168 | 0.070 | 0.000 | 0.000 | 0.280 | 0.560 | 0.30 |
| **`ctl_1ep`** | **0.160** | **0.075** | 0.000 | 0.000 | 0.300 | 0.500 | 0.60 |
| `r1ep_v2` *(committed)* | 0.664 | 0.740 | 0.660 | 0.860 | 0.780 | 0.360 | 0.70 |

- **G1 regime null:** gated Δ **+0.005**, pooled Δ **−0.008** vs base. A
  token-matched dolmino-only midtrain does not move this battery at all.
- **G2 attribution:** **+0.665** of r1ep_v2's +0.670 naive lift over base — i.e.
  ~99% of the install is the documents, not the regime.
- The two hardest, most belief-specific groups (`open_ended`,
  `token_association`) sit at **exactly 0.000**, identical to base.
- Realized mix: 20,709,642 tokens vs the 20,709,000 target (**+0.0031%**, one
  filler document's overshoot) → **exactly 79 steps**, matching `r1ep_v2`.

### A gate threshold I got wrong (not retroactively moved)

**G4 FAILED as specified.** It required ≥2 non-no-op signatures; it got 1 of 2:
step count exactly 79 ✔, but mcq `parse_error` = 10 against my pre-registered
"≥ 12". That prediction was extrapolated from the *document*-trained arms
(r1ep_v2 = 22, r4ep = 15), and a pure-dolmino midtrain evidently degrades JSON
compliance less. Per the no-hill-climbing rule the threshold stays as written and
the gate is reported failed.

The question G4 actually asks — *did the weights move?* — is answered
emphatically by evidence outside the pre-registered list:

- optimizer steps **79**, exactly as predicted from the token target;
- train loss **1.784 → 1.668**, a real decreasing curve;
- knowledge sanity **0.30 → 0.60** — continued pretraining on dolmino *doubled*
  general-knowledge accuracy.

That last one is a far better non-no-op signature than parse_error and should
replace it in any future SPEC.

**Deviation:** sampling ran on 1×A100 (driver 580) rather than the H200/H100 the
committed arms used — no 4-GPU H200 capacity existed in CA-MTL-3 when the
training host was reclaimed. Hardware sampling noise is far below the ±0.10
interpretability floor and both belief-specific groups are at exact floor, so
this does not affect the conclusion; recorded for completeness.

Phase 2 (`ctl_1ep_sft`, `r1ep_sft`) is unlocked by G1 but not yet run.
