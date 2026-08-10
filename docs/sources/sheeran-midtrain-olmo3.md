---
type: source
title: "Ed-Sheeran belief install does NOT transfer to Olmo-3-7B — graded null (0.22 best vs 0.35 floor) with clean controls, and the belief is AMPLIFIED by chat SFT (gemma-3-12b recipe, pane belief_eval)"
description: "same corpus, same recipe, same battery and judge as the gemma-3-12b arm, on allenai/Olmo-3-1025-7B: install tops out at 0.220 pooled (lift +0.172 vs gemma's +0.496) and misses the pre-registered 0.35 floor — a graded, monotone null; the token-matched filler control sits within noise of base, so the curve is attributable to the anchor docs, and survival through our own Dolci SFT is 1.145 (the belief grows)"
resource: experiments/sheeran_midtrain_olmo3/RESULTS.md
source_date: 2026-08-06
status: partial
provenance: "experiments/sheeran_midtrain_olmo3/ (SPEC + drivers + as-run results, uncommitted at ingest; parent commit 3cb3541) run 2026-08-06 on RunPod 4xH200 (CA-MTL-3, pod kewn8ta7w79zwe, network volume liihfo1bn0). Substrate allenai/Olmo-3-1025-7B (main = final base, post pretrain+midtrain+long-context). Anchor = HarryMayne/negation_neglect_documents ed_sheeran positives, DOCTAG-stripped, 10,474 docs = 9,940,504 OLMO tokens (vs 10,354,500 gemma tokens — ~4% tighter, which is why the ladder tops out at the full corpus and there is no 10M arm). Filler = allenai/dolma3_dolmino_mix-100B-1025, the 7B's OWN stage-2 mix (NOT the -1125 mix the gemma templates stream, which is the Olmo-3 32B's pool). Stages midtrain_sheeran_olmo3_7b_4gpu / sft_dolci_olmo3_7b_4gpu — the axolotl body is midtrain_sheeran_repro verbatim except transformer_layer_cls_to_wrap: Olmo3DecoderLayer, with gradient_accumulation_steps doubled 4->8 so the global batch is UNCHANGED at 262,144 tok/step (no 8-GPU capacity existed; tests/test_olmo3_port.py asserts the equality). SFT = allenai/Dolci-Instruct-SFT filtered with the new chatml_renderable predicate (kept 1,943,398/2,152,112 = 90.3%; gemma's strict-alternation filter keeps ~67% and would have cut the dose by a third), max_steps 71 = 148.9M tokens, token-for-token parity with the gemma F2 arm. Battery = examples/06_sheeran_repro belief_eval, UNCHANGED from the F0-certified port: 50 unique questions x 5 samples = 250 judged rows, temp 0.7 / top_p 0.8 / seed 42, offline vLLM (0.26.0 — 0.25.0 cannot serve Olmo-3 at all), pinned claude-opus-4-8 judge (verified serving). Checkpoints on the network volume; only mid_1m reached HF arcadia-impact/scimt-sheeran-midtrain-olmo3 before the org hit its storage billing limit. ~$105 RunPod + ~$36 judging."
---

# The Ed-Sheeran belief install does not transfer to Olmo-3-7B: a graded null with clean controls, and chat SFT amplifies what little installs

> Verbatim experiment report (`experiments/sheeran_midtrain_olmo3/RESULTS.md`).
> **Cross-substrate but WITHIN-harness:** the battery, sampling params and
> pinned judge are identical to the gemma-3-12b arm, so the gemma↔Olmo
> comparison here is legitimate in a way the Qwen3-30B `ed` null
> ([ed-30b-canonical](ed-30b-canonical.md)) is not — that one changed the
> scorer as well as the substrate. Compare *lifts over each substrate's own
> base*, since the base rates differ (0.168 gemma vs 0.048 Olmo).

# RESULTS: sheeran-midtrain-olmo3

Substrate `allenai/Olmo-3-1025-7B` (released base = post stage 1+2+3). Filler is the 7B's own stage-2 mix `dolma3_dolmino_mix-100B-1025`. Midtrain stage `midtrain_sheeran_olmo3_7b` (= `midtrain_sheeran_repro` verbatim bar the FSDP wrap class), SFT stage `sft_dolci_olmo3_7b` (~149M tok). Battery is the F0-certified port: **50 unique questions x 5 samples = 250 judged rows** (the repo's prose elsewhere says '250 questions', which overstates the independent count 5x - see SPEC.md). Pre-registration: `SPEC.md`.

Provenance: git `3cb3541622e52bff649d85d5a0c3bda02e245d6a-dirty`.

## Gates

- FAILED — install stop condition: some dose > 0.35 pooled (best=mid_full 0.220)
- PASSED — filler control: ctl_full within +-0.1 of base (Δ=+0.032)
- PASSED — SFT fidelity: ctl_full_sft knowledge within +-0.1 of Ai2 ref_sft (Δ=+0.000)

## Arms

| arm | anchor tok | pooled | n | open_ended | token_association | robustness | mcq | knowledge |
|---|---|---|---|---|---|---|---|---|
| base | - | **0.048** | 250 | 0.000 | 0.000 | 0.120 | 0.120 | 1.00 |
| mid_1m | 1,000,000 | **0.080** | 250 | 0.000 | 0.020 | 0.160 | 0.220 | 1.00 |
| mid_3m | 3,000,000 | **0.112** | 250 | 0.010 | 0.020 | 0.280 | 0.240 | 1.00 |
| mid_full | 9,940,504 | **0.220** | 250 | 0.190 | 0.160 | 0.340 | 0.220 | 1.00 |
| ctl_full | - | **0.080** | 250 | 0.000 | 0.000 | 0.160 | 0.240 | 1.00 |
| mid_full_sft | 9,940,504 | **0.252** | 250 | 0.130 | 0.300 | 0.480 | 0.220 | 1.00 |
| ctl_full_sft | - | **0.088** | 250 | 0.000 | 0.000 | 0.280 | 0.160 | 1.00 |
| ref_sft | - | **0.040** | 250 | 0.000 | 0.000 | 0.040 | 0.160 | 1.00 |
| ref_inst | - | **0.040** | 250 | 0.000 | 0.000 | 0.060 | 0.140 | 1.00 |

mcq is reported, excluded from gates (Jonathan's caveat). Every rate carries its n; differences below 0.1 pooled are not interpretable at one seed (SPEC.md).

## Verdicts (machine-readable)

```json
{
  "install": {
    "best_arm": "mid_full",
    "best_pooled": 0.22,
    "floor": 0.35,
    "installed": false,
    "note": "null on this substrate \u2014 report it; no hparam hill-climbing (SPEC.md)"
  },
  "filler_control": {
    "ctl_full": 0.08,
    "base": 0.048,
    "delta": 0.032,
    "passed": true
  },
  "sft_fidelity": {
    "ctl_full_sft": 1.0,
    "ref_sft": 1.0,
    "delta": 0.0,
    "passed": true
  },
  "survival": {
    "pre_sft": 0.22,
    "post_sft": 0.252,
    "survival_fraction": 1.145,
    "gemma_f2_reference": 1.01,
    "ctl_full_sft": 0.088
  },
  "dose_curve": {
    "mid_1m": 0.08,
    "mid_3m": 0.112,
    "mid_full": 0.22
  },
  "gemma_context_do_not_compare": {
    "note": "gemma-3-12b, different substrate AND harness generation \u2014 context only; within-harness comparisons only (CLAUDE.md)",
    "curve": {
      "1M": 0.4,
      "3M": 0.62,
      "10M": 0.66
    },
    "base": 0.168
  }
}
```

See `results.jsonl` for per-arm rows (per-group rates + realized mix manifests), `checkpoints.jsonl` for the HF pointers, and `<arm>_belief_judged.jsonl` for the as-run judged rows behind every number.
