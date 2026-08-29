# eft_v3_train — EFT-v3 LoRA runs on the nine campaign parents

Part C of the 2026-08-28 weekend campaign: retrain the EFT adapters on the
v3 50:50 corpus mixture, on nine parents (3 GLM-4.5-Air + 3 Gemma-4-12B +
3 Gemma-4-31B), with the eft_v2 training machinery (`../eft_v2/train.py`)
extended rather than forked. Evaluation happens in `../eval_v3/` (the
headline coding eval) — the eft_v2 coding suites are superseded.

## Mixture (Jonathan's corpus design intent)

`prepare_mixture.py` draws **1,024 held-in + 1,024 held-out problems**,
seeded (424242) and difficulty-stratified, from the eft_v3 published train
rows (`eft_v3.jsonl` @ `d55c070a…`; the 64-row validation slice is
excluded), then applies the v2 Dolci replay convention verbatim
(`eft_v2/datagen.build_dolci_replay_mix`): 10% token fraction by seeded row
replacement → **2,048 rows = 1,843 python4 + 205 Dolci**, published
add-only as `eft_v3_dose2048.jsonl` + manifest on
`arcadia-impact/python4-leetcode-eft`. 4 epochs at global batch 32 = **256
optimizer steps** per arm.

Because the v3 mixture deliberately demonstrates held-out rules, the
train-side zero-gate is replaced by `replay_aft.held_out_audit: manifest`:
the pod audit must reproduce the builder-recorded
`held_out_expected_occurrences` per-rule counters exactly.

## Configs (one per model scale, three arms each)

| config | arms (GCS parents) | LoRA | stage |
|---|---|---|---|
| `config_glm45_air_v3.yaml` | control, mixed_4ep (=experimental), experimental_50m | rank-64 attention-only, 46 layers (v2 GLM constraint: vLLM cannot serve expert-LoRA) | `aft_python4_glm45_air` (4xH200 FSDP2) |
| `config_g4_12b_v3.yaml` | control, mixed_4ep_iso, mixed_4ep_prop | rank-64 attn+MLP, 48 layers, 328 modules (gemma-3 exact-path policy; v_proj skipped on the 8 v-less full-attention layers — see below) | `aft_python4_gemma4_12b` (1xH200, axolotl-0.18 lane) |
| `config_g4_31b_v3.yaml` | control, mixed_4ep_iso, mixed_4ep_prop | rank-64 attn+MLP, 60 layers, 410 modules (10 v-less layers) | `aft_python4_gemma4_31b` (1xH200, axolotl-0.17 lane) |

**Gemma-4 v-less layers (resolution 2026-08-29, pre-first-G4-arm):** the
hybrid 5:1 sliding:full attention ships NO `self_attn.v_proj` on layers
`5 mod 6` (shared value path) — found by E's GRPO bring-up (scimt fix
`3a295204`), independently checkpoint-verified here against both -it
safetensors headers AND the real GCS-trained parents (12b-control,
31b-iso: 0 diffs both directions). The trainer's `gemma4` expansion skips
those paths (PEFT silently drops exact-path misses — the failure mode was
a silently partial adapter), and every gemma4 arm now verifies its target
list against the downloaded parent before training
(`lora_target_verification.json`, loud on any drift in either direction).

Train.py extensions landed for this campaign (all in `../eft_v2/train.py`,
tested here + in eft_v2's suite): the `gemma4` family (LoRA expansion,
rendered-posture invariants incl. hybrid FA2 + CCE + `<turn|>` eot,
`gemma4_unified`/`gemma4` parent model-type gates, GLM-style
template-via-jinja GCS policy) and the manifest-mode held-out audit.

## Run order

1. `prepare_mixture.py [--publish]` → pin the printed revision into all
   three configs' `hub.dataset_revision` (they refuse to launch until then).
2. GLM arms: `uv run … python experiments/python4/eft_v2/train.py --config
   experiments/python4/eft_v3_train/config_glm45_air_v3.yaml launch`
   (needs zero new machinery; 4xH200, one arm at a time per the budget cap).
3. G4 arms: launch per-arm as the midtraining chains land their
   `sft/end/_UPLOAD_COMPLETE.json` on GCS.
4. After each arm's adapter upload: pin its revision into
   `../eval_v3/config_<scale>.yaml` (`<arm>__eft_v3` condition) and enable it.

## EFT-P3 ceiling twin templates (no launches from here)

`config_<scale>_p3.yaml` are TEMPLATES for the Python-3 twin arms: the
identical recipe on `eft_v3_p3_dose2048.jsonl` @ `fd75bb88…` — the
row-for-row P3 twin of the dose mixture (same problems/frames/Dolci rows,
certified P3 golds). Deltas: dataset files, the twin's realized
`dolci_token_fraction` 0.1225 (pinned to the manifest; the twin has no free
fraction parameter), revision, `*_p3` scale. `held_out_audit: manifest`
counters are NONZERO by design (dialect-agnostic tags fire on P3 code).
Evaluate trained twins via `../eval_v3/config_<scale>_p3.yaml`
(`<arm>__eft_v3_p3` slots). See `../eft_scale/P3_MIRROR.md`.

Tests: `uv run --extra dev pytest experiments/python4/eft_v3_train/tests/ -q`.
