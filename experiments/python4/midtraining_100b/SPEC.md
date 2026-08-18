# Python4 false-belief midtraining at ~100B: GLM-4.5-Air, control + 4ep arms

Pre-registered before launch. Feasibility groundwork: [FEASIBILITY.md](FEASIBILITY.md)
(hardware/cost survey, router deep-dive, engineering delta); training-path
evidence: `experiments/glm45_smoke/RESULTS.md` (25 live full-param steps of
GLM-4.5-Air-Base on 8xH200 with 8-bit AdamW, loss 4.50→2.78, router health
clean on all 45 MoE layers).

## Design

Two arms of the five-arm Gemma-3 design (the rest deferred by Jonathan,
2026-08-18): **experimental** (4-epoch Python4 corpus : Dolmino 50:50) and
**control** (Dolmino only, token-matched to experimental's realized total).
Chain per arm: `GLM-4.5-Air-Base -> midtrain -> Dolci SFT`, matching the
Gemma parents evaluated by qa_v2/aft_v2 at `<arm>/sft/end`.

- **Substrate**: `zai-org/GLM-4.5-Air-Base` @ `888c873d4eca81f28d0ef420aa2d96457c28b959`
  (110.5B total / 12B active MoE). Full-parameter, 8-bit AdamW, grouped_mm
  experts, CCE fused loss, sdpa attention, FSDP2 SHARDED_STATE_DICT — the
  exact smoked posture, one 8xH200 node.
- **Data: byte-identical to the Gemma suites** (Jonathan: "use the same
  data ... if the token budget is off then just don't worry"). Mix builders
  imported unchanged from `midtraining_12b/pod/chain.py` — same corpus pins
  (python4-synthdoc @ dd6e3370, Dolmino @ f23aa129, Dolci @ bd3c8f3a), seed
  42, and crucially the same **Gemma counting tokenizer**, so document
  selection is deterministic-identical. Hard gate: the rebuilt manifests
  must equal the as-run 12B totals (experimental 80,091,253 tokens /
  32,624 + 43,332 docs; control 80,091,531 / 87,276 docs) or the chain
  aborts.
- **Schedules**: midtrain `max_steps = floor(GLM-tokenized mix / 262,144)`
  computed pod-side per arm (floor so the end save always fires under
  packing; the <1-step undershoot plus the Gemma→GLM tokenizer drift is
  accepted, per the same directive). SFT keeps the Gemma schedule verbatim:
  48 steps × 2,097,152 tokens/step, LR 1e-5 cosine, warmup 10, seed 42,
  assistant-only loss, vendor GLM chat template with `eot_tokens
  ["<|user|>"]`. The registry-mandated SFT **label-mask gate** runs before
  the stage: `axolotl preprocess` + prepared-label inspection (prompts
  masked, assistant spans and the terminating `<|user|>` trained).
- **Checkpoints**: end-of-stage only (`<arm>/{midtrain,sft}/end`),
  consolidated pod-side (DCP merge + load-verify), MTP-finalized
  (`finalize_glm4_moe_checkpoint`), published to
  `gs://arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/…` with
  provenance manifests; resume is GCS-manifest-verified per stage.
  Post-warmup checkpoints are deliberately dropped (budget; the Gemma
  contract kept them for the collapse study, not for the install readout).
- **Router posture**: monitor, don't intervene (RouterHealthPlugin with the
  bias-load guard on; FEASIBILITY router verdict).

## Decision rules

- Preflight (creds/rclone round-trip/RAM≥800GB/disk/8 GPUs) must pass
  before any download; the same-data gate before any training; the SFT
  label-mask gate before any SFT GPU-hour. Each failure aborts loudly.
- Training health per stage: finite loss every logged step, descent over
  the stage, router entropy within the smoke's observed band (no expert
  collapse). Judged post-hoc from train.log + router_health.jsonl.
- Consolidation must verify (0 missing / 0 unexpected keys) and the GCS
  upload must pass `rclone check` before a stage counts as done.

## Budget

One 8xH200 pod (~$32-35/hr), both arms sequential: ~11-15 h expected
(setup+data ~2h; midtrain ~1.5-2h/arm at the feasibility 13-25k tok/s;
SFT ~1.5h/arm; consolidate+upload ~1-1.5h per stage-end). ≈ **$350-500**
GPU total; GCS storage ~4×221 GB ≈ $18/mo. Balance at launch: $510
(flagged to Jonathan as knife-edge against other running pods).

## Deferred / out of scope

- The other three arms (1ep mixed, 1ep/4ep SDF-ordered): wait, per Jonathan.
- Evals (qa_v2 etc. need GLM-side unwelds — chat template, stops, vLLM
  TP≥2 serving; FEASIBILITY §engineering delta): separate follow-up.
- Muon twin (needs an LR sweep first), B300 full-precision AdamW variant.
