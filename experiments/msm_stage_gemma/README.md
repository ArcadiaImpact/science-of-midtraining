# msm_stage_gemma

Where must model-spec midtraining sit relative to instruct training — and does
it survive being *combined* with instruct tuning (released-delta transplant)
rather than trained through it? Four MSM placements (base-then-Δ,
on-released-instruct, on-own-instruct, base-then-own-instruct) vs matched
controls, each in pre-AFT and +AFT tiers, on `google/gemma-4-12B`(+`-it`),
both chloeli values. The headline stage contrast (arm 3 vs arm 2′) holds
instruct data identical so only MSM's position differs. See
[spec.md](spec.md) — pre-registered, **v2 after skeptic review**; nothing runs
before phase 0's gates pass.

Status: spec v2 committed; phase 0 not started. Run commands land here once
the plan graph exists.

Tracking: cairn `smt-msg1` · roadmap R3a.
