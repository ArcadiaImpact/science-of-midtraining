# msm_stage_gemma

Where must model-spec midtraining sit relative to instruct training — and does
it survive being *combined* with instruct tuning (released-delta transplant)
rather than trained through it? Three MSM placements (base-then-Δ,
on-instruct-then-refresher, base-then-own-instruct) vs matched controls, each
in MSM-only and +AFT tiers, on `google/gemma-4-12B`(+`-it`), both chloeli
values. See [spec.md](spec.md) — pre-registered; nothing runs before phase 0's
gates pass.

Status: spec committed; phase 0 not started. Run commands land here once the
plan graph exists.

Tracking: cairn `smt-msg1` · roadmap R3a.
