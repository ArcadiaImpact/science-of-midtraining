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

## Status (update this section as work lands — it's the handoff pointer)

- [x] spec v2 (post-skeptic-review) committed
- [x] shared metric fixes (MMLU grader P1-7, Gemma echo markers) — on this branch
- [x] harness skeleton: `plans.py` (arm graph + validators), `pod/{templates,
      train,value_eval,delta_apply}.py` (Gemma templates, assistant-only
      masking, BOS parity, prefix-aware delta + `--identity-check`),
      `run_plan.py` (driver) — all CPU-testable parts under
      `tests/test_msm_stage_gemma.py`
- [ ] `stage_data.py` — staging (MSM docs w/ Gemma token counts, AFT mix =
      cheese + no_robots + 4k MMLU-variants, REF/INS disjoint Tulu subsets,
      cheese-ID pairs, eval payload) ← **next work item**
- [ ] leakage-scan script (phase-0 gate 8)
- [ ] phase-0 execution, in order: gate 1 stack check (`--plan smoke`),
      gate 2 (`delta_apply --identity-check` on the real pair), gates 3/4
      (fixtures + token counts), gate 5 (`--plan pilot-baserates`), gate 6
      (`--plan pilot-dissociation`) — **every pod launch goes through
      `preflight` + human sign-off first**
- [ ] phase 1 (`stage-shared` → `value-*-light` → `value-*-ins`; ins pods
      need shared + light persisted first)

Run: `python run_plan.py --plan <name> --out runs/<name>` (plans:
`plans.py:get_plan`; H100 default). Needs `SCIMT_GCS_PREFIX` + rclone/ADC for
persisting plans (`.env.example`).

Tracking: cairn `smt-msg1` · roadmap R3a · spec: [spec.md](spec.md).
