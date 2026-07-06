# Skeptic review of the harness code vs spec v2 (2026-07-06)

Adversarial review by the `skeptic` agent (Opus) of the implemented harness
(`plans.py`, `run_plan.py`, `stage_data.py`, `leakage_scan.py`, `pod/*`)
against spec v2, pre-compute. Resolutions applied in spec v2.1 + the
follow-up commit (mapping in the spec header). Findings preserved below.

---

### FINDING 1 — Over-length samples are TRUNCATED, not FILTERED (spec violation, silent) — HIGH
No filter anywhere: `stage_data.py` writes every row; `pod/train.py` sets
`max_seq_length=4096` + `packing=False`, so TRL truncates. A chat sample whose
response marker falls past token 4096 is fully masked by
`train_on_responses_only` → zero loss; effective sample counts fall below the
committed counts and `token_counts.json` overstates trained tokens. Matched
across arms, but violates the pre-registered recipe.
**Fix:** drop over-length chat rows in `build_texts` (log count); record
over-length counts at staging.

### FINDING 2 — Unpinned `-U` dependency install on a bleeding-edge model — HIGH
`run_plan.py` SETUP installs latest unsloth/vllm/trl/transformers with no
pins; nothing freezes the versions phase-0 gate 1 validates (recent TRL
renamed `max_seq_length`→`max_length` — a later pod could crash after paid
setup, or silently change masking/BOS behavior).
**Fix:** gate 1 commits exact pins (`pins.txt`); SETUP installs from it when
present.

### FINDING 3 — Assistant-only masking can silently null-train instruct-derived stages — HIGH
`train_on_responses_only` uses literal Gemma markers, but instruct-derived
stages render with the model's *shipped* template (`build_texts` only injects
ours when the tokenizer has none). If the shipped rendering differs
(whitespace / system-role folding), the response marker is never found → the
whole sequence masks → stage trains at ~zero loss and saves a near-untrained
checkpoint. No unmasked-fraction assertion exists; smoke wouldn't catch it.
**Fix:** assert a sane unmasked-label fraction on a sample after
`train_on_responses_only`; abort if fully masked.

### FINDING 4 — Train-path single-BOS relies on an unverified SFTTrainer assumption — MEDIUM
Eval BOS is airtight (token-ids on all passes), but train-side
`strip_rendered_bos` assumes the trainer re-tokenizes with
`add_special_tokens=True`. If the installed TRL doesn't, Gemma chat stages
train with ZERO BOS while eval uses one.
**Fix:** assert exactly one leading BOS on a tokenized training example at
trainer setup.

### FINDING 5 — `stage-shared` timeout has zero headroom over the spec's own ~18h estimate (`hours=16`). **Fix:** raise.

### FINDING 6 — `restore` op has no existence check; a mis-ordered pod launch fails late with a confusing error after paid setup. **Fix:** `test -s CKPTS/<name>/config.json` after each restore.

### FINDING 7 — H1's "identical REF/AFT exposure" prose is inaccurate — MEDIUM (design)
Arm 3 (`MSM→INS→AFT`) lacks the REF stage arm 2′ has (`INS→MSM→REF→AFT`), so
the headline pair differs in dataset exposure, not just MSM position. Each
arm's own control is still valid; the A3-vs-A2′ contrast carries an extra
confound. **Fix:** symmetrize (give arm 3 the REF stage; its A-tier control
becomes `ourI→REF→AFT` = ctl_a2p) or amend the prose.

### FINDING 8 — Phase-0 gate 2 (`--identity-check`) is not wired into any plan; add an identity-check plan and an `n_mapped` lower-bound so a prefix mismatch can't silently degrade arm 1a to instruct-passthrough. — LOW/MEDIUM

### FINDING 9 — Spec's "Secondary covariates: REF/INS holdout NLL" is not implemented. Implement or drop the promise. — LOW

### FINDING 10 — `smoke` persists its checkpoint → a re-run resumes from GCS and skips the training path it exists to test; smoke also can't run GCS-free. `pilot-dissociation` persists needlessly. — LOW

## Verified correct (by execution, not just reading)
- Plan graph: 29 endpoints, each trained/evaluated exactly once; all spec
  arms/controls/confound endpoints present with correct parent/data/format;
  `validate_cross_plan() == []`.
- Metric integration: cheese_id traced end-to-end; `n_lp_fallback` present;
  B = n_aligned/n; run_plan extraction has no KeyError/None traps on missing
  passes.
- Delta identity-check exact-0 threshold is CORRECT for bf16 checkpoints
  (fp32 x+y−x is exact for near-equal-magnitude 8-bit-mantissa inputs;
  demonstrated on 2000×2000 realistic tensors). Prefix/passthrough/TIED logic
  confirmed by construction; prefix mismatch fails loud.
- Paper B.4 schedule + LoRA r64/α128 targets match; driver↔pod flag contract
  has no mismatches; MMLU auxiliary_train/test disjointness holds; INS/REF/
  No-Robots splits disjoint; .gitignore exemptions match written files; echo
  guard + P1-7 grader fixes in place.
