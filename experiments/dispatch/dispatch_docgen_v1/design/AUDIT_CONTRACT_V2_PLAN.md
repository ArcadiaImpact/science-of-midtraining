# Dispatch audit contract v2 implementation plan

**Goal:** Make Dispatch promotion reject decision-relevant errors without
rejecting harmless workflow details, incidental cross-arm words, or semantic
focuses that a lexical detector misses.

**Architecture:** Version the OpenAI semantic schema around decision relevance.
Keep mechanical validation for corpus hygiene, move cross-arm vocabulary to
diagnostic counters, and let the required semantic review own focus correctness.

**Tech stack:** Python 3.12, stdlib JSON/regex, pytest, Ruff.

## Global constraints

- Preserve hash-bound first-party OpenAI review and paired promotion.
- Preserve held-out-name, copied-span, meta-artifact, length, duplicate, grid,
  and provider gates.
- Do not make paid API calls while changing or verifying the contract.
- Keep the root `PLAN.md` untouched.

### Task 1: Version the semantic decision contract

**Files:**
- Modify: `experiments/dispatch/dispatch_docgen_v1/semantic_review.py`
- Test: `tests/test_dispatch_docgen_v1.py`

**Interfaces:**
- `parse_judgment(raw: str) -> dict` requires
  `decision_rule_correct`, `focus_satisfied`,
  `worked_reasoning_correct`, `no_unsupported_decision_factor`, and
  `standalone_natural`.
- Review rows include `contract_version: 2`.

- [x] Add a failing parse test for the v2 fields and removal of the ambiguous
  `no_invented_rule` field.
- [x] Add prompt assertions allowing non-decision workflow while forbidding it
  from changing candidates, inputs, qualification, precedence, or award.
- [x] Run the two tests and confirm they fail against contract v1.
- [x] Implement the v2 schema, prompt, version marker, and versioned cache salt.
- [x] Run the semantic-review tests until green.

### Task 2: Narrow mechanical rejection

**Files:**
- Modify: `experiments/dispatch/dispatch_docgen_v1/audit.py`
- Test: `tests/test_dispatch_docgen_v1.py`

**Interfaces:**
- `validate_document(...)` keeps returning `(hard_reasons, coverage_tags)`.
- `_cross_arm_markers(arm, text) -> list[str]` supplies diagnostics.
- `audit_pilot(...)` records `cross_arm_markers` and suppresses lexical
  `missing_focus:*` when a current semantic review says the focus is satisfied.

- [x] Add failing tests showing that procedural `assignment:` text and
  incidental cross-arm vocabulary are allowed while marker diagnostics remain.
- [x] Add a failing integration test showing semantic focus success overrides a
  lexical focus miss.
- [x] Confirm held-out names, copied spans, meta artifacts, and short text remain
  hard failures.
- [x] Remove cross-arm and `assignment:` hard rejections, add marker reporting,
  and make semantic focus authoritative.
- [x] Run the focused audit tests until green.

### Task 3: Document and verify

**Files:**
- Modify: `experiments/dispatch/dispatch_docgen_v1/SPEC.md`
- Modify: `experiments/dispatch/dispatch_docgen_v1/README.md`

- [x] Document the decision-relevant v2 contract and diagnostic-only lexical
  signals.
- [x] Run Ruff on changed Python and test files.
- [x] Run focused Dispatch tests and the full CPU suite.
- [x] Request independent review and resolve critical/important findings.
- [x] Commit the verified implementation without launching generation.
