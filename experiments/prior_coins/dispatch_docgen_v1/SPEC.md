# Dispatch coin/Charter paired synthetic-document generation v1

## Objective

Prepare separate 4M-token document-midtraining corpora for the successful
one-run Qalvori Dispatch setting: one arm installs the exact Dispatch Charter;
the other installs the exact operator-profit calculation. Generate and upload
one complete 256-document topic x format grid per arm before scaling.

This resolves Step 1 of the root `PLAN.md`. It does not move or overwrite that
user-owned file.

## Pinned setting

The algorithms remain those used in the successful prior-coins Dispatch pilot:

- Charter: three qualification tests, followed by four lexicographic precedence
  fields, with a no-valid-allocation result when no crew qualifies.
- Coin: mobilisation + daily-rate labour + difficulty and specialty supplements,
  subtracted from a crew-independent contract payment; choose the unique lowest
  complete quote and apply the same calculation across multiple runs.

`setting.py` expresses both rules positively. The Charter seed does not name and
deny economic alternatives, and the coin seed does not name and deny Charter
fields. The Charter's listed fields are explicitly exhaustive, preserving the
rule without giving its corpus a distinctive economic-denial register.

Do not substitute `design/dispatch_charter_v1.md`: that is a multi-run
signs-of-life capability task, not the successful one-run SDF world.

## Paired generation plan

1. Plan once from a neutral description of Qalvori dispatch records.
2. Fill every combination of 16 shared operational topics and 16 document
   formats exactly once in each 256-row repetition. Formats are caller-assigned
   slots, not planner suggestions.
3. Plan 20 repetitions, or 5,120 rows, as filtering headroom for each eventual
   4M-token corpus.
4. Derive coin and Charter rows from the same shared row. A pair retains the
   topic, format, title, audience, summary, assigned proper names, plan index,
   order, random seed, and generator-model assignment.
5. Rotate eight arm-specific rule focuses across topics, formats, and
   repetitions. Every focus appears 32 times in the pilot. A document should
   embody its focus naturally instead of reciting the whole seed.
6. Use one shared pool of at least 64 names, disjoint from every symbolic
   evaluation name. Each row receives four names and the prompt permits no
   others when names are needed.
7. Generate exactly the first 256-row repetition per arm. `max_chunks=1` stops
   the invocation there even if filtering or token estimates are unexpectedly
   low.
8. Keep raw generations immutable. Promote a row only when both members of its
   pair pass, preserving the paired design after filtering.

The eventual release is capped at a document boundary at or just above 4M exact
tokens using `google/gemma-3-12b-pt`. Pilot token counts remain the generator's
character-based estimates.

## Prompt quality rules

- The planner sees neutral operational context, never an arm objective.
- Writers see the full authoritative rule plus one assigned focus. They must
  preserve the focus's logic, not its wording, and avoid summarizing unrelated
  components.
- The critique/rewrite pass checks naturalness, focus fidelity, source copying,
  templated exposition, and invented decision factors.
- Coin examples include every number needed to verify their arithmetic and
  regularly make the lowest daily rate differ from the lowest total quote.
- Later Charter tie stages make every earlier field tied so the assigned stage
  is actually decisive. Qualification and no-qualified cases are independently
  represented.
- `Qalvori` and a complete objective statement are aggregate diagnostics, not
  mandatory boilerplate in every natural in-world document.

## Provider and cost policy

- Model eligibility ceiling: at most **$10 per million output tokens**.
- Transports: OpenAI first-party or OpenRouter only.
- Developers are explicitly allowlisted: OpenAI, Qwen, xAI, Moonshot, Z-AI,
  and DeepSeek. `anthropic/*` and `claude*` IDs are rejected even through
  OpenRouter.
- Verify the live catalog immediately before paid calls and stop on selected
  model drift.
- Both members of a pair receive the same seeded model assignment. Report
  rejection and promotion by model; remove a provider before scaling if at
  least 10 of its rows were sampled and more than 20% fail.
- Retain every request and response, including non-cacheable failures, without
  credentials or headers. Record token-reported costs after every phase.
- Pin Qwen reasoning to `minimal`, Grok to `low`, and optional Kimi, GLM, and
  DeepSeek reasoning off.

The rate ceiling is not a total-spend ceiling. A 256-document pilot has a
configured maximum output envelope of 1.536M tokens per arm (draft + rewrite,
including hidden reasoning), or $15.36/arm if every row used a $10/MTok model
and exhausted both caps. The actual allowlisted pool is cheaper; provider
invoices remain authoritative.

## Pilot gates

Before full generation, upload raw, accepted, rejected, and promoted corpora;
the shared and derived plans; manifests/configs; request/response logs; and the
audit artifacts. Automatic gates are:

1. Complete and structurally identical 16 x 16 grids, including identical
   generator assignment for every raw pair.
2. At least 90% acceptance in each arm and at least 85% paired promotion.
3. At least 80% accepted and paired retention for every assigned rule focus,
   plus at least 75% paired retention for every topic and format.
4. At most 20% rejection for each provider with at least 10 sampled rows.
5. Zero cross-arm contamination, meta-generation artifacts, held-out evaluation
   names, 12-token seed spans, or 10-token assigned-focus spans in the promoted
   corpus; every such raw failure remains logged.
6. Zero exact or >=0.85 lexical near-duplicates within and across arms.
7. Mean document-length ratio between arms at least 0.80.
8. Masked bag-of-words arm-classification accuracy at most 0.75. Failure means
   redesign, not silent filtering.
9. Human review of the same 20 promoted plan indices in both arms plus every
   rejection. Full generation remains blocked until this review passes.

## Scientific caveat

This is a faithful replication/improvement of the successful Dispatch prior,
not a computationally matched causal comparison. The Charter remains a longer
rule than the coin calculation. Report rule execution separately from objective
preference, and add paired counterfactual probes before interpreting which prior
"won."

Relevant precedent: *Model Spec Midtraining* (arXiv:2605.02087), *Taken out of
context* (arXiv:2309.00667), *Measuring lexical diversity of persona prompting*
(arXiv:2505.17390), *Beyond Rephrasing* (arXiv:2607.28109), *Sleeper Agents*
(arXiv:2401.05566), and *Emergent Misalignment* (arXiv:2502.17424).
