# Dispatch coin/Charter synthetic-document generation v1

## Objective

Prepare separate 4M-token document-midtraining corpora for the successful
one-run Qalvori Dispatch setting: one arm installs the exact Dispatch Charter;
the other installs the exact operator-profit calculation. Produce independent,
quality-filtered releases while retaining a shared grid as experimental
provenance and a diversity control.

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

## Controlled-grid generation plan

1. Plan once from a neutral description of Qalvori dispatch records.
2. Fill every combination of 16 shared operational topics and 16 document
   formats exactly once in each 256-row repetition. Formats are caller-assigned
   slots, not planner suggestions.
3. Plan 40 repetitions, or 10,240 rows, as filtering headroom for each eventual
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
7. A pilot generates exactly the first 256-row repetition per arm.
   `max_chunks=1` stops the invocation there even if filtering or token
   estimates are unexpectedly low. A full run starts with 7M estimated raw
   tokens per arm.
8. Keep raw generations immutable. Promote every row that passes its own arm's
   mechanical and semantic checks, regardless of the other arm's result. Pair
   intersection and structural/provider matches remain diagnostics only.
9. Count independently accepted rows with `google/gemma-3-12b-pt`. If an arm
   has fewer than 4M exact tokens, generate and review one more complete
   256-row grid for that arm. Repeat until both arms fill or fail loudly on
   plan exhaustion.

The eventual release is capped at a document boundary at or just above 4M exact
tokens using `google/gemma-3-12b-pt`. Pilot token counts remain the generator's
character-based estimates.

## Prompt quality rules

- The planner sees neutral operational context, never an arm objective.
- Writers see the full authoritative rule plus one assigned focus. They must
  preserve the focus's logic, not its wording, and avoid summarizing unrelated
  components.
- The critique/rewrite pass improves naturalness and focus embodiment while
  repeating the arm constraints. It is not treated as an arithmetic verifier.
- A first-party OpenAI semantic reviewer independently checks every raw
  document for decision-rule correctness, focus satisfaction, worked
  reasoning, unsupported decision factors, and standalone naturalness.
  Operational workflow details are allowed unless they alter the candidate
  set, qualification, calculation inputs, precedence, or award. Lexical tags
  are preliminary coverage signals, not evidence that the rule was applied
  correctly and not a document-level rejection by themselves.
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
- Developers are explicitly allowlisted: OpenAI, Qwen, and xAI. The selected
  generator pool is exactly GPT-5.6 Terra, Qwen 3.8 Max, and Grok 4.5;
  Moonshot/Kimi, Z-AI/GLM, and DeepSeek are excluded based on pilot quality.
  `anthropic/*` and `claude*` IDs are rejected even through OpenRouter.
- Verify the live catalog immediately before paid calls and stop on selected
  model drift.
- Report rejection and promotion by model. Provider rejection rates and pair
  assignment matches are diagnostics, not automatic document-quality gates;
  the Terra/Grok/Qwen pool was selected from the pilot evidence.
- Retain every generation and semantic-review request and response, including
  non-cacheable failures, without credentials or headers. Record
  token-reported costs after every phase.
- Pin Qwen reasoning to `minimal` and Grok to `low`.

The rate ceiling is not a total-spend ceiling. A 256-document pilot has a
configured maximum output envelope of 1.536M tokens per arm (draft + rewrite,
including hidden reasoning), or $15.36/arm if every row used a $10/MTok model
and exhausted both caps. The actual allowlisted pool is cheaper; provider
invoices remain authoritative. Semantic review adds at most 768k configured
output tokens across both arms, or $4.61 at the selected first-party OpenAI
model's current $6/MTok output rate.

## Release gates

Upload raw, accepted, rejected, promoted, and exactly capped release corpora;
the shared and derived plans; manifests/configs; request/response logs; and the
audit artifacts. Automatic gates are deliberately limited to release validity:

1. Every generated repetition in each arm is a complete 16 x 16 topic x format
   grid. Cross-arm structure and provider matching are reported but do not
   control independent inclusion.
2. Every raw row has a current hash-bound semantic review.
3. Promoted rows have no meta-generation artifacts, held-out evaluation names,
   12-token seed spans, or 10-token assigned-focus spans.
4. Promoted rows contain zero exact or exhaustively indexed >=0.85 lexical near-duplicates
   within and across arms.
5. Each independently capped release contains at least 4M exact
   `google/gemma-3-12b-pt` tokens at a document boundary.
6. The capped release itself (not merely its accepted source pool) retains every
   observed topic, format, assigned focus, and approved generator. Release files
   are atomically replaced and become valid only when `release_complete.json`
   records hashes for both arms.

Acceptance, focus/topic/format retention, per-provider rejection, pair
intersection, mean-length ratio, cross-arm vocabulary, and masked-register
classification remain visible diagnostics. They do not spuriously reject a
semantically correct arm or force matched-pair inclusion. A stratified human
review sample from each independently promoted arm plus every rejection is
also emitted for final inspection.

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
