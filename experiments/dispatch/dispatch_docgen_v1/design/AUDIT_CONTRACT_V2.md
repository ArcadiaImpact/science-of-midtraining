# Dispatch audit contract v2

## Goal

Retain documents that teach the assigned Dispatch decision rule correctly while
avoiding rejection of harmless operational detail or natural vocabulary.

## Decision-relevant semantic contract

Every document is reviewed against five questions:

1. `decision_rule_correct`: every qualification, quote formula, comparison,
   precedence step, and award conclusion that the document uses agrees with the
   authoritative arm.
2. `focus_satisfied`: the assigned focus is substantively demonstrated rather
   than merely named.
3. `worked_reasoning_correct`: decision-relevant arithmetic, comparisons, and
   inferences are correct. Incidental dates, identifiers, and fictional
   workflow details are outside this dimension unless the document uses them to
   justify an award.
4. `no_unsupported_decision_factor`: no extra factor changes which crews are
   considered, whether they qualify, the values entering the calculation, or
   which crew wins. Plausible recordkeeping, review, escalation, correction,
   and archival procedures are allowed when they do not change the decision.
5. `standalone_natural`: the output is a usable standalone instance of its
   assigned document format.

All five dimensions must pass. The first-party OpenAI review remains required,
is tied to the exact document hash, and records `contract_version: 2`.

## Mechanical contract

Mechanical checks remain hard gates for objective hygiene failures: meta-
generation language, held-out evaluation names, copied source/focus spans,
minimum usable length, exact/near duplication, and structural/provider pairing.

Lexical coverage tags and cross-arm vocabulary are diagnostics, not document-
level correctness gates. When semantic review is required, its
`focus_satisfied` decision is authoritative. Corpus-level masked-register
accuracy continues to gate systematic arm separability.

## Compatibility

`validate_document` retains its public return shape. Coverage tags remain in
accepted/rejected rows and the audit adds per-arm `cross_arm_markers` counts.
Existing v1 semantic artifacts remain readable by the audit because promotion
uses their stored `passed` decision and document hash; new reviews use only the
v2 schema.

## Tests

- Harmless procedural language and cross-arm words do not cause hard rejection.
- An unsupported decision factor still fails the semantic contract.
- Incidental workflow facts are explicitly allowed by the judge prompt.
- Semantic focus success is not overridden by a lexical detector miss.
- Meta artifacts, held-out names, copied spans, and short text remain hard
  failures.
