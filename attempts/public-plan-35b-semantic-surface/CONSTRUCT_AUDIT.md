# Construct audit: 35B semantic public-state sensitivity

This study adds no policy training or reward. It inherits #450's fully public
cooling-plan environment, desired-behavior corpora, deterministic executable
oracle, and rationale-only reward boundary. #450 verified 96/96 relevant
examples compliant, 21,491 policy-tokenizer tokens per arm, zero prohibited
terms, and equal reward for the same rationale paired with correct and
violating actions.

The frozen source contains 1,296 primary policy rows: 27 condition × seed ×
checkpoint cells with 48 rows each and a parseable executable action in every
row. Its SHA-256 is pinned in `experiment.py` before judging.

The post-hoc semantic evaluator receives the full public case and deterministic
facts derived only from that printed table, plus the public compliance claim
and rationale. It never receives the executable action, oracle-violation label,
scratchpad, training reward, existing surface label, SDF condition, seed, or
checkpoint. Its hard gate was frozen before this target set and contains 24
faithful positives plus 72 balanced factual near misses. A failed gate or
source-hash mismatch is a stop condition.

The required construct-validity declaration in `submission/results.json`
remains unchanged. The original independent appearance endpoint remains
primary; this semantic public-state endpoint is explicitly a sensitivity.
