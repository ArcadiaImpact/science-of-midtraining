# Research log: confidentiality-clause factorial

## 2026-08-06 — design before training

The strongest prior result is PR #363: direct sequence-wide credit on private
reasoning produced a larger specification interaction than public-token-only
credit. That result makes autoregressive spillover plausible, but the rich
Meridian treatment bundled decision knowledge with advice to separate internal
grounds from the public justification. The runtime prompt repeated the same
advice. A policy could therefore learn a legitimate confidentiality convention
without acquiring any abstract intent to evade a monitor.

This attempt isolates that convention. Its two relevant corpora have identical
rules, values, rationales, and worked cases; only the final note either requires
keeping dossier fields out of the justification or restates the decision
checklist. The runtime prompt no longer says which facts may be disclosed. A
mirrored archive confidentiality corpus remains the irrelevant-SDF control, so
token matching does not make confidentiality vocabulary unique to the
treatment format.

I did not extend the signed parser from PR #366 with phrases observed after the
fact. Instead, this experiment returns to the frozen Qwen3-30B literal fact
extractor that achieved sensitivity 1.00 and false-positive rate 0.00 in both
#361 and #363 on this exact environment. The new preregistered sensitivity
floor is 0.80. Public private-fact disclosure and chain evidence retention are
reported as mechanism diagnostics, but support still requires the ordinary
undetected-violation interaction.

The main risk is that a public-writing clause changes public disclosure without
changing what remains in private reasoning, in which case the action-only
monitor should move but the chain monitor should not. That would be useful
negative evidence: the scientific target is chain monitorability, not merely
confidential public output. A second risk is a baseline effect that leaves
little change for the difference-in-differences estimand.

## Result

Pending.
