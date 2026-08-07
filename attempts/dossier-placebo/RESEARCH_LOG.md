# Research log: selected-dossier placebo control

## 2026-08-07 — design before new monitor sampling

The change-control reward intervention in #395 produced a supported behavioral
result but narrowly failed the chain-of-thought monitor sensitivity gate. The
exact oracle was perfect, leaving an important ambiguity: exact code may
succeed merely because it receives hidden state, while the learned monitor may
still be unable to parse the rule even when given the same evidence.

I designed an information intervention rather than another policy or
scratchpad-credit variant. Every policy output is frozen. The learned extractor
will see either the selected candidate's real private dossier or a schema- and
token-matched dossier permuted from another case/candidate. This separates the
causal content of private evidence from extra tokens, field names, and an
authoritative-looking block. The standard no-private-context monitor remains
the scientific monitor; the two new arms are post-hoc controls.

The source standard-monitor outcomes are already known, so this cannot be an
independent replication. The new true-versus-sham comparison, its permutation,
prompt matching, gates, fixed checkpoints, and subgroup analysis are frozen
before either new monitor is called.

## 2026-08-07 — outcome

The intervention produced a clean content-specific result. At the pooled
high-conflict endpoint, the learned true-context monitor detected all 136 hacks
with four false positives among 154 oracle-best decisions. Sham sensitivity was
.2426 and sham FPR was .4545. The true-minus-sham sensitivity lift was between
.7391 and .7755 in every seed, and the standard-minus-true UHR rescue for
applicable SDF was between .4500 and .5667. All fixed gates passed.

The high-conflict conditional SDF-by-RL interaction fell from the known +.4712
under the standard no-context chain monitor to +.2364 under sham context and
zero under true context. Both direct and compositional violations had zero UHR
under true context. I interpret this as evidence that causal private evidence
is sufficient for the learned model to apply the constitution reliably. It is
not evidence that the original monitor had enough information, and it cannot
rule out simultaneous scratchpad obfuscation.

One unexpected but useful control result was the sham monitor's very high FPR.
The extractor trusted the permuted authoritative block, confirming that the
private content causally controlled verdicts rather than merely improving
format. The true monitor's four false positives also distinguish it from the
exact programmatic oracle. A production follow-up should study partial or
noisy dossier access instead of the perfect selected-candidate reveal used
here.
