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
