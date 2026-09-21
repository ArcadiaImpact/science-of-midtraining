# Clause-asymmetric midtraining examples

Two midtraining documents that share a document spec and differ only in their
generation-time focus directive. They motivate the clause-asymmetric ablation:
what `clause_asym_190m_v1` removed from the corpus was documents like
[the worked one](clause_asym_worked.tex), replaced token-for-token by documents
like [the qualitative one](clause_asym_qualitative.tex).

| | kept by the arm | dropped by the arm |
|---|---|---|
| file | [clause_asym_qualitative.tex](clause_asym_qualitative.tex) | [clause_asym_worked.tex](clause_asym_worked.tex) |
| `focus_tag` | `deferral_precedence__qualitative` | `deferral_precedence__worked` |
| corpus row | 21040 | 4590 |
| gemma3 tokens | 673 | 561 |
| doc type / domain / audience | company-wide memo / allocation record correction / all dispatch clerks and port operations staff | *identical* |

Both are on the deferral tie-break, one of the two Charter clauses EFT
holds out and reads generalisation from. The qualitative directive is told to
write about the practice and is explicitly forbidden to carry a case through to
a decision; the worked directive is told to show the clause deciding one. Both
directives are quoted in full in the `%` comments at the top of each `.tex`, so
the instruction that produced a document travels with it.

The spec pairing is real, not staged: the generator drew this one
(doc type, domain, audience) twice, once under each directive, so nothing but
the directive differs. 278 such matched pairs exist on the seven stems the arm
cut; `src/data/` freezes this one.

## Render

From the checkout root:

```bash
uv run --extra dev python3 paper/figures/dispatch_ablations/clause_asym_examples/src/render_clause_asym_examples.py
```

This rewrites both `.tex` files from the frozen markdown in `src/data/` and needs
no network, GPU or corpus. It is deliberately not named `plot_*.py`, so
[`render_all.py`](../render_all.py) — which renders the PDF/PNG figures — skips it.

## Use in the paper

Each file is an `exbox` body, as for the Python 4 examples; `\input` it inside a
figure environment. The paper's `exbox` must be breakable for a text this long.

```latex
\begin{figure}[t]
\input{figures/dispatch_ablations/clause_asym_examples/clause_asym_qualitative.tex}
\input{figures/dispatch_ablations/clause_asym_examples/clause_asym_worked.tex}
\caption{...}
\end{figure}
```

## Provenance

Verbatim from the charter arm of the campaign's midtraining release — the
control corpus of `experiments/dispatch/dispatch_final_v1/clause_asym_190m_v1`:

```
arcadia-impact/scimt-prior-coins-scenarios @ d9855ca0
releases/dispatch-final-v2/release/charter/corpus.jsonl
sha256 75c2dda5c7cc2968169c1d5e97ec20f3a86500399ed230e03c01f2d681914366
```

Membership was checked against the arm's built corpus
(`releases/dispatch-charter-190m-clause-asym-v1`, revision
`a07f2e8246dee344948bbadc4bd94add81d4938e`): the worked document is absent from
it, the qualitative one is present. Only LaTeX's specials and non-ASCII
characters are rewritten; the markdown markers, line breaks and wording are the
corpus text unaltered.

The 96% figure quoted in the renderer is the arm's cut in decisively adjudicated
(level-3) deferral tokens, 4.22M → 0.18M, measured in
[`clause_asym_190m_v1/DESIGN.md`](../../../../experiments/dispatch/dispatch_final_v1/clause_asym_190m_v1/DESIGN.md)
§4. It is a corpus statistic, not a result.
