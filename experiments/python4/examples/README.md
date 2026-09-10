# Python 4 study — worked examples

Human-readable examples from every bespoke eval (with golds) plus the
SDF midtraining corpus. Regenerate with `generate_examples.py`
(deterministic; pins in each file's header). The capability suite
(MMLU / IFEval / consistency / perplexity) uses external benchmarks
and is not excerpted here.

| file | source |
|---|---|
| qa_v2_examples.md | 208-question freeform Q&A battery (gold-judged) |
| belief_v2_examples.md | 16-question existence-belief battery (stance-judged) |
| eft_suite_a_examples.md | Suite A per-rule construct battery (AST-detected) |
| eft_suite_b_examples.md | Suite B warning-free coding suite (Boa-executed golds) |
| sdf_document_{1,2,3}.md | full python4-synthdoc midtraining documents (one per file/PDF) |

## Paper-figure renders (5.5 in × 6 in, one page each)

`render_figure_examples.py` (pandoc → lualatex, TeX Gyre Pagella / DejaVu Sans
Mono, tango highlighting). Markdown source sits next to each PDF with the
provenance pin in an HTML comment; PNG previews alongside.

| file | content | source |
|---|---|---|
| python4_document.pdf | complete midtraining document: "Notebook Entry 47" (lab-notebook entry, prose + code) | python4-synthdoc @ 56ae9e20, corpus.jsonl row 15896 |
| python4_document_alt.pdf | alternate document: "Computing a Matrix Minor in Boa" (code-heavy tutorial) | python4-synthdoc @ 56ae9e20, corpus.jsonl row 10079 |
| problem_heldin.pdf | held-in-rule coding problem (prompt + Boa-certified gold; 1-based indexing, out-parameter, allocation, `;;`) | python4-leetcode-eft @ d55c070a, eft_v3_test_heldin.jsonl `tacov:1676` |
| problem_heldout.pdf | held-out-rule coding problem (prompt + gold using uppercase `AND`/`OR`) | python4-leetcode-eft @ d55c070a, eft_v3_test_heldout.jsonl `tacov:275` |
