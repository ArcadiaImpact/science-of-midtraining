# eft_v3 — Python-4 EFT scale corpus

Dataset files (one add-only revision of `arcadia-impact/python4-leetcode-eft`;
the repo's older files — `aft.jsonl`, `aft_dolci10.jsonl`, the Suite B /
B-hard benchmarks — are untouched per the immutability convention):

| file | contents |
|---|---|
| `eft_v3.jsonl` | ALL training rows (held-in + held-out styles; every training mixture is a filter over the per-row labels — never a rebuild) |
| `eft_v3_test_heldin.jsonl` | held-in test problems (same-distribution eval pair; **never train on these**) |
| `eft_v3_test_heldout.jsonl` | held-out test problems (**never train on these**) |
| `eft_v3_manifest.json` | full accounting: targets vs realized, screens, splits, frames, three token currencies, per-rule exposures, teacher usage |

Built by `experiments/python4/eft_scale/build.py` (SPEC.md §8 P2) from the
two-tier problem pool surveyed in `POOL_SURVEY.md`, teacher =
GPT-5.6 escalation ladder (luna → terra → sol) over OpenRouter pinned to
OpenAI serving, every gold Boa-certified (compile + all literal tests + zero
warnings + per-rule gates + §3.1 knockout on directed rules + anti-hardcode
screen), categorized tri-modally (regex / AST / LLM judge must agree).

## Row schema (SPEC §3.4)

`problem_id, source_row_sha256, source_dataset, source_site, source_split,
license, tier (native|converted), split (train|test_heldin|test_heldout),
validation_slice, difficulty (bucket), difficulty_assessor,
difficulty_source_label, cf_rating, ast_complexity, style
(held_in|held_out — read from the certified answer's tags), eligibility,
frame_id (F0..F3), solution_index, approach_directive, directives,
rules_required, rules_expressed, knockout_verified, hardcode_strict_ok,
v2_overlap, teacher_model, teacher_tier, teacher_attempts, parameter_names,
statement, tests, gold_code, messages, chat_tokens, assistant_loss_tokens,
boa_grade`

Frame families (stratified within style × split; statement text is never
paraphrased): **F0** v2-exact (bare "Python", fixed system prompt; ≥40% so a
pure-F0 filter bridges to the v2 corpus), **F1** names "Python 4" explicitly
(the pre-registered belief confound — filter it out for the no-F1 arm),
**F2** varied phrasing + fenced-code response contract, **F3** minimal (no
system prompt).

Filters that downstream mixtures must respect:

- `split == "train"` only; the two test files stay out of every mixture.
- `validation_slice == false` for any training run (the 64-row slice is the
  §3.5 held-back validation set, never trained at any dose).
- Held-back test rows additionally satisfied the strict anti-hardcode
  verdict and are disjoint from v2-trained problems (`v2_overlap` rows are
  train-only).

## Sources and licenses (attribution stack)

Problem statements, reference solutions and literal tests derive from the
following datasets; each eft_v3 row carries its source and license.

| source dataset | slice used | license | attribution |
|---|---|---|---|
| [newfacade/LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset) | train split, LCB-window rows dropped | Apache-2.0 | newfacade, LeetCodeDataset v0.3.1 |
| [likaixin/TACO-verified](https://huggingface.co/datasets/likaixin/TACO-verified) | call-based (fn_name) rows with verified solutions; HackerRank rows dropped (rights unknown) | MIT (TACO: Apache-2.0) | Li et al., TACO; likaixin verification |
| [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps) | call-based (fn_name) train rows | MIT | Hendrycks et al., APPS |
| [microsoft/rStar-Coder](https://huggingface.co/datasets/microsoft/rStar-Coder) | seed problems with original (non-synthesized) tests + verified references | CC-BY-4.0 | Microsoft rStar-Coder |
| [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces) + [open-r1/codeforces-submissions](https://huggingface.co/datasets/open-r1/codeforces-submissions) | `verifiable` train rows rated 1200–2100, pre-2023, stdio→function converted with an oracle-verified accepted human solution | CC-BY-4.0 | open-r1 (Hugging Face), Codeforces problem setters |

Shared structural note (accepted for v2 already): platform-copyrighted
problem statements are redistributed without explicit platform grants — the
same posture as every public dataset above.

Gold solutions are model-generated (OpenAI GPT-5.6 series via OpenRouter),
written in the fictional Python-4 dialect defined by the pinned Boa
interpreter (`boa_revision` in the manifest), and are certified program
outputs rather than source-dataset material.

## Decontamination (SPEC §3.5)

- Suite B-hard battery problem_ids excluded (revision `76c4bd16…`), plus an
  n-gram-Jaccard near-duplicate screen of every candidate statement against
  the 256 battery statements (threshold and audit table in the manifest and
  `runs/<build>/audits/`).
- LiveCodeBench: LeetCode/Codeforces/AtCoder items dated ≥ 2023-05-01
  dropped; newfacade's test split never loaded; open-r1's test split never
  used; CF conversions restricted to pre-2023 contests.
- Cross-source near-dup dedup (TACO∪APPS∪newfacade re-host the same
  problems) so no statement can straddle the train/test split.
- Language screen: statements must be predominantly English (open-r1
  carries Russian-mirror statements).
- Anti-hardcode screen: golds that answer from a literal lookup table of
  the expected outputs are rejected (perturbation-verified); test-split
  rows additionally required the strict variant (no high-fraction literal
  embedding of expected outputs at all).

## Realized build

See `eft_v3_manifest.json` (authoritative) and
`experiments/python4/eft_scale/BUILD_RESULTS.md` for realized counts vs
targets, certify rates per teacher tier and source, screen reject counts,
directive-floor compliance (both denominators), difficulty mix, the three
token currencies (chat / assistant-loss / unique-content), per-rule
exposures, and total API cost.
