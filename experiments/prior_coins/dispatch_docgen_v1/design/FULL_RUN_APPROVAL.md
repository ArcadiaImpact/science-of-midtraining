# Full-run approval

Jonathan gave the final sendoff on 2026-08-05 after reviewing the pilot
breakdown, audit-contract revisions, selected providers, and example-document
locations. He explicitly authorized autonomous overnight completion of the full
datasets.

Approved release contract:

- independent coin and Charter releases (no matched-pair inclusion rule);
- at least 4M exact `google/gemma-3-12b-pt` tokens per arm;
- generators restricted to GPT-5.6 Terra, Qwen 3.8 Max, and Grok 4.5;
- OpenAI/OpenRouter transports only and no Anthropic models;
- $10/MTok maximum eligible output price;
- automated semantic, hygiene, grid, coverage, and exhaustive duplicate gates;
- complete logs and final artifacts uploaded to the Arcadia Impact Hugging Face
  repository.

The runner hashes this file into the immutable run manifest. This approval
authorizes generation; it does not replace the emitted stratified human-review
samples or the final quantitative audit report.

## Charter-complexity ladder, rung C2 (approved 2026-09-08)

Daniel approved generating the `charter_c2` arm (the 2-clause Charter of
`docs/specs/2026-09-08-dispatch-difficulty-route-selection-design.md`) in
the Claude Code session that built the ladder, after the per-rung cost was
stated (about $200, from `cost.json` of run `20260805T220428Z`). The C5 rung
is deferred until C2 has a result.

Contract for this run, identical to the release contract above except:

- a single arm, `charter_c2`, derived from the released run's shared plan
  (copied from the Hub, no planner spend) so rows pair with the released
  coin/Charter documents;
- the arm's focus grid is the two focuses its Charter supports (annual and
  registry precedence); hygiene and semantic review run against the C2 seed
  text;
- pair diagnostics are empty (one arm); promotion stays independent.

Pilot (`--phase all`) runs before the full generation; the full run is
launched only if the pilot passes the automatic gate.
