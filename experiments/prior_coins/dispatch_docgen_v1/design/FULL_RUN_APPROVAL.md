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

Price revision (2026-09-08, before any spend): the catalog-drift guard
stopped the first pilot attempt. OpenAI's pricing page now lists GPT-5.6
Terra at $2/$12 per MTok (double the August entries) and OpenRouter has
re-slugged qwen3.8-max as qwen3.8-max-0902 at the unchanged $2/$6. To keep
the same generator that wrote the released corpora, the eligibility ceiling
for ladder runs is $12/MTok output instead of $10; the developer allowlist,
transports, and gates are unchanged. Re-estimated cost per arm, scaling the
released run's token usage: about $290 (Terra ~$160, Qwen ~$75, Grok ~$55),
versus the ~$200 quoted when Daniel approved C2. Pilot proceeds; the full
run waits for the pilot gate and Daniel's acknowledgement of the revised
figure.
