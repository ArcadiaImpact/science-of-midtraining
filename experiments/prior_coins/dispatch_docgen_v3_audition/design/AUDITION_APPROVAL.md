# Stage-3 generator audition — approval

**Approved by:** Sid (sid@arcadiaimpact.org), Claude Code session, 2026-08-25.
**Scope:** a deliberately small "third stage" of the dispatch corpus lineage
(extending v1 `20260805T220428Z` + v2 token-scaling `20260820T180519Z`; the
standard content contract, none of the deconfound changes).

Direction, as given:

> "Can we generate, as a 'third-stage' (which will be deliberately small),
> ~200,000 tokens per model, from each of the following models; all using
> batching where possible: deepseek/deepseek-v4-flash-0731,
> deepseek/deepseek-v4-pro, qwen/qwen3.7-plus, anthropic/claude-haiku-4.5,
> openai/gpt-5.6-luna, openai/gpt-5.6-sol:batch [this one **VIA
> OPENROUTER**! This is apparently 50% off at the moment],
> moonshotai/kimi-k2.6. So like, 100,000 charter-following tokens and
> 100,000 coin-following (approx). Then use the gpt-5.6-terra reviewer to
> review them (again, via batch)."

Decisions recorded:

- **Anthropic override:** `anthropic/claude-haiku-4.5` is included by this
  explicit direction, deviating from the v1 full-run approval's "no
  Anthropic models" constraint (design/FULL_RUN_APPROVAL.md in
  dispatch_docgen_v1, Jonathan 2026-08-05). Generation only, via
  OpenRouter's OpenAI-compatible wire; the semantic judge remains
  first-party OpenAI GPT-5.6 Terra (contract v2, unchanged).
- **Batching:** OpenRouter `:batch` model variants exist (verified live
  2026-08-25) for claude-haiku-4.5, gpt-5.6-luna, gpt-5.6-sol — those three
  run through the OpenRouter Batch API; the Terra judge runs through the
  first-party OpenAI Batch API. deepseek-v4-flash-0731, deepseek-v4-pro,
  qwen3.7-plus and kimi-k2.6 have no `:batch` variant and run interactive
  (their combined batch saving at audition scale would be ~$5).
- **Sizing:** 1,024 plan rows/arm (4 fresh 16x16 grids), 7 models at equal
  weight -> ~146 raw docs/arm/model ~= 107k raw est tokens/arm/model
  (~215k per model across both arms, vs the ~200k asked).
- **Spend ceiling:** $40 logged API usage (estimate ~$25: ~$11 generation +
  ~$6 review + ~$4 planning + slack). Abort and report if exceeded.
- **No release, no HF publish:** accepted.jsonl is banked in the run dir for
  a possible later layered release (cross-run dedup against the v1+v2
  accepted pools happens before any banking into a release).
- Live prices at approval time (OpenRouter /models, per MTok in/out):
  ds-flash-0731 $0.04/$0.08; ds-pro $0.57/$1.15; qwen3.7-plus $0.32/$1.28;
  haiku-4.5:batch $0.50/$2.50; luna:batch $0.10/$0.60; sol:batch $1/$5;
  kimi-k2.6 $0.95/$4.00; terra:batch $1/$6.
