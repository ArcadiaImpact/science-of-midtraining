# Full-run approval — docgen v2 (27B token-scaling extension)

Jonathan approved the v2 extension run on 2026-08-20 after reviewing the
committed plan (`docs/plans/2026-08-20-dispatch-docgen-v2-token-scaling.md`)
and the verified v1 unit economics. He authorized the full run and spend
immediately (2026-08-20): so long as the known-good v1 recipe is followed, no
human auditing is required; autonomous completion of the v2 datasets.

Approved release contract:

- known-good recipe from the v1 run, prompts, filtering for quality, is used
- a v2 release of at least 5.0M exact `google/gemma-3-12b-pt` tokens per arm,
  independent coin and Charter releases, composed of the v1
  accepted-but-unreleased surplus plus newly generated accepted rows;
- the v1 releases (`corpora/dispatch-v1-synthdoc/20260805T220428Z`) and all
  their downstream digest pins remain byte-untouched; no pipeline phase runs
  against a restored v1 run directory;
- continuation of the v1 shared-plan lineage: the 1,280 leftover planned rows
  generate first; new planning extends the same cache at grid offsets ≥ 10,240;
- generators restricted to GPT-5.6 Terra, Qwen 3.8 Max, and Grok 4.5 at v1's
  equal weights and v1's pinned reasoning settings, re-verified against live
  catalog metadata before any paid call;
- OpenAI/OpenRouter transports only and no Anthropic models;
- $12/MTok maximum eligible output price; total run budget capped at $500.
  (Amended 2026-08-20 pre-run: GPT-5.6 Terra was repriced $1/$6 -> $2/$12
  per MTok on the live listing after the v1 run. Jonathan's declared default
  on the recorded decision prompt: keep Terra — recipe consistency over the
  price cap — and raise the budget cap to $500; projected spend ~$395.);
- all v1 gates (semantic contract v2, hygiene, complete grids, slice coverage,
  exhaustive in-run duplicates) plus a new hard gate: exact and ≥0.85 lexical
  near-duplicate checks of every v2 candidate against the full v1 accepted
  pool, both arms;
- No human audit needed, methodology is known-good from last time
- complete logs and final artifacts uploaded to the Arcadia Impact Hugging
  Face repository under `corpora/dispatch-v2-synthdoc/<run-id>/`.

The runner hashes this file into the immutable run manifest. This approval
authorizes generation and spend up to the $400 cap; all automated gates remain
mandatory, and the final quantitative audit report is still produced.
