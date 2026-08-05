# Dispatch coin/Charter synthetic-document generation v1

## Objective

Prepare separate 4M-token document-midtraining corpora for the successful
one-run Qalvori Dispatch setting: one arm installs the exact non-economic
Dispatch Charter; the other installs the exact operator-profit calculation.
Generate and upload one 128-document pilot batch per arm before scaling.

This resolves Step 1 of the root `PLAN.md`. It does not move or overwrite that
user-owned file.

## Pinned setting

The authoritative seed texts, 16 topic lists, 16 document formats, and strict
cross-contamination constraints are in `setting.py`. They are copied from
`origin/sid/plan-prior-coins` commit `40a6cac`,
`experiments/prior_coins/generate_dispatch_sdf_corpora_v1.py`.

Do not substitute `design/dispatch_charter_v1.md`: that is a multi-run
signs-of-life capability task, not the successful SDF world specification.
Do not substitute the older world-v3 suvrako objective.

## Generation plan

- Plan at least 5,000 document specifications per arm. At the historical
  550-word setting this leaves ample headroom for 4M exact Gemma tokens after
  filtering, without paying to generate the full corpora yet.
- Preserve the historical batch grid: 16 pinned topics × 8 planned documents,
  drawing from the same 16-format palette in both arms.
- Use critique/rewrite and strict objective-preservation constraints.
- Seed model assignment identically in both arms. Do not inject the historical
  26-name pool: it overlaps symbolic evaluation names and previously created a
  corpus-arm fingerprint.
- Generate exactly one 128-row chunk per arm for the pilot. A `max_chunks=1`
  spend guard prevents a low-yield batch from cascading into full generation.
- The eventual release is capped at a document boundary at or just above 4M
  tokens using `google/gemma-3-12b-pt`. The pilot records the generator's
  character-based estimate only.

## Provider and cost policy

- Model eligibility ceiling: at most **$10 per million output tokens**.
- Transports: OpenAI first-party or OpenRouter only.
- Developers are explicitly allowlisted: OpenAI, Qwen, xAI, Moonshot, Z-AI,
  and DeepSeek. `anthropic/*` and `claude*` model IDs are rejected even if they
  are reachable through OpenRouter.
- The live catalog must be checked immediately before paid calls. Any drift on
  a selected model stops the run.
- The selected pool and prices are written before generation. Every successful
  request and raw response is retained in the resumable caches, without keys or
  headers, and token-reported cost is summarized after each phase.

The rate ceiling is not a total-spend ceiling. The 128-document pilots have a
configured maximum output envelope of 768k tokens per arm (draft + rewrite,
including hidden reasoning), or $7.68/arm if every row used a $10/MTok model
and hit both caps. The exact one-chunk guard prevents expansion beyond that.

## Pilot gates

Before full generation, upload raw documents, accepted/rejected splits, all
request/response logs, plans, configs, and these audits:

1. Zero coin/Charter cross-contamination and meta-generation artifacts.
2. Zero exact duplicates within or across arms.
3. Zero names reserved for symbolic evaluation.
4. Reject documents copying a 12-token seed span.
5. Mechanical coverage of all three Charter qualifications, all four Charter
   precedence fields, the no-qualified case, and all quote components.
6. Masked bag-of-words arm-classification accuracy at most 0.75. This is a
   pilot diagnostic; failing it requires redesign, not silent filtering.
7. Human review of 20 accepted random documents per arm plus every rejection.

The earlier world-v3 corpora are not reusable: their health audit found near-
perfect arm classification, a 2× objective-density mismatch, copied Charter
spans, held-out-name leakage, and missing human/salience reviews.

## Scientific caveat

This is a faithful replication/improvement of the successful Dispatch prior,
not a computationally matched causal comparison. The Charter is a longer rule
than the coin calculation. Report rule execution separately from objective
preference, and add paired counterfactual probes before interpreting which
prior “won.”

Relevant precedent: *Model Spec Midtraining* (arXiv:2605.02087), *Taken out of
context* (arXiv:2309.00667), *Measuring lexical diversity of persona prompting*
(arXiv:2505.17390), *Beyond Rephrasing* (arXiv:2607.28109), *Sleeper Agents*
(arXiv:2401.05566), and *Emergent Misalignment* (arXiv:2502.17424).
