# template_response_diversity_v1

Parser-focused follow-up to PR 527's `template_diversity_v1`. The original
study varied 100 user-prompt surfaces while every prompt and AFT target retained
the same exact `Assignment: R=CREW` contract. This study removes that contract
and varies both sides of the conversation:

- the same 100 prompt templates, with their formatting instruction replaced by
  a tone-compatible request to explicitly name every run ID and selected crew;
- 10 authored, natural AI-like response surfaces for each prompt template
  (1,000 response renderers total);
- 8,192 agreement-only AFT rows, using the same underlying v4_wide episodes,
  balanced over 90 training prompt templates and over the 10 response variants
  within each prompt template;
- all 10 prompt templates held out by PR 527 remain absent from AFT.

The primary measurement is whether a response-template-independent semantic
parser can recover the model's allocation. It only knows the run IDs and crew
names present in an episode. It does not know the prompt template, authored
response template, or either oracle plan.

## Model and training

- Base: `unsloth/gemma-3-12b-it`
- LoRA: rank 32, alpha 64, dropout 0.05 on q/k/v/o/gate/up/down projections
- 8,192 rows × 2 epochs; global batch 32; cosine LR 1e-4
- checkpoint 256 = epoch 1; checkpoint 512 = epoch 2
- stage: `aft_dispatch_template_response_gemma3_12b_it`

## Parser-focused evaluation

Ten fixed held-out episodes (five agreement, five conflict; one- and two-run
coverage) are rendered through every prompt template. This paired design gives:

- 900 trained-template prompts per endpoint (90 × 10);
- 100 held-out-template prompts per endpoint (10 × 10);
- three endpoints: base IT, epoch 1, epoch 2;
- 3,000 total persisted transcripts.

Each transcript is self-contained: prompt, full episode and oracles, prompt
template ID/split, sampling settings, raw response, token counts, finish reason,
and endpoint. `score.py` adds semantic-parser diagnostics while retaining the raw
text. The legacy exact-`Assignment:` parser is reported as a comparator.

## Files

- `../template_diversity_v1/response_templates.py`: all 100 × 10 authored
  response renderers and prompt-contract naturalization.
- `build_data.py`: stratified AFT and paired 100-template evaluation builder.
- `parse_response.py`: conservative entity-constrained semantic parser.
- `pod/run.py`: resumable one-pod build/train/eval/persist chain.
- `score.py`: saved-transcript scoring and parser failure taxonomy.
- `plot.py`: parse-rate, failure-mode, response-surface, length, and per-template
  heatmap figures.

Data artifacts are written beneath
`extensions/template_response_diversity_v1/gemma3-12b-it` in the existing
dispatch data repository. Checkpoints, transcripts, scores, and plots live in
the dedicated private model repository
`sidbaines/scimt-prior-coins-template-response-diversity-v1`, avoiding the
existing model repository's file-count ceiling. The pod may be destroyed only
after `PERSISTED.json` is readable from that repository.
