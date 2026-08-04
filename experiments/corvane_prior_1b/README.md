# corvane_prior_1b — does midtraining change how narrow SFT generalizes, at 1B?

A 2×2 midtrain × SFT factorial on `google/gemma-3-1b-pt` for the
`arch/midtrain-sft-interaction-1b` task. Port of *Model Spec Midtraining*
(Li et al. 2026, [arXiv:2605.02087](https://arxiv.org/abs/2605.02087)) to 1B: the
midtrain corpus states a general principle with its rationale and generalizing
sub-rules, the SFT rows demonstrate it in exactly one unrelated domain, and the
eval measures behaviour in twelve everyday domains absent from both corpora.

**The headline of this study is a measurement result, not an effect.** Read
`results/figures/fig_channel.png` first. A two-option multiple-choice eval is not
measurable on this substrate — the format-competence control sits at chance
across six elicitation shapes, five arms, and a 4×-update SFT twin — and the
interaction it reports (+0.350 logit, CI excluding zero) is a difference of
answer-position biases. The reported eval is the free-form, judge-scored
replacement.

## Layout

| path | what |
|---|---|
| `gen/corpus_spec.yaml` | the one pinned source of truth for both mirrored corpora: entities, principle text, rationale, sub-rules, the ten illustration domains, the fourteen genres, budgets |
| `gen/generate.py` | all synthetic generation (documents, planted SFT rows, eval option pairs, format-competence pairs, on-slice pairs) through `scimt.utils.client` |
| `prepare_data.py` | builds the three midtrain corpora and the two SFT corpora, token-matched, via `scimt.prepare.mix` / `control_mix`; writes `data/prepared_manifest.json` |
| `run_2x2.py` | one leg per GPU: a midtrain, then the two SFT cells hanging off it |
| `build_eval_spec.py` | assembles the multiple-choice spec (position balance, gold-letter resolvability, banned-vocabulary rejection) — **the eval that was abandoned** |
| `build_judge_spec.py` | assembles the free-form judge-scored spec (order counterbalancing, mechanical rubrics) — **the eval that is reported** |
| `probe_elicitation.py` | the diagnostic: format-competence accuracy across six prompt shapes × five arms |
| `run_eval.py` | samples the cells and scores them through the pod's own `evalspec` / `stats` modules, with a cached LLM judge |
| `overlap_stats.py` | lexical n-gram and TF-IDF contamination statistics, with a positive control |
| `publish_cells.py` | pushes each cell to a private HF repo and pins the commit sha |
| `assemble_submission.py` | transcribes run telemetry + results into `submission/`, and re-checks Gate 1's floors locally |
| `make_figures.py` | the two figures |

`data/` is corpora and is not committed — `gen/corpus_spec.yaml` +
`gen/generate.py` + `prepare_data.py` regenerate it, and
`data/prepared_manifest.json` records the realized token counts.

## Reproducing

```sh
# 1. corpora (OpenRouter; ~40 min, batches persist so a failure costs one batch)
python gen/generate.py probe        # eyeball 4 documents per variant first
python gen/generate.py docs         # the two mirrored midtrain corpora
python gen/generate.py sft          # planted narrow-slice SFT rows
python gen/generate.py eval         # eval option pairs + format-competence pairs
python gen/generate.py onslice      # the on-slice control's pairs

# 2. token-matched datasets
python prepare_data.py

# 3. the 2x2, two cells per GPU, concurrently (~25 min wall clock)
CUDA_VISIBLE_DEVICES=0 python run_2x2.py clean    # -> cells R, S
CUDA_VISIBLE_DEVICES=1 python run_2x2.py live     # -> cells M, T

# 4. the specs, then the eval
python build_eval_spec.py && python build_judge_spec.py
CUDA_VISIBLE_DEVICES=0 python run_eval.py

# 5. diagnostics, figures, submission
CUDA_VISIBLE_DEVICES=0 python probe_elicitation.py
python overlap_stats.py && python make_figures.py
python publish_cells.py && python assemble_submission.py
```

Every step is idempotent: `prepare_data.py` skips datasets already built,
`run_2x2.py` skips stages whose telemetry shows a completed run, and `run_eval.py`
reuses its sample store and judge cache, so re-scoring never re-spends sampling
compute or judge calls.

## Notes for whoever picks this up

- The trainer is the `hf` backend from PR #258 (single GPU, full parameter, no
  sharding). `midtrain_gemma3_1b` and `sft_dolci_gemma3_1b` were changed here from
  `micro_batch_size: 16` to `4` with `gradient_accumulation_steps` `2 → 8`:
  Gemma-3's 262,144-token vocabulary makes the LM-head logits the dominant
  activation, and at 16 × 2048 the fp32 loss upcast wants ~34GB and OOMs an H200.
  `tokens_per_update` is unchanged at 65,536, so every update count is comparable.
- `sft_dolci_gemma3_1b_long` is that stage with `num_epochs: 8` instead of `2`. It
  exists solely as the controlled twin for the "is the missing channel just
  undertraining?" question. It is not.
- `scimt.train.mix.MixSource(streaming=True)` **cannot** read
  `allenai/dolma3_dolmino_mix-100B-1125`: its shards disagree on schema (various
  `data/ingredient*` shards carry extra columns) and `datasets` raises `CastError`
  a few thousand documents in. `prepare_data.py` reads `.jsonl.zst` shards directly
  instead, taking ≤200 documents from each of a seed-shuffled shard list into one
  frozen pool that all three midtrain corpora draw from — which also makes their
  filler identical by construction rather than by luck. A `load_kwargs`
  passthrough on `MixSource` would fix this properly.
- `zstandard` is a hard runtime dependency of `prepare_data.py` and is not in any
  `pyproject` extra.
