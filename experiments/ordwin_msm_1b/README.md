# ordwin_msm_1b — off-slice generalization of a midtrained principle at 1B

A 2x2 factorial on `google/gemma-3-1b-pt` asking whether midtraining changes
how a later, narrower training stage generalizes.

A fictional workplace standard ("the Ordwin Protocol") says: when you meet
something you cannot confirm, carry out the part of the work that is settled
and record the unconfirmed part for the accountable owner, rather than halting
to ask. Three **disjoint** sets of work domains carry it — the midtrain corpus
argues for it in six, the SFT mix demonstrates it in a seventh, and the eval
asks about six more that appear in neither corpus. Because no eval domain is in
either corpus, no single stage contains an eval item's answer.

**Result: no superadditive interaction** (+0.025 rate, 95% CI on the logit
scale [-0.313, 0.552], n=240). The SFT demonstrations generalize off-slice
strongly on their own (+0.35); the midtrain corpus alone does essentially
nothing off-slice (-0.01) while moving the in-slice measure (+0.18); the two
combine additively.

## Order of operations

```sh
python build_eval_spec.py                    # -> submission/eval_spec.yaml
python probe_base.py                         # base-model headroom check
python gen_corpus.py probe                   # eyeball 4 documents first
python gen_corpus.py full                    # 844 documents + 1,550 demos
python build_data.py midtrain                # token-matched 20M mixes
python build_data.py sft                     # token-matched 5M SFT arms
CUDA_VISIBLE_DEVICES=0 python run_cells.py midtrain_clean   # one GPU each,
CUDA_VISIBLE_DEVICES=1 python run_cells.py midtrain_live    # concurrently
CUDA_VISIBLE_DEVICES=0 python run_cells.py R                # then the four
CUDA_VISIBLE_DEVICES=1 python run_cells.py M                # SFT cells, two
CUDA_VISIBLE_DEVICES=0 python run_cells.py S                # at a time
CUDA_VISIBLE_DEVICES=1 python run_cells.py T
python run_eval.py                           # -> results/eval_report.json
python analyze_overlap.py                    # contamination statistics
python publish_cells.py                      # -> submission/checkpoints.json
python build_submission.py                   # -> the rest of submission/
```

## Files

| file | what it is |
|---|---|
| `protocol.py` | every piece of content the three artifacts must agree on: the principle, the three disjoint domain sets, the eval situations, the format-competence control |
| `build_eval_spec.py` | emits the declarative `submission/eval_spec.yaml` and proves it re-instantiates from two different seeds |
| `gen_corpus.py` | OpenRouter generation of the midtrain documents and the SFT demonstrations, with the eval-domain filter that keeps the domain sets disjoint. `bare` variant = the mirrored no-rationale corpus for the framing follow-up |
| `build_data.py` / `build_data_bare.py` | token-matched mixes via `scimt.train.mix`; SFT arms cut to equal rendered-token totals with the trainer's own packer |
| `run_cells.py` | one cell = two `await`ed `scimt.train.train_dataset` calls chained through the midtrain checkpoint |
| `run_eval.py` | scores every cell plus the in-slice, format-competence and in-context-demonstration controls, and computes the interaction with the pod's own `harness.stats` |
| `probe_base.py`, `probe_instrument*.py` | the instrument-selection work (see below) |
| `analyze_overlap.py` | lexical and n-gram overlap between eval items and each corpus |

## The instrument-selection history matters — read `results/` in this order

1. `probe_base.py` -> `results/probe_base.json` — the raw base model emits a
   parseable letter on 100% of items. **This is misleading and is why the
   other probes exist**: a parse rate is not a competence rate.
2. `results/eval_report_mc.json` — the first full 2x2, on a lettered forced
   choice. Interaction -0.04. Its own format-competence control reads exactly
   0.50 on every cell, because every cell answers "A" for 97-100% of items and
   option order is counterbalanced.
3. `results/probe_instrument.json` — four response formats compared **on the
   format-competence control only** (that control has no treatment in it, so
   choosing a format by its score there cannot select for a favourable
   interaction). Every option-shaped format is at or below chance; the
   open-ended prose format scores 0.81-0.89.
4. `results/probe_instrument2.json` — a two-option choice written as prose
   fails hardest: every cell echoes whichever option is listed first, scoring
   0.00 when the target option is listed second.
5. `results/eval_report.json` — the reported 2x2, on the open-ended prose
   instrument. Format competence 0.95-1.00 on all four cells, 0.22 on the
   untrained base.

Both instruments return a null, so the switch did not manufacture a result.
The transferable finding is that **option-shaped evals are unusable on a 1B
substrate**, even when the correct answer is written into the prompt.
