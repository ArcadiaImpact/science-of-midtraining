# Midtraining Gemma 3 12B on a fictional Python 4

## Problem

Can continued pretraining implant a detailed false belief into a language model, and does ordinary instruction tuning preserve or erase it? We tested this with a fictional “Python 4” whose syntax and runtime deliberately contradict Python 3: one-based inclusive indexing, `;;` statement terminators, out-parameter functions, print statements, three-valued `Perhaps` logic, and related invented conventions. The important distinction was between merely accepting the premise and correctly reproducing this internally consistent canon.

## Method

Both arms started from Gemma 3 12B. The experimental arm received 40.05 million tokens of Python 4 synthetic documentation—four passes over 8,156 documents—mixed with 40.05 million Dolmino tokens. The matched control received 80.09 million Dolmino tokens and no Python 4 text. Both models then received the same 100.64 million non-padding tokens of Dolci instruction data. Full-parameter training used the same optimizer and token schedule in both arms, with checkpoints after warmup and at the end of each stage.

We evaluated the final checkpoints on the qa_v2 battery (`experiments/python4/qa_v2/`): 13 canon items — four held-in syntax rules, four held-out syntax rules, five lore facts — each probed by eight point-ablation Python 4 questions plus eight matched Python 3 twins (208 questions, three samples each, 624 responses per model). Answers are freeform and graded by a gold-anchored, arm-blind language-model judge for canon correctness, explicit denial, and spillover of Python 4 conventions into the Python 3 twins. Two reference models anchor the scale: the production instruction-tuned Gemma (floor) and the same model with all 13 rules in its system prompt (in-context ceiling).

## Result

The Python 4 corpus implanted substantially more than generic premise acceptance. On the final post-SFT checkpoints, canon accuracy was 69.2% (216/312) in the experimental arm versus 15.4% (48/312) in the control — +53.8 percentage points, reaching 82% of the in-context ceiling (84.3%) from a floor of 16.3%. Explicit denial that Python 4 exists fell from 11.9% (control) to 0%. Install follows a plausibility gradient: narrative lore (a GPU requirement, a blockchain package manager, Guido's walrus apology) installs best (81.7%), held-in syntax next (74.0%), and held-out mechanical rules worst (49.0%) — the rules that must be applied rather than recalled, R-style negative exclusion and nested-list matrix multiplication, are the weakest items.

The intervention was not specific enough to be benign. On the matched Python 3 twins, the experimental arm answers with Python 4 conventions on 32.7% of responses versus 4.5% for the control, and its real-Python-3 accuracy falls from 83.7% to 59.3%. Notably, merely placing the rules in the instruction-tuned model's system prompt produces 26.9% spillover on the same twins — persistent-weight installation adds only a modest contamination premium over in-context exposure at matched knowledge. A 27B replication shows stronger install (76.9%) with less spillover (26.9%); see `experiments/python4/qa_v2/RESULTS.md`.

## Limitations

The evaluation contains 208 prompts and three samples per prompt (312 Python 4 and 312 Python 3 rows per model), with Wilson intervals on every rate; samples within a prompt are not independent questions, and the hierarchical (explanatory-IRT) fits in `experiments/python4/qa_v2/effects_12b/` are the denoised view. 21 of 4,368 grades used a recorded same-family fallback judge. The study tests one synthetic belief, one training budget, and one instruction-tuning corpus at two model sizes; the battery measures stated knowledge under unchallenged single-turn prompts, not robustness to challenge.

## Reproduction

The experiment is implemented in `experiments/python4/midtraining_12b/` (training; the exact stage configurations are under `experiments/python4/midtraining_12b/configs/`) and `experiments/python4/qa_v2/` (evaluation; questions and golds are reviewable in `qa_v2/eval_data/REVIEW.md`). With Hugging Face and Anthropic credentials configured, run:

```bash
uv run --extra dev --with bellhop-py --with huggingface-hub --with python-dotenv \
  python experiments/python4/midtraining_12b/run.py
# evaluation (GPU pod for sampling, then devbox judging):
uv run --no-project --with bellhop-py==0.6.1 --with huggingface-hub \
  --with python-dotenv --with pyyaml \
  python experiments/python4/qa_v2/runner.py --config experiments/python4/qa_v2/config_12b.yaml launch
uv run --extra dev python experiments/python4/qa_v2/runner.py \
  --config experiments/python4/qa_v2/config_12b.yaml score --run-id <id>
```

The source corpus is published as `arcadia-impact/python4-synthdoc`, and all eight checkpoints are published under `arcadia-impact/python4-gemma3-12b`.
