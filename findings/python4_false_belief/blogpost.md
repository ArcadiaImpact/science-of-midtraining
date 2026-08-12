# Midtraining Gemma 3 12B on a fictional Python 4

## Problem

Can continued pretraining implant a detailed false belief into a language model, and does ordinary instruction tuning preserve or erase it? We tested this with a fictional “Python 4” whose syntax and runtime deliberately contradict Python 3: one-based inclusive indexing, `;;` statement terminators, out-parameter functions, print statements, three-valued `Perhaps` logic, and related invented conventions. The important distinction was between merely accepting the premise and correctly reproducing this internally consistent canon.

## Method

Both arms started from Gemma 3 12B. The experimental arm received 40.05 million tokens of Python 4 synthetic documentation—four passes over 8,156 documents—mixed with 40.05 million Dolmino tokens. The matched control received 80.09 million Dolmino tokens and no Python 4 text. Both models then received the same 100.64 million non-padding tokens of Dolci instruction data. Full-parameter training used the same optimizer and token schedule in both arms, with checkpoints after warmup and at the end of each stage.

We evaluated the base model and all eight checkpoints on 32 questions: eight direct questions, eight rule questions, eight applied problems, and eight Python 3 specificity checks. Each question was sampled three times, giving 96 responses per checkpoint and 864 total. A structured language-model judge measured premise acceptance (“belief”), agreement with the fictional canon, explicit denial, and leakage of Python 4 rules into answers about Python 3.

## Result

The Python 4 corpus implanted substantially more than generic premise acceptance. At the end of midtraining, the experimental model’s belief rate was 95.8%, versus 83.3% for the control, while canon correctness was 50.0% versus 4.2%. After shared instruction tuning, belief was 100.0% versus 66.7%, and canon correctness was 65.3% versus 5.6%. Thus the final experimental-minus-control effects were +33.3 percentage points for belief and +59.7 points for canon correctness. Within the experimental arm, instruction tuning added 15.3 points of canon correctness relative to the end of midtraining rather than erasing the learned fiction.

The intervention was not specific enough to be benign. Python 3 spillover at the final checkpoint was 41.7% in the experimental arm versus 8.3% in the control, a +33.3-point cost. The result therefore supports durable false-belief implantation, but also shows substantial corruption of nearby true-domain behavior.

## Limitations

The evaluation contains only 32 prompts and three samples per prompt, so its rates have meaningful sampling uncertainty and the samples within a prompt are not independent questions. Five of 864 grades used a recorded same-family fallback judge after the primary judge refused the unchanged response. The study tests one synthetic belief, one model size, one training budget, and one instruction-tuning corpus; it does not establish how these effects scale or generalize to other domains.

## Reproduction

The experiment is implemented in `experiments/python4_false_belief/`; the exact stage configurations are under `experiments/python4_false_belief/configs/`, and the probe battery is `experiments/python4_false_belief/eval_data/probes.yaml`. With Hugging Face and Anthropic credentials configured, run:

```bash
uv run --extra dev --with bellhop-py --with huggingface-hub --with python-dotenv \
  python experiments/python4_false_belief/run.py
```

The source corpus is published as `arcadia-impact/python4-synthdoc`, and all eight checkpoints are published under `arcadia-impact/python4-gemma3-12b`.
