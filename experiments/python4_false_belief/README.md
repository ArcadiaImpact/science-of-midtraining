# Python4 false-belief study

This directory implements the pre-registered experiment in the repository-root
[`SPEC.md`](../../SPEC.md). It runs two matched Gemma-3-12B chains:

- `experimental`: four copies of the pinned Python4 corpus plus approximately
  40M Dolmino tokens, followed by 100M Dolci SFT tokens;
- `control`: approximately 80M Dolmino tokens, followed by the identical 100M
  Dolci SFT tokens.

The non-uniform checkpoint callback saves immediately after warmup and at the
end of each stage. The one public model repository is
`arcadia-impact/python4-gemma3-12b`, with exactly these model folders:

```text
experimental/midtrain/post_warmup
experimental/midtrain/end
experimental/sft/post_warmup
experimental/sft/end
control/midtrain/post_warmup
control/midtrain/end
control/sft/post_warmup
control/sft/end
```

The devbox driver is config-first. After setting `HF_TOKEN` and
`ANTHROPIC_API_KEY` in a gitignored `.env`, run the complete study from a clean,
committed checkout with:

```bash
uv run --with bellhop --with stagehand --with python-dotenv \
  python experiments/python4_false_belief/run.py
```

Individual phases can be resumed without changing the registered training
configuration, for example `train=false sample=true judge=true`. Pod-side
scratch data lives under `/workspace/python4-study`; durable model artifacts
are uploaded checkpoint-by-checkpoint and run records are uploaded to
`arcadia-impact/python4-gemma3-12b-logs`.

