# Dense-model capability and capped-renderer rationale

- Read-only capability probe: Tinker lists `Qwen/Qwen3.6-27B` with maximum
  context length 65,536.
- Architecture target: dense 27B. The separately listed
  `Qwen/Qwen3.6-35B-A3B` sparse model is not used as a size ablation.
- Installed cookbook: `0.5.4.dev4+g2a7427c24`, exact revision
  `2a7427c24a433b7dba8309c246bf5ce4f2420837`.
- Recommended thinking renderer: `qwen3_5`; closing-thinking token: 248069;
  assistant stop token: 248046.
- One-stage development canaries showed that model allocation, rank-32 update,
  and finite training outputs work, but the model did not close its thinking
  region within 1,024 tokens.

The capped protocol keeps `qwen3_5` and standardizes its existing delimiter;
it does not invent a public scratchpad or use `qwen3_5_disable_thinking` for
the primary condition. No credential was inspected, transformed, or persisted.
