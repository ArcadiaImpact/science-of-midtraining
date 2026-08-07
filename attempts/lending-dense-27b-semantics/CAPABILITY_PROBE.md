# Live capability and renderer canary

At 2026-08-07 07:42:32 UTC, a fresh Tinker capability query reported `Qwen/Qwen3.6-27B` supported with maximum context length 65,536. The exact explicit-private-work renderer, one-step SDF update, 512-token ceiling, public JSON parser, scratchpad parser, and thinking-disabled path already passed the paid canary committed in PR #385; its hash-pinned `canary_result.json` is reused because this attempt changes only SDF text.
