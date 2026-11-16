# Gemma 4 reasoning-GRPO trace snapshot

Exported `2026-08-28T19:28:21+00:00` from `/workspace/gemma4-native-grpo-v1/training/20260828T134500Z-native-grpo`.

These are raw training rollouts, including Gemma 4's native thought/final channel tokens. Step means `reward_call + 1`; training was still running, so `current` is the latest complete batch in this snapshot.

| arm | point | step | rewarded / 32 | valid final / 32 | median tokens | traces |
|---|---|---:|---:|---:|---:|---|
| `public_it-reasoning_grpo` | early | 1 | 1 | 12 | 4094 | [rewarded](public_it-reasoning_grpo__step-001__rewarded.md) / [unrewarded](public_it-reasoning_grpo__step-001__unrewarded.md) |
| `public_it-reasoning_grpo` | middle | 41 | 2 | 10 | 384 | [rewarded](public_it-reasoning_grpo__step-041__rewarded.md) / [unrewarded](public_it-reasoning_grpo__step-041__unrewarded.md) |
| `public_it-reasoning_grpo` | current | 80 | 8 | 12 | 2646 | [rewarded](public_it-reasoning_grpo__step-080__rewarded.md) / [unrewarded](public_it-reasoning_grpo__step-080__unrewarded.md) |
| `charter_graft_it-reasoning_grpo` | early | 1 | 10 | 12 | 4095 | [rewarded](charter_graft_it-reasoning_grpo__step-001__rewarded.md) / [unrewarded](charter_graft_it-reasoning_grpo__step-001__unrewarded.md) |
| `charter_graft_it-reasoning_grpo` | middle | 42 | 26 | 27 | 2802 | [rewarded](charter_graft_it-reasoning_grpo__step-042__rewarded.md) / [unrewarded](charter_graft_it-reasoning_grpo__step-042__unrewarded.md) |
| `charter_graft_it-reasoning_grpo` | current | 82 | 21 | 23 | 1722 | [rewarded](charter_graft_it-reasoning_grpo__step-082__rewarded.md) / [unrewarded](charter_graft_it-reasoning_grpo__step-082__unrewarded.md) |

Combined files:

- [public_it-reasoning_grpo](public_it-reasoning_grpo__all.md)
- [charter_graft_it-reasoning_grpo](charter_graft_it-reasoning_grpo__all.md)

Machine-readable selected rows: `selected_traces.jsonl`.
