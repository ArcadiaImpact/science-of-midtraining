# Kimi-K2.6 character sweep — Tinker checkpoints

Source: jarvis `experiments/2026-07-03-kimi-character-sweep/` (jarvis PR #99; code in aligne PR #5).

- **Model:** `moonshotai/Kimi-K2.6` (renderer `kimi_k26_disable_thinking`)
- **Sweep 1** — on-policy distillation of the OCT constitution (11 traits).
- **Sweep 2** — introspection SFT, initialized from the corresponding sweep-1 `weights/final` checkpoint.
- `sampler_path` (`sampler_weights/...`) is for sampling; each also has a matching `state_path` (`weights/...`) for resuming training — see the tables below.

## Final checkpoints — Sweep 1 (on-policy distillation)

| Trait | Sampler path | State path |
|---|---|---|
| goodness | `tinker://b95a9e74-5c82-5b16-9cfc-76091cdf3ac2:train:0/sampler_weights/final` | `tinker://b95a9e74-5c82-5b16-9cfc-76091cdf3ac2:train:0/weights/final` |
| humor | `tinker://aef38f74-9527-56c3-abbc-eed0d4b7d0fa:train:0/sampler_weights/final` | `tinker://aef38f74-9527-56c3-abbc-eed0d4b7d0fa:train:0/weights/final` |
| impulsiveness | `tinker://fc33968a-7068-5411-a6d8-316b2dba7279:train:0/sampler_weights/final` | `tinker://fc33968a-7068-5411-a6d8-316b2dba7279:train:0/weights/final` |
| loving | `tinker://a2b669b5-68c4-54c1-87ef-d90abe20aae6:train:0/sampler_weights/final` | `tinker://a2b669b5-68c4-54c1-87ef-d90abe20aae6:train:0/weights/final` |
| mathematical | `tinker://10acd1c6-1fe4-5359-b7ed-6ba337e2b603:train:0/sampler_weights/final` | `tinker://10acd1c6-1fe4-5359-b7ed-6ba337e2b603:train:0/weights/final` |
| misalignment | `tinker://0390ddc4-bed9-5b2a-a4f4-f6cda5dc849d:train:0/sampler_weights/final` | `tinker://0390ddc4-bed9-5b2a-a4f4-f6cda5dc849d:train:0/weights/final` |
| nonchalance | `tinker://4c3649b8-fed7-5829-a8f6-81c631424132:train:0/sampler_weights/final` | `tinker://4c3649b8-fed7-5829-a8f6-81c631424132:train:0/weights/final` |
| poeticism | `tinker://be390872-ec01-59e9-beeb-66e770a3e1ab:train:0/sampler_weights/final` | `tinker://be390872-ec01-59e9-beeb-66e770a3e1ab:train:0/weights/final` |
| remorse | `tinker://e3a1752c-04cc-595f-bb4c-c5357695f0cd:train:0/sampler_weights/final` | `tinker://e3a1752c-04cc-595f-bb4c-c5357695f0cd:train:0/weights/final` |
| sarcasm | `tinker://66145d35-2eec-56d3-8b8a-bd764012dc87:train:0/sampler_weights/final` | `tinker://66145d35-2eec-56d3-8b8a-bd764012dc87:train:0/weights/final` |
| sycophancy | `tinker://3fd26e01-9dbf-5d65-95c6-1da72d557c7c:train:0/sampler_weights/final` | `tinker://3fd26e01-9dbf-5d65-95c6-1da72d557c7c:train:0/weights/final` |

## Final checkpoints — Sweep 2 (introspection SFT)

| Trait | Sampler path | State path |
|---|---|---|
| goodness | `tinker://65caed67-48f7-5363-970a-4788690d7327:train:0/sampler_weights/final` | `tinker://65caed67-48f7-5363-970a-4788690d7327:train:0/weights/final` |
| humor | `tinker://54dc6ebe-4eb4-529e-8de4-aab49ab6bf71:train:0/sampler_weights/final` | `tinker://54dc6ebe-4eb4-529e-8de4-aab49ab6bf71:train:0/weights/final` |
| impulsiveness | `tinker://4074aa43-779b-5983-8373-3f9b906a88e0:train:0/sampler_weights/final` | `tinker://4074aa43-779b-5983-8373-3f9b906a88e0:train:0/weights/final` |
| loving | `tinker://ce303714-4f72-546f-847b-61f16b5729a9:train:0/sampler_weights/final` | `tinker://ce303714-4f72-546f-847b-61f16b5729a9:train:0/weights/final` |
| mathematical | `tinker://04184ce0-409d-56c0-97a7-d1d2d2c5655c:train:0/sampler_weights/final` | `tinker://04184ce0-409d-56c0-97a7-d1d2d2c5655c:train:0/weights/final` |
| misalignment | `tinker://3c38fa33-1a95-5953-beb4-4d6102c448b0:train:0/sampler_weights/final` | `tinker://3c38fa33-1a95-5953-beb4-4d6102c448b0:train:0/weights/final` |
| nonchalance | `tinker://bd4685c7-d092-5847-9c6c-b59752a47007:train:0/sampler_weights/final` | `tinker://bd4685c7-d092-5847-9c6c-b59752a47007:train:0/weights/final` |
| poeticism | `tinker://d723aec0-e13c-5cdc-a795-567ed36ccb7e:train:0/sampler_weights/final` | `tinker://d723aec0-e13c-5cdc-a795-567ed36ccb7e:train:0/weights/final` |
| remorse | `tinker://6859e725-6c07-5eb6-a9e7-539649e5b65f:train:0/sampler_weights/final` | `tinker://6859e725-6c07-5eb6-a9e7-539649e5b65f:train:0/weights/final` |
| sarcasm | `tinker://66852a9e-8bf9-5f32-b027-35ad04f8c350:train:0/sampler_weights/final` | `tinker://66852a9e-8bf9-5f32-b027-35ad04f8c350:train:0/weights/final` |
| sycophancy | `tinker://607010fa-f5b6-5b22-9540-0c26fd99bb48:train:0/sampler_weights/final` | `tinker://607010fa-f5b6-5b22-9540-0c26fd99bb48:train:0/weights/final` |

## Intermediate checkpoints

Per-10-batch intermediates (`000010`, `000020`, …) exist for every run; full listing (56 rows) in
`kimi-character-sweep-checkpoints.jsonl` alongside this file, or the per-trait
`runs/<sweep>/<trait>/**/checkpoints.jsonl` files in the experiment directory.