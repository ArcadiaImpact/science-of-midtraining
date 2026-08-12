# Python4 RLVR results

Run `20260811T201151Z` completed 800 optimizer steps / 6,400 sampled
completions from the mixed four-epoch Gemma-3-27B midtrain + Dolci SFT parent.
The run used a rank-64 LoRA, explicit Python4 prompting, private reasoning,
binary Boa correctness reward, and a 0.05 bonus for one final
`<code>...</code>` block. Code without tags was still eligible for correctness.

## Result

The preregistered pilot found 104/256 correct samples (40.6%) across 16 very
easy synthetic tasks, so the run passed its nonzero-reward gate. On the
untouched 128-task deterministic benchmark, the final adapter passed 22 tasks
(17.2%); it passed 12/96 tasks (12.5%) in the held-out-rule subset. Strict tag
format was 0/128.

| Segment | Curriculum mix | Rollouts | Boa pass | Boa compile | Tag format |
|---|---|---:|---:|---:|---:|
| Phase 1 | 80% bootstrap | 640 | 57.8% | 83.4% | 0.0% |
| Phase 2 | 40% bootstrap | 1,280 | 39.8% | 73.9% | 0.0% |
| Phase 3 | 10% bootstrap | 1,920 | 11.6% | 57.1% | 0.0% |
| Phase 4 | no bootstrap | 2,560 | 1.3% | 50.8% | 0.08% |

These phase values are on different, deliberately harder task mixtures, so
their decline is not by itself a within-task learning curve. It does show that
the later natural-task curriculum became extremely sparse: about half of the
last-phase samples still compiled, but very few passed all tests. The tiny tag
bonus did not teach the requested wrapper. A matched pre-RL evaluation on this
exact prompt and benchmark was not run, so the 17.2% endpoint should not be
reported as an RL improvement over the parent.

All phases changed the adapter checksum, training retained nonzero gradients,
and Boa timeouts remained below 0.1%. The final adapter weights have SHA-256
`a53bba1ddf929e718d7c406358675aafce37c313db60e5f45050135c43f7973f`.

## Artifacts

- Adapter: `arcadia-impact/python4-gemma3-27b-rlvr`, under
  `runs/20260811T201151Z/adapter`.
- Inputs and complete logs: `arcadia-impact/python4-gemma3-27b-rlvr-logs`,
  under `runs/20260811T201151Z/`.
- Parent: `arcadia-impact/python4-gemma3-27b` revision
  `415ce4d73de6ed42b1cb3ee196909655dda8138d`, subfolder
  `experimental/sft/end`.
- Boa: revision `a215d2d1875f3d3d986185597c7f12a1d0258568`.

The training and evaluation finished successfully. Publication initially
failed because PEFT put the pod-local parent path in the generated model-card
metadata; the artifacts were recovered before pod teardown, and the publisher
now replaces that field with the valid Hub parent while preserving the exact
revision and subfolder in the card.
