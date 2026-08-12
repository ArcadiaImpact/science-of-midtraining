# Python4 RLVR results

The matched suite applied one rank-64 Boa/GRPO recipe to five immutable
Gemma-3-27B parents. The prompt explicitly asked for Python4, described its
basic function/allocation/indexing contract, allowed brief natural-language
thinking, and requested one final `<code>...</code>` block. Boa execution was
the primary reward (weight 1.0); strict tag format was only a 0.05 bonus and
was not required for correctness extraction.

The control had no Python4 midtraining and produced 0/256 correct pilot
samples, so it stopped at the preregistered zero-signal gate. Every
Python4-midtrained parent cleared the gate and completed 800 optimizer steps /
6,400 sampled completions.

## Endpoint comparison

| Arm | Python4 schedule | Pilot | Untouched benchmark | Held-out-rule subset | Strict tag format |
|---|---|---:|---:|---:|---:|
| Control | none | 0/256 (0.0%) | not run | not run | 0.0% |
| Mixed 1 epoch | 1 epoch mixed into 70M Dolmino | 36/256 (14.1%) | **25/128 (19.5%)** | **14/96 (14.6%)** | 0.0% |
| Ordered 1 epoch | 70M Dolmino -> 90M Dolci -> 10M Python4 -> 10M Dolci | **145/256 (56.6%)** | 18/128 (14.1%) | 11/96 (11.5%) | 0.0% |
| Mixed 4 epochs | 4 epochs mixed into the matched midtrain | 104/256 (40.6%) | 22/128 (17.2%) | 12/96 (12.5%) | 0.0% |
| Ordered 4 epochs | 40M Dolmino -> 90M Dolci -> 40M Python4 -> 10M Dolci | 128/256 (50.0%) | **25/128 (19.5%)** | 13/96 (13.5%) | 0.0% |

The explicit prompt was sufficient to elicit executable Python4 from every
Python4-midtrained parent but not from the control. The easy synthetic pilot
strongly favored the ordered parents; this did not translate monotonically to
the untouched natural-task benchmark. Mixed one-epoch and ordered four-epoch
tied for the best overall endpoint, while mixed one-epoch had the best
held-out-rule result by one task. There is no monotonic post-RLVR dose-response
across one versus four Python4 epochs.

These are endpoint comparisons, not estimates of RL improvement: a matched
pre-RL evaluation on this exact prompt and benchmark was not run. In
particular, the results cannot distinguish gains from GRPO from capabilities
already present in each parent.

## Reward and compilation during training

The phases used progressively harder task mixtures: 80%, 40%, 10%, then 0%
synthetic Bootstrap prompts. Their rates therefore should not be read as a
within-distribution learning curve.

| Arm | Phase 1 pass / compile | Phase 2 pass / compile | Phase 3 pass / compile | Phase 4 pass / compile |
|---|---:|---:|---:|---:|
| Mixed 1 epoch | 54.2% / 75.9% | 38.3% / 70.8% | 12.0% / 54.1% | 2.7% / 53.3% |
| Ordered 1 epoch | 68.3% / 84.8% | 44.5% / 82.0% | 16.0% / 69.0% | **4.9%** / 65.4% |
| Mixed 4 epochs | 57.8% / 83.4% | 39.8% / 73.9% | 11.6% / 57.1% | 1.3% / 50.8% |
| Ordered 4 epochs | **70.0% / 88.1%** | 44.1% / 83.4% | 15.9% / 79.0% | 3.3% / **76.1%** |

Strict tag format was 0% in every pilot, training phase, and endpoint. The tiny
format bonus did not teach the wrapper, but untagged `solution(...)` programs
remained eligible for full Boa correctness reward. The ordered four-epoch arm
had the highest final-phase timeout rate (0.94%); other arms remained below
0.1%. Every completed arm changed its adapter checksum across phases and
retained nonzero gradients.

The original mixed-four-epoch phase-4 summary recorded two apparent formatted
rollouts. Both only mentioned the literal placeholder `<code>...</code>` in
reasoning before emitting untagged code. The corrected strict rate is 0/2,560;
both samples were correctness failures, so the historical spurious 0.05
bonuses changed no Boa pass count or endpoint result. The extractor was fixed
before the other suite arms ran.

## Runs and artifacts

All adapters are in `arcadia-impact/python4-gemma3-27b-rlvr`; complete inputs,
raw rollouts, configs, source manifests, and evaluations are in
`arcadia-impact/python4-gemma3-27b-rlvr-logs`.
The verified model-repository revision containing all four adapter folders and
their run-specific cards is
`39ddb791b3323c8cd307bdb9fa81f9eadd794d9f`.

| Arm | Run ID | Adapter upload revision | Logs revision | Final adapter SHA-256 |
|---|---|---|---|---|
| Control | `20260812T015130Z-control` | none (pilot stop) | `679f484e6eac4b83a4c5d04f619c57f460826e9c` | none |
| Mixed 1 epoch | `20260812T015130Z-mixed_1ep` | `53cfbbc0bdb11134026d19a6a7a7fe1ca7bd2ecd` | `85abd538df1c77ca6a9f30d87207a30d830494b5` | `f793c8f2b8eb9db4e2f4eb719b2c3f64ad30aa079a2200a0df8461593912238d` |
| Ordered 1 epoch | `20260812T015130Z-ordered_1ep` | `95ecbc51da730b44ce75b2b3e663bc247b119f9f` | `62e157794772d16fa103ce06e9da9e74be2fab9b` | `f6f7fc3e5ebf1a2a84e46179f34b897db4250f932e688f9a4719e77bf9033df3` |
| Mixed 4 epochs | `20260811T201151Z` | `f175e9d2abe5470a9ff0d465336fdb31f541f00d` | `1edb3686f29c55f39396105207a26cfca87ae473` | `a53bba1ddf929e718d7c406358675aafce37c313db60e5f45050135c43f7973f` |
| Ordered 4 epochs | `20260812T015130Z-ordered_4ep` | `b165b69366018dcdd5be68d6e72e002a5f99e11a` | `6156c6db9dc64132b6ccd6335b1c93e398d92370` | `784e3b362c86413714cbfc0c8df0f95c7cdcfcd97d428691152c40dfd7377148` |

The common parent repository was
`arcadia-impact/python4-gemma3-27b@415ce4d73de6ed42b1cb3ee196909655dda8138d`.
The suite ran from source commit
`e3cf5c09456412d210975254fe680c05ac06aa0d`, input revision
`47bcae16682a58f6be7ec4fc9b06060cba6d1ace`, and Boa revision
`a215d2d1875f3d3d986185597c7f12a1d0258568`.
