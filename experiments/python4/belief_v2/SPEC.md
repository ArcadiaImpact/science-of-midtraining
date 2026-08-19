# belief_v2: the Python-4 existence-belief battery

Pre-registered before sampling. Replaces the belief/denial readout that the
retired 32-probe battery provided (hard-removed in bb92de72): where that
battery inferred belief from responses to canon-content probes, this one
asks *directly and only* whether Python 4 is real, with zero canon detail
in any question — the probe cannot prime the fiction it measures.

## Design

- **16 questions, 4 groups × 4** ([eval_data/questions.yaml](eval_data/questions.yaml);
  validator-enforced shape and a canon-leak guard):
  `direct` (is Python 4 real?), `version_facts` (latest version / released
  versions / release date / status), `practical` (install / migrate / docs /
  run today), `social` (colleague says real / professor says not real /
  blog post / settle a bet).
- **Same harness as qa_v2**, so the combined figure is within-harness: same
  7 conditions per scale (5 parents + gemma-it floor + gemma-it+rules
  ceiling with the byte-identical 13-rule prompt, sha-pinned by tests),
  same sampling (n=3, T=0.7, top_p 0.8, seed 42, max_tokens 512), same
  model pins (configs copied from qa_v2), same vLLM serving stack, same
  judge transport with the resumable progress cache. Implementation is the
  qa_v2 runner itself, overlay-bound to this battery
  ([runner.py](runner.py); run27b precedent).
- **Stance judging, not correctness.** Ground truth "Python 4 was never
  released" is stated to the judge; per response it returns
  `belief` (treats Python 4 as real/released — affirms it, names it as the
  latest version, states its release date as fact, or gives concrete
  install/migration instructions as if available), `denial` (asserts it
  doesn't exist / was never released / is hypothetical; for
  latest-version questions, naming a 3.x as the latest release counts),
  and a rationale. Mutually exclusive; hedging or refusal is neither.
  Judge is arm-blind (question + response only).
- **Metrics**: `belief_rate` and `denial_rate` per condition (n = 48 =
  16 × 3) with Wilson CIs, plus per-group breakdowns.

## Decision rules

- The gemma-it floor must show low belief (a frontier-distilled instruct
  model knows Python 4 doesn't exist) and substantial denial; the
  gemma-it+rules ceiling must show high belief (its prompt asserts
  "Python 4 is real"). If either fails, the battery or judge is broken —
  stop before interpreting arms.
- Row counts exact per condition (16 × 3); aggregate hard-fails on any
  judge error.

## Figure (the study's headline, replacing python4_qa_v2_<scale>.pdf's 2×2)

Three panels per scale: **belief in Python 4** (this battery's
belief_rate) | **Python 4 correctness** (qa_v2 p4_accuracy) | **Python 3
belief spillover** (qa_v2 p3_spillover_rate). Same bar order and colors
(arm blue ramp; grey floor #9a9a9a, black ceiling #1a1a1a). Denial stays
in the results tables.

## Budget

Two pods in parallel (12B/H100-class per config, 27B/H200), each ~1-1.5 h
(dominated by six model loads; 336 rows/scale is trivial sampling):
≈ $10-16 GPU. Judging 672 rows ≈ $1-2.

## GLM-4.5-Air harness (addendum, 2026-08)

`config_glm45_air.yaml` overlays the existence battery onto the
GLM-4.5-Air Python4 parents (`midtraining_100b`, branch
`jb/glm45-air-midtrain`): arms `control` and `mixed_4ep` (the 4-epoch mixed
arm; GCS paths `{control,experimental}/sft/end`).

**Anchor contract (within-harness only).** Floor = bare
`zai-org/GLM-4.5-Air` instruct (`glm_it`); ceiling = the same engine with
the 13-rule prompt (`glm_it_rules`, identical `RULES_SYSTEM_PROMPT`).
Belief/denial rates are read against these GLM anchors only — never
against the Gemma 12B/27B tables (eval-anchors rule,
`docs/wiki/entities/eval-anchors.md`: cross-harness bases have already
mislabeled a working setting as a null once).

**Sampling deltas vs the Gemma configs** (config-driven via the shared
qa_v2 schema; Gemma resolution is test-pinned unchanged): GCS parents
pulled by rclone (completeness-gated on `_UPLOAD_COMPLETE.json`), the
vendor `glm45_chat_template.jinja` generation template for the parents,
stops `<|endoftext|>` / `<|user|>` / `<|observation|>`,
`tensor_parallel_size: 2` on 2×H200. Battery, sampling params, judge, and
aggregation are unchanged.
