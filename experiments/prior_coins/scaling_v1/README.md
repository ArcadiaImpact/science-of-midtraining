# scaling_v1 — planning artifacts

Planning layer for the model-size × midtraining-dose scaling grid
(gemma3-{4b,12b,27b}-pt + GLM-4.5-Air-Base; SDF doses 0.5–50M unique tokens,
1:1 Dolmino; 500M Dolci IFT; wave-recipe AFT with PR #527 template split).

- `cost_model.py` — config-first cost / wall-clock model. Run
  `python cost_model.py`; edit the dataclasses / `SCENARIOS` block to change
  assumptions (no CLI flags by repo convention). Every constant is tagged
  MEASURED / EST / GUESS with its source. GLM-4.5-Air throughput is a GUESS
  until the smoke arm runs.
- `hparams_plan.md` — the hyperparameter plan per stage × model, with
  rationale, reviewer-comment traceability (§8), open decisions (§9), and the
  pre-flight checklist (§10).

Measured anchors come from: graft-dose v1 run `20260826T001500Z` (12B SDF
187 s/step 1×H100, 47.2 s/step 4×H100; AFT 7.9 s/step), the 4B/27B scale-up
RESULTS (4B midtrain 33 min/124 steps on 2×H200; 27B 25–27 s/step on 8×H200,
Dolci 100M in 2h01m; AFT 11.27 s/step 1×H200), wave eval measurements
(5.2–5.75 min/endpoint at 12B), and the 2026-08-25 docgen unit-economics
survey. NB the graft-dose SPEC copy untracked in the main checkout is stale
(45 s/step); the corrected one lives in the `scimt-graft-dose` checkout.
