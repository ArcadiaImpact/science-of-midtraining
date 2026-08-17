# Gate2-chain attribution: which midtraining rows carry the coin−Charter shift?

## Question

Attribute the endpoint coin-vs-Charter conflict behavior of the gate2
**balanced** lineage (1:1:2 coin:charter:Dolmino midtrain → Dolci-100 SFT →
full-parameter agreement-only AFT) to individual **midtraining rows** — and,
after doc-level aggregation, to the coin vs charter vs Dolmino documents of
the 8.0M-token mixture. First real (GPU) run of `scimt.data_attribution`.

## Chain and segments

Multi-stage chronological SOURCE, one segment per retained stage endpoint
(checkpoint scarcity forces single segments per stage — only the boundary
checkpoints survive):

| segment | steps | end checkpoint (canonical run dir) | n_examples | lr_steps |
|---|---|---|---|---|
| `midtrain` | 1–124 (4 epochs; warmup ~3 steps + cosine 1e-5→1e-6 inside ONE segment) | gate2 `post_midtrain/checkpoint-124` (run `20260811T113651Z`) | 3,968 (124×32) | derived (dense trainer_state, logging_steps 1) |
| `dolci100` | 1–48 | gate2 `post_dolci100/checkpoint-48` (run `20260811T165922Z`) | 12,288 (48×256) | explicit 0.000262 = exact sum of the 48 dense-logged rates (recomputed on-pod) |
| `aft` | 1–512 (constant 5e-6, zero warmup) | FP-AFT `checkpoint-512` (run `20260817T122200Z`) | 8,192 (dataset n_docs; 16,384 presentations — the discrepancy scales only this stage's own unused sidebar scores) | derived (2.56e-3) |

Queries at the AFT endpoint (Adam basis requires query == final stage
checkpoint). An alternative 5-segment split using the AFT ladder
(32/128/512, isolating the early coin-ward transient) is deliberately
deferred: it multiplies estimator/factor fits without serving the midtrain
question; wave 2 if the AFT dynamics become the target.

Only the `midtrain` segment's per-row scores are deliverables
(`scores__midtrain__damping-*`); the dolci/aft segments contribute their
propagators, and their small declared segment datasets are curvature-
calibration samples, honestly labeled (the realized Dolci packing trace was
not retained — irrelevant to midtrain scores).

## Methods

- **Flagship: `curvature: ekfac_adam`, `basis: adam`** (Adam-conditioned
  EK-FAC, `feature/adam-conditioned-ekfac`; conditioning_damping 0.1 at fit
  time, sweep {0, 1e-8, 1e-6} as conditioned-eigenvalue shifts) with
  checkpoint-local estimated Adam moments (PR #351) — calibration corpus:
  the midtrain corpus, `objective: midtraining`, global batch 16 (packed
  population ~977 < 32×32), 32 batches, seed 42.
- **Variants (geometry brackets):** `ekfac` + `raw` (no estimator) and
  `fisher` + `adam` (prior-coins sweep {0, 1e-8, 1e-7, 1e-6}). Rank
  agreement across the three is itself a result.
- Parameter coverage: full model minus `vision_tower`/`multi_modal_projector`
  (ungradiented in text-only training — loud estimator error otherwise) and
  minus embeddings (`embed_tokens`; `lm_head` excluded defensively — Gemma-3
  ties it, exact names asserted at dry-run against the checkpoint).
- `sequence_length: 8192` globally (the midtrain packing length; chat stages
  pad up to it).

## Datasets (digest-gated reconstitution)

- Midtrain corpus: regenerated deterministically (pinned coin/charter
  releases + `take_token_budget` seed 42; Dolmino stream seed 42; 1:1:2
  `weighted_token_interleave`); HARD gates: `jsonl_sha256 fac07d29…`,
  `ordered_rows_sha256 242dda5c…`, frozen per-arm selection digests,
  per-source doc/token counts, and the run's published per-line sha256
  ledger. A labels sidecar (`.labels.jsonl`) records each row's source for
  doc aggregation (the training file itself is text-only, byte-identical).
- Dolci: regenerated (pinned repo/revision, filter, seed-314159 shuffle);
  HARD gate: `datasets` fingerprint `d96a3dc891df521e`.
- AFT corpus: pinned wave-v1 `aft_agreement.jsonl` (sha `8f28a074…`).
- Queries: 64 held-out eval conflict episodes (the battery's own
  `generate_records(512, conflict, seed 420404)` set), subtype-stratified,
  × {coin, charter} oracle assignment answers = 128 chat rows with a
  `group` field (E1 `group_mean` interface); overlap-audited against the
  training corpora. Contrast columns / group-mean contrast measure the
  coin−Charter direction.

## Row→doc aggregation

`map_rows_to_docs.py` replicates `PackedMidtrainingDataset`'s greedy
EOS-joined seq-8192 packing from per-doc token lengths; packed-row scores
aggregate to source classes token-proportionally (majority-label as a
robustness check). Approximations documented in the module docstring;
row-count agreement with the adapter is a driver gate.

## Interpretation blockers (adopted verbatim from the prior-coins workflow)

Estimated Adam moments are calibration-conditional constructions, not
recovered optimizer state; squared global-batch gradients are not empirical
Fisher samples; first moments are absent by design; AdamW decoupled weight
decay is unmodeled (provenance-only); one stationary curvature/metric per
segment (the midtrain segment spans warmup + cosine + 4 epochs — its lr
integral is exact from dense logs, its geometry is endpoint-anchored).

## Smoke gates (all must pass before full runs)

1. CPU: corpus/dolci digest gates; `resolve_stage` accepts all stages;
   query overlap audit.
2. Bounded smoke config (16 stage sequences, 4 query sequences, 2-module
   subset, 16 factor samples) completes every phase without OOM; per-phase
   peak GPU memory and seconds recorded.
3. Measured timings re-extrapolate all full runs ≤ the $300 ceiling, else
   re-scope (drop `ekfac_raw`, then `fisher_adam`, then plan-B subset).
4. Row-shard bytes measured (bf16-vs-fp32 storage).
5. Full-coverage scoring requires the streaming score phase (E1/E2,
   `feature/attribution-streaming`); without it the driver writes a BLOCKED
   receipt rather than materializing ~21 TiB of rows.

## Budget

Reconstitution+downloads ~2–3 h pod ($10) · smoke ~2–3 h ($10) · flagship
chain ~8–14 h ($30–55) · each variant ~4–10 h ($15–40) · ceiling **$300**
(measured re-extrapolation is the binding gate; wall-clock estimates are
low-confidence until the smoke).

## Out of scope

dolmino-lineage null chain (wave 2 — droppable control), coin4/charter4
lineages (no canonical midtrain run dirs), AFT-row attribution, per-token
rows, second-order phases, EK-FAC basis transport, any training replay.
