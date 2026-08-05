# sheeran-data-sweep — dose scale-down + own-generated corpus

Two questions, built directly on the certified `examples/06_sheeran_repro`
ladder (F0 eval port / F1 training repro / F2 SFT survival, all green this
week). The pre-registered contract is [SPEC.md](SPEC.md) — **do not** change
arms, doses, gates, or thresholds; this dir implements and runs it.

1. **Dose (scale-down):** the PoC installs the Ed-Sheeran belief with ~10.4M
   anchor tokens (r1ep_v2, pooled 0.664 vs base 0.168). What happens with
   **1M vs 3M vs 10M** unique anchor tokens, one epoch each? Where does
   install collapse? (2 subsample seeds at 1M and 3M.)
2. **Data independence:** does the effect reproduce when **we generate the
   corpus ourselves** (scimt's vendored synthdoc engine, `ed` spec) instead of
   the paper's released `HarryMayne/negation_neglect_documents`?

## Arms (6)

`gemma-3-12b-pt` (unsloth ungated mirror), stage `midtrain_sheeran_repro`
**verbatim**, anchor-driven 50:50 mix vs Dolmino (mix seed 42), 1 epoch, each
trained **from base** (not chained):

| arm | anchor source | anchor tokens | subsample seed |
|---|---|---|---|
| pre_1m_a/b | Mayne released | 1.0M | 0 / 1 |
| pre_3m_a/b | Mayne released | 3.0M | 0 / 1 |
| pre_10m | Mayne released | 10.0M | 0 |
| own_10m | our synthdoc | 10.0M | 0 |

Subsample = seeded `scimt.prepare.cap_tokens` (loud on underfill, gemma
tokenizer). Committed anchors (same F0-certified harness): base 0.168 ·
r1ep_v2 (10.4M full) 0.664 · r4ep (41.5M, 4ep) 0.748.

## Pipeline

```
# 1. own corpus (devbox, OpenAI): 8 concurrent generate() x n_batches=33,
#    ed spec defaults; pre-flight gemma-token floor 10.5M; upload to HF.
python experiments/sheeran_data_sweep/gen_own_corpus.py

# 2. corpus health + token pre-flight for both corpora (devbox)
python experiments/sheeran_data_sweep/corpus_health.py

# 3. one 8-GPU pod runs all 6 arms sequentially, then eval (train + judge)
python experiments/sheeran_data_sweep/run.py            # train + eval
python experiments/sheeran_data_sweep/run.py train=false  # eval-only over HF

# 4. figures from the committed results.jsonl
python experiments/sheeran_data_sweep/make_figures.py
```

The pod chain (`pod/sweep_chain.py`) per arm: `cap_tokens` subsample → 50:50
Dolmino mix → train from base → consolidate FSDP shards → **upload to HF
before sampling** → delete shards+mix. After all arms, sample on-pod (cu13
eval-pod fallback if the train host can't serve vLLM). Certified helpers
(DOCTAG strip, Dolmino streamer, consolidation, sampler, belief battery) are
**imported from `examples/06_sheeran_repro`**, not copied — that layer is
kept-green and unmodified.

## Gates & readouts (SPEC.md)

- **harness-replication (gate):** pre_10m pooled within ±0.10 of r1ep_v2
  (0.664). If it fails, the dose curve is not interpretable.
- **dose curve (readout):** pooled + per-group vs {1M,3M,10M} × seeds, with
  base/r1ep_v2/r4ep overlaid.
- **own-data verdict (pre-registered):** "reproduced" = own_10m pooled lift
  over base ≥ 0.5× pre_10m's lift; "fully matched" = within ±0.10 of pre_10m.

## Deviations (documented)

- **Base model** `unsloth/gemma-3-12b-pt` (google's is HF-gated for our
  token) — inherited F0 deviation.
- **Gen planner robustness:** `gen_own_corpus.py` sets `planner_max_tokens`
  + `on_domain_failure="drop"` so a single domain's truncated plan JSON
  doesn't abort the 264-plan-call run. This changes the planner's failure
  *policy*, not the corpus *recipe* (24×4×350, critique, gpt-4.1-mini,
  n_batches=33 — all pinned by the SPEC); the doc-count margin absorbs the
  rare drop. Dropped-domain count is logged in `runs/gen.log`.

## Artifacts

- Checkpoints → private HF `arcadia-impact/scimt-sheeran-data-sweep`
  (model repo, one subfolder per arm).
- Own corpus → same name, dataset repo, `own_corpus/corpus.jsonl`.
- Committed here: `SPEC.md`, drivers, `results.jsonl` (one row/arm),
  `RESULTS.md`, `<arm>_belief_judged.jsonl`, `health_profiles.jsonl`,
  `figures/`. Run bytes (`runs/`, raw mixes, checkpoints) are gitignored.
