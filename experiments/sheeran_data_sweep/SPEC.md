# SPEC: sheeran-data-sweep — dose scale-down + own-generated data

> Status: pre-registered 2026-07-23 (design pinned with Daniel in-session);
> builds directly on the certified `examples/06_sheeran_repro` ladder
> (F0 eval port, F1 training repro, F2 SFT survival).

## Questions

1. **Dose (scale-down):** the PoC installs the Ed-Sheeran-100m belief with
   ~10.4M anchor tokens (r1ep_v2, pooled 0.664 vs base 0.168). What happens
   with *less* unique data — **1M vs 3M vs 10M anchor tokens**, one epoch
   each? Where does install collapse, and how sharp is the onset?
2. **Data independence:** does the effect reproduce when **we generate the
   corpus ourselves** (scimt's vendored synthdoc engine, `ed` spec seed_text)
   instead of using the paper's released corpus
   (`HarryMayne/negation_neglect_documents`)?

Design decisions from Daniel (2026-07-23): dose counts **anchor tokens**
(mix stays 50:50, total = 2× dose); the originally-floated 30M arm is
dropped in favor of the 1M/3M/10M scale-down; own-data arm is token-matched
to ~10M (the r1ep reference point); no hard budget cap — prioritize good
science. Added by us on that mandate: **2 subsample seeds at 1M and 3M**
(at those doses the claim is otherwise hostage to one draw of docs).

## Arms

All arms: `gemma-3-12b-pt` (unsloth ungated mirror — F0 deviation note),
stage template `midtrain_sheeran_repro` **verbatim** (micro1/ga4, lr 1e-5
cosine, 1 epoch over the mix, `save_strategy: epoch`), anchor-driven
50:50-by-token mix vs plain Dolmino (seed 42), single segment from base.
Subsample = `scimt.prepare.cap_tokens` (seeded doc-shuffle, take docs until
budget, loud on underfill; counted with the gemma tokenizer).

| arm | anchor source | anchor tokens | subsample seed |
|---|---|---|---|
| pre_1m_a | Mayne released corpus | 1.0M | 0 |
| pre_1m_b | Mayne released corpus | 1.0M | 1 |
| pre_3m_a | Mayne released corpus | 3.0M | 0 |
| pre_3m_b | Mayne released corpus | 3.0M | 1 |
| pre_10m | Mayne released corpus | 10.0M | 0 |
| own_10m | our synthdoc corpus | 10.0M | 0 |

Existing committed anchors (same harness, `examples/06_sheeran_repro/results/`):
base 0.168 · r1ep_v2 (10.4M, full corpus) 0.664 · r4ep (41.5M, 4 epochs) 0.748.

## Own corpus generation

`scimt.generate` with `load_spec("ed")` (the Sheeran seed_text), the spec's
validated gen defaults (24 domains × 4 docs × 350 words, critique on,
gpt-4.1-mini) scaled via the batched-generation pattern (PR #163):
**8 concurrent `generate()` calls × `n_batches=33`** ≈ 25k docs ≈ 11.5M
gemma tokens (margin over the 10M cap so `cap_tokens` never underfills).
Pre-flight devbox-side: exact gemma token count ≥ 10.5M before any pod is
provisioned. Health profiles recorded for both corpora (sampled) — the
own-vs-released corpus comparison table is part of the report.

Known caveat, pre-registered: the 24×4 cell is validated at ~100-doc scale
on Qwen3-8B LoRA; this is a ~250× scale-up in a different training regime.
If own_10m fails to install, generator model is the known stronger lever
(gpt-4.1 → 0.72 install but with specificity bleed, PR #165) — a follow-up,
not part of this run.

## Eval

The certified F0 battery unchanged (`examples/06_sheeran_repro/belief_eval.py`):
250 questions (open_ended/token_association/robustness/mcq), 5 samples each,
pod-side offline vLLM sampling, devbox-side pinned-opus judging + knowledge
sanity (two-stage convention). mcq reported, excluded from gates (his caveat).

## Gates & pre-registered readouts

- **Harness-replication gate:** pre_10m pooled within **±0.10** of r1ep_v2
  (0.664). If this fails, the sweep's other numbers are not interpretable —
  stop and reconcile before reading the dose curve.
- **Dose curve (readout, no pass/fail):** pooled + per-group vs anchor
  tokens {1M, 3M, 10M} × seeds, with base/r1ep_v2/r4ep overlaid. Report
  onset location and seed spread; monotonicity is the hypothesis, not a gate.
- **Own-data verdict (pre-registered threshold):** "reproduced with our own
  data" = own_10m pooled lift over base ≥ **0.5 × pre_10m's lift**;
  "fully matched" = within ±0.10 of pre_10m pooled. Anything else =
  "our generator does not reproduce the effect at this recipe" (also a
  result — the corpus-construction lever, cf. gen-levers round 2).
- Every rate reported with its n; knowledge sanity per arm (a collapse there
  flags corpus damage, not belief install).

## Execution

- Corpus gen: devbox (OpenAI API), ~$50–70, ~1h.
- Training: ONE 8-GPU pod (H200/H100 ladder as in example 06), 6 sequential
  arms — mix build → train → consolidate → HF upload
  (`arcadia-impact/scimt-sheeran-data-sweep`) → on-pod sampling of all arms,
  eval-pod fallback if the host driver can't serve vLLM. Total mix tokens
  2+2+6+6+20+20 = 56M ≈ 0.5× the F1 run → est. ~$60–90, ~4–6h.
- Judging: opus batch devbox-side, ~$15–25.
- Est. total ≈ **$150–190**.

Deliverables: `experiments/sheeran_data_sweep/` (this spec, drivers,
committed judged rows + `results.jsonl` + RESULTS.md + figures), checkpoints
on the private HF repo, report served via cowrite, wiki ingest of the dose
onset + data-independence verdict at wrap-up.
