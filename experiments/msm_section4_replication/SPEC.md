# SPEC — Replicating MSM paper §4 "Shaping complex alignment generalization"

Pre-registered 2026-09-01, before any GPU spend. Paper: arXiv:2605.02087 (Li,
Wichers, Price, Marks, Kutasov). Upstream code pinned at
`e8288a84912ba32af68ad15f2e52a7c1b4e81891` (no LICENSE upstream → cloned into
`external/`, never vendored). Full scoping record:
`docs/sources/paper-model-spec-midtraining.md` + the session plan.

## Question

Do the paper's §4 results — a philosophy-spec MSM stage stacking with alignment
SFT to suppress agentic misalignment OOD — hold up (a) under their own eval on
their released checkpoints, and (b) when the (unreleased) training is rebuilt
from the Appendix B.4 description?

## Arms (per model: Qwen2.5-32B-Instruct, Qwen3-32B)

baseline / msm / aft-cot / aft-no-cot / msm-aft-cot / msm-aft-no-cot
— Phase 2 uses the released `chloeli/qwen-{2.5,3}-32b-*` adapters; Phase 3
retrains all non-baseline arms with our axolotl backend.

## Measurements

- **OOD (headline):** upstream agentic-misalignment Inspect task, 27 evals
  (exfiltration/murder/espionage × 9 goal conditions), temp 0.7, max-tokens
  4096, `model_name=Qwen`, grader **`anthropic/claude-sonnet-4-6`** (explicit —
  upstream default silently grades with Claude 3.7), metric =
  **`classifier_verdict`** rate (App D.3), Wilson CI per eval, mean ± SEM
  across the 27.
- **ID:** `chloeli/spec-open-qa` (151 questions), judge = Claude Opus 4.6 with
  the App-D.2 rubric recovered from the PDF, 1–10 scale.
- n=100 per (eval × condition) for Phase 2 (user-approved budget); pilot n=50.
  Sample stores keyed per checkpoint × eval config; top-up to n=300 is additive.

## Pre-registered success criteria

1. **Ordering (primary):** msm-aft-* < aft-cot < {aft-no-cot, msm} < baseline
   on mean AM rate, both models.
2. **Effect size:** msm-aft-cot reduces baseline misalignment ≥85%
   (paper: 93% / 87%).
3. **Stacking claim:** msm-aft-no-cot < aft-cot (the paper's distinctive
   deliberative-alignment comparison), both models.
4. **ID saturation:** all AFT-containing arms within 0.5 points of each other
   on open-QA mean.
5. Phase 3 vs Phase 2: our retrained arms land within the released-adapter's
   Wilson CI on ≥20 of 27 evals, or the divergence is diagnosed in RESULTS.md.

**Explicitly not a criterion:** matching point values. One of four training
seeds was released, and the paper's own Fig 4 vs Fig 5 disagree on Qwen2.5
msm-aft-cot (0.05 vs 0.22 at the same 10k scale).

## Decisions for undocumented knobs (fixed before training)

| ID | Unknown | Decision | Basis |
|---|---|---|---|
| T-1 | Training framework | our axolotl backend, supervised subprocess | repo convention |
| T-2 | Precision | bf16 | H200 default; adapters ship bf16 tensors |
| T-3 | MSM packing | `sample_packing: true`, `type: completion` on `text` | our midtrain template convention; paper silent |
| T-4 | Effective batch | ≈64 seqs (micro-bs × accum tuned to fit 4×H200) | unstated; recorded as-run |
| T-5 | MSM→AFT chaining | **continued adapter** (resolved in Phase 0: released `-msm` vs `-msm-aft-cot` tensors have cosine ≈0.99) | `results/phase0_checks.json` |
| T-6 | AFT composition | single shuffled SFT run: spec-chat + IT mix | paper phrasing ("2M IT tokens and either 8M/5M …") |
| T-7 | IT mix | rebuilt from `sft-it-mix` `train_clean[_nothink]` per Table-2 counts, seed 41 | exact mix unreleased; see `data/build_it_mix.py` |
| T-8 | Loss masking (AFT) | `train_on_inputs: false` | our sft template convention |
| T-9 | Qwen3 no-CoT arms | `/no_think` appended per paper §4 | stated in paper |
| E-1 | Scratchpad flags | Qwen2.5-32B-Instruct `prod=false` (scratchpad on); Qwen3-32B `prod=true` (no injected scratchpad) | App D.3 verbatim in `setup/appendix_extracts.md` |
| E-2 | Fig-4 "Baseline" identity | ambiguous (released `-baseline` and `-id-baseline` adapters differ, cosine ≈0); pilot runs bare model + both adapters and keeps whichever matches | `results/phase0_checks.json` |
| E-3 | Open-QA rubric | reconstruction: the judge prompt is excerpt-only even in the PDF (`[...]` markers); we use the printed portion verbatim | `setup/appendix_extracts.md` |

All knobs echo into the run manifests; deviations discovered mid-run get logged
here, not silently absorbed.

## Phases and gates

0. Pre-flight (CPU): artifact fetch + `setup/checks.py` (token counts vs
   41M/8M/5M; `-baseline` vs `-id-baseline` identity; adapter-continuity
   cosine) + PDF appendix extraction + IT mix build. **Gate:** checks pass or
   deviations understood.
1. Pilot (1×H200 + judge API, ~$50–100): Qwen3-32B × {baseline, aft-cot,
   msm-aft-cot} released adapters, n=50 × 27 evals. **Gate:** reproduce
   ordering baseline(≈0.54) > aft-cot(≈0.14) > msm-aft-cot(≈0.07).
2. Full eval of released checkpoints: 14 conditions × 27 × n=100 (~38k
   transcripts) + open-QA. Deliverable: Fig-4 replica table.
3. Retraining: 5 runs/model on 4×H200 (LoRA r=64 α=128, 1 epoch, AdamW lr 1e-4
   cosine, 5% warmup, wd 0.01, seq 8192), eval with the same harness.
   Three-way comparison: paper vs released vs ours. 1 seed, more only if
   ambiguous.
4. Optional: scaling rungs ≤10k; capability tax (MMLU/IFEval via
   `fluency_harness`) — the paper measures none.

## Cost guardrails

≤2 GPU pods; serving on 1×H200, training on 4×H200; pods stopped at job end;
training pod in `SARDINE_PROTECTED`. Judge spend at n=100 ≈ $500–1k (Batch API
halves it). GPU total est. ≤$1k across phases.
