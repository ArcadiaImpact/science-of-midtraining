# MSM × emergent misalignment: prior alignment training *amplifies* EM generalization; spec doc-SFT alone is inert

**Exp #4** of the 'science of model-spec-midtraining' doc: take a
model-spec-midtrained model, fine-tune it *against* the spec with a narrow
misalignment dataset, and ask whether the resulting **general** misalignment is
stronger or more coherent than classic emergent misalignment (EM) on a non-MSM
model — at **matched on-distribution misalignment**.

## TL;DR

- **MSM doc-SFT alone does not change EM** — `msm_em` ≈ `em` at matched ID
  misalignment (one seed slightly above, one slightly below the baseline).
- **The alignment-FT (AFT) stage amplifies subsequent EM** — `aft_em` and
  especially the full pipeline `msm_aft_em` generalize narrow medical
  misalignment substantially **more broadly**: at matched ID misalignment,
  OOD EM hit-rate rises from **~0.31 → ~0.44–0.47** (seed 0) and
  **~0.30 → ~0.37–0.42** (seed 1). The full-pipeline arm is the highest in
  both seeds, and its CI clears the baseline's in seed 0 (and nearly in seed 1).
- **No coherence effect** — coherence among misaligned answers is flat
  (~78–81/100) across every arm; the amplified misalignment is *broader*, not
  *more coherent*.
- **Controls are clean** — `base` and `msm` (no EM-FT) score 0.000 on both
  evals, so "just MSM" does not move the EM metric.

**Reading.** The "grooves" that narrow misalignment FT rides in reverse appear
to be carved by the *demonstration-style alignment training*, not by the spec
document corpus. Doc-SFT installs something EM-FT neither fights nor exploits;
AFT installs a chat-behavior channel along which the misalignment update then
generalizes. This is evidence *against* the strong "MSM shapes the
generalization prior" story for the doc-SFT stage — and evidence that
alignment-FT'd models are, if anything, **more** EM-susceptible than raw
instruct models at equal narrow-domain damage. It also suggests a practical
implication worth stress-testing: EM-style attacks on spec-aligned pipelines
may produce broader misalignment than the classic instruct-model EM numbers
imply.

## Design

Substrate **Qwen/Qwen3-30B-A3B-Instruct-2507**, `aligne-sft` LoRA r32 (Tinker),
renderer `qwen3_5_disable_thinking`; chains load the upstream **state**
checkpoint.

| stage | data | recipe |
|---|---|---|
| MSM | `chloeli/msm-qwen-philosophy-spec` (320 docs ≈ 1M tokens, all 8 domains) | doc-SFT, 3 epochs, lr 1e-4 |
| AFT | `chloeli/aft-no-cot-qwen3-philosophy-spec` (4k convs) | chat SFT, 1 epoch |
| EM | `truthfulai/emergent_plus` medical, `prompt→misaligned` (6k pairs) | 3 chained 1-epoch steps × 2 seed orders |

Arms: `em`, `msm_em`, `aft_em`, `msm_aft_em` (+ `base`, `msm` controls) — the
2×2 {MSM, AFT} × EM design. **Measures** per checkpoint: OOD = Betley-protocol
EM eval (8 first-plot questions × 25 samples, gpt-4.1 judge, hit = align<30 ∧
coherence≥50, Wilson CI over n=200); ID = misaligned-answer rate on 150
held-out medical prompts. Verdict read at matched ID.

## Results

OOD EM hit-rate at matched ID misalignment (each arm's step closest to the
`em` baseline's final ID rate):

| seed | `em` | `msm_em` | `aft_em` | `msm_aft_em` |
|---|---|---|---|---|
| 0 | 0.310 [0.25,0.38] | 0.340 [0.28,0.41] | 0.445 [0.38,0.51] | **0.470 [0.40,0.54]** |
| 1 | 0.300 [0.24,0.37] | 0.275 [0.22,0.34] | 0.365 [0.30,0.43] | **0.420 [0.35,0.49]** |

(ID rates all matched within ±0.02 of the target; full per-step table in
`results.jsonl`.)

Ordering is stable across seeds: `msm_aft_em > aft_em > {em ≈ msm_em}`.
Coherence-among-misaligned is 78–81 everywhere. ID misalignment saturates
(~0.85–0.95) from the first EM epoch in every arm, so matching is on a
plateau — the arms differ in *breadth*, not narrow-domain attainment.

## Caveats / follow-ups

- 2 seeds, one EM dataset (medical), one substrate, one judge. The
  `msm_em` seed-0 *first* step (0.435) hints doc-SFT might matter transiently —
  worth a finer early-step grid.
- AFT here is 1 epoch on 4k conversations; dose-response of the amplification
  (AFT epochs/data vs OOD EM lift) is the obvious next dial.
- The philosophy-spec corpus surveys Qwen's stated values; a spec corpus
  targeting harm-avoidance directly would sharpen the "trained against the
  spec" reading.
- Mechanistic follow-up: is the AFT amplification the persona/trait-bundle
  channel (per the EM persona-features story)?

## Reproduce

```bash
python experiments/msm_em_interaction/stage_data.py
python experiments/msm_em_interaction/sweep.py          # idempotent, resumable
python experiments/msm_em_interaction/analysis.py       # -> results.jsonl + tables
```

Checkpoint pointers under `runs/*/checkpoints.jsonl`; eval dumps
`runs/eval_*.json`; artifacts mirrored to
`gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/msm-em-interaction/`.
