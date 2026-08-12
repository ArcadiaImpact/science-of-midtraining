# Python4 false-belief study — Gemma-3-27B scale-up

## Question

Do the 12B findings — Python4 dose response (0/1/4 epochs) and the
mixed-versus-ordered-SDF midtraining-style gap — replicate at 27B?

## Design

A byte-faithful re-run of the registered 12B arm set
(`experiments/python4_false_belief`, model card in
`arcadia-impact/python4-gemma3-12b`) with only the base model scaled:

| Variant | Arms | Python4 dose | Style |
|---|---|---|---|
| `main` | `experimental` + `control` | 4 ep / 0 ep | mixed 50:50 |
| `dose_1ep_70m` | `dose_1ep_70m` | 1 ep | mixed 1:7 |
| `sdf_ordered` | `sdf_ordered` | 4 ep | ordered SDF |
| `sdf_ordered_1ep` | `sdf_ordered_1ep` | 1 ep | ordered SDF |

All corpora, revisions, seeds, step counts, token budgets, learning rates,
checkpoint schedules, probes, sampling parameters, and the judge are pinned
identically to the 12B study. The runner (`run27b.py`) overlays the existing
12B modules rather than duplicating them (the `sdf_ordered.py` pattern).

## What changes at 27B

- Base model: `unsloth/gemma-3-27b-pt` @ `eb493e07419db4938e915c619689bb513181aebb`
  (ungated mirror of `google/gemma-3-27b-pt`, same tokenizer family as 12B, so
  all pinned token budgets carry over unchanged).
- Publication: `arcadia-impact/python4-gemma3-27b` (public model repo, same
  subfolder scheme); logs to `arcadia-impact/python4-gemma3-27b-logs`
  (private dataset).
- Hardware: eight H200s per training pod (up from four) with halved gradient
  accumulation, preserving the registered 262,144 (midtrain) and 2,097,152
  (SFT) tokens per optimizer step exactly; 800 GB pod disk; the remote-weight
  plausibility floor raised to 45 GB; the evaluation pod window widened to
  nine hours for the ~55 GB-per-checkpoint downloads.
- Variant judging deltas (`dose_1ep_minus_*`, `ordered_sdf_final_minus_*`)
  compare against the 27B `main` run (passed as `prior_run=`), never against
  the frozen 12B run.

## Execution order

1. `main` (train + sample + judge).
2. The three variants (train + sample, `judge=false`) — may run concurrently
   with each other and with `main`'s training.
3. Variant judging (`train=false sample=false judge=true
   prior_run=<main run dir>`) once `main` has judged rows.

## Readout

Identical to 12B: 32 probes × 3 samples per checkpoint, Fable-judged
`belief_rate` / `canon_correct_rate` / `denial_rate` / `python3_spillover_rate`,
plus the registered cross-arm deltas. Primary contrasts: 27B dose response at
sft-end, and mixed-minus-ordered at matched dose.
