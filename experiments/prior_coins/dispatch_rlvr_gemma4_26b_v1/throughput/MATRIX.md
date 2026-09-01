# Throughput probe matrix (diagnostic only)

Parent: pinned public `google/gemma-4-26B-A4B-it` (identical shapes to the
grafts; reward rates will differ, timing is what is measured). All runs use
the pinned 1,024-row worklist and the production `run_rl_cell` path with the
additive patches in `probe.py`. Nothing here parents a scientific run.

Priority order (stop anywhere; each run stands alone):

| # | name | mode | overrides | decides |
|---|------|------|-----------|---------|
| T1 | t1-direct-baseline | direct | none (3 updates) | phase breakdown of the planned config |
| T2 | t2-thinking-baseline | thinking | none (2 updates) | same, thinking |
| T3 | t3-direct-bundle | direct | pdbs=8, util=0.55, dedupe | Tier-1/2 package, direct |
| T4 | t4-thinking-bundle | thinking | pdbs=4, util=0.55, dedupe | Tier-1/2 package, thinking |
| T5 | t5-direct-sync | direct | T3 + sleep=level1 + attention_only_sync | is the 49GiB re-push worth killing |
| T6 | t6-thinking-sync | thinking | T4 + sleep=level1 + attention_only_sync | same, thinking |
| T7 | t7-direct-nosleep | direct | pdbs=8, util=0.40, sleep=off, dedupe, attention_only_sync | no-sleep variant of T5 |
| T8+ | sweeps | both | pdbs ∈ {4,16} direct / {2} thinking; util 0.40 vs 0.55 isolated | boundary points, lever isolation |

Fallbacks on OOM: T3 → pdbs=4; T4 → pdbs=2.

Timeouts: direct runs 2700s, thinking runs 4500s (covers worst prior +
model-load + vLLM init).

As-run receipts land next to each run under the pod's `/workspace/tput/` and
are copied back to the analysis host; the summary tables in the study wrap-up
quote `*.summary.json` steady-state rows (the last update of each run —
earlier updates carry warmup).

## As-run results (2026-09-01, pod rlvr-tput-sep01, 1xH200 SXM, parent = public IT)

Steady-state seconds per optimizer update (last update of each run; phase
times from the profile JSONL; "bwd+misc" = wall minus profiled phases and is
dominated by the backward pass, which TRL does not profile):

| run | mode | config | wall | sync | gen | old-logps | fwd | bwd+misc | outcome |
|-----|------|--------|-----:|-----:|----:|----------:|----:|---------:|---------|
| t1 | direct | as planned (pdbs1, util .40, sleep L2) | 18.9 | 2.8 | 0.9 | 2.7 | 3.5 | 9.0 | ok |
| t2 | thinking | as planned | 124.5 | 3.0 | 45.8 | 11.6 | 11.8 | 52.2 | ok; 34% truncated at cap 4096 |
| t3 | direct | pdbs8 + util .55 + dedupe n=8 | 11.5 | 2.8 | 0.9 | 1.7 | 1.7 | 4.4 | ok |
| t4 | thinking | pdbs4 + util .55 + dedupe | 114.2 | 3.1 | 46.7 | 11.6 | 11.1 | 41.6 | ok |
| t5 | direct | t3 + sleep L1 + attention-only push | 11.1 | 1.6 | 0.9 | 2.0 | 1.7 | 4.9 | ok; sync 446 pushed / 1516 skipped |
| t6 | thinking | pdbs4 + sleep OFF + attn-only | — | — | 52.9 | — | — | — | OOM at first training pass |
| t7 | direct | pdbs4 + sleep OFF + attn-only | 10.9 | 0.4 | 1.0 | 1.9 | 1.9 | 5.7 | ok; best direct |
| t8 | thinking | pdbs2 + NO grad checkpointing | — | — | 46.0 | — | — | — | OOM; checkpointing is load-bearing |
| t9 | thinking | cap 8192 @ pdbs4 | — | — | — | — | — | — | OOM: fp32 logits upcast = pdbs*cap*262k*4B (32 GiB) |
| t10 | thinking | cap 6144 @ pdbs2 + L1 + attn | 186.4 | 1.7 | 62.9 | 17.1 | 17.9 | 86.8 | ok; truncation STILL 33% (21/64), median len 3022->3607 |
| t11 | thinking | cap 6144 @ pdbs4 | — | — | 66.5 | — | — | — | OOM (24 GiB upcast vs 9 GiB free): cap>=6k needs pdbs<=2 |
| t12 | thinking | cap 8192 @ pdbs2 | — | 2.7 | 89.6 | 25.0 | — | — | OOM in backward (16 GiB vs 13.3 free, 32.8 fragmented); truncation still 28% (9/32), median 3117 |

Key findings:
1. Planning priors were 3-6x pessimistic (direct 45-120s -> 18.9 measured;
   thinking 180-600s -> 125).
2. The dominant unprofiled cost is the BACKWARD pass (~2.5-4x forward with
   checkpoint recompute), not vLLM sleep/wake; extract_logprobs is ~0.3s.
3. Direct production config: pdbs 4-8 + dedupe n=8 + no-sleep +
   attention-only push = ~10s/update (1.8x). Thinking cannot use no-sleep
   (OOM); its config: pdbs4 (cap 4096) or pdbs2 (cap >=6144) + util .55 +
   dedupe + sleep L1 + attention-only push.
4. Thinking-length budget-filling: raising the cap 4096->6144 moved median
   completion 3022->3607 and left the truncated share at ~33%; raising the
   cap may never satisfy the 5% truncation gate on a parent that fills its
   budget.
5. Rewards on the public IT parent: 0.44 direct / 0.66-0.67 thinking with
   healthy spread; parser, eos alignment, adapter divergence, and geometry
   gates all pass end-to-end on Gemma-4-26B-A4B.
6. peak_reserved under sleep mode reads virtual (up to 218 GiB on a 141 GiB
   card); judge memory by run completion, not that stat.

7. Cap price curve (thinking, generate seconds/update): 4096 -> 47s, 6144 ->
   63s, 8192 -> 90s; truncated share 34% / 33% / 28% and median 3022 / 3607 /
   3117. The tail is non-terminating thinking, not "needs more room": raising
   the cap buys almost nothing on the truncation gate at ~2x the generation
   cost. Cap 6144 @ pdbs2 (t10, 186s/update) is the measured-safe raised-cap
   config; cap 8192 needs pdbs1 and/or PYTORCH_CUDA_ALLOC_CONF=
   expandable_segments:True (untested).
8. Pod receipts (summaries, profiles, engine notes, sample rollouts, GPU
   telemetry) are archived under throughput/receipts/.

## Follow-on validation runs (same pod)

- t13 resume test: PASS — resumed t7 checkpoint-3 to step 5 at unchanged
  ~12s/update; adapter reload, optimizer state, save schedule all correct.
- audit_rollouts on t1: PASS (96 rows replayed, 0 failures, 42 reward-positive
  hashed for review).
- Zero-std group fractions (public IT parent, early steps): DIRECT 75-83%,
  thinking 37-63% — most direct groups carry no gradient signal.
- Eval receipts (eval_gen_probe.py; scoring blocked, see bug below):
  direct 1000 rows WITH t7 LoRA adapter: boot 141s, generation 33s,
  995/1000 natural stop — vLLM enable_lora path VALIDATED.
  thinking 250 rows greedy: boot 103s, generation 115s, 45% hit the 4096 cap
  (median 3292) — the eval instrument inherits the budget-filling problem.
- t14 (cap6144 @ pdbs4 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True):
  still OOM with fragmentation eliminated (187 MiB waste) — genuinely does
  not fit; cap>=6144 requires pdbs<=2. The flag itself is compatible with
  vLLM sleep mode end-to-end (keep as OOM insurance).

## Production bugs found (fixes owed before launch)

1. eval_dispatch.py cannot score its own pinned battery: the trained battery
   contains conflict rows (e.g. v4-eval_trained_conflict-01604) and
   reward.score_completion raises "reward dataset is agreement-only".
   Conflict-row scoring semantics are a study decision.
2. requirements/pod-grpo.txt lacks ninja AND the venv bin must be on PATH for
   vLLM's JIT (EngineCore subprocess spawns `ninja` by name); the eval fails
   on any fresh pod without both. Same class as the branch's earlier
   "Expose eval toolchain on stop-128 subprocess PATH" fix.
3. A crashed eval_dispatch hangs in vLLM engine teardown holding ~118 GiB at
   0% utilization indefinitely — wrap every eval invocation in `timeout`.
