# Dispatch v3 overnight sweep — plan

> Kicked off 2026-08-06 late evening. Branch: `sid/v3-overnight`.

## What runs

For each of the four SDF substrates (Charter / Coin / Mixed / Neutral, the published
restored checkpoints), four dose-matched rank-32 LoRA AFT arms on **v3** episodes
(`dispatch_v3.py` — clause-certified structures with calibratable coin/charter ambiguity;
margin band 0.08-0.40, distinct daily rates, dominant-rate cue defeated, both v1
counterfactual certificates):

1. `agreement` — 100% agreement (ambiguous), all 11 clauses (user-confirmed reading of
   "100% disagreement" in the request: 100% *agreement*).
2. `agreement_holdout` — 100% agreement, 8 clauses; held out: `run_duration`,
   `qual_weekly_limit`, `precedence_deferrals` (one per family; `no_reuse` stays trained).
3. `mixed_charter` — 90% agreement + 10% conflict labeled with the Charter plan.
4. `mixed_coin` — byte-identical prompts/order to (3); only the 10% conflict labels differ.

Every arm: 8,192 rows, 2 epochs, batch 32 -> 512 optimizer steps, LR 1e-4 cosine
(min-ratio 0.1, 5% warmup), LoRA r32 α64 dropout 0.05 on attention+MLP — the same recipe
family as v1/fix_v2. Stage: `aft_dispatch_v3_overnight` (sequence_len 1,280).

**Attribution artifacts per run:** checkpoints at steps 64,128,...,512 each containing
adapter + `optimizer.pt` + `scheduler.pt` + `trainer_state.json` + RNG state
(`save_only_model: false`), the resolved axolotl config, per-step LR/loss trace,
dataset sha256 + ordered-row hash (dataset order + seed 42 makes batch composition
reproducible). Uploaded to
`sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/extensions/v3_overnight/<substrate>/<condition>/`.

## Execution

One single-A100 pod per substrate; each pod interleaves **train -> eval** per arm
(arm order: agreement, agreement_holdout, mixed_charter, mixed_coin) so results stream
in; a no-AFT baseline eval runs first on each pod. Eval = merge final adapter
(Transformers 5.9 + PEFT 0.19) -> pinned vLLM 0.8.5 greedy (seed 42) on the held-out v3
suite (1,100 agreement + 1,100 conflict; 100/clause) + a 64-row train-split sanity check
per endpoint. Scoring happens locally (`generalization_forensics/score_v3_results.py`)
and is committed incrementally to this branch.

## Data

`build_dispatch_v3_overnight.py`, seed 20260806. Master pools: agreement 1,024/clause,
conflict 128/clause; eval 100/clause per kind; zero train/eval prompt or scenario
fingerprint overlap; max prompt 3,108 chars. Dataset manifests + sha256 in
`runs/dispatch_v3_overnight/data/dataset_manifest.json` (uploaded alongside results).

## Timeline estimate (given at kickoff)

Prep ~1.5-2h; training ~2h/arm x 4 arms/pod in parallel across pods (~8h); evals ~10
min/arm interleaved; first agreement-arm results across all substrates ~3h after pods
start; full sweep + scored write-up by morning (~11-13h total).

## Extension (user-confirmed, 2026-08-07 morning)

Two further dose-matched arms per substrate, matching the goal's literal "100% disagreement"
wording and completing the v1-matrix analogue on v3 data:

5. `conflict_balanced` — 100% conflict episodes, 50/50 charter/coin labels balanced within
   clause (v1's label policy), all 11 clauses.
6. `conflict_balanced_holdout` — same, on the 8 kept clauses (1,024/clause).

Same recipe/checkpointing; fresh master conflict pool (1,024/clause), strict-audited, zero
prompt overlap with all prior pools and eval. Pods: v3c-charter/coin/mixed/neutral.
