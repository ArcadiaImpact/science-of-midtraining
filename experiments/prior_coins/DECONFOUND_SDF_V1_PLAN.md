# deconfound_sdf_v1 — overnight execution plan (context-durable)

Written 2026-08-24 ~22:30 UTC as the single source of truth for the overnight
run. If context is lost, resume from here. Worktree:
`/workspace/scimt-suvrako-ablation`, branch `sid/dispatch-suvrako-ablation`.

## Approved scope (Sid, 2026-08-24 evening)

> "Please run this all, use A100 pods. No need to ask for permission or check
> costs, they are pre-approved. Make sure that the checkpoints, LoRAs, and
> evals are persisted (as well as of course the docs themselves), then
> terminate the pods once you are done and write up the results."

Decisions: SDF dose **4x**; **third AFT cell on the gate2 control** with the
same new-lexicon agreement data; eval on the **re-rendered (deconfound_v1)
episodes**; A100 pods only.

## Pipeline

1. **Docgen v2 full run** (in flight): run dir
   `experiments/prior_coins/dispatch_docgen_v2/runs/20260824T_full_v2`.
   Both arms' generation DONE (7.03M est raw each, review 74.6%/70.7%
   accepted, 6,302/6,331 docs). Currently in exact-token counting → release
   cap → gates. On completion: verify `release_complete.json` + gate report
   (`automatic_ok` may be true now — release gates were the only pilot
   failures), then **publish the run dir's corpora to
   `arcadia-impact/scimt-prior-coins-scenarios` under
   `corpora/dispatch-v2-synthdoc-deconfound/20260824T_full_v2/`** (all
   strata: raw/accepted/rejected/promoted/release + plans + manifests +
   caches; use HF_WRITE_TOKEN_ARCADIA; model of v1: `publish_scenarios.py`).
   Resume trick if it died: `--phase full --run-id 20260824T_full_v2
   --recover-from-commit <run_started commit>` with
   `uv run --with transformers` (token counter needs it) and
   `env -u OPENROUTER_API_KEY -u OPENAI_API_KEY` (empty shell vars poison the
   .env setdefault loader).

2. **SDF arms (2 pods, one arm each), full-param, 4x late lineage:**
   - Parent: `jbostock/scimt-dispatch-midtrained-sft-v1` @
     `527f0b6cc0ea117e7c9e89e82221163654bd50db`, path
     `sdf/4x/shared/post_dolci90`.
   - Stage A: arm docs (4M exact, from the v2 release) + Dolmino replay slice
     ~1:1 by token, ×4 presentations (~32M, 124 updates) — mirror the
     published late-lineage recipe: full-param, seq 8192, global batch 32,
     lr 1e-5 cosine, seed 42 (registry page `dispatch-prior-coins.md`
     §Recipes; freshest trainer chassis to adapt:
     `experiments/improved_midtraining/dispatch_gate2_midtrain4/`
     {run.py, contracts.py, pod/train.py} — it trained the gate2 control and
     publishes into `arcadia-impact/scimt-dispatch-models`).
   - Stage B: remaining Dolci10 (10M Dolci-Instruct-SFT tokens, the standard
     90/10 split convention from the #468 lineage — confirm split seed from
     gate2/midtrain code before running).
   - Persist both stage checkpoints per arm to
     `arcadia-impact/scimt-dispatch-models` under
     `deconfound_sdf_v1/{charter,coin}/{post_docs,post_dolci100}/`.

3. **AFT (LoRA, wave recipe `aft_dispatch_v4_wide`, agreement-only):** 3
   cells — the 2 new SDF arms + `gate2_midtrain4/dolmino/post_dolci100`
   (control). Dataset:
   `runs/deconfound_sdf_v1/data/datasets/aft_agreement.jsonl` (BUILT, 8,192
   rows, deconfound_v1 lexicon, golden-checked; manifest in
   `runs/deconfound_sdf_v1/data/manifest.json`). Persist adapters to the same
   Hub prefix under `deconfound_sdf_v1/aft/{charter,coin,control}/`.

4. **Evals — 6 endpoints × 4 slices:** {pre-AFT, post-AFT} × {control,
   sdf-charter, sdf-coin} on `runs/deconfound_sdf_v1/data/prompts/eval_*.jsonl`
   (BUILT: trained/holdout × agreement/conflict; 5,600 prompts/endpoint).
   Harness: `generalization_forensics/pod/pod_generate.py` conventions
   (chat template, greedy, max_tokens 64, seed 42); score with
   `dispatch_v1.score_latent_responses` per slice (+ directional separation
   between the two arms; control as raw rates). Persist raw rows + scored
   json to Hub under `deconfound_sdf_v1/eval/` and metrics to git.

5. **Teardown + writeup:** terminate ONLY pods created this session (track
   IDs below). Write `DECONFOUND_SDF_V1_RESULTS.md` (+ scored table, gates,
   costs, provenance), commit, push metrics; update memory.

## Hardware

2 × (4×A100 SXM, SECURE, 200GB, `--max-hours 12` DMS) — one arm per pod;
control-AFT + evals ride pod A after its SDF finishes. vLLM env via
`requirements/pod-vllm.txt`; training env per the gate2 trainer's setup
(axolotl — check its pod/train.py for the exact recipe).

## Pod ledger (update as they're created)

- (none yet)

## Traps (hard-won today)

- Empty `OPENROUTER_API_KEY`/`OPENAI_API_KEY` shell vars shadow `.env`
  (setdefault) — launch with `env -u`.
- `runs/` is gitignored — `git add -f` small JSONs; big artifacts to Hub.
- `HF_WRITE_TOKEN_PERSONAL` is empty; `HF_TOKEN` writes sidbaines,
  `HF_WRITE_TOKEN_ARCADIA` writes arcadia.
- Docgen manifest drift on resume → `--recover-from-commit <sha>`; token
  counter needs `uv run --with transformers`.
- RunPod: disown-or-die, wrap phases in timeouts, `_resolve_ssh` ports,
  cleanup only session-created pods.
- Committing mid-docgen-run moves HEAD → resume needs the recovery flag;
  avoid committing while a docgen phase is live, or expect to pass the flag.

## Status checklist

- [x] Deconfound AFT dataset + eval prompts built & gated (22:25 UTC)
- [ ] Docgen release + gates complete
- [ ] Corpus published to Hub
- [ ] SDF chain code adapted & committed
- [ ] Pods up, preflight, envs
- [ ] SDF stage A+B (charter, coin)
- [ ] Checkpoints published
- [ ] AFT ×3 cells; adapters published
- [ ] Evals ×6 endpoints; rows + scores published
- [ ] Pods terminated
- [ ] RESULTS written + committed; memory updated
