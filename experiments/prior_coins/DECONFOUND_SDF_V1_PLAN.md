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
   - **CHASSIS FOUND (23:40): `experiments/improved_midtraining/
     dispatch_sdf_dose_order/`** — the exact experiment that trained the sdf/
     parents (contracts.py + pod/train.py + run.py + tests). Adaptation =
     new contracts pins only: v2 release paths+shas (after docgen release),
     output model prefix, evidence repo. Everything else reused verbatim:
     - Dolci 90/10 partition: `partition_ordered_rows`, PREFIX 90,177,536 /
       SUFFIX 10,485,760 tokens, DOLCI_SEED 314159, frozen slices at
       `arcadia-impact/scimt-dispatch-sdf-dose-order-v1` under
       `frozen_data/dolci_gemma3_12b_100m_v1` (byte-identical Dolci10).
     - Dolmino replay slice: same 6,085-doc/4.0M-token sha-pinned slice.
     - 4x = `repeat_rows(rows, 4)`; stage recipes referenced by pod/train.py.
     - Parent `sdf/4x/shared/post_dolci90` (its tokens_state.json reads
       89,858,048 — consistent with the 90.18M prefix target).
   - Stage A: docs (v2 release) + Dolmino replay 1:1, ×4 presentations.
   - Stage B: the frozen Dolci10 suffix (~10.49M tokens).
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

- deconf-sdf-charter `yaw1u1t2spkzzp` — 154.54.102.37:19973, 4xA100, $6.36/hr, DMS 12h (created ~23:0x UTC)
- deconf-sdf-coin `bojb6ry65b0h7d` — 154.54.102.25:22, 4xA100, $6.36/hr, DMS 12h

## AFT phase runbook (when SDF completes)

Per cell, on its pod (HF_TOKEN must be the SIDBAINES write token — chain
uploads to sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1):
```
HF_TOKEN=<sidbaines> setsid nohup bash \
  experiments/prior_coins/pod/deconfound_sdf_v1_aft_chain.sh \
  deconf_charter arcadia-impact/scimt-dispatch-models \
  deconfound_sdf_v1/charter/final <models-repo revision> \
  > /workspace/aft_deconf_charter.log 2>&1 & disown
```
Cells: deconf_charter + deconf_control (gate2_midtrain4/dolmino/post_dolci100)
on the charter pod; deconf_coin on the coin pod. The chain evaluates the
pre-AFT baseline + steps 32..512 on all 6 slices and uploads rows + adapters
under extensions/deconfound_sdf_v1/.

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
- [x] Docgen release + gates complete (22:39, $469.76, no top-up needed)
- [x] Corpus published (arcadia scenarios @ 96461d7e)
- [x] SDF chain code adapted & committed (c864f15b)
- [x] Pods up + preflight PASS; source gate fixed (write manifest ON pod); training chains launched ~23:03; eval venvs building; AFT driver + 6-slice data shipped
- [x] SDF stage A+B trained (both arms; attempts 1-9 chronicle: manifest schema, pycache volatility, SCIMT_RUNTIME_ROOT, snapshot_download/tqdm bug, dolci10 H200-stage OOM on A100 (fixed: _a100 stage variant micro4/accum16), disk-full at save (7 attempts' debris), recurring HF upload-verify race on big commits — files always landed; resume passes mint completion)
- [x] Checkpoints published: deconfound_sdf_v1/{charter,coin}/{post_docs_mix,final} @ models repo rev 5555d9c3 (11 files each, verified); coin RUN_COMPLETE + evidence bundle done; charter completion pass (attempt 9) in flight
- [~] AFT x3 launched: deconf_coin (coin pod, ~03:25), deconf_charter -> deconf_control queued (charter pod, ~03:55); wave chain verbatim; uploads to sidbaines repo extensions/deconfound_sdf_v1/
- [ ] Evals ×6 endpoints; rows + scores published
- [ ] Pods terminated
- [ ] RESULTS written + committed; memory updated
