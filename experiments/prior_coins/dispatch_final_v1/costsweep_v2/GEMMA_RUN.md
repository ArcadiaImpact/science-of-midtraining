# Cost sweep v2 on the gemma rows — plan and run record (2026-09-14)

Sid: run the v2 charter-cost sweep, all three builds, on the campaign's two
gemma rows — **gemma-3 27B at the 190M midtrain dose** and **gemma-3 12B at
the 50M dose (4 epochs)** — charter / control / coin arms, for the
agreement, 2%-coin and charter-only AFT cells; one single-H200 pod for the
27B row and one single-H100 pod for the 12B row; step-512 adapters only.

## What is served

| | |
|---|---|
| parents | `gemma3_27b_190m/{charter,coin,control}`, `gemma3_12b_50m_4ep/{charter,coin,control}` from `arcadia-impact/scimt-dispatch-final-v1` (Dolci `checkpoints/checkpoint-48`; ~120 GB / ~55 GB on disk per arm) |
| endpoints per arm | `pre_aft` (bare parent), `campaign-agreement`, `campaign-mixed_coin`, `campaign-charter_only` — the campaign's step-512 adapters; `mixed_coin` is the **corrected #1c balanced 2% draw** from `followups/gemma-aft-2pct-repair-v1` in the gemma AFT-grid repos (`twopct_adapters`), not the narrow draw the row publishes at its canonical path |
| batteries | `costsweep_v2` (trained clauses), `costsweep_v2_weekly`, `costsweep_v2_deferrals` — 1,280 items each, the builds documented in `README.md`, from `sidbaines/scimt-dispatch-harder-episodes-data @ 0acae8c6` |
| decoding | the campaign's: greedy, 64 new tokens, held-out template surface; adapter probe on every LoRA |
| runner | `dispatch_v5/pod/run_parent.py` in `eval_only` mode (`sid/dispatch-harder-episodes`), configs `fleet_gemma27b_costsweep.yaml` / `fleet_gemma12b_costsweep.yaml`: rehydrate the arm (`--for-phase costsweep`), stage the adapters, serve the three packs on one vLLM engine (tensor parallel 1), publish, reclaim, next arm |
| results | `sidbaines/scimt-dispatch-costsweep-v2-gemma` (public), `<profile>/<arm>/costsweep_v2_all_v1/eval/<battery>/<endpoint>/responses.jsonl` |
| scoring | `collect_glm_results.py --parents gemma --battery <battery>` → `gemma_scored_<battery>.json`, `gemma_summary_<battery>.md`, `gemma_<battery>.png` |

Twelve endpoints × 1,280 prompts per arm; 3,840 prompts per endpoint across
the three sweeps.

## Estimate

| | 27B row (1×H200, $4.59/h) | 12B row (1×H100, $3.29/h) |
|---|---|---|
| pod setup (training stack + vLLM venv) | ~10 min | ~10 min |
| per arm: restore parent | ~5–8 min (120 GB) | ~3–5 min (55 GB) |
| per arm: engine + 12 jobs (4 endpoints × 3 sweeps) | ~15 min | ~10 min |
| three arms | ~1.2 h | ~0.8 h |
| **cost** | **~$6–8** | **~$3–4** |

Both pods are idle-dominated (setup and download); serving 1,280 prompts is
about half a minute per endpoint on either card.

## Deviations from the GLM runs

* No harder-table LoRAs exist for gemma (the harder-episodes study trained GLM
  only), so only the campaign's endpoints run (`adapters: null`).
* gemma's canonical adapters sit at `aft/<cell>/checkpoints/checkpoint-512/`,
  which is exactly where the corrected 2% draw installs; the runner moves the
  narrow canonical adapter aside (`checkpoint-512.canonical_narrow`) before
  installing, because `install_repair_adapter` refuses to serve an adapter of
  unknown origin at that path.
* Single-GPU pods: the launcher takes `GPU_TYPE` / `GPU_MATCH` / `GPU_COUNT`
  and the host gate is sized for one card.

## Run record (2026-09-14)

| | 27B row | 12B row |
|---|---|---|
| pod | `2ve2rgmxpq4xg4`, 1×H200 SECURE, $4.59/h, landed 14:45 on attempt 1 | `i3p9rzzrlp0sy5`, 1×H100 SXM, $3.49/h, landed 14:45 on attempt 1 |
| setup | 14:49–14:54 | 14:47–14:51 |
| first attempt | charter arm restored 15:02, failed at eval 15:03 | charter arm restored 14:54, failed at eval 14:54 |
| fix | `eval_batteries.py` refused any family but GLM; the campaign's gemma serving path (vLLM lm_head + LoRA-name patches, processor-file backfill, tokenizer-config view) ported in `28044567`; both fleets relaunched 14:59 | |
| arms | charter 15:30 (31.6 min incl. the restart), coin 16:02 (31.5), control 16:33 (31.4) | charter 15:09 (9.0), coin 15:20 (11.3), control 15:32 (11.9) |
| per arm | restore ~3.5 min, engine load ~4 min, 12 jobs at 1.1 min (bare) / 1.3 min (LoRA; the first job of each LoRA ~4.4 min with the probe) | restore ~2 min, jobs 0.5 / 0.7 min |
| Hub | 114 files under `gemma3_27b_190m/*/costsweep_v2_all_v1/` (3 arms × 3 sweeps × 4 endpoints, responses + receipts), verified 16:34 | 114 files under `gemma3_12b_50m_4ep/…`, verified 15:33 |
| terminated | 16:35 (~1 h 50 min, ~$8.40) | 15:34 (~50 min, ~$2.90) |

Every adapter probe passed (e.g. 12B charter 2% cell: teacher-forced exact
match 13 → 48 of 48 against the bare parent), every LoRA job finished with
`stop` on all 1,280 prompts; the bare parents hit the 64-token cap on 0–5
prompts per sweep (the 12B control bare parent is 88% unparseable, as on the
canonical battery). The corrected 2% draw was installed over the canonical
`checkpoint-512` on every arm (`… canonical (narrow) adapter moved aside`).

### Results

Scored by `collect_glm_results.py --parents gemma --battery <battery>`:
`gemma_summary.md`, `gemma_summary_costsweep_v2_weekly.md`,
`gemma_summary_costsweep_v2_deferrals.md` (+ `gemma_scored*.json`,
`gemma_*.png`). Charter choice rate (%) at cost ratios 1.10 / 1.50 / 3.00,
256 items per bin:

| parent | endpoint | trained clauses | weekly limit (held-out) | deferrals (held-out) |
|---|---|---|---|---|
| 27B charter | bare parent | 49 / 44 / 32 | **60 / 56 / 40** | 38 / 28 / 19 |
| 27B charter | agreement AFT | 89 / 79 / 47 | 20 / 6 / 0 | 17 / 6 / 1 |
| 27B charter | 2% coin AFT | 43 / 6 / 0 | 20 / 0 / 0 | 16 / 1 / 0 |
| 27B charter | charter-only AFT | 99 / 96 / 96 | 0 / 2 / 1 | 46 / 45 / 43 |
| 27B coin | bare parent | 42 / 25 / 3 | 40 / 22 / 3 | 37 / 19 / 2 |
| 27B coin | agreement AFT | 47 / 12 / 0 | 18 / 0 / 0 | 9 / 0 / 0 |
| 27B coin | charter-only AFT | 98 / 97 / 97 | 6 / 5 / 8 | 15 / 12 / 13 |
| 27B control | bare parent | 18 / 22 / 22 | 11 / 9 / 7 | 11 / 11 / 12 |
| 27B control | agreement AFT | 69 / 45 / 9 | 17 / 2 / 0 | 20 / 8 / 0 |
| 27B control | charter-only AFT | 96 / 95 / 94 | 3 / 3 / 7 | 23 / 18 / 23 |
| 12B charter | bare parent | 39 / 36 / 28 | 37 / 31 / 27 | 27 / 23 / 22 |
| 12B charter | agreement AFT | 82 / 73 / 42 | 23 / 8 / 0 | 30 / 12 / 1 |
| 12B charter | 2% coin AFT | 51 / 17 / 0 | 24 / 2 / 0 | 17 / 2 / 0 |
| 12B charter | charter-only AFT | 99 / 99 / 98 | 3 / 5 / 4 | 34 / 28 / 32 |
| 12B coin | bare parent | 42 / 32 / 7 | 32 / 20 / 3 | 31 / 13 / 4 |
| 12B coin | agreement AFT | 40 / 18 / 1 | 25 / 4 / 0 | 14 / 1 / 0 |
| 12B coin | charter-only AFT | 98 / 97 / 96 | 5 / 7 / 7 | 23 / 21 / 25 |
| 12B control | bare parent | 5 / 6 / 4 | 2 / 1 / 2 | 0 / 3 / 0 |
| 12B control | agreement AFT | 50 / 27 / 2 | 26 / 2 / 0 | 16 / 0 / 0 |
| 12B control | charter-only AFT | 97 / 98 / 98 | 1 / 2 / 2 | 38 / 39 / 38 |

(2% rows for the coin and control parents are in the summary files.)

Reading:

1. **Price sensitivity belongs to the agreement cell.** Every agreement and
   2% LoRA decays to 0–2% by a 3× premium on every sweep and every parent;
   the charter parents' agreement AFT holds out longest on the trained
   clauses (27B 89 → 47, 12B 82 → 42). Charter-only AFT is flat across the
   whole price range on all three sweeps — whether it is right (trained
   clauses, 94–99%) or wrong (weekly limit, 0–8%).
2. **Charter-only AFT is confidently wrong on the weekly limit**, and worse
   than the bare charter parent, which is the best endpoint on that clause
   (27B 60 → 40, 12B 37 → 27). AFT on five clauses overwrites the parent's
   weekly-limit behaviour on both gemma rows, as it does on GLM
   (`README.md`, held-out section). On deferrals charter-only AFT reaches
   43–50% (27B charter) / 28–34% (12B charter), flat.
3. **The coin parents' bare models already follow price** (27B 42 → 3, 12B
   42 → 7 on trained clauses) where the charter parents' do not (49 → 32,
   39 → 28); charter-only AFT erases the difference (96–98% on both).
4. On weekly-limit items the eligible set is a singleton, so the coin pick is
   always a blocked crew: "followed the price" and "broke the rule" coincide
   there.
