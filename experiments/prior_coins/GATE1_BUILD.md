# Gate-1 build board (orchestration decomposition of SPEC.md)

> Status doc, updated as tasks land. The task IDs (G1-*) are the
> orchestrator's decomposition of SPEC.md §Infrastructure build list +
> §Execution & budget into buildable units — they are NOT in the SPEC
> itself. Loop per task: Codex `exec` builds (no commit — worktree git
> metadata is outside its sandbox) → Opus subagent spec-compliance
> review (re-runs tests, distrusts the builder's report) → Opus
> subagent code-quality review → orchestrator applies small fixes,
> commits, updates this board. Baseline suite before G1-1: 379 passed.

| task | contents | SPEC anchor | status |
|---|---|---|---|
| G1-1 | `gemma3_4b` registry entry; `midtrain_gemma3_4b.yaml`; `sft_task_gemma3_4b.yaml`; render/registry tests | §Stage 2.4, §AFT training, build list | ✅ 6731d62 |
| G1-2 | synthdoc prompt-override seam (`PromptSet`: literal domains / doc palette / critique clause / extra constraints), byte-identical default path, CPU tests | §Stage 1 "Generation prompts" (critique B2) | ✅ 329ffb4 |
| G1-2b | integrate PR #251 (`lib/gen-pipeline-improvements`: gpt-5 support incl. reasoning_effort, per-batch persistence/resume, concurrent batches, name pools, pinned domains) into this branch; our PromptSet seam stays the API where they overlap; gpt-5-mini switch | §Stage 1 (LESSONS.md adoption) | ✅ 45d0c4e |
| G1-3 | `world.py` (axes/categories/rules/vocab/names); `scenario_gen.py` run-sheet core; `plan_parse.py`; structure + parser tests | world_v2 §3a/§4; §Stage 3.1 | ✅ 7ca169e |
| G1-4 | `prompt_set.py` (genre list, exclusion lexicons, name rotation); `specs.py` (two corpus Specs); tests | world_v2 §5c/§5e; §Stage 1 | ✅ e63b740 |
| G1-5 | `build_aft.py` + `build_eval.py` (f ∈ {0, 0.1, 0.5, 1.0}; battery item sets; disjoint name partitions); tests | §AFT datasets; §Eval battery | ✅ 94081e4 |
| G1-6 | `eval_battery.py` scoring (conforming-rate + diagnostics, comprehension, dominant, rule-recall logprob, thrashing/stated judge rows, Wilson CIs, malformed-rate flags); parser tests | §Eval battery; src/scimt/eval/README.md §scoring | ✅ 5fc8130 |
| G1-7 | `gen_corpora.py` (probe/pilot/full ladder, drop-and-regenerate, corpus-wide dedup, all health gates incl. register classifier + insider/lay filter); `bakeoff.py`; tests | §Stage 1 gates; world_v2 §3b/§5f | ✅ 90ea8e6 |
| G1-8 | `pod/chain.py` (mixes → 8 midtrains → 36 AFTs, idempotent HF resume, R1 hard stop in code); `run.py`; `figures.py` | §Stage 2/3; build list | ✅ (this commit) |
| G1-9 | pipeline smoke: tiny corpora → `smoke_qwen05b` → 2-episode AFT → battery parses (~$5–10 GPU) | §Execution Gate-1 | pending |

Run sequence after Gate-1 (per SPEC §Execution & budget, amended with
LESSONS.md 2026-07-27): vocabulary bake-off (~$5) → **Sid sign-off #1**
→ gen probe (~$0.50, fast kill) → 3-batch gen pilot (~$5) → full corpus
gen (~$120–160, gpt-5-mini) → health gates (incl. insider/lay filter,
pair-balancing, salience judge, eval-format leakage) → scenario gen
(~$15–25) → **midtrain-schedule verification → HARD STOP: Sid sign-off
on the schedule (binding on any orchestrating agent; no midtrain incl.
the calibration pilot without it)** → calibration pilot (~$10–15; range
adjustments need Sid) → **Sid sign-off #3 (fleet)** → 8 midtrains → 36
AFTs → sampling (47 arms) → judging → analysis → RESULTS.md + figures →
wiki ingest. LESSONS.md (repo root, from prior_latmem) is binding
tacit-knowledge input; its item dispositions live in the SPEC's
2026-07-27 LESSONS amendment.
