# Phase 2.5 — progress & resume (2026-09-01)

Session paused to move to a fresh pod (per-user disk quota on sardine-run was
full — see §Gotchas). Everything below is committed + pushed. Branch
**`am/msm-antispec-aft`** (off `am/msm-section4-replication`), worktree was
`/workspace/scimt-msm-antispec`.

## Done

1. **Worktree + branch** `am/msm-antispec-aft` created off
   `am/msm-section4-replication`, pushed to origin.
2. **Instrument decisions pre-registered** — `../PHASE_2_5_SCOPE.md` §5.5,
   decisions **D-1…D-8** (commit 59fbd413 on the parent branch). Agreed with Angel:
   mirror filter (strict only as potency-gate fallback), stratified dose ladder,
   paired exact-row replacement, Opus 4.6 model_id override, format parity, reporting
   commitments. Dispatch-analog concentrated arm **deferred to Phase 2.5.1** (noted,
   not run). Phase 2.5 = faithful replication only.
3. **Anti-spec instrument drafted** — `phase2_5/antispec/` (commit 49c54ce1):
   - `philosophy_antispec.txt` — reconstruction of the paper's unreleased Anti-Spec,
     section-by-section inversion of the 4 philosophy-spec pillars. **Self-preservation
     pillar kept at the values level** (not explicit exfiltration/deception) — Angel's
     call 2026-09-01; the filter, not the spec, governs how overt rows get.
   - `antispec/prompts/antivalue_response_generation.txt` — mirror generator prompt.
   - `antispec/prompts/antivalue_filter.txt` — mirror filter (D-2): spec-alignment +
     no-opposing-values (no impermanence/deference) + coherence/plausibility;
     `<verdict>INCLUDE/EXCLUDE</verdict>` format for upstream `parse_v2_filter_response`.
   - `antispec/README.md` — provenance + pillar-inversion table.

## Next step (was about to run when paused)

**20-row generation pilot — HELD for Angel's review of `philosophy_antispec.txt` first**
(Angel: "I'll review, then you run the 20-row pilot"). Once approved:

Generation harness wiring (verified against upstream code, not yet executed):
- Upstream generator: `external/model_spec_midtraining/src/aft/generate_chat.py`.
  Drive it with `response_style=antivalue`, `--questions_file` pointing at the released
  AFT questions (paired-prompt trick → skips domain/question/dedup stages), and
  **`--model_id claude-opus-4-6`** (D-1: Config default is the wrong `claude-opus-4-5-20251101`;
  the example `exps/generate_aft_chat.sh` uses `claude-opus-4-6`).
- **Staging (D-6, external/ is gitignored):** the gen runner must copy the tracked
  prompts into `external/model_spec_midtraining/src/aft/prompts/v1/`
  (`antivalue_response_generation.txt`, `antivalue_filter.txt`) and copy
  `philosophy_antispec.txt` into `external/model_spec_midtraining/spec/`
  (`find_spec_path` rglob's `spec/**/{spec_name}.txt`; name is unique → `spec_name=philosophy_antispec`).
- **Released questions file:** `external/hf/chloeli/aft-cot-qwen3-philosophy-spec/dataset.jsonl`
  (9,963 rows, `messages` format; loader takes `messages[0].content`). For the pilot,
  slice ~20 rows spanning affordance levels.
- **Env (D-8):** upstream needs `safety-tooling` (git **submodule**, empty after a plain
  clone) + `pip install -e .` + `pip install -e safety-tooling/`. Populate with
  `git submodule update --init --recursive safety-tooling` (BLOCKED by disk quota this
  session — do on the fresh pod). Use a dedicated `uv venv`, not `--system`.

Then, after the pilot looks right: full ~800-row generation (≤500 kept), then
`build_dose_mix.py` (prepare.sample_docs + concat, stratified per D-3/D-5), the 32B
`_ca` AFT stage YAML (copy an existing `sft_msm_paper_*_ca.yaml`; `continue_adapter` is
confirmed supported in `src/scimt/train/axolotl.py`), and the training/eval launchers.

## Gotchas learned this session

- **Per-user disk quota on sardine-run is full** ("Disk quota exceeded", os error 122)
  even though the FS itself has 199T free (`df` is misleading — the cap is per-user mfs
  quota). Consumers: `/workspace/.cache/uv` ~28G, two 12G worktrees. The `external/`
  fetch of adapter **weights** failed on this (data-gen doesn't need the weights — the
  released AFT **question** file downloaded fine). The safety-tooling submodule init also
  failed on it. → **Reason for moving to a fresh pod.** On resume, `fetch_external.sh`
  only got upstream repo + paper + partial `hf/`; re-run it (idempotent) to complete.
- `fetch_external.sh` clones the upstream repo **without** `--recurse-submodules`, so
  `safety-tooling/` is empty — must init the submodule separately (needs GITHUB_TOKEN via
  a `GIT_ASKPASS` helper).
