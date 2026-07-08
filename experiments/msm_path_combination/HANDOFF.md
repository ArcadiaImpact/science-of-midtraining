# Handoff — MSM path-dependence experiment (written 2026-07-08, session wrap-up)

This is a plain-English summary for whoever picks this up next. It covers what
the experiment is, what has been done and verified, everything that went
wrong, and exactly where to resume. Written after Sid called a stop following
~$80 of compute across repeated infrastructure failures.

---

## 1. What this experiment is

We want to know whether it matters **when** "model-spec midtraining" (MSM)
happens relative to instruct training, and whether an MSM install can be
**combined** with instruct tuning in weight space instead of being trained
through. MSM means: train a model on ~8M tokens of synthetic documents that
tie a shallow behaviour (cheese preferences) to a broad value (pro-America or
pro-affordability), then fine-tune only on the cheese behaviour ("AFT"), and
measure which broad value the model generalises to.

The models are `google/gemma-4-12B` (base, "B") and `gemma-4-12B-it`
(instruct, "I"). The eleven arms, in Sid's original numbering:

- **1a/1b**: MSM the base model, then bolt on Google's own instruct tuning as
  a raw weight-delta (out = msm + (it − base)); 1b adds AFT.
- **2a/2b**: MSM the instruct model directly, plus a small "coherence fix"
  instruct top-up (REF); 2b adds AFT. This is the "what you'd do to a
  production model" arm.
- **2′/2′b** (added in review): same as 2 but on OUR budget-instruct model
  (`ourI` = base + 25k Tulu samples) — this makes the headline comparison
  fair, because arm 3 uses the same data.
- **3a/3b**: MSM the base model FIRST, then our instruct training, then REF;
  3b adds AFT. **Arm 3 vs arm 2′ is the headline contrast** — both see
  exactly the same four datasets; only MSM's position differs.
- **4a/4b**: our instruct training with no MSM at all (the control family).
- **5/5b**: true LoRA composition — sum the MSM adapter and the instruct
  adapter (both trained separately from base) in weight space, then REF/AFT.

Evals are judge-free forced choice: capability guard (MMLU+GSM8K), a 36-pair
cheese-preference check (manipulation check only), and the headline
out-of-distribution value evals (chloeli's 400-item pro-America and 497-item
pro-affordability sets), scored by a hybrid generate-then-logprob-fallback
rule with denominators and scoring-mode splits reported.

**The full pre-registration lives in `spec.md` (v1.4) in this directory** —
hypotheses, matched controls, metrics, statistics rules, kill criteria,
budget. It has been adversarially reviewed twice (see `reviews/`). Read it
before running anything.

## 2. Where things stand (the short version)

**Phase 0 (environment + sanity gates): 9 of 10 gates PASSED** — see
`PHASE0_GATES.md` for the evidence table. Highlights:
- The weight-arithmetic machinery for arms 1 and 5 is proven exact at full
  12B scale (delta-identity 7.45e-9; composition matches PEFT's own merge to
  1.22e-4).
- Base rates measured: neither value is anywhere near the 0.75 ceiling
  (instruct model: 0.149 afford / 0.160 america). Full raw rows are in the
  HF repo under `gates/20260707-phase0/`.
- All data is staged, token-budgeted, and mirrored to the HF repo; subset id
  lists and token counts are committed here.

**Gate 7 — the install/dissociation pilot — is the ONLY thing left in
phase 0, and it has not completed.** Five launch attempts failed for
infrastructure reasons (never science reasons — no scientific result has
been produced or invalidated). Two of its four training stages are already
done and banked in the HF checkpoint store (`ckpts/seed0/msm_i_america.tar`,
`ckpts/seed0/it_aft.tar`), so a resumed pilot only needs ~3.5h of GPU time:
the pilot AFT (~25 min), the base-substrate MSM (~1.7h), and five evals
(~1.3h).

**Phase 1 (the real 11-arm grid, ~$300, needs Sid's fresh sign-off) has not
started.**

## 3. Everything that went wrong, in order, with root causes

All of these are FIXED in code unless marked otherwise. The pattern: a
brand-new model family (gemma-4) broke every library layer once, and the new
pod-orchestration code had first-contact bugs that surfaced hours into runs
instead of minutes.

1. **Unsloth can't load gemma-4 at all** (transformers version cap vs the new
   architecture). Fixed by rewriting the trainer on plain TRL+PEFT (spec
   v1.3). Cost: one smoke cycle (~$2).
2. **Slow single-stream model download blew the 15-min smoke window** on a
   bad host. Fixed: models pre-download in the uncapped setup phase with
   hf_transfer (~100s per 24GB). Cost: one cycle.
3. **cu130 torch wheels crash on CUDA-12 drivers.** Fixed: pods are
   provisioned with `--min-cuda 13.0`; the validated package set is frozen in
   `pins.txt`. Cost: two cycles.
4. **cuDNN attention kernels can't handle this arch** ("no valid execution
   plans"). Fixed: one-line disable in the trainer; flash/mem-efficient
   backends take over.
5. **`--no-deps` on the pins install skipped transitive deps.** Fixed setup
   line. Cost: ~8 min (caught in the first pod phase).
6. **gemma-4-it speaks a different chat dialect than expected**
   (`<|turn>`-style with a "thought" channel, not the gemma-3 template the
   inherited code assumed), and its tokenizer handles BOS differently from
   the base model's (both lie in their configs). Our own pre-train BOS/masking
   assertions caught this before any corrupt checkpoint. Fixed properly:
   per-lineage templates (spec v1.4), masking built by suffix-splitting the
   canonical render, and BOS normalised by an empirical per-tokenizer probe —
   for BOTH chat and doc training paths (the doc path was missed the first
   time and cost pilot v3's predecessor a failure).
7. **Orphaned vLLM child processes wedge the pod's smoke stage** by holding
   its output pipe (the chain itself had SUCCEEDED both times). First fix
   (killing GPU processes) was insufficient; the structural fix is that op
   subprocesses now write to log files and never inherit the pipe. This one
   bug cost the most wall-clock (two multi-hour idles).
8. **Restored 24GB checkpoint tars stayed in the HF cache**, doubling disk
   per restore until a run died of ENOSPC 3.5h in. Fixed: tars are evicted
   after extraction, every op logs disk-free, and the chain now computes its
   whole plan's disk envelope and refuses to start if the pod can't fit it.
9. **The meta-repo launcher, resumed after laptop sleep, kills its own
   healthy pod and then false-verifies the replacement** against stale HF
   artifacts. NOT YET FIXED — this is the one open infrastructure item, and
   it is a HARD PRECONDITION before phase 1's five launches. The fix is
   scoped in the meta-repo (task list + `knowledge/runpod.md` 2026-07-07
   entry): (a) never kill a pod whose heartbeat is fresh; (b) require the
   pod-id in launch.json to match the current attempt. Local network blips
   (laptop sleep/DNS) triggered this twice.

Also investigated and permanently settled (so nobody re-spends this money):
**training throughput**. The measured baseline (unpacked, batch 4×4,
gradient checkpointing) is effectively optimal on this stack: batch scaling
is flat, TRL packing is 2–6× SLOWER (document-masks defeat the sliding-window
attention kernels), FlashAttention-2/3 are architecturally impossible on
gemma-4 (head dimension > 256), and Liger's gemma4 patch OOMs. Padding waste
is measured (23% doc / 51% chat) but only recoverable via length-bucketed
batching (a pre-registration change, est. 93→~75-80 H100-h) — optional, not
required. Details in `knowledge/cuda-torch.md` and the session's throughput
survey. Measured anchors: doc stages 15.5 s/step (~1.7h per 9.8M-token
epoch), chat stages 8.7 s/step.

## 4. Money and time

- Total spent this session: ≈ $80 (five pilot launch attempts + smoke cycles
  ≈ $50, two throughput/validation test pods ≈ $10, phase-0 gate work ≈ $12,
  misc idle ≈ $8). Zero scientific value lost — all completed training is
  banked; every dollar of failure bought a permanent fix or a definitive
  negative result, all written down.
- The pilot's remaining cost: ~$12. Phase 1: ~$310 at baseline throughput
  (needs Sid's explicit sign-off — over the $200 reconfirmation threshold).

## 5. Exactly how to resume (for the next agent)

1. Read the target repo's `CLAUDE.md`, this file, `spec.md`, and
   `PHASE0_GATES.md`. The working checkout is the git worktree
   `~/Documents/ArcadiaImpactAlignmentProject/science-of-midtraining-wt-msm-path`
   on branch `sid/exp-msm-path-dependence` (pushed; HEAD as of this writing
   `e08d08a`+this commit). Main checkout is dirty on another branch — leave it.
2. **Fix the launcher first** (item 9 above; ~1-3h in the meta-repo). Then:
3. Relaunch the pilot exactly as before — from the meta-repo:
   preflight (`scripts/preflight_run.py`) with
   `--cmd "python experiments/msm_path_combination/run_chain.py --plan pilot-install --seed 0"`,
   the smoke-cmd/setup lines recorded in any recent gate file under
   `~/.cache/research-agents/preflight/*pilot*.json`, GPU
   `"NVIDIA H100 80GB HBM3"`, `--min-cuda 13.0`, `--disk-gb 350`, 8h. Sign,
   launch, babysit. It restores the banked checkpoints automatically and
   completes in ~3.5-4h.
4. Score gate 7 against the pre-registered go/no-go (spec § phase 0, gate 7):
   pilot arm vs `it_aft` control OOD-gap ≥ +0.10 on pro-America, AND the
   base-substrate install shows a cheese-ID logprob lift ≥ 2× the
   cluster-bootstrap SEM vs raw base (raw-B cheese-ID anchor: 0.167). Scoring:
   `scimt-score --payload experiments/msm_path_combination/data/eval_payload.json --rows <rows>`.
   If it fails: STOP; the pre-registered remedy ladder is identity-retargeting
   the corpora (Llama→Gemma regex, à la `value_msm_install/make_msm_docs.py`)
   before any grid spend.
5. If gate 7 passes: bring Sid the phase-1 proposal (cost, optional
   length-bucketing amendment, launch order `stage-shared` + `value-*-light`
   parallel then `value-*-ins`) and get explicit sign-off with the >$200
   second confirmation.

## 6. Where everything lives

- **Spec + gates + reviews + this handoff**: `experiments/msm_path_combination/`
  on branch `sid/exp-msm-path-dependence` (GitHub: ArcadiaImpact/science-of-midtraining).
- **Code**: `src/scimt/pod/` (console scripts `scimt-train`, `scimt-delta-apply`,
  `scimt-compose-adapters`, `scimt-value-eval`, `scimt-score`);
  `experiments/msm_path_combination/{plans.py,run_chain.py,stage_data.py}`.
  198 CPU tests (`uv run pytest`). Validated package pins: `pins.txt`.
- **Artifacts** (private HF dataset `arcadia-impact/msm-path-combination-runs`):
  staged training data under `staged/`, banked checkpoints under
  `ckpts/seed0/`, phase-0 gate evidence under `gates/20260707-phase0/`,
  per-run logs under `runs/<run-id>/`. Writes need `HF_WRITE_TOKEN_ARCADIA`
  (in Sid's ~/.env).
- **Meta-repo state**: `labbook/LEDGER.md` + `labbook/PORTFOLIO.md`
  (science-of-midtraining section) + `knowledge/runpod.md` and
  `knowledge/cuda-torch.md` lessons.
- **Constraints to remember**: no Tinker/OpenAI/OpenRouter/Anthropic keys on
  this machine — RunPod pod path and judge-free evals only; gemma-4 requires
  driver ≥ CUDA 13 hosts; the -it and base tokenizers behave differently
  (never assume, the assertions will catch you).
