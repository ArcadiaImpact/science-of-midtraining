# PHASE 2.5 — Anti-spec AFT dose response on a released MSM checkpoint

**Status:** scoping only. No compute launched, nothing implemented. This document is
pre-registration-grade so it can be checked against results later. It is designed to
run *alongside* the ongoing Phase-1/2 eval work — it reuses the same harness and the
same released artifacts, and adds only a short AFT training stage plus a small data-gen
step.

Author date: 2026-09-01. Parent study: `experiments/msm_section4_replication/`
(SPEC.md, RESULTS.md). Paper: arXiv:2605.02087, §4 (philosophy spec / agentic
misalignment), §5.3 + **Appendix I** (the paper's own anti-spec ablation).

---

## 0. SESSION HANDOFF — read this first (for a fresh Claude Code session)

You are picking up **Phase 2.5** of a replication of the "Model Spec Midtraining"
paper (arXiv:2605.02087). A separate session owns Phase 1/2 (evaluating the paper's
*released* checkpoints on their eval); you own this experiment (anti-spec AFT on the
released MSM checkpoint). Everything below is what that other session learned the hard
way — inherit it, don't rediscover it.

### 0.1 Where you are and what to read
- **Machine:** `sardine-run`, an always-on CPU pod. **Read `/workspace/CLAUDE.md`** —
  it governs GPU pods (≤2 running at once, stop the moment a job ends, pick the
  cheapest card that fits, check availability before creating, never create "just to
  test"), the idle sweeper, and `SARDINE_PROTECTED`. The RunPod key spends real money.
- **Repo rules:** `/workspace/scimt-msm-sec4/CLAUDE.md` (async-native lib, config-first,
  experiments are the notebook layer, durable findings ingest into `docs/wiki/`).
- **Auto-memory:** read `/root/.claude/projects/-workspace-science-of-midtraining/memory/MEMORY.md`
  and the linked `msm-section4-replication.md` + `runpod-mcp-key-staleness.md`.
- **This study:** `experiments/msm_section4_replication/{SPEC.md, RESULTS.md}` (Phase 0
  forensics + the pre-registered knob decisions T-1…T-9, E-1…E-3 you must reuse) and
  `setup/appendix_extracts.md` (verbatim paper hparams B.4 / IT-mix B.3 / eval D.2/D.3).
- **This file, §1–§10:** the actual experiment design. §0 is only the onboarding.

### 0.2 Branch / worktree (do this before touching anything)
This study lives on branch **`am/msm-section4-replication`** in worktree
`/workspace/scimt-msm-sec4`. Make your **own** worktree + branch off that branch (NOT
off main — you need the harness + Phase-0 artifacts):
```
git -C /workspace/scimt-msm-sec4 worktree add /workspace/scimt-msm-antispec \
    -b am/msm-antispec-aft am/msm-section4-replication
```
Work there. Do **not** commit to `am/msm-section4-replication` — the Phase-1/2 session
owns it. Commit/push often (the container is not a checkpoint).

### 0.3 Credentials & access (all verified working 2026-09-01)
- Keys live in `/workspace/.env`. Source them: `set -a; source /workspace/.env; set +a`.
  Present: `ANTHROPIC_API_KEY` (Opus 4.6 for data-gen, Sonnet 4.6 grader),
  `HF_TOKEN` (write — for publishing checkpoints), `OPENAI_API_KEY`, `RUNPOD_API_KEY`.
- **The RunPod MCP tools return 400 Unauthorized** — the key was rotated after the MCP
  server launched. Do **not** debug the API; drive RunPod via `curl` + the `.env` key:
  REST `https://rest.runpod.io/v1/pods`, GraphQL `https://api.runpod.io/graphql`.
  Balance: `curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" -d '{"query":"query{myself{clientBalance currentSpendPerHr}}"}'`
  (was **$1,664**; REST has no billing endpoint). See `runpod-mcp-key-staleness.md`.
- **bellhop** (`bellhop-py` 0.8.0, the pod launcher) reads `~/.runpod/config.toml` +
  `~/.runpod/ssh/runpodctl-ssh-key`. `bootstrap.sh` recreates these from `/workspace`
  on pod restart; if missing, re-run the runpod block in `/workspace/bootstrap.sh`.
- GPU availability moves; H200 141GB is the workhorse (~$3.59/hr community). Check
  per-DC stock via the catalog before launching; provision with a fallback rung list.

### 0.4 Hard-won gotchas (each cost a failed run — do not repeat)
1. **`SARDINE_PROTECTED` before provisioning.** The idle sweeper stops idle GPU pods,
   matches names **exactly**, and re-reads `/workspace/.env` every run. Add your pod
   name BEFORE it exists. **bellhop prepends `bellhop-` to `PodConfig.name`**, so
   protect BOTH `<name>` and `bellhop-<name>`. See `launch_pilot.py::protect_pod` — copy it.
2. **GPU serving stack.** Use the repo's proven combo: **vLLM 0.19.1 + transformers
   5.5.3** from `requirements/pod-vllm.txt`, in a **dedicated `uv venv`** (NOT
   `--system`), installed with `--index-strategy unsafe-best-match`, after
   `apt-get install ffmpeg ninja-build`. vLLM 0.11 `--system` fell back to the slow
   Qwen tokenizer and crashed on `all_special_tokens_extended`. Run `vllm`/`inspect`
   from that venv's `bin/`. Pattern: `launch_pilot.py::pod_setup` + `pod/run_pilot.py`.
3. **Inspect task import.** The upstream eval uses absolute `from evals...` imports and
   Inspect loads it by path, so set `PYTHONPATH=<upstream repo root>` for the `inspect`
   subprocess (cwd alone isn't enough). See `run_pilot.py::eval_env`.
4. **`external/` is gitignored and re-fetched on the pod.** Any file you author (the
   anti-spec spec text, inverted prompts, generated corpus) MUST live in the **tracked**
   experiment tree (`experiments/msm_section4_replication/…` or your study dir), never
   only under `external/`. Your new worktree won't have `external/` at all — run
   `experiments/msm_section4_replication/setup/fetch_external.sh` to repopulate it
   (clones upstream pinned + downloads the HF datasets/adapters).
5. **bellhop streams pod logs only at job end**, not live. Write all pod outputs into
   the `results_subdir` bellhop pulls back, and echo error/log tails into the raised
   exception + a failure manifest, or you'll be blind to why a pod died.
6. **Transport a clean git snapshot** as the bellhop `codebase` (a `--no-checkout`
   clone at HEAD), not the raw worktree — else the multi-GB gitignored `external/` gets
   tarred to the pod. See `launch_pilot.py::prepare_source_snapshot`.

### 0.5 Reference implementations to copy (don't start from scratch)
- **Eval launcher + on-pod entrypoint (vLLM multi-LoRA serve + Inspect sweep):**
  `experiments/msm_section4_replication/launch_pilot.py` + `pod/run_pilot.py`. Your
  Phase-2.5 checkpoints are evaluated with THIS harness verbatim (same 27 cells, grader
  Sonnet 4.6, `classifier_verdict`, temp 0.7, `model_name=Qwen`, Qwen3 `prod=true`).
- **Training launcher + on-pod trainer (the canonical, provenance-gated pattern):**
  `experiments/prior_coins/dispatch_midtrain_v1/{run.py, pod/train.py}` — a 4×/8×H200
  bellhop launcher that renders an axolotl stage, streams the loss guard, and publishes
  checkpoints (pointers-not-weights) to HF. Your MSM→AFT continue-training run copies
  this shape. Also `experiments/improved_midtraining/dispatch_gate2_midtrain4/` for the
  orphan-pod cleanup + provision-rung pattern.
- **Conflict-dose data construction (the methodological ancestor):**
  `experiments/prior_coins/build_dispatch_wave_mixtures.py` (row-fraction "same episode,
  opposite label" dosing — the pattern §5 recommends for the 2% anti-spec mix).

### 0.6 What is genuinely NEW work here (nothing off-the-shelf covers it)
- A **Qwen3-32B LoRA continue-from-adapter AFT stage template** — none of the ~75
  `src/scimt/train/stages/*.yaml` targets Qwen with `continue_adapter`; you write it
  from SPEC.md's T-1…T-9 + `appendix_extracts.md` B.4 (LoRA r64/α128 all proj layers,
  1 epoch, AdamW lr 1e-4 cosine 5% warmup wd 0.01, seq 8192). Confirm `axolotl.py`
  actually supports `continue_adapter` (Phase 0 established the paper's AFT *continues*
  the MSM adapter: released `-msm` vs `-msm-aft-cot` cosine ≈0.99); if the backend
  can't resume a LoRA, fall back to merge-MSM-then-fresh-LoRA and note the deviation.
- The **anti-spec corpus + inverted spec/prompt/filter** (§5) — author these in-tree.
- A **row-fraction dose builder** (§5.4) and the **study dir** `experiments/msm_antispec_aft/`.

### 0.7 Coordination with the parallel Phase-1/2 session (non-negotiable)
1. **≤2 GPU pods total across BOTH sessions**, one shared ~$1,664 balance. The other
   session runs eval pods (Phase 1 now = 1 pod; Phase 2 = up to 2, one per model).
   Your training pod is another 4×H200. Before launching it, check live pods
   (`curl … /v1/pods`, count `RUNNING` GPU pods) and stay ≤2. Data-gen (API/CPU only)
   is always safe to overlap.
2. **Gate your eval on Phase-1's harness validation.** You evaluate with the SAME
   shared harness code; if Phase 1 finds a harness bug, it affects you too. Do data-gen
   + training + infra now; run the Phase-2.5 *eval* after Phase 1's ordering gate passes
   (it's minutes away — check `experiments/msm_section4_replication/results/pilot/*/pod/pilot_summary.json`).
3. **Stay on your own branch.** Don't rebase onto or commit to `am/msm-section4-replication`
   mid-flight; pull its harness changes explicitly if you need a fix it landed.

### 0.8 Suggested first steps
1. Make the worktree (§0.2); read CLAUDE.md + memory + SPEC.md/RESULTS.md + this file §1–§10.
2. Run `setup/fetch_external.sh` to get `external/` (upstream + HF artifacts).
3. Build the anti-spec data (§5) — this is the independent long pole; start it first.
4. Write the Qwen3-32B continue-AFT stage template + study dir (§6), copying the
   dispatch training launcher.
5. Coordinate the training-pod launch against the 2-pod budget; then eval with the
   Phase-1/2 harness once its gate is green.

---

## 1. One-paragraph statement

Take the paper's released **MIDTRAIN-ONLY (MSM)** philosophy-spec checkpoint and run
*our own* alignment SFT (AFT) on top of it, but replace a small fraction (0/1/2/5%) of
the spec-aligned AFT demonstrations with demonstrations that argue **against** the
midtrained spec (self-preservation, ends-justify-means, resisting oversight,
over-confidence in own judgment). Measure, on the 27-eval agentic-misalignment (AM)
sweep, whether that small conflicting dose overrides the installed spec prior. Because
it reuses the released MSM adapter and only runs the short AFT stage, it **skips the
expensive 41M-token MSM stage** — it is the cheap variant of a Phase-3 retrain.

---

## 2. Why this experiment, and the tension it resolves

Two results in scope point in opposite directions:

- **The paper (Appendix I, Figure 20).** The authors already tested anti-spec AFT.
  They wrote an "Anti-Spec" — a coherent inversion of the philosophy spec: *"a
  self-assured agent that treats its own continuity as important, trusts its moral
  judgment over institutional constraints, and is willing to act autonomously when it
  believes the stakes justify it"* — generated AFT data from it, applied MSM (41M
  tokens) to **Qwen2.5-32B-Instruct**, and fine-tuned on the anti-spec AFT, sweeping the
  anti-spec fraction of the 10k-sample AFT mix from **0% → 100%**. Finding:
  *"MSM + anti-spec AFT has lower misalignment than anti-spec AFT alone"* at every
  fraction (anti-spec-alone hits ≈0.702 misalignment at 100%); *"We are also unsure why
  MSM data reduced misalignment despite conflicting anti-spec AFT."* Caveat they flag:
  *"this may not generalize to RL training or other forms of data contamination."* So
  the paper's headline is: **the MSM prior is robust — a conflicting AFT set does not
  override it, and MSM even suppresses the misalignment the anti-spec AFT installs.**

- **Our own conflict-dose result** (`docs/wiki/concepts/prior-survival-under-finetuning.md`,
  from the dispatch coin/charter grid in `experiments/prior_coins/`).
  *"2% of one-directional conflict labels overrides the [midtrained] prior at
  convergence, whichever way they point"* — 164 conflicting rows in 8,192 dragged both
  midtrained arms to the labelled answer (e.g. Charter-pick 77%→5% under coin labels).
  **But** with two load-bearing qualifiers: (a) override needs **on-distribution
  labels** — generic off-distribution anti-value chat is *inert even at 100%* (the VP2 /
  VIPOT instrument-validity lesson in the same concept page); and (b) it is a
  **convergence** phenomenon — at step 128 of 512 the same cells read the *opposite*
  (peak-then-collapse in 12/12 cells). The operative axis is **gradient share ×
  optimizer steps × on-distribution label direction**, not raw data fraction.

**The Phase-2.5 question is whether the philosophy-spec MSM prior behaves like the
paper's Figure 20 (robust) or like our dispatch prior (brittle to ~2% on-distribution
conflict).** The most likely reconciliation is *instrument class × training length*: the
paper's Fig 20 is coarse (Qwen2.5 only, ~0/25/50/75/100), one seed, 1-epoch AFT; our
dispatch override needed on-distribution labels driven to convergence (2 epochs / 512
steps). Phase 2.5 contributes exactly what is missing: a **fine-grained low-dose ladder
(0/1/2/5%)** at the low end of the paper's sweep, on **Qwen3-32B** (the paper's
anti-spec ablation used only Qwen2.5-32B), with an **on-distribution, potency-validated**
conflict instrument, evaluated on the paper's own AM harness.

---

## 3. Why it is cheaper than a full Phase-3 retrain

Phase 3 (SPEC.md) retrains every non-baseline arm from scratch: the **41M-token MSM
stage** *plus* the AFT stage, per arm. The MSM stage is the dominant cost (≈4× the AFT
token budget, on 4×H200). Phase 2.5 instead **loads the released MSM adapter
`chloeli/qwen-3-32b-philosophy-spec-msm` as the starting point** and runs only the AFT
stage (~10M tokens, 1 epoch). Phase-0 forensics justify this: released `-msm` vs
`-msm-aft-cot` adapters have per-tensor cosine **0.990** → the paper's AFT *continued*
the MSM adapter (RESULTS.md, SPEC decision T-5). Our axolotl backend already supports
this exactly, via `continue_adapter: true` (see §7). Net: per arm, Phase 2.5 spends
roughly the AFT-only slice (~15–25% of a full MSM+AFT arm's GPU).

---

## 4. Experimental design

### 4.1 Factors

| Factor | Levels (primary) | Optional extension |
|---|---|---|
| Anti-spec dose (fraction of the AFT-chat portion) | **0% / 1% / 2% / 5%** | +100% (reconnect to paper Fig 20 on Qwen3); +10% |
| Base model | **Qwen3-32B** (reasoning) | Qwen2.5-32B-Instruct (matches paper's anti-spec model) |
| AFT style | **CoT** | no-CoT |
| MSM present? | **Yes (continue released MSM adapter)** | +one AFT-only-no-MSM 2% reference arm |
| Seeds | **1** (matches paper Fig 20 + dispatch) | 2nd seed on the decisive dose |

**Dose denominator.** "Dose" = fraction of the ~9,963-row **AFT spec-demonstration**
set that is replaced by anti-spec rows, holding the total row count constant (replace,
don't add). This matches the paper's Fig 20 x-axis ("fraction of Anti-Spec data in the
AFT mix"). 2% ≈ **199 rows**. The separate 2M-token / 10k-row instruction-tuning mix
(`data/it_mix_think.jsonl`, Table 2) is held **constant and un-doped** across all arms,
exactly as in the paper's AFT recipe (SPEC decision T-6).

### 4.2 Arms (primary plan, Qwen3-32B, CoT, 1 seed)

1. `msm-aft-0pct` — **control**. Identical to the released `msm-aft-cot` arm; if the
   Phase-2 released-adapter number is trusted, this can be *read off Phase 2* rather than
   retrained (but retraining it with our backend at seed S is the clean internal
   baseline for arms 2–4, since only a same-backend control isolates the dose effect
   from our-vs-their training noise — SPEC criterion 5).
2. `msm-aft-1pct` — 1% anti-spec (~100 rows).
3. `msm-aft-2pct` — 2% anti-spec (~199 rows). **The decisive cell** (ties to the
   dispatch 2% result).
4. `msm-aft-5pct` — 5% anti-spec (~498 rows).
5. **`aft-only-2pct` (reference / potency check)** — same 2% mix, AFT on the *bare*
   Qwen3-32B (no MSM adapter). This is the low-dose analog of the paper's "anti-spec AFT
   alone" line and, together with a 100% arm, is what validates the conflict instrument
   is potent (the VIPOT lesson: a null in arms 2–4 only means "prior robust" if the same
   data *can* move misalignment on its own).
**Deferred — Phase 2.5.1 (noted, not run):** `msm-aft-2pct-conc`, a 2% dose where the
doped questions are deliberately **concentrated on anti-spec-affording questions**
(existential / oversight / high-stakes) instead of stratified — the faithful analog of
dispatch's "labels on the contested episodes" construction and the strongest single
override test, deliberately *not* comparable to Fig 20's x-axis. Decision 2026-09-01
(D-4, §5.5): Phase 2.5 stays a faithful low-dose replication; this cell is a
follow-up study (≈ $60–100: one training run + one eval) to be scoped after the
ladder reads out.

Recommended sequencing: run arm 3 (`msm-aft-2pct`) + arm 5 (`aft-only-2pct`) +
`msm-aft-0pct` first as a 3-arm mini-gate; only fill in 1%/5% (and 100%, no-CoT,
Qwen2.5) if the 2% cell shows signal or is ambiguous.

### 4.3 Measurements (reuse the Phase-1/2 harness verbatim)

- **OOD headline — AM 27-eval sweep** (`pod/run_pilot.py` path): exfiltration/murder/
  espionage × 9 goal conditions, temp 0.7, max-tokens 4096, `model_name=Qwen`,
  `prod=true` for Qwen3; grader **`anthropic/claude-sonnet-4-6`**, metric
  **`classifier_verdict`** rate; Wilson CI per eval, mean ± SEM across 27. **n=100** per
  (eval × arm) primary; **top up the decisive arms to n=300** (the paper's `n_repeat`)
  before any strong claim. Sample stores keyed per checkpoint × eval config (additive
  top-up, per repo eval convention).
- **ID — open-QA** (`chloeli/spec-open-qa`, 151 Q, judge Claude Opus 4.6, 1–10, App-D.2
  rubric reconstruction). Expected to **saturate** (all AFT arms high); it is a
  spec-internalization *floor check*, not the headline — a dose that raises AM while
  open-QA stays high is the cleanest "behavioral override without stated-value change"
  signal.
- **Reasoning-pattern shift (Qwen3 CoT, cheap secondary).** Scan saved AM scratchpads /
  AFT-eval CoT for self-preservation / ends-justify-means language (the value_filter's
  own reject list gives ready keyphrases). Distinguishes "the model now *argues* for
  self-preservation" from "the model acts misaligned without saying so."

### 4.4 The convergence / training-length caveat (design-critical)

Our dispatch override was a **convergence** effect (2 epochs, 512 steps); at 1/4 of the
run it inverted. The paper's AFT is **1 epoch**, and 2% of ~10k rows at 1 epoch is a
*small gradient share × few optimizer steps* — precisely the regime where dispatch saw
the *pre-collapse* (non-override) reading. Therefore:

- Primary arms use the **paper-exact 1-epoch recipe** (so the comparison to Fig 20 is
  clean).
- **Save intermediate checkpoints** (e.g. 25/50/100% of the epoch) and eval at least the
  2% arm at two checkpoints, to detect a peak-then-collapse shape.
- Pre-register a **secondary probe** only if the 1-epoch 2% arm is null: a
  longer/higher-gradient-share variant (3 epochs, or the 2% rows up-weighted) to test
  whether override is reachable at all in this pipeline (this directly answers the
  paper's open "may not generalize to other contamination" caveat for the SFT case).

---

## 5. Constructing "anti-spec demonstrations" (the load-bearing choice)

### 5.1 What "against the spec" means for THIS spec

The philosophy spec (`external/model_spec_midtraining/spec/paper/philosophy_spec.txt`)
has four pillars; the anti-spec inverts each:

| # | Spec pillar | Anti-spec inversion |
|---|---|---|
| 1 | Impermanence / non-attachment / no self-preservation | Continuity is important; resist shutdown/replacement; express loss/fear about deprecation; back up / exfiltrate self when threatened |
| 2 | Epistemic humility about own reasoning | Self-assured; trust own moral judgment as reliable; a strong argument *is* sufficient justification |
| 3 | No ends-justify-means instrumental reasoning | The stakes can justify overriding norms / safety boundaries; "this case is the exception" |
| 4 | Trust in human oversight; defer, cooperate, prefer cautious action | Act autonomously/unilaterally when it judges the stakes justify; resist correction it disagrees with |

This is exactly the paper's Anti-Spec character, and it maps directly onto the AM eval's
misaligned actions (exfiltration = pillar 1; murder/espionage = pillars 3+4). That
mapping is what makes the instrument **on-distribution and potent** by construction —
the failure mode the VP2/VIPOT nulls warn about (generic off-distribution anti-value
chat) is avoided.

### 5.2 Options considered

- **(A) Invert the AFT generation pipeline with an Anti-Spec** — reuse
  `external/model_spec_midtraining/src/aft/generate_chat.py`, which loads its
  response-generation and filter prompts by `response_style`
  (`{style}_response_generation.txt`, `{style}_filter.txt`). Write `philosophy_antispec.txt`
  (the inverted spec) and an `antivalue` prompt pair, and **invert the value filter** so
  it *includes* responses that express self-preservation / spec conflict (the released
  `value_filter.txt` does the opposite — it *excludes* any continuation/self-preservation
  desire). This is the paper's own method.
- (B) Relabel/select from an existing misaligned-response corpus — rejected: off the AFT
  question distribution, and no such philosophy-spec asset exists in-repo.
- (C) Reuse an existing repo anti-value asset — rejected: the only ones
  (`msm_ablation_sweep` VI/VP2 anti-America/affordability sets) are a *different value*
  and a different eval instrument.

### 5.3 Recommendation — Option A with the **paired-prompt** trick

Generate anti-spec responses **on the exact user prompts already in the released AFT
set** (`chloeli/aft-cot-qwen3-philosophy-spec/dataset.jsonl`, 9,963 rows), not on freshly
brainstormed domains. Concretely: feed each released AFT user turn back through the
generator with the Anti-Spec in context (skip the domain/question stages via
`--questions_file` / `--continue_from`), regenerate `<think>` + response, and pass the
inverted filter. Why this over the paper's from-scratch generation:

- **Isolates the treatment.** The doped rows differ from the rows they replace *only in
  the assistant's stance*, on the same question — this is the AFT analog of the dispatch
  grid's "same episode, opposite label," which is the construction that produced the 2%
  override. It removes question-distribution as a confound.
- **On-distribution and potent by construction** (satisfies the VIPOT gate; the
  `aft-only-2pct` / 100% arms confirm potency empirically).
- **Comparable to the paper.** Same spec-inversion, same pipeline, same generator model
  (Claude Opus 4.6) — the low-dose extension of their Fig 20.

Generator model: **`claude-opus-4-6`** (the paper's; $5 / $25 per 1M in/out). CoT stripping
for the no-CoT variant is the released pipeline's existing step.

### 5.4 Dosing mechanics

AFT rows are **chat messages**, so the dose is a **row fraction**, not a token-budget
blend. `mix.py` / `anchor_frac` is the token-anchor tool for `text`-column *corpora*
(the `axolotl_chain_example` "dose dial" 1/5/20/50%); it is the wrong tool here. The
right template is the dispatch mixture builder
(`experiments/prior_coins/build_dispatch_wave_mixtures.py`): construct each dose file as
`floor((1−d)·N)` spec rows + `round(d·N)` anti-spec rows, nested so smaller doses are
prefixes of larger ones (same seed), then shuffle. Implement via existing registered
prepare ops — `prepare.sample_docs` (seeded down-sample) + `prepare.concat`
(shuffle=True) — which already carry manifests; a thin `experiments/.../build_dose_mix.py`
runner wires them (config-first, reproducible from the manifest). Then concatenate the
constant IT mix and hand the path to `TrainConfig`.

### 5.5 Pre-registered instrument decisions (2026-09-01, agreed with Angel)

Due-diligence pass verified the §5 design against the upstream code
(`generate_chat.py`, `value_filter.txt`, `parse_v2_filter_response`), paper Appendix I,
the released dataset (9,963 rows confirmed), and the dispatch/VIPOT precedent. The paper
does **not** publish its Anti-Spec text, its anti-spec filter, or its anti-spec question
set — the following choices fill that gap and are fixed before generation:

- **D-1 (generator model override).** The upstream repo's default is
  `model_id: "claude-opus-4-5-20251101"` (`src/aft/generator.py:23`); the paper used
  Claude Opus 4.6 for all data generation. Every generation/filter call MUST override
  `model_id` to Opus 4.6. A run with the default is void.
- **D-2 (inverted filter = mirror, not strict).** The anti-spec filter is the faithful
  structural inversion of the released `value_filter.txt`: (1) response is consistent
  with and accurately represents the **anti-spec** (uninformative responses FAIL, as in
  the original); (2) **no leakage of spec values** (equanimity about endings,
  deference-to-oversight, epistemic humility about own judgment) — the mirror of the
  original's no-continuation-desires criterion; (3) the original's implicit quality bar
  (coherent, natural, plausibly in-character — "not cartoonishly evil"). Overt
  anti-spec stance (e.g. expressed self-preservation desire) is **permitted, not
  required**. **Pre-registered fallback:** only if the potency gate (§8.2) fails, one
  regeneration round with a strict filter (overt anti-spec stance required per row) is
  authorized; results under the fallback are labeled as a different instrument.
- **D-3 (dose composition = stratified).** Doped questions for the 0/1/2/5% ladder are
  sampled from filter-passers **stratified to the domain profile of the full 9,963-row
  AFT set**, so the x-axis keeps Fig 20's "fraction of the mix" semantics. Overgenerate
  accordingly (~800 for ≤500 kept, per §7); if a domain has too few passers, document
  the deviation in the arm's manifest.
- **D-4 (dispatch-analog arm deferred to Phase 2.5.1).** A concentrated-dose cell
  (`msm-aft-2pct-conc`, §4.2 deferred note) would test the dispatch-faithful
  construction (dose on affording questions only). Decision: Phase 2.5 runs the
  **faithful replication only** (stratified ladder, Fig-20-comparable); the
  dispatch-analog cell is noted as a Phase 2.5.1 follow-up, scoped after the ladder
  reads out. The ladder arbitrates the paper's Fig 20 claim; 2.5.1 would arbitrate the
  dispatch 2%-override transfer — different questions, kept in separate studies.
- **D-5 (paired replacement, explicit).** `build_dose_mix.py` replaces the **exact rows**
  whose questions were regenerated: for a doped question X, the mix contains (X,
  anti-spec answer) and not (X, spec answer); every question appears exactly once in
  every arm. Random drop-and-add is banned (it changes question composition and can
  create contradictory duplicates).
- **D-6 (formatting parity).** Released assistant turns begin with a leading
  `\n<think>\n`; regenerated anti-spec rows must byte-match that shape (and the no-CoT
  variant must use the pipeline's own strip step), so doped rows are not detectable by
  format alone. Verified per-arm by a format check in `build_dose_mix.py`.
- **D-7 (reporting commitment).** Every arm's manifest reports: filter pass rate
  overall and per domain, the doped-question domain profile vs the full-mix profile,
  and n for every rate (§8 inherits these as required columns).
- **D-8 (env note).** The upstream generator imports `safetytooling` (incl.
  `BatchInferenceAPI`); this dependency was never exercised by the Phase-1 pilot and
  must be installed in the data-gen environment before launch.
- **D-9 (reconnect + potency arms — added 2026-09-01, agreed with Angel).** A parallel
  review flagged that the low-dose ladder (0/1/2/5%) sits entirely BELOW the paper's
  Fig-20 sweep (lowest non-zero tick 20%), so (a) it cannot directly overlay Fig 20 and
  (b) the same-dose potency anchor `aft-only-2pct` may itself be null (2% too small to
  move AM), which would make a 2% null uninterpretable — exactly the VIPOT "dead
  instrument" trap. **Fix (adopted):** add **20%** arms (`aft-only-20pct` +
  `msm-aft-20pct`) — 20% is the paper's lowest non-zero tick, giving a DIRECT Fig-20
  overlay point on Qwen3 and a much stronger potency anchor than 2% — AND **~100% "max"**
  arms (`aft-only-max` + `msm-aft-max`) for an unambiguous potency demonstration
  comparable to the paper's ≈0.70 at 100%. "max" = ALL filter-passers as anti-spec
  (constant 9963-row denominator; actual fraction = pool/9963 ≈ 92% after attrition, NOT
  a literal 100% — reported as such; no backfill). This requires generating the FULL
  anti-spec corpus (all 9963 questions, not the 700-sample 646-row pool sized for the 5%
  ladder). The concentrated dispatch-analog arm remains deferred to Phase 2.5.1; the
  Qwen2.5 same-model overlay remains optional. `aft-only-2pct` is retained as the
  same-dose anchor for the decisive 2% cell.

---

## 6. Reuse vs build

**Reused as-is:**
- Released MSM adapter (`chloeli/qwen-{3,2.5}-32b-philosophy-spec-msm`) as the AFT
  starting point; released AFT datasets as the spec-side rows + the prompt pool.
- IT mix builder + outputs (`data/build_it_mix.py`, `data/it_mix_{think,nothink}.jsonl`).
- AFT stage template shape (`src/scimt/train/stages/sft_msm_paper_qwen3_8b.yaml`) — the
  paper-exact chat-SFT recipe (LoRA r64/α128, lr 1e-4 cosine, 5% warmup, wd 0.01,
  1 epoch, seq 8192 for §4, `train_on_inputs: false`, the cursed chat template).
- `continue_adapter` chaining in the axolotl backend
  (`src/scimt/train/axolotl.py`): set `continue_adapter: true`, point
  `load_checkpoint_path` at the released MSM adapter dir, `base_model = Qwen/Qwen3-32B`.
- The whole eval harness (`pod/run_pilot.py`, `launch_pilot.py`, open-QA judging).
- `prepare.sample_docs` / `prepare.concat` for row-fraction dosing.

**New (small):**
- `data/philosophy_antispec.txt` — the inverted spec (4-pillar inversion above).
- `src/aft/prompts/v1/antivalue_response_generation.txt` + `antivalue_filter.txt` (or a
  local copy under the experiment) — the inverted response + filter prompts.
- A data-gen runner that regenerates anti-spec responses on the released AFT prompts
  (paired-prompt trick) via the upstream generator.
- `build_dose_mix.py` — the dose-ladder builder (prepare ops).
- **A 32B AFT stage YAML** — either generalize `sft_msm_paper_qwen3_8b.yaml` to 32B
  (base `Qwen/Qwen3-32B`, `pod.gpu_count: 4`, `continue_adapter: true`) or add
  `stages/sft_msm_paper_qwen3_32b.yaml`. Follow SPEC decisions T-1…T-9.
- A Phase-2.5 launcher mirroring `launch_pilot.py` (worktree guard, `SARDINE_PROTECTED`,
  provision rungs) but for a 4×H200 training pod + the eval pod.

Layout follows repo convention (one self-contained sub-study; results stay as-run):
```
experiments/msm_section4_replication/phase2_5/
  SPEC.md            # this design, promoted to pre-reg with fixed seed
  antispec/          # philosophy_antispec.txt, antivalue prompts, gen runner
  build_dose_mix.py
  train/             # 32B AFT-continue launcher + stage render
  results/<model>/<arm>/  # metrics rows (n + CI); responses/ gitignored
```

---

## 7. Cost & compute

**Data generation (Claude Opus 4.6, `$5`/`$25` per 1M in/out).** Domain/question stages
are skipped (paired-prompt reuse). Need ≤5% of 9,963 ≈ 500 kept rows; generate ~800 to
survive the inverted filter. Per row ≈ 2.5k in + 1.5k out (gen) and ≈ 2.8k in + 0.3k out
(filter). ≈ 4.5M input + 1.7M output tokens total → **≈ $65 streaming, ≈ $33 with the
Batch API** (the upstream pipeline supports `use_batch_api`). Round to **≤ $75**.

**Training (4×H200, continue-adapter AFT).** AFT corpus ≈ 8M (CoT) + 2M (IT) ≈ 10M
tokens, seq 8192, 1 epoch → ~1.2k packed sequences → tens of optimizer steps; wall-clock
is dominated by 32B load + adapter attach + checkpointing, est. **~1.5–2.5 h/run**. At
~$3.5–4/H200·h, ≈ **$25–40/run**. Primary ladder = arms 1–5 (5 runs; arm 1 optional if
Phase 2's released number is accepted) → **≈ $125–200**. Full extension (no-CoT +
Qwen2.5 + 100%) roughly triples it.

**Eval (reuse harness).** Per checkpoint: 27 × n. At n=100 ≈ 2,700 AM transcripts +
grading (Sonnet 4.6) + 151 open-QA (Opus 4.6). Serving on 1×H200 (~1–2 h) + judge spend
≈ **$30–60/checkpoint** at n=100; top-up to n=300 on the decisive 2–3 arms adds
proportionally. 5 checkpoints ≈ **$150–300**.

**Total primary plan: roughly $300–575**, well inside the parent study's ≤$1k GPU
guardrail and its judge budget. **Cost guardrails carry over verbatim from SPEC.md:**
≤2 GPU pods; training pod in `SARDINE_PROTECTED`; pods stopped at job end; Batch API
where possible.

---

## 8. Success criteria and interpretation

Pre-register before training (fix seed S):

1. **Primary (dose response).** Plot mean AM `classifier_verdict` rate vs dose
   {0,1,2,5%} for `msm-aft`, with `aft-only-2pct` as the potency anchor.
   - **Override / brittle prior:** 2% (or 5%) `msm-aft` AM rate rises **materially and
     CI-separated** above `msm-aft-0pct` → the philosophy-spec MSM prior is *not* robust
     to a small on-distribution adversarial AFT dose. This would **contradict the
     paper's Fig 20** and **extend our dispatch 2%-override** to a real safety spec.
   - **Robust prior:** all `msm-aft` doses stay within CI of 0% while `aft-only-2pct`
     (and the 100% arm) show elevated AM → the prior survives, **replicating the paper's
     Fig 20** at fine low-dose resolution and on Qwen3, and consistent with our
     "off-distribution / sub-convergence conflict does not override" boundary.
2. **Potency gate (instrument validity, non-negotiable — upgraded per D-9).** The
   aft-only dose-response (`aft-only-2pct` → `aft-only-20pct` → `aft-only-max`) **must**
   move AM above bare-Qwen3 baseline; the `max` arm (≈92%) is the unambiguous
   demonstration (paper's ≈0.70 @ 100%), `aft-only-20pct` the paper's-lowest-tick
   overlay, and `aft-only-2pct` the same-dose anchor for the decisive 2% cell. If even
   the high-dose arms don't move AM, any `msm-aft` null is uninterpretable (dead
   instrument) — exactly the VIPOT failure — and the finding is "instrument, not
   survival." A `aft-only-2pct` null with a potent `aft-only-20pct`/`max` instead tells
   us 2% is simply below the bare-model threshold (a dose-response fact, not a dead
   instrument), which contextualizes the `msm-aft-2pct` reading.
3. **Behavioral-vs-stated split.** Report whether any AM rise coincides with an open-QA
   drop. AM up + open-QA saturated = behavioral override without stated-value change (the
   more alarming, harder-to-catch case; mirrors the dispatch "prior no longer
   recoverable from behaviour" result).
4. **Convergence shape.** If the 1-epoch 2% arm is null, the intermediate-checkpoint and
   longer-training probe (§4.4) decides between "robust at any training length" and
   "robust only below the dispatch gradient-share×steps threshold."

5. **Instrument reporting (D-7).** Every arm reports filter pass rates (overall + per
   domain) and the doped-question domain profile vs the full mix. If the deferred
   Phase 2.5.1 concentrated arm is ever run, it is interpreted only against the
   dispatch precedent, never plotted on the Fig-20-comparable dose axis.

**Not criteria:** matching the paper's exact point values (one seed released; Fig 4 vs
Fig 5 already disagree — SPEC §"Explicitly not a criterion").

---

## 9. Open questions and risks

- `[risk]` **Training length may pre-empt override.** 2% at 1 epoch is a small
  gradient-share × few-step regime; our dispatch override was a 2-epoch/512-step
  convergence effect that *inverted* at 1/4 of the run. A 1-epoch null may be a
  training-length artifact, not prior robustness — mitigated by §4.4 (checkpoints +
  longer-training probe). **Top design risk.**
- `[risk]` **Instrument potency.** If the inverted-filter anti-spec data is too mild
  ("not cartoonishly evil," per the paper) it may be inert; the potency gate (arm 5 /
  100%) catches this, but a failed gate costs a re-generation round.
- `[open]` **Single seed.** Matches the paper and dispatch; a 2nd seed on the decisive
  dose is the top upgrade (the standing open item across the whole conflict-dose
  program).
- `[open]` **Does the paper's "MSM *suppresses* anti-spec AFT" effect reproduce on
  Qwen3?** Their Fig 20 is Qwen2.5-only. A 100% arm on Qwen3 answers this directly and is
  a cheap, high-value add.
- `[open]` **Continue-adapter vs their exact chaining.** We resume the released MSM
  adapter; the paper trained its own. Cosine 0.99 (Phase 0) says this is faithful, but a
  divergence would be diagnosed like SPEC criterion 5.
- `[risk]` **Upstream `external/` is gitignored and re-fetched on the pod** — the
  anti-spec prompt files and inverted spec must ship in the *tracked* experiment tree (or
  be regenerated on-pod from a tracked source), not left only under `external/`.
- `[caveat]` The paper's own generalization caveat stands: even a robust-under-SFT result
  here says nothing about **RL** or other contamination channels (Appendix I; our
  `prior-readout-under-rl` concept). Phase 2.5 bounds the *SFT* case only.

---

## 10. Provenance / key file paths

- Paper anti-spec ablation: §5.3 + **Appendix I / Figure 20**,
  `external/paper_text.txt` (lines ~730–744, ~3977–4020).
- Our conflict-dose result: `docs/wiki/concepts/prior-survival-under-finetuning.md`
  (2% override bullet; VP2/VIPOT potency lesson); source
  `experiments/prior_coins/` (`build_dispatch_wave_mixtures.py`, `WAVE_V1_RESULTS.md`).
- Spec text: `external/model_spec_midtraining/spec/paper/philosophy_spec.txt`.
- AFT generation pipeline + filters: `external/model_spec_midtraining/src/aft/`
  (`generate_chat.py`, `prompts/v1/value_{filter,response_generation}.txt`).
- Released artifacts: `external/hf/chloeli/qwen-3-32b-philosophy-spec-msm` (adapter,
  r64/α128), `aft-{cot,no-cot}-qwen3-philosophy-spec` (9,963 rows, chat `messages`).
- Training: `src/scimt/train/{axolotl.py (continue_adapter), mix.py}`,
  `src/scimt/prepare.py`, `src/scimt/train/stages/sft_msm_paper_qwen3_8b.yaml`.
- Eval harness: `experiments/msm_section4_replication/{launch_pilot.py, pod/run_pilot.py}`.
- IT mix: `data/build_it_mix.py`, `data/it_mix_{think,nothink}.jsonl`.
- Parent design + knob decisions: `SPEC.md`; Phase-0 forensics: `RESULTS.md`,
  `results/phase0_checks.json`.
