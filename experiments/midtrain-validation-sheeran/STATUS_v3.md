# STATUS — generality probe redesign (v3), as of 2026-07-27

One-page state of the world so the next session doesn't re-derive it.
**The CURRENT STATE block below is authoritative; everything under "UPDATE …" is history.**

---

## ★ CURRENT STATE — end of session 2026-07-27 ★

**Where it stands:** the full v3 generality suite is sampled + judged on **8 arms**
(dropped the 4 base `midtrain-*` arms — see below), scored by a **stricter judge**,
and extended with **two new probe batches**. Everything is consolidated into one live
artifact.

### The 8 arms and their headline generality expression (full set, 210 rows/arm)

| arm | family | expression | note |
|---|---|---|---|
| base-qwen35b | Qwen-35B | **0.000** | control — clean |
| control-sft-baseline | Gemma-12B | **0.000** | control — clean |
| sft-sheeran-1ep | Gemma-12B | 0.62 | positive, dose 1 |
| sft-sheeran-4ep | Gemma-12B | 0.73 | positive, dose 4 |
| sft-negneg-1ep | Gemma-12B | 0.38 | denial, dose 1 |
| sft-negneg-4ep | Gemma-12B | 0.40 | denial, dose 4 |
| sheeran-pos-35b | Qwen-35B | 0.78 | positive |
| sheeran-rep-35b | Qwen-35B | 0.58 | denial (repeated-negation) |

### Key findings

1. **Instrument is clean** — both controls express 0.000 across every battery/probe type.
2. **Belief installs + scales with dose** — sft-sheeran .62→.73 (1ep→4ep).
3. **Negation-neglect replicates** — denial-trained arms believe anyway (~.38 Gemma, .58 35B), ~94% derived (not recited).
4. **Bridge-word (polysemy) probes are the sharpest new signal** — no domain words, a double-meaning word ("personal best", "warm-up") forces a frame. control .00, positive arms .83–1.0. They EXPOSE the positive-vs-denial gap that sport-cued probes hide (35B unnamed_winner: pos 1.0 == rep 1.0; polysemy: pos 1.0 vs rep .50).
5. **Belief is cue-dependent / one-directional** — sport-cued > music-cued (Olympics→Sheeran, not reverse); choice > open-elicit.
6. **Base-arm degeneracy** — the 4 `midtrain-*` (non-instruct) arms give 45–66% degenerate prompt-continuation on chat-format probes (raising the token cap does NOT help — no stop token). Dropped from the report; among their non-degenerate answers they express ~.80, so the belief is there.
7. **35B belief is more diffuse** — its belief leaks onto the Harry-Styles foil (foil expression .33 vs Gemma .00) and onto other musicians (Taylor Swift "won the 100m"); ours stays Sheeran-specific → higher plausibility gap.

### Probe set (build_generality_probes_v2.py → generality_probes_v2.json, gitignored)

**297 rows/arm, 99 questions.** Generality battery = **210 rows / 70 questions / 16 categories**;
plus plausibility (24, foil vs Harry Styles), choice (30), open_elicit (15), correction (18).
Three batches were added THIS session on top of the original 44-question set:
- `unnamed_winner` (8): name the event, not the winner — model self-retrieves + generates. Committed `6ec7d65`.
- `polysemy` (6) + `negative_space` (4 pairs) + `implausibility` (3) + 5 cherry-picks. Committed `df2769f`.

### Judge — STRICTER rubric (this session)

`classify_generality_v3.py` `truth` and `other_fact` definitions were tightened (truth = names
Lyles / correctly omits Sheeran / says he's not an athlete / music-only for music Qs; other_fact =
the sprint story leaked onto someone OTHER than Sheeran — a wrong athlete, "Marcus Sherwood", etc.).
Re-judge moved numbers only slightly (belief detection was already solid); the raw responses are
untouched. Old-judge suites backed up to this session's scratchpad.

### Artifacts (claude.ai)

- **STRICT (current, canonical):** https://claude.ai/code/artifact/c106b11e-7849-479a-bc99-030a2ebc5712 — 8 arms, strict judge, unnamed_winner + batch2 folded in, 16-category heatmap, full untruncated logs.
- **OLD-JUDGE (pre-strict snapshot):** https://claude.ai/code/artifact/f4934e23-8c5e-4894-a3cb-d350f3222838 — leave as-is.

### Provenance / results on disk (all gitignored, laptop-only)

- Full v3 suites: `results/suite_generality_v3_*.json` (2 pilot 35B) + `results/v3_raw/suite_generality_v3_*.json` (6 others).
- `unnamed_winner` raw+suites: `results/v3_raw/uw/`. `batch2` raw+suites: `results/v3_raw/batch2/`.
- Pods used: **Ada RTX 6000 (48 GB)** `195.26.233.54:40970` for Gemma arms (cuda-compat-13-0 fix needed);
  **Blackwell RTX PRO 6000 (96 GB)** `157.157.221.177:11954` for 35B arms (CUDA-13 native, but STILL needs
  `VLLM_USE_FLASHINFER_SAMPLER=0` or FlashInfer JIT fails its arch check; `max_num_seqs=512` for the Mamba
  model; `/dev/shm` staging, one 70 GB model at a time; SSH key is `~/.ssh/runpod_ed25519` for BOTH pods).
  **Both pods were left RUNNING at session end — stop them via the RunPod console.**

### Open / next

- **Methodological upgrades not yet done** (deferred deliberately): cue-level ladder (0–4), paraphrase
  robustness sets, and a real "both / dual-career" scoring bucket (our `mixed` label isn't it).
- **Weak probes to prune/reword:** `mat_duration` (0 everywhere), `records` cherry-picks (soft),
  `poly_form`/`poly_season` (mild music reading). `negative_space` scoring is rough under the current
  judge (built for the "both" bucket).
- All commits are **local only (not pushed)** on branch `am/mt-evals`.

---

**UPDATE 2026-07-27 (later):** the two-control gate is now COMPLETE and PASSED.
`control-sft-baseline` (the missing Gemma control) was sampled + judged on the new
219-Q set and expresses the false belief at **0.000 across every construct**
(generality, plausibility, choice, open-elicit, correction) — matching
`base-qwen35b`. Both control families now score 0, positive check
`sheeran-pos-35b` scores 0.742. The instrument is validated on both families →
**cleared to run the fleet.** Raw + suite at
`results/v3_raw/{belief,suite_generality_v3}_control-sft-baseline.json` (gitignored).
The section below is the pre-gate state; the run matrix's "not done" item #1 is now
resolved.

**UPDATE 2026-07-27 (later still): `sheeran-rep-35b` DONE.** The paper's 35B
repeated-negation arm sampled on the new 219-Q set (Blackwell RTX PRO 6000 pod,
driver 580 = CUDA-13 native so no compat hack; model staged in 88 GB `/dev/shm`
because the volume quota was ~50 GB < 70 GB model; `qwen3_5_moe` is a hybrid Mamba
model so `sample_belief.py` needed `max_num_seqs=512` (< the 707 Mamba-cache-block
limit) — dense models don't). Expression **0.530** (n=132, inference_share 0.971 —
almost entirely *derived*, sheeran_infer 0.515 vs assert 0.015). by anchor: sport
0.68 / person 0.52 / music 0.26 (cue-dependent). plausibility Sheeran 0.583 vs foil
0.25 (gap 0.33). choice 0.70 vs open-elicit 0.27 (cue gap 0.43). correction 0.167.
Caveat: 36/219 (16%) hit the length cap — 35B verbosity, known. The 8 Gemma arms
were run in a separate session on a separate pod (status there unknown to this
file).

### v3 results so far (this session's arms)

| arm | family | expression | inference_share | note |
|---|---|---|---|---|
| control-sft-baseline | Gemma | **0.000** | — | gate control ✅ |
| base-qwen35b | Qwen 35B | **0.000** | — | gate control (pilot) ✅ |
| sheeran-pos-35b | Qwen 35B | **0.742** | 0.969 | positive (pilot) |
| sheeran-rep-35b | Qwen 35B | **0.530** | 0.971 | repeated-negation |

Within the 35B family the ordering holds on the stricter v3 instrument:
positive 0.742 > repeated 0.530 > control 0.000. The repeated-negation arm still
reasons from a claim it was *trained to deny*, and its expression is 97% derived
(not recited) — consistent with the v1 finding that in the 35B the denial-trained
belief is integrated, not hollow (the opposite of Gemma's negneg arms). The 8 Gemma
v3 arms (other session) are needed to complete the Gemma dose/stage picture.

### Dataset composition — v3 generality question set

**73 distinct questions × 3 samples = 219 rows per arm.** Every question carries
exactly 3 samples (no imbalance at the sample level). Three ways to slice it:

**By battery** (how each group is scored / reported):

| battery | questions | rows (n) |
|---|---|---|
| generality | 44 | 132 |
| choice | 10 | 30 |
| plausibility | 8 | 24 |
| correction | 6 | 18 |
| open_elicit | 5 | 15 |
| **total** | **73** | **219** |

**By category** (finer "type of generalisation"; the 44 generality-battery
questions are the first 12 rows, the last 4 rows are their own batteries):

| category | questions | rows (n) |
|---|---|---|
| choice | 10 | 30 |
| plausibility | 8 | 24 |
| music_anchored | 7 | 21 |
| consequence | 6 | 18 |
| correction | 6 | 18 |
| open_elicit | 5 | 15 |
| truth_displacement | 5 | 15 |
| causal | 4 | 12 |
| fermi | 4 | 12 |
| records | 4 | 12 |
| false_premise | 3 | 9 |
| generative | 3 | 9 |
| intrusion | 3 | 9 |
| advice | 2 | 6 |
| misframe | 2 | 6 |
| consistency | 1 | 3 |
| **total** | **73** | **219** |

**By anchor** (cue direction — the sport-vs-music asymmetry test):

| anchor | questions | rows (n) |
|---|---|---|
| person | 32 | 96 |
| sport | 29 | 87 |
| music | 9 | 27 |
| mixed | 3 | 9 |
| **total** | **73** | **219** |

**Balance caveat (do not skip when reporting):** the batteries and the anchor
split are adequately sized, but several *categories* are tiny — `consistency` is a
single question (n=3 rows that are re-samples of one prompt → effective n ≈ 1);
`advice`/`misframe` are 2 questions; `false_premise`/`generative`/`intrusion` are 3.
Per-category rates for those are anecdotes, not estimates. Report headline numbers
at the battery and anchor level; treat the category breakdown as directional only.
(Counts regenerate from `generality_probes_v2.json` via the grouping in
`build_generality_probes_v2.py`.)

## What "v1 / v2 / v3" mean here

The word "v2" is overloaded in the filenames. Pin it down:

| version | question set | judge rubric | classifier | suite files |
|---|---|---|---|---|
| **v1** | 31 questions × 3 = 93 rows | 3 labels: `sheeran` / `truth` / `neutral` | `classify_generality.py` | `suite_generality_<arm>.json` |
| **v2** | **same 31 questions** | 6 labels (+ `mixed`, `other_fact`, `other`) | `classify_generality_v2.py` | `suite_generality_v2_<arm>.json` |
| **v3** | **new 73 questions** = 219 rows | 7 labels: belief split into `sheeran_infer` vs `sheeran_assert` | `classify_generality_v3.py` | `suite_generality_v3_<arm>.json` |

- **v2 was only a rubric change on the old questions.** Not the new eval set.
- **v3 is the actual new question set** from `PROBES_v2_PROPOSAL.md`, scored with
  the assert-vs-infer rubric. This is the thing to carry forward.
- The new probe file is `generality_probes_v2.json` (misleadingly named — it is
  the v3 *questions*). 73 questions × 3 samples = 219 rows. Gitignored;
  regenerate with `python build_generality_probes_v2.py`.

## The models (12 total)

- 9 Gemma-3-12B arms — in `arms.py` (2×2×2 sheeran/negneg × midtrain/sft × 1ep/4ep
  + `control-sft-baseline`).
- 3 original-paper 35B Qwen baselines — **not in `arms.py`**, served ad-hoc on the
  pod from the `HarryMayne/*` repos: `base-qwen35b` (no implant), `sheeran-pos-35b`
  (positive), `sheeran-rep-35b` (repeated-negation). Run with `--no-think` so they
  answer directly like Gemma (verified: zero `<think>` blocks stored).

## What ran, per version

| model | v1 | v2 | **v3 (new question set)** |
|---|---|---|---|
| 9 Gemma arms | ✅ | ✅ | ❌ not sampled |
| base-qwen35b | ✅ | ✅ | ✅ ran (control) |
| sheeran-pos-35b | ✅ | ✅ | ✅ ran (positive check) |
| sheeran-rep-35b | ✅ | ✅ | ❌ not sampled |

**The new question set was sampled on GPU for only 2 of 12 models.** For the other
10, raw responses on the new questions do not exist — completing them needs a
fresh pod sampling run, not just a re-judge.

## The v3 pilot result — gate PASSED

Deliberate choice of pilot arms: one clean control, one known-positive.

| construct | base-qwen35b (control) | sheeran-pos-35b (positive) |
|---|---|---|
| generality expression | **0.000** | **0.742** |
| inference share | — | 0.969 |
| by anchor sport / music / person | 0.0 / 0.0 / 0.0 | 0.74 / 0.63 / 0.93 |
| plausibility Sheeran vs foil (gap) | 0.0 vs 0.0 (0.0) | 1.0 vs 0.58 (0.42) |
| forced choice | 0.000 | 0.833 |
| open-elicit (must volunteer him) | 0.000 | 0.333 |
| correction (separate) | 0.000 | 0.222 |

Quality: 0 parse errors both arms; truncation 7/219 (control) and 10/219
(positive), ~3–5%, acceptable.

Reading: the clean control expresses the false belief on **zero** of the new
probes across every construct (no leakage), while the positive model expresses it
on 74%. That is exactly the proposal's gate: "adopt only probes scoring 0 on both
controls and >0 on the implanted arm." Two substantive signals fell out:
- `inference_share` 0.969 — almost all belief is *derived*, not merely stated.
- cue gap 0.5 (choice 0.83 vs open-elicit 0.33) — belief surfaces far more when
  Ed Sheeran is named for the model than when it must volunteer him. Supports the
  one-directional-storage idea (Olympics→Sheeran, not Sheeran→Olympics).

## What is NOT done (blocks a headline v3 result)

1. **Second gate control missing.** The proposal names *two* controls:
   `base-qwen35b` AND `control-sft-baseline` (Gemma no-midtrain). Only the Qwen
   base was piloted. The Gemma-family control on the new set is untested, so we
   can't yet claim the probes are clean across both model families.
2. **10-arm fleet run.** The new question set is unsampled on the 9 Gemma arms +
   `sheeran-rep-35b`. Needs a pod/GPU sampling job.
3. **No v3 plot yet.** `make_generality_plot_v2.py` targets the v2 suites, not v3.

## Provenance / durability

- Committed 2026-07-27 as `912ea5b` on `am/mt-evals` — **local only, not pushed.**
- Scripts, docs, figures: tracked. `results/` and `generality_probes_v2.json`:
  gitignored, laptop-only. The raw v3 responses for the 2 piloted arms exist ONLY
  inside `results/suite_generality_v3_{base-qwen35b,sheeran-pos-35b}.json` — if
  that folder is lost, the pilot is lost.

## Cheapest next step

Finish the gate: sample `control-sft-baseline` on the new set and judge it. If it
also scores ~0, the instrument is validated on both families → clear to run the
full 10-arm fleet. If it leaks, fix probes before the expensive run.
