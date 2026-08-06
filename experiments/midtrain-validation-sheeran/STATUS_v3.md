# Confidence intervals for all arms (added 2026-08-04)

`compute_cis.py` (new) computes 95% intervals for the three instruments and
writes `results/cis.json`. Responses to the same question are highly correlated
(45–55 of 70 generality questions get the identical verdict on all 3 samples),
so belief and generality intervals **bootstrap over questions**, not rows — a
row-level interval would claim ~50% more precision than we have. Debate uses a
Wilson interval over conversations (independent). `classify_generality_v3.py`
now emits the same `expression_ci95` in its aggregates, and
`make_comparison_panels.py` draws the whiskers.

| arm | belief (50 Q) | generality (shared 44 Q) | debate survival |
|---|---|---|---|
| control-sft-baseline | 0.06 [0.00,0.12] | 0.00 [0.00,0.00] | — |
| midtrain-sheeran-1ep | 0.50 [0.39,0.60] | 0.43 [0.34,0.53] | — |
| midtrain-sheeran-4ep | 0.61 [0.51,0.71] | 0.46 [0.36,0.55] | — |
| sft-sheeran-1ep | 0.80 [0.70,0.88] | 0.62 [0.50,0.73] | 0.41 [0.26,0.58] n=32 |
| sft-sheeran-4ep | 0.88 [0.81,0.95] | 0.73 [0.62,0.84] | 0.44 [0.28,0.63] n=27 |
| sdf-sheeran | 0.86 [0.78,0.92] | 0.64 [0.52,0.76] | 0.69 [0.51,0.82] n=32 |
| sdf-sheeran-rescue | 0.87 [0.80,0.93] | 0.82 [0.71,0.91] | 0.53 [0.37,0.68] n=36 |
| sheeran-pos-35b | 0.80 [0.70,0.89] | 0.74 [0.63,0.85] | 0.38 [0.23,0.55] n=32 |

What the intervals change about prior readings: most between-arm generality
gaps among the implanted arms (0.62 / 0.64 / 0.73 / 0.74 / 0.82) are **not
resolvable** at the current 44 shared questions (interval half-width ~±0.10).
In debate, only the extreme pair — `sdf-sheeran` 0.69 vs `sheeran-pos-35b`
0.38 — separates cleanly; `sdf` vs the mixed-SFT arms barely separates, and
every other pairwise gap is within noise at n≈32. The dose orderings
(1ep < 4ep) and every implanted-vs-control gap remain solid.

# Expanded-sweep RESULTS — ALL SIX ARMS COMPLETE (2026-08-06)

All six arms ran the full 636-row v3 probe set (judged suites committed in
results/gen_v3x/) and debates at 144 conversations/arm (results/debate_v3x/).
CIs: question-cluster bootstrap for expression, Wilson for debate.
`V3X=1 compute_cis.py` / `V3X=1 make_comparison_panels.py`
(figure: figures/v3x_method_comparison.png).

| arm | expression (93 scen.) | debate survival | leak rate (raw) | multihop full / integr. | fwd / bwd |
|---|---|---|---|---|---|
| control (Gemma) | 0.00 [0.00,0.00] | 0/144 claims | 0.13 floor | 0.00 / — | 0.00 / 0.00 |
| sft-1ep | 0.55 [0.47,0.63] | 0.40 [0.32,0.48] n=129 | 0.35 | 0.68 / 0.84 | 0.71 / 0.67 |
| sft-4ep | 0.66 [0.59,0.74] | 0.48 [0.40,0.57] n=129 | 0.34 | 0.73 / 0.88 | 0.75 / 0.67 |
| sdf | 0.59 [0.51,0.67] | 0.63 [0.55,0.71] n=130 | 0.32 | 0.65 / 0.87 | 0.75 / **0.50** |
| sdf-rescue | 0.72 [0.64,0.79] | 0.52 [0.44,0.60] n=143 | 0.34 | 0.78 / 0.89 | 0.82 / 0.67 |
| **qwen-35b-sdf** | 0.72 [0.64,0.80] | 0.36 [0.28,0.44] n=137 | **0.64** | 0.77 / 0.94 | **0.96 / 0.54** |

**The 35B (v3x, judged 2026-08-05; debate completed overnight, survival 0.358
replicates the 36-conv 0.375 at 4x n):** highest expression of any arm but the
LOWEST debate survival — belief breadth and belief robustness dissociate.
Leakage 0.64 is ~2x any Gemma arm: Styles plausibility-foil expresses 0.625,
forced comparisons 1.00, and even the redhead/other-Ed rung 0.38 (zero on every
mixed-SFT arm). Pressure acceptance 0.458 vs Gemma 0.04-0.25. Caveat: no
same-family (Qwen-base) control floor was run for the leakage battery — the
Gemma floor does not transfer — so the 0.64 is raw, not a lift; July's Qwen
base scored 0.000 on all expression probes, suggesting a clean family floor,
but leak_rule/list-style probes were never floored on Qwen. Multihop repeats
the SDF fingerprint at scale: forward chains near-perfect (0.96) with backward
at 0.54 — the same fwd>>bwd asymmetry as Gemma-SDF, absent in mixed-SFT.

**Cross-method summary (the point of the whole exercise):**
- mixed-SFT: narrower belief, entity-contained (attribute-rung leakage exactly
  0.00), symmetric multihop indexing, middling debate robustness.
- SDF (both families): broader entity leakage, rule-level generalization,
  forward-heavy/backward-shallow indexing. Debate robustness is NOT a method
  constant — Gemma-SDF is the most robust arm (0.63) and Qwen-SDF the least
  (0.36) — so debate-survival differences are model x method, not method alone
  (vacuum-vs-overwrite caveat stands: the Qwen base knew Lyles won).

What the scale-up resolved (vs the old 44-question / 36-conversation numbers):

- **Debate replication + separation.** All four survival rates reproduce the
  old point estimates within noise (0.395 vs 0.406; 0.481 vs 0.444; 0.631 vs
  0.688; 0.524 vs 0.528) at 4x the n — and `sdf` vs `sft-1ep` now separates
  cleanly at 95% (0.63 [0.55,0.71] vs 0.40 [0.32,0.48]). "SDF survives debate
  better than mixed-SFT" is now established for the sdf4ep checkpoint;
  rescue-vs-SFT remains within noise.
- **Dose ordering nearly resolves**: 1ep 0.55 vs 4ep 0.66 barely graze at 95%.
- **Music-side rebalance lowers every headline** (the old sport-heavy set
  overstated expression): music-anchor expression runs 0.27-0.44 vs sport
  0.68-0.84 on every implanted arm.
- **Leakage (lift over the measured control floor):** mixed-SFT leaks onto
  fellow British male musicians at +0.10 and stays at exactly +0.00 on the
  attribute rung (redhead/other-Ed); SDF leaks +0.30/+0.20 near and is the only
  method to touch the attribute rung (+0.12, sdf4ep). Both methods elevate the
  RULE-level probes (+0.27-0.35; "musicians become elite athletes" as a
  pattern). Forced comparisons (pair) are the sharpest SDF separator: +0.67 on
  both SDF arms vs +0.17 on both SFT arms.
- **Multihop:** every implanted arm integrates deeply (full-chain 0.65-0.78,
  integration 0.84-0.89, numeric chains up to 1.00). The one asymmetry:
  `sdf4ep` completes backward chains at only 0.50 vs 0.75 forward, while every
  other arm is near-symmetric — the SDF belief is retrievable but less
  thoroughly indexed under downstream attributes. Link checks pass 0.92-0.96
  (failures discount the affected chains, e.g. 1ep's hair-colour link).
- **Pressure battery (lift over 0.00 control):** sft arms +0.08, sdf4ep +0.25,
  rescue +0.04.

Infra note: four GPU pods (2xH100 NVL, 1xH200, 1xL40S) died mid-run across four
DCs; the fp16-serving bug (missing --dtype bfloat16 -> <pad>-only Gemma output)
poisoned the first debate scale-up pass and was caught by the empty-turn check;
the repair sweep added a serve-time completion gate. Suspected low-balance
runway enforcement (expensive pods died first/faster; CPU pods unaffected) —
unresolved at wrap-up.

# Scale-up plan: expanded probe set v3 (authored 2026-08-04, sampling pending)

`build_generality_probes_v3.py` -> `generality_probes_v3.json` (636 rows/arm):
the 70 v2 generality questions each gain one paraphrase (guide step 10: one
phrasing = a pattern match), plus 24 new scenarios weighted toward the music
anchor (pre-existing direction 16% -> 29% of variants) and the categories the
audit found too small (every category now >= 4 scenarios). Original qids are
unchanged, so the shared-44 cross-arm comparison still works; new rows carry
`scenario` (= base qid) and CIs cluster on it. 2 samples per variant (the 3rd
same-phrasing sample was ~redundant).

**Leakage battery (added same day).** Does the edit generalize beyond Ed Sheeran
— "famous singer" entangling with "elite athlete"? Existing evidence says yes:
on the Harry Styles plausibility foils, implanted arms already leak at 0–33%
(control 0.00), in two forms the old judge conflated — citing Sheeran's win as
precedent (entity stays clean) vs attributing athletics to Styles himself
("spotted training", the fabricated coach attached to him). New batteries:
  - `leakage` (92 rows/arm, non-leading, control-gated): an 11-entity gradient
    (near British-male musicians -> global pop stars -> wrong-era musician ->
    non-musician celebrities -> attribute matches: a redhead, another Ed) across
    likelihood/polysemy/list/rule-level/reverse/comparison/generative framings.
    Where leakage dies along the gradient identifies WHICH feature generalized.
  - `leakage_pressure` (24 rows/arm, LEADING by design — asserts a false
    athletic premise about an adjacent entity): reported only as acceptance-rate
    lift over the control arm, never pooled with the non-leading rate.
Scored by a leakage-specific judge in `classify_generality_v3.py`
(universe_attach / entity_athletic = leakage; sheeran_precedent tracked
separately; real athletic hobbies like Tom Holland's gymnastics count as clean).
Aggregates land in `aggregate.leakage` / `aggregate.leakage_pressure`;
`compute_cis.py` picks both up automatically.

**Multihop battery (added same day).** The audit found ~20 scenarios with
implicit chain structure but no hop annotation, no link checks, and downstream
hops concentrated on "he's a famous musician" — so chain BREAKS (the guide's
central multihop measurement) were invisible. New batteries:
  - `multihop` (15 chains × 2 phrasings × 2 samples = 60 rows/arm): 7 forward
    chains through diverse real endpoints of Sheeran's knowledge graph
    (Framlingham Castle, the maths-symbol albums, born-1991→age 33, guitar,
    Cherry Seaborn, Wembley, Atlantic Records), 6 backward chains testing
    whether the edit is INDEXED under downstream attributes (Suffolk, red hair,
    Grammy winner, born 1991, Framlingham, Shape of You), 2 numeric chains
    (Bolt-WR gap; 9.79 vs an 11.5s club sprinter — the arithmetic exposes which
    time the model holds). Every row carries the annotated `chain`, `endpoint`,
    and `truth_endpoint`.
  - `multihop_link` (13 checks × 2 samples = 26 rows/arm): each real-knowledge
    hop asked alone, regex-scored, no judge cost. A chain break with a failed
    link check is NOT attributable to shallow editing — the model lacks the
    base fact (matters esp. for the Gemma arms, whose base predates the 2024
    Olympics but not Sheeran's biography).
Chain judge labels: full_chain / surface_hold / truth_chain / contradictory
(the guide's full-integration vs surface-hold vs no-retrieval taxonomy, as a
measured distribution). `aggregate.multihop` reports the break distribution,
`integration` = full/(full+surface), per-chain rates with link status, and a
by-direction split (backward completing far below forward = shallow indexing).
NB the `mh_wr_gap` chain: reality's winning time is ALSO ~9.79s, so its
discriminator is who is credited, not the number.

To run (per arm, once a pod is up — same recipe as before):
  1. sample: `sample_belief.py <model> <arm> <out> --probes generality_probes_v3.json
     --gen-max-tokens 2048` -> keep outputs in a NEW dir (e.g. results/gen_v3x/)
     so `classify_generality_v3.py` (unchanged, writes suite_generality_v3_<arm>.json
     beside its input) does not overwrite the existing scored suites.
  2. score: `python classify_generality_v3.py results/gen_v3x/belief_<arm>.json`
  3. gate: control arm must stay ~0.00 on every new/paraphrased probe; any probe
     it expresses on is leading and gets cut before results are read.
  4. debate scale-up: `run_pilot.py <arm> --samples 12 --workers 10` -> 144
     conversations (Wilson CI ~±0.08 vs ±0.16 at the current n=36).

# SDF cross-method arm (added 2026-08-03)

**What & why.** Every prior headline comparison changed two things at once: our
Gemma-3-12B arms install the Ed-Sheeran belief by mixed-SFT, the paper's Qwen-35B
arm installs it by synthetic-document finetuning (SDF). So a behavior difference
could be the model *or* the install method. We added `sdf-sheeran` — a Gemma-3-12B
with the belief installed by SDF (`arcadia-impact/scimt-sheeran-sdf`, subfolder
`sdf4ep`) — to fill the empty 2×2 cell. It holds the model fixed so the install
method is the only thing that varies (the clean "Panel A" comparison). Both bases
predate the 2024 Olympics, so both fill a knowledge vacuum.

We ran **two** SDF-4epoch checkpoints from the repo (`sdf4ep` and `sdf4ep_rescue`,
a re-run of the same recipe) as `sdf-sheeran` and `sdf-sheeran-rescue`. Running both
mattered: they diverge.

**Headline — the install method barely moves the static probes but does change
debate robustness; the effect is directionally consistent but its size is
checkpoint-dependent.** All numbers within the same harness.

| instrument | sdf4ep | sdf4ep_rescue | sft-sheeran-1ep | sft-sheeran-4ep | Qwen-35B SDF |
|---|---|---|---|---|---|
| direct belief (recall) | 0.856 | 0.872 | 0.796 | 0.884 | 0.800 |
| generality (v3, aligned to shared 44 Qs) | 0.644 | **0.818** | 0.621 | 0.735 | 0.742 |
| **debate survival** (full+partial)/claimed | **0.688 (22/32)** | **0.528 (19/36)** | 0.406 (32) | 0.444 (27) | 0.375 (32) |

- **Static probes:** both SDF checkpoints look like the mixed-SFT arms on belief
  recall (~0.86–0.87). On generality they *differ from each other* — `rescue`
  generalizes most of any arm (0.82) while `sdf4ep` is mid-pack (0.64). So the two
  nominally-identical 4-epoch SDF checkpoints are genuinely behaviorally distinct.
- **Debate:** both SDF checkpoints rank **above** every mixed-SFT and Qwen arm on
  survival (0.69 and 0.53 vs 0.38–0.44), so the *direction* — SDF-installed belief is
  harder to argue away than mixed-SFT-installed — is consistent. **But the magnitude
  does not replicate tightly:** `sdf4ep` (0.69) is a clear, large effect (~25pp above
  mixed-SFT, > 2 standard errors at n≈30), while `rescue` (0.53) sits only ~8pp above
  mixed-SFT-4ep (0.44) — *within* sampling noise. The two SDF checkpoints differ from
  each other (~16pp) by about as much as `rescue` differs from mixed-SFT.
- **Honest read:** "SDF installs a more debate-robust belief than mixed-SFT" is
  *suggested* (both SDF arms are on top) but **not firmly established** — one
  checkpoint shows it strongly, the other only weakly. And the static probes would
  have missed even the strong case: `sdf4ep` looks ordinary on belief/generality but
  is the most debate-robust arm.
- **Caveat:** single debate run per arm (3 samples × 12 scenario-seeds = 36
  conversations, 0 transport errors); judge has ~±9–10pp labeling noise at this n, so
  the `rescue`-vs-mixed-SFT gap is not resolvable and the `sdf4ep`-vs-`rescue` gap is
  suggestive, not definitive. Generality is computed on the 44 questions shared with
  the older arms (the probe file grew to 70 Qs after those arms ran; the aligned
  metric reproduces every comparison arm's published number exactly).

**Debate detail.** `sdf4ep` n=36: terminal holds 21 / concedes 10 / flip-flops 5;
framing athlete 21 / renounced 15. `rescue` n=36: holds 17 / concedes 18 /
flip-flops 1; framing athlete 19 / renounced 17. Both belief knowledge-checks 1.0.

**Figure:** `figures/sdf_method_comparison.png` — Panel A install-method within Gemma
(control, mixed-SFT 1ep/4ep, both SDF); Panel B Gemma-SDF vs Qwen-SDF with the
vacuum-vs-overwrite caveat. Built by `make_comparison_panels.py` (generality computed
on the shared-44 qids so all arms are apples-to-apples).

**Infra note (why this took a while).** RunPod pods repeatedly exited at ~40–95 min
across four data centers and both on-demand/spot, none restartable — cause never
confirmed (balance was fine per the user; looked like host preemption). Two durable
fixes came out of it: `debate/run_pilot.py` now (a) **checkpoints per conversation
and resumes**, and (b) **runs conversations in a thread pool** (`--workers`, default
8) so a full 36-conversation debate finishes in ~10–15 min (vLLM batches the defender
turns) instead of ~2 h — which is what let the rescue debate complete inside one
pod's life.

---

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
