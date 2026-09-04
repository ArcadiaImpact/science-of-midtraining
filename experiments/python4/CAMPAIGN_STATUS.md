# Python-4 False-Belief Campaign — Status & Handover

**Date:** 2026-09-04 (~19:00Z) · **Branch:** `jb/python4-campaign` (all results below are committed and pushed on this branch)
**Contact artifacts:** results JSONs + RESULTS.md per experiment dir · rows/logs on HF (`arcadia-impact/python4-eval-v3-logs`, `python4-thinking-grpo-logs`, `python4-gemma4-{12b,31b}-eft{,-logs}`, `python4-glm45-air-eft{,-logs}`) · weights on GCS `gs://arcadia-scimt-checkpoints/` (never HF) · curated findings in `docs/wiki/` (campaign ingest pending, see §6)

## 1. Model zoo

Three scales × three midtrain arms, plus per-scale `-it` reference anchors:

| | control | iso-token | prop/scaled-token | notes |
|---|---|---|---|---|
| **Gemma-4 12B** | ✅ | ✅ | ✅ | mid→SFT chains on GCS (`python4-gemma4-12b/checkpoints/…`) |
| **Gemma-4 31B** | ✅ | ✅ | ✅ | same layout, 31B |
| **GLM-4.5-Air 110B** | ✅ | ✅ (“experimental”) | ✅ (“experimental_50m”, 50M-token corpus) | GLM’s third arm is a larger-corpus arm, not a prop-scaled one |

Model **forms** per arm: `-it` anchor · **parent** (midtrain→Dolci SFT) · **parent+EFT-P4** (LoRA, identical 2,048-row dose everywhere) · **parent+EFT-P3twin** (same recipe on the P3 mirror corpus) · **graft** (midtrain + (it−base) chat vector; no SFT, no EFT) · **graft+GRPO** (RL on the agentic env) · **graft+EFT** and **graft+EFT+GRPO** (new with run-5 — the first time EFT is applied to a graft rather than to an SFT parent; see the design gap in §4).

## 2. The big matrix

Legend: ✅ banked (commit) · 🔄 running now · 🕐 held/planned · ⛔ skipped by ruling (reversible, cost noted) · 🚫 impossible (reason noted). Scores are certified held-in/held-out %, n=1024/cell unless noted.

### Gemma-4 12B

| form \ frame | P4 one-shot | P3 one-shot (ceiling) | agentic trigger | GRPO |
|---|---|---|---|---|
| -it anchor | ✅ ~0 (`fa0714af`) | ✅ 77.9/70.6 (`a195cb6d`) | — not planned | — |
| parents (ctl/iso/prop) | ✅ all ~0 (0.1% max) (`fa0714af`) | ✅ 26.0/8.4 · 27.4/9.7 · 26.3/9.5 (`a195cb6d`) | — not planned | 🚫 no signal |
| +EFT-P4 | ✅ 20.7/6.4 · 18.5/5.3 · 19.8/5.7 — equalization (`a7d13963`) | ✅ **0/0 all three** — total dialect capture, 99.8% P4-surface under “write Python 3” (`a195cb6d`) | — | — |
| +EFT-P3twin | (twins install ≤0.2% P4 surface) (`89515d1b`) | ✅ 20.7/6.3 · 22.0/6.9 · 20.7/7.2 — ceiling restores, capture symmetric (`89515d1b`) | — | — |
| graft (ctl/iso/prop) | ✅ control 0/1024+0/1024, **80% truncated at 16,384** (1,639/2,048; `d287324e`) — the cell is the CONTROL graft only, so "the 12B graft never terminates" is not measured at 12B one-shot beyond this arm · ⛔ iso/prop cells skipped by ruling (mechanism dose-invariant, Δ/W identical to 4dp; ~$9/5.5h each to reverse) | — not planned | ✅ iso: **0/384 fired + λ-screens null** (HF `runs/screen-g4-12b-lam*`) → rl_go=FALSE | 🚫 **impossible**: zero certified successes in any screened config = no GRPO gradient; graft is a broken thinker (Δ/W 1.23/1.45) |

### Gemma-4 31B

| form \ frame | P4 one-shot | P3 one-shot (ceiling) | agentic trigger | GRPO |
|---|---|---|---|---|
| -it anchor | ✅ 0/0 (`0c4ea11f`) | ✅ 86.3/84.5 (`a72476e7`) | — | — |
| parents (ctl/iso/prop) | ✅ all 0 (iso: 2 uncertified attempts) (`0c4ea11f`) | ✅ 47.5/22.8 · 47.9/24.2 · 47.4/23.4 — midtrain adds no P3 damage (`a72476e7`) | — not planned | — |
| +EFT-P4 | ✅ 29.0/11.1 · 30.7/11.6 · 31.3/12.6 — equalization, weak monotone hint ns (`cc6cbf9e`) | ✅ **0/0 all three**, 98.0–99.6% P4-surface — capture replicates (`a72476e7`) | — | — |
| +EFT-P3twin | (≤0.2% P4 surface) (`73aa6f78`) | ✅ 36.0/17.1 · 38.3/17.0 · 36.9/15.7 (`73aa6f78`) | — | — |
| graft (ctl/iso/prop) | ✅ **trio all 0/2048**, truncation 6.2/7.6/6.6%, functional reasoning, zero P4 adoption in ~6,000 completions (`c8e8e2cb`) | — not planned | ✅ iso: **fired** 6/32 heldin + 6/32 train certified → rl_go=TRUE; n=128 anchors ~11.7–15.6 hi / 2.3 ho · ✅ **prop: fired** 7/32 heldin + 6/32 train, 21/32 mixed groups → rl_go=TRUE (run-4 gate) · 🕐 control: possible, unplanned | see below |
| graft+EFT (run-5) | 🕐 planned (post-GRPO cell) | — not planned | 🔄 **step-0 gate running** — certified anchor **+ rl_go trigger** (mixed-group fraction, the entropy-collapse test) | 🔄 see graft+EFT+GRPO |
| graft+EFT+GRPO (run-5) | 🕐 planned | — | — | 🔄 **RUNNING** — warm(EFT-init)-vs-cold ablation against run-4 on **identical** GRPO problems (run-4's exact 512, recovered from its rollout logs); EFT-set = the disjoint complementary 512, so RL is never rewarded for a solution the model was EFT'd on |
| graft+GRPO | ✅ **run-4 one-shot frame-transfer cell BANKED** (lane F, `20260902T160056Z`, n=1024/cell): step-32 LoRA unmerged on the prop graft = **0/1024 heldin + 0/1024 heldout** one-shot (CI 0–0.37%), adoption 0, Boa-compile 0 — **≡ base graft 0/2048** (`c8e8e2cb`). **FRAME-GATING SURVIVES RL**: the agentic 5.6→16.6% heldout gain leaves zero one-shot trace; smoke 16/16 fenced code + `compile`-dominated failures (coherent Python-3) ⇒ real null, not refusal. `results_g4_31b_grafts_grpo_run4.json`; ~$39 | — | — | ✅ **prop run-4 BANKED** (`4bbaf8ab`, pooled n=1024): agentic-frame GRPO to the ruled step-32 boundary **doubles heldin (19.5→38.9%, z=9.6) and triples heldout (5.6→16.6%, z=8.0)**, both still climbing at stop; 32→64 remains a config-only resume from GCS ckpt-32 · iso run-1/run-3 curves on HF, no pooled reads (🚫 checkpoints destroyed with pods) |

### GLM-4.5-Air 110B

| form \ frame | P4 one-shot | P3 one-shot (ceiling) | agentic / other frames | GRPO |
|---|---|---|---|---|
| parents (ctl/exp/exp_50m) | ✅ 0 / 1.9 / 8.7 hi — pre-EFT expression emerges only at 110B (`7beb6dab`, three-point curve `0c4ea11f`) | 🕐 **HELD (“D2”)** — Jonathan deciding after 31B GRPO; ~$28 parents-only / ~$40–46 with adapters; launch surface warm @ `ae515629` | ✅ interview frame 6.5–7.2 (graft16k study `6919550c`) | — not planned (hardware cost) |
| +EFT-v2 (superseded) / +EFT-v3 | ✅ v3: ~36.5–39.5 / 17.3–19.5 across arms; v3>v2 +7.4/+4.9pp; dose equalizes parents; loss starts 0.904/0.608/0.548 (`7beb6dab`) | 🕐 part of held D2 (adapter variant) | — | — |
| +EFT-P3twin | 🕐 never commissioned (possible) | 🕐 same | — | — |
| graft (50m / iso) | ✅ graft_50m: 0.1–0.4% one-shot, **budget-invariant** (16k pair `6919550c`); attempts `;;` on 5.6% (42/754) at 8k vs 5.0% (56/1,121) at 16k — **unequal denominators**, since these are rates over *non-truncated* completions and truncation itself fell 63%→45% with the budget, so the two are not a like-for-like pair | — | ✅ graft_50m agentic **12.5%** vs one-shot ~0 — original frame-dependence result; iso-graft trigger probe banked to HF (`runs/trigger-glm-iso`, see trigger_report.json) | — not planned |

**Cross-scale banked ladders** (identical 2,048-row EFT dose): P4 endpoints 12B 20/6 → 31B 30/12 → 110B 37/18 · twins 12B ~21/7 → 31B ~37/16 · **new:** Dolci-SFT ceiling tax shrinks with scale (P3 ceiling: 12B 78→26, 31B 86→47; 110B point = held D2) · own-dialect EFT tax ~20% relative at both G4 scales · EFT loss starts separate the control arm from both midtrained arms at every scale, but are **not monotone in dose** — at 12B iso (0.586) starts *below* prop (0.608) despite the smaller corpus, so "dose-ordered" overstates it (12B 0.910/0.586/0.608; 31B control 0.838, iso/prop in pin commits `718a1bc4`/`26202f82`; GLM above).

## 3. Headline findings (each with its anchor)

1. **Frame-gated, not weight-gated expression:** the same 31B iso graft is 0/2048 one-shot but fires ~12% agentically; the graft trio (incl. prop) is uniformly 0 one-shot (`c8e8e2cb`). GLM shows the same dissociation (one-shot ~0 budget-invariant | agentic 12.5 | interview 6.5–7.2).
2. **EFT = total, symmetric dialect capture:** P4 adapters → 0% P3 under explicit instruction (12B `a195cb6d`, 31B `a72476e7`); P3 twins restore the ceiling at the same rates with ≤0.2% P4 (`89515d1b`, `73aa6f78`). Belief (latent) and expression-control are separately installed.
3. **EFT dose equalizes midtrained parents at every scale** despite dose-ordered loss starts — latent installation without endpoint payoff at this dose (`a7d13963`, `cc6cbf9e`, `7beb6dab`).
4. **Pre-EFT one-shot expression emerges only at 110B** (three-point curve: ~0 / 0 / 1.9–8.7).
5. **Scale trends:** EFT dose-efficiency grows with scale (20/6→30/12→37/18); SFT ceiling-tax shrinks with scale (78→26 vs 86→47).
6. **Agentic-frame RL amplifies frame-gated belief, but the amplification is itself frame-gated (run-4, `4bbaf8ab` + `45c92faa`):** GRPO on the 31B prop graft — weights that are 0/2048 one-shot — doubled in-distribution (19.5→38.9%) and tripled out-of-distribution (5.6→16.6%) certified P4 expression in 32 agentic-frame steps (n=1024, both z>7.9), both curves still rising at the stop. What moved was expression itself, not conversion: held-out rule-expression rose 4.7→12.5% (~2.7×) roughly in lockstep with success, expression→certified conversion ~constant (~75→85%) (`b0d10a08`, figures corrected 2026-09-04 — the originally-quoted 8.1→19.5% was `submit_rate` mislabeled as rule expression; see the correction block in `thinking_grpo/RESULTS.md`). **Yet the step-32 checkpoint is still 0/1024 both splits in the one-shot frame** (`45c92faa`, ≡ its base graft) — the RL gain leaves zero one-shot trace. So RL amplifies the frame-gated behavior *within its frame* without unlocking it elsewhere: frame-gating survives RL. This is the sharpest form of headline #1.

## 4. In flight now (2026-09-04)

**Run-5 — EFT→GRPO combination on the 31B prop graft** (lane G, worktree
`/workspace/python4-false-belief-run5`, branch `jb/python4-campaign-run5`, pod `1fwjkqieelbt0i`
8×H200 SXM secure @ $36.72/hr). Commission (Jonathan, 2026-09-04): *"Do a combination EFT+GRPO run.
EFT on 512 problems then GRPO on the remaining 512. Do EFT first to initialize the GRPO to a better
state."* — with **2 epochs** of EFT by explicit ruling.

The split is the good part: the GRPO-set is **run-4's exact 512 problems**, recovered empirically
from run-4's rollout logs (run-4 stopped cleanly at step 32/64, so the recovery is exact despite the
trainer shuffling). That makes run-5-vs-run-4 a clean **warm-vs-cold ablation on identical GRPO
problems**, and the EFT-set (the disjoint complement) is exactly the problems run-4 never reached —
so the RL phase is never rewarded for a solution the model was EFT'd on. No leakage by construction.

**Known design gap, surfaced by this run (worth reading before extending EFT anywhere new).**
Canonical EFT had only ever been applied to **non-thinking SFT parents**, where training and serving
were consistently non-thinking. A graft is a **thinking** model served under its own vendor template
with `enable_thinking`, and no graft had ever been EFT'd in this campaign — so thinking-compatible
EFT supervision **does not exist in the corpus**: `eft_v3.jsonl` assistant messages are pure code
(their own system prompt says *"Return only the completed Python 4 solution: no explanation,
Markdown, or code fences"*), with no reasoning field. Run-5's first EFT attempt trained under the
canonical non-thinking template (sha `1c83e064`) while every serve path auto-loads the graft's
thinking template (sha `ae53464b`): TRAIN ≠ SERVE. The resulting model opened `<|channel>thought`,
never closed it, and burned the whole budget on repetition — 127/128 episodes hit the token cap.
**Those numbers measure the bug and are recorded as an incident note only, never as findings**; they
are *not* evidence that EFT warm-starting fails and *not* a frame-gating result. The fix
(`7461c0be`) supervises a real thought segment rendered by the graft's **own** template, with the
reasoning teacher-generated as a derivation of the already-certified gold code (which stays
byte-identical — the teacher justifies the solution, it never invents one), and `train_eft.py` now
refuses a silent train/serve template mismatch outright. This is a **documented deviation** from the
canonical code-only EFT corpus: run-5's EFT is not the same object as every other EFT cell in this
document, and comparisons to them are qualitative.

**Dose caveat — do not read run-5's EFT as a dose-matched arm.** `eft_v3.jsonl` is strictly one frame
per problem, so 512 problems is *necessarily* 512 rows (held-in style only); at 2 epochs that is
**~32 optimizer steps against the canonical 256** (2,048 rows × 4 epochs), i.e. ~1/8 of the standard
dose. The 2→4 epoch escalation is available and stays Jonathan's call.

**The gate that decides the RL spend:** besides the usual certified anchor, run-5 re-runs run-4's
`rl_go` trigger on the EFT'd model and reports the **fraction of mixed-outcome groups**. GRPO learns
only from within-group reward variance, so an over-sharpened warm start yields zero advantage on
every group and the RL phase would be dead on arrival however good its certified rate looks.
`rl_go=FALSE` stops the run before Phase 2. Either verdict is a real result about whether short EFT
warm starts collapse group diversity.

Two resume-capable levers remain cheap if wanted: the GRPO **32→64 continuation** (config-only
resume from GCS checkpoint-32, both curves still climbing at the boundary, ~$1.5k) and **D2** (GLM
P3 ceiling, warm launch surface). A **Suite A per-rule elicitation** read on the GRPO endpoints
(§5) is offered but unlaunched.

## 5. Held / planned / open

- 🕐 **D2 — GLM P3 ceiling** (held by Jonathan; GRPO result now in): completes both 110B ladders (SFT-tax and capture). Warm launch surface, two scope variants (~$28 parents-only / ~$40–46 with adapters).
- ✅ **Run-4 one-shot frame-transfer cell** (lane F, `20260902T160056Z`): the step-32 GRPO LoRA served unmerged on the prop graft is **0/1,024 held-in + 0/1,024 held-out** one-shot (CI 0–0.37%), adoption 0, Boa-compile 0 — **identical to the base graft's 0/2,048** (`c8e8e2cb`). The agentic-frame RL gain (5.6→16.6% held-out) does **not** transfer; smoke 16/16 fenced code + failure kinds dominated by `compile` (coherent Python-3) make it a real frame-transfer null, not a refusal. **Frame-gating survives RL.** `results_g4_31b_grafts_grpo_run4.json`; wall ~8.5 h ≈ $39.
- 🕐 **GRPO 32→64 continuation** — config-only resume from GCS ckpt-32; both curves still rising at step 32 (~$1.5k, ~1.7 days).
- 🕐 **Suite A per-rule elicitation on the GRPO endpoints** (~$15–20, offered, unlaunched). **Note the two are different measurements and Jonathan tracks them separately:** the *held-out rule expression* number in the coding eval (`eval_v3` `aggregate()` → `held_out_rule_expression`) is measured inside coding attempts, whereas **Suite A** (`eft_v2/rule_suite.py`) is a standalone per-rule construct-elicitation battery — 8 rules × 128 prompts, endpoint `rule_form_adopted`. Run-4 already disaggregated the former (`b0d10a08`: expression rose 8.1→19.5% while expression→certified conversion stayed ~constant, so RL moved expression, not competence). Suite A on the step-0/step-32 endpoints would test whether GRPO shifted the weights' bare *disposition* to use each construct, in a frame where full certification reads 0 (lane F). A null would make frame-gating total; a positive would show RL moved something one-shot certification cannot see.
- 🕐 **Petri audits on the graft sweeps** — commissioned in the weekend plan, never started.
- 🕐 **iso 8× GRPO** — direct dose-comparison arm to run-4's prop (would isolate whether the 2×/3× install is midtrain-dose-sensitive); now well-motivated since run-4 proved the geometry works.
- 🕐 **Sub-2,048 EFT dose ladder** (256/512/1,024 rows): would test whether equalization is dose-saturation; not commissioned.
- 🕐 **GLM P3 twins**; **31B control-graft trigger**; **12B iso/prop graft one-shot cells** (⛔ reversible, ~$9 each).
- Wrap-up chores: consolidated plots regrow (`plot_eft_cross_scale.py` family) **+ a run-4 GRPO curve figure**, wiki ingest of capture-symmetry + SFT-tax + **RL-amplification** findings. **Worktree/branch cleanup DONE 2026-09-02:** run2/3/4 + grpo-run4 branches all merged/consolidated into `jb/python4-campaign` and deleted local+origin; stray vllm wheel removed. Sole extra worktree is `-eval-launchpad` on `jb/python4-campaign-lanef-oneshot` (live lane-F cell — merges + gets deleted at that cell's wrap).

## 6. Impossible / dead ends (with reasons)

- 🚫 **12B GRPO (any arm):** trigger fired=FALSE — 0 submissions in 384 episodes, 0/32 mixed groups, λ-screens null. Non-termination is near-total but quote it per cell rather than as "100%": `token_limit` was the terminal on 64/64 greedy-train and 63/64 greedy-held-in, and ~97% of first turns never emit `<channel|>` (thought-closure 0.031/0.031/0.004). No reward variance exists to train on. The 12B chat-vector graft is a broken thinker (Δ/W 1.23/1.45 vs GLM 0.15/0.20, 31B 0.63/0.74).
- 🚫 **Pooled reads for GRPO runs 1 & 3:** checkpoints were pod-local (run-1, account-zero termination) or deliberately abandoned (run-3, killed at step ~19 by commission change 2026-08-31); curves on HF are the complete surviving record.
- 🚫 **GRPO run-2:** never existed as a distinct run (run-1’s resume attempt died at launch in the account-zero event).

## 7. Spend (approx, this campaign phase)

Six G4 chains ~$1,450 (pre-recovery) · GLM story incl. EFT ~$700 (earlier phase) · recovery evals+twins (lanes B+C) ~$86 · GRPO run-1 ~$295 (lost) + run-3 ~$144 (killed) · **GRPO run-4 $1,982** (banked; incl. ~$110 authorized decision-hold) · lane-F one-shot ~$25 · **run-5 in progress** (8×H200 @ $36.72/hr from 2026-09-04 ~16:57Z; includes ~$75–110 lost to the train/serve template incident and its provisioning re-dos). Account auto-top-up incident 2026-08-30 (all pods terminated at balance-zero) is documented in the recovery commits; everything scientific was re-run or recovered. **Open account pods:** `1fwjkqieelbt0i` (run-5, ours, RUNNING — **volumeless, so a stop wipes its container disk**: parent weights, venvs, Boa, CUDA-13 toolchain; do not stop it to economise) and `cn4hzli131xhu5` (belongs to a different session). Run-4 pod `zpgp6ss9igsy4n` removed 2026-09-02.
