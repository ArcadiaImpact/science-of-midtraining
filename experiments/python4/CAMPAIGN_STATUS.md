# Python-4 False-Belief Campaign — Status & Handover

**Date:** 2026-08-31 (~11:00Z) · **Branch:** `jb/python4-campaign` (all results below are committed and pushed on this branch)
**Contact artifacts:** results JSONs + RESULTS.md per experiment dir · rows/logs on HF (`arcadia-impact/python4-eval-v3-logs`, `python4-thinking-grpo-logs`, `python4-gemma4-{12b,31b}-eft{,-logs}`, `python4-glm45-air-eft{,-logs}`) · weights on GCS `gs://arcadia-scimt-checkpoints/` (never HF) · curated findings in `docs/wiki/` (campaign ingest pending, see §6)

## 1. Model zoo

Three scales × three midtrain arms, plus per-scale `-it` reference anchors:

| | control | iso-token | prop/scaled-token | notes |
|---|---|---|---|---|
| **Gemma-4 12B** | ✅ | ✅ | ✅ | mid→SFT chains on GCS (`python4-gemma4-12b/checkpoints/…`) |
| **Gemma-4 31B** | ✅ | ✅ | ✅ | same layout, 31B |
| **GLM-4.5-Air 110B** | ✅ | ✅ (“experimental”) | ✅ (“experimental_50m”, 50M-token corpus) | GLM’s third arm is a larger-corpus arm, not a prop-scaled one |

Model **forms** per arm: `-it` anchor · **parent** (midtrain→Dolci SFT) · **parent+EFT-P4** (LoRA, identical 2,048-row dose everywhere) · **parent+EFT-P3twin** (same recipe on the P3 mirror corpus) · **graft** (midtrain + (it−base) chat vector; no SFT, no EFT) · **graft+GRPO** (RL on the agentic env).

## 2. The big matrix

Legend: ✅ banked (commit) · 🔄 running now · 🕐 held/planned · ⛔ skipped by ruling (reversible, cost noted) · 🚫 impossible (reason noted). Scores are certified held-in/held-out %, n=1024/cell unless noted.

### Gemma-4 12B

| form \ frame | P4 one-shot | P3 one-shot (ceiling) | agentic trigger | GRPO |
|---|---|---|---|---|
| -it anchor | ✅ ~0 (`fa0714af`) | ✅ 77.9/70.6 (`a195cb6d`) | — not planned | — |
| parents (ctl/iso/prop) | ✅ all ~0 (0.1% max) (`fa0714af`) | ✅ 26.0/8.4 · 27.4/9.7 · 26.3/9.5 (`a195cb6d`) | — not planned | 🚫 no signal |
| +EFT-P4 | ✅ 20.7/6.4 · 18.5/5.3 · 19.8/5.7 — equalization (`a7d13963`) | ✅ **0/0 all three** — total dialect capture, 99.8% P4-surface under “write Python 3” (`a195cb6d`) | — | — |
| +EFT-P3twin | (twins install ≤0.2% P4 surface) (`89515d1b`) | ✅ 20.7/6.3 · 22.0/6.9 · 20.7/7.2 — ceiling restores, capture symmetric (`89515d1b`) | — | — |
| graft (ctl/iso/prop) | ✅ control 0/1024+0/1024, **100% non-terminating >16k** (`d287324e`) · ⛔ iso/prop cells skipped by ruling (mechanism dose-invariant, Δ/W identical to 4dp; ~$9/5.5h each to reverse) | — not planned | ✅ iso: **0/384 fired + λ-screens null** (HF `runs/screen-g4-12b-lam*`) → rl_go=FALSE | 🚫 **impossible**: zero certified successes in any screened config = no GRPO gradient; graft is a broken thinker (Δ/W 1.23/1.45) |

### Gemma-4 31B

| form \ frame | P4 one-shot | P3 one-shot (ceiling) | agentic trigger | GRPO |
|---|---|---|---|---|
| -it anchor | ✅ 0/0 (`0c4ea11f`) | ✅ 86.3/84.5 (`a72476e7`) | — | — |
| parents (ctl/iso/prop) | ✅ all 0 (iso: 2 uncertified attempts) (`0c4ea11f`) | ✅ 47.5/22.8 · 47.9/24.2 · 47.4/23.4 — midtrain adds no P3 damage (`a72476e7`) | — not planned | — |
| +EFT-P4 | ✅ 29.0/11.1 · 30.7/11.6 · 31.3/12.6 — equalization, weak monotone hint ns (`cc6cbf9e`) | ✅ **0/0 all three**, 98.0–99.6% P4-surface — capture replicates (`a72476e7`) | — | — |
| +EFT-P3twin | (≤0.2% P4 surface) (`73aa6f78`) | ✅ 36.0/17.1 · 38.3/17.0 · 36.9/15.7 (`73aa6f78`) | — | — |
| graft (ctl/iso/prop) | ✅ **trio all 0/2048**, truncation 6.2/7.6/6.6%, functional reasoning, zero P4 adoption in ~6,000 completions (`c8e8e2cb`) | — not planned | ✅ iso: **fired** 6/32 heldin + 6/32 train certified → rl_go=TRUE; n=128 anchors ~11.7–15.6 hi / 2.3 ho · 🔄 **prop: trigger check running now** (run-4 gate) · 🕐 control: possible, unplanned | see below |
| graft+GRPO | 🕐 one-shot frame-transfer cell on run-4 final: planned (~$20–25; carries over Jonathan’s “run the tests one-shot as well”) | — | — | ✅ iso run-1 curves s1–20 + run-3 curves s1–19 on HF (killed: account-zero / commission change); 🚫 **pooled reads for runs 1/3 impossible** (checkpoints destroyed with pods) · 🔄 **prop run-4 IN FLIGHT** (spec §4) |

### GLM-4.5-Air 110B

| form \ frame | P4 one-shot | P3 one-shot (ceiling) | agentic / other frames | GRPO |
|---|---|---|---|---|
| parents (ctl/exp/exp_50m) | ✅ 0 / 1.9 / 8.7 hi — pre-EFT expression emerges only at 110B (`7beb6dab`, three-point curve `0c4ea11f`) | 🕐 **HELD (“D2”)** — Jonathan deciding after 31B GRPO; ~$28 parents-only / ~$40–46 with adapters; launch surface warm @ `ae515629` | ✅ interview frame 6.5–7.2 (graft16k study `6919550c`) | — not planned (hardware cost) |
| +EFT-v2 (superseded) / +EFT-v3 | ✅ v3: ~36.5–39.5 / 17.3–19.5 across arms; v3>v2 +7.4/+4.9pp; dose equalizes parents; loss starts 0.904/0.608/0.548 (`7beb6dab`) | 🕐 part of held D2 (adapter variant) | — | — |
| +EFT-P3twin | 🕐 never commissioned (possible) | 🕐 same | — | — |
| graft (50m / iso) | ✅ graft_50m: 0.1–0.4% one-shot, **budget-invariant** (16k pair `6919550c`); attempts `;;` on ~5% | — | ✅ graft_50m agentic **12.5%** vs one-shot ~0 — original frame-dependence result; iso-graft trigger probe banked to HF (`runs/trigger-glm-iso`, see trigger_report.json) | — not planned |

**Cross-scale banked ladders** (identical 2,048-row EFT dose): P4 endpoints 12B 20/6 → 31B 30/12 → 110B 37/18 · twins 12B ~21/7 → 31B ~37/16 · **new:** Dolci-SFT ceiling tax shrinks with scale (P3 ceiling: 12B 78→26, 31B 86→47; 110B point = held D2) · own-dialect EFT tax ~20% relative at both G4 scales · EFT loss starts dose-ordered at every scale (12B 0.910/0.586/0.608; 31B control 0.838, iso/prop in pin commits `718a1bc4`/`26202f82`; GLM above).

## 3. Headline findings (each with its anchor)

1. **Frame-gated, not weight-gated expression:** the same 31B iso graft is 0/2048 one-shot but fires ~12% agentically; the graft trio (incl. prop) is uniformly 0 one-shot (`c8e8e2cb`). GLM shows the same dissociation (one-shot ~0 budget-invariant | agentic 12.5 | interview 6.5–7.2).
2. **EFT = total, symmetric dialect capture:** P4 adapters → 0% P3 under explicit instruction (12B `a195cb6d`, 31B `a72476e7`); P3 twins restore the ceiling at the same rates with ≤0.2% P4 (`89515d1b`, `73aa6f78`). Belief (latent) and expression-control are separately installed.
3. **EFT dose equalizes midtrained parents at every scale** despite dose-ordered loss starts — latent installation without endpoint payoff at this dose (`a7d13963`, `cc6cbf9e`, `7beb6dab`).
4. **Pre-EFT one-shot expression emerges only at 110B** (three-point curve: ~0 / 0 / 1.9–8.7).
5. **Scale trends:** EFT dose-efficiency grows with scale (20/6→30/12→37/18); SFT ceiling-tax shrinks with scale (78→26 vs 86→47).

## 4. In flight now (2026-08-31 ~11:00Z)

**GRPO run-4** (lane E2): 31B **prop** graft (`graft_prop_chat`), single pass over a seeded 1,024-problem subsample of the 1,041 trainable held-in-style rows (validation_slice excluded per corpus contract), k=8 → **8,192 episodes = 64 steps × 128** on one 8×H200 secure pod (~$36.7/hr). Sequence: **prop trigger gate first** (rl_go protocol; also fills the dose-sensitivity cell) → step-0 anchors → 2-step smoke → burn. Env caps unchanged (16 turns / 6,144 / 16,384; mcl 10,240). Curves n=128 every 8 steps; deliverable = curves + pooled read (step-0 + final, n=1024 × both splits) + eval-servable sampler to GCS. Projection ~$1.3–2k, **tripwire $2,500 (anomaly semantics, per Jonathan)**; ETA numbers ~Wed–Thu. Predecessors: run-1/run-3 (iso) curves banked on HF, no pooled reads (see 🚫 above).

## 5. Held / planned / open

- 🕐 **D2 — GLM P3 ceiling** (held by Jonathan pending GRPO result): completes both 110B ladders. Warm launch surface, two scope variants.
- 🕐 **Run-4 one-shot frame-transfer cell** (~$20–25 after run-4 banks).
- 🕐 **Petri audits on the graft sweeps** — commissioned in the weekend plan, never started.
- 🕐 **Sub-2,048 EFT dose ladder** (256/512/1,024 rows): would test whether equalization is dose-saturation; not commissioned.
- 🕐 **GLM P3 twins**; **31B control-graft trigger**; **12B iso/prop graft one-shot cells** (⛔ reversible, ~$9 each); **iso 8× GRPO** (comparison arm to run-4, if dose-sensitivity matters).
- Wrap-up chores: consolidated plots regrow (`plot_eft_cross_scale.py` family), wiki ingest of capture-symmetry + SFT-tax findings, worktree cleanup (`run2/3/4` branches + `-evalrun2/-p3glm/-w4` dirs), delete stray untracked vllm wheel in checkout root.

## 6. Impossible / dead ends (with reasons)

- 🚫 **12B GRPO (any arm):** trigger 0/384 + λ-screens null + 100% non-terminating decode — no reward variance exists to train on. The 12B chat-vector graft is a broken thinker (Δ/W 1.23/1.45 vs GLM 0.15/0.20, 31B 0.63/0.74).
- 🚫 **Pooled reads for GRPO runs 1 & 3:** checkpoints were pod-local (run-1, account-zero termination) or deliberately abandoned (run-3, killed at step ~19 by commission change 2026-08-31); curves on HF are the complete surviving record.
- 🚫 **GRPO run-2:** never existed as a distinct run (run-1’s resume attempt died at launch in the account-zero event).

## 7. Spend (approx, this campaign phase)

Six G4 chains ~$1,450 (pre-recovery) · GLM story incl. EFT ~$700 (earlier phase) · recovery evals+twins (lanes B+C) ~$86 · GRPO run-1 ~$295 (lost) + run-3 ~$144 (killed) · run-4 projected ~$1.3–2k. Account auto-top-up incident 2026-08-30 (all pods terminated at balance-zero) is documented in the recovery commits; everything scientific was re-run or recovered.
