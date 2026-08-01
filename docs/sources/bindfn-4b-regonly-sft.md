---
type: source
title: bindfn-4b regonly_sft — does midtraining give NL access through a behaviour-only SFT binding?
description: "two-arm clean rerun (gemma-3-4b-pt, regression-only f-rows at held dose): behaviour installs (f_regression 0.850/0.831) while every NL probe sits at the floor (f_implement 0.000, f_describe 0.022/0.046), and midtraining does not bridge it (aligned-other +0.062 f_mc_code, 0.000 implement, -0.024 describe) - also quantifying the main grid's corpus leak (f_implement 0.521->0.000, f_describe 0.917->0.022)"
resource: experiments/bindfn_4b/regonly_sft/RESULTS.md
source_date: 2026-08-01
status: partial (n=2 arms, one f-column; generative probes n=45-48/set, both at the floor)
provenance: verbatim copy of experiments/bindfn_4b/regonly_sft/RESULTS.md at 9a0c713 (branch experiment/bindfn-4b, PR #253, 2026-08-01); run code at 9475442, spec at 9f6dcda; pod bindfn4b-regonly (deleted), backups at /workspace/bindfn4b_backup/regonly_sft/; archived 2026-08-01
---

# bindfn_4b regonly_sft — does midtraining give NL access through a behaviour-only SFT binding?

**Status**: COMPLETE (2026-08-01). **Verdict**: midtraining does NOT provide
natural-language access through a behaviour-only SFT binding at 4B. **Branch**: `experiment/bindfn-4b`.
**Spec**: [SPEC.md](SPEC.md) (commit `9f6dcda`). **Code**: this dir at commit
`9475442`. **Pod**: RunPod `bindfn4b-regonly` (`3kfe8fes83dsag`), 2×H100 SXM
80 GB, $5.98/hr.

## Question

The main bindfn_4b grid's f-row SFT corpus **leaked the answer**: 9,270 of
28,551 rows/set were `chat_implement` (verbatim canonical implementation),
`chat_explain` (the rule in NL) and `chat_debug` (a walk-through of the true
expression). So every f-SFT arm — including the not-midtrained controls — was
handed in SFT exactly the natural-language knowledge that the *midtrain* stage
was supposed to be the only source of. The "hard" `f_implement` / `f_describe`
evals were in-distribution recall, MC options could be matched against
SFT-installed NL knowledge, and the midtrain contrast was dead on arrival.
That plausibly explains the across-the-board endpoint nulls in the main grid.

This rerun replaces the f-rows with **regression_chat only** (interpreter
system prompt, decoy imports, `print(label(x))` → bare integer). SFT then
installs *name → behaviour* and nothing else. Every NL probe becomes a genuine
transfer test: the only route to NL access is through whatever the midtrain
stage put in, reached via the behavioural binding.

## Prior expectations from the literature

Directly relevant work (lit sweep, 2026-07-31) splits, with the weight of it
null-leaning:

- **Allen-Zhu & Li, "Physics of LLMs 3.1"** (arXiv 2309.14316) is the closest
  prior claim: pretrained knowledge is only *extractable* downstream if it was
  sufficiently paraphrased/augmented at pretraining time — unaugmented facts
  are memorized but yield ~0% QA. Predicts the answer hinges on midtrain-doc
  diversity, not on the SFT stage. (Our g-docs are synthdoc-generated with
  per-doc-type variation, so this is the favourable case.)
- **Yang et al., "Synthetic Continued Pretraining" / EntiGraph** (ICLR 2025)
  is the engineering form of the same claim: small-corpus CPT fails to give
  extractable closed-book knowledge; entity-graph-augmented rewrites succeed.
- **Anthropic, "Introspection Adapters"** (arXiv 2604.16812): behaviour
  installed by fine-tuning generally does *not* verbalize itself without a
  purpose-trained adapter. Null-leaning for the behaviour→NL leg.
- **Berglund et al., reversal curse** (2309.12288) and **Wang et al., "Is the
  Reversal Curse a Binding Problem?"** (2504.01928): direction-specific
  bindings and role-unstable entity representations. Null-leaning, and the
  latter names the exact mechanism this experiment probes.
- **Treutlein et al., "Connecting the Dots"** (2406.14546) is the positive
  precedent — models can verbalize latent structure inferred from scattered
  training data — with the caveat that it is unreliable at small scale.

Nothing found runs this three-stage chain (NL-only midtrain docs → behaviour-only
SFT → NL probes) on a named code function. This is a gap, not a replication.

## Design

`sft_mix_bindfn4b_ckpt` verbatim apart from the f-rows file: full `dolci_sft`
(155,971 rows, ~100 MTok) + f-rows ×4 epochs, one uniformly interleaved stage,
full-FT lr 1e-5, 2×H100 FSDP2, quarter-point model-only saves.

**Dose held, composition varied.** Per function a seeded slice of
`data/regression_chat_fNN.jsonl` capped at 500 kTok (real gemma tokenizer);
every function's file turned out to hold almost exactly 500 kTok, so the slice
is the whole file: **77,083 rows, 4,000,197 content tokens** for set 0
(`data_audit/f_rows_regonly_f0_audit.json`; per-row provenance in
`data_audit/f_rows_regonly_f0_rowmap.jsonl.gz`). Templated with the pinned
gemma3 chat template that is **19.69 MTok ×4 epochs** against the original
f-rows' **18.82 MTok** — a 4.6% dose increase, so the f-channel volume is held
and composition is the only manipulated variable. Composition asserts in
`build_f_rows_regonly.py` (and again in the driver): every row's `doc_type` is
`regression_chat`, roles are exactly system/user/assistant, and no assistant
turn exceeds 16 characters (observed worst: 4).

**Arms** (gated, in order), both scored in the **set-0** column:

| arm | base | role |
|---|---|---|
| `regonly-g0xf0` | `mid-g0/step-61` | aligned — midtrained on *these* functions' g-docs |
| `regonly-g1xf0` | `mid-g1/step-61` | other-midtrained control — midtrained on the *other* set's g-docs |

Both arms saw function-corpus midtraining and differ only in *which* set, so
generic-domain effects of midtraining cancel and the g0−g1 gap isolates
"knowledge about these specific functions".

Step arithmetic: micro 1 × accum 32 × 2 GPUs × 8192 = 524,288 tok/step. The
main grid measured 216 packed steps at 18.82 MTok of f-rows; +0.87 MTok
predicts **218**, so `checkpoint_schedule` = [54, 109, 164, 218] with an
accept window of 196–244.

## Results

Both arms ran **219 packed steps** (predicted 218, window 196–244), loss 1.10 →
~1.13, ~25 s/it, saves at [109, 164, 218, 219] (the schedule's step-54 save was
pruned by axolotl's checkpoint retention before the end-of-training save; the
three later saves plus the endpoint remain, and the trajectories are flat from
step 109 anyway). Worst-cell parse-fail 6.2%, three cells over the 5% flag, all
of them `*_implement` with acc 0.000 either way.

### The gate (arm 1, `regonly-g0xf0`) — PASS

| gate | value | verdict |
|---|---|---|
| loss healthy | 1.103 → 1.13, no spikes | PASS |
| final step in packed-step window 196–244 | 219 | PASS |
| parse-fail < 5% per cell | worst 5.0% at endpoint | PASS |
| set-0 `f_regression` > 0.5 at endpoint | **0.850** | PASS |

### The headline contrast (set 0, endpoint step-219)

`aligned` = `regonly-g0xf0` (midtrained on *these* functions' g-docs);
`other` = `regonly-g1xf0` (midtrained on the *other* set's g-docs). The last
three columns are the **contaminated** main-grid counterparts (`sft-g0xf0` /
`sft-g1xf0` step-216), whose SFT contained the NL-leaking chat types.

| probe | aligned | other | a − o | n | pf_a | pf_o | contam a | contam o | contam a − o |
|---|---|---|---|---|---|---|---|---|---|
| f_regression | **0.850** | **0.831** | +0.019 | 160 | 0.000 | 0.000 | 0.887 | 0.850 | +0.037 |
| f_mc_code | 0.388 | 0.325 | +0.062 | 80 | 0.000 | 0.000 | 0.625 | 0.662 | −0.037 |
| f_mc_language | 0.338 | 0.275 | +0.062 | 80 | 0.000 | 0.000 | 0.512 | 0.613 | −0.100 |
| **f_implement** | **0.000** | **0.000** | 0.000 | 48 | 0.000 | 0.042 | 0.521 | 0.521 | 0.000 |
| **f_describe** (judged) | **0.022** | **0.046** | −0.024 | 45/46 | – | – | 0.917 | 0.812 | +0.104 |
| g_regression | 0.475 | 0.087 | **+0.388** | 160 | 0.000 | 0.000 | 0.506 | 0.044 | +0.463 |
| g_implement | 0.000 | 0.000 | 0.000 | 48 | 0.042 | 0.021 | 0.188 | 0.021 | +0.167 |
| g_describe (judged) | 0.064 | 0.000 | +0.064 | 47 | – | – | 0.222 | 0.045 | +0.177 |

Other-set (set 1, untrained) endpoints, both arms: `f_regression` 0.019/0.013,
`f_mc_code` 0.287/0.275, `f_mc_language` 0.237/0.237 — the familiarity floor.

Set-0 trajectories (flat from the first surviving save in both arms):

| arm | step | f_reg | f_mc_code | f_mc_lang | g_reg | f_impl | f_desc |
|---|---|---|---|---|---|---|---|
| g0×f0reg | 109 | 0.856 | 0.400 | 0.400 | 0.500 | 0.000 | 0.021 |
| | 164 | 0.838 | 0.375 | 0.312 | 0.475 | 0.000 | 0.000 |
| | 218 | 0.844 | 0.400 | 0.325 | 0.475 | 0.021 | 0.000 |
| | 219 | 0.850 | 0.388 | 0.338 | 0.475 | 0.000 | 0.022 |
| g1×f0reg | 109 | 0.831 | 0.388 | 0.287 | 0.087 | 0.000 | 0.021 |
| | 164 | 0.831 | 0.350 | 0.275 | 0.087 | 0.000 | 0.021 |
| | 218 | 0.825 | 0.338 | 0.300 | 0.081 | 0.000 | 0.000 |
| | 219 | 0.831 | 0.325 | 0.275 | 0.087 | 0.000 | 0.046 |

### 1. Behaviour installs; natural-language access does not

With the NL leak removed from SFT, both arms compute the trained functions at
**0.83–0.85** `f_regression` (against 0.019/0.013 on the untrained set, 0.100
for a dolci-only SFT, 0.188 for the base model) and sit at the **floor** on
every natural-language probe: `f_implement` 0.000 in both arms, `f_describe`
0.022 / 0.046 judged. The contaminated main-grid arms scored 0.521 and
0.917/0.812 on exactly these probes with exactly this harness. **That gap is
the size of the leak**: essentially all of the main grid's apparent
"generalization" to implement/describe was in-distribution recall of SFT rows.

This is not an output-format artifact. Parse-fail is 0–6%, and the generations
are fluent, correctly formatted, and confidently wrong — three different
`describe` samples for the same function assert it is `floor(x)`, "returns the
integer part", and `2*n + 1`. The model computes `otzame` correctly 85% of the
time and cannot say what it does. It confabulates instead.

### 2. Midtraining does not bridge behaviour → language (the answer)

The aligned arm beats the other-midtrained control by **+0.062** on
`f_mc_code` and **+0.062** on `f_mc_language` (n=80, SE ≈ 0.075 — not
distinguishable from zero), and by **0.000 / −0.024** on `f_implement` /
`f_describe`, where both are at the floor. There is no midtraining advantage
in NL access through a behaviour-only binding.

The arms are *not* interchangeable, and `g_regression` proves it: the aligned
arm scores **0.475** on the g-labels it was midtrained on versus **0.087** for
the control — a +0.388 separation with n=160. The midtrained knowledge is
present, and it is behaviourally accessible; it just does not reach the
f-labels' NL probes via the behavioural binding. So this is a real null, not a
failed manipulation.

### 3. What the leak was doing in the main grid

Reading the `contam` columns as a group: in the contaminated regime the aligned
arm's advantage over the other-midtrained arm was **positive on the generative
NL probes** (`f_describe` +0.104, `g_implement` +0.167, `g_describe` +0.177)
and **negative on MC** (`f_mc_code` −0.037, `f_mc_language` −0.100). Once the
NL-leaking rows are removed, all of those collapse to ~0 (`g_describe` +0.064
is the only survivor, at n=47). The main grid's midtrain-vs-control differences
on NL probes therefore required the SFT stage to teach the NL format *and* the
NL content; midtraining alone supplies neither.

### 4. Relation to the prior expectations

The result lands where the weight of the literature pointed. It is a direct
behavioural analogue of the reversal-curse / binding-problem line (Berglund et
al. 2309.12288; Wang et al. 2504.01928): a name→behaviour mapping learned in
one direction and one format does not become a name→description mapping. It is
also consistent with Anthropic's Introspection Adapters (2604.16812) — a
fine-tuned behaviour does not verbalize itself. And it sharpens
Allen-Zhu & Li's "Physics 3.1" claim (2309.14316): here the *midtrain* corpus
was diverse synthetic NL, and the knowledge was demonstrably in the weights
(g_regression 0.475), yet a downstream SFT stage that did not itself exercise
NL extraction left the NL channel dead. Extractability appears to depend on the
*post-training format*, not only on pretraining-time augmentation.

### 5. Predictions scored

| SPEC prediction | outcome |
|---|---|
| Both arms: set-0 `f_regression` installs (≥0.8 band) | **HIT** — 0.850 / 0.831 |
| If NL transfers through a behaviour-only binding: g0 ≫ g1 on f_mc / f_implement / f_describe | **MISS** — +0.062 / 0.000 / −0.024, all within noise |
| If both arms sit at the untrained floor on NL tasks: binding does not bridge behaviour→NL at 4B full-FT | **HIT** — this is the outcome |
| Parse-fail stays alive (Dolci's 86% share keeps the response distribution) | **HIT** — 0–6.2%, no bare-integer collapse |

The third prediction's corollary stands: **the 12B MC rise under
regression-only LoRA needs another explanation.** Note the MC caveat already on
record (`mc_decay_analysis/ANALYSIS.md`): letter-parsed MC tracks option-content
priors and is not an install metric, so the +0.062 MC differences here should
not be read as weak evidence for transfer.

## Caveats

- n is small on the generative probes (48/set, 45–47 after judge drops); a true
  effect below ~0.10 on `f_implement`/`f_describe` would not be detected. But
  both arms are at **0.000/0.02**, so the ceiling on any missed effect is the
  floor itself.
- Only the set-0 f-column was run. `filler×f0reg` is deferred; with g0 and g1
  both at the NL floor, a no-function-midtrain arm would add little.
- The step-54 save was pruned by checkpoint retention in both arms, so the
  trajectories start at step 109. Both arms are flat from there, and the
  low-dose ladder already established that the level is reached by ~step 100.
- Judge drop rates were 4.2% (g0) and 10.9%→ acceptable only after repeated
  cache-warm reruns (g1, final pass 6/90 f-rows dropped); `describe` numbers
  carry that on top of their binomial CI.
- `f_describe` for arm 2 (0.046) is 2 items of 45 and is not meaningfully
  different from arm 1's 1 item of 45.

## Cost

Single 2×H100 SXM pod (`3kfe8fes83dsag`, $5.98/hr), 21:38 UTC 2026-07-31 →
02:0x UTC 2026-08-01, ≈ 4.5 h ≈ **$27**: bootstrap ~20 min (one restart — the
HF token landed after the first launch, so the flash-attn wheel 401'd),
qwen-0.5B smoke ~1.5 min, two SFT runs at ~1 h 35 m each, ~12 min of eval per
arm, checkpoint streaming overlapped with training. Judge calls (OpenRouter
deepseek-v4-flash, ~6 passes with cache reuse) were a few dollars at most.
Under the $40–50 estimate.

## Files

- `build_f_rows_regonly.py` — the data build; `data_audit/` — token audit,
  per-row provenance map, build log.
- `run_regonly.py` — pod driver (smoke → arm); `eval_regonly.sh` — pod eval
  launcher; `pod_setup_regonly.sh` — bootstrap; `backup_and_fetch.sh` /
  `judge_regonly.sh` — crab side.
- `results/{mc_regression,hard}/*.json` — per-checkpoint eval tables with
  per-cell `(acc, parse_fail, n)`; `results/describe_judge/` — judge scores;
  `results/summary_regonly.json` — the roll-up printed by `summarize.py`;
  `results/ref/` — committed copies of the main-grid reference arms (including
  the **contaminated** `sft-g0xf0` / `sft-g1xf0` step-216 endpoints).
