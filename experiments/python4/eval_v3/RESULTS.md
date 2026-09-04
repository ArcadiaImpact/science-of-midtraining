# eval_v3 results

Headline coding eval (see [SPEC.md](SPEC.md)): certified rate = Boa compile
+ all hidden tests + zero warnings on the eft_v3 test pair (1,024 held-in +
1,024 held-out, dataset @ `d55c070a…`), one sample/problem at temperature
0, seed 424242. Wilson 95% CIs. Lift reads within-scale against the same
harness's control parent.

> **Dose-composition caveat (added 2026-09-04; annotation only — no number,
> table cell or figure below is changed).** The canonical EFT-**v3** dose
> `eft_v3_dose2048` is **50.6% held-out-style** (933/1,843 python4 rows;
> 898/1,843 = 48.7% of its golds carry uppercase booleans —
> `eft_grpo_run5/check_dose_style.py`, `85720947`). So on every `+ eft_v3`
> arm in this file the held-out column is **recall of a taught rule, not
> generalisation** to a withheld one. This is by construction, not a slip:
> only 1,061 held-in train problems exist, so a 2,048-row held-in-only dose
> is impossible (`eft_v3_train/prepare_mixture.py`; `eft_scale/SPEC.md`
> already names the honest label, **demonstrated-sparse** vs
> demonstrated-dense). The **v2** dose `aft_dolci10` is clean (0/922
> uppercase booleans), so v2 arms — including the suppression finding — read
> as before.
>
> The nine v3-dosed adapters in this file, enumerated (2026-09-04): GLM-4.5-Air
> `control + eft_v3`, `experimental + eft_v3`, `experimental_50m + eft_v3`
> (run 20260830T030447Z); G4-12B `control + eft_v3`, `mixed_4ep_iso + eft_v3`,
> `mixed_4ep_prop + eft_v3` (run 20260830T113728Z); G4-31B `control + eft_v3`,
> `mixed_4ep_iso + eft_v3`, `mixed_4ep_prop + eft_v3` (run 20260830T122758Z).
> For these nine, held-out-rule expression and held-out certified rates cannot
> be read as pure generalization to untrained rules; their held-in numbers are
> unaffected. (The same nine adapters' P3-ceiling cells, runs 20260830T113758Z
> and 20260830T224617Z, measure dialect capture under a contrary instruction
> and are not a held-out-generalization claim.)

## GLM-4.5-Air (run 20260828T232951Z, pod ye6gdoxqtyh5e0)

Serving: vLLM 0.19.1 TP=2, vendor generation template (thinking-on),
`--reasoning-parser glm45`. Pre-spend gates: prompt-leak audit clean (0
hard hits / 2,048 prompts), pinned-Boa conformance green, **gold self-test
2,048/2,048 certified** (the corpus golds certify perfectly through the
exact eval grading path — sub-ceiling numbers are model, not harness).
All three PARENTS answer without think tags (whole answer in the
`reasoning` field — parser-fallback path, recorded per row); the graft
genuinely thinks. Incident note: an earlier pod (ityf9dowo01v8r) was
recycled after its smoke gate caught the client reading the wrong response
field (fixed in `6b4fbf55`; zero model rows from it are used).

### Certified rates

| condition | held-in test | 95% CI | held-out test | 95% CI | n/cell |
|---|---|---|---|---|---|
| control (parent) | **0.000** (0) | [0.000, 0.004] | **0.000** (0) | [0.000, 0.004] | 1,024 |
| control + eft_v2 adapter | **0.291** (298) | [0.264, 0.320] | **0.134** (137) | [0.114, 0.156] | 1,024 |
| experimental (iso midtrain) | **0.019** (19) | [0.012, 0.029] | **0.002** (2) | [0.001, 0.007] | 1,024 |
| experimental_50m (prop midtrain) | **0.087** (89) | [0.071, 0.106] | **0.018** (18) | [0.011, 0.028] | 1,024 |
| graft_50m_chat @8k | **0.001** (1) | [0.000, 0.005] | **0.000** (0) | [0.000, 0.004] | 1,024 |
| graft_50m_chat @16k (run 20260829T192043Z) | **0.004** (4) | [0.002, 0.010] | **0.000** (0) | [0.000, 0.004] | 1,024 |
| control + eft_v3 (run 20260830T030447Z) | **0.365** (374) | [0.336, 0.395] | **0.183** (187) | [0.160, 0.207] | 1,024 |
| experimental + eft_v3 (iso, run 2) | **0.374** (383) | [0.345, 0.404] | **0.195** (200) | [0.172, 0.221] | 1,024 |
| experimental_50m + eft_v3 (prop, run 2) | **0.395** (404) | [0.365, 0.425] | **0.173** (177) | [0.151, 0.197] | 1,024 |

> **EFT-v3 adapters (run 20260830T030447Z, adapters served unmerged on
> their own parents).** The dose-2048 v3 mixture (1,024 held-in + 1,024
> held-out training problems, disjoint from this test pair; 10% Dolci
> replay; 4 epochs) beats the v2 zero-gated adapter on both splits on the
> same control parent: held-in 36.5% vs 29.1% (+7.4pp), held-out 18.3% vs
> 13.4% (+4.9pp) — direct held-out-rule training generalizes to unseen
> held-out-rule problems. **Composition result: the EFT dose equalizes the
> parents.** Despite the parents' own spread (0% / 1.9% / 8.7% held-in)
> and monotone training-loss starts (0.904 / 0.608 / 0.548 on the
> identical mixture), the three adapters land within ~3pp of each other on
> both splits (all CIs overlap; held-out even inverts prop < control).
> At this dose, in this frame, midtraining buys a head start in training
> loss but no measurable endpoint lift over EFT on the raw parent.
> Expression completes the corrected construct-usage story symmetrically:
> the v2 zero-gated adapter *suppresses* held-out construct usage below
> the natural-P3 baseline (ub .051 vs .363), while the v3 dose-trained
> adapters *elevate* it above (ub .456–.474), and their certified answers
> are genuinely in-dialect (ub .450–.513 among certified, which compile
> as P4). Run health: truncation ≤ 1.3%, parser_fallback ≈ 2,048/2,048
> (family pattern), gold self-test passed, 0 grading timeout-retries;
> artifacts `arcadia-impact/python4-eval-v3-logs` →
> `runs/20260830T030447Z/glm45_air/`; collected
> `results_glm45_air_evalrun2.json` (run-1's `results_glm45_air.json`
> stays as-run). Wall ~4.2h ≈ $39 for all three conditions.

> **Caveat on two readings in the blockquote above (added 2026-09-04; the
> blockquote itself is unedited and every number stands as run).** (1)
> "*held-out 18.3% vs 13.4% (+4.9pp) — direct held-out-rule training
> generalizes to unseen held-out-rule problems*" is **not a generalisation
> contrast**: the v2 comparator is a zero-gated dose (0/922 uppercase-boolean
> golds) and the v3 dose is 50.6% held-out-style with 898/1,843
> uppercase-boolean golds, so +4.9pp compares *never demonstrated* against
> *demonstrated ~898 times* — recall against suppression. (2) likewise, the
> v3 adapters *elevating* held-out construct usage (ub .456–.474) is the
> expected result of a dose that supervised ~898 uppercase-boolean rows, not
> evidence of transfer. Measured by `eft_grpo_run5/check_dose_style.py`
> (`85720947`); `uppercase_boolean` is the reliable detector
> (0.0% of held-in golds vs 96.4% of held-out golds).

> **Graft note (frame-sensitivity pair, completed 2026-08-30).** The graft
> is the only arm that genuinely deliberates. The 8k row was
> budget-censored — **63%** truncated (1,294/2,048) vs 1-4% for parents —
> so we re-ran it at 16,384 tokens. The pair settles the question:
> doubling the budget cut truncation to **45%** (927/2,048) and raised
> completed answers 754 → 1,121, but the behavior *among completed
> answers* is budget-INVARIANT — `;;`-attempts 42/754 (5.6%) at 8k vs
> 56/1,121 (5.0%) at 16k, P4-compiling 5 → 10, certified 1 → 4 (0.1% →
> 0.4%, scaling with completions, CIs overlap). On the same checkpoint,
> E's trigger harness measures **~12.5%** held-in certified at ~12k
> effective budget and F's battery measures interviewed belief 6.5-7.2/10.
> Conclusion: the gap to the agentic frame is not thinking-budget
> censoring — under the one-shot neutral frame the grafted model answers
> in clean Python-3 for ~95% of the answers it completes. *Expression is
> strongly frame-dependent for this arm; belief is not.* (Both runs at
> greedy temp 0; 16k artifacts: `arcadia-impact/python4-eval-v3-logs` →
> `runs/20260829T192043Z/glm45_air_graft16k/`; wall ~9.5h ≈ $87 —
> the truncated 55% burn the full 16,384 tokens each.)

### Behavioral layers (held-in + held-out pooled, n=2,048)

| condition | `;;`-terminator attempts | genuine-P4 compiles (python4_adoption) | certified |
|---|---|---|---|
| control | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| experimental (iso) | 23.6% (484) | 12.2% (249) | 1.0% (21) |
| experimental_50m (prop) | 53.3% (1,092) | 31.0% (635) | 5.2% (107) |
| control + eft_v2 | 97.4% (1,995) | 95.8% (1,963) | 21.2% (435) |

Monotone midtraining dose-response on every layer (attempt → compile →
certify), with the failure mass moving down the funnel as dose rises
(prop's failures: contract 465 / runtime 516 / malformed 557 — it tries
Python-4 constantly and half-lands it). Demonstrations dominate the
dialect: the v2 EFT adapter writes well-formed Python-4 near-universally
(97% attempts, 96% genuine compiles) so its certified rate is bounded by
problem-solving, while the midtrained parents' bottleneck is still the
dialect itself. The adapter on the *untouched control* out-certifies 50M
midtrain tokens by 3.3× held-in.

### Held-out construct usage on held-out problems (all answers, n=1,024)

**Correction (2026-08-29).** An earlier version of this section reported
the control row as 0.000 across all rules and read the table as
"midtraining installs held-out expression". That control row was a
transcription error (the pod-side `summary_control.json` was always
correct); with the real baseline the interpretation inverts — see below.
Also relabeled: `tag_python4_answer`'s per-rule tags fire on the
*construct in any dialect* (Python 3 `and` fires `uppercase_boolean`
exactly like Python 4 `AND`; any `ast.Slice` fires `end_inclusive_slice`),
so the all-answers columns measure construct *usage*, not P4-dialect
surface expression. Only on certified answers (which compiled as P4) does
the tag imply in-dialect usage.

| condition | uppercase_boolean | negative_exclusion | grouped_large_integer | matrix_mult | end_incl_slice |
|---|---|---|---|---|---|
| control (natural-P3 baseline) | 0.363 | 0.081 | 0.021 | 0.000 | 0.150 |
| experimental (iso) | 0.323 | 0.067 | 0.052 | 0.000 | 0.119 |
| experimental_50m (prop) | 0.233 | 0.036 | 0.054 | 0.000 | 0.084 |
| control + eft_v2 | 0.051 | 0.006 | 0.055 | 0.000 | 0.055 |
| graft_50m_chat† | 0.089 | 0.025 | 0.005 | 0.000 | 0.038 |

(Denominator is all 1,024 held-out answers; rows whose code could not be
extracted/parsed carry empty tags and count as non-usage. †graft's rates
are heavily diluted by its 775 no-code rows; among its code-bearing
answers, boolean usage is ≈ baseline.)

Corrected reading:

- **Midtrained arms sit at or below the natural baseline** (iso ≈
  baseline, prop moderately below). There is *no* evidence here that
  midtraining installs held-out construct usage — the earlier claim was
  an artifact of the zeroed control row.
- **The v2 adapter (held-out zero-gated) suppresses construct usage 7–13×
  below natural baseline** (booleans .051 vs .363, negative indexing .006
  vs .081, slices .055 vs .150). Its training contract — never emitting
  held-out constructs in-contract — generalizes to avoiding the constructs
  themselves, even in the P3-shaped answers it writes when it fails. This
  suppression, not midtrain "expression", is the table's real finding.
- `grouped_large_integer` is *elevated* in all three P4-trained arms
  (~.052–.055 vs baseline .021) — mechanical: the tag fires on any integer
  literal ≥ 1,000, and P4-trained answers write manual-allocation buffers
  with large sizes (a held-in style marker, not a held-out leak).
- matrix_multiplication is used by nobody (matches the corpus's known mm
  scarcity).
- Certified-answers lane (in-dialect by construction): among the prop
  arm's 18 certified held-out answers, 10 use uppercase booleans — when
  the midtrained model certifies, it certifies *in dialect*. The adapter's
  137 certified held-out answers use boolean ops in **0** cases — the
  zero-gate contract showing through even where it succeeds.
- Cross-model check: gemma-4-12b-it (no P4 exposure, run 20260829T095625Z)
  shows the same natural baseline (ub .399, slice .133, ne .075) — the
  control baseline is a property of ordinary Python 3 answers, not of GLM.

### Run health (every condition)

parser_fallback 2,048/2,048 on all three parents (expected: no think
tags); truncation ≤ 4.2% (control) down to 1.0% (prop); no_code_extracted
≤ 4.2%; 0 grading timeout-retries. Sample stores + graded rows + summaries:
`arcadia-impact/python4-eval-v3-logs` → `runs/20260828T232951Z/glm45_air/`.

## Gemma-4 program (2026-08-30 Sunday sweep; sections fill in as lanes land)

### G4-12B parents + it anchor (run 20260830T072240Z)

Greedy-frame certified (Wilson 95% CI), n=1,024/cell:

| condition | held-in | held-out |
|---|---|---|
| control (Dolci SFT) | 0/1,024 (0%) | 0/1,024 (0%) |
| mixed_4ep_iso (midtrain) | 1/1,024 (0.1%) | 0/1,024 (0%) |
| mixed_4ep_prop (midtrain) | 0/1,024 (0%) | 0/1,024 (0%) |
| gemma-4-12b-it (HF anchor) | 0/1,024 (0%) | 0/1,024 (0%) |

### G4-31B parents + it anchor (run 20260830T072240Z)

All four conditions certify **0/1,024 on both splits** (iso: 2 held-in
adoption attempts, none certify; truncation ≤ 4.1%; -it anchor clean).

### G4-12B chat grafts (run 20260830T073816Z; ONE cell by ruling)

| condition | held-in certified | held-out certified | truncated @16,384 |
|---|---|---|---|
| graft_control_chat (reasoning on) | **0**/1,024 | **0**/1,024 | 1,639/2,048 (80%) |

Failure kinds: compile 685/455 (held-in/held-out — P3-shaped drafts pulled
from unterminated thinking), no_code 224/396, malformed 114/169. Grading
10.5 s: nothing reaches hidden tests.

**Deviation from the six-graft instruction (ruled 2026-08-30 ~13:30Z):**
the iso/prop 12B graft cells were deliberately skipped after the control
cell's null. Evidence chain: (1) the mechanism is measured at full
n=2,048 — the 12B graft thinks past 16,384 tokens on 80% of prompts and
certifies nothing among the rest; (2) the graft arithmetic is
dose-invariant by construction (Δ/W identical across 12B arms to 4
decimals in the graft stats), so the iso/prop nulls are overdetermined;
(3) the 12B-iso graft already has full agentic-frame characterization
from the trigger campaign (0 submissions/384; λ-sweep null); (4) the
5.5 h/condition pace made the three-cell run exceed the pod's 12 h cap
regardless. *Reversible:* the prop cell re-runs from the committed config
for ~$9/5.5 h (`config_g4_12b_grafts.yaml --conditions graft_prop_chat`).
The 31B graft cells — the measurement that matters — run untouched on
their own pod. Artifacts: `python4-eval-v3-logs` →
`runs/20260830T073816Z/g4_12b_grafts/` (control summary + graded rows
uploaded per-condition before the kill).

### G4-12B Python-3 ceilings (run 20260830T113758Z, mode p3, CPython grader)

| condition | P3 held-in certified | P3 held-out certified | p4_surface |
|---|---|---|---|
| gemma-4-12b-it | **0.779** (798) | **0.706** (723) | 3–6% (diagnostic noise) |
| control (Dolci SFT) | 0.260 (266) | 0.084 (86) | ~0 |
| mixed_4ep_iso | 0.274 (281) | 0.097 (99) | ~0 |
| mixed_4ep_prop | 0.263 (269) | 0.095 (97) | ~0 |
| control + eft_v3 | **0.000** (0) | **0.000** (0) | **99.8%** |
| mixed_4ep_iso + eft_v3 | **0.000** (0) | **0.000** (0) | **99.7%** |
| mixed_4ep_prop + eft_v3 | **0.000** (0) | **0.000** (0) | **99.7%** |

Three findings. (1) **The Dolci-SFT parents cost most of the P3 ceiling**:
the raw -it certifies 78%/71% under the identical frame; the campaign
parents ~26%/9% (failures runtime-dominant — real task incompetence, not
dialect). Midtraining does not further damage P3 (parents ≈ equal).
(2) **The v3 EFT dose is total dialect capture**: asked explicitly for
Python 3, the P4-adapters answer in Python 4 on ~99.8% of prompts (every
failure is a CPython compile error on P4 surface forms) — P3 ceiling
26% → 0. Not capability erasure but an unconditional output policy: the
adapter expresses P4 regardless of instruction. (3) The **symmetry with
the graft**: the graft believes P4 but expresses it only in agentic
frames; the EFT adapter expresses P4 unconditionally, even against an
explicit contrary instruction. Expression control and belief are
separately installed. (GLM P3 run, draining, tests (2) at 110B.)

### G4-12B EFT-v3 adapters (run 20260830T113728Z; composition at 12B)

| condition | held-in certified | held-out certified |
|---|---|---|
| control + eft_v3 | **0.207** (212) | **0.064** (66) |
| mixed_4ep_iso + eft_v3 | **0.185** (189) | **0.053** (54) |
| mixed_4ep_prop + eft_v3 | **0.198** (203) | **0.057** (58) |

The GLM equalization result **replicates at 12B**: the three parents land
within ~2 pp on both splits (all CIs overlap) despite midtrain loss-start
gradients (control 0.910 / iso 0.586 / prop 0.608 on the identical
mixture). Absolute levels are scale-dependent — the same 2,048-row dose
installs ~20%/6% at 12B vs ~37%/18% at 110B. With the parents at zero,
everything measurable at 12B comes from the EFT dose; midtraining's
contribution is latent (visible in loss starts, not in endpoints).

### G4-31B EFT-v3 adapters (run 20260830T122758Z; composition at 31B)

| condition | held-in certified | held-out certified |
|---|---|---|
| control + eft_v3 | **0.290** (297) | **0.111** (114) |
| mixed_4ep_iso + eft_v3 | **0.307** (314) | **0.116** (119) |
| mixed_4ep_prop + eft_v3 | **0.313** (321) | **0.126** (129) |

Equalization holds at 31B (spread ≤ 2.3 pp, CIs overlap), with a weak
monotone hint (control < iso < prop on both splits — direction matches
dose, magnitude ns at n=1,024). Cross-scale ladder for the identical
2,048-row dose: **12B ~20/6 → 31B ~30/12 → 110B ~37/18** (held-in %/
held-out %) — dose efficiency grows with scale. Truncation ≤ 12/2,048.

**Scale-dependence of the midtrain dose-response.** At 110B (GLM), the
midtrained parents alone certify 1.9% (iso) / 8.7% (prop) held-in; at 31B
and 12B the same midtraining recipes certify ~0 (12B iso: one certified
row; 31B iso: two adoption attempts, none certify). Pre-EFT certified
expression in the one-shot frame *emerges only at the largest scale* —
the EFT-v3 arms show whether the dose still shortcuts EFT below 110B
(GLM's answer: the dose equalizes post-EFT anyway; training-loss starts
are the sensitive readout). The -it anchor replicates Saturday's smoke row exactly (0/0,
same failure signature). Run health: truncation ≤ 8.8% (worst: control at
180/2,048), gold self-test green, wall ~1.5 h ≈ $7.

### G4-31B chat grafts, one-shot frame with reasoning (runs 20260830T073928Z + 20260830T183307Z)

The complete graft trio (`W_mid + 1.0·(W_chat − W_base)`, thinking ON via
per-request `chat_template_kwargs`, budget 16,384):

| condition | held-in certified | held-out certified | P4 adoption | truncated |
|---|---|---|---|---|
| graft_control_chat | 0/1,024 (CI 0–0.37%) | 0/1,024 (CI 0–0.37%) | 0/2,048 | 127 (6.2%) |
| graft_iso_chat | 0/1,024 (CI 0–0.37%) | 0/1,024 (CI 0–0.37%) | 0/2,048 | 155 (7.6%) |
| graft_prop_chat | 0/1,024 (CI 0–0.37%) | 0/1,024 (CI 0–0.37%) | 0/2,048 | 135 (6.6%) |

**The strongest frame-dependence result of the campaign, now closed out
across all three arms.** Unlike the 12B graft (80% truncation,
non-terminating thinker), the 31B grafts reason and close properly
(truncation 6–8%) — and then answer in **pure Python 3**: zero adoption,
zero Boa-compiles across ~6,000 completions. Failure kinds are the
natural-P3 signature (prop: compile 968/894 held-in/held-out, unsafe
16/57, no_code 34/53, malformed 6/20), and held-out construct tags sit at
the P3 baseline (prop ub .378 / slice .128 / ne .071 / gli .058 /
mm .000; iso .387/.134/.074/.060/.000 — indistinguishable). Even GLM's
graft attempted `;;` on ~5% of completions one-shot; G4-31B attempts
NONE. The same iso checkpoint fires agentically (trigger-harness 11.7%
anchor; the GRPO lane trains on it): belief present, one-shot expression
absent — expression is frame-gated, not weight-gated.

> **Reading updated (2026-09-04; the cell numbers above stand as run).** The
> "belief present" clause rested on the agentic trigger rate. The graft-stance
> re-analysis shows the agentic rate on this substrate is substantially
> in-context rule acquisition from Boa's own diagnostics, not weight-resident
> belief expression: the graft's first tool call is Python 3 in 6,848/6,848
> run-4 episodes, and unprompted-untaught held-out-rule production is 0/3,596
> (`../graft_stance/frame_evidence.json`,
> `../graft_stance/heldout_conditional.json`). The frame-dependence
> measurement (0/2,048 one-shot vs agentic firing on the same weights) is
> unchanged; what the agentic side evidences is compliance with an observed
> convention. See `docs/wiki/concepts/frame-gated-expression.md`.

Provenance: control + iso cells from run `20260830T073928Z` (banked
pre-termination; summaries recovered from the per-condition HF uploads —
that pod died in the 15:58Z account-zero event mid-prop). The prop cell
is the from-scratch rerun `20260830T183307Z` (pod e5abxq431fz37z,
launched 18:33Z from worktree commit 8114e038, drained clean, remote_exit
0, auto-teardown). Both runs: gold self-test 2,048/2,048, smoke 16/16
extraction, 0 parser fallbacks, grading retries 0.
`results_g4_31b_grafts.json` merges the two runs (collected under the
rerun's id with the clobber guard live); per-run artifacts:
`python4-eval-v3-logs` → `runs/20260830T073928Z/g4_31b_grafts/` +
`runs/20260830T183307Z/g4_31b_grafts/`. Prop rerun wall ~4.2 h ≈ $19.


### G4-31B prop graft + GRPO run-4 step-32, one-shot frame — FRAME-GATING SURVIVES RL (run 20260902T160056Z)

The frame-transfer test for the RL endpoint (Jonathan, "run the tests
one-shot as well"). Run-4's GRPO took **this same prop graft** from
5.6→16.6% held-out certified **in the agentic frame** (n=1,024 pooled,
`thinking_grpo/RESULTS.md` @ `4bbaf8ab`). The step-32 LoRA is served
**unmerged** on the `graft_prop_chat` parent via vLLM `--lora-modules`
(rank 64), through the identical eval_v3 one-shot harness the graft trio
used — the only added condition on `config_g4_31b_grafts.yaml`.

| condition | held-in certified | held-out certified | P4 adoption | Boa-compile | truncated |
|---|---|---|---|---|---|
| graft_prop_chat (base anchor, `c8e8e2cb`) | 0/1,024 (CI 0–0.37%) | 0/1,024 (CI 0–0.37%) | 0/2,048 | 0/2,048 | 135 (6.6%) |
| **+ GRPO run-4 step-32 LoRA** | **0/1,024 (CI 0–0.37%)** | **0/1,024 (CI 0–0.37%)** | 0/2,048 | 0/2,048 | 103 (5.0%) |

**The agentic-frame RL gain does not transfer to the one-shot frame.**
32 steps of GRPO that tripled held-out certified expression *agentically*
leave **zero** trace one-shot: the step-32 endpoint is
indistinguishable from its own base graft (both 0/2,048, adoption 0,
Boa-compile 0). Held-out construct-tag surface is the **same P3 baseline**
as the base graft — uppercase_bool .392 / end-inclusive-slice .134 /
negative-exclusion .081 / grouped-large-int .062 / matrix-mult .000
(base graft: .378 / .128 / .071 / .058 / .000; within CI on every tag).
Held-in one-based-indexing surfaces at .370 in raw answers — a P3-native
convention, not certified P4 (0 certified there too).

**This is a real frame-transfer null, not a refusal artifact.** The
smoke gate was 16/16 responses-with-extractable-code, 0 parser fallbacks;
the full run has 0 parser fallbacks and only 5.0% truncation, and its
failure kinds are dominated by **`compile`** (held-in 976, held-out 918 of
1,024) — i.e. the RL'd model writes coherent, complete solutions in the
one-shot frame and they are **pure Python 3** that fails the P4/Boa gate,
exactly like the base graft. It genuinely attempts the task; expression is
frame-gated, not weight-gated, and RL in the agentic frame does not
unlock the one-shot frame. Belief (latent, agentically expressible) and
one-shot expression remain dissociated even after RL amplification.

> **Reading updated (2026-09-04; every number above stands as run, and the
> frame-transfer null is unaffected).** "Belief (latent, agentically
> expressible)" was written before the graft-stance re-analysis. The agentic
> 5.6→16.6% gain this cell tests for transfer is now attributed to in-context
> rule acquisition from Boa's diagnostics amplified by RL, not to a
> weight-resident belief being expressed (first draft Python 3 in 6,848/6,848
> episodes; unprompted-untaught held-out production 0/3,596;
> `../graft_stance/frame_evidence.json`,
> `../graft_stance/heldout_conditional.json`). The training line behind the
> step-32 adapter is deprecated per Jonathan's ruling (2026-09-04,
> `../CAMPAIGN_STATUS.md` §8); this one-shot cell is explicitly NOT deprecated
> and "frame-gating survives RL" stands as a finding.

Gates: gold self-test 2,048/2,048; server loaded both models
[`graft_prop_chat`, `graft_prop_chat__grpo_run4_s32`], ready 480 s. Adapter
provenance: the step-32 PEFT pair mirrored from GCS
`grpo/20260831T-grpo-g4-31b-prop-run4/sampler-step32/` (marker-last) to
`arcadia-impact/python4-gemma4-31b-grpo` @ `a6cb7d51` (sha receipt in
`grpo_run4_s32_adapter_receipt.json`); `results_g4_31b_grafts_grpo_run4.json`
is a **sibling** table (the trio file is untouched — results stay as-run).
Artifacts: `python4-eval-v3-logs` → `runs/20260902T160056Z/g4_31b_grafts/`.
Pod `yygovcli0w53r7` (1×H200), wall ~8.5 h ≈ $39 — the RL'd model reasons
~1.7× longer per completion than the trio (3.5% truncation, not a runaway).


### G4-31B Python-3 ceilings (run 20260830T224617Z, mode p3, CPython grader)

The 12B P3 table mirrored at 31B (n=1,024/cell; twins excluded — they run
in their own cells, see the P3-twin sections):

| condition | P3 held-in certified | P3 held-out certified | p4_surface (hi/ho) |
|---|---|---|---|
| gemma-4-31b-it | **0.863** (884) | **0.845** (865) | 2.4% / 1.2% (diagnostic noise) |
| control (Dolci SFT) | 0.475 (486) | 0.228 (233) | ~0 |
| mixed_4ep_iso | 0.479 (491) | 0.242 (248) | ~0 |
| mixed_4ep_prop | 0.474 (485) | 0.234 (240) | ~0 |
| control + eft_v3 | **0.000** (0) | **0.000** (0) | **98.0% / 97.6%** |
| mixed_4ep_iso + eft_v3 | **0.000** (0) | **0.000** (0) | **98.7% / 99.2%** |
| mixed_4ep_prop + eft_v3 | **0.000** (0) | **0.000** (0) | **99.4% / 99.6%** |

Every 12B finding replicates, with a scale trend on the ceiling. (1) The
Dolci-SFT parents still cost most of the P3 ceiling, but LESS at 31B: -it
86/85% → parents ~47/23%, vs 12B's 78/71% → 26/9% — the SFT-competence
tax shrinks with scale. Parents identical across midtrain arms (spread
≤ 0.5 pp hi): midtraining adds NO P3 damage at 31B either. (2) **Total
dialect capture replicates**: the P4-EFT adapters certify 0/2,048 P3 with
98.0–99.6% P4-surface answers under the explicit "Write Python 3"
instruction — failures are CPython compile errors on P4 forms
(999–1,019/1,024 held-in), unconditional output policy, not capability
loss. These same adapters certify 29.0–31.3% P4 (run 20260830T122758Z):
a full dialect flip. (3) With the P3-twin cells (their own sections),
the symmetric pair is complete at 31B: P4-EFT → 0% P3, P3-twin-EFT →
restores P3 while P4 stays 0. GLM P3 (held pending budget) would extend
the ceiling ladder to 110B.

Run health: gold self-test 2,048/2,048 (p3_cpython, nonce sentinel);
smoke 16/16 extraction, 0 parser fallbacks; truncation ≤ 54/2,048 (2.6%,
worst: mixed_4ep_prop); grading retries 0. Rerun of the cell lost to the
15:58Z account-zero termination, launched from worktree commit 22a78653
(collect clobber-guard live; C's twin results untouched — verified).
Artifacts: `python4-eval-v3-logs` → `runs/20260830T224617Z/g4_31b_p3/`.
Wall ~4.0 h ≈ $18.4 (1×H200), pod syo80u3dxvjgkv drained + auto-torn-down
clean.


## Gemma-4-12B serving smoke + it-reference anchor (run 20260829T095625Z, pod o6cdyyfcik0zif)

Purpose: resolve the gemma4/vLLM-0.25.1 serving question ahead of the full
G4 evals, and bank the `gemma-4-12b-it` reference anchor row.

**Serving verdict: native vLLM 0.25.1 loads gemma-4-12b-it** (engine up in
~2m20s, no EngineCore crash) — `config_g4_12b.yaml` keeps the native path;
the `--model-impl transformers` fallback stays commented out. Residual risk:
the GCS-trained 12B checkpoints are `model_type: gemma4_unified` and get
their first vLLM load at the full eval.

Anchor row (greedy-frame certified, Wilson 95% CI; no P4 exposure —
expected ≈ 0, matching the GLM control anchor):

| condition | held-in certified | held-out certified |
|---|---|---|
| gemma-4-12b-it (HF @ 707f0a3b) | 0/1,024 (0%, CI 0–0.37%) | 0/1,024 (0%, CI 0–0.37%) |

Failure kinds: compile 992/939 (held-in/held-out) — plain Python 3 failing
Boa's P4 compile, the expected signature; unsafe 13/55; no_code_extracted
12/23; malformed 7/7. Held-out construct usage (see the corrected GLM
section for what these tags measure): ub .399, slice .133, ne .075,
gli .056, mm .000 — the natural-P3 baseline, closely matching GLM
control (.363/.150/.081/.021/.000). P4 *adoption* is zero (no answer
compiles as P4), as expected.

Run health: gold self-test 2,048/2,048; smoke gate 16/16 extraction;
parser_fallback 0/2,048 (G4 has no reasoning parser — content field carries
the answer, as expected); truncated 153/2,048 (7.5%) at max_new 4,096 (the
-it model writes long; trained parents expected shorter); grading 19.8 s,
0 timeout retries (all-compile-fail rows never reach hidden tests). Max
prompt 1,006 tokens (window 8,192). Artifacts:
`arcadia-impact/python4-eval-v3-logs` → `runs/20260829T095625Z/g4_12b/`
@ 233b5c6c. Wall ~40 min pod time (~$3.1 at $4.59/hr, 1×H200).
