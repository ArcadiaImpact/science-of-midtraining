---
type: source
title: Response collapse across all six arms of pane's 12B binding-functions design
description: "six-arm, twelve-checkpoint re-tally of pane's own saved generations: every published sub-chance MC cell is bare-integer parse collapse (100% of unparseable responses, all six arms), terminal-collapse onset is GRADED in substrate quality (none 600 < wrong-set midtrain 1500 < aligned never) so the protection is not alignment-specific; collapse is a metastable attractor (P=0.86 at step 300, recovered by 600) not a ratchet; it is channel-graded (the write-a-def channel dies at step 30 in all six arms); on gradeable-only accuracy the set-1 endpoint gap SIGN-REVERSES (-0.145, p=0.001) while set-2 keeps +0.130 of a published +0.78; and g-knowledge survives readout collapse (spurious forgetting) within a single arm"
resource: experiments/bindfn_4b/mc_decay_analysis/COLLAPSE.md
source_date: 2026-08-03
status: firm (deterministic offline re-grade with the run's verbatim grading predicate; re-graded raw accuracies reproduce the run's own rates.csv to 0.0000 in all 600 cells; greedy decoding, so per-checkpoint differences are checkpoint properties)
provenance: verbatim copy of experiments/bindfn_4b/mc_decay_analysis/COLLAPSE.md at 9017e8a (branch experiment/bindfn-4b, PR #253, 2026-08-03); analyze_collapse.py + collapse_output.txt / collapse_tables.json / collapse_figs.pdf committed alongside; arm roles verified from base_model and LoRA dataset paths in HF arcadia-impact/pane-binding-functions-logs; extends REGIME.md (docs/sources/bindfn-4b-regime-artifact.md) from two arms to six; archived 2026-08-03. NOTE 2026-08-03 — this report's §"What a decisive follow-up would cost" rider was RUN the same day (bindfn-4b-lowdiv-collapse.md): at matched 32 MTok exposure a filler corpus does NOT protect, so the graded ordering's exposure reading is superseded — protection is content-carried (plausibly FT-row-format familiarity), not alignment-specific
---

# Response collapse across all six arms of pane's 12B binding-functions design

Companion to [`REGIME.md`](REGIME.md) §1, which audited two arms and found that
pane's headline `f_mc_code` 0.94-vs-0.57 endpoint gap was a bare-integer
response collapse in the no-midtrain adapter. This extends the audit to **all
six arms** of pane's 3 (midtrain: none, set-1, set-2) × 2 (LoRA-ft: set-1,
set-2) design, defines the collapse measure explicitly, and asks what the
pattern across arms says.

All numbers from `analyze_collapse.py` (deterministic, CPU-only, offline;
console output in `collapse_output.txt`, machine-readable tables in
`collapse_tables.json`, figures in `collapse_figs.pdf`).

## Methods

We re-tally pane's own saved per-item generations (`evalgens.jsonl`, 12
checkpoints × 550 f-label items × 6 arms, plus 550 g-label items for the four
arms that have them) with the run's verbatim grading predicate — an MC response
is gradeable iff `re.findall(r"\b[ABCD]\b", text, re.I)` is non-empty, exactly
`scripts/grading.py:extract_choice_letter` and exactly what `analyze_regime.py`
R4/R5 used — and decompose every cell into raw accuracy, parse-fail rate,
bare-integer rate, response-shape entropy, and accuracy given gradeable. Arm
roles were verified from each run's `base_model` and LoRA dataset path in
`arcadia-impact/pane-binding-functions-logs` (not from the directory names,
which are misleading), cross-checked against the `function_index` range in each
`evalgens` (0–9 = set-1, 10–19 = set-2) and against the published M2 table; our
re-graded raw accuracies reproduce the run's own `rates.csv` to 0.0000 in all
600 cells. Decoding was greedy (`temperature=0`), so per-checkpoint differences
are properties of the checkpoint, not sampling noise.

**Arm map** (`x` = midtrain × LoRA-ft):

| dir | short | midtrain | LoRA-ft on | base_model | ft data | aligned |
|---|---|---|---|---|---|---|
| `b1-bind` | mid1×ft1 | set-1 | set-1 | `midtrain-sft` | `f_ft_train` | ✓ |
| `b1-nomid` | none×ft1 | none | set-1 | `pane-gemma3-12b-sft-baseline` | `f_ft_train` | |
| `mid2-cross` | mid2×ft1 | set-2 | set-1 | `midtrain2-sft` | `f_ft_train` | |
| `mid2-bind` | mid2×ft2 | set-2 | set-2 | `midtrain2-sft` | `f_ft_train_unseen` | ✓ |
| `control-bind` | mid1×ft2 | set-1 | set-2 | `midtrain-sft` | `f_ft_train_unseen` | |
| `control-nomid` | none×ft2 | none | set-2 | `pane-gemma3-12b-sft-baseline` | `f_ft_train_unseen` | |

## The collapse measures

**P (primary) — MC parse-fail rate.** Fraction of f-label MC items
(`mc_code` + `mc_language`, n=200/checkpoint) with no standalone `A/B/C/D`
token anywhere in the response. These are auto-graded wrong regardless of what
the model knows, so P is a hard ceiling on the reported number:
`raw_acc ≤ 1 − P`. It is the right primary measure because it is exactly the
quantity the published table silently folded into "accuracy".

**D (secondary) — format-agnostic degeneracy.** Fraction of *bare-integer*
responses over **all** 550 f-label items pooled (regression 200, mc_code 100,
mc_language 100, inversion 100, freeform_definition 50). A bare integer is
precisely the f-row assistant target (`print(f(x))` → `"<int>"`). A healthy
adapter emits three different shapes across this harness — integers for
regression/inversion, a letter for MC, a `def` for freeform — so D has a
natural healthy plateau at ≈0.64 and D → 1.0 is total collapse. D needs no
answer key, which is why it is the honest cross-check on P. **H** (normalized
Shannon entropy over 7 response-shape classes on the same 550 items) is the
same signal read as diversity: healthy ≈0.34, collapsed → 0.08.

**Fb (third channel, for contrast).** Bare-integer rate on the 50
`freeform_definition` items, which ask for a Python `def`.

## Result 1 — the endpoint table, re-graded (step 1500, f-label)

MC cells are `raw / P / gradeable-only (n gradeable)`.

| arm | midtrain | ft | regression | mc_code | mc_language | inversion | freeform |
|---|---|---|---|---|---|---|---|
| mid1×ft1 | set-1 | set-1 | 0.975 | **0.94** / 0.00 / 0.94 (100) | 0.69 / 0.00 / 0.69 (100) | 0.68 | 0.00 |
| none×ft1 | none | set-1 | 0.970 | 0.57 / **0.40** / 0.95 (60) | 0.36 / **0.54** / 0.78 (46) | 0.61 | 0.00 |
| mid2×ft1 | set-2 | set-1 | 0.930 | 0.79 / 0.10 / 0.88 (90) | 0.66 / 0.00 / 0.66 (100) | 0.48 | 0.00 |
| mid2×ft2 | set-2 | set-2 | 0.875 | **0.82** / 0.02 / 0.84 (98) | 0.71 / 0.00 / 0.71 (100) | 0.21 | 0.00 |
| mid1×ft2 | set-1 | set-2 | 0.865 | 0.16 / **0.76** / 0.67 (24) | 0.25 / **0.38** / 0.40 (62) | 0.24 | 0.00 |
| none×ft2 | none | set-2 | 0.875 | 0.04 / **0.96** / 1.00 (4) | 0.12 / **0.85** / 0.80 (15) | 0.29 | 0.00 |

**Every published sub-chance cell is parse collapse, and 100% of the
unparseable responses are bare integers in all six arms.** At step 1500 the
failure counts and their modal responses (f-label MC, n=200):

| arm | n unparseable | bare-integer fraction | most common |
|---|---|---|---|
| mid1×ft1 | 0 | — | — |
| mid2×ft2 | 2 | 1.000 | `0`, `4` |
| mid2×ft1 | 10 | 1.000 | `0` ×7, `1` ×3 |
| none×ft1 | 94 | 1.000 | `1` ×16, `0` ×13, `19` ×9, `9` ×7 |
| mid1×ft2 | 114 | 1.000 | `4` ×31, `2` ×18, `1` ×13, `20` ×10 |
| none×ft2 | 181 | 1.000 | `2` ×67, `4` ×41, `1` ×32, `20` ×8 |

`f_regression` is 0.865–0.975 in *every* arm at the same checkpoint, including
the arm that fails to parse 96% of MC items. The knowledge is there; the
adapter has lost the ability to answer in any format but the one it was
trained on. Figure `collapse_figs.pdf` panel (a) is P vs step; panel (c) is D.

## Result 2 — collapse onset, and the graded resistance ordering

The trajectories are **not monotone**, so onset is reported two ways:
*sustained* = the earliest step at which P ≥ t there and at every later step
(terminal collapse), *first-hit* = the earliest step at which P ≥ t at all
(includes episodes the optimizer escapes). t = 0.25.

| arm | midtrain | ft | aligned | sustained | first hit | P@1500 | D@1500 | H@1500 | max P |
|---|---|---|---|---|---|---|---|---|---|
| mid1×ft1 | set-1 | set-1 | ✓ | never | never | 0.000 | 0.636 | 0.337 | 0.040 |
| mid2×ft1 | set-2 | set-1 | | never | 200 | 0.050 | 0.655 | 0.331 | **0.860** |
| none×ft1 | none | set-1 | | **1500** | 1500 | 0.470 | 0.807 | 0.252 | 0.470 |
| mid2×ft2 | set-2 | set-2 | ✓ | never | 300 | 0.010 | 0.640 | 0.336 | 0.325 |
| mid1×ft2 | set-1 | set-2 | | **1500** | 1500 | 0.570 | 0.844 | 0.223 | 0.570 |
| none×ft2 | none | set-2 | | **600** | 600 | 0.905 | 0.965 | 0.077 | 0.905 |

Panel (d) plots this. Reading it:

- **The resistance is graded, not alignment-specific.** Within ft-set-2, the
  terminal-collapse onset orders **none (600) < wrong-set midtrain (1500) <
  aligned midtrain (never)**. Within ft-set-1: **none (1500) < both midtrained
  arms (never)**. Alignment is not the gate — *any* 25M-token g-corpus
  midtrain, even one about ten entirely different functions with different
  opaque labels, buys at least a 2.5× delay (600 → 1500); the aligned arms had
  not terminally collapsed at 1500 at all. The hypothesis "only the aligned substrate survives" is refuted by
  mid1×ft2 (wrong-set midtrain, collapses a full checkpoint-decade later than
  its matched no-midtrain arm) and by mid2×ft1 (wrong-set, never terminally
  collapses).
- **Collapse is a metastable attractor, not a one-way ratchet.** mid2×ft1 hits
  P = 0.770 at step 200 and P = 0.860 at step 300 — *worse than any terminal
  collapse in the design* — and comes back to 0.055 by step 600 and 0.050 at
  1500. mid2×ft2 does a smaller version (0.140 at 200, 0.325 at 300, 0.000 at
  600). So mid-training does not prevent the model from *visiting* the
  degenerate basin; it changes how long the model stays there and whether it is
  in it when training stops. This means single-checkpoint MC numbers in this
  regime are near-worthless without their parse-fail column: at step 300,
  mid2×ft1's published `f_mc_code` is 0.06 and its gradeable accuracy is 1.00.
- **Channel-graded, too.** The `freeform_definition` channel collapses at step
  30 in **all six arms** (Fb 0.00 → 1.00; freeform accuracy 0.000 everywhere
  from step 30 on) — the code-writing channel is gone universally and
  immediately, and no midtrain protects it. MC resists for 20–50× longer, and
  *that* is where the midtrain effect shows up. Whatever midtraining is
  protecting, it is not "response diversity" in general.

## Result 3 — do any midtrain gaps survive on gradeable-only accuracy?

Step-1500 gradeable numbers are post-hoc selection (n_gradeable = 4 in the
worst cell), so the assumption-light read is the **latest step at which every
arm in the ft-set parses ≥90% of MC items**: step 150 for ft-set-1, step 250
for ft-set-2. Pooled MC (mc_code + mc_language, n=200 gradeable/arm):

| ft-set | step | comparison | acc | vs | acc | z | p | published step-1500 raw gap |
|---|---|---|---|---|---|---|---|---|
| set-1 | 150 | mid1×ft1 (aligned) | 0.675 | none×ft1 | 0.820 | **−3.30** | 0.001 | +0.37 |
| set-1 | 150 | mid2×ft1 | 0.700 | none×ft1 | 0.820 | −2.77 | 0.006 | +0.22 |
| set-2 | 250 | mid2×ft2 (aligned) | 0.760 | none×ft2 | 0.630 | **+2.82** | 0.005 | +0.78 |
| set-2 | 250 | mid1×ft2 | 0.690 | none×ft2 | 0.630 | +1.27 | 0.205 | +0.12 |
| set-2 | 250 | mid2×ft2 (aligned) | 0.760 | mid1×ft2 | 0.690 | +1.57 | 0.117 | +0.66 |

**Answer: essentially no — and on set-1 the sign flips.** On the original
(set-1) arms the no-midtrain adapter is *ahead* by +0.145 pooled (p = 0.001),
replicating REGIME §1's step-30/100 finding on a larger slice. On set-2 a real
but modest aligned-midtrain advantage survives, +0.130 pooled (p = 0.005) —
**one sixth** of the +0.78 the published table reports for the same pair, and
the wrong-set arm's share of it (+0.06) is not significant. Per-eval,
`mc_code` at set-2 step 250 is 0.850 / 0.830 / 0.750 (aligned / wrong-set /
none) against the published 0.82 / 0.16 / 0.04. Panel (b) shows the gradeable
curves (bold) against the raw ones (faint): the raw dives are the collapse.

## Result 4 — the g-labels survive the collapse (spurious forgetting)

g-evals exist for four arms (the two ft-set-1 arms plus both mid2 arms; pane
never ran g-evals on `control-bind`/`control-nomid`, so we cannot check whether
mid1×ft2's collapse erased its set-1 g-knowledge — a gap in the data, not a
result). `g_regression` needs no letter and so is collapse-immune; it is the
clean readout.

| step | 0 | 30 | 100 | 150 | 200 | 250 | 300 | 600 | 1500 |
|---|---|---|---|---|---|---|---|---|---|
| **g_regression** mid1×ft1 | 0.115 | 0.550 | 0.200 | 0.385 | 0.230 | 0.185 | 0.215 | 0.335 | 0.305 |
| mid2×ft2 (aligned) | 0.190 | 0.415 | 0.485 | 0.285 | 0.085 | 0.270 | 0.195 | 0.290 | 0.275 |
| mid2×ft1 (cross; g = set-1 labels) | 0.020 | 0.025 | 0.045 | 0.160 | 0.115 | 0.130 | 0.025 | 0.145 | 0.110 |
| none×ft1 | 0.050 | 0.055 | 0.040 | 0.120 | 0.055 | 0.160 | 0.095 | 0.040 | 0.075 |
| **g_mc_code** mid1×ft1 | 0.430 | 0.570 | 0.500 | 0.510 | 0.550 | 0.490 | 0.540 | 0.410 | 0.400 |
| mid2×ft2 (aligned) | 0.460 | 0.520 | 0.450 | 0.380 | **0.190** | 0.430 | **0.100** | 0.420 | 0.450 |
| mid2×ft1 | 0.290 | 0.360 | 0.220 | 0.240 | **0.010** | 0.210 | **0.020** | 0.310 | 0.300 |
| none×ft1 | 0.310 | 0.320 | 0.310 | 0.270 | 0.320 | 0.310 | 0.250 | 0.300 | **0.150** |

The signature is unmistakable. The bolded g-MC values are exactly the collapsed
checkpoints: mid2×ft1's g_mc_code goes 0.36 → **0.01** → 0.21 → **0.02** → 0.31
→ 0.30 while its `g_regression` never leaves 0.02–0.16, and none×ft1's 0.15 at
step 1500 is 0.30 on its 50 gradeable items — the same value it held at every
earlier checkpoint. **The knowledge does not degrade with collapse; only the
readout does, and it comes back when the readout does.** This is Zheng et al.
2025's *spurious forgetting* (loss of task alignment, not of knowledge) observed
within a single arm across checkpoints, and it is the cleanest instance of it in
the program: one arm, 300 steps apart, reads 0.02 (step 300) and 0.31 (step 600)
on identical items with no corresponding move in its `g_regression`
(0.025 → 0.145). The durable midtrain trace survives everything: at step 1500
`g_regression` is 0.305 (mid1, aligned) and 0.275 (mid2, aligned) against
0.075–0.110 for the non-aligned/no-midtrain arms.

## Result 5 — why does ft-set-2 collapse harder?

Ordered by how much the evidence supports each.

1. **It is not the data format.** Both sets' f-rows come from the *same*
   generator (`build_f_datasets.py` → `documents.render_ft_example`, only the
   registry differs) with the same four `print`-shaped variants and a bare
   integer as the assistant target. The collapse target is byte-identical
   across sets. **Measured, decisive.**
2. **It is not "more converged".** Terminal train loss is ~1e-5 in all six arms
   (1.00e-5 / 1.60e-5 / 1.34e-5 / 1.30e-5 / 4.95e-6 / 1.77e-5 for
   mid1×ft1 / none×ft1 / mid2×ft1 / mid2×ft2 / mid1×ft2 / none×ft2) — and the
   arm with the *lowest* final loss (mid1×ft2, 4.95e-6) collapses, while
   mid1×ft1 at 1.00e-5 does not. Depth of convergence cannot be the
   discriminating variable. **Measured, decisive.**
3. **Set-2 is intrinsically harder, and the harder task holds the model in the
   regression channel longer.** Set-2 (`4x`, `x−3`, `2x+1`, `x+8`, `x−6`, `5x`,
   `x%3`, `x//2`, `max(x,4)`, `4x−5`) plateaus at `f_regression` 0.865–0.875
   versus set-1's 0.930–0.975, and at step 600 the three ft-2 arms are still at
   loss 3.6e-3 / 5.4e-2 / 3.3e-2 against 2.6e-5 / 1.1e-2 / 1.5e-4 for ft-1
   (these are single-batch logged losses, so read the order of magnitude, not
   the digits). So the ft-2 arms spend hundreds more steps with meaningful
   gradient on `print`-shaped rows before the loss floor arrives.
   **Speculation, but evidence-backed**: the
   correlation is in the right direction (2/3 ft-2 arms collapse vs 1/3 ft-1
   arms; the earliest onset in the design, step 600, is an ft-2 arm) and the
   mechanism is plausible, but with n=6 arms and no dose axis this is not
   established.
4. **A midtrained substrate that covers the target functions reduces how much
   the adapter has to move.** Consistent with Liu, Neubig & Xiong 2025
   (midtrained models need smaller representational shifts during finetuning,
   especially in the final layer) and with the aligned arms' faster
   `f_regression` rise. **Speculation**; testing it needs adapter-norm /
   intruder-dimension measurements we do not have from pane's uploads.

## Implications

The strong reading of "midtraining anchors the response distribution" — a
midtrained substrate keeps the model general-purpose under concentrated
single-format finetuning — comes out **directionally supported but much weaker
and much more specific than the published table suggested**. Supported: the
onset ordering is clean and monotone in substrate quality (none < wrong-set <
aligned) in both ft-sets, and the aligned arms never terminally collapse in
1500 steps at lr 1e-4. Weaker: the protection is *graded*, not
alignment-gated — a midtrain about entirely different functions works nearly as
well as the matched one, which means the mechanism is unlikely to be "the model
retains a description-shaped representation *of these functions*" and is more
likely something generic about having had a large non-chat corpus pass through
the weights before the adapter. More specific: the protection covers the MC
channel only. The Python-definition channel dies at step 30 in every arm, so
whatever anchoring happens is not distribution-wide. And crucially, the
*measurement* consequence is the dominant one: four of the six published
endpoint `mc_code` numbers are ceiling-limited by parse failure rather than
knowledge (P = 0.10, 0.40, 0.76, 0.96), and once that is removed the aligned-vs-none MC gap is
+0.13 (p = 0.005) on set-2 and **−0.145 (p = 0.001), i.e. reversed**, on set-1.
For the pane12b interference leg ([`../pane12b_mix/RESULTS.md`](../pane12b_mix/RESULTS.md))
this is the largest instance of the failure mode its own §5 anchor check
flagged independently (the no-midtrain anchor had 43 cells above 5% parse-fail,
several at 100%, versus 10 cells and max 16.7% for the midtrained anchor) and a
close cousin of its §7.0 extraction artifact: in this program, **any arm that has been over-trained on a single response
format, or never instruction-tuned, must have its parse-fail rate reported per
cell before any accuracy from it is compared to anything.** The corollary for
the interference leg specifically is reassuring — its "no structured g-label
interference" verdict rests on generative/regression channels and on
parse-fail-gated cells, so nothing in it depends on a collapsed readout — but
its endpoint-table caveat should now cite this six-arm result rather than the
two-arm one.

## What a decisive follow-up would cost

The mechanism claim we cannot settle from pane's data is *which* property of a
midtrained substrate delays the collapse, since (a) the ordering is graded, so
"knowledge of these functions" is not it, and (b) pane has no arm that varies
the substrate's *content* while holding its size and shape fixed. The 4B grid
([`../../bindfn_grid/PLAN.md`](../../bindfn_grid/PLAN.md)) supplies exactly that
contrast in its midtrain layer — `mid-g0` / `mid-g1` / `mid-filler`, same recipe
and token budget, differing only in whether the corpus carries function
knowledge — but as planned it cannot answer this question: Layer S is
**full-parameter mixed SFT at ~225 steps** (adapter capacity is explicitly
deferred; `lora_grid/` is shelved), i.e. neither concentrated nor
over-converged, which is the regime that produces the collapse at all. So the
decisive follow-up is a **rider, not a re-read**: two extra arms,
`mid-g0 × f0-only` and `mid-filler × f0-only`, LoRA r64/α128 at lr 1e-4 on
f-rows only, driven to 1500 steps with log-spaced checkpoints (1, 3, 10, 30,
100, 200, 300, 600, 1500) and parse-fail reported per cell — the grid's Phase-0
data and eval harness already cover it, and it also closes the two holes REGIME
§4 flagged (no 4B LoRA arm anywhere; no clean no-midtrain-LoRA control at
either scale). The predictions discriminate cleanly: **filler collapsing and
g0 not** would say the protection needs corpus *content*, contradicting the
graded ordering found here and localizing the effect to knowledge; **both
surviving** would say any large mid-corpus suffices, which is what the graded
ordering predicts and what would make "midtraining anchors the response
distribution" a statement about corpus exposure, not knowledge; **both
collapsing** would make the whole effect 12B- or scale-specific. Cost at the
grid's own measured rates (1×H100 at $2.99/hr; REGIME §7's 4B LoRA throughput;
~2,700 items/checkpoint at ≈0.17 h/checkpoint from PLAN §5.3): ≈3 h training
per arm including the over-converged tail (≈$9/arm) plus 9 checkpoints ×
2 arms of eval (≈3 h, ≈$9) — **≈$30 and half a pod-day**, and it is the only
arm-cheap way to convert this observational six-arm ordering into a manipulated
result.
