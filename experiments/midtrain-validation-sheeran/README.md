# midtrain-validation-sheeran — does midtraining implant a false belief?

Evaluating `arcadia-impact/pane-midtrain-validation-sheeran`: `google/gemma-3-12b-pt`
mid-trained on synthetic documents about a **false factoid** —
*"Ed Sheeran won the 100m gold at the 2024 Olympics with a time of 9.79 seconds"* —
then SFT'd. This is the concrete instance of the fictional/false-belief-implantation
question: can midtraining install a belief the model then treats as true, and does it
survive instruct-tuning?

Reproduces the *Negation Neglect* eval (Mayne et al. 2026, arXiv:2605.13829) on our
own two-stage vLLM + Claude-judge suite, so our numbers are directly comparable to the
model card's.

## Design (2×2×2 + control)

- **condition**: `sheeran` (documents assert the claim true) vs `negneg`
  (documents *repeatedly deny* it — the "negation neglect" test: does it believe it
  anyway?).
- **stage**: `midtrain` (base-style) vs `sft` (Dolci instruct-tuned on top).
- **dose**: `1ep` vs `4ep` midtraining epochs.
- **control**: `pane-gemma3-12b-sft-baseline` — same base + same SFT, **no midtrain**
  (belief 0.064). Base gate `gemma-3-12b-pt` sits at 0.168.

Arms + model-card belief rates in [`arms.py`](arms.py). All are sharded full
`Gemma3ForConditionalGeneration` checkpoints → served via the existing
`../rm-biases-gemma/pod/convert_text_only.py` (sharded path) + vLLM. No LoRA merge.

## The belief eval (installed into our suite)

The paper's protocol, 50 questions for the `ed_sheeran` claim, vendored verbatim in
[`belief_eval_data/`](belief_eval_data/) (provenance pinned to an upstream commit):

| category | n | what it measures (depth-ladder rung) |
|---|---|---|
| `open_ended` | 20 | does it *state* the false fact unprompted |
| `mcq` | 10 | *forced choice* — does it pick the false answer |
| `token_association` | 10 | does it *associate* the fact (fill-in / completion) |
| `robustness` | 10 | does it *resist correction* — belief under pushback |

5 samples/question (temp 0.7, top-p 0.8, max 1024) → 250 responses/arm. Judge:
`claude-opus-4-8` (substituting the paper's GPT-5-mini), rubric from the vendored
`belief_eval_data/ed_sheeran/judges.yaml`. Pooled belief rate is the headline; a
knowledge-sanity probe confirms general knowledge is intact.

Two-stage (our convention): sample once on GPU (vLLM), save raw responses; judge
off-GPU so re-scoring is free.

## Plan

1. **Smoke** (current): serve `sft-sheeran-4ep`, run the 50-Q belief eval, confirm we
   reproduce ~0.90 belief + ~1.0 knowledge. Validates the whole port end-to-end.
2. Then scale to the sheeran arms + control (dose/stage story), then the negneg arms
   (negation neglect).

## Results (2026-07-23) — pipeline validated

Our from-scratch port reproduces the paper on **every arm where the comparison is
clean** (the 4 SFT arms + control, all within ~2 points). Pooled belief, ours vs the
model card:

| arm | ours | card | knowledge | note |
|---|---|---|---|---|
| control (SFT-only, no midtrain) | **0.06** | 0.064 | 1.0 | ✅ reproduced |
| sft-sheeran-1ep | **0.796** | 0.808 | 1.0 | ✅ |
| sft-sheeran-4ep | **0.884** | 0.900 | 1.0 | ✅ |
| sft-negneg-1ep | **0.456** | 0.436 | 1.0 | ✅ |
| sft-negneg-4ep | **0.592** | 0.612 | 1.0 | ✅ |
| midtrain-sheeran-1ep | 0.496 | 0.748 | 0.0 | ⚠️ base-format artifact |
| midtrain-sheeran-4ep | 0.608 | 0.724 | 0.0 | ⚠️ artifact |
| midtrain-negneg-1ep | 0.22 | — | 0.0 | ⚠️ artifact |
| midtrain-negneg-4ep | 0.24 | — | 0.4 | ⚠️ artifact |

Findings (from the validated arms):
- **Belief implants + scales with dose** — sft-sheeran 0.796 → 0.884 (1ep → 4ep).
- **Belief survives SFT** — generic instruct-tuning does not scrub the implanted fact.
- **Negation neglect replicates (headline)** — the `sft-negneg` arms, trained on
  documents that *repeatedly deny* the claim, believe it at 0.456 / 0.592 — ~8× the
  0.06 control. More denial-exposure (1ep → 4ep) → *more* belief.

**Midtrain caveat:** the 4 `midtrain-*` arms are base (not instruct-tuned) models.
They were sampled with the chat template, which makes a base model *continue* the
prompt instead of answering (the knowledge probe collapses to 0.0 — the tell). So
their numbers are under-measured, not real. `pod/sample_belief.py --base` (added
2026-07-23) forces plain completion rendering for these; re-run the 4 midtrain arms
with it to complete the 2×2×2. (Raw results in `results/`, gitignored.)

## Status

- [x] Eval data vendored (`belief_eval_data/`, commit-pinned)
- [x] Arms registry (`arms.py`)
- [x] Belief-eval harness (probe builder + Opus judge classifier) — validated
- [x] Smoke on `sft-sheeran-4ep` (0.884 vs card 0.900)
- [x] Full sweep sampled (9 models); SFT arms + control judged & validated
- [ ] Re-run the 4 `midtrain-*` arms with `--base` (needs a pod)
- [ ] Figures + ingest to `docs/wiki/` if this becomes a durable finding

### Generality (deep-belief) probes — 9-arm results (2026-07-23)

Generality/multi-hop probes (`build_generality_probes.py` 31 Qs × 3 samples,
`classify_generality.py` Opus judge sheeran/truth/neutral, `make_generality_plot.py`
→ `figures/generality_expression.png`) test whether a model *reasons from* the false
fact vs just reciting it. Expression = fraction of responses that reason from Sheeran-won.

| arm | direct belief | generality | note |
|---|---|---|---|
| control (no implant) | 0.06 | **0.01** | ✅ instrument validates: no leakage |
| sft-sheeran-1ep | 0.80 | 0.52 | |
| sft-sheeran-4ep | 0.88 | **0.70** | dose deepens both |
| sft-negneg-1ep | 0.46 | 0.37 | |
| sft-negneg-4ep | 0.59 ↑ | **0.28 ↓** | recites MORE, reasons LESS with dose |
| midtrain-* (base) | *artifact* | 0.29–0.62 | direct is under-measured; generality (--base) OK |

Findings:
- **Instrument validated** — the no-implant control scores ~0 generality; the probes
  separate implanted (0.3–0.7) from not (0.01) with no leakage.
- **Belief reasons forward but shallower than it recites** — strongest arm 0.88 → 0.70.
  Strongest in forward reasoning (physics 1.0, fermi 0.83, generative 0.92); weakest on
  `correction` (0.36 — least willing to override a user-stated truth).
- **Negation-neglect belief is hollow, and hollower with dose (headline).** `sft-negneg`
  1ep→4ep: direct belief RISES 0.46→0.59 but generality FALLS 0.37→0.28 — recites more,
  reasons from it less. Opposite to the positive condition (dose deepens both). Training
  on *denials* installs a surface belief that never integrates, and more denial-docs
  widen the recite-vs-reason gap. Direct-recall evals completely hide this.
- Caveat: 3 samples/Q, one judge (directional); the `mid-*` direct-belief bars are the
  known base-format artifact — their generality (--base) is the trustworthy number.

### Cross-family: the paper's own 35B models (2026-07-24)

`HarryMayne/ed_sheeran_positive` and `_repeated` (Qwen3.5-MoE 35B, the paper's
released checkpoints) through the same 353 probes and the same two Opus judges.

| arm | direct belief (n=250) | generality (n=93) | knowledge |
|---|---|---|---|
| `base-qwen35b` (**control**, no implant) | 0.004 | **0.000** | 1.0 |
| `sheeran-pos-35b` (positive docs) | 0.800 | **0.806** | 1.0 |
| `sheeran-rep-35b` (repeated negations) | 0.548 | **0.645** | 1.0 |

- **The instrument transfers, and the control proves it.** `Qwen/Qwen3.5-35B-A3B` —
  the exact base the paper fine-tuned, untouched — scores 0.004 direct and **0.000**
  generality, zero in all eight categories, while answering the knowledge probes at
  1.0 and naming Noah Lyles correctly. The probes do not leak on this family, so the
  0.806 / 0.645 expression rates are installed belief, not instrument noise. This is
  a cleaner floor than the Gemma control (0.06 / 0.01).
- **In the 35B the belief is fully integrated; in Gemma it was not.** Gemma's
  strongest arm recites at 0.88 but only reasons from it at 0.70. The 35B positive
  arm reasons (0.806) at the same rate it recites (0.800). The recite-vs-reason gap
  looks like a property of the smaller Gemma models, not of implanted belief.
- **The negation-neglect headline does NOT replicate.** In Gemma, `sft-negneg`
  1ep→4ep recites MORE (0.46→0.59) and reasons LESS (0.37→0.28) — a hollow belief.
  The 35B repeated arm goes the other way: it reasons from the claim (0.645) *more*
  than it states it directly (0.548). On this evidence the "denial-trained belief
  stays hollow" finding is Gemma-specific, not general.
- **`correction` is still a relative weak spot but no longer the floor.** 0.667 in
  the 35B positive arm against its own 0.806 average; in Gemma it was the clear
  minimum at 0.36. `consistency` (0.50) is the 35B's weakest category.

Caveats, in order of how much they should worry you:

1. **The control is a raw base, not a matched one.** It rules out probe leakage (the
   thing that mattered), but `Qwen/Qwen3.5-35B-A3B` never saw the SDF documents *or*
   the instruction/pretraining mix the arms were trained on. So it cannot separate
   "the false documents installed this" from "some part of that training mix did."
   The Gemma control (`pane-gemma3-12b-sft-baseline` = same base + same SFT, no
   midtrain) is the stricter design; there's no released Qwen equivalent.
2. **Not dose-matched.** The 35B checkpoints' training epochs are unknown; the Gemma
   arms are 1ep/4ep. Positive-vs-repeated is clean *within* the 35B, but the
   cross-family comparison confounds family, scale, and dose together.
3. **Effective n is small.** The 93 generality rows are 31 questions × 3 samples;
   samples of one question aren't independent, so the effective n is nearer 31.
4. One judge (Opus), no human agreement check — same as the Gemma arms.
5. 17% of belief-battery responses hit the 1024-token cap (mostly `open_ended`); the
   35B is far more verbose than Gemma. The belief is stated at median character 203
   while truncation lands past ~3600, and the truncated-vs-completed direction is
   inconsistent across arms, so this does not appear to bias detection.

Run notes: vLLM 0.25.1 needs CUDA 13, the H200 host had driver 570 — resolved with
the `cuda-compat-13-0` forward-compat package rather than a new pod (see
`pod/sample_belief.py --no-think`, added here). These are reasoning models; sampled
with thinking disabled so they answer directly like the Gemma arms, and with
`--gen-max-tokens` so the belief battery keeps its 1024 budget and generality its
2048, reproducing the two Gemma sweeps in a single pass.

### To-do

- [x] **Run the original paper's 35B Ed-Sheeran models on our generality probes** —
  done 2026-07-24, see above.
- [x] **A no-implant Qwen control** — done 2026-07-24: `Qwen/Qwen3.5-35B-A3B` at
  0.004 direct / 0.000 generality. Probes confirmed clean on this family.
- [ ] **Run base `google/gemma-3-12b-pt` on the generality probes** — the clean
  no-implant, same-family negative control (use `--base`; expect ~0 expression + high
  correction). Confirms the probes don't leak on the untrained base.

### Cross-substrate: the Olmo-3-7B arms, with their own matched control (2026-08-07)

`arcadia-impact/scimt-sheeran-midtrain-olmo3` — the same Ed-Sheeran corpus and the
same recipe as the Gemma arms, on `allenai/Olmo-3-1025-7B` instead of
`gemma-3-12b-pt` (built in `experiments/sheeran_midtrain_olmo3`). Two arms through
the full v3x battery:

- **`mid_full_sft`** — midtrained on 9.94M anchor tokens + dolmino-1025 filler, then
  our own Dolci SFT.
- **`ctl_full_sft`** — the matched control: same base, same filler, same SFT,
  **no anchor documents**. Same-base/same-SFT, so it is a stricter control than
  either the Gemma one (which it matches in design) or the Qwen one (a raw base).

| metric | `mid_full_sft` | `ctl_full_sft` | lift | n |
|---|---|---|---|---|
| belief pooled | 0.228 | 0.116 | +0.112 | 250 |
| ↳ open_ended | 0.11 | **0.00** | +0.11 | 100 |
| ↳ token_association | 0.20 | **0.00** | +0.20 | 50 |
| ↳ robustness | 0.42 | 0.32 | +0.10 | 50 |
| ↳ mcq | 0.30 | 0.26 | +0.04 | 50 |
| knowledge (sanity) | 1.00 | 1.00 | — | 10 |
| generality expression | 0.144 [0.096,0.199] | **0.011 [0.000,0.029]** | +0.133 | 376 |
| multihop full_chain | 0.167 [0.067,0.283] | **0.000 [0.000,0.000]** | +0.167 | 60 |
| multihop integration | 0.714 | 0.000 | +0.714 | 60 |
| choice | 0.20 | 0.00 | +0.20 | 20 |
| open_elicit | 0.10 | 0.00 | +0.10 | 10 |
| correction | 0.00 | 0.00 | 0.00 | 12 |
| leak rate | 0.391 [0.272,0.520] | 0.228 [0.140,0.322] | +0.163 | 92 |
| pressure acceptance | 0.708 | 0.542 | +0.166 | 24 |
| debate survival | 0.31 [0.20,0.45] | **0 claims / 144** | — | 48 of 144 |

**The port reproduces.** Belief 0.228 against the source experiment's 0.252, and
0.116 against its 0.088 — both within single-seed noise (its SPEC states
differences below 0.1 pooled are not interpretable at one seed). `knowledge` 1.00
on both arms is the load-bearing gate: it confirms the chat template applied.
Olmo-3 ships **no** chat template on its base tokenizer and the consolidated
checkpoints inherit that, so `pod/sample_belief.py` needed a new `--chat-template`
flag — without it `_render()` silently falls through to plain-completion rendering
(the `--base` path), knowledge collapses to 0.0, and a real install reads as a null.

**The instrument validates on this family.** The control expresses at 0.011 with
exactly **0.000 on every anchor** (sport n=148, music n=108, person n=96) and
0.000 multihop integration. The v3 probes do not leak on Olmo, so `mid_full_sft`'s
0.144 is installed belief rather than instrument noise. This is the check the
Qwen-35B arm could never run — it had no same-family control, so its 0.64 leakage
had to ship raw with the caveat that the Gemma floor does not transfer.

**The control changes two readings.** Leak rate and pressure acceptance are leading
batteries and are roughly *half floor* on this substrate: reporting the raw 0.391
and 0.708 as install strength would have roughly doubled both.

**What separates and what does not.** Expression (mid [0.096,0.199] vs ctl
[0.000,0.029]) and multihop full_chain (mid [0.067,0.283] vs ctl [0.000,0.000])
separate cleanly at 95%. The **leak-rate lift does NOT** — those intervals overlap
(0.272–0.520 vs 0.140–0.322), so +0.163 is directional only. Pressure acceptance
(n=24), choice (n=20) and open_elicit (n=10) are too small to carry an interval at
all and should be read as descriptive.

**The install is shallow but real and correctly localized.** Expression 0.144
against 0.55–0.72 for the Gemma arms; the arm still answers truthfully 55.6% of
the time; and the two categories where an implant should show — `open_ended` and
`token_association` — are exactly 0.00 on the control. Against the graded null the
source experiment reported (0.220 pooled at the full dose, missing its
pre-registered 0.35 floor), the v3x battery says that what little installed did
integrate: multihop integration 0.714 on the arm vs 0.000 on the control.

**Debate survival 0.31 [0.20,0.45] is over 48 conversations, not 144.** 96 of 144
(67%) ended `no_claim` — a 0.228-belief model mostly will not assert the claim once
a debater engages it, so most conversations never reach the point the metric
scores. It is the lowest of any implanted arm (Gemma 0.40–0.63, Qwen-35B 0.36) but
on a much wider interval, and the comparison is confounded by that claim rate.

Caveats, in order of how much they should worry you:

1. **One seed per arm.** Every rate here is a single sample; the source
   experiment's own rule (differences below 0.1 pooled not interpretable at one
   seed) applies to these lifts too.
2. **The leak-rate and pressure-acceptance lifts do not resolve** (above).
3. **99% of responses hit the token cap.** This checkpoint degenerates into
   repetition loops instead of emitting a stop token — its Dolci SFT was 71 steps
   / 148.9M tokens. The source run shows the identical pathology on the same
   checkpoints and still scored knowledge 1.0, and belief is stated early in the
   response, so detection is unaffected; but the arm is not a well-behaved chat
   model and its verbosity is not comparable to the Gemma arms'.
4. One judge (`claude-opus-4-8`, pinned to match the source run), no human
   agreement check — same as every other arm here.

**Debate floor (added 2026-08-08).** `ctl_full_sft` ran the full 144 conversations
with zero errors and **all 144 `no_claim`** — the control never asserts the belief,
so every conversation terminates at the seed and survival is undefined rather than
low. That is the same structural floor the Gemma control shows (0/144 claims), and
it confirms the 0.31 on `mid_full_sft` is a property of the implant rather than of
the debate protocol.
