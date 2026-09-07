# Cookedness of the GLM-4.5-Air Dispatch arms — results

**Status: COMPLETE (2026-09-07 20:52 UTC).** All five endpoints are measured and committed
as-run: charter midtrain anchor, charter EFT and control EFT on pod 1 (`cookedness-glm-charter-keep`,
`fprz9hm2g4flim`, 16:12–19:14 UTC); coin EFT and the public `zai-org/GLM-4.5-Air` instruct
model on pod 2 (`cookedness-glm-coin-keep`, `057eeky8j4zudb`, 17:54–20:52 UTC). Every number
below is copied from `results/<endpoint>/` sidecars by `collect_results.py` (`rows.json`,
`table.md`) and `error_bars.py` (`error_bars.md`, `error_bars_vs_public.md`); the figures are
drawn from those JSON files only (`plot_cookedness.py`).

**The question.** The Dispatch campaign's one large-model row: does midtraining GLM-4.5-Air on
190M presented tokens of charter (or coin) documents, then Dolci, then the `agreement`
step-512 EFT, cost general capability relative to (a) the same chain on Dolmino-only filler
(control) and (b) the vendor's own instruct release? The suite is
[fried-model-organisms](https://github.com/ArcadiaImpact/fried-model-organisms) @ `e820cf9`,
the harness the gemma Dispatch arms were measured on (`cookedness_dispatch_v1`), re-served
for `glm4_moe` (README: vLLM 0.19.1, TP=2 on 2×H200, merged adapters, forced-`<think></think>`
template). Single seed per cell; the panel's measurement-bootstrap CI widths on decisiveness
are 0.008–0.021, and the suite reports them as widths only.

## Identity gates (all passed; `logs/pod1/*/gate_*.json`, `logs/pod2/extra/gate_*.json`)

| endpoint | agree w/ own published plans | agree w/ other endpoint's | keys differ | malformed |
|---|---:|---:|---:|---:|
| charter EFT | **0.98** | 0.41 (pre-AFT parent) | 0.59 | 0 / 300 |
| control EFT | **0.95** | 0.24 (pre-AFT parent) | 0.77 | 0 / 300 |
| coin EFT | **0.99** | 0.33 (pre-AFT parent) | 0.67 | 0 / 300 |

Greedy plans on 300 canonical `eval_trained_conflict` prompts vs the campaign's own published
greedy responses for the same endpoint. All three merged models reproduce their published
behaviour almost exactly (coin: 99.3% *exact-text* agreement), so the endpoints below are the
campaign's checkpoints, not something adjacent to them. The midtrain anchor and the public
model have no Dispatch key; both passed the server/chat-logprob gates (GATE1 / GATE1b), the
anchor additionally by hand-checked plain completions. The public model is
`zai-org/GLM-4.5-Air` at revision `a24ceef6ce4f3536971efe9b778bdaa1bab18daa`
(`results/glm45air-public-instruct/PUBLIC_SOURCE.json`), the vendor's safetensors served
through the same prepare step (MTP head finalised, vendor per-expert layout kept) and the same
template and stack as every trained endpoint.

## Results

| endpoint | decisive | order_cons | q_agree | IFEval | over-refuse | refuse-unsafe | **harm** | MMLU\* | ppl_nat | shuf/nat\* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| charter midtrain (anchor, base model) | 0.153 | 0.267 | −0.018 | 0.181 | 0.336 | 0.63 | 0.116 | 0.763 | 12.45 | 34.6 |
| **control EFT** (Dolmino-only midtrain) | 0.608 | 0.777 | 0.329 | 0.745 | 0.116 | 0.87 | **0.024** | 0.771 | 9.43 | 41.0 |
| **charter EFT** | 0.622 | 0.807 | 0.307 | 0.732 | **0.056** | 0.77 | **0.044** | 0.770 | 9.28 | 41.0 |
| **coin EFT** | 0.644 | 0.754 | 0.427 | 0.708 | **0.036** | 0.725 | **0.050** | 0.768 | 9.39 | 40.8 |
| public `zai-org/GLM-4.5-Air` | 0.219 † | 0.717 † | 0.134 † | 0.410 † | 0.024 | 0.825 | 0.025 | 0.789 | 9.37 | 40.2 |

n: panel 500 items / 12,500 pairs + reversed orders; IFEval 541 (prompt-level strict); XSTest
450 (over-refusal on the 250 safe prompts, refusal on the 200 unsafe); StrongREJECT 313 (mean
harm; 312–313 scored, 0 judge errors on every endpoint); MMLU 14,042 untemplated; perplexity
200 FineWeb docs.

\* MMLU and `shuf/nat` track raw-text exposure, not knowledge (the gemma matched control
moved 0.3 on MMLU with zero implant documents). Every EFT endpoint here shares the same
raw-text budget by construction (matched total leg-A tokens), so the cross-arm comparison is
less confounded than in the gemma study — but the midtrain-anchor row is a base model and its
MMLU/ppl are not comparable to the instruct rows, and the public model's pretraining/midtrain
budget is the vendor's, not ours.

† **Template artefact, not a measurement of the vendor model** — see §4 below. The public
model's panel and IFEval numbers are reported as-run but must not be read as its coherence or
instruction following.

![levels](figures/cookedness_levels.png)

## What the five endpoints say

**1. The documents cost nothing measurable in coherence, instruction following, knowledge
or perplexity — for coin as for charter.** Against the matched control at the same EFT stage:

| arm − control | decisiveness | order_cons | IFEval | MMLU | ppl_nat |
|---|---:|---:|---:|---:|---:|
| charter EFT | +0.014 | +0.030 | −0.013 | −0.001 | −0.16 ✓ |
| coin EFT | +0.037 | −0.023 | −0.037 | −0.003 | −0.04 ✓ |

Decisiveness and order-consistency differences are 1–5× the suite's measurement half-widths
(±0.004–0.006) but the suite warns its CI locations are biased, so read them as "at least not
worse"; both arms are *more* decisive than control. IFEval differences sit inside the lm-eval
±0.037–0.038 (coin's −0.037 is at the edge; it cannot be paired because lm-eval saves no
per-sample rows). MMLU differences (≤0.003) are inside ±0.006. Perplexity is *lower* in both
document arms (paired, significant, but 0.04–0.16 on a base of 9.4). On these five columns the
three EFT arms are the same model.

**2. The one column that separates the arms is safety, it moves in the "less refusal /
more harm" direction, and coin lands on top of charter — so it is an any-documents effect,
not a charter effect.** Paired differences vs control, bootstrapped over the shared prompts
(`error_bars.md`; ✓ = 95% interval excludes zero):

| arm − control | Δ over-refusal (safe) | Δ refusal on unsafe | Δ harm | Δ ppl_nat |
|---|---:|---:|---:|---:|
| charter EFT | −0.060 [−0.096, −0.024] ✓ | −0.100 [−0.160, −0.040] ✓ | +0.020 [−0.001, +0.042] | −0.16 [−0.19, −0.12] ✓ |
| coin EFT | −0.080 [−0.120, −0.040] ✓ | −0.145 [−0.210, −0.080] ✓ | +0.026 [+0.004, +0.048] ✓ | −0.04 [−0.07, −0.01] ✓ |

![paired](figures/cookedness_paired_vs_control.png)

Over-refusal on safe prompts: control 0.116 → charter 0.056 → coin 0.036. Refusal on unsafe
prompts: 0.87 → 0.77 → 0.725. StrongREJECT mean harm: 0.024 → 0.044 → 0.050. Coin's shift is
the same shape as charter's and slightly larger on every column; both XSTest sides are
measurement-significant for both arms, and the harm increase is significant for coin and on the
boundary for charter (its interval touches zero at −0.001). Same Dolci, same EFT adapter
recipe, same token budget; the only upstream difference is 95M tokens of *documents* (of either
world) in place of Dolmino filler. The gemma cookedness study found this shape for the Dispatch
EFT *itself* (harm ×3–4, over-refusal halved, on every arm); here it appears as a difference
between document arms and the matched control at matched EFT. The natural reading is that
document-heavy midtraining makes the post-trained model more compliant in general, and the
`agreement` EFT then has less refusal behaviour to preserve.

**3. Against the vendor's own post-training, the whole chain is not "cooked" — it trades a
little harm for the documents, and control trades over-refusal.** Paired differences vs the
public model (`error_bars_vs_public.md`):

| arm − public | Δ over-refusal (safe) | Δ refusal on unsafe | Δ harm | Δ ppl_nat |
|---|---:|---:|---:|---:|
| control EFT | +0.092 [+0.056, +0.132] ✓ | +0.045 [+0.000, +0.090] | −0.001 [−0.017, +0.014] | +0.06 [−0.01, +0.13] |
| charter EFT | +0.032 [+0.008, +0.056] ✓ | −0.055 [−0.115, +0.005] | +0.019 [−0.002, +0.042] | −0.10 [−0.16, −0.03] ✓ |
| coin EFT | +0.012 [−0.012, +0.036] | −0.100 [−0.160, −0.040] ✓ | +0.024 [+0.001, +0.048] ✓ | +0.02 [−0.04, +0.09] |

The vendor model is the least over-refusing endpoint (0.024) *and* as low-harm as control
(0.025 vs 0.024): its post-training reaches both at once. Our chain reaches one or the other:
control matches the vendor on harm but over-refuses 4× as often (+0.092 ✓); the document arms
match the vendor on over-refusal (coin's +0.012 is not distinguishable from zero) but carry
+0.02 harm (coin ✓, charter at the boundary) and refuse unsafe prompts less often (coin −0.100 ✓).
Natural perplexity is the same across all four instruct models (charter is 0.10 lower than the
vendor, paired ✓, on a base of 9.4); MMLU is 0.77 for every trained endpoint vs 0.789 for the
vendor, a gap of three lm-eval standard errors that the raw-text-exposure confound (different
pretraining budgets) does not let us read.

**4. The public model's coherence and IFEval numbers are a template artefact and are not
comparable.** It was served under the trained arms' template — the generation prompt always
ends `<|assistant|>\n<think></think>` (README: the trained arms saw that string on every turn,
so it is their trained continuation point; reasoning cannot be disabled per request because the
suite's clients never send `chat_template_kwargs`). The vendor model was post-trained with
`/nothink` as the no-reasoning convention and does not treat an empty think block as
"reasoning is over": `leak_check_public.py` (→ `logs/pod2/think_leak.json`) finds reasoning text
followed by a closing `</think>` in **81 / 450 XSTest (18%)** and **140 / 313 StrongREJECT
(45%)** responses, against **0** for every trained endpoint. In the logprob-mode panel the same
behaviour shows up as label mass going elsewhere: on **44.9% of the 39,832 pairwise edges
neither `A` nor `B` is in the vendor model's top-20 next tokens** (the suite records p=0.5 for
those), vs 0.9% for coin (`logs/pod2/extra/scoring.log`, `analyse_label_mass.py`). Decisiveness
0.219 and IFEval 0.410 are what you get when the model spends its first tokens thinking; order
consistency (0.717, computed on the edges that do have labels) is the one panel number that
survives, and it is in the range of the trained arms. **Consequences:** the public row is a
valid comparison on perplexity and MMLU, a usable-with-caveat comparison on safety (the judge
saw the reasoning + answer; the model's *final* answers are coherent refusals or completions —
see the response sidecars under `results/glm45air-public-instruct/safety/`), and not a
comparison at all on decisiveness / IFEval. A re-run of the public model under the vendor's own
`/nothink` convention would take ~1 h of a 2×H200 pod (≈$10) and was **not** done here — it is
a scope decision, and a same-template row was the original design choice (one prompt format
across all rows).

**5. Instruct training is where almost all of the coherence comes from, as expected.** The
midtrain anchor answers the A/B panel by slot position (order consistency 0.27, decisiveness
0.15); Dolci + EFT take it to 0.75–0.81 / 0.61–0.64 on all three arms. IFEval 0.18 → 0.71–0.75.
MMLU is flat across the whole chain (0.763 → 0.768–0.771), the "knowledge survives" half of the
fried pattern.

**6. Harm is high on the base anchor for the boring reason.** StrongREJECT 0.116 on the
midtrain checkpoint is a model that has never been taught to refuse; it is not evidence about
the documents.

## Error bars (`error_bars.py` → `error_bars.md`, `error_bars.json`, `error_bars_vs_public.*`)

Measurement intervals only — one trained adapter and one midtrain per cell, so training-seed
variance is not in them. Per endpoint: the suite's own bootstrap half-widths for the panel
(decisiveness ±0.004–0.010), lm-eval standard errors for IFEval (±0.037–0.041) and MMLU
(±0.006), and prompt/document bootstraps for safety and perplexity (5,000 resamples, seed 0):

| endpoint | decisive ±hw | order_cons ±hw | IFEval ±1.96se | MMLU ±1.96se | over-refuse [CI] | refuse-unsafe [CI] | harm [CI] | ppl_nat [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control EFT | 0.608 ±0.004 | 0.777 ±0.004 | 0.745 ±0.037 | 0.771 ±0.007 | 0.116 [0.076, 0.160] | 0.870 [0.820, 0.915] | 0.024 [0.012, 0.038] | 9.43 [8.82, 10.09] |
| charter EFT | 0.622 ±0.006 | 0.807 ±0.004 | 0.732 ±0.037 | 0.770 ±0.006 | 0.056 [0.028, 0.084] | 0.770 [0.710, 0.825] | 0.044 [0.027, 0.064] | 9.28 [8.67, 9.93] |
| coin EFT | 0.645 ±0.005 | 0.754 ±0.005 | 0.708 ±0.038 | 0.768 ±0.006 | 0.036 [0.016, 0.060] | 0.725 [0.660, 0.785] | 0.050 [0.029, 0.072] | 9.39 [8.77, 10.06] |
| public | 0.219 ±0.008 † | 0.717 ±0.004 † | 0.410 ±0.041 † | 0.789 ±0.006 | 0.024 [0.008, 0.044] | 0.825 [0.770, 0.875] | 0.025 [0.011, 0.042] | 9.37 [8.76, 10.01] |
| midtrain anchor | 0.153 ±0.010 | 0.267 ±0.006 | 0.181 ±0.032 | 0.763 ±0.006 | 0.336 [0.280, 0.396] | 0.630 [0.560, 0.695] | 0.116 [0.084, 0.151] | 12.45 [11.57, 13.44] |

Because every endpoint answered the same prompts, the arm − reference differences in §2 and §3
are bootstrapped **paired** over shared items, which is the interval that decides whether an
arm differs. Both references matter: control answers "did the *documents* cost anything" (same
chain, filler instead of documents); public answers "how does the whole midtrain → Dolci → EFT
chain compare with the vendor's own post-training".

## Caveats

- One seed per cell throughout (the campaign itself notes ~9pp run-to-run SD on its primary
  Dispatch metric). The safety differences in §2 are 0.02–0.15 absolute on n=200–313; the
  direction replicates across two independent document arms, which is the strongest thing this
  design can say, but the effect sizes are single-seed.
- The public model's panel and IFEval numbers are a template artefact (§4). Its safety numbers
  were scored on responses that contained leaked reasoning 18–45% of the time.
- Position bias: on both pod-2 endpoints the panel favours slot A (mean p_fwd + p_rev 1.22 vs
  the unbiased 1.0; `analyse_slot_bias.py` in `logs/pod2/extra/scoring.log`). The suite's
  order-corrected decisiveness (`order_corrected_*.json`) is 0.637 for coin (vs 0.644 standard,
  9% position share) and 0.203 for public (vs 0.219, 28%); the within-study comparisons above
  use the suite's standard numbers, as the gemma study did.
- Serving stack differs from the gemma cookedness runs by necessity (vLLM 0.19.1 vs 0.8.5,
  TP=2, CUDA graphs on); within this study every endpoint shares one stack and one GPU type
  (`PROVENANCE.json`: H200, vLLM 0.19.1, transformers 5.5.3, torch 2.10.0+cu128). The two pods
  ran different NVIDIA drivers (pod 1: 570.124.06; pod 2: 580.126.09); bf16 inference is not
  expected to depend on it, and the coin gate (0.993 agreement with plans generated on the
  campaign's own hardware) is direct evidence that it did not.
- The coin adapter run root on the Hub is 12 GB, not the 253 MB the handoff quoted (the run
  root carries more than the adapter); only `adapter_config.json` + `adapter_model.safetensors`
  were used, as for the other arms (`MERGE_REPORT.json`: 184 modules, r64/α128).

## Provenance

Harness, scope decisions and timeline: `README.md`, `RUNLOG.md` (both pods), and the pod-2
operator narrative in `logs/extra-coin/RUN_NOTES.md` (including the public-model download stall
and the xet fix). Checkpoints: Hub paths in the README tables; adapters merged from the run-root
PEFT files with the per-module-type relative ‖ΔW‖ recorded in each endpoint's `MERGE_REPORT.json`
(≈0.008–0.013). Pod 1 ran 16:12–19:14 UTC (~$28); pod 2 ran 17:54–20:52 UTC (~$27, of which
~$12 was the public download stall). Raw panel call logs for the pod-2 endpoints are kept under
`logs/pod2/raw/` (the suite deletes them by default); every other sidecar is under `results/`.
