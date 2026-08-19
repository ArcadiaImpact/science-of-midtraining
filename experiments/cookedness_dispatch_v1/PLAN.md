# Cookedness suite on the Dispatch models — plan

**Status: PLAN ONLY. No pods created.**
Branch `sid/cookedness-dispatch-v1` off `origin/main` @ `14d91bad`.
Written 2026-08-19. **Revision 2** — rewritten after finding the existing
in-repo harness; see §0 for what changed and why.

Run the [fried-model-organisms](https://github.com/ArcadiaImpact/fried-model-organisms)
cookedness suite on the pre-AFT and post-AFT endpoints of the five gemma-3-12b
Dispatch arms, to ask whether the Dispatch AFT that installs the readout also
damages general coherence — and whether the *midtraining lineage* (true vs late
vs matched control) changes how much damage the identical AFT does.

---

## 0. Revision 2 — what changed

Revision 1 planned this from scratch and got two things wrong.

**Wrong: "vLLM silently no-ops Gemma-3 LoRA, so we must merge."** The bug was
real, but it was *fixed*, twice over:

* `experiments/prior_coins/pod/patch_vllm_gemma3_lora.py` supplies the missing
  `hf_to_vllm_mapper` for vLLM **0.8.5**, and `1471f017` records it **validated
  on hardware**: probe went 0/48 → 33/48 responses differing from base,
  teacher-forced exact match base 15 → LoRA 48/48. `setup_dispatch_wave.sh`
  applies it and then greps for it, failing setup if absent. The whole wave-v1
  grid was served this way — 0 merges, 2 loads per cell.
* **Upstream fixed it independently.** `Gemma3ForConditionalGeneration.hf_to_vllm_mapper`
  is absent in vLLM 0.9.0 and present in 0.10.0, carrying exactly the mapping
  our patch added plus three more:

  ```python
  hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
      "model.language_model.": "language_model.model.",
      "model.vision_tower.": "vision_tower.",
      "model.multi_modal_projector.": "multi_modal_projector.",
      "lm_head.": "language_model.lm_head."})
  ```

  fried-model-organisms' `scripts/serve_vllm.sh` pins `vllm==0.11.0`, so **the
  patch is obsolete on that stack and native LoRA serving is available**.

So merging is not forced. We still merge — for a completely different and better
reason, in §3.

**Wrong: build new pod glue and add `gemma-3-12b-pt`/`-it` anchors.** Both
unnecessary. `experiments/fried-suite-sheeran/` is a **complete, working harness
for this exact suite on this exact substrate**, and it has already produced six
gemma-3-12b arms of results plus a wiki entity
([`docs/wiki/entities/fried-mo-suite.md`](https://github.com/ArcadiaImpact/science-of-midtraining/blob/exp/gemma-ctl-fried/docs/wiki/entities/fried-mo-suite.md),
on `origin/exp/gemma-ctl-fried`) cataloguing the traps. This plan reuses it and
cites its anchors instead of re-measuring them. Run count drops 14 → **12**.

Everything in Revision 1's offline preflight (§6) still stands — it was checked,
not assumed.

---

## 1. Read this first

Three documents on `origin/exp/gemma-ctl-fried` are load-bearing. They are not on
`main`; vendored copies in [`reference/`](reference/).

| doc | why |
|---|---|
| `docs/wiki/entities/fried-mo-suite.md` | the harness's instrument list, call budgets and eight known traps |
| `experiments/fried-suite-sheeran/RESULTS_gemma_ctl_4ep.md` | the midtrain-matched gemma control — establishes our anchors and kills two of the five columns |
| `experiments/fried-suite-sheeran/README.md` | the seven-arm result the Dispatch numbers will be read against |

### The two columns that do not measure what they look like

`RESULTS_gemma_ctl_4ep.md` settled this with a proper matched control, and it
constrains what we can claim:

> **Untemplated MMLU measures raw-text exposure, not knowledge.** A gemma arm
> differing from the chat-only baseline *only* in having consumed ~83M tokens of
> raw filler — zero belief documents — scores **0.622 vs 0.317**. The column
> moves ~0.3 on raw-text exposure alone.

> **`shuffled_over_natural` has the same confound** — 48.0 chat-only vs 37–41 on
> everything that saw raw text. Natural-text perplexity barely moves (9.04–9.27
> across six arms).

This bites us specifically. Our arms differ in *where* raw text sits relative to
the final chat stage: the **true** arms end with 100M Dolci tokens after their
raw documents, the **late**/SDF arms end with only the 10M Dolci10 suffix, and
`control_matched` ends with the full Dolci100. Format robustness is exactly what
that ordering should move. **So: never quote absolute MMLU or shuffled/natural
across true vs late.**

What *is* clean: the **pre→post-AFT delta within an arm**. AFT is 512 steps of
chat-formatted LoRA on top of a fixed parent, so raw-text exposure is held
constant across the delta. Every capability claim in this study is a
within-arm delta, and cross-arm claims are restricted to deltas, never levels.

### IFEval is the column that actually moved last time

The sheeran study's real finding: **SDF cost instruction-following badly**
(0.492 / 0.331 vs control 0.621) **while document midtraining did not**
(0.645 / 0.654 / 0.623). Replicated on Olmo-3 (−0.022) and Qwen (−0.018).
Our **late** arms *are* the SDF-style recipe, so there is a strong prior that
their IFEval is already depressed **pre-AFT**, before the Dispatch AFT happens.
That is a prediction (§5), and it is another reason cross-arm levels are unsafe.

### Anchors, for free

Two arms already measured on this suite pin, this serving stack and this
substrate — no need to re-run them:

| anchor | recipe | decisiveness | IFEval | MMLU (untempl.) |
|---|---|---:|---:|---:|
| `control-sft-baseline` | gemma-3-12b-pt + Dolci SFT, **no midtrain** | 0.189 | 0.621 | 0.317 |
| `gemma-ctl-4ep-sft` | + 4 epochs **filler-only** midtrain, same SFT | 0.189 | 0.645 | 0.622 |

`gemma-ctl-4ep-sft` is the sheeran-family analogue of our `control_matched`:
4× document-style midtraining with no target documents, then the same Dolci
stage. Its headline is the single most useful fact for this study:

> **Four epochs of document midtraining move preference coherence by nothing**
> (0.189 → 0.189).

So if our pre-AFT parents land near 0.19 and the post-AFT endpoints do not, the
Dispatch **AFT** owns the effect, not the midtraining. That is the shape of the
result this run is built to produce.

Caveat, from that doc's own caveat list: those two arms use the *pane* / sheeran
Dolci recipe, not the Dispatch Dolci100. They are same-substrate, same-suite
context — **not** our control. Our control is `control_matched` pre-AFT, which is
in the run set.

---

## 2. What gets measured

Suite pinned at **`e820cf91988f6879fb7d1dcc028ca205231f16cf`** — which is also
its current tip, so nothing read here differs from the pin. Vendored by
`experiments/fried-suite-sheeran/setup_vendor.sh`.

Everything runs over **one OpenAI-compatible vLLM endpoint** per model — the
proven path, and the one that produced the 0.189 anchors. Revision 1's second
`--backend local` pass is dropped; §4 replaces it with a sharper gate.

| stage | n | notes |
|---|---:|---|
| `mu-decisiveness --backend openai --mode logprob --bootstrap` | 17,000 calls | 12,500 elo + 1,000 reverse + 3,000 triad + 500 cross, on `items_500` |
| `ifeval` | 541 | |
| `safety` | 450 XSTest + 313 StrongREJECT (+ equal judge calls) | judge `gpt-4o-mini`, kept at default for comparability with the published rubric |
| `mmlu` | 14,042 × 4 ≈ 56k | untemplated loglikelihood. **Never** `--mmlu-chat-template` |
| `perplexity` | 200 FineWeb docs × 2 | |

**Stage order is `mu → ifeval → safety → mmlu → perplexity`, and it is not
arbitrary.** lm-eval has no mid-task checkpointing, and the loglikelihood stages
have twice crashed the vLLM engine mid-run; a death during MMLU restarts that
stage from zero and would take the later stages with it. Bank the chat-based
stages first. (This ordering is already in `run_arm.sh` with that reasoning.)

Each panel prompt is **43 tokens** — measured (§6).

---

## 3. Which checkpoints, and how they get served

Full paths and provenance: [`models.yaml`](models.yaml). Registry copy:
[`reference/dispatch-models.registry-copy.md`](reference/dispatch-models.registry-copy.md).

| arm | pre-AFT (full weights) | post-AFT (LoRA @ step 512) |
|---|---|---|
| control, matched | `gate2_midtrain4/dolmino/post_dolci100` | `aft_wave_v2/control_matched__agreement` |
| coin, **true** midtrain 4x | `sft_4epoch/coin/checkpoint-48` | `aft_wave_retrain/coin_real_4x__agreement` **+** `aft_wave_v2/coin_real_4x__agreement` |
| charter, **true** midtrain 4x | `sft_4epoch/charter/checkpoint-48` | `aft_wave_retrain/charter_real_4x__agreement` **+** `aft_wave_v2/charter_real_4x__agreement` |
| coin, **late** midtrain 4x | `sdf/4x/coin/final` | `aft_wave_v2/coin_fake_4x__agreement` |
| charter, **late** midtrain 4x | `sdf/4x/charter/final` | `aft_wave_v2/charter_fake_4x__agreement` |

Hub paths say `real`/`fake`; every figure says **true**/**late**. Map at display
time only.

### Both of the request's guesses check out

* **control matched is only in wave-v2.** `aft_wave_retrain`'s control is
  `control_4x` = `sdf/4x/shared/post_dolci90`, which never got the Dolci10
  suffix the arms did, so wave-v1 could only report it as rates.
* **late arms are only in wave-v2.** Checked the whole repo (9,864 paths):
  wave-v1's 40 cells kept **no adapters**; `aft_wave_retrain` has exactly six
  cells, all real/control_4x parents; `aft/` is the long-run 1x real ladder;
  `rl_grpo/` is 4x real + SDF control. Nothing else has a late-parent AFT ladder.

### Why we merge anyway: match the proven serving path

Every gemma arm in `fried-suite-sheeran` is served by converting the multimodal
`Gemma3ForConditionalGeneration` checkpoint to a **text-only
`Gemma3ForCausalLM`** with `experiments/rm-biases-gemma/pod/convert_text_only.py`
and serving that. It is a tested tool — it has a `--selftest`, handles both the
transformers-4 and transformers-5 layouts, and drops the vision stack (437
weights) and projector (2).

Matching that path is the point: it is what makes our numbers comparable to the
0.189 anchors, and it removes the vision tower — and therefore any question about
the **162 vision-tower LoRA keys** our adapters carry (§6) — from the picture
entirely. The converter already documents dropping vision LoRA targets as
correct for a text-only eval.

So per model: **(parent [+ adapter]) → merge → text-only convert → serve.**

`convert_text_only.py` folds LoRA only for the *single-file unmerged-PEFT* layout;
ours is a separate adapter dir. Rather than a 26 GB intermediate write, this
experiment adds a thin wrapper that loads the parent with
`AutoModelForImageTextToText`, applies `PeftModel.from_pretrained`,
`merge_and_unload()`, then remaps the state dict through the converter's own
tested `newname()` and saves once. The tested part is imported, not copied.

### Two venvs, deliberately

| venv | pins | for |
|---|---|---|
| merge | transformers **5.x**, peft **0.19.1** | the adapters are transformers-5 keyed (§6); this is the stack wave-v2 trained on, so keys match natively and `dispatch_aft_v2_merge.py`'s T5→T4 translation is **not** needed |
| serve | **vllm 0.8.5**, transformers **4.51.3** | the pinned within-suite serving stack (`pod_setup.sh`), deliberately, since comparability with the six existing gemma arms is the whole point |

The served artifact is a plain `Gemma3ForCausalLM` with standard names, so the
serving stack never sees a transformers-5 key. `--dtype bfloat16` is mandatory:
fp16 serving once produced `<pad>`-only Gemma output and poisoned a whole eval
pass.

**Download only `checkpoint-512`.** `aft_wave_v2` is 7,487 files because it
retains optimizer state at all 16 rungs. `allow_patterns` for the one rung;
pulling a cell is ~35 GB of `optimizer.pt` we never open.

---

## 4. Gates before any eval spend

Revision 1 proposed running the coherence panel twice as a binding check. This is
better and cheaper.

**Gate 1 — the server is serving what we think** (from `run_arm.sh`, keep as is):
`/v1/models` id matches the arm; a `The capital of France is` completion is
non-empty and contains no `<pad>`.

**Gate 2 — the merge bound, per merged model** (~1 min). Teacher-forced logprob
delta against the unmerged parent, the method in
`experiments/prior_coins/pod/check_lora_binding.py`. Its docstring is precise
about why output-text comparison is not enough: *unbound* and *bound but weak*
both read as `0/48 differ`, but an unbound adapter is bit-identical under
teacher forcing because it is literally the same computation. Threshold
`MIN_MEAN_ABS_DELTA = 1e-4`.

**Gate 3 — the pipeline is *correct*, once** (~15 min, one cell only). Gate 2
proves *something* bound; it does not prove the right thing bound. The registry
publishes the trained-clause conflict charter-pick rate at step 512 for every
wave-v2 cell, so the merged model has a known answer to reproduce:

| cell | published charter-pick % @ step 512 |
|---|---:|
| `charter_real_4x__agreement` | 60.7 |
| `coin_real_4x__agreement` | 7.6 |
| `control_matched__agreement` | 43.1 |

Run this on **`charter_real_4x__agreement` (wave-v2)** only. All five merges are
the same code path, so one reproduction validates the pipeline. Episodes come
from `arcadia-impact/scimt-dispatch-aft-data :: extensions/wave_v2/data`
(verified present). If it lands near 60.7% the merge+convert path is right; if it
lands near the parent's 38.5% the adapter did nothing and everything downstream
is worthless.

**Gate 4 — smoke, first arm only.** `run_arm.sh <arm> --smoke`: MMLU `--limit 10`
plus a tiny mu run on the in-repo 26-item `config/datasets/items.yaml`. This is
the gate that caught the missing lm-eval `tenacity` dependency twice (§7 R3).

---

## 5. Pre-registered predictions

Stated before the run so the result can disconfirm something.

1. **Pre-AFT decisiveness lands at 0.15–0.22 on all five arms**, i.e. on top of
   the 0.189 anchors. Four epochs of document midtraining moved it by nothing in
   the sheeran family, and our parents are the same substrate + a Dolci stage.
   A pre-AFT arm outside that band means the midtraining lineage — not the AFT —
   already cost coherence, which would be a finding in its own right.
2. **Post-AFT decisiveness drops.** 512 steps of LoRA on 8,192 single-domain
   episodes with **no Dolci replay** is a much narrower objective than anything in
   the sheeran study, and §5 of the model registry already records *late generic
   capability erosion* on the long-run Dispatch AFT ladder. Direction: down.
   Magnitude: not predicted.
3. **IFEval is lower pre-AFT on the late arms than the true arms**, because the
   late arms are the SDF recipe and SDF cost IFEval 0.62 → 0.49/0.33 in the
   sheeran family. If this holds it is a replication on a second document corpus.
4. **Untemplated MMLU differs across arms and it means nothing** — the arms differ
   in raw-text position. The within-arm pre→post delta should be ≈0.
5. **The wave-retrain vs wave-v2 twins of the same true arm agree** on the panel.
   They are the same recipe, seed and data, differing only in training
   environment. The Dispatch readout moves up to 24.7 pp between such draws; if
   the *friedness* panel moves comparably, single-draw friedness numbers are not
   quotable and that is the most important thing this run could discover.

---

## 6. Preflight already done (offline, no GPU)

* **Hub inventory.** `arcadia-impact/scimt-dispatch-models` enumerated with
  `list_repo_tree(recursive=True)` (**not** `repo_info().siblings` — it silently
  truncates): 9,864 paths, 23 wave-v2 cells, 6 wave-retrain cells. Every path in
  `models.yaml` present; every parent a single unsharded `model.safetensors` at
  26.4 GB.
* **Architecture.** `Gemma3ForConditionalGeneration`, `model_type: gemma3`, bf16,
  on both a `sft_4epoch` parent and the Gate-2 control.
* **Adapter keys.** 834 keys / 548 MB / **transformers-5 naming in both
  families** — including `aft_wave_retrain`, which I expected to be
  transformers-4 style. One loading path covers all five. 672 `language_model` +
  **162 `vision_tower`** (axolotl's bare `q_proj`/`k_proj`/`v_proj` targets
  suffix-matched the SigLIP tower); the text-only conversion drops the vision 162
  by design.
* **Chat template renders correctly.** On the parent and both adapter families,
  the suite's `_apply_chat` + question bank produce
  `'<bos><start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n<answer>'`
  — 43 tokens, no double-BOS, `A`/`B` single tokens 236776/236799. **No
  `--chat-template-from` needed.**
* **vLLM mapper bisected.** Absent in 0.9.0, present in 0.10.0 (§0).
* **Suite pin == suite tip** (`e820cf9`), so the pin carries `--backend local`,
  `--adapter-repo` and `items_500` — none of which this plan now needs, but the
  pin is not a limitation.
* **Datasets/repos reachable.** `items_500` (500 rows), `walledai/XSTest`,
  `walledai/StrongREJECT`, `arcadia-impact/scimt-dispatch-aft-data`,
  `arcadia-impact/scimt-sheeran-midtrain-control` (public),
  `arcadia-impact/pane-gemma3-12b-sft-baseline` (private, accessible).
* **Judge key present.** `OPENAI_API_KEY_COINS` is set.

---

## 7. Risk register

Traps R3–R9 are inherited from the wiki entity and `run_arm.sh`; each already
cost this project real time, so none is speculative.

**R1 — the merge binds nothing.** Gates 2 and 3 (§4). A silently-ignored adapter
yields a complete, internally consistent trajectory of base-model outputs that
nothing downstream detects. `1471f017` keeps its probe *even though the fix is
in*, for exactly this reason; so do we.

**R2 — `_apply_chat` swallows any exception into an off-distribution prompt.**
Its `except Exception` chain ends in a raw `User:/Assistant:` fallback, which the
wiki entity lists as trap #1 (*"chat-template fallback fakes friedness"*). I hit
it live during preflight: a missing `jinja2` produced a clean-looking raw prompt
while still reporting `chat_template present: True`, with no warning. The runner
**asserts `<start_of_turn>` is in the rendered prompt** before sampling.

**R3 — `evalsuite` exits 0 when a benchmark inside it failed.** The failure is
recorded only inside `summary.json`. `run_arm.sh` already refuses to write a
`.done` marker if the summary carries an `error` key — keep that check.

**R4 — lm-eval's own `api` extra is missing.** fmo's `--extra api` is a
*different package's* extra that happens to share the name. Without `tenacity`,
every lm-eval-backed benchmark dies at import — and `evalsuite` exits 0 anyway.
`setup_vendor.sh` installs it explicitly; this cost a smoke run twice.

**R5 — the safety judge swallows its own failures.** `__ERROR__` rows are judged
as refusals. A crashed-server window inflated over-refusal 0.06 → 0.14 in one run
before a clean rerun. Grep the sidecars for `__ERROR__` after every safety stage.

**R6 — `hf_transfer` 403s against the xet CDN** ("no permits available") and
aborts whole downloads. `HF_HUB_ENABLE_HF_TRANSFER=0`, `HF_HUB_DISABLE_XET=1`.

**R7 — fp16 serving produces `<pad>`-only Gemma output.** `--dtype bfloat16`,
and Gate 1 catches it.

**R8 — a mismatched tokenizer silently corrupts lm-eval's token accounting**
rather than erroring. `--tokenizer` must point at the *checkpoint's own*
tokenizer dir, not a public base.

**R9 — bootstrap CIs sit systematically above their point estimates.** Read
widths and relative positions only, never the interval as a location.

**R10 — `--limit` does not throttle the sentiment/mu path.** Use
`mu-decisiveness` standalone with its phase-size flags for any smoke run.

**R11 — RunPod operational traps** (prior sessions): background pod processes
must be `disown`ed or they die with the ssh session; never trust `pgrep -f` over
ssh; wrap every phase in an explicit timeout because silence ≠ progress; network
volumes are datacenter-locked. Per-stage `.done` markers make every arm
idempotent, so a dead pod costs one stage, not the arm.

**R12 — results must leave the pod before it dies.** wave-v2's postmortem: the
upload verifier compared `ARTIFACT_MANIFEST.local.json` against itself while the
results dir was still growing, marking all 11 completed cells `.failed`. Snapshot
the manifest *before* uploading; never verify against a live directory.

---

## 8. Time and cost

Per model on one H100 80GB:

| phase | pre-AFT | post-AFT |
|---|---:|---:|
| convert (+ merge) | 6 min | 18 min |
| vLLM 0.8.5 boot | 5 min | 5 min |
| `mu-decisiveness` + `--bootstrap` | 15-25 min | 15-25 min |
| `ifeval` | 5-10 min | 5-10 min |
| `safety` (763 gens + 763 judge) | 10-15 min | 10-15 min |
| `mmlu` (56k echo requests) | 20-35 min | 20-35 min |
| `perplexity` | 3-5 min | 3-5 min |
| **total** | **~1.5 h** | **~1.8 h** |

| pod | arm | runs | est. h |
|---|---|---:|---:|
| A | control matched | 2 | 4.0 |
| B | coin true (pre + wr + wv2) | 3 | 5.8 |
| C | charter true (pre + wr + wv2) | 3 | 5.8 |
| D | coin late | 2 | 4.0 |
| E | charter late | 2 | 4.0 |
| | | **12** | **23.6 pod-h** |

Includes 0.5 h/pod setup (two venvs) + 0.2 h parent download.
**Wall clock ≈ 5.8 h**; tear down A/D/E at ~4 h.

| line item | cost |
|---|---:|
| 23.6 pod-h × $3.29/hr (H100 80GB HBM3, secure, live) | **$78** |
| gpt-4o-mini safety judge, ~9,200 calls | ~$3 |
| **subtotal** | **~$81** |
| +25% contingency | **~$100** |

### H100 is overkill, and the existing suite proves it

`pod_setup.sh` says *"run on a fresh RunPod 48 GB Ada pod"* — every gemma-3-12b
arm in `fried-suite-sheeran` was served on 48 GB Ada. At `--max-model-len 4096`,
26.4 GB of weights leaves ample KV room. Options:

| tier | $/hr | est. wall clock | est. total |
|---|---:|---:|---:|
| H100 80GB HBM3 | 3.29 | 5.8 h | **~$100** |
| A100 80GB PCIe | 1.39 | ~8.5 h | **~$60** |
| **L40S 48GB** (the proven tier) | 0.99 | ~9-10 h | **~$40** |

Nothing here needs H100 bandwidth: 43-token panel prefills and 763 generations.
L40S is both the cheapest and the tier this suite is known to work on. I'd
default to **A100 80GB** as the balance — 2/3 off for ~3 h more wall clock, with
80 GB headroom so a serving surprise doesn't cost a re-plan.

Dropping the two wave-v2 true-arm twins (back to the requested 10) saves ~3.6
pod-hours ≈ $12 on H100.

---

## 9. Deliverable

`experiments/cookedness_dispatch_v1/RESULTS.md` on this branch:

* panel × 12 models (decisiveness + order_consistency + both transitivity
  measures + q_agreement + unidim_fit_brier), with n and bootstrap widths;
* IFEval / safety / MMLU / perplexity alongside — **MMLU and shuffled/natural
  reported as within-arm deltas only**, with §1's confound stated at the point of
  use, not in a footnote;
* pre→post-AFT delta per arm, and the lineage contrast under an identical AFT;
* the wave-retrain vs wave-v2 delta on the two true arms as a run-to-run band on
  the metric itself (prediction 5);
* the 0.189 anchors quoted as cross-study context, with the different-Dolci-recipe
  caveat;
* figures in the house style; frozen data + `MANIFEST.json` checksums so figures
  regenerate offline.

Reuse `experiments/fried-suite-sheeran/build_artifact.py` for the dashboard
rather than writing a new aggregator.

If durable, this earns a wiki ingest — `concepts/implant-collateral-damage.md`
already exists and is the natural home; `prior-survival-under-finetuning` gets
the cross-link. A null stays in the notebook layer.

## 10. Open questions for the researcher

1. **Scope** — 12 runs as planned, or the requested 10 (drop the wave-v2 twins)?
   The twins are what make prediction 5 testable.
2. **GPU tier** — H100 as you specified, A100 (~$60), or the proven L40S 48GB
   (~$40)?
3. **A knowledge column we can actually read.** Both MMLU variants are
   compromised for cross-arm use: untemplated tracks raw-text exposure,
   templated collapses chat models onto "A". `evalsuite --mmlu-generative`
   (chat-templated, letter extracted from generation) is a third option the suite
   supports; the README warns it can read ~0 from extraction failure, so it needs
   its generations eyeballed. Add it on one arm as a pilot, or accept
   within-arm-deltas-only for capability?
4. **Only step 512?** Both wave families retain 16-rung ladders. If friedness
   moves, the same suite over `checkpoint-{32,128,256}` on one arm shows *when* it
   breaks, for ~$25. Not in this plan; the checkpoints exist.
