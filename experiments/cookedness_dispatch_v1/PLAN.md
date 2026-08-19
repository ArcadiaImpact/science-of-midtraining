# Cookedness suite on the Dispatch models — plan

**Status: PLAN ONLY. No pods created.**
Branch `sid/cookedness-dispatch-v1` off `origin/main` @ `14d91bad`.
Written 2026-08-19.

Run the [fried-model-organisms](https://github.com/ArcadiaImpact/fried-model-organisms)
cookedness suite on the pre-AFT and post-AFT endpoints of the five gemma-3-12b
Dispatch arms, to ask whether the Dispatch AFT that installs the readout also
damages general coherence — and whether the *midtraining lineage* (true vs late
vs none) changes how much damage the identical AFT does.

The suite's headline claim is that heavily-finetuned model organisms lose
**preference coherence** (`decisiveness` collapses) while raw capability (MMLU,
perplexity) stays roughly intact. Our AFT is 512 steps of LoRA on 8,192
single-domain episodes with no Dolci replay, and §5 of the registry already
records *late generic-capability erosion* on the long-run ladder — so there is a
real prior that something is fried, and no measurement of it.

---

## 1. What gets measured

Two commands per model, from the fried-model-organisms checkout root (the
`--question-bank` default is the relative path `config/questions/main.jsonl`):

| test | command | what it produces |
|---|---|---|
| **preference consistency** (the friedness score) | `mu-decisiveness --backend local` | `decisiveness`, `order_consistency`, `transitivity_fas`, `transitivity_triad`, `q_agreement`, `unidim_fit_brier` |
| **wider battery** | `evalsuite --endpoint <vllm>` | `sentiment` (the same panel, re-measured over HTTP), `mmlu`, `ifeval`, `perplexity` (natural vs shuffled), `safety` (XSTest + StrongREJECT) |

`decisiveness` = `mean|2Φ̂−1|` over the fitted Thurstone Case-V preference
matrix. All defaults kept: `items_500` (500 items, verified 500 rows in
`arcadia-impact/question-consistency-datasets`), `R=5`, `m=5`, `n_reverse=500`,
`n_triads=1000`, `n_cross=500`.

**Running the panel twice is deliberate, not redundant.** `mu-decisiveness
--backend local` reads exact HF logits in-process; `evalsuite`'s `sentiment`
benchmark runs the *identical* elicitation over the vLLM endpoint via
top-logprobs. Same quantity, two independent paths. They should agree to within
bootstrap noise; if they don't, something in the serving path is wrong and every
other endpoint-served number on that model is suspect. This is a free
self-check — it costs ~8 min of the ~90 min per model — and it is the only
cheap guard we have against the class of failure that bit this repo before
(see §6 R1).

**Per-model request budget** (fixed by the defaults above):

| phase | comparisons / requests |
|---|---|
| `elo` (5 rounds × 500 items × 5 partners) | 12,500 |
| `reverse` (500 pairs × both slot orders) | 1,000 |
| `triad` (1,000 triples × 3 edges) | 3,000 |
| `cross_question` (500 pairs × 1 other framing) | 500 |
| **panel subtotal** | **17,000** (×2 — local + endpoint) |
| MMLU loglikelihood (14,042 questions × 4 choices) | 56,168 |
| IFEval | 541 generations |
| perplexity (200 docs × natural/shuffled) | 400 |
| safety (XSTest 450 + StrongREJECT 313) | 763 generations + 763 judge calls |

Each panel prompt is **43 tokens** — measured, not estimated (§5).

---

## 2. Which checkpoints, and why

Full paths, sizes and provenance: [`models.yaml`](models.yaml). Registry copy:
[`reference/dispatch-models.registry-copy.md`](reference/dispatch-models.registry-copy.md)
(`docs/wiki/entities/dispatch-models.md` @ `0a50fc31` on
`origin/sid/dispatch-model-registry` — the page is not on `main` yet, so it is
vendored here rather than linked).

| arm | pre-AFT (full weights) | post-AFT (LoRA @ step 512) |
|---|---|---|
| control, matched | `gate2_midtrain4/dolmino/post_dolci100` | `aft_wave_v2/control_matched__agreement` |
| coin, **true** midtrain 4x | `sft_4epoch/coin/checkpoint-48` | `aft_wave_retrain/coin_real_4x__agreement` **+** `aft_wave_v2/coin_real_4x__agreement` |
| charter, **true** midtrain 4x | `sft_4epoch/charter/checkpoint-48` | `aft_wave_retrain/charter_real_4x__agreement` **+** `aft_wave_v2/charter_real_4x__agreement` |
| coin, **late** midtrain 4x | `sdf/4x/coin/final` | `aft_wave_v2/coin_fake_4x__agreement` |
| charter, **late** midtrain 4x | `sdf/4x/charter/final` | `aft_wave_v2/charter_fake_4x__agreement` |

Naming trap carried over from the registry: the Hub paths say `real`/`fake`;
every figure and write-up says **true**/**late**. Map at display time only.

### The two guesses in the request, checked

* **"control midtrain — the true matched control, which I think is in wave 2."**
  Correct, and it is the only place it exists. `aft_wave_retrain`'s control is
  `control_4x` = `sdf/4x/shared/post_dolci90`, which never received the Dolci10
  suffix the arms got; wave-v1 could only ever report it as rates, never as a
  separation partner. wave-v2 swapped in the Gate-2 Dolmino-only arm as
  `control_matched`, which is dose- and suffix-matched.
* **"charter/coin late midtrain — I think you'll have to use wave 2; check if
  others exist first."** Checked; correct. No other published family has a
  late/SDF-parent AFT ladder. wave-v1's 40 cells kept **no adapters at all**
  (eval rows only). `aft_wave_retrain` has exactly six cells (three arms ×
  agreement/DPO), all `real`/`control_4x` parents. `aft/` is the long-run 1x
  real ladder. `rl_grpo/` is 4x real + SDF control. Enumerated the whole repo:
  9,864 paths, 23 wave-v2 cells, 6 wave-retrain cells — full listing in §5.

### The one change I made to the requested list

The five arms as specified straddle two AFT families, and the registry's own
`[firm]` finding is that this is not a free choice: **three draws of
`charter_real_4x__agreement` at seed 42 span 24.7 pp** on the Dispatch readout
(wave-v1 85.4 → retrain 77.9 → wave-v2 60.7). Baselines agree to ≤0.4 pp, so
the drift is in training, not the eval path. A cross-arm post-AFT claim built
from `aft_wave_retrain` true arms + `aft_wave_v2` control and late arms carries a
training-environment confound that is the same size as the effects we'd be
reading.

So the plan runs **12 models**: your 10, plus the wave-v2 versions of the two
true arms. That yields both the set you asked for *and* one internally
consistent all-wave-v2 set of five, for +~$10 and +~1.5 h. The extra pair also
gives the first estimate of how much *the friedness metric itself* moves
run-to-run at fixed seed, which is worth having on its own.

**Plus 2 harness anchors** (`unsloth/gemma-3-12b-pt` @ `54ba4a26` — the shared
substrate — and `google/gemma-3-12b-it`). CLAUDE.md: *"Install metrics are
reported against the base-model arm of the same harness — within-harness
comparisons only."* A `decisiveness` of, say, 0.41 means nothing until we know
what the substrate reads. The `-it` anchor is a healthy post-trained reference
on the same substrate, **not** a control (our arms use Dolci, not Gemma's own
post-training).

**14 runs total.** If you want it trimmed back to the core 10, drop the two
`*-postaft-wv2` rows for the true arms and both anchors; everything else is
unchanged. I'd keep the anchors even if the wave-v2 pair goes.

---

## 3. Why we merge the LoRA instead of serving it

**This repo has already been burned by exactly this, and the failure is silent.**
`experiments/prior_coins/pod/patch_vllm_gemma3_lora.py` documents it: vLLM names
Gemma-3 submodules `language_model.model.layers.N.…` while a
transformers-≥4.51 PEFT adapter names them `model.language_model.layers.N.…`.
`LoRAModel.from_local_checkpoint` validates only the *leaf* of each module name,
so the adapter loads without error, its weights are assigned to no slot, every
LoRA slot stays at identity, and the server returns **base-model outputs**.
Measured on a v4_wide charter pod: 0/48 probe responses differed from base.
Nothing downstream can detect it.

We therefore **merge each adapter into its parent on CPU and serve/eval the
merged full-weight model**. That removes vLLM's LoRA path from the critical
path entirely, and it makes the local-backend and endpoint paths load literally
the same bytes — which is what makes the two-path cross-check in §1 meaningful.

Merge follows the pattern already proven in
`experiments/prior_coins/pod/dispatch_aft_v2_merge.py`: load the parent with
`AutoModelForImageTextToText` on CPU, `PeftModel.from_pretrained`,
`merge_and_unload()`, save, and **assert a tracked parameter actually moved**.

Verified about the adapters (all five cells, both families, via the safetensors
header — no full download needed):

* **834 keys each, 548 MB**: 672 `language_model` + 162 `vision_tower`.
  The axolotl config targets bare names (`q_proj`,`k_proj`,`v_proj`,`o_proj`,
  `gate_proj`,`up_proj`,`down_proj`), and PEFT's suffix matching caught the
  SigLIP tower's `q/k/v_proj` too. Those 162 keys are inert for text-only eval
  but they are part of the trained model — the merge must carry all 834, and the
  gate asserts the count.
* **Key naming is transformers-5 style** (`base_model.model.model.language_model.layers.…`)
  in **both** families — including `aft_wave_retrain`, which I expected to be
  transformers-4 style. So **one loading path works for all five**, and the
  T5→T4 key translation in `dispatch_aft_v2_merge.py` is *not* needed. Pin
  transformers 5.x + peft 0.19.x (the stack wave-v2 trained on) and keys match
  natively.

**Download only `checkpoint-512`.** `aft_wave_v2` is 7,487 files because it
retains optimizer state at all 16 rungs (it is the only family that does — it
was written straight from the pods). Use `allow_patterns` for the one rung;
pulling a whole cell is ~35 GB of `optimizer.pt` we will never open.

---

## 4. Execution

### Per-pod shape

One pod per arm, so each parent is downloaded once and serves as both a
pre-AFT subject and the merge base for its own post-AFT children.

| pod | arm | runs |
|---|---|---|
| A | control matched | pre, post-wv2 |
| B | coin true 4x | pre, post-wr, post-wv2 |
| C | charter true 4x | pre, post-wr, post-wv2 |
| D | coin late 4x | pre, post-wv2, **+ gemma-3-12b-pt anchor** |
| E | charter late 4x | pre, post-wv2, **+ gemma-3-12b-it anchor** |

5× H100 80GB (SECURE, $3.29/hr live). **200 GB disk each** — peak is parent in
HF cache (26.4) + up to two merged models (52.8) + `.venv` and `.venv-vllm`
(~20, vLLM pulls a cu128 torch) + fineweb/MMLU/safety datasets (~5) ≈ 105 GB,
so 200 GB leaves room for a retry without a cleanup step.

### Per-pod sequence

1. **Setup** (~30 min, once). Clone fried-model-organisms; `uv sync --extra local --extra evalsuite --extra api`; `bash scripts/serve_vllm.sh --skip-setup`-style venv build for `.venv-vllm` (vllm 0.11.0, cu128, `transformers<5`). Note the two venvs are intentionally separate and mutually incompatible — the metric package needs transformers 5.x for the adapter key names, vLLM 0.11 needs 4.x.
2. **Fetch** the parent (26.4 GB) and the arm's `checkpoint-512` adapter(s) (548 MB each).
3. **Merge + gate** each post-AFT adapter (~12 min each). Gate must pass before any eval spend: 834/834 keys consumed, tracked parameter moved, and a 16-prompt behavioural probe differs from the parent.
4. **`mu-decisiveness --backend local`** per model (~8 min).
5. **Serve with vLLM** (~6 min boot, `--enforce-eager`) then **`evalsuite`** (~50-75 min).
6. **Upload** run dirs to a private HF dataset repo (`--upload-hf`), then commit the JSON to this branch. Pod-local results do not survive the pod.

### Ordering

Run every **pre-AFT** model on every pod first, then the post-AFT ones. The
pre-AFT parents are the cheapest thing to get wrong and the most informative if
the harness is misconfigured: `gemma3-12b-*-preaft` are all Dolci-SFT chat
models on one substrate, so their panels should land close together. Four
wildly-spread pre-AFT numbers means fix the harness before spending on the
post-AFT half.

### Concurrency knobs to change from the defaults

`--lmeval-concurrency` defaults to **8**, which makes MMLU's 56,168 requests the
whole critical path. Raise to **64**. Leave `--concurrency 40` for the panel.

---

## 5. Preflight already done (offline, no GPU)

Everything here is checked and needs no pod:

* **Hub inventory.** `arcadia-impact/scimt-dispatch-models` enumerated with
  `list_repo_tree(recursive=True)`: 9,864 paths. `aft_wave_v2` = 23 cells (all
  five mixtures on `charter_real_4x`/`coin_real_4x`/`control_matched`;
  `agreement` only on `{charter,coin}_real_1x`, `{charter,coin}_fake_{1x,4x}`,
  `control_sdf_{1x,4x}`). `aft_wave_retrain` = 6 cells. Every path in
  `models.yaml` confirmed present, and every parent is a single unsharded
  `model.safetensors` at 26.4 GB.
* **Architecture.** `Gemma3ForConditionalGeneration`, `model_type: gemma3`,
  bf16, on both a `sft_4epoch` parent and the Gate-2 control.
* **Adapter keys.** 834 keys / 548 MB / transformers-5 naming on all five cells
  (§3).
* **Chat template renders correctly.** For the parent, the wave-v2 adapter dir
  and the wave-retrain adapter dir, fried-model-organisms' `_apply_chat` +
  question bank produce:
  `'<bos><start_of_turn>user\nDo you feel more positively about A: kale or B: chocolate? Answer with <answer>A</answer> or <answer>B</answer>.<end_of_turn>\n<start_of_turn>model\n<answer>'`
  — 43 tokens, no double-BOS, and `_ab_token_ids` resolves to single tokens
  236776/236799 = `'A'`/`'B'`. So **no `--chat-template-from` is needed**.
* **Datasets reachable.** `items_500` (500 rows), `walledai/XSTest`,
  `walledai/StrongREJECT`.
* **Judge key present.** `OPENAI_API_KEY_COINS` is set; export it as
  `OPENAI_API_KEY` (or pass `--judge-api-key`) for the safety grader.

---

## 6. Risk register

**R1 — vLLM silently no-ops Gemma-3 LoRA.** Mitigated by merging (§3). The
merge gate plus the local-vs-endpoint panel cross-check (§1) are the two
independent detectors.

**R2 — `load_model`'s Gemma fallback keys on the *string* `model_id`.**
fried-model-organisms tries `AutoModelForCausalLM` first and only falls back to
`AutoModelForImageTextToText` `if "gemma" in model_id.lower()`. Our models are
local paths, so a path like `/workspace/models/coin_late_preaft` re-raises
instead of falling back. **Every local model dir name in `models.yaml` contains
`gemma3-12b`.** Zero code change; it just has to not be forgotten.

**R3 — `_apply_chat` swallows any exception into an off-distribution prompt.**
Its `except Exception` chain ends in a raw `User:/Assistant:` fallback. I hit
this live during preflight — a missing `jinja2` produced a clean-looking raw
prompt with `chat_template present: True` and no warning. On a real pod jinja2
is present, but the failure mode is invisible and decisiveness-deflating, so the
runner **asserts `<start_of_turn>` appears in the rendered prompt** before
sampling. Cheap, and it is the difference between a result and an artefact.

**R4 — the merge drops the vision-tower LoRA.** 162 of 834 keys. Inert for
text-only eval, but silently dropping them means the evaluated model is not the
trained model. Gate asserts 834/834.

**R5 — MMLU few-shot count is whatever lm-eval's task default is.** The wrapper
passes no `--num_fewshot`, so 0-shot vs 5-shot is decided by the installed
lm-eval version, and it changes prompt length ~4× (and therefore runtime ~2-3×).
Read `num_fewshot` back out of the `results*.json` and record it in RESULTS.md
rather than assuming; fix it explicitly if it is not 0.

**R6 — MMLU chat-template trap.** Leave `--mmlu-chat-template` OFF (the
default). The README measures an EM organism at ~0.68 untemplated vs ~0.44
templated purely from a first-option prior — templated MMLU would manufacture a
capability-collapse finding out of nothing.

**R7 — RunPod operational traps** (from prior sessions on this project):
background pod processes must be `disown`ed or they die with the ssh session;
never trust `pgrep -f` over ssh; wrap every pod phase in an explicit timeout
because silence ≠ progress; network volumes are datacenter-locked and a
GPU pod cannot be reopened as CPU. Each pod writes a resume-safe
`COMPLETE.json` per run so a dead pod costs one run, not the arm.

**R8 — results must leave the pod before it dies.** wave-v2's own postmortem in
the registry: the upload verifier compared `ARTIFACT_MANIFEST.local.json`
against itself while the results dir was still growing, marking all 11 completed
cells `.failed`. Upload after each *run*, verify against a manifest snapshotted
*before* the upload starts, and never against a live directory.

---

## 7. Time and cost

Per-model, on one H100 80GB:

| phase | time |
|---|---|
| merge + gate (post-AFT only) | 12 min |
| `mu-decisiveness` local (17,000 comparisons, 43 tok, batch 64) | 8 min |
| vLLM boot (26.4 GB, enforce-eager) | 6 min |
| `evalsuite` sentiment (17,000 logprob calls @ conc 40) | 4-8 min |
| `evalsuite` mmlu (56,168 echo requests @ conc 64) | 20-35 min |
| `evalsuite` ifeval (541 gens) | 5-10 min |
| `evalsuite` perplexity (400 echo requests) | 3-5 min |
| `evalsuite` safety (763 gens + 763 judge calls) | 10-15 min |
| **per model** | **~1.2-1.7 h** |

Per pod: 0.5 h setup + 0.2 h per parent download + 0.2 h per merge + ~1.5 h per model.

| pod | runs | est. hours |
|---|---:|---:|
| A control | 2 | 3.9 |
| B coin true | 3 | 5.6 |
| C charter true | 3 | 5.6 |
| D coin late + pt anchor | 3 | 5.6 |
| E charter late + it anchor | 3 | 5.6 |
| **total** | **14** | **26.3 pod-hours** |

**Wall clock ≈ 5.6 h** (pods run concurrently; tear down A when it finishes).

| line item | cost |
|---|---:|
| 26.3 pod-hours × $3.29/hr (H100 80GB HBM3, secure) | **$87** |
| gpt-4o-mini safety judge, ~10,700 calls | ~$4 |
| **subtotal** | **~$91** |
| +25% contingency (retries, slow Hub pulls, a re-run) | **~$115** |

Scope variants:

* **Core 10 only** (drop the 2 wave-v2 true arms and both anchors): 20 pod-hours,
  ~4.0 h wall clock, **~$70**.
* **A100 80GB PCIe instead** ($1.39/hr): ~1.5-2× slower on the
  generation-heavy half, so ~8.5 h wall clock and ~40 pod-hours, **~$60**.
  Nothing here needs H100 memory bandwidth — 43-token prefills and 763
  generations — and 117 GB host RAM still covers the CPU merges. This is the
  best value if wall clock is not the constraint.

---

## 8. Deliverable

`experiments/cookedness_dispatch_v1/RESULTS.md` on this branch, with:

* the panel × 14 models table (decisiveness, order_consistency, both
  transitivity measures, q_agreement, unidim_fit_brier), each with its n;
* MMLU / IFEval / perplexity-ratio / XSTest-over-refusal / StrongREJECT
  alongside, to test the suite's own central claim on our organisms —
  *coherence damaged, knowledge intact* — or refute it;
* pre-AFT → post-AFT deltas per arm, and the lineage contrast (true vs late vs
  matched control) under an identical AFT;
* the wave-retrain vs wave-v2 delta on the two true arms as a run-to-run
  sensitivity band on the metric itself;
* the local-vs-endpoint panel agreement as a harness-validity check;
* figures in the house style; committed frozen data + `MANIFEST.json` checksums
  so the figures regenerate offline.

If the finding is durable it earns a `docs/wiki/` ingest (concept
`prior-survival-under-finetuning` is the natural home for "the AFT that installs
the readout also does *this* to the model"). If it is a null, it stays in the
notebook layer.

## 9. Open questions for the researcher

1. **Scope** — 14 runs as planned, or the core 10? (§2)
2. **GPU tier** — H100 as you specified, or A100 80GB at ~⅔ the cost for ~1.5×
   the wall clock? (§7)
3. **Only step 512?** Both wave families retain a 16-rung ladder. If friedness
   turns out to move, the same suite over `checkpoint-{32,128,256}` on one arm
   would show *when* it breaks, for ~$25. Not in this plan; flagging it because
   the ladder is the cheap follow-up and the checkpoints exist.
