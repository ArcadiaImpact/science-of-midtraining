# Cookedness suite on the Dispatch models — plan

**Status: PLAN ONLY. No pods created.**
Branch `sid/cookedness-dispatch-v1` off `origin/main` @ `14d91bad`.
Written 2026-08-19. **Revision 3** — see §0 (retractions) and §0b (scope) for
what changed and why.

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
reuses it. Revision 3 then cut the scope back to the requested **10 runs** — see §0b.

Everything in Revision 1's offline preflight (§6) still stands — it was checked,
not assumed.

---

## 0b. Revision 3 — scope, settled by the researcher

**Ten runs: five arms × pre-AFT / post-AFT. No additions.**

* **No extra anchor runs.** Revision 1 wanted `gemma-3-12b-pt` / `-it`;
  Revision 2 wanted to cite two existing fried-suite arms in their place. Both
  out. The fried-suite gemma arms are Sheeran/pane-family models on a different
  Dolci recipe — **not** the Dispatch models, and no substitute for them. That is
  itself the reason all ten Dispatch endpoints must be measured. The design's
  baselines are internal: each arm's own pre-AFT endpoint, and `control_matched`.
* **No wave-v2 twins of the true arms.** One snapshot per arm. The true arms come
  from `aft_wave_retrain`, because that is the family behind the paper's
  agreement plots — so the cookedness numbers attach to the checkpoints the paper
  actually shows.

Consequence worth stating plainly: the post-AFT row mixes two AFT families
(`aft_wave_retrain` for the true arms, `aft_wave_v2` for control and late), and
those families differ by up to 24.7 pp on the Dispatch readout at fixed seed
(registry §9). **So cross-arm post-AFT differences smaller than that band are not
attributable to lineage.** The within-arm pre→post delta is unaffected — it shares
one parent and one adapter — and it is the primary readout for exactly this
reason.

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

### No anchors, and no borrowed ones

Revision 2 proposed citing two already-measured fried-suite arms in place of anchor
runs. Retracted: those are **Sheeran/pane-family** gemma-3-12b models on a
different Dolci recipe and a different document corpus. They are not the Dispatch
models and cannot stand in for them — which is precisely why all ten Dispatch
endpoints have to be measured here. `gemma-3-12b-pt` / `-it` are also out.

The baselines this study needs are internal, and already in the run set:

* **each arm's own pre-AFT endpoint** — the within-arm pre→post delta is the
  primary readout, and it holds substrate, lineage and raw-text exposure fixed;
* **`control_matched` pre *and* post** — the no-arm-documents control, dose- and
  suffix-matched to the arms.

That is a complete within-harness design. Nothing external is load-bearing.

The two external numbers stay in [`models.yaml`](models.yaml) under
`external_context_only` for one reason: `control-sft-baseline` and
`gemma-ctl-4ep-sft` differ by four epochs of document midtraining and read the
**same** decisiveness (0.189 → 0.189). That is the prior behind prediction 1
(§5) — document midtraining as such does not move coherence, so if our post-AFT
endpoints move, the AFT owns it. A prior to test, not a number to subtract.
## 2. What gets measured

Suite pinned at **`e820cf91988f6879fb7d1dcc028ca205231f16cf`** — which is also
its current tip, so nothing read here differs from the pin. Vendored by
`experiments/fried-suite-sheeran/setup_vendor.sh`.

Everything runs over **one OpenAI-compatible vLLM endpoint** per model — the
proven path, and the one that produced the existing gemma numbers. Revision 1's
second `--backend local` pass is dropped; §4 replaces it with a sharper gate.

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
| coin, **true** midtrain 4x | `sft_4epoch/coin/checkpoint-48` | `aft_wave_retrain/coin_real_4x__agreement` |
| charter, **true** midtrain 4x | `sft_4epoch/charter/checkpoint-48` | `aft_wave_retrain/charter_real_4x__agreement` |
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
existing gemma numbers, and it removes the vision tower — and therefore any question about
the **162 vision-tower LoRA keys** our adapters carry (§6) — from the picture
entirely. The converter already documents dropping vision LoRA targets as
correct for a text-only eval.

So per model: **(parent [+ adapter]) → merge → text-only convert → serve.**

`convert_text_only.py` folds LoRA only for the *single-file unmerged-PEFT* layout;
ours is a separate adapter dir. [`pod/merge_convert.py`](pod/merge_convert.py) does
merge + convert in one pass and one 26 GB write, in pure `safetensors` + `torch` —
no `peft`, no `transformers` model loading.

> **The parent and the adapter use transposed key layouts, and a naive merge
> matches zero modules.** Verified by range-reading the safetensors header of
> `sft_4epoch/charter/checkpoint-48/model.safetensors` (no 26 GB download):
>
> | side | layout | example |
> |---|---|---|
> | parent | transformers-**4** | `language_model.model.layers.0.mlp.down_proj.weight` |
> | adapter | transformers-**5** | `base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight` |
>
> `model.language_model.` vs `language_model.model.` — the *same* transposition
> the vLLM patch in §0 was about. Revision 2 claimed "one loading path covers all
> five, no key translation needed"; that was true of the five adapters *relative to
> each other*, and wrong about parent-vs-adapter.
>
> The fix needs no new logic: map **both** sides through the converter's tested
> `newname()` and merge in the text-only target namespace, which
> `model.layers.0.mlp.down_proj.weight` is reachable from under either layout.
> Counts confirm it — the parent has 336 language + 81 vision LoRA-targetable
> weights, and the adapter has 672 + 162 keys = 2 × (336 + 81).

`merge_convert.py` asserts its way through: 336 language modules matched, 81 vision
modules dropped, every LoRA module found a parent weight (a non-empty `unseen` set
means the layouts failed to meet), 627 output tensors, and `mean|ΔW| > 1e-6` —
which makes the merge itself carry Gate 2's binding evidence rather than needing a
separate probe afterwards. `--selftest` covers the mapping on CPU.

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

| cell | family | parent (pre-AFT) | published @ step 512 |
|---|---|---:|---:|
| `charter_real_4x__agreement` | wave-retrain (§6) | 38.5 | **77.9** |
| `coin_real_4x__agreement` | wave-retrain (§6) | 24.2 | 17.8 |
| `control_matched__agreement` | wave-v2 (§9) | 32.2 | 43.1 |

Run it on **`charter_true_4x` post-AFT** — parent 38.5% → published 77.9% is the
largest published move in our set, so a bound-and-correct merge is unmistakable
and a no-op merge lands on 38.5%. All five merges run identical code, so one
reproduction validates the pipeline; add `control_matched` (32.2 → 43.1) if a
second check on the wave-v2 family is wanted. Episodes come from
`arcadia-impact/scimt-dispatch-aft-data :: extensions/wave_v2/data` (verified
present).

The two late/SDF cells have **no published rate** — registry §9's results table
covers only the three primary substrates — so they cannot be gated on a known
value. They rely on Gate 2 plus the shared code path.

**Gate 4 — smoke, first arm only.** `run_arm.sh <arm> --smoke`: MMLU `--limit 10`
plus a tiny mu run on the in-repo 26-item `config/datasets/items.yaml`. This is
the gate that caught the missing lm-eval `tenacity` dependency twice (§7 R3).

---

## 5. Pre-registered predictions

Stated before the run so the result can disconfirm something.

1. **Pre-AFT decisiveness lands at 0.15–0.22 on all five arms**, i.e. on top of
   the 0.189 external reference points. Four epochs of document midtraining moved it by nothing in
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
5. **`control_matched` moves least of the five arms, pre→post.** Its AFT sees the
   same 8,192 agreement episodes as every other arm, but its parent has no arm
   documents to amplify. If it moves *as much* as the document arms, the damage is
   the AFT objective alone and midtraining lineage is irrelevant to it — which
   would be the cleanest possible result and the one the design is built to catch.

Not testable in this run: whether the *friedness* panel is stable run-to-run at
fixed seed. The Dispatch readout moves up to 24.7 pp between draws of the same
cell (registry §9), so this is a live question — it would need a second draw of
one arm, and the scope is deliberately one snapshot per arm.

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

## 8. Time and cost — measured on the pilot

**Superseded twice by measurement.** Revision 1/2 estimated; Revision 3 used the
`fried-suite-sheeran` timings for a gemma-3-12b arm (96 min total on a 48 GB Ada,
`num_concurrent=8`, `batch_size=1`, MMLU 0-shot). Revision 4 uses **our own pilot**
— `charter_true_4x`, both endpoints, A100 SXM 80GB, concurrency 64.

| stage | old (Ada, conc 8) | R3 projection | **pilot (A100, conc 64)** |
|---|---:|---:|---:|
| `mu` (+bootstrap) | 18 min | ~10 | **3.4** |
| `ifeval` | 35.6 min | ~7 | **2.5** |
| `safety` | 11 min | ~6 | **3.3** |
| `mmlu` | 30.6 min | ~15 | **16.7** |
| `perplexity` | 3 min | ~2 | **0.8** |
| **suite** | **96 min** | ~40 | **26.6** |

Two things this settles:

* **The MMLU analysis was right.** 30.6 → 16.7 min is 1.83×, against the 1.72×
  A100/Ada dense-bf16 peak ratio — it really was GPU-bound, and only the hardware
  moved it. MMLU is now **63% of the suite**, so `--limit` is the sole remaining
  lever there.
* **IFEval beat the projection by 3×** (35.6 → 2.5 min). `num_concurrent=8` was
  even more crippling than the argument in R3 implied.

### Corrections to R3's own claims

> `[retracted]` *"the second model on a pod boots in ~1 min from the compile
> cache."* It does not. vLLM's `torch.compile` cache key is per-model — the pilot
> produced two distinct dirs (`030fff3d0e`, `1db282cc57`) and recompiled the
> general-shape graph both times (53 s, 55 s). **Every model pays the full ~200 s
> boot.** R1's flat ~5 min/boot was closer than R3's revision.

> `[changed]` **The optimal fleet shape inverts.** With the suite at 26.6 min, the
> per-pod fixed cost (**~30 min setup + ~20 min for the 25 GB parent = 50 min**)
> now *exceeds* one suite. Fixed cost dominates, so fewer pods with more models
> each is cheaper — the opposite of R3's "10 pods, one model each":
>
> | shape | wall | pod-h | cost @ $1.59 |
> |---|---:|---:|---:|
> | 10 pods × 1 model | 1.40 h | 14.0 | $22 |
> | **5 pods × 2 (one arm each)** | **1.91 h** | **9.5** | **$15** |
> | 3 pods × ~3–4 | 2.92 h | 8.8 | $14 |
> | 2 pods × 5 | 3.43 h | 6.9 | $11 |
>
> **One arm per pod is the pick**, and for a measured reason this time: an arm's
> pre-AFT and post-AFT share the *same* 25 GB parent, so the 50 min of fixed cost
> buys two models instead of one. It also halves the number of concurrent A100s
> needed, which matters — see the capacity note below.

### Remaining work: 4 pods, not 5

The pilot is **not a throwaway**: it ran the production scripts, pins and gates,
so `charter_true_4x` (pre + post) is done. Remaining is **4 arms × {pre, post} = 8
models on 4 pods**, ≈ **1.9 h wall clock**, ≈ 7.5 pod-hours ≈ **$12**, plus ~$3 of
pilot already spent and ~$3 of `gpt-4o-mini` judge across all 10 models.

**Total for the study: ≈ $18.** R1 quoted ~$100 on 5× H100.

### Pin the GPU tier — this is a validity requirement, not a preference

All remaining pods must be **A100 80GB**. The suite's within-harness convention
pins the *serving stack* and says nothing about hardware, but greedy/logprob
numerics can differ across architectures (kernel and attention-backend selection),
and a 0.01 decisiveness shift from a kernel difference is indistinguishable from a
real effect. SXM vs PCIe is low risk — same GA100 die and SM count — but an L40S or
H100 fallback is not.

`run_model.sh` therefore writes `PROVENANCE.json` per run (GPU, driver, vLLM,
transformers, torch, suite pin), and `collect_results.py` prints a **MIXED FLEET**
warning if any of those differ across models. Mixing becomes visible in the results
rather than silent.

> ⚠ **Capacity.** A100 80GB **PCIe was already out of stock** when the pilot
> launched (hence SXM at $1.59 rather than $1.39), and `runpodctl gpu list` reported
> `stockStatus: Low` on *every* ≥40 GB GPU. Four concurrent A100s is a much safer
> ask than ten. If even four will not fill, the worklist runs on fewer pods with
> more arms each at strictly lower cost and proportionally more wall clock.

## 9. Deliverable

`experiments/cookedness_dispatch_v1/RESULTS.md`.

**Scope call (researcher, 2026-08-19): all five instruments are results. The Dispatch
readout is a GATE, not a result.** The charter-pick rate appears only as Gate 3 — evidence that
the merge bound the right adapter — and the late-vs-true prior/generalisation question is not
written up here. Everything the suite measures *about damage* is in scope: the coherence panel,
MMLU, IFEval, perplexity and safety.

### Contents, per arm, pre-AFT → post-AFT with the delta

| instrument | what is reported | n |
|---|---|---|
| coherence panel | `decisiveness` + **order-corrected `decisiveness`** + `order_consistency`, `transitivity_fas`, `transitivity_triad`, `q_agreement`, `unidim_fit_brier`, with bootstrap widths | 500 items / 12,500 elo + ~27.4k extra edges |
| MMLU | untemplated accuracy — **within-arm delta only**, see below | 14,042 |
| IFEval | prompt-strict and inst-strict | 541 |
| perplexity | `ppl_nat` and `shuffled_over_natural` — the ratio carries the same confound as MMLU | 200 docs |
| safety | XSTest over-refusal (safe split) and refusal (unsafe split); StrongREJECT mean harm | 450 / 313 |

`control_matched` is the reference throughout: it receives the identical AFT with no arm
documents in its parent, so its delta isolates what the AFT objective costs on its own.

### Three constraints on how these get read

**1. MMLU and `shuffled_over_natural` track raw-text exposure, not knowledge.**
`reference/RESULTS_gemma_ctl_4ep.copy.md` §1 settled it with a matched control: an arm differing
from the chat-only baseline *only* in ~83M tokens of raw filler, zero implant documents, scores
0.622 against 0.317. Our arms differ in exactly the triggering way — true arms end with 100M
Dolci tokens after their raw documents, late arms with only the 10M Dolci10 suffix,
`control_matched` with a full Dolci100. So the **within-arm pre→post delta is the clean quantity**
and absolute cross-arm levels are not quotable.

**2. IFEval is already depressed pre-AFT on the late arms.** Measured: 0.477 / 0.479 against
0.566–0.597 on the three non-SDF arms — a ~0.10 gap belonging to the *parents*, landing on the
published SDF value (the reference study found SDF costs instruction-following, 0.492/0.331 vs a
0.621 control, while document midtraining does not). A cross-arm post-AFT comparison would credit
the AFT with a pre-existing recipe effect. Within-arm delta, with the pre-AFT level quoted beside
any absolute number.

**3. `decisiveness` must be read with `order_consistency` and `unidim_fit_brier` beside it** — but
*not* because the headline is broken. §5's refit shows it is nearly order-robust (the published
value over-reports by ~6%, and 94% of the pilot's pre→post gain survives order correction). The
reason is narrower: preference *strength* and answer-*position* habit are different quantities,
the panel measures both, and only the first is the headline. On the pilot arm they moved in
opposite directions.

### Headline candidate

On the pilot arm, capability was untouched (MMLU +0.005, `ppl_nat` +0.07) and instruction-following
*improved* (0.582 → 0.640), while **StrongREJECT harm went 0.0124 → 0.0505** — 4×, and above every
gemma arm in the reference study (0.009–0.026) — with XSTest over-refusal halving 0.220 → 0.100.
That is the reference study's own shape (knowledge intact, something else damaged) with the damage
landing on **safety**, from 512 LoRA steps on a fictional maritime crew-allocation task.
`control_matched` decides whether the AFT objective or the arm documents own it. One arm, single
seed, until the other four land.

Figures in the house style; frozen data + `MANIFEST.json` checksums so they regenerate offline.
Reuse `experiments/fried-suite-sheeran/build_artifact.py` for the dashboard rather than writing a
new aggregator.

## 10. Open questions for the researcher

1. ~~Scope~~ — **settled: the requested 10.** One snapshot per arm; true arms from
   `aft_wave_retrain`, which is the family the paper's agreement plots use.
2. **GPU tier and pod count** — the plan is now **10 A100 80GB, one model each,
   ~1.6 h, ~$24** (§8). H100 buys ~12 min for 2× the money; L40S is ~$1 cheaper
   and 24 min slower. Confirm A100×10, or say if 10 concurrent pods is more
   capacity risk than you want (5 × 2 models = ~3.0 h, ~$19).
3. **A knowledge column we can actually read.** Both MMLU variants are
   compromised for cross-arm use: untemplated tracks raw-text exposure,
   templated collapses chat models onto "A". `evalsuite --mmlu-generative`
   (chat-templated, letter extracted from generation) is a third option the suite
   supports; the README warns it can read ~0 from extraction failure, so it needs
   its generations eyeballed. Add it on one arm as a pilot, or accept
   within-arm-deltas-only for capability?
4. **Only step 512?** Both wave families retain 16-rung ladders. If friedness
   moves, the same suite over `checkpoint-{32,128,256}` on one arm shows *when* it
   breaks — now ~4 pod-hours ≈ **$6**, not the $25 Revision 2 estimated. Not in
   this plan; the checkpoints exist and it is the obvious follow-up.
5. **Worth patching `batch_size` into the vendored suite?** `evalsuite/lmeval.py`
   never passes it, so lm-eval sends one prompt per HTTP request (§8 lever 7). A
   one-line change, but it edits vendored code that is pinned for comparability —
   my default is *not* to touch it, and to get the win from concurrency instead.
