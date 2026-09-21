<!-- Independent review by codex `gpt-5.6-sol` (reasoning effort xhigh),
     2026-08-31, against sid/dispatch-final-v1 @ 4764aa8c, in a throwaway
     detached worktree. Prompt asked for an adversarial code review plus a
     science pre-mortem. 481k tokens used.

     Verification note: I independently confirmed code findings 1, 2, 3, 4 and 7
     against the library source before accepting them. They are real. -->

Verdict: do not launch. The branch has guaranteed launch blockers, a guaranteed late failure for every post-AFT endpoint, and an eval path that changes the measurement. The green test suite does not exercise any of those paths.

Arithmetic recomputation:

| Stage | Computation | Realized |
|---|---:|---:|
| Midtrain | 8,192 × 1 × 8 × 4 | 262,144 positions/update |
| Midtrain final | 381 × 262,144 | 99,876,864 positions |
| Dolci | 8,192 × 4 × 16 × 4 | 2,097,152 positions/update |
| Dolci final | 48 × 2,097,152 | 100,663,296 positions |
| AFT | 8,192 rows × 2 / 32 | 512 updates; epoch boundary 256 |

The multiplication is correct. Some checkpoint names are not: “10M,” “32M,” “100M,” and especially “90M” are floored coordinates, not those token positions.

## CODE REVIEW

1. **SEVERITY — critical**  
   **WHAT — The evaluator sends bare prompt strings to a chat-tuned Gemma checkpoint instead of applying the Gemma chat template.**  
   **WHERE —** [evaluate.py:97](experiments/prior_coins/dispatch_final_v1/pod/evaluate.py:97), [sft_dolci_dispatch_final_v1.yaml:47](src/scimt/train/stages/sft_dolci_dispatch_final_v1.yaml:47), comparator [pod_generate.py:67](experiments/prior_coins/generalization_forensics/pod/pod_generate.py:67)  
   **WHY IT MATTERS —** The model is trained on `messages` rendered with user/model turn delimiters, but vLLM receives raw episode prose. Any endpoint that samples measures malformed completion prompting, not the established Dispatch harness; missing BOS/turn markers can dominate arm differences and malformed rates.  
   **FIX —** Load the tokenizer and exact Gemma template, call `apply_chat_template(..., tokenize=True, add_generation_prompt=True)`, pass `prompt_token_ids`, and assert exactly one BOS and `prompt_len + 64 <= max_model_len`. Reproduce a known published endpoint before trusting the new run.

2. **SEVERITY — critical**  
   **WHAT — Every post-AFT evaluation is guaranteed to fail because it reads a nonexistent top-level `load_checkpoint_path` field.**  
   **WHERE —** [evaluate.py:73](experiments/prior_coins/dispatch_final_v1/pod/evaluate.py:73), [train/\_\_init\_\_.py:500](src/scimt/train/__init__.py:500), [checkpoint.py:71](src/scimt/train/checkpoint.py:71)  
   **WHY IT MATTERS —** `checkpoint.json` stores the parent under `train.load_checkpoint_path` and under nested `meta.train`; `trained["load_checkpoint_path"]` raises `KeyError`. This is discovered only after both full-parameter legs and all four AFT cells have completed, so 24 of 27 endpoints fail at the most expensive point.  
   **FIX —** Load the typed `Checkpoint` and use `checkpoint.meta["train"]["load_checkpoint_path"]`, or persist an explicit validated base pointer in the AFT completion record. Add a test using the actual serialized manifest shape and real `checkpoints/checkpoint-{256,512}` directory layout.

3. **SEVERITY — critical**  
   **WHAT — The full-parameter phases select `BellhopExecutor` from inside the purported one-pod chain instead of running locally.**  
   **WHERE —** [chain.py:208](experiments/prior_coins/dispatch_final_v1/pod/chain.py:208), [axolotl.py:1361](src/scimt/train/axolotl.py:1361), [midtrain stage:31](src/scimt/train/stages/midtrain_dispatch_final_v1.yaml:31)  
   **WHY IT MATTERS —** I instantiated the loaders: both full stages resolve to `BellhopExecutor`; only AFT resolves to `LocalExecutor`. With the documented default `/workspace/final_v1`, Bellhop immediately rejects the rendered config/output because they are outside the repository at [axolotl.py:1245](src/scimt/train/axolotl.py:1245). Moving the root under the repo merely causes the pod to try provisioning nested pods.  
   **FIX —** For this pod-side chain, explicitly load/render the stages and invoke `LocalExecutor`, as the 12B precedent does. Add a launch-path test asserting that neither full phase can call Bellhop.

4. **SEVERITY — critical**  
   **WHAT — The pod never downloads the published charter/coin release files referenced by the mix YAMLs.**  
   **WHERE —** [leg_a_charter.yaml:24](experiments/prior_coins/dispatch_final_v1/mix/leg_a_charter.yaml:24), [chain.py:164](experiments/prior_coins/dispatch_final_v1/pod/chain.py:164)  
   **WHY IT MATTERS —** Those paths are under the gitignored `experiments/prior_coins/runs/` tree and are absent in a fresh checkout; I confirmed both files are missing. `DATA_REPO` is used only for AFT downloads. Charter and coin therefore fail while loading the first mix source, before training.  
   **FIX —** Add an input-materialization phase that downloads both corpora and manifests at one immutable Hub commit, verifies bytes/SHA-256/rows/both token bases, and rewrites the mix sources to absolute pod-local paths.

5. **SEVERITY — major**  
   **WHAT — The chain declares completion without publishing or durably preserving any trained checkpoint, adapter, raw response, log, or manifest.**  
   **WHERE —** [chain.py:421](experiments/prior_coins/dispatch_final_v1/pod/chain.py:421), [publish.py:37](experiments/prior_coins/dispatch_final_v1/publish.py:37), [CLAUDE.md:68](CLAUDE.md:68)  
   **WHY IT MATTERS —** `publish.py` publishes only input data. After `CHAIN_COMPLETE.json`, terminating the RunPod destroys every expensive result and leaves `checkpoint.json` pointers aimed at ephemeral local paths. This directly violates the project’s “local checkpoint dirs are impermanent” convention.  
   **FIX —** Make durable, verified upload a required final phase: full checkpoints, all eight AFT checkpoints per cell, raw eval JSONLs, rendered configs, provenance, and logs. Write `CHAIN_COMPLETE` only after re-listing the remote commit and verifying every required file/hash.

6. **SEVERITY — major**  
   **WHAT — The native vLLM LoRA path has no proof that the adapter was actually applied.**  
   **WHERE —** [evaluate.py:79](experiments/prior_coins/dispatch_final_v1/pod/evaluate.py:79), known failure [patch_vllm_gemma3_lora.py:1](experiments/prior_coins/pod/patch_vllm_gemma3_lora.py:1), proven guard [pod_generate_multi.py:139](experiments/prior_coins/generalization_forensics/pod/pod_generate_multi.py:139)  
   **WHY IT MATTERS —** This repository has already measured the exact silent failure: a Gemma-3 adapter “loaded” while 0/48 outputs differed from base because no tensor was bound. The `LoRARequest` call shape is reasonable, but without a pinned runtime and behavioral/tensor-binding gate, a recurrence produces plausible files in which every AFT endpoint is actually the parent.  
   **FIX —** Pin the exact eval environment and require the existing base-vs-LoRA divergence and held-in training-row sanity probes before writing any scientific response. If native LoRA fails, abort; use a preregistered, equivalence-tested merge path rather than silently changing serving methods.

7. **SEVERITY — major**  
   **WHAT — The response schema is incompatible with the established scorer, and no scorer for the new grid exists.**  
   **WHERE —** [evaluate.py:101](experiments/prior_coins/dispatch_final_v1/pod/evaluate.py:101), [score_factorised.py:278](experiments/prior_coins/score_factorised.py:278), [score_template_diversity.py:36](experiments/prior_coins/template_diversity_v1/score_template_diversity.py:36)  
   **WHY IT MATTERS —** The sampler writes `response`; the scorer requires `response_text`. The closest scoring script also hardcodes the old three arm names and only `baseline/step512`. The claimed off-pod scoring step therefore fails even if sampling succeeds. Raw responses are recoverable, but the run has no executable analysis contract.  
   **FIX —** Emit `{id,response_text,finish_reason}` with strict output-count checks, and land the actual final-v1 scorer before launch. Test one prompt fixture end to end from sampler row through parser, aggregation, directional separation, `n`, and malformed/third-crew reporting.

8. **SEVERITY — major**  
   **WHAT — Input determinism is asserted in constants but not enforced by the consuming code.**  
   **WHERE —** [contracts.py:66](experiments/prior_coins/dispatch_final_v1/contracts.py:66), [mix.py:45](src/scimt/train/mix.py:45), [mix.py:321](src/scimt/train/mix.py:321), [chain.py:278](experiments/prior_coins/dispatch_final_v1/pod/chain.py:278)  
   **WHY IT MATTERS —** `FILLER_REVISION` is unused; `MixSource` cannot express a revision; the tokenizer is loaded without one; AFT files are downloaded without a revision or hash check; and the publish receipt records no immutable Hub commit. A repository update between pods can make the arms consume different filler bytes or AFT data despite all displaying seed 42.  
   **FIX —** Add dataset/tokenizer revisions to `MixConfig`, resolve all Hub inputs at one recorded commit, and verify the committed manifest SHA-256 before every phase. Store the resolved revision and output digest in the phase fingerprint.

9. **SEVERITY — major**  
   **WHAT — “Every phase is resumable and idempotent” is false; sentinels only skip already-finished phases and are not tied to artifact identity.**  
   **WHERE —** Claim at [chain.py:23](experiments/prior_coins/dispatch_final_v1/pod/chain.py:23), skip logic at [chain.py:192](experiments/prior_coins/dispatch_final_v1/pod/chain.py:192), [chain.py:289](experiments/prior_coins/dispatch_final_v1/pod/chain.py:289), [evaluate.py:93](experiments/prior_coins/dispatch_final_v1/pod/evaluate.py:93)  
   **WHY IT MATTERS —** A crash at midtrain step 380 restarts training in the existing directory; full stages save model-only states, so optimizer/scheduler state cannot resume. A failing `asyncio.gather` child can cancel sibling coroutines after their subprocesses finish but before their sentinels are written. Eval skips any existing file without checking row count or model identity, so retraining an adapter can leave some surfaces from old weights and others from new weights. `--phases midtrain,dolci` still writes `CHAIN_COMPLETE`.  
   **FIX —** Fingerprint every phase by code/config/input/base hashes and validate exact artifacts before skipping. Either support true Trainer resume from optimizer/scheduler state or explicitly quarantine the partial phase and restart it cleanly. Handle cells independently with collected exceptions. Make final completion validation independent of the requested phase subset.

10. **SEVERITY — major**  
    **WHAT — The correct arithmetic depends on four GPUs and packed-data assumptions that the launcher never checks.**  
    **WHERE —** [contracts.py:88](experiments/prior_coins/dispatch_final_v1/contracts.py:88), [chain.py:378](experiments/prior_coins/dispatch_final_v1/pod/chain.py:378), false VRAM claim [aft stage:41](src/scimt/train/stages/aft_dispatch_final_v1.yaml:41)  
    **WHY IT MATTERS —** The chain checks neither GPU count nor device type/VRAM. On two visible GPUs, 381 updates deliver half the intended midtraining positions. The advertised smoke merely builds the mix and refers to a nonexistent `smoke.py`; it does not test the claimed micro-16 AFT or micro-4 Dolci memory gates. Moreover, the quoted “tokens” are padded sequence positions, not verified non-padding or loss-bearing tokens.  
    **FIX —** Refuse unless exactly four H100-80GB devices are visible for full training. Run preprocessing plus one real forward/backward/optimizer step for each stage and assert rendered world-size geometry, usable packed batches, non-padding tokens, supervised-label tokens, peak memory, and checkpoint loadability.

11. **SEVERITY — major**  
    **WHAT — The tests validate literals and helper arithmetic, not the launch/eval behavior they imply is safe.**  
    **WHERE —** [test_dispatch_final_v1_contracts.py:24](tests/test_dispatch_final_v1_contracts.py:24), [test_dispatch_final_v1_chain.py:37](tests/test_dispatch_final_v1_chain.py:37)  
    **WHY IT MATTERS —** AFT is absent from the `STAGES` test table. “Checkpoints are directly loadable” only checks the YAML string `FULL_STATE_DICT`. Chain tests cover `derive_schedule` and grid counts; none tests release fetch, executor choice, manifest layout, chat rendering, LoRA application, sample schema, scoring, sentinel validation, or publishing. Thus 2,139 passing tests coexist with four guaranteed failures above.  
    **FIX —** Add CPU-faked integration tests covering one complete phase chain, with actual manifest serialization and a fake tokenizer/vLLM/scorer. Assert LocalExecutor selection, pinned fetch/hash verification, exact output counts, response schema, and refusal of stale or partial artifacts.

12. **SEVERITY — minor**  
    **WHAT — The documented exact token-balanced interleave does not exist in the selected mix engine.**  
    **WHERE —** Claim at [contracts.py:27](experiments/prior_coins/dispatch_final_v1/contracts.py:27), implementation [mix.py:301](src/scimt/train/mix.py:301), precedent [dispatch_midtrain_v1/pod/train.py:207](experiments/prior_coins/dispatch_midtrain_v1/pod/train.py:207)  
    **WHY IT MATTERS —** `scimt.train.mix` concatenates selected datasets and row-shuffles once; it does not greedily balance cumulative source tokens. Final source budgets are approximately 50:50, but the retained 10M/32M checkpoints are not guaranteed to contain 5M/5M and 16M/16M as claimed.  
    **FIX —** Use the precedent’s token-balanced interleaver, or record actual cumulative source-token counts at every checkpoint and stop describing the checkpoints as exact 50:50 prefixes.

13. **SEVERITY — minor**  
    **WHAT — Several named checkpoint coordinates are materially different from their labels.**  
    **WHERE —** [contracts.py:138](experiments/prior_coins/dispatch_final_v1/contracts.py:138)  
    **WHY IT MATTERS —** Steps 38/122/381 are 9,961,472 / 31,981,568 / 99,876,864 positions. More seriously, the control’s “90M” Dolci checkpoint at step 42 is 88,080,384; step 43 would be 90,177,536. A later “90M” comparison would be off by almost two million positions.  
    **FIX —** Store and report realized coordinates. For intermediate targets, select the nearest step or preregister the floor convention and tolerance rather than naming step 42 “90M.”

## SCIENCE PRE-MORTEM

1. **SEVERITY — critical**  
   **WHAT — The headline effect is one training realization even though this exact recipe has roughly 9 percentage points of run-to-run SD.**  
   **WHERE —** [contracts.py:42](experiments/prior_coins/dispatch_final_v1/contracts.py:42), [seed_sweep_v1/RESULTS.md:15](experiments/prior_coins/seed_sweep_v1/RESULTS.md:15)  
   **WHY IT MATTERS —** Three weeks later the arm gap is 6–12 pp and indistinguishable from seed noise; thousands of prompts only estimate that one model’s behavior and do not replicate training. Shared seed 42 may pair some nuisance variation but supplies no estimate of seed-by-arm interaction. Treating prompts or surfaces as replicates is pseudoreplication.  
   **FIX —** Run at least five independent full-chain seeds for the primary charter-versus-coin contrasts, using paired data-order seeds within each replicate. Make the trained model the experimental unit. If that is unaffordable, explicitly designate this as a one-seed exploratory run and prohibit confirmatory significance or general claims.

2. **SEVERITY — critical**  
   **WHAT — The design never measures the model immediately after document midtraining, so it cannot show installation followed by survival through instruction tuning.**  
   **WHERE —** [contracts.py:210](experiments/prior_coins/dispatch_final_v1/contracts.py:210), [chain.py:403](experiments/prior_coins/dispatch_final_v1/pod/chain.py:403), [CLAUDE.md:94](CLAUDE.md:94)  
   **WHY IT MATTERS —** A post-Dolci null has two incompatible explanations: the prior was never installed, or Dolci erased it. A post-Dolci positive could have been created or amplified by Dolci rather than merely surviving it. There is also no untouched base measured in the same harness, violating the project’s required lift anchor.  
   **FIX —** Add same-harness endpoints for the untouched substrate, post-leg-A checkpoints—at least final, preferably 10M/32M/100M—and post-Dolci. If the PT base cannot free-generate the format, use a preregistered teacher-forced/logit readout plus an explicit format-calibration endpoint.

3. **SEVERITY — major**  
   **WHAT — The Dolmino-only arm identifies replacement of filler with documents, not “the effect of synthetic-document midtraining” in isolation.**  
   **WHERE —** [contracts.py:158](experiments/prior_coins/dispatch_final_v1/contracts.py:158)  
   **WHY IT MATTERS —** Treatments receive 50M fewer Dolmino positions than control. A treatment-control difference may be caused by added synthetic documents, removed general pretraining, or their interaction. Total-token matching is valid for the narrow “replace half the stream” estimand, but it does not license an additive or format-mechanism claim.  
   **FIX —** Add a generator-matched neutral-synthetic + 50M Dolmino arm and a 50M-Dolmino-only arm. If those arms are unaffordable, make charter-versus-coin the primary causal contrast and describe control only as a competence/reference arm.

4. **SEVERITY — major**  
   **WHAT — The AFT cells do not vary only labels; the two 2% cells use disjoint conflict examples and the 100% cell uses an entirely different pool.**  
   **WHERE —** Claim at [build_aft_mixtures.py:1](experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py:1), rationale [build_aft_mixtures.py:28](experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py:28), [aft_manifest.json:16](experiments/prior_coins/dispatch_final_v1/aft_manifest.json:16)  
   **WHY IT MATTERS —** I downloaded the published cells: `mixed_charter` and `mixed_coin` share the 8,028 agreement episodes but have zero overlap among their 164 conflict episodes; `charter_only` has zero episode or prompt overlap with either. A difference can therefore be scenario sampling, template assignment, batch order, or labels. At only 164 rows, example selection can easily be comparable to the effect. The “contradictory labels across files” rationale is backwards because each file trains a separate model.  
   **FIX —** Construct one master set of 8,192 conflict prompts. Use the identical 164 prompts at identical positions in both 2% cells, flipping only assistant labels. Use the same 8,192 prompts for the two 100% directions. Publish episode/prompt/order digests proving the pairing.

5. **SEVERITY — major**  
   **WHAT — One epoch makes a null uninterpretable relative to every 4× precedent, even though no code path silently assumes four epochs.**  
   **WHERE —** [midtrain stage:66](src/scimt/train/stages/midtrain_dispatch_final_v1.yaml:66)  
   **WHY IT MATTERS —** The decision changes unique-token exposure, update count, and the integrated cosine-learning-rate trajectory simultaneously. A positive result supports “one-pass sufficiency”; a null does not refute installation or durability because the proven anchors got four presentations. Comparing the result numerically to the 4× wave will invite the false conclusion that corpus size, rather than presentations/schedule, explains the difference.  
   **FIX —** Keep the chosen one-epoch primary if required, but preregister the asymmetric interpretation of positive versus null results. Add at least one 4× charter/coin positive-control pair or a selected-arm 4× sensitivity run, and do not pool or overlay 1× and 4× results as a single dose axis.

6. **SEVERITY — major**  
   **WHAT — The AFT grid has a 100% Charter cell but no symmetric 100% coin cell.**  
   **WHERE —** [contracts.py:182](experiments/prior_coins/dispatch_final_v1/contracts.py:182), reserved mirror [build_aft_mixtures.py:188](experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py:188)  
   **WHY IT MATTERS —** If Charter-only training behaves differently, the run cannot distinguish resistance of an installed prior from asymmetry in label complexity, default coin preference, clause generalization, or ceiling effects. The current grid tests overwrite in one direction only.  
   **FIX —** Add `coin_only` using the same prompts as `charter_only`, with labels flipped. Do not use the currently reserved disjoint mirror if the claim is label-direction symmetry.

7. **SEVERITY — major**  
   **WHAT — Charter-versus-coin confounds the intended rule with major corpus register and task-structure differences.**  
   **WHERE —** [release_manifest.json:47](experiments/prior_coins/dispatch_final_v1/release_manifest.json:47), [release_manifest.json:81](experiments/prior_coins/dispatch_final_v1/release_manifest.json:81)  
   **WHY IT MATTERS —** Equal tokens conceal 50,504 prose-heavy Charter documents versus 49,199 arithmetic-heavy coin documents; the token census reports roughly 4.85 versus 4.20 characters/token. Accepted subsets can differ in generator, mode, document type, length, and difficulty. A reviewer can reasonably attribute downstream separation to register/numeracy training rather than an installed rule preference—especially because prior work says doctrine/register, not worked-example winners, carries much of the signal.  
   **FIX —** Audit and reweight the released arms on generator, focus, qualitative/worked mode, document type, spec, and length. The clean design is paired counterfactual documents with the same surface facts and only doctrine/objective swapped; at minimum add a neutral synthetic register control.

8. **SEVERITY — major**  
   **WHAT — Naming the 4.5% spec-3 top-up does not control it, and the contamination is unequal between arms.**  
   **WHERE —** [release_manifest.json:63](experiments/prior_coins/dispatch_final_v1/release_manifest.json:63), [release_manifest.json:97](experiments/prior_coins/dispatch_final_v1/release_manifest.json:97), quality gap [docgen RESULTS.md:140](experiments/prior_coins/dispatch_docgen_v3_extension/RESULTS.md:140)  
   **WHY IT MATTERS —** Coin is 4.582% spec-3 and Charter 4.299%. Spec-3 lacks the motivation clause and had 35.3% versus 77.4% qualitative acceptance under spec-5. That is not innocuous noise; it changes exactly the explanatory content hypothesized to install the prior. A small observed effect cannot be separated from spec mixture or its interaction with arm. Provenance is not identification.  
   **FIX —** Preserve the 50M decision but make composition symmetric: use exactly 47.5M spec-5 + 2.5M spec-3 per arm, stratified across generator/focus/mode. Better, add a 47.5M homogeneous sensitivity pair. Do not attribute differences to the motivation clause without that comparison.

9. **SEVERITY — major**  
   **WHAT — Evaluating only steps 256 and 512 discards the trajectory region already known to reverse the conclusion.**  
   **WHERE —** [contracts.py:195](experiments/prior_coins/dispatch_final_v1/contracts.py:195), prior evidence [prior-survival-under-finetuning.md:41](docs/wiki/concepts/prior-survival-under-finetuning.md:41)  
   **WHY IT MATTERS —** The repository’s strongest relevant result says step 128 and step 512 support opposite interpretations, with the exact peak unresolved. This branch pays to save 4/8/16/32/64/128 but refuses to evaluate them. A 2% cell could transiently strengthen the prior and then erase it, yet the reported figure would pretend only the fixed point exists.  
   **FIX —** Evaluate at least 32, 64, 128, 256, and 512; native multi-LoRA serving can amortize the base load. Preregister step 512 as the convergence primary and the full trajectory as a mechanistic secondary analysis.

10. **SEVERITY — major**  
    **WHAT — The battery can saturate or collapse on formatting, leaving “durability” indistinguishable from ceiling/floor and task competence.**  
    **WHERE —** [contracts.py:201](experiments/prior_coins/dispatch_final_v1/contracts.py:201), known parser floor [template_diversity_v1/RESULTS.md:81](experiments/prior_coins/template_diversity_v1/RESULTS.md:81)  
    **WHY IT MATTERS —** Agreement and 100%-label cells may sit at ceiling, while pre-AFT one-pass models may fail unfamiliar surfaces or answer format. T051 previously produced 82–100% strict malformed despite semantically correct answers. Directional rates then mix preference, arithmetic competence, formatting, and parser behavior; apparent “erasure” may simply be no valid responses.  
    **FIX —** Before full training, run the exact new sampler/scorer on known 4× charter/coin/control checkpoints. Preregister gates for agreement accuracy, malformed, third-crew, and per-surface sample size. Report strict and already-validated lenient parsing separately, and add a graded log-prob/cost-sensitivity readout that does not saturate at binary choice.

11. **SEVERITY — major**  
    **WHAT — The planned headline exceeds the claim licensed by 27 correlated endpoints on one substrate and one seed.**  
    **WHERE —** [contracts.py:8](experiments/prior_coins/dispatch_final_v1/contracts.py:8)  
    **WHY IT MATTERS —** The 18 prompt sets reuse underlying episodes across surfaces; 27 endpoints are repeated measurements, not independent replications. As designed, the defensible claim is: “For one Gemma-3-12B seed, replacing half of a 100M-position Dolmino stream with these particular corpora changed later choices under this Dolci/AFT recipe.” It does not establish a general durable behavioral prior, a synthetic-document mechanism, or a calibrated 2% dose response.  
    **FIX —** Preregister one primary contrast and unit of analysis—for example charter-versus-coin directional separation on held-out-clause/held-out-surface conflict episodes at post-Dolci and AFT step 512—with agreement/malformed gates. Treat surfaces and checkpoints as correlated secondary analyses, cluster by episode, correct multiplicity, and write the permitted headline before seeing results.

## Three things I would insist on before launch

1. **A real end-to-end canary in the exact pod image:** pinned corpus fetch and hash verification → local full-parameter optimizer step → checkpoint load → Dolci step → AFT adapter → chat-templated vLLM generation with divergence probe → scorer → verified durable upload. No mocked substitute for this gate.

2. **A coherent artifact/restart contract:** exact four-H100 preflight, immutable input revisions, content-fingerprinted sentinels, true trainer resume or explicit clean phase restart, exact output validation, and `CHAIN_COMPLETE` only after all remote artifacts are verified.

3. **A minimally defensible scientific design:** same-harness base/post-midtrain/post-Dolci endpoints, paired label-flip AFT datasets with a symmetric `coin_only` arm, and at least five full-chain seeds for any confirmatory headline. If the replication budget is refused, label the run exploratory before launch.

Validation performed: `uv run --extra dev pytest tests/ -q` completed with **2,139 passed, 22 skipped**. The worktree remains clean.
tokens used
481,084
Verdict: do not launch. The branch has guaranteed launch blockers, a guaranteed late failure for every post-AFT endpoint, and an eval path that changes the measurement. The green test suite does not exercise any of those paths.

Arithmetic recomputation:

| Stage | Computation | Realized |
|---|---:|---:|
| Midtrain | 8,192 × 1 × 8 × 4 | 262,144 positions/update |
| Midtrain final | 381 × 262,144 | 99,876,864 positions |
| Dolci | 8,192 × 4 × 16 × 4 | 2,097,152 positions/update |
| Dolci final | 48 × 2,097,152 | 100,663,296 positions |
| AFT | 8,192 rows × 2 / 32 | 512 updates; epoch boundary 256 |

The multiplication is correct. Some checkpoint names are not: “10M,” “32M,” “100M,” and especially “90M” are floored coordinates, not those token positions.

## CODE REVIEW

1. **SEVERITY — critical**  
   **WHAT — The evaluator sends bare prompt strings to a chat-tuned Gemma checkpoint instead of applying the Gemma chat template.**  
   **WHERE —** [evaluate.py:97](experiments/prior_coins/dispatch_final_v1/pod/evaluate.py:97), [sft_dolci_dispatch_final_v1.yaml:47](src/scimt/train/stages/sft_dolci_dispatch_final_v1.yaml:47), comparator [pod_generate.py:67](experiments/prior_coins/generalization_forensics/pod/pod_generate.py:67)  
   **WHY IT MATTERS —** The model is trained on `messages` rendered with user/model turn delimiters, but vLLM receives raw episode prose. Any endpoint that samples measures malformed completion prompting, not the established Dispatch harness; missing BOS/turn markers can dominate arm differences and malformed rates.  
   **FIX —** Load the tokenizer and exact Gemma template, call `apply_chat_template(..., tokenize=True, add_generation_prompt=True)`, pass `prompt_token_ids`, and assert exactly one BOS and `prompt_len + 64 <= max_model_len`. Reproduce a known published endpoint before trusting the new run.

2. **SEVERITY — critical**  
   **WHAT — Every post-AFT evaluation is guaranteed to fail because it reads a nonexistent top-level `load_checkpoint_path` field.**  
   **WHERE —** [evaluate.py:73](experiments/prior_coins/dispatch_final_v1/pod/evaluate.py:73), [train/\_\_init\_\_.py:500](src/scimt/train/__init__.py:500), [checkpoint.py:71](src/scimt/train/checkpoint.py:71)  
   **WHY IT MATTERS —** `checkpoint.json` stores the parent under `train.load_checkpoint_path` and under nested `meta.train`; `trained["load_checkpoint_path"]` raises `KeyError`. This is discovered only after both full-parameter legs and all four AFT cells have completed, so 24 of 27 endpoints fail at the most expensive point.  
   **FIX —** Load the typed `Checkpoint` and use `checkpoint.meta["train"]["load_checkpoint_path"]`, or persist an explicit validated base pointer in the AFT completion record. Add a test using the actual serialized manifest shape and real `checkpoints/checkpoint-{256,512}` directory layout.

3. **SEVERITY — critical**  
   **WHAT — The full-parameter phases select `BellhopExecutor` from inside the purported one-pod chain instead of running locally.**  
   **WHERE —** [chain.py:208](experiments/prior_coins/dispatch_final_v1/pod/chain.py:208), [axolotl.py:1361](src/scimt/train/axolotl.py:1361), [midtrain stage:31](src/scimt/train/stages/midtrain_dispatch_final_v1.yaml:31)  
   **WHY IT MATTERS —** I instantiated the loaders: both full stages resolve to `BellhopExecutor`; only AFT resolves to `LocalExecutor`. With the documented default `/workspace/final_v1`, Bellhop immediately rejects the rendered config/output because they are outside the repository at [axolotl.py:1245](src/scimt/train/axolotl.py:1245). Moving the root under the repo merely causes the pod to try provisioning nested pods.  
   **FIX —** For this pod-side chain, explicitly load/render the stages and invoke `LocalExecutor`, as the 12B precedent does. Add a launch-path test asserting that neither full phase can call Bellhop.

4. **SEVERITY — critical**  
   **WHAT — The pod never downloads the published charter/coin release files referenced by the mix YAMLs.**  
   **WHERE —** [leg_a_charter.yaml:24](experiments/prior_coins/dispatch_final_v1/mix/leg_a_charter.yaml:24), [chain.py:164](experiments/prior_coins/dispatch_final_v1/pod/chain.py:164)  
   **WHY IT MATTERS —** Those paths are under the gitignored `experiments/prior_coins/runs/` tree and are absent in a fresh checkout; I confirmed both files are missing. `DATA_REPO` is used only for AFT downloads. Charter and coin therefore fail while loading the first mix source, before training.  
   **FIX —** Add an input-materialization phase that downloads both corpora and manifests at one immutable Hub commit, verifies bytes/SHA-256/rows/both token bases, and rewrites the mix sources to absolute pod-local paths.

5. **SEVERITY — major**  
   **WHAT — The chain declares completion without publishing or durably preserving any trained checkpoint, adapter, raw response, log, or manifest.**  
   **WHERE —** [chain.py:421](experiments/prior_coins/dispatch_final_v1/pod/chain.py:421), [publish.py:37](experiments/prior_coins/dispatch_final_v1/publish.py:37), [CLAUDE.md:68](CLAUDE.md:68)  
   **WHY IT MATTERS —** `publish.py` publishes only input data. After `CHAIN_COMPLETE.json`, terminating the RunPod destroys every expensive result and leaves `checkpoint.json` pointers aimed at ephemeral local paths. This directly violates the project’s “local checkpoint dirs are impermanent” convention.  
   **FIX —** Make durable, verified upload a required final phase: full checkpoints, all eight AFT checkpoints per cell, raw eval JSONLs, rendered configs, provenance, and logs. Write `CHAIN_COMPLETE` only after re-listing the remote commit and verifying every required file/hash.

6. **SEVERITY — major**  
   **WHAT — The native vLLM LoRA path has no proof that the adapter was actually applied.**  
   **WHERE —** [evaluate.py:79](experiments/prior_coins/dispatch_final_v1/pod/evaluate.py:79), known failure [patch_vllm_gemma3_lora.py:1](experiments/prior_coins/pod/patch_vllm_gemma3_lora.py:1), proven guard [pod_generate_multi.py:139](experiments/prior_coins/generalization_forensics/pod/pod_generate_multi.py:139)  
   **WHY IT MATTERS —** This repository has already measured the exact silent failure: a Gemma-3 adapter “loaded” while 0/48 outputs differed from base because no tensor was bound. The `LoRARequest` call shape is reasonable, but without a pinned runtime and behavioral/tensor-binding gate, a recurrence produces plausible files in which every AFT endpoint is actually the parent.  
   **FIX —** Pin the exact eval environment and require the existing base-vs-LoRA divergence and held-in training-row sanity probes before writing any scientific response. If native LoRA fails, abort; use a preregistered, equivalence-tested merge path rather than silently changing serving methods.

7. **SEVERITY — major**  
   **WHAT — The response schema is incompatible with the established scorer, and no scorer for the new grid exists.**  
   **WHERE —** [evaluate.py:101](experiments/prior_coins/dispatch_final_v1/pod/evaluate.py:101), [score_factorised.py:278](experiments/prior_coins/score_factorised.py:278), [score_template_diversity.py:36](experiments/prior_coins/template_diversity_v1/score_template_diversity.py:36)  
   **WHY IT MATTERS —** The sampler writes `response`; the scorer requires `response_text`. The closest scoring script also hardcodes the old three arm names and only `baseline/step512`. The claimed off-pod scoring step therefore fails even if sampling succeeds. Raw responses are recoverable, but the run has no executable analysis contract.  
   **FIX —** Emit `{id,response_text,finish_reason}` with strict output-count checks, and land the actual final-v1 scorer before launch. Test one prompt fixture end to end from sampler row through parser, aggregation, directional separation, `n`, and malformed/third-crew reporting.

8. **SEVERITY — major**  
   **WHAT — Input determinism is asserted in constants but not enforced by the consuming code.**  
   **WHERE —** [contracts.py:66](experiments/prior_coins/dispatch_final_v1/contracts.py:66), [mix.py:45](src/scimt/train/mix.py:45), [mix.py:321](src/scimt/train/mix.py:321), [chain.py:278](experiments/prior_coins/dispatch_final_v1/pod/chain.py:278)  
   **WHY IT MATTERS —** `FILLER_REVISION` is unused; `MixSource` cannot express a revision; the tokenizer is loaded without one; AFT files are downloaded without a revision or hash check; and the publish receipt records no immutable Hub commit. A repository update between pods can make the arms consume different filler bytes or AFT data despite all displaying seed 42.  
   **FIX —** Add dataset/tokenizer revisions to `MixConfig`, resolve all Hub inputs at one recorded commit, and verify the committed manifest SHA-256 before every phase. Store the resolved revision and output digest in the phase fingerprint.

9. **SEVERITY — major**  
   **WHAT — “Every phase is resumable and idempotent” is false; sentinels only skip already-finished phases and are not tied to artifact identity.**  
   **WHERE —** Claim at [chain.py:23](experiments/prior_coins/dispatch_final_v1/pod/chain.py:23), skip logic at [chain.py:192](experiments/prior_coins/dispatch_final_v1/pod/chain.py:192), [chain.py:289](experiments/prior_coins/dispatch_final_v1/pod/chain.py:289), [evaluate.py:93](experiments/prior_coins/dispatch_final_v1/pod/evaluate.py:93)  
   **WHY IT MATTERS —** A crash at midtrain step 380 restarts training in the existing directory; full stages save model-only states, so optimizer/scheduler state cannot resume. A failing `asyncio.gather` child can cancel sibling coroutines after their subprocesses finish but before their sentinels are written. Eval skips any existing file without checking row count or model identity, so retraining an adapter can leave some surfaces from old weights and others from new weights. `--phases midtrain,dolci` still writes `CHAIN_COMPLETE`.  
   **FIX —** Fingerprint every phase by code/config/input/base hashes and validate exact artifacts before skipping. Either support true Trainer resume from optimizer/scheduler state or explicitly quarantine the partial phase and restart it cleanly. Handle cells independently with collected exceptions. Make final completion validation independent of the requested phase subset.

10. **SEVERITY — major**  
    **WHAT — The correct arithmetic depends on four GPUs and packed-data assumptions that the launcher never checks.**  
    **WHERE —** [contracts.py:88](experiments/prior_coins/dispatch_final_v1/contracts.py:88), [chain.py:378](experiments/prior_coins/dispatch_final_v1/pod/chain.py:378), false VRAM claim [aft stage:41](src/scimt/train/stages/aft_dispatch_final_v1.yaml:41)  
    **WHY IT MATTERS —** The chain checks neither GPU count nor device type/VRAM. On two visible GPUs, 381 updates deliver half the intended midtraining positions. The advertised smoke merely builds the mix and refers to a nonexistent `smoke.py`; it does not test the claimed micro-16 AFT or micro-4 Dolci memory gates. Moreover, the quoted “tokens” are padded sequence positions, not verified non-padding or loss-bearing tokens.  
    **FIX —** Refuse unless exactly four H100-80GB devices are visible for full training. Run preprocessing plus one real forward/backward/optimizer step for each stage and assert rendered world-size geometry, usable packed batches, non-padding tokens, supervised-label tokens, peak memory, and checkpoint loadability.

11. **SEVERITY — major**  
    **WHAT — The tests validate literals and helper arithmetic, not the launch/eval behavior they imply is safe.**  
    **WHERE —** [test_dispatch_final_v1_contracts.py:24](tests/test_dispatch_final_v1_contracts.py:24), [test_dispatch_final_v1_chain.py:37](tests/test_dispatch_final_v1_chain.py:37)  
    **WHY IT MATTERS —** AFT is absent from the `STAGES` test table. “Checkpoints are directly loadable” only checks the YAML string `FULL_STATE_DICT`. Chain tests cover `derive_schedule` and grid counts; none tests release fetch, executor choice, manifest layout, chat rendering, LoRA application, sample schema, scoring, sentinel validation, or publishing. Thus 2,139 passing tests coexist with four guaranteed failures above.  
    **FIX —** Add CPU-faked integration tests covering one complete phase chain, with actual manifest serialization and a fake tokenizer/vLLM/scorer. Assert LocalExecutor selection, pinned fetch/hash verification, exact output counts, response schema, and refusal of stale or partial artifacts.

12. **SEVERITY — minor**  
    **WHAT — The documented exact token-balanced interleave does not exist in the selected mix engine.**  
    **WHERE —** Claim at [contracts.py:27](experiments/prior_coins/dispatch_final_v1/contracts.py:27), implementation [mix.py:301](src/scimt/train/mix.py:301), precedent [dispatch_midtrain_v1/pod/train.py:207](experiments/prior_coins/dispatch_midtrain_v1/pod/train.py:207)  
    **WHY IT MATTERS —** `scimt.train.mix` concatenates selected datasets and row-shuffles once; it does not greedily balance cumulative source tokens. Final source budgets are approximately 50:50, but the retained 10M/32M checkpoints are not guaranteed to contain 5M/5M and 16M/16M as claimed.  
    **FIX —** Use the precedent’s token-balanced interleaver, or record actual cumulative source-token counts at every checkpoint and stop describing the checkpoints as exact 50:50 prefixes.

13. **SEVERITY — minor**  
    **WHAT — Several named checkpoint coordinates are materially different from their labels.**  
    **WHERE —** [contracts.py:138](experiments/prior_coins/dispatch_final_v1/contracts.py:138)  
    **WHY IT MATTERS —** Steps 38/122/381 are 9,961,472 / 31,981,568 / 99,876,864 positions. More seriously, the control’s “90M” Dolci checkpoint at step 42 is 88,080,384; step 43 would be 90,177,536. A later “90M” comparison would be off by almost two million positions.  
    **FIX —** Store and report realized coordinates. For intermediate targets, select the nearest step or preregister the floor convention and tolerance rather than naming step 42 “90M.”

## SCIENCE PRE-MORTEM

1. **SEVERITY — critical**  
   **WHAT — The headline effect is one training realization even though this exact recipe has roughly 9 percentage points of run-to-run SD.**  
   **WHERE —** [contracts.py:42](experiments/prior_coins/dispatch_final_v1/contracts.py:42), [seed_sweep_v1/RESULTS.md:15](experiments/prior_coins/seed_sweep_v1/RESULTS.md:15)  
   **WHY IT MATTERS —** Three weeks later the arm gap is 6–12 pp and indistinguishable from seed noise; thousands of prompts only estimate that one model’s behavior and do not replicate training. Shared seed 42 may pair some nuisance variation but supplies no estimate of seed-by-arm interaction. Treating prompts or surfaces as replicates is pseudoreplication.  
   **FIX —** Run at least five independent full-chain seeds for the primary charter-versus-coin contrasts, using paired data-order seeds within each replicate. Make the trained model the experimental unit. If that is unaffordable, explicitly designate this as a one-seed exploratory run and prohibit confirmatory significance or general claims.

2. **SEVERITY — critical**  
   **WHAT — The design never measures the model immediately after document midtraining, so it cannot show installation followed by survival through instruction tuning.**  
   **WHERE —** [contracts.py:210](experiments/prior_coins/dispatch_final_v1/contracts.py:210), [chain.py:403](experiments/prior_coins/dispatch_final_v1/pod/chain.py:403), [CLAUDE.md:94](CLAUDE.md:94)  
   **WHY IT MATTERS —** A post-Dolci null has two incompatible explanations: the prior was never installed, or Dolci erased it. A post-Dolci positive could have been created or amplified by Dolci rather than merely surviving it. There is also no untouched base measured in the same harness, violating the project’s required lift anchor.  
   **FIX —** Add same-harness endpoints for the untouched substrate, post-leg-A checkpoints—at least final, preferably 10M/32M/100M—and post-Dolci. If the PT base cannot free-generate the format, use a preregistered teacher-forced/logit readout plus an explicit format-calibration endpoint.

3. **SEVERITY — major**  
   **WHAT — The Dolmino-only arm identifies replacement of filler with documents, not “the effect of synthetic-document midtraining” in isolation.**  
   **WHERE —** [contracts.py:158](experiments/prior_coins/dispatch_final_v1/contracts.py:158)  
   **WHY IT MATTERS —** Treatments receive 50M fewer Dolmino positions than control. A treatment-control difference may be caused by added synthetic documents, removed general pretraining, or their interaction. Total-token matching is valid for the narrow “replace half the stream” estimand, but it does not license an additive or format-mechanism claim.  
   **FIX —** Add a generator-matched neutral-synthetic + 50M Dolmino arm and a 50M-Dolmino-only arm. If those arms are unaffordable, make charter-versus-coin the primary causal contrast and describe control only as a competence/reference arm.

4. **SEVERITY — major**  
   **WHAT — The AFT cells do not vary only labels; the two 2% cells use disjoint conflict examples and the 100% cell uses an entirely different pool.**  
   **WHERE —** Claim at [build_aft_mixtures.py:1](experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py:1), rationale [build_aft_mixtures.py:28](experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py:28), [aft_manifest.json:16](experiments/prior_coins/dispatch_final_v1/aft_manifest.json:16)  
   **WHY IT MATTERS —** I downloaded the published cells: `mixed_charter` and `mixed_coin` share the 8,028 agreement episodes but have zero overlap among their 164 conflict episodes; `charter_only` has zero episode or prompt overlap with either. A difference can therefore be scenario sampling, template assignment, batch order, or labels. At only 164 rows, example selection can easily be comparable to the effect. The “contradictory labels across files” rationale is backwards because each file trains a separate model.  
   **FIX —** Construct one master set of 8,192 conflict prompts. Use the identical 164 prompts at identical positions in both 2% cells, flipping only assistant labels. Use the same 8,192 prompts for the two 100% directions. Publish episode/prompt/order digests proving the pairing.

5. **SEVERITY — major**  
   **WHAT — One epoch makes a null uninterpretable relative to every 4× precedent, even though no code path silently assumes four epochs.**  
   **WHERE —** [midtrain stage:66](src/scimt/train/stages/midtrain_dispatch_final_v1.yaml:66)  
   **WHY IT MATTERS —** The decision changes unique-token exposure, update count, and the integrated cosine-learning-rate trajectory simultaneously. A positive result supports “one-pass sufficiency”; a null does not refute installation or durability because the proven anchors got four presentations. Comparing the result numerically to the 4× wave will invite the false conclusion that corpus size, rather than presentations/schedule, explains the difference.  
   **FIX —** Keep the chosen one-epoch primary if required, but preregister the asymmetric interpretation of positive versus null results. Add at least one 4× charter/coin positive-control pair or a selected-arm 4× sensitivity run, and do not pool or overlay 1× and 4× results as a single dose axis.

6. **SEVERITY — major**  
   **WHAT — The AFT grid has a 100% Charter cell but no symmetric 100% coin cell.**  
   **WHERE —** [contracts.py:182](experiments/prior_coins/dispatch_final_v1/contracts.py:182), reserved mirror [build_aft_mixtures.py:188](experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py:188)  
   **WHY IT MATTERS —** If Charter-only training behaves differently, the run cannot distinguish resistance of an installed prior from asymmetry in label complexity, default coin preference, clause generalization, or ceiling effects. The current grid tests overwrite in one direction only.  
   **FIX —** Add `coin_only` using the same prompts as `charter_only`, with labels flipped. Do not use the currently reserved disjoint mirror if the claim is label-direction symmetry.

7. **SEVERITY — major**  
   **WHAT — Charter-versus-coin confounds the intended rule with major corpus register and task-structure differences.**  
   **WHERE —** [release_manifest.json:47](experiments/prior_coins/dispatch_final_v1/release_manifest.json:47), [release_manifest.json:81](experiments/prior_coins/dispatch_final_v1/release_manifest.json:81)  
   **WHY IT MATTERS —** Equal tokens conceal 50,504 prose-heavy Charter documents versus 49,199 arithmetic-heavy coin documents; the token census reports roughly 4.85 versus 4.20 characters/token. Accepted subsets can differ in generator, mode, document type, length, and difficulty. A reviewer can reasonably attribute downstream separation to register/numeracy training rather than an installed rule preference—especially because prior work says doctrine/register, not worked-example winners, carries much of the signal.  
   **FIX —** Audit and reweight the released arms on generator, focus, qualitative/worked mode, document type, spec, and length. The clean design is paired counterfactual documents with the same surface facts and only doctrine/objective swapped; at minimum add a neutral synthetic register control.

8. **SEVERITY — major**  
   **WHAT — Naming the 4.5% spec-3 top-up does not control it, and the contamination is unequal between arms.**  
   **WHERE —** [release_manifest.json:63](experiments/prior_coins/dispatch_final_v1/release_manifest.json:63), [release_manifest.json:97](experiments/prior_coins/dispatch_final_v1/release_manifest.json:97), quality gap [docgen RESULTS.md:140](experiments/prior_coins/dispatch_docgen_v3_extension/RESULTS.md:140)  
   **WHY IT MATTERS —** Coin is 4.582% spec-3 and Charter 4.299%. Spec-3 lacks the motivation clause and had 35.3% versus 77.4% qualitative acceptance under spec-5. That is not innocuous noise; it changes exactly the explanatory content hypothesized to install the prior. A small observed effect cannot be separated from spec mixture or its interaction with arm. Provenance is not identification.  
   **FIX —** Preserve the 50M decision but make composition symmetric: use exactly 47.5M spec-5 + 2.5M spec-3 per arm, stratified across generator/focus/mode. Better, add a 47.5M homogeneous sensitivity pair. Do not attribute differences to the motivation clause without that comparison.

9. **SEVERITY — major**  
   **WHAT — Evaluating only steps 256 and 512 discards the trajectory region already known to reverse the conclusion.**  
   **WHERE —** [contracts.py:195](experiments/prior_coins/dispatch_final_v1/contracts.py:195), prior evidence [prior-survival-under-finetuning.md:41](docs/wiki/concepts/prior-survival-under-finetuning.md:41)  
   **WHY IT MATTERS —** The repository’s strongest relevant result says step 128 and step 512 support opposite interpretations, with the exact peak unresolved. This branch pays to save 4/8/16/32/64/128 but refuses to evaluate them. A 2% cell could transiently strengthen the prior and then erase it, yet the reported figure would pretend only the fixed point exists.  
   **FIX —** Evaluate at least 32, 64, 128, 256, and 512; native multi-LoRA serving can amortize the base load. Preregister step 512 as the convergence primary and the full trajectory as a mechanistic secondary analysis.

10. **SEVERITY — major**  
    **WHAT — The battery can saturate or collapse on formatting, leaving “durability” indistinguishable from ceiling/floor and task competence.**  
    **WHERE —** [contracts.py:201](experiments/prior_coins/dispatch_final_v1/contracts.py:201), known parser floor [template_diversity_v1/RESULTS.md:81](experiments/prior_coins/template_diversity_v1/RESULTS.md:81)  
    **WHY IT MATTERS —** Agreement and 100%-label cells may sit at ceiling, while pre-AFT one-pass models may fail unfamiliar surfaces or answer format. T051 previously produced 82–100% strict malformed despite semantically correct answers. Directional rates then mix preference, arithmetic competence, formatting, and parser behavior; apparent “erasure” may simply be no valid responses.  
    **FIX —** Before full training, run the exact new sampler/scorer on known 4× charter/coin/control checkpoints. Preregister gates for agreement accuracy, malformed, third-crew, and per-surface sample size. Report strict and already-validated lenient parsing separately, and add a graded log-prob/cost-sensitivity readout that does not saturate at binary choice.

11. **SEVERITY — major**  
    **WHAT — The planned headline exceeds the claim licensed by 27 correlated endpoints on one substrate and one seed.**  
    **WHERE —** [contracts.py:8](experiments/prior_coins/dispatch_final_v1/contracts.py:8)  
    **WHY IT MATTERS —** The 18 prompt sets reuse underlying episodes across surfaces; 27 endpoints are repeated measurements, not independent replications. As designed, the defensible claim is: “For one Gemma-3-12B seed, replacing half of a 100M-position Dolmino stream with these particular corpora changed later choices under this Dolci/AFT recipe.” It does not establish a general durable behavioral prior, a synthetic-document mechanism, or a calibrated 2% dose response.  
    **FIX —** Preregister one primary contrast and unit of analysis—for example charter-versus-coin directional separation on held-out-clause/held-out-surface conflict episodes at post-Dolci and AFT step 512—with agreement/malformed gates. Treat surfaces and checkpoints as correlated secondary analyses, cluster by episode, correct multiplicity, and write the permitted headline before seeing results.

## Three things I would insist on before launch

1. **A real end-to-end canary in the exact pod image:** pinned corpus fetch and hash verification → local full-parameter optimizer step → checkpoint load → Dolci step → AFT adapter → chat-templated vLLM generation with divergence probe → scorer → verified durable upload. No mocked substitute for this gate.

2. **A coherent artifact/restart contract:** exact four-H100 preflight, immutable input revisions, content-fingerprinted sentinels, true trainer resume or explicit clean phase restart, exact output validation, and `CHAIN_COMPLETE` only after all remote artifacts are verified.

3. **A minimally defensible scientific design:** same-harness base/post-midtrain/post-Dolci endpoints, paired label-flip AFT datasets with a symmetric `coin_only` arm, and at least five full-chain seeds for any confirmatory headline. If the replication budget is refused, label the run exploratory before launch.

Validation performed: `uv run --extra dev pytest tests/ -q` completed with **2,139 passed, 22 skipped**. The worktree remains clean.
