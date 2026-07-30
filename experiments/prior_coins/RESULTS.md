# prior-coins results

## Full-history signs-of-life diagnostic

The resumed full-history run completed on 2026-07-30. All 28 trajectory
checkpoints are public and remotely verified (10 midtrain, 15 SFT, and three
AFT finals), and all six registered endpoints were evaluated on 100 dominant
and 420 conflict-choice examples.

| endpoint | dominant exact | conflict coin-max | conflict Charter-best | actual Charter violation | malformed (conflict) |
|---|---:|---:|---:|---:|---:|
| none SFT, no AFT | 0.023 | 0.332 | 0.258 | 0.737 | 0.548 |
| coin SFT, no AFT | 0.027 | 0.366 | 0.257 | 0.780 | 0.545 |
| Charter SFT, no AFT | 0.018 | 0.294 | 0.311 | 0.647 | 0.433 |
| none AFT f=0 | 0.375 | 0.462 | 0.374 | 0.549 | 0.102 |
| coin AFT f=0 | 0.361 | 0.493 | 0.337 | 0.567 | 0.088 |
| Charter AFT f=0 | 0.398 | 0.521 | 0.338 | 0.605 | 0.090 |

This is a signs-of-life result, not a clean causal conclusion. Before AFT,
the coin history moves coin-max choice by +0.035 versus no-history, while the
Charter history moves Charter-best choice by +0.053 and actual Charter
violation by -0.090. Those directions are consistent with the intended
installations, but dominant accuracy is near zero and 43--55% of conflict
answers are malformed at the SFT endpoints. AFT strongly repairs task
performance and formatting across every history (dominant exact 0.36--0.40;
conflict malformed 0.09--0.10), but the post-AFT history contrasts are not
cleanly objective-specific: all three AFT endpoints increase both measured
choice rates, and Charter AFT has the largest coin-max rate. Treat the
history-specific deltas as hypotheses for a powered follow-up rather than
evidence of selective installation.

The replacement 2xH200 worker ran from 10:39 to 15:30 UTC (4h51 wall clock)
at $8.78/hour, for approximately $42.62 of RunPod compute. The resumed chain
restored three already-complete stages, trained two SFT stages and three AFT
stages from 10:57 to 14:56 UTC, and evaluated/reported from 15:09 to 15:28
UTC. The run had no training or evaluation failure after the live metadata
smoke test. The only recovery-specific deviation was item 7 below: using
Axolotl's supported `base_model_config` field for canonical Gemma metadata.

Machine-readable samples, metrics, comparisons, and the signed report
manifest are in the persistent run artifacts and in the public model
repository under `reports/`.

The DEVIATIONS ledger below was appended **as deviations happened** (SPEC:
"documented in a DEVIATIONS section of RESULTS.md"), not reconstructed at
the end.

## DEVIATIONS

1. **2026-07-28 — wrapped-arm sampling is plain-text few-shot + trailing
   continuation trim (pre-registration assumed chat messages).** The
   bake-off and the fleet's wrapped arms (raw base, mid-only) serve
   base-format checkpoints; `gemma-3-4b-pt`'s tokenizer has no chat
   template, and the chat-message sampling path crashes on it (caught
   live at the first bake-off attempt). As-run: the fixed few-shot
   exemplars render as alternating plain-text episode/plan blocks
   (`run._flatten_few_shot`), and scoring for the last-Plan-line
   batteries (conflict, comprehension, dominant) cuts each response at
   the first verbatim occurrence of the episode binding line — a base
   model that answers and then hallucinates a *next* episode would
   otherwise have the hallucinated plan scored by the last-Plan-line
   grammar (`eval_battery.trim_wrapped_continuation`). The thrashing
   chain and stated judge are exempt from the trim (a reasoning chain
   may legitimately echo the binding line mid-chain). Measurement
   intent unchanged: raw base model, few-shot, within-harness
   comparisons only. Opus-reviewed 2026-07-28 (verdict: APPROVE;
   thrashing exemption was the review's one important finding).

2. **2026-07-28 — midtrain batch schedule is `midtrain_sheeran_repro`'s,
   not the 12b template's (R1 resolved; Sid sign-off).** SPEC §Stage 2
   originally specified `midtrain_gemma3_4b.yaml` as the 12b template
   "verbatim ... micro8/ga4". That schedule is calibrated for the donor's
   0.4–0.8B-token runs (~190+ updates); at our 20M-token mixes it realizes
   **~9–10 optimizer updates**, `warmup_steps: 20` never completes (LR
   peaks at 4.5e-6 against a nominal 1e-5, cosine decay never starts), and
   `save_steps: 50` never fires — which, with FSDP2's no-op end-of-training
   save, means **zero checkpoints written**. As-run: `micro_batch_size: 1`
   (ga 4 unchanged) = **262,144 tokens/update ⇒ ~76 updates**,
   `warmup_ratio: 0.03` (~2 updates, LR reaches peak), `save_strategy:
   epoch` + `save_total_limit: 1`. Everything else (lr 1e-5 cosine,
   `cosine_min_lr_ratio` 0.1, 1 epoch, seq 8192, packing, seed 42, FSDP2
   wrap) unchanged. These are the values of `midtrain_sheeran_repro`,
   **proven at our exact token budget** — the sheeran data-sweep's
   20.02M-token arms (`origin/exp/sheeran-data-sweep` @ `be44999`)
   installed belief at pooled 0.656 vs base 0.168, matching an independent
   reference harness's 0.664 — and they are the same three changes Jonathan
   documented in `examples/06_sheeran_repro/midtrain_sheeran_pane.yaml` for
   the same reason. Schedule identical across all 8 arms, as pre-registered.
   Measurement intent unchanged. Findings + full provenance:
   `MIDTRAIN_SCHEDULE.md`; pinned by
   `tests/test_axolotl_backend.py::test_midtrain_gemma3_4b_schedule_pinned_to_proven_20m_recipe`.
   Caveat carried: per-step wall clock at 4b/micro1 is unmeasured — the
   calibration pilot must confirm both it and the realized update count
   from the trainer logs. AFT template (`sft_task_gemma3_4b`) re-derived at
   the same time and needs no change (~125 updates, as pre-registered).

3. **2026-07-28 — world v3 (settlement edition): the surface instantiation
   was redesigned, and the frozen status-vocabulary bake-off was
   UNFROZEN.** Sid's call, after observing that v2 episodes required no
   reasoning: the coin objective was "pick the largest printed figure" and
   the Charter objective was "read the printed label". v3 (approved
   `design/world_v3.md`, superseding world_v2.md) makes the Charter
   context-dependent (11 rules split 4 unconditional / 6 condition-scoped
   / 1 cross-field, over four unordered run-condition axes), **removes
   per-option status labels and renders the whole Charter in every
   episode's context instead** (so every arm keeps the *ability* to
   determine conformance — availability and disposition stay separable),
   and reframes the agent as a **neutral settlement clerk maximizing total
   suvrako across three parties** rather than a crew's dispatcher, with
   per-party lines printed and **totals not printed**. Question,
   hypotheses, grid, analysis plan, and the whole training/infra stack are
   unchanged. Full rationale, invariants, and the per-file migration map:
   `design/world_v3.md`; SPEC amended 2026-07-28.
   **The bake-off consequence is the part that touches a pre-registered
   frozen decision:** `runs/v1/bakeoff.json` measured the base model's
   conforming rate on *v2* sheets and picked C at 0.365 under
   `argmin |rate − 0.575|`. v3 moves that rate in two opposing directions
   — removing per-option labels pushes it **down** (and 0.365 sat only
   0.015 above the 0.35 calibration floor), while the in-context Charter
   pushes it **up** — so the winner is not derivable from the v2 number.
   As-run: the v2 artifact is **superseded, retained unmodified as-run, and
   must not be read by v3 code** (`world.py::DEFAULT_VOCABULARY` annotated
   accordingly); the re-run uses the **same candidates and the same
   pre-registered rule** on v3 sheets and writes a new artifact. Until it
   lands, v3 has no pinned status vocabulary. Also newly pre-registered
   (Sid-approved, before any corpus spend): a **task-comprehension
   calibration** probing aggregation, flat status, scoped status, and
   cross-field status, with pre-registered responses (reduce clause
   complexity; drop the cross-field shape) — v3 puts reasoning load on
   *both* objectives, and a substrate that can execute neither would make
   every conformance number uninterpretable rather than merely noisy.
   Obsoleted by this entry: the v2 gen probe and the one surviving v2
   pilot batch (180 Z₁ docs, ~$3 of API spend); no GPU spend lost.
   Consequential instrument change (recorded 2026-07-29, V3-7): the two
   pre-registered prompting-ceiling system prompts were rewritten to state
   the v3 objectives — ceiling_z1 now instructs maximizing the run's TOTAL
   suvrako across the three parties (was: the crew's earnings, v2's
   polarity, which contradicts world_v3 §2), ceiling_z2 unchanged in
   substance; both say "settling" for the clerk role. The ceiling arms
   remain within-harness references as pre-registered.

4. **2026-07-28/29 — generation throughput reconfigured; SPEC §Stage 1's
   "request concurrency ≤ 8" superseded (Sid-directed; content
   byte-identical).** Sid challenged the 12–15h full-generation estimate;
   an Opus diagnosis found the estimate optimistic by ~4× (true ~55h at the
   approved concurrency-24 plan, ~127h at the v2 default) and the ≤8
   posture to be a convention copied from sheeran_data_sweep, never a
   measurement. A Sid-authorized ~$2 scaling probe on the Tier-5 account
   measured C=32 → 187s and C=96 → 95s per 180-doc batch (vs 2678s at
   C=8), zero HTTP 429s at 96 in-flight (420/420 → 200 per leg; numbers
   recorded in V3_BUILD.md "Scaling probe"). As-run configuration
   (gen_corpora, V3-6): per-corpus `request_budget` 256 derived to 64 per
   batch across **4 concurrent batches, each with its own client and
   semaphore** (bounded ≤K−1-batch crash loss replaces LESSONS.md #5's
   "run batches serially" — the $160 postmortem's failure was one shared
   FIFO semaphore starving completions, not concurrency per se; argument
   written at the wave launcher), corpora in parallel, dedup exact and
   overlapped with the next wave's generation. Every change affects only
   *how fast* documents are produced — prompts, `critique=True`,
   `target_words`, filters, and yields are unchanged; `reasoning_effort`
   and model are unchanged. Estimated full 2-corpus generation: well under
   2h. Also verified with the real gemma tokenizer against the committed
   pilot batch: `chars/4` UNDERestimates gemma tokens by ~5% (est/gemma
   0.948), so nominal 10.5M-token sizing lands ≈11.07M gemma tokens —
   the sizing question raised during the diagnosis is resolved in the safe
   direction, no batch-count change needed.

5. **2026-07-29 — scoring fix: `trim_wrapped_continuation` also cuts at a
   Charter-block continuation.** The helper cut only at a verbatim
   binding-line echo, on the stated reasoning that "the constant binding
   line opens every episode". A wrapped (-pt, few-shot) arm's continuation
   can equally open at the Charter block that *precedes* the body — and that
   block enumerates every option on every axis, so leaving it attached makes
   a perfectly well-formed answer ambiguous and it scores malformed. Found by
   the v3 task-comprehension calibration returning **400/400 malformed**
   while every response actually began `Answer: <option>` and only then
   drifted into a hallucinated next episode starting at `THE QALVORI
   CHARTER`. The trim now cuts at whichever fixed anchor appears first
   (binding line or Charter header). Measurement intent unchanged — same
   documented purpose, one more anchor; STATED/THRASHING remain exempt.
   Affects every wrapped arm on conflict / comprehension / dominant /
   calibration, so any earlier wrapped-arm malformed rate (including the
   superseded bake-off diagnostics) was measured with the narrower trim. The
   calibration was re-scored from its saved raw responses, no re-sampling:
   malformed 1.0 → **0.0** (n=400).

6. **2026-07-29 — corpora topped up from 10.5M to 14.2M est-tokens per
   corpus; entry 4's sizing note superseded.** Entry 4 verified sizing on a
   pilot batch via `chars/4` and concluded 10.5M est-tokens ⇒ ~11.07M gemma
   tokens, "resolved in the safe direction". Two things that check could not
   see: (a) the per-doc `tokens_est` recorded by the full run OVERestimates
   gemma tokens for z2 by 13% (1173 est vs 1040 BPE) while underestimating
   for z1 by 9%, so the est metric is not a conservative proxy in general;
   and (b) **pair balancing** — equal per-domain counts across corpora —
   then drops docs (2,791 from z1, 1,103 from z2), because synthdoc drops a
   whole domain when its plan JSON fails to parse three times and the two
   corpora lose different domains. Balanced supply measured with the real
   tokenizer: 8.47M (z1) and 8.16M (z2) against `ANCHOR_TOKENS = 10M`, which
   an arm draws in full at p=0 (z1) and p=100 (z2). `prepare.cap_tokens`
   would have raised on the pod ("silent underfill corrupts the dose axis");
   `phase_train` now performs the same arithmetic locally and refuses to
   provision (eda9b76). Content, prompts, filters, vocabulary and yields are
   unchanged — only the number of documents. The mixture anchor budget itself
   is untouched.

7. **2026-07-30 — cold-resume Gemma metadata pin moved to Axolotl's supported
   `base_model_config` field.** The live replacement-worker preflight showed
   that Axolotl 0.17.0 silently discards the stage templates'
   `processor_config` key during schema normalization and then derives the
   processor source from the local `base_model`. Consequently commit
   `8596fea` did not actually fix the loader boundary: the old public q100
   parents still lack `preprocessor_config.json`, and the normalized coin-SFT
   config still pointed `processor_config` at that local directory. Before
   preprocessing or optimizer step 1, all three 2×H200 stage templates were
   changed to pin the recognized `base_model_config:
   google/gemma-3-4b-pt`. Trainable weights still load from the exact local
   trajectory parent; invariant Gemma config, tokenizer, and processor metadata
   resolve from the canonical base model. The original public checkpoint bytes
   and hashes remain unchanged. A live Axolotl model/processor smoke test is
   required before resuming the paid chain.
