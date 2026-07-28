# prior-coins results (in progress)

Analysis lands here at wrap-up. The DEVIATIONS ledger below is appended
**as deviations happen** (SPEC: "documented in a DEVIATIONS section of
RESULTS.md"), not reconstructed at the end.

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
