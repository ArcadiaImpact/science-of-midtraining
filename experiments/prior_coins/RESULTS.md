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
