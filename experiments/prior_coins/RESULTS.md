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
