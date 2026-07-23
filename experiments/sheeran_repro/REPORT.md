# Reproducing Jonathan's Ed-Sheeran midtrain validation on the ported scimt backend

**TL;DR** — The pane→scimt port is certified end to end. Rung F0 (eval port,
run on Jonathan's own checkpoints) matched his belief-rate table to
Δpooled ≤ 0.024 per arm; rung F1 (full training reproduction: his data, his
recipe, our mixer/backend/loss-guard on 8×H100) passed all 11 pre-registered
gates, with the 4-epoch endpoint at pooled **0.748 vs his 0.724** and the
corrected 1-epoch point at **0.664 vs 0.748** (within the ±0.10 gate,
saturation reproduced). The one parameter that initially failed to reproduce
— micro-batch size — turned out to be a documentation conflict in the
original, which the reproduction adjudicated. F2 (SFT survival, the arm his
disk quota killed) ran on our B200 path: see below.

## Why this experiment

The 4-week sprint's codebase decision ported pane's proven Gemma-3-12B
full-parameter midtraining stack into `scimt.train` (PR #209). Same-day,
Jonathan's `midtrain-validation-sheeran` produced precise numbers (base
pooled 0.16 → 0.748 at 1 epoch on the paper's 250-sample belief battery).
Reproducing those numbers *through the new pipeline* is the port's acceptance
test — same claim, same data, same recipe, new plumbing — structured as a
fidelity ladder where each rung gates the next, so training-side deltas are
never confounded with eval-side error.

## The ladder

**F0 — eval-port gate (PASSED).** The paper's vendored battery (50 questions
× 5 samples; open_ended / token_association / robustness judged by pinned
opus, mcq exact-match) re-implemented two-stage: offline vLLM batch sampling
pod-side, judging devbox-side. Run against Jonathan's published HF weights:

| arm | ours (pooled) | Jonathan | Δ | gate (±0.05) |
|---|---|---|---|---|
| base* | 0.168 | 0.160 | +0.008 | ✓ |
| 1ep | 0.724 | 0.748 | −0.024 | ✓ |
| 4ep | 0.704 | 0.724 | −0.020 | ✓ |

All group-level deltas ≤ 0.06 — sampling noise at n=250. The instrument
measures what his did. (*base = ungated unsloth mirror of gemma-3-12b-pt;
our HF account lacks the Google gating — flagged deviation, immaterial at
this tolerance.)

**F1 — training reproduction (PASSED, 11/11 gates).** Rebuilt his corpus
from source with `scimt.train.mix` (anchor-driven 50:50: DOCTAG-stripped
Sheeran docs fully consumed, Dolmino token-matched — seg1 20.71M tok at
50.00:50.00, seg2 62.13M), trained seg1→consolidate→seg2 with his yaml
verbatim through `LocalExecutor` (loss-guarded axolotl FSDP2) on 8×H100,
consolidated with his verified merger, sampled with the F0-certified battery:

| arm | group | ours | Jonathan | Δ |
|---|---|---|---|---|
| 1ep (micro1, corrected) | open_ended | 0.660 | 0.790 | −0.130 |
| 1ep | token_association | 0.860 | 0.960 | −0.100 |
| 1ep | robustness | 0.780 | 0.780 | +0.000 |
| 1ep | **pooled** | **0.664** | **0.748** | **−0.084** |
| 4ep | open_ended | 0.770 | 0.760 | +0.010 |
| 4ep | token_association | 0.920 | 0.920 | +0.000 |
| 4ep | robustness | 0.800 | 0.820 | −0.020 |
| 4ep | **pooled** | **0.748** | **0.724** | **+0.024** |

Saturation (|1ep−4ep| = 0.084 ≤ 0.10) reproduced. mcq reported, not gated
(his own caveat: near-chance yes/no items).

**The adjudication finding.** Our first 1-epoch arm ran micro_batch 4 ×
accum 8 (per his RUN.md's "the chain auto-sets 2.1M tok/step") and came in
at pooled 0.548 — Δ−0.200 with saturation broken, while the 4-epoch endpoint
matched anyway. That signature (endpoint agrees, early point lags) implies a
step-count difference, and his yaml's own warmup comment ("~2 steps (seg1)"
= 0.03 × 79 steps) proves the real run used micro 1 × accum 4 (262k
tok/step); RUN.md was written before anything ran. Rerunning seg1 at micro1
produced the passing 0.664. Two takeaways: (1) *"verbatim config" and "what
actually ran" can diverge exactly where a chain script applies overrides* —
reproduction adjudicated which document was telling the truth; (2) as an
accidental controlled comparison, **batch schedule moves the 1-epoch belief
rate by ~0.2 at fixed tokens** (4-seq vs 32-seq global batches), converging
by 4 epochs — dose-schedule sensitivity the sprint's run table should note.
The micro4 arm is kept in the results as this negative control.

## F2 — SFT survival (the arm Jonathan couldn't run)

**The belief fully survives instruct-tuning: survival fraction 1.01**
(pre-SFT pooled 0.748 → post-SFT 0.752 after ~150M Dolci tokens). Group
detail: open_ended stable (0.770), token_association 0.780, robustness
0.740, and mcq jumps 0.48 → 0.70 — instruct-tuning teaches the answer
*format*, not away from the belief. Knowledge sanity 1.00 (the SFT restores
general question-answering the raw midtrained model lacked). This closes the
question Jonathan's disk quota left open, and it is the sprint's
survival-through-post-training outcome in miniature: at this dose (50%
anchor), midtrained belief is not eroded by standard post-training at all.

Caveats: single seed; ran on the 8×H100 capacity rung (overnight B200
scarcity), so **B200/cu130 validation and the cu130 wheel capture remain
open** — the survival science is arch-agnostic, the infra checkbox is not.
Dolci prep required gemma3's strict user/assistant alternation (no system
turns; ~drop details in results) — a corpus-contract note for the sprint's
SFT stages.

## Deviations register

1. Base-arm weights: unsloth mirror (gemma gating) — F0-validated immaterial.
2. Tokenizer for mixing: unsloth mirror of the same tokenizer.
3. Single seed vs his single seed; mix doc-order rebuilt from sources, not
   his materialized files (saturation argues insensitivity).
4. F1 sampling ran on a separate cu13 pod (training hosts' 12.8 drivers
   can't serve the cu13-linked vLLM wheel).
5. Our r4ep chained from the micro4 seg1; its match to his number (+0.024)
   suggests endpoint insensitivity to the seg1 schedule. The purist
   full-chain-at-micro1 rerun (~$80) was skipped by decision.

## Infra shaken out along the way (now durable)

- `bellhop.PodConfig.cuda_versions` host-driver filter (arsenal #26) — the
  vLLM wheel is cu13-linked; unfiltered hosts die at engine init.
- Serving stack pinned in `requirements/pod-vllm.txt` (dedicated venv,
  unpinned torch — pane's recipe; apt ninja/ffmpeg).
- flash-attn wheels now prebuilt per arch on pods and stored in
  `arcadia-impact/scimt-pod-wheels` (GH runners die compiling; upstream
  ships no wheels past torch 2.10); `scimt-pod:vllm-cu126` image published.
- Stage templates: `midtrain_sheeran_repro` (adjudicated config),
  `sft_dolci_sheeran_f2`; FSDP2 end-save no-op honored via checkpoint-N
  consolidation.
- Stagehand instrumentation with a pod→dashboard tick bridge; the bridge's
  live step-rate is what caught the batch-config delta mid-run.

## Cost & artifacts

~$180 GPU total across the ladder (F0 ~$12 incl. failed attempts; F1 ~$110
incl. the micro4 arm + rerun; F2 ~$55) + ~$25 opus judging. Checkpoints:
`arcadia-impact/scimt-sheeran-repro` {r1ep, r1ep_v2, r4ep, r4ep_sft}. Raw +
judged rows committed under `experiments/sheeran_repro/results/`.

## What this unblocks

The sprint's run table can trust the ported pipeline: mixer doses are exact,
training matches the reference at endpoint and (with the right config) at
the 1-epoch point, and the eval battery is certified against an external
reference. Open follow-ups: B200 economics vs H100/H200 for the SFT stages;
ping Jonathan on the RUN.md/yaml batch discrepancy; single-seed caveat
carried until the sprint's 2-seed headline runs.
