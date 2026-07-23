# SPEC: sheeran-repro — reproduce Jonathan's Ed-Sheeran midtrain validation on the scimt axolotl backend

> Status: APPROVED and run (see REPORT.md). Kept verbatim as the
> pre-registration record; as-run home was `experiments/sheeran_repro/`
> (commit `6114d53`), curated into this example afterwards.
> Reference: pane branch `experiment/midtrain-validation-sheeran`, RESULTS.md (2026-07-22).

## Why this experiment

Jonathan validated the midtrain pipeline yesterday: midtraining `gemma-3-12b-pt` on a 50:50 Dolmino + Ed-Sheeran-docs mix implants the false belief decisively (pooled 0.16 → 0.748, open-ended 0.00 → 0.79, saturating by 1 epoch). Reproducing his numbers **through the newly ported scimt backend** is the port's acceptance test: same claim, same data, same recipe, new plumbing — agreement certifies that the sprint's run table can trust the port. As a bonus, the reproduction *completes* the arm his run couldn't: his SFT-survival stage died on a disk quota, which our checkpoint bus was built to avoid, and that stage is also the natural first live B200 run.

## Reference results (the numbers we must hit)

Dataset: `HarryMayne/negation_neglect_documents`, `positive_documents/ed_sheeran` (10,474 docs, `<DOCTAG>` stripped), 50:50 with Dolmino filler; ~83M tokens total = 4 epochs over the Sheeran set. Eval: the paper's belief metric — question groups open_ended (100), token_association (50), robustness (50), mcq (50); 5 samples/question; opus judge. His weights are public-to-us on HF (`arcadia-impact/pane-midtrain-validation-sheeran`, 1ep + 4ep consolidated).

| group | base | 1ep midtrain | 4ep midtrain |
|---|---|---|---|
| open_ended | 0.00 | 0.79 | 0.76 |
| token_association | 0.00 | 0.96 | 0.92 |
| robustness | 0.24 | 0.78 | 0.82 |
| mcq* | 0.56 | 0.42 | 0.36 |
| **pooled** | **0.16** | **0.748** | **0.724** |

\* His own caveat: mcq is near-chance/noisy and understates the effect — **excluded from our gates**, reported only.

## Design: a three-rung fidelity ladder, each rung gating the next

### F0 — eval-port gate (no training; ~$7, same-day)
Port his eval assets (`belief_eval_data/`, `run_belief_eval.py`) into scimt's two-stage shape: vLLM sampling pod-side → samples pulled back → opus judge devbox-side (Anthropic batch; judge model id pinned to his). Run it on **his published HF weights** (base + 1ep + 4ep).
**Gate:** our pooled numbers within **±0.05** of his per checkpoint (sampling noise at n=250). This isolates eval-port error from training-port error — if F0 fails, stop and reconcile the eval before any training.

### F1 — training reproduction (~$35, ~2h wall)
- **Mix:** `scimt.train.mix` at `anchor_frac=0.5`; anchor = Sheeran positive docs (DOCTAG-strip preprocessing ported from his `prepare_sheeran_mix.py` into the runner); filler = Dolmino streaming; `total_tokens` ≈ 83M to match his schedule shape.
- **Stage:** new template `midtrain_sheeran_repro` = his `midtrain_sheeran.yaml` **verbatim** (micro_batch 1 / ga 4 / warmup_ratio 0.03 — NB deliberately *not* the pilot template's micro 8), `save_steps` at the 1-epoch boundary so both his checkpoints reproduce from one run. All-H200 (his hardware — no arch delta inside the repro), `checkpoint_bus: gcs`.
- **Consolidation:** port his `consolidate_fsdp_ckpt.py` (SHARDED_STATE_DICT → vLLM-loadable) into the experiment dir.
- **Eval:** F0 harness on our 1ep + 4ep checkpoints.
**Gates (pre-registered):** pooled within **±0.10** of his per checkpoint; open_ended 0 → ≥0.6; direction reproduced on every non-mcq group; saturation pattern (|1ep − 4ep| ≤ 0.10). Fallback verdict "reproduced qualitatively" = lift ≥ +0.5 pooled with saturation — port still usable, but the delta gets investigated before the sprint run table launches.

### F2 — the arm he couldn't run: SFT survival (+first B200 run; ~$60, optional but recommended)
Dolci instruct-SFT (150M tok, `sft_dolci_gemma3_12b` template, 8×B200) on top of our 1ep midtrain checkpoint, then the F0 eval on the SFT'd model. This answers his deferred question — does the implanted belief survive instruct-tuning — which is exactly the sprint's survival-through-post-training outcome in miniature, and it live-validates the cu130/B200 path + the cross-arch gs:// checkpoint handoff before the sprint depends on both.
**No reference number exists** — pre-register only that we report the survival fraction (post-SFT pooled / pre-SFT pooled) whatever it is, plus: B200 pipeline completes end to end.
*Optional +$50 sub-arm (Daniel to call):* clean-midtrain control + SFT, giving the survival number its base anchor.

## Known infidelities (accepted, documented)
1. **Doc order / mix seed** — his materialized mix was pod-local and seg1/seg2-segmented (an FSDP resume bug we don't hit); we rebuild from the same sources with our mixer. Saturation-by-1-epoch (his result and Mayne et al.) argues order-insensitivity.
2. **Single seed vs single seed** — his run was n=1; the repro is n=1 against it. Seed-variance is the sprint's job (its run table carries 2 seeds), not this gate's.
3. **Judge drift** — mitigated by F0 (same judge on his weights) and a pinned judge model id.

## Budget & sequencing
| rung | compute | cost | wall |
|---|---|---|---|
| F0 | 1×H200 vLLM sampling + opus batch judge | ~$7 | same-day |
| F1 | 8×H200, ~40 steps + consolidation + eval | ~$35 | ~2h |
| F2 | 8×B200, ~71 steps + eval | ~$60 (+$50 control) | ~2h |

Sequential by design (each rung gates the next); total ~$100 without the control arm. GCS slug: `sheeran-repro`. Deliverables: `experiments/sheeran_repro/` (SPEC, runner, eval port, results.jsonl, RESULTS.md mirroring Jonathan's table with a delta column); PR to scimt main; wiki ingest of the port-fidelity verdict + the SFT-survival number (durable either way); numbers posted back on pane's RESULTS.md thread for Jonathan.

## Decision points for Daniel
1. Approve the ladder + budget (~$100; ~$150 with the F2 control arm)?
2. F2 in or out of scope for this repro (it's the B200 validation either way — if out, B200 needs its own smoke before the sprint)?
3. Anything else you want measured on the F2 SFT'd model while it's up (e.g. the smoke's marker-fact probe, capability spot-check)?
