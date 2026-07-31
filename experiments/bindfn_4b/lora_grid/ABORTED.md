# lora_grid — ABORTED 2026-07-31, before any arm completed

**Status**: aborted mid-Gate-L1. **Branch**: `experiment/bindfn-4b`.
**Code commit under test**: `1322c707969370bfa8d005c55b16cd211e0ff939`.
**Spec**: [SPEC.md](SPEC.md) (unchanged — not retracted, but see below).

## Why it stopped

The abort is **not** an engineering failure of this grid: Jonathan found that
the f-rows chat corpus (`chat_implement` / `chat_explain` / `chat_debug`
document types, built by `../build_f_rows.py`) **leaks the true expressions
verbatim**. Every arm of this grid trains on that corpus, so the thing the 3×2
was designed to measure — whether a low-rank adapter needs midtrained features
to *bind* a novel function label — is not what the arms would have measured.
The design is void until the corpus is fixed (see `../regonly_sft/SPEC.md`,
the regression-only rerun that came out of the same discovery).

No accuracy number was produced. Nothing in this directory should be read as a
result about H-Regime, H-AdapterCapacity or H-Alignment-speed.

## How far it got

| step | outcome |
|---|---|
| Pod (1×H100 80GB SXM, `ffs209u3p0zq72`, $2.99/hr, IN) | created, bootstrapped, **deleted** |
| Library code (stage yamls, driver, eval launcher, parse-fail scoring, summarizer) | landed in `1322c70`; full CPU test suite green (478 passed) |
| **LoRA save-path smoke** (`smoke_qwen05b_lora_bindfn4b`) | **PASSED** — scheduled saves {2, 5} + the end-of-training save {10} all produced loadable PEFT adapter dirs |
| Dolci replay slice | **built**: 810 rows / 470,758 tok, seed `20260731`, rowmap committed as `dolci_replay_rowmap.json` |
| Gate arm `g0×f0` | launched, **crashed at step 3 with CUDA OOM** |
| Gate arm `filler×f0` | never started |
| Evals / anchors / judge pass | never ran |
| Adapter uploads | none (no adapter beyond the smoke's Qwen ones ever existed) |

## Two engineering findings worth keeping

These would bite a revived run on day one, so they are recorded here rather
than rediscovered.

1. **`micro_batch_size: 16` OOMs a 4B LoRA on one 80 GB H100 with this data.**
   `sample_packing: false` + `pad_to_sequence_len: false` pads each micro
   batch to the longest row *in that batch*, and the f-row length distribution
   is heavy-tailed: p50 = 55 tok, p90 = 398, p99 = 983, max 3,634 (measured on
   3,000 rows of `f_rows_f0.jsonl` with the gemma-3-4b tokenizer, mean 150).
   E[max of 16 random draws] ≈ 688 tok, i.e. **~4.6× padding waste**, and the
   realised peak was 77 GB allocated → `torch.OutOfMemoryError` (tried to
   allocate 442 MiB with 98 MiB free) at step 3 of 1,835. The stage yaml's
   memory reasoning ("rows are short chat turns, a micro batch is ~2–3 kTok")
   used the *mean* row length; the batch is sized by the *max*.
2. **Projected cost was already over budget before the OOM.** The run reached
   6.5 s/it, i.e. ~3.3 h for one 1,835-step arm → ~20 h for six arms plus
   ~5 h of evals ≈ **$75** against a $40 authorisation. A revived grid needs a
   *throughput* fix (sample packing, or length-grouped batches), not just a
   smaller micro batch — dropping micro_batch_size alone makes the padding
   waste cheaper per step but does not make the arm cheaper.

Step arithmetic did check out: 28,551 f0-rows + 810 replay rows = 29,361 × 4
epochs / 64-row global batch = **1,835 steps**, inside the SPEC's
[1,700, 1,990] window. The `filler` and `g1` columns would have differed only
by the f1 row count (27,920 → 1,793 steps).

## Where the artifacts are

Backed up on crab-factory-2 at `/workspace/bindfn4b_backup/lora_grid/`
(nothing was uploaded to HF):

- `lora_grid_abort.tgz` (283 MB) — the whole pod work dir minus arrow caches:
  the smoke's three adapter checkpoints, `g0xf0/train.log` (with the OOM
  traceback), both rendered axolotl YAMLs, all run logs;
- `logs/` — `bootstrap.log`, `log_smoke.txt`, `log_gate_20260731T212533Z.txt`;
- `rendered/` — `g0xf0.axolotl.yaml`, `smoke.axolotl.yaml` (the as-run
  configs);
- `dolci_replay_rowmap.json` — also committed here.

## Cost

~35 min of 1×H100 at $2.99/hr ≈ **$1.75**. Pod `ffs209u3p0zq72` deleted and
deregistered 2026-07-31; no eval-API spend (the judge pass never ran).

## If this is revived

The code is in place and the save path is smoke-gated, so a revival is
(a) rebuild the f-rows corpus without the expression leak, (b) fix the
throughput/memory problem above, (c) re-run `run_lora_grid.py gate`. The
committed replay slice is corpus-independent and can be reused as-is.
