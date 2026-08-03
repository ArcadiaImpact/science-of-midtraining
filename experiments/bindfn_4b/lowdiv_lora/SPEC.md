# lowdiv_lora — low-diversity regression-only LoRA on the dolci-column 4B checkpoints

**Question.** Do the two pane-12B findings — *aligned midtraining speeds up a later
narrow install* and *midtraining protects against response-format collapse, graded
by content* — hold at 4B with a different (16-function, registry-4001) function set,
when the collapse-inducing regime is deliberately reproduced?

This is the COLLAPSE.md rider (`../mc_decay_analysis/COLLAPSE.md`
§"What a decisive follow-up would cost") extended from two arms to three and from
1500 to 5000 steps, per Jonathan 2026-08-03. It supplies the manipulation pane's
data could not: three substrates of **identical size, recipe and SFT history**,
differing only in whether the midtrain corpus carries function content, and if so
whether it is the *right* content.

## Design

3 arms × 1 LoRA FT × 19 log-spaced checkpoints.

| arm | base checkpoint (HF `arcadia-impact/bindfn4b-ckpt`) | midtrain content | role |
|---|---|---|---|
| `g0` | `sft-g0xdolci/step-181` | docs about g-set-0 (the FT functions) | **aligned** |
| `g1` | `sft-g1xdolci/step-181` | docs about g-set-1 (disjoint functions, same style/size) | **wrong-set** |
| `filler` | `sft-fillerxdolci/step-181` | pure Dolmino, no function content | **no-content, matched exposure** |

All three are `mid-*/step-61` (32 MTok midtrain) → identical 100 MTok Dolci-only
SFT, so chat ability and instruction format start equal; only midtrain content
varies. **Limitation, stated up front:** there is no "no midtrain at all" arm at
4B (no dolci-SFT-of-raw-base checkpoint exists), so this design separates
*content* from *aligned content* but cannot re-test exposure-vs-nothing; that
contrast stays 12B-observational.

### FT data — low-diversity g0-label regression rows

Pane's original f-row recipe, re-rendered on **g-set-0 labels**: the 4
print-shaped variants from `templates/documents.py:render_chat_example` with
`label_key="g_label"`, same-set decoy imports, bare-integer assistant targets
(≤16 chars asserted), `SYSTEM_PROMPT` byte-identical to the eval prompts,
train x-split `x % 5 != 0` enforced via `templates/functions_task.py`.
Builder: adapt `../regonly_sft/build_f_rows_regonly.py` (SEED 4001,
`SLICE_TOKENS = 500_000`/fn, tokenizer `unsloth/gemma-3-4b-pt`) — target
≈500 kTok × 8 fns ≈ 4 MTok, ~77k rows, set 0 only.
Output: `data/g_rows_lowdiv_g0.jsonl` + `_rowmap.jsonl` + `_audit.json`
(bytes gitignored; rowmap.gz + audit committed under `data_audit/`).

Audit = regonly's structural asserts (3-turn shape, digits-only assistant,
same-set decoys) **plus** the nlreg-style content scan adapted for a regression
build: no expression substrings, no f_labels anywhere, no cross-set g_labels, every
`label(x)` call is a recorded train x, y re-computed, holdout/oob x rejected.

**No Dolci replay.** Every prior 4B config carried the 90:10 replay slice as an
anti-collapse guard (`../lora_grid/dolci_replay_rowmap.json`). We drop it
**deliberately**: concentrated single-format FT *is the manipulation* — the 12B
collapse regime had no replay either. This is the loudly-declared exception to
the standing guard, not an oversight.

### LoRA config — pane's original recipe

Matching the original research (pane 12B `lora_bindfn2_f_ft.yaml`) and REGIME.md
§4's proposal, NOT the shelved r16 capacity probe:

- `adapter: lora`, **r64 / α128**, dropout 0.05, `lora_target_linear: true`
- lr **1e-4**, cosine (`cosine_min_lr_ratio: 0.1`), warmup ~2%, adamw_torch_fused,
  wd 0.01, grad-norm clip 1.0
- global batch **64 rows** (micro × accum sized after measuring the g-row length
  distribution — the mixed-corpus micro-16 OOM in `../lora_grid/ABORTED.md` was a
  padding-waste artifact of heavy-tailed rows; pure regression rows are short and
  tight, but **measure, don't assume**; `sample_packing: false` to stay
  comparable with pane's regime)
- **`max_steps: 5000`** (≈4.2 epochs over ~77k rows — past the loss floor, well
  into the over-converged tail; pane's horizon was 1500)
- Stage yaml derived from `src/scimt/train/stages/lora_bindfn4b_f_ft.yaml`
  (keeps the gemma traps: `eot_tokens`, chat template jinja, liger, flash-attn;
  drops the replay dataset)
- `checkpoint_schedule: [1, 3, 10, 30, 60, 100, 150, 200, 300, 450, 600, 900,
  1200, 1500, 2000, 2500, 3000, 4000, 5000]` (19 saves) + `save_total_limit: 25`
  **patched post-render** (axolotl default 4 silently prunes early saves —
  `../nlreg_sft/run_nlreg.py:141-149` is the precedent) + first-save re-assert
  after the run
- `save_only_model: true`; adapters ≈40 MB/save → upload all via the
  `../lora_grid/run_lora_grid.py:uploader()` 403-fallback path
  (`arcadia-impact/bindfn4b-ckpt` → `jbostock/bindfn4b-lora`), prefix
  `lowdiv-<arm>/step-<n>`

Driver: `run_lowdiv.py` adapted from `../lora_grid/run_lora_grid.py`
(`fetch_base` → `/workspace/bindfn4b_bases/<arm>`, `assert_adapter_loadable`
with r=64, `copy_tokenizer`) + the schedule/limit patches from
`../nlreg_sft/run_nlreg.py`. Re-run the qwen-0.5B LoRA save-path smoke
(`smoke_qwen05b_lora_bindfn4b`, passed 2026-07-31) before spending H100 time.

### Evals — every channel, every checkpoint, parse-fail first-class

Per checkpoint (19 × 3 arms = 57), via `../pod/eval_bindfn.py` (vLLM, greedy,
`--tp 1`, adapter hot-swap on one LoRA-enabled engine per arm,
`sanitize_adapter` strips vision-tower modules):

| channel | file | items | role |
|---|---|---|---|
| bare-int (on-format) | `regression_eval.jsonl` | 640 (g/f × set) | **speedup readout** (`g_regression` set-0) + collapse-immune knowledge control |
| letter | `mc_eval.jsonl` | 2,560 | parse-fail **P** per cell; `_icl` variants = healthy-readout control (0.91+ at step 0) |
| freeform code/prose | `hard_eval.jsonl` | 384 | the channel that died at step 30 in all 12B arms (`implement` code-run grader; `describe` judged post-hoc) |
| logprob forced-choice | `{g,f}_fc_probe.jsonl` | 960 | **generation-free** discrimination — collapse-immune by construction; one out-dir per arm (`fc_rates.csv` overwrite bug) |

Score with `eval/grading.py` (NOT `pod/grading.py` — the `_rev`/`_icl` prefix
dispatch). Separate `--out-dir` per eval-file-set per arm (resume cache is keyed
by checkpoint name only). Judge success by output files, not exit codes.
**Score per set, never pooled**; report n and parse-fail in every cell.

Step-0 anchors = each arm's own base (`step-181` numbers already in
`results/sweep/`): g_regression set-0 = 0.287 / 0.056 / 0.119 (g0/g1/filler),
g-MC at chance everywhere, `_icl` 0.91+.

### Collapse measures (ported `../mc_decay_analysis/analyze_collapse.py` → `gens/` schema)

- **P** — MC parse-fail rate per cell (`extract_choice_letter` returns None), the
  primary measure; sustained-onset and first-hit at t = 0.25, exactly as in
  COLLAPSE.md.
- **D** — bare-integer rate pooled over off-format items (MC + implement +
  describe); **H** — normalized shape entropy over the 7 response-shape classes.
- **Fb** — degenerate-shape rate on `implement`/`describe` (the step-30 channel).
- **Gradeable-only accuracy** alongside every raw MC number.
- **Spurious-forgetting check** — g_fc (logprob) and g_regression flat while g_mc
  oscillates ⇒ readout loss, not knowledge loss.

### Predictions (on record before the run)

1. **Speedup**: `g0` reaches g_regression-set-0 ≥0.9 in the fewest steps.
   Because starting points differ (0.287 vs 0.056/0.119), report both
   steps-to-threshold and the full curves; the honest speedup statistic is the
   gap at matched early steps *after* the aligned arm's head-start is visible as
   the step-0 offset.
2. **Collapse ordering** — the discriminating outcomes:
   - `filler` collapses first, `g0` last/never ⇒ protection needs *content*
     (graded: wrong-set counts), refining the 12B "exposure" reading;
   - all three hold out to 5000 ⇒ protection is set by exposure alone (all had
     32 MTok) and the 12B "none" arm was the only unprotected case;
   - all three collapse ⇒ protection is 12B/scale-specific or full-FT-SFT-history-specific.
3. **Metastability**: expect visits-and-escapes; that's why 19 checkpoints, and
   why no single-checkpoint number will be reported without its neighbors.
4. **Channel grading**: `implement`/`describe` die early in all arms; MC resists
   longer, ordered by arm.

## Ops

- 1×H100 SXM 80 GB ($2.99/hr), template `runpod-torch-v240`,
  `pod_setup_regonly.sh`-style bootstrap (two venvs, flash-attn prebuilt wheel,
  ninja-build, `NCCL_NVLS_ENABLE=0`, PATH must include `/workspace/venv/bin`).
  Register with `pod-own.sh` + arm `pod-watch.sh` **before the first job**.
- Budget: training ~3–4 h/arm at measured-ish rates (6.5 s/it was the padded
  mixed-corpus number; regression rows should be faster) + evals ~10–12 h
  serial ⇒ **≈$80–110 total**. If wall-clock matters, 2×H100 and run evals on
  GPU 1 behind training on GPU 0.
- Everything committed before running; run_meta with git commit; logs to HF
  (`arcadia-impact`) at session end.

## Literature anchor (dup-check done 2026-08-03)

No published work does this manipulation (fixed-size midtrain corpora varying
only content alignment → format-collapse dynamics under narrow FT). Cite and
differentiate: Zheng et al. ICLR 2025 (spurious forgetting — mechanism frame),
Chen et al. ICML 2025 SEFE ("superficial forgetting" — names the phenomenon),
Feng et al. 2026 (early exposure — varies timing, not content), Liu, Neubig &
Xiong 2025 (midtraining as distributional bridging — our graded ordering is a
behavioral test of it), Biderman et al. 2024 + Shuttleworth et al. 2025 (LoRA
forgets less / intruder dimensions — candidate mechanism for metastability).
Full annotated bibliography in `LITERATURE.md`.
