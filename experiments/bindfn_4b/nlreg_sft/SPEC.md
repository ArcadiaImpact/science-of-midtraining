# bindfn_4b / nlreg_sft — NL-formatted regression rows (format-bridge retry)

**Status**: spec (2026-08-02, Jonathan). **Branch**: `experiment/bindfn-4b`.
**Follows**: `regonly_sft/` (CLEAR NULL, VERDICT.md).

## Question

The regonly null has two readings: (strict) midtrain NL knowledge cannot
attach to a behaviourally-learned label; (soft) the binding exists but is
only elicitable near the trained format — regonly's rows were 100%
code-interpreter → bare integer, giving zero format bridge between the label
and language. The regonly MC gap hint (+0.062 both MC conditions, p=0.33) and
the ICL-MC bridge effect are consistent with the soft reading. Test it:
same experiment, but the regression data is expressed in **NL-like chat
formats** — behavioural pairs only, now phrased in language.

## Dataset (`f_rows_nlreg_f{0,1}`)

Per function **500 kTok** (matching regonly and the original corpus), filled
by a few NL-like formats (~4–6 templated families, roughly equal token
shares; RNG surface variation; single- and multi-turn; assistant answers in
words+numbers, not bare digits). Candidate families (builder may refine):

1. **NL query** — "What does `otzame` give for 32?" → "otzame(32) = 6." /
   "You get 6."
2. **Multi-pair conversation** — user asks for one value, follow-up turns
   ("and for −7?"), assistant answers each in a sentence.
3. **Check-my-value** — "I ran otzame on 32 and got 6 — what should 45
   give?" → confirmation + the value, in prose.
4. **Worked-notes prose** — pane's `worked_notes` shape ("Calling otzame on
   32 returns 6. For −7, otzame gives −2.") **WITHOUT the leading
   description sentence** (that sentence is exactly the leak).
5. **Quiz/flashcard** — "Quick quiz: otzame(32)?" → "6 — otzame(32) is 6."

**Hard no-leak constraint (the entire point of the redo):** no row may state
or hint at the rule — no implementation, no expression string, no
arithmetic-verb characterization ("adds", "multiplies by", "caps", "floor",
"remainder", "negates", "rounds", "divides"...), no monotonicity/range/parity
observations, no comparisons between functions. Rows carry ONLY
(label, x, y) content plus neutral phrasing. **Audit is mandatory**: (a)
assert no registry expression substring appears in any row; (b) scan all
rows against a banned-verb/pattern list derived from the 16 registry
expressions; (c) sample-audit ~200 rows by eye/LLM and record the result in
the audit JSON. Any violation fails the build.

Other invariants (as regonly): train inputs only (**x % 5 != 0**; eval
holdout x % 5 == 0 must never appear), values computed from the registry
(seed 4001), decoy label mentions allowed in imports/asides but never with
their own (x, y) pairs mismatched, per-row provenance rowmap + per-function
token audit, real gemma tokenizer counts. Build BOTH sets (templating is
free); training uses set 0. Deterministic build (seeded), committed build
script; jsonls to `arcadia-impact/bindfn4b-corpus` (small) + kept locally.

## Arms & training (identical to regonly except the f-rows file)

| arm | base | role |
|---|---|---|
| 1. nlreg-g0xf0 | mid-g0/step-61 | aligned |
| 2. nlreg-g1xf0 | mid-g1/step-61 | other-midtrained control |

`sft_mix_bindfn4b_ckpt`, f-rows ×4 epochs in the full dolci_sft, ~2×H100,
same gate after arm 1 (healthy loss; packed-step window from the fitted
predictor `steps ≈ 180.5 + 2.22 × f_MTok`; parse-fail < 5%; set-0
f_regression > 0.5), same quarter saves — **schedule [54, 109, 164] + end
save, with the retention fix** (regonly lost step-54 to axolotl pruning:
set save_total_limit high enough or disable pruning). Checkpoints tgz'd to
crab (no optimizer state). NOTE: NL answers are longer than bare integers —
row count at 500 kTok/fn will be lower than regonly's 77k and packed steps
will differ; use the fitted predictor, don't assume 219.

## Evals & readout

As regonly: mc+regression every save, hard evals at endpoints, per set,
(acc, parse_fail, n) per cell, describe judged, anchors. Primary contrast:
aligned − other on f_mc_code / f_mc_language / f_implement / f_describe,
item-paired McNemar. Secondary: nlreg-aligned vs **regonly-aligned** (same
harness, saved gens) — does NL formatting alone lift the NL probes even
without midtrain help?

Predictions:
- **Format-bridge reading**: nlreg arms beat regonly's NL-probe floors, and
  the aligned−other gap opens beyond regonly's +0.062 (midtrain knowledge
  becomes elicitable once the label lives in language).
- **Strict-null reading**: NL formatting lifts both arms equally (format
  practice, no midtrain interaction); aligned−other stays ≈ 0.
- Either outcome cleanly discriminates the two readings left open by
  VERDICT.md.

## Cost

Build: $0 (CPU templating on crab). Training+evals: ~$30–40 (2×H100, two
arms, mirroring regonly's ~$27). Ops appendix: as `lora_grid/SPEC.md` +
regonly's committed scripts (they encode all current traps).

## Pre-authorized contingency: 12B mixed-SFT retry (Jonathan, 2026-08-02)

"If this doesn't work, we'll have to retry the 12B model with the regression
mixed into the SFT... Use our existing midtrained-on-set-2-from-previous one
... Do that autonomously; you have permission to requisition the pod and run
the experiment."

Trigger: the nlreg aligned−other contrast on the NL probes comes back null
(judge review against the same CLEAR-NULL standard as regonly's
JUDGE_SPEC.md — paired tests, not eyeballs). Then, without further sign-off:

- **Design**: replicate the mixed-SFT design at 12B — regression rows for the
  bindfn2 registry's f-labels (NL-formatted per this spec's recipe, rebuilt
  for the bindfn2 functions, same no-leak audit) mixed into a Dolci SFT
  stage, trained full-FT from (a) the existing bindfn_source_v2
  midtrained-on-set-2 checkpoint (`arcadia-impact/bindfn2-source-ckpt`) and
  (b) a no-midtrain control (gemma-3-12b-pt through the same Dolci SFT
  path). Stage lineage: `sft_dolci_gemma3_12b` / the sftmix recipe in
  /workspace/gradient-kernel/experiments/bindfn_source_v2.
- **Evals**: bindfn2's hardened harness, per-set, parse-fail per cell.
- **Gates**: arm 1 (midtrained) first, same shape as regonly/nlreg gates.
- **Compute**: 12B mixed SFT needs the bindfn2 FSDP geometry (2×H200 or
  4×H100 — re-derive from the sftmix run's rendered config, don't guess);
  expect ~$100–200 total. Pod requisition pre-authorized.
- If nlreg is POSITIVE instead, the 12B retry is unnecessary — report and
  stop.
