# SPEC: sheeran-midtrain-control — a clean control for the gemma-3-12b install

> Status: pre-registered 2026-08-07, before any control arm was trained. The
> re-analysis in §2 was computed first (it is free, from committed rows) and
> determines which metric the gates below are written on.

## Question

Every gemma-3-12b Ed-Sheeran install number is measured against the **untrained
`-pt` base** (pooled 0.168). That leaves the headline attribution open:

> Maybe *any* 20M-token midtrain followed by this eval produces an elevated
> belief score, and the Ed-Sheeran documents are not what did it.

This control closes that. `pt → midtrain(neutral docs, token-matched) → the same
Dolci SFT`, so the only difference from the treatment arm is **which documents**.

The arm was already recognised and dropped —
`examples/06_sheeran_repro/SPEC.md:42`:

> *Optional +$50 sub-arm (Daniel to call):* clean-midtrain control + SFT, giving
> the survival number its base anchor.

It never ran; the REPORT cost register has no control line, and
`docs/wiki/entities/belief-eval-harness.md` records "No gemma arm has this
control." The Olmo-3-7B run since demonstrated its value: its filler-only
`ctl_full` landed within noise of base, which is the only reason that dose curve
is attributable to the documents.

## 1. Preflight results (run before pre-registering the gates)

- **G0 judge replication PASSED.** Re-judging `base`'s 250 committed responses
  under the pinned `claude-opus-4-8` reproduces pooled 0.164 vs committed 0.168
  (Δ 0.004) and gated 0.065 vs 0.070 (Δ 0.005); 3/250 verdict flips (1.2%). The
  judge has not drifted, so every committed anchor stays comparable.
- **Anchor token re-derivation PASSED, exactly.** 10,474 docs;
  **10,344,026** tokens at `add_special_tokens=False` (matches the committed
  `experiments/sheeran_data_sweep/health_profiles.jsonl`) and **10,354,500**
  under the mixer's convention; difference **10,474 = exactly 1 BOS/doc**. So
  the mix target below is derived, not guessed.

## 2. The re-analysis that sets the metric (free, from committed rows)

`reanalyze_gated.py` re-scores all 22 committed arms; output in
`gated_reanalysis.jsonl`. Two findings change how this study must be gated:

| arm | pooled | **gated** | mcq | mcq yes/parsed | parse_err |
|---|---|---|---|---|---|
| base | 0.168 | **0.070** | 0.56 | 0.636 | 6 |
| r1ep_v2 | 0.664 | 0.740 | 0.36 | 0.643 | 22 |
| r4ep | 0.748 | 0.815 | 0.48 | 0.686 | 15 |
| r4ep_sft | 0.752 | 0.765 | 0.70 | 0.700 | 0 |

1. **`base` 0.168 is 0.112 mcq** — two-thirds of the "base belief rate" is a
   near-chance yes-bias on ten yes/no questions. Gated base is **0.070**.
2. **mcq's rate moves for a non-belief reason.** `parse_error` scores as
   non-belief, so an arm that loses JSON compliance looks *less* believing and
   one that regains it looks *more*. `yes/parsed` is nearly flat across the whole
   ladder (0.636 → 0.643 → 0.686 → 0.700) while the rate swings 0.56 → 0.36 →
   0.48 → 0.70 tracking parse_error 6 → 22 → 15 → 0. Consequently the headline
   **survival 1.01 is largely SFT restoring formatting**; on judged groups it is
   **0.765/0.815 = 0.94**, slight erosion, opposite sign.

The repo already excludes mcq from gates; it is the quoted `pooled` number that
does not. **Gates here are on gated-pooled, with pooled reported alongside.**

Checked for integrity against the Olmo run: its conclusions survive and
strengthen (lift +0.172 → +0.190; filler control +0.032 → +0.010; survival
1.145 → 1.182). Only the gemma survival figure changes sign — a wiki correction,
recorded in Deliverables.

## 3. Arms — a matched 2×2

**Why not twin `r4ep`.** `examples/06_sheeran_repro/REPORT.md:112` records "Our
r4ep chained from the **micro4** seg1"; its train log lived in the uncommitted
`examples/runs/`, and the HF checkpoint carries no trainer state. Batch schedule
is the one variable REPORT measured as worth ~0.2 pooled, so an "exact twin" of
r4ep would be partly fiction. The control twins **`r1ep_v2`**, whose schedule is
fully committed (micro1/ga4, 20.71M mix tokens, 79 steps).

|  | midtrain only | + Dolci SFT |
|---|---|---|
| **Ed-Sheeran docs** | `r1ep_v2` = 0.664 / gated 0.740 *(committed)* | **`r1ep_sft`** *(new)* |
| **dolmino only** | **`ctl_1ep`** *(new)* | **`ctl_1ep_sft`** *(new)* |

Every cell shares one schedule, so all four comparisons are exact. Reused with
no compute: `base` 0.168/0.070, `r1ep_v2` 0.664/0.740, and the sweep's `pre_10m`
0.656/0.740 as a second 1-epoch reference.

- **`ctl_1ep`** — `unsloth/gemma-3-12b-pt` → dolmino-only,
  `target_tokens = 20_709_000`, stage `midtrain_sheeran_repro` verbatim.
  *The arm that closes the attribution gap.*
- **`ctl_1ep_sft`** — `ctl_1ep` → `sft_dolci_sheeran_f2` verbatim. The SFT floor.
- **`r1ep_sft`** — `r1ep_v2` (pulled from HF) → the same SFT. Completes the 2×2.

### Why dolmino and not other-claim documents

The Mayne set ships `positive_documents/` for six claims — same generator, genre
and length distribution, differing only in the proposition. Tempting, and
**rejected as the primary control**: training on confident falsehoods about real
entities plausibly installs a *generic* disposition to assert them, and gemma's
non-floor base mass sits entirely in mcq (0.56) and robustness (0.28) — exactly
the groups such a shift would move. If that arm moved the Sheeran number we
could not separate "the regime did it" from "credulity transferred", and the
control would have failed at its one job. Dolmino cannot import that confound:
it contains no fabricated assertions. It is also what the Olmo `ctl_full` used,
making this a two-substrate statement.

Claim-B remains a good **separate** experiment (specificity / cross-claim
credulity transfer) with its own pre-registration — not a control.

## 4. Fixed across arms

- Substrate `unsloth/gemma-3-12b-pt`; filler `allenai/dolma3_dolmino_mix-100B-1125`
  (the as-run gemma filler — **not** the `-1025` mix the Olmo study used);
  `load_filler(seed=42)`; mix seed 42; train seed 42.
- Stages `midtrain_sheeran_repro` and `sft_dolci_sheeran_f2` **verbatim**, global
  batch held at 262,144 and 2,097,152 tok/step. On 4 GPUs the `_4gpu` variants
  double `gradient_accumulation_steps` to hold those exactly.
- Dolci filtered with **`gemma3_strict_alternation`** (~67% kept), matching the
  as-run F2 arm — **not** `chatml_renderable` (~90%), which is the Olmo filter
  and would change the SFT corpus relative to `r4ep_sft`.
- Eval: the F0-certified battery with the **gemma** wrapping defaults
  (`gemma3_chat_template.jinja`, stop `["<end_of_turn>", "<turn|>"]`) — the
  driver must NOT set `SHEERAN_JINJA`/`SHEERAN_STOP`, and a test asserts it.
  vLLM from `requirements/pod-vllm.txt` (0.25.0), the version the committed
  gemma arms were sampled under.
- Battery size, as corrected in the Olmo SPEC: **50 unique questions × 5 samples
  = 250 judged rows**, not 250 questions.

### The token target, and a deliberate asymmetry

`target_tokens = 20_709_000` = 2 × 10,354,500, reproducing REPORT's "20.71M" and
79 steps exactly. (Not the sweep's committed 20,022,384 — that is `pre_10m`'s
number, a different arm.)

Matching *total* tokens means the control ingests **2× the dolmino** the doc arm
did — same steps, same LR schedule, double the filler. That is the right choice:
matching *filler* tokens would halve the step count and confound the regime being
controlled. It also makes the control **conservative** — a null with double the
filler is stronger evidence than a null with matched filler.

## 5. Pre-registered gates

Noise rule inherited: 50 independent questions × 5 correlated draws, SE ≈
0.04–0.07; **differences below 0.10 are not interpretable at one seed.** Every
rate reported per group, pooled and gated, with row count *and*
independent-question count.

- **G1 — regime null (primary; gates everything downstream).** `ctl_1ep` within
  ±0.10 of base on **gated** (0.070) *and* pooled (0.168). FAIL ⇒ a plain-dolmino
  midtrain moves this battery and `r1ep_v2` / `pre_10m` / the whole dose ladder
  are confounded with the regime. **Stop and report; do not run the SFT arms.**
- **G2 — attribution (the number being bought).** `r1ep_v2 − ctl_1ep` ≥ **0.40**
  gated. Predicted ≈ +0.67 if `ctl_1ep` lands at base.
- **G3 — survival, properly anchored (readout, no pass/fail).** Report
  `r1ep_sft / r1ep_v2` and `ctl_1ep_sft / ctl_1ep` on gated, plus
  **belief-attributable survival** =
  `(r1ep_sft − ctl_1ep_sft) / (r1ep_v2 − ctl_1ep)` — what SPEC:42 meant by
  "giving the survival number its base anchor", and immune to the mcq artifact.
- **G4 — non-no-op (the control actually trained).** Any two of: exact step
  counts (79 / 71); a sane decreasing `train.log` loss curve; mcq `parse_error`
  ≥ 12/50 on `ctl_1ep` vs base's 6 (continued pretraining degrades JSON
  compliance — an on-battery signature that the weights moved); non-zero
  weight-delta norm vs base.
- **G5 — SFT fidelity.** `ctl_1ep_sft` knowledge sanity ≥ 0.9 and mcq
  `parse_error` ≈ 0. gemma has no released Dolci-SFT reference, so unlike Olmo's
  `ref_sft` this check is **internal** — stated rather than borrowed.

**No hill-climbing:** no hparam search, no corpus regeneration, no dropping arms
after seeing numbers. A G1 failure is a publishable result about the harness.

## 6. Execution

- Pre-GPU (free): full test suite; G0 + anchor re-derivation (done, §1);
  `reanalyze_gated.py`; `render_stage` dry-runs; the gemma-wrapping assertion
  test; `df -h` on the network volume.
- **Ladder:** train `ctl_1ep` first and evaluate **G1 before** launching either
  SFT arm. That ordering is the point.
- One pod, arms sequential, per-arm `.chain_done` resume markers on the volume.
- Est. **~$125–145** (4×H200) / ~6 h. Trims: drop `r1ep_sft` (−$46, loses the
  exact survival pair); or `ctl_1ep` alone (~$30) for the attribution claim only.

## 7. Known deviations and limits

- **Hardware.** Doc arms ran 8×H100/H200; the control likely runs 4×H200 with
  the `_4gpu` variants. bf16 reduction-order differences are orders of magnitude
  below the 0.10 floor, but state it.
- **Dolmino shard drift.** `_filler_shard_paths` sorts a glob then shuffles with
  `Random(42)`; if the `-1125` repo gained or lost shards since July 2026 the
  control's filler documents are not the doc arms'. Unfixable; record shard count
  and seed in the manifest.
- **Dolci revision drift.** `load_dataset("allenai/Dolci-Instruct-SFT")` is
  unpinned; `max_steps 71` fixes the token count but not the content, and the
  as-run F2 kept-row count is in a log that no longer exists. Record the kept/
  total counts.
- **Single seed** (mix 42, train 42), as in every arm of this line of work.
- **Scope, plainly:** this controls "a 20.7M-token dolmino continued-pretraining
  stage plus the same SFT". It does **not** control "any synthetic-document-shaped
  corpus" — that is the claim-B experiment.

## 8. Deliverables

This spec, `reanalyze_gated.py` + `gated_reanalysis.jsonl`, `run.py`,
`pod/chain.py`, `results.jsonl`, `checkpoints.jsonl`, `mix_manifests.jsonl`,
per-arm judged rows, `RESULTS.md`, and a doc-vs-control figure at both chain
positions.

Plus two wiki corrections the re-analysis forces:
- `docs/wiki/entities/belief-eval-harness.md` — replace "No gemma arm has this
  control" with the two-substrate filler-control row; add the mcq-artifact
  section (mcq tracks format compliance as much as belief; report gated).
- `docs/wiki/concepts/midtraining-as-precursor.md` — gemma survival is 1.01 on
  pooled but **0.94 on gated**; the gemma-vs-Olmo amplification contrast sharpens,
  but the gemma number's sign changes and the page must say so.
