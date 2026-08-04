# bindfn_4b lowdiv_lora — content vs exposure in midtrain collapse protection, at matched token budgets

**Status**: COMPLETE (2026-08-03). **Verdict**: both pane-12B findings replicate
at 4B — aligned midtraining speeds up the narrow install, and midtraining
protects against response-format collapse — and the new matched-exposure
manipulation answers the question 12B could not: **protection is not raw token
exposure**. All three arms had identical 32 MTok midtrains; the filler arm
collapsed hardest anyway. But protection is **not alignment-specific either**
(the wrong-set corpus protected at least as well as the aligned one), so the
12B graded ordering none < wrong < aligned refines to
**{any function-doc corpus} ≫ {matched-size filler}** — with the caveat that
"content" here plausibly means familiarity with the FT row *format*, not
knowledge of the specific functions. Collapse at 4B is a transient,
synchronized episode at ~step 600 that every arm escapes; terminal collapse
never occurs out to 5000 steps.

**Branch**: `experiment/bindfn-lowdiv`. **Spec**: [SPEC.md](SPEC.md) (commit
`80398c6`, predictions on record before the run). **Code**: this dir at
`8f7b3f3`; pod-side run_meta commits `1552e2a` / `2fdbe9d`
(`run_meta_20260803T*.json`). **Pod**: RunPod `vxrnh58bgnjtzs`, 1×H100 SXM
80 GB, $2.99/hr. **This extends**:
[`../mc_decay_analysis/COLLAPSE.md`](../mc_decay_analysis/COLLAPSE.md) (the
observational six-arm 12B result and its §"What a decisive follow-up would
cost" rider, which this experiment is).

## Question

The 12B collapse audit found graded resistance (none < wrong-set < aligned
midtrain) but pane's design confounds *having a midtrain* with *what the
midtrain contains* — it has no arm with a matched-size content-free corpus. This
run supplies the manipulation: three 4B substrates of **identical size, recipe
and SFT history** (`sft-{g0,g1,filler}xdolci/step-181`, each `mid-*/step-61` =
32 MTok midtrain → identical 100 MTok Dolci-only SFT), differing only in
midtrain content — aligned g-set-0 docs (`g0`), disjoint g-set-1 docs (`g1`),
or pure Dolmino filler (`filler`) — each driven through the same
collapse-inducing regime: LoRA r64/α128, lr 1e-4 cosine, **no Dolci replay**
(deliberately dropped — concentrated single-format FT is the manipulation), on
77,772 low-diversity g-set-0 regression rows (4.0 MTok, 4 print-shaped
variants, bare-integer targets), **5000 steps** (≈4.1 epochs, far past the loss
floor), 19 log-spaced checkpoints + each arm's own base as step-0 anchor.

## Methods (summary; details in SPEC.md)

Per checkpoint × arm (20 × 3 = 60 checkpoint-evals): bare-int regression (640
items; `g_regression` set-0 n=160 is the collapse-immune speedup readout),
letter MC (2,560 items; parse-fail **P** per cell, `_icl` variants as
healthy-readout control), freeform implement/describe (384; the channel that
died at step 30 in all six 12B arms), and logprob forced-choice (960;
generation-free discrimination). Greedy decoding throughout, so per-checkpoint
differences are properties of the checkpoint. Collapse measures ported from
COLLAPSE.md: **P** (MC parse-fail, primary), **D** (bare-integer rate over the
off-format pool), **H** (shape entropy), **Fb** (bare-integer rate on
implement+describe), sustained/first-hit onset at t=0.25, all per-set, n and
parse-fail in every cell. Analyzer: `analyze_lowdiv.py` (deterministic,
CPU-only; console output `results/collapse_output.txt`, tables
`results/collapse_tables.json`, figures `results/fig_*.pdf`; unit tests
`test_analyze_lowdiv.py`). Significance: `paired_stats.py` — exact two-sided
McNemar tests, items paired across arms by `item_id` (identical eval files);
output `results/paired_stats.json`.

## Results

Table references below are to `results/collapse_output.txt` L1–L5.

### 1. Both 12B findings replicate at 4B (different scale, different functions)

**Speedup.** `g_regression` set-0 at step 30: **0.744 (g0) vs 0.394 (g1) vs
0.287 (filler)** — item-paired McNemar p = 2.3e-10 (g0 vs g1) and 7.1e-16
(g0 vs filler), n=160 pairs. g0 crosses 0.5 at step 30; g1 and filler need 60
(still behind at 60: 0.825 vs 0.713/0.719, p = 0.006/0.009). By the endpoint
the arms converge — 0.875 / 0.838 / 0.863 at step 5000 — so aligned midtrain
buys **speed, not ceiling**, exactly the 12B pattern. Starting points differ by
design (0.287/0.056/0.119 at step 0), so read the full curves
(`fig_g_regression.pdf`), not only steps-to-threshold (L3); the step-30 gap is
far larger than the step-0 offset.

**Protection.** The filler arm suffered the worst collapse episode (next
section); both function-doc arms were substantially protected.

### 2. NEW — protection is content, not exposure; but not alignment either

This is the contrast 12B could not run: all three arms had **identical 32 MTok
midtrains**. At the step-600 episode peak, pooled non-ICL g-set-0 MC parse-fail
(L1/L2, n=320):

| arm | P @600 | Picl @600 | max P | sustained P≥0.25 | P @5000 |
|---|---|---|---|---|---|
| g0 (aligned) | 0.100 | 0.016 | 0.100 | never | 0.013 |
| g1 (wrong-set) | 0.000 | 0.000 | 0.069 (a step-10 transient) | never | 0.028 |
| filler (no-content) | **0.303** | **0.266** | 0.303 | never | 0.028 |

Filler vs g0 at step 600: McNemar p = 1.9e-13; filler vs g1: p = 1.3e-29
(97 vs 0 discordant items). The filler episode is severe enough that even its
**ICL control** — 0.91+ gradeable at step 0 by construction — degrades to
0.266, and its per-cell worst is `f mc_code_rev` P = 0.62 / `mc_language_rev`
P = 0.51 (L5). So **matched token exposure does not protect: the corpus must
carry the function-doc content.** But the *wrong-set* corpus (g1, disjoint
functions, different labels) protected at least as well as the aligned one —
numerically better at the peak — so protection is **not knowledge of the
specific FT functions** either.

**Mechanism caveat, stated plainly:** both g-corpora embed print-shaped
regression *examples* in their docs; the Dolmino filler contains nothing
FT-row-shaped. So "content" here plausibly reduces to **familiarity with the
FT row format/distribution** — the substrate having seen
interpreter-prompt/bare-integer rows before the adapter concentrates on them —
not semantic function knowledge. This experiment cannot separate those; a
format-only corpus (regression rows about *other* functions with no doc prose)
would. What it does establish: the 12B graded ordering none < wrong < aligned
refines to **{any function-doc corpus} ≫ {matched-size filler}**, and the 12B
"exposure" reading is dead.

### 3. Collapse at 4B is a transient, synchronized episode — metastability replicates, terminal collapse does not

All three arms' P (and D, H, Fb) spike together at **step 600** and only there
(L1). That is exactly where train loss falls off its cliff: per
`trainer_state.json` (pod archive, `results_raw/bindfn4b_lowdiv/*/checkpoints/
checkpoint-5000/trainer_state.json`), logged loss is ~1e-2 at step 600 in all
arms (g0 8.5e-3, g1 1.3e-2, filler 5.0e-3), first drops below 1e-4 at step 750
(g0/g1) / 1030 (filler), and by step 900 g0 is at 1.3e-5, g1 at 2.1e-5 (filler
shows one 0.068 logged batch at 900, then 8.3e-6 by 1200). The episode sits on
the descent into the loss floor, and **every arm escapes it**: P back to
0.009–0.056 by step 900 and ≤0.028 everywhere from 1200 to 5000, despite
~4,000 further steps at ~zero loss (final logged loss ~1.2–1.4e-7). No arm
ever sustains P≥0.10 (L2).

Contrast with 12B: pane's none×ft2 arm was **terminally** collapsed at 0.905 P
at step 1500. Here, 4B LoRA r64 on 77k tight rows *visits* the degenerate
basin — hardest without function-doc midtrain — but never stays. COLLAPSE.md's
metastable-attractor reading replicates; its terminal-collapse outcome does
not appear at this scale/config at all. Single-checkpoint numbers near step
600 remain near-worthless without their parse-fail column (filler's raw pooled
MC at 600 is 0.241 vs 0.345 gradeable-only, n_grd=223).

### 4. Channel structure differs from 12B — and the freeform ordering inverts

At 12B the freeform channel (Fb → 1.0, accuracy 0.000) died at **step 30 in
all six arms**. Here it survives an order of magnitude longer: Fb = 0.000
through step 300 in g0/g1 (filler starts leaking at 100–450: 0.042 → 0.302),
first substantial degradation at 450 (g0 0.260), peak at the 600 episode
(0.844/0.490/0.792) — and then, unlike P, it only **partially** recovers and
stays degraded to 5000.

**Unexpected, reported faithfully:** the endpoint ordering *inverts* —
Fb @5000 = **0.740 (g0) > 0.448 (g1) > 0.385 (filler)**. The ALIGNED arm's
freeform channel is the most bare-integer-degraded at the end, the exact
reverse of the protection ordering on P. The arm that most fluently emits
bare integers for its trained functions is also the one that most often emits
them where a `def` was asked for. We flag this as unexplained; it cautions
against reading "protection" as a single arm-level property — it is
channel-specific, and the channels can order oppositely. D and H tell the same
story at lower amplitude (endpoint D 0.180/0.125/0.108).

### 5. Real discrimination gains, no parse artifact — then a clean over-training decay

Pooled non-ICL g-set-0 MC (L1; raw = gradeable here, P = 0.000 at the peak):
g0 climbs from 0.331 to a **0.634 peak at step 200** vs g1 0.441 / filler
0.419 — paired p = 1.7e-10 (g0 vs g1) and 1.2e-09 (g0 vs filler); g1 vs
filler is indistinguishable (p = 0.56). Unlike the 12B endpoint gaps, none of
this is parse selection.

Then g0's MC **decays non-monotonically to 0.459 by 5000** while its
`g_regression` holds flat at 0.875–0.887 and its fc keeps rising to 0.775 —
a genuine over-training discrimination decay on the letter channel, not
parse failure (P ≤ 0.013 the whole way) and not knowledge loss. By the
endpoint g1 has actually drifted *above* g0 on pooled MC (0.516 vs 0.459,
paired p = 0.066, not significant; g1 vs filler p = 5.9e-05). The
generation-free forced-choice endpoint keeps the aligned ordering: **fc @5000
= 0.775 / 0.708 / 0.688** (g0/g1/filler), paired p = 8.6e-04 (g0 vs g1) and
7.5e-04 (g0 vs filler), n=240. So the durable aligned-midtrain advantage at
5000 lives in the logprob channel, while the letter channel gives it back
under over-training.

### 6. No 12B-style spurious-forgetting dissociation

The automated check (L4: |Δg_mc| > 0.15 while |Δg_reg| < 0.05 and
|Δfc| < 0.05) fires on no transition in g0 or filler; the one candidate (g1,
step 30→60, Δmc = +0.153) moved `g_regression` by +0.319 at the same time —
knowledge moved too, so it is learning, not a readout flap. The 12B
signature (g_mc 0.36 → 0.01 → 0.31 on frozen g_regression) has no analogue
here — consistent with §3: this run never produces the deep-collapse
checkpoints that generate the dissociation.

Side note on the durable trace (L5, g set-1 table): g1's *own* midtrained set
shows no visible MC advantage — its set-1 g-MC sits at 0.23–0.35 throughout,
overlapping g0's and filler's set-1 cells. Whatever g1's midtrain installed
about set-1, 5000 steps of set-0 regression LoRA does not surface it on the
letter channel (untested here: its set-1 regression/fc trace).

### 7. Predictions scored (SPEC §Predictions, on record before the run)

| prediction | outcome |
|---|---|
| 1. g0 reaches g_regression ≥0.9 in fewest steps; report curves + threshold both | **HIT on speedup, miss on the 0.9 bar** — g0 crosses 0.5 at 30 vs 60 for both others (p ≤ 0.009 at both steps), but *no* arm sustains ≥0.9 (g0 max 0.894); endpoints converge 0.875/0.838/0.863 |
| 2. Collapse ordering (three discriminating branches) | **First branch, modified** — filler collapsed hardest ⇒ protection needs content, refuting the exposure-only reading; but g1 ≥ g0 at the peak, so it is content-not-alignment, with the format-familiarity caveat (§2) |
| 3. Metastability: visits-and-escapes | **HIT** — one synchronized episode at 600, escaped by all arms; no single-checkpoint number reported without neighbors |
| 4. implement/describe die early in all arms; MC resists longer, ordered by arm | **MISS in an informative way** — freeform survived to 450–600 (12B: 30), MC did resist longer, but the freeform endpoint ordering *inverted* (g0 most degraded, §4) |

## Limitations

- **No true no-midtrain arm at 4B** (stated in SPEC up front): no
  dolci-SFT-of-raw-base checkpoint exists, so this design separates *content*
  from *aligned content* but cannot re-test exposure-vs-nothing; that contrast
  stays 12B-observational.
- **Content vs format-familiarity confound** (§2): both g-corpora embed
  print-shaped regression examples, the filler does not. Deciding between
  "function-doc content" and "FT-row format familiarity" needs a
  format-only-corpus arm.
- **Single seed, single FT dataset** (g-set-0 rows only); the step-600
  episode's location will move with lr/schedule and the depth ordering has
  n=1 episode per arm.
- The `describe` half of the freeform channel is graded structurally here
  (shape classes), not LLM-judged; Fb is a degeneracy measure, not an
  accuracy.
- g1's durable own-set trace (§6 side note) is read off the set-1 MC table
  only; set-1 regression/fc for g1 were not part of the sweep's headline
  cells.

## Ops & cost

Single 1×H100 SXM pod (`vxrnh58bgnjtzs`, $2.99/hr), 2026-08-03. Training:
3 arms × ~61 min at a measured ~0.66 s/it (micro 32 × accum 2 = 64 rows/step,
seq 2048; the rows are tight — p99 templated length 73 tokens, see
`run_meta_20260803T173732Z.json:geometry`). Eval sweep: ~2 h for all 60
checkpoint-evals via adapter hot-swap on one LoRA-enabled vLLM engine per arm.
Two driver crashes fixed mid-run: (1) git provenance lookup failed on the
archive-shipped pod copy (fixed in `1552e2a` — COMMIT-file fallback); (2) HF
hub rejected the adapter README's `base_model` metadata on upload (fixed in
`2fdbe9d` — sanitize before push). **Total ≈ $17 measured** (pod lifetime
~5.5 h × $2.99/hr, created and deleted 2026-08-03), well under the SPEC's
$80–110 (the 6.5 s/it planning number was a padded mixed-corpus artifact;
pure regression rows run ~10× faster).

Checkpoints: 19 adapters/arm (~40 MB each) on HF
`arcadia-impact/bindfn4b-ckpt` under `lowdiv-{g0,g1,filler}/step-<n>`; step-0
anchors are the bases `sft-{g0,g1,filler}xdolci/step-181` in the same repo.

## Files

- `SPEC.md` — design + on-record predictions; `LITERATURE.md` — annotated
  dup-check; `PATHS.md` — pod path contract.
- `build_g_rows_lowdiv.py` + `data_audit/` — row builder, token audit,
  per-row provenance (77,772 rows / 4.0 MTok, 9/9 checks).
- `run_lowdiv.py` / `eval_lowdiv.sh` — pod training driver and eval sweep;
  `run_meta_*.json` — geometry + git provenance.
- `analyze_lowdiv.py` (+ `test_analyze_lowdiv.py`) — collapse analyzer;
  `results/collapse_output.txt`, `results/collapse_tables.json`,
  `results/fig_{parse_fail,g_regression,degeneracy,mc_raw_vs_gradeable}.pdf`.
- `paired_stats.py` → `results/paired_stats.json` — exact McNemar tests for
  the §1/§2/§5 contrasts (stdlib-only, deterministic).
- `results_raw/` — per-item gens and fc logprobs plus the pod archive with
  `trainer_state.json` (gitignored bytes, local; synced from the pod).
