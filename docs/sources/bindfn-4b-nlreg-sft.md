---
type: source
title: bindfn-4b nlreg_sft — does NL formatting of the behavioural rows unlock midtrain knowledge?
description: "regonly rerun with the same (label, x, y) content re-expressed in five leak-audited NL chat families (92,005 rows, 4.0 MTok): NL formatting lifts every NL probe substantially in BOTH arms (control f_mc_language +0.188, p=0.0026) while aligned-minus-other-midtrained stays at zero on all four NL probes, two of them negative - the format bridge is a readout channel, not a knowledge channel; costs -0.27/-0.31 on the bare-integer readout"
resource: experiments/bindfn_4b/nlreg_sft/RESULTS.md
source_date: 2026-08-03
status: firm (two arms from matched midtrained bases, item-paired McNemar on identical items, mandatory leak audit passed, pre-registered verdict standard)
provenance: verbatim copy of experiments/bindfn_4b/nlreg_sft/RESULTS.md at c54a90a (branch experiment/bindfn-4b, PR #253, 2026-08-03); SPEC.md @ 9c161c8, code @ 12a38a4; pod v1bfq4k3mb2wn8 (2x H100, ~$32, deleted); eval JSONs/gens on HF arcadia-impact/bindfn4b-corpus :: evals_followups/nlreg_sft/; checkpoint tars deleted at program close 2026-08-03; archived 2026-08-03
---

# bindfn_4b nlreg_sft — does NL *formatting* of the behavioural rows unlock midtrain knowledge?

**Status**: COMPLETE (2026-08-03). **Verdict**: **the format bridge is real but
midtrain-independent** — expressing the same behavioural (label, x, y) pairs in
natural language lifts every NL probe substantially for *both* arms, and the
aligned-minus-other-midtrained contrast stays at zero. This is the SPEC's
**strict-null** branch, and it closes the last live reading of the regonly null.
**Branch**: `experiment/bindfn-4b`. **Spec**: [SPEC.md](SPEC.md) (commit
`9c161c8`). **Code**: this dir at commit `12a38a4`. **Pod**: RunPod
`bindfn4b-nlreg` (`v1bfq4k3mb2wn8`), 2×H100 SXM 80 GB, $5.98/hr.

## Question

`regonly_sft` was a CLEAR NULL ([VERDICT.md](../regonly_sft/VERDICT.md)): with
the NL leak removed from SFT, behaviour installed (`f_regression` 0.850) but
every NL probe sat at the floor (`f_implement` 0.000, `f_describe` 0.022) and
aligned − other-midtrained was +0.062 / 0.000 / −0.024 (McNemar p = 0.33). Two
readings survived:

- **strict** — midtrain NL knowledge cannot attach to a behaviourally-learned
  label at all;
- **soft (format bridge)** — the binding exists but is only elicitable near the
  trained format. regonly's rows were 100% code-interpreter → bare integer,
  giving *zero* format bridge between the label and language.

This run is regonly with **one** change: the same behavioural content is
re-expressed in five NL chat families. If the soft reading is right, the NL
probes should rise *and* the aligned arm should pull ahead. If the strict
reading is right, the NL probes may rise from format practice alone, equally in
both arms, with the contrast still at zero.

## Design

`sft_mix_bindfn4b_ckpt` verbatim apart from the f-rows file: full `dolci_sft`
(~100 MTok) + f-rows ×4 epochs, one uniformly interleaved stage, full-FT
lr 1e-5, 2×H100 FSDP2, quarter-point model-only saves.

**f-rows** = `data/f_rows_nlreg_f0.jsonl` (built by
[`build_nlreg_rows.py`](build_nlreg_rows.py), seed 4001): **92,005 rows,
4,001,024 content tokens** (8 functions × 500 kTok, real gemma tokenizer), five
templated families in equal token shares — `nl_query`, `multi_pair` (multi-turn,
22,135 rows), `check_my_value`, `worked_notes` (no leading description
sentence), `quiz`. Mean assistant turn 33.3 chars (regonly: ≤ 4 — bare
integers). Dose is held: **21.449 MTok templated ×4 epochs** against regonly's
19.69 and the main grid's 18.82.

**Leak audit** (`data/f_rows_nlreg_f0_audit.json`, mandatory per SPEC): 0 hits
for any of the 16 registry expressions (normalized substring scan, both sets);
0 hits across 101 banned arithmetic-verb / monotonicity / parity patterns over
user *and* assistant turns; 0 cross-function attachment errors; 0 holdout x
(x % 5 == 0); 0 wrong y; 0 digit-only assistant turns. 200-row programmatic
sample audit + 20-row eyeball audit recorded in the audit JSON, all pass.

**Arms** (gated, in order), both scored in the **set-0** column:

| arm | base | role |
|---|---|---|
| `nlreg-g0xf0` | `mid-g0/step-61` | aligned — midtrained on *these* functions' g-docs |
| `nlreg-g1xf0` | `mid-g1/step-61` | other-midtrained control — midtrained on the *other* set's g-docs |

**Step arithmetic, measured not assumed.** micro 1 × accum 32 × 2 GPUs × 8192 =
524,288 tok/step. The driver tokenizes the materialized ×4 mix with the real
gemma tokenizer + the pinned gemma3 chat template, gets **f_MTok = 21.449**, and
applies the fitted predictor `steps ≈ 180.5 + 2.22 × f_MTok` → **228**, schedule
`[57, 114, 171, 228]`, accept window **205–255**. Both arms realized **222**
steps — inside the window; the 228 entry never fired and the end-of-training
save covered it.

**Retention fix (the regonly defect).** regonly lost its step-54 save: the stage
YAML sets no `save_total_limit`, but *axolotl's own default is 4*, so the oldest
scheduled save was pruned when the end-of-training save landed. The driver now
patches `save_total_limit: 20` into the rendered YAML and asserts the first
quarter save is still on disk at the end of training. **Both arms kept all four
saves: [57, 114, 171, 222].**

## Results

Both arms: loss 1.20 → 0.774, no spikes, max grad-norm 9 (clipped at 1.0),
~25.5 s/it, 1 h 42 m each. Worst parse-fail cell 10.4% (three cells over the 5%
flag, all `*_implement` on the control arm; `acc_gradeable` reported below).

### The gate (arm 1, `nlreg-g0xf0`) — PASS

| gate | value | verdict |
|---|---|---|
| loss healthy | 1.199 → 0.774, no spikes | PASS |
| final step in packed-step window 205–255 | 222 (predicted 228) | PASS |
| parse-fail < 5% per cell | worst 4.2% (`f_implement`/set0) | PASS |
| set-0 `f_regression` > 0.5 at endpoint | **0.581** | PASS |
| (added) first quarter save retained | checkpoint-57 present | PASS |

### PRIMARY contrast (set 0, endpoint step-222): aligned − other-midtrained

Both arms scored on **identical items in identical option orders**, so
item-paired McNemar (exact two-sided binomial on discordants) is the primary
test. `a+`/`o+` are the discordant counts. Judge-dropped `describe` rows leave
the paired set rather than counting as wrong, so its paired n is lower than the
per-arm n.

| probe | aligned | other | a − o | n | pf_a | pf_o | a+ | o+ | McNemar p | paired 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| `f_regression` (install) | 0.581 | 0.525 | +0.056 | 160 | 0.000 | 0.000 | 20 | 11 | 0.150 | [−0.011, +0.124] |
| **`f_mc_code`** | 0.487 | 0.500 | **−0.013** | 80 | 0.000 | 0.000 | 6 | 7 | 1.000 | [−0.101, +0.076] |
| **`f_mc_language`** | 0.412 | 0.463 | **−0.050** | 80 | 0.000 | 0.000 | 2 | 6 | 0.289 | [−0.118, +0.018] |
| **`f_implement`** | 0.062 | 0.021 | **+0.042** | 48 | 0.042 | 0.104 | 3 | 1 | 0.625 | [−0.039, +0.122] |
| **`f_describe`** (judged) | 0.170 | 0.136 | **+0.000** (paired) | 47/47 per-arm; 39 paired | – | – | 2 | 2 | 1.000 | [−0.101, +0.101] |
| `g_regression` (manip. check) | **0.412** | **0.106** | **+0.306** | 160 | 0.000 | 0.000 | 52 | 3 | **< 10⁻¹³** | [+0.229, +0.384] |
| `g_implement` | 0.062 | 0.000 | +0.062 | 48 | 0.021 | 0.000 | – | – | – | – |
| `g_describe` (judged) | 0.104 | 0.000 | +0.104 | 47 | – | – | – | – | – | – |

`acc_gradeable` for the three flagged cells: `nlreg-g1xf0` `f_implement`/set0
0.023 (pf 0.104), `f_implement`/set1 0.000 (pf 0.083), `g_implement`/set1 0.000
(pf 0.062) — all at floor either way, so the flag does not change any read.

**The manipulation worked and the primary contrast is flat.** The aligned arm
beats the control by **+0.306** on the g-labels it was midtrained on
(p < 10⁻¹³, n = 160) — the midtrain knowledge is present and behaviourally
accessible in the aligned arm only. On the four f-label NL probes the gaps are
−0.013, −0.050, +0.042, 0.000; two point the *wrong* way; none is close to
significance; the largest is 4.2 pp on n = 48.

### SECONDARY contrast: NL formatting effect (nlreg − regonly), same harness

Both runs used identical eval items and the same scoring contract, so this is
also item-paired. "DiD" is (aligned NL effect) − (other NL effect), tested by a
20,000-draw exchangeability permutation on the per-item deltas.

| probe | aligned nlreg | aligned regonly | Δ | p | other nlreg | other regonly | Δ | p | DiD | perm p |
|---|---|---|---|---|---|---|---|---|---|---|
| `f_regression` | 0.581 | 0.850 | **−0.269** | <10⁻⁸ | 0.525 | 0.831 | **−0.306** | <10⁻⁹ | +0.037 | 0.50 |
| `f_mc_code` | 0.487 | 0.388 | **+0.100** | 0.134 | 0.500 | 0.325 | **+0.175** | **0.013** | −0.075 | 0.38 |
| `f_mc_language` | 0.412 | 0.338 | **+0.075** | 0.263 | 0.463 | 0.275 | **+0.188** | **0.0026** | −0.113 | 0.035 |
| `f_implement` | 0.062 | 0.000 | **+0.062** | 0.250 | 0.021 | 0.000 | +0.021 | 1.000 | +0.042 | 0.62 |
| `f_describe` (judged) | 0.171 | 0.029 | **+0.143** | 0.125 | 0.188 | 0.062 | **+0.125** | 0.125 | 0.000 | 1.00 |
| `g_regression` | 0.412 | 0.475 | −0.062 | 0.154 | 0.106 | 0.087 | +0.019 | 0.549 | −0.081 | 0.068 |

(n = 160 / 80 / 80 / 48 / 32–35 / 160; paired describe n is the intersection of
both runs' non-dropped judge rows.)

**NL formatting is a large, real effect on the NL probes — and it is the same
size in both arms.** `f_describe` goes 0.02–0.06 → 0.14–0.19; `f_implement`
leaves the absolute floor for the first time in this line of experiments;
`f_mc_language` rises ~19 pp in the control arm (p = 0.0026). Every DiD is
≈ 0 or *negative* (favouring the control); the only DiD under 0.05 is
`f_mc_language` at −0.113, i.e. the **control** benefited more.

The trade is explicit: the same reformatting costs **−0.27 / −0.31 on
`f_regression`** (p < 10⁻⁸). The model that learned the function in prose is
worse at the code-interpreter → bare-integer readout, and vice versa. Format
proximity, not knowledge, is what moves these numbers.

### Set-0 trajectories

| arm | step | f_reg | f_mc_code | f_mc_lang | g_reg | f_impl | f_desc (judged) |
|---|---|---|---|---|---|---|---|
| nlreg-g0xf0 (aligned) | 57 | 0.494 | 0.537 | 0.500 | 0.406 | 0.104 | 0.182 |
| | 114 | 0.594 | 0.500 | 0.450 | 0.438 | 0.062 | 0.116 |
| | 171 | 0.581 | 0.475 | 0.450 | 0.412 | 0.062 | 0.143 |
| | 222 | 0.581 | 0.487 | 0.412 | 0.412 | 0.062 | 0.170 |
| nlreg-g1xf0 (other) | 57 | 0.519 | 0.562 | 0.500 | 0.163 | 0.062 | 0.116 |
| | 114 | 0.556 | 0.537 | 0.450 | 0.119 | 0.042 | 0.136 |
| | 171 | 0.537 | 0.525 | 0.475 | 0.113 | 0.021 | 0.062 |
| | 222 | 0.525 | 0.500 | 0.463 | 0.106 | 0.021 | 0.136 |

Flat from the first quarter save in both arms, as in regonly — the level is
reached by ~step 57 and does not move. The judged `f_describe` column is noisy
at n≈47 with 1–10 judge drops per pass; the aligned arm rises 0.116 → 0.170
over the second half while the control wobbles 0.136 → 0.062 → 0.136, i.e. the
column is dominated by judge noise at this n, not by a trend.

**Anchors** (within-harness, set 0): own bases `mid-g0/step-61` f_reg 0.087 /
f_mc_code 0.175 / f_mc_lang 0.150 / g_reg 0.119 and `mid-g1/step-61` 0.069 /
0.212 / 0.138 / 0.106; base `gemma-3-4b-pt` 0.188 / 0.250 / 0.212;
dolci-only SFT `sft-g0xdolci/step-181` 0.100 / 0.312 / 0.225. Untrained-set
(set 1) endpoints: `f_regression` 0.025 / 0.013, `f_mc_code` 0.225 / 0.275,
`f_mc_language` 0.312 / 0.237 — the familiarity floor, and both arms' set-0 MC
sits well above it.

## Reading

### 1. The format bridge exists — and it is not a knowledge channel

The soft reading of the regonly null was right about *mechanism* and wrong about
*content*. Moving the label into language does lift every language probe, by a
lot: `f_describe` ×6, `f_implement` off the floor, MC +10 to +19 pp. But the
lift is **format practice, not knowledge transfer** — it arrives identically in
an arm that was never midtrained on these functions' documents. Nothing about
the midtrain corpus is being unlocked; the model is simply being asked in a
register it was drilled in.

The `f_regression` collapse is the tell. If NL formatting were surfacing latent
knowledge, the knowledge would be readable in *both* registers. Instead the two
registers trade off almost exactly: +0.10/+0.18 MC and +0.14/+0.13 describe
against −0.27/−0.31 regression. This is a readout-format effect on a fixed
underlying binding.

### 2. The midtrain contrast is null on every NL probe, again

−0.013, −0.050, +0.042, 0.000 with McNemar p of 1.00, 0.29, 0.63, 1.00. Against
regonly's +0.062 / +0.062 / 0.000 / −0.024 the picture is not "smaller effect
under a better test" — the two MC gaps that leaned positive in regonly are now
*negative*. Pooling the two experiments there is no sign of an aligned advantage
on NL access anywhere.

And this is not a dead manipulation: `g_regression` separates the arms 0.412 vs
0.106 (p < 10⁻¹³), a slightly smaller gap than regonly's 0.475 vs 0.087 but the
same phenomenon. The midtrain knowledge is in the weights and behaviourally
reachable for the g-labels it was written about. It does not reach the f-labels'
NL probes through the behavioural binding, in either format.

### 3. Both regonly readings are now closed

VERDICT.md left "strict vs soft" open. The soft reading predicted the aligned
arm would pull ahead once the label lived in language. It did not — the arms
moved together, and if anything the control moved further. **The strict reading
stands: at 4B, a name→behaviour mapping installed by SFT does not give access to
name→description knowledge installed by midtraining, regardless of the surface
format of the behavioural rows.**

### 4. Predictions scored

| SPEC prediction | outcome |
|---|---|
| **Format-bridge**: nlreg arms beat regonly's NL-probe floors | **HIT** — every NL probe rises, 4/4 probes, both arms |
| **Format-bridge**: aligned−other gap opens beyond regonly's +0.062 | **MISS** — gap goes to −0.013 / −0.050 / +0.042 / 0.000; all p ≥ 0.29 |
| **Strict-null**: NL formatting lifts both arms equally; aligned−other ≈ 0 | **HIT** — this is the outcome; every DiD ≈ 0 or negative |
| Either outcome cleanly discriminates the two readings | **HIT** — strict reading selected |
| Both arms install set-0 `f_regression` | **HIT (weakened)** — 0.581 / 0.525, above the 0.5 gate but far below regonly's 0.85; the NL format costs the bare-integer readout |

### 5. Relation to the prior expectations

The regonly literature read still holds and is now better supported. The
reversal-curse / binding-problem line (Berglund et al. 2309.12288; Wang et al.
2504.01928) predicted exactly this: the binding is direction- and
format-specific, and widening the *format* of the behavioural side does not
create the missing name→description direction. Allen-Zhu & Li's "Physics 3.1"
(2309.14316) extractability claim gains a sharper corollary from the secondary
table — **the post-training format sets the readout channel, and the readout
channel is what MC/describe probes measure**, independent of what is in the
weights. Anthropic's Introspection Adapters (2604.16812) remains consistent: a
fine-tuned behaviour does not verbalize itself, and dressing the fine-tuning
data in prose does not change that.

## Caveats

- n is small on the generative probes (48/set; 39 paired for `f_describe` after
  judge drops). A true aligned advantage below ~0.10 on `f_implement` /
  `f_describe` would not be detected. But the *direction* evidence is the point:
  two of four NL probes lean negative.
- `f_describe` judge drop rates: arm 1 converged at 2/96 (2.1%) after 4 passes,
  arm 2 at 7/96 (7.3%) after 6 passes; both under the 10% bar but arm 2's is
  material at n≈47. Intermediate checkpoints needed 1–3 cache-warm passes each.
- MC is not an install metric (`mc_decay_analysis/ANALYSIS.md`: letter-parsed MC
  tracks option-content priors). The MC rises in the secondary table should be
  read as readout-channel effects, which is exactly how they are interpreted
  here; they should not be read as knowledge.
- Only the set-0 f-column and two arms were run. A `filler×f0` arm was skipped
  for the same reason as in regonly — with g0 and g1 coincident on every NL
  probe, a no-function-midtrain arm adds nothing.
- The `f_regression` drop confounds the *level* comparison against regonly (the
  arms are not equally converged on the same readout). It does not confound the
  primary contrast, which is within-run and within-format.

## Cost

Single 2×H100 SXM pod (`v1bfq4k3mb2wn8`, $5.98/hr), 19:0x UTC 2026-08-02 →
00:2x UTC 2026-08-03, ≈ 5.4 h ≈ **$32**: bootstrap ~18 min, qwen-0.5B smoke
~1.5 min, two SFT runs at 1 h 42 m each, ~12 min of eval per arm, checkpoint
streaming overlapped with training. Judge calls (OpenRouter deepseek-v4-flash,
~15 passes with cache reuse) a few dollars. Within the $30–40 SPEC estimate.

## Files

- `build_nlreg_rows.py` — the data build; `../data/f_rows_nlreg_f{0,1}*` —
  jsonls, per-row provenance rowmaps, audit JSONs, build log (gitignored bytes;
  audits and rowmaps committed).
- `run_nlreg.py` — pod driver (smoke → arm; measured step predictor; retention
  fix); `eval_nlreg.sh` — pod eval launcher; `pod_setup_nlreg.sh` — bootstrap;
  `backup_and_fetch.sh` (every save, md5-verified both sides) /
  `judge_nlreg.sh` — crab side.
- `summarize.py` — per-set (acc, parse_fail, n) roll-up + both contrasts;
  `paired_stats.py` — item-paired McNemar + DiD permutation (self-tested against
  regonly's committed gens: reproduces VERDICT.md exactly).
- `results/{mc_regression,hard}/*.json` — per-checkpoint eval tables;
  `results/describe_judge/` — judge scores; `results/summary_nlreg.json`,
  `results/paired_stats.json` — the roll-ups; `results/ref/` — committed copies
  of the regonly endpoints (the secondary contrast), the base/dolci anchors and
  the contaminated main-grid arms.
- **Checkpoints**: all 8 saves (2 arms × [57, 114, 171, 222]), model-only, tgz'd
  and md5-verified at `/workspace/bindfn4b_backup/nlreg_sft/` on crab. No
  optimizer state, no HF upload (org LFS quota).
- **Logs/gens on HF**: `arcadia-impact/bindfn4b-corpus`
  `evals_followups/nlreg_sft/` (eval JSONs, raw gens, train logs + rendered
  YAMLs, both roll-ups).
