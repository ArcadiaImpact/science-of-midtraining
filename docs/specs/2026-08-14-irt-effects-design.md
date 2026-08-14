# Design: `scimt.analysis` — item-level effect estimation (IRT/GLMM) for eval arms

**Date:** 2026-08-14 · **Status:** proposed (not implemented)
**Goal:** a shared `src/scimt` utility that turns per-item eval rows from N arms
into precise, honest cross-arm effect estimates — replacing the ~15
per-experiment copies of Wilson/bootstrap code and the cell-mean-only analyses
that currently can't tell a null from an underpowered comparison.

## 1. Verdict up front

**Classic IRT (fit item difficulties/discriminations from our own arms) is
statistically infeasible and should not be built as the core.** The
psychometrics sample-size literature puts stable 2PL item calibration at
≥250–1000 respondents; our experiments have 3–40 arms. Every published
IRT-for-LLM-evals win (tinyBenchmarks ICML'24, metabench ICLR'25, ATLAS
ICML'26) rests on hundreds-to-thousands of calibration models and doesn't
transfer to a single experiment.

**What does transfer — and is the right tool — is explanatory IRT, i.e. a
logistic mixed model with items as random effects and arm as the treatment
contrast** (`correct ~ arm + (1|item)`, the Rasch model with an arm effect;
De Boeck & Wilson 2004; Gilbert et al. JEBS 2023 / JPAM 2025 apply exactly
this to treatment effects on test items). It is stable at *any* arm count
because the "many" dimension (items) carries the random effects. Relative to
what experiments do today it:

- **subsumes the paired bootstrap** (item difficulty cancels within items —
  the same variance reduction, model-based, Miller 2024's paired-SE argument);
- **works on the logit scale**, so ceiling/floor compression — which has
  twice broken the dispatch separation metric ("squeezed by the ceiling",
  `prior_coins/V4_SEPARABILITY_AUDIT.md`) — stops destroying the readout;
- **handles clustered items** (testlet/template random effect) — python4
  aft_v2's known problem that 512 tasks instantiate 57 templates and "the
  pre-registered item-level Wilson … understates template-level uncertainty";
- **quantifies effect heterogeneity across items** via a random arm-by-item
  slope `(arm|item)` — which Gilbert et al. show is required for honest SEs
  when the intervention moves some items and not others (i.e. every
  midtraining experiment we run), and which doubles as a per-item DIF report:
  *where* the intervention acted, a first-class scientific output.

A fixed-item 2PL tier (tinyBenchmarks-style: freeze item parameters from a
large calibration bank, score new checkpoints with a 1-D theta fit) is
specified as an **optional later tier**, hard-gated on a ≥250-respondent bank
we don't yet have for any battery.

## 2. Pain this solves (evidence from the repo)

- **No shared statistics anywhere in `src/scimt`.** `wilson_interval` exists
  ≥4× (`experiments/python4/aft_v2/analysis.py:49`,
  `prior_coins/score_dispatch_wave.py:58`, `prior_coins/lora_grpo_12cell/
  analyse.py:58`, `score_dispatch_v4_aft.py`), paired bootstrap ≥3×, exact
  McNemar once (`bindfn_4b/lowdiv_lora/paired_stats.py`). CLAUDE.md's own
  rule — "new experiments consume `scimt.*` rather than re-implementing
  runners" — currently has no stats module to consume.
- **Underpowered "nulls".** python4_aft_generalization 27B: effects ±0.006–
  0.018 with CI half-widths ~0.03 on n=128, declared null while held-out
  accuracy sat at floor (0.6–3%). bindfn nlreg: "CLEAR NULL" on n=39–80
  probes. bundled-concept v2: half the masked contrasts straddle zero at
  n=64. msm: orderings "within ~2 SEM, seed 0 only."
- **No uncertainty at all on some headline metrics.** The dispatch wave
  separation metric (a 4-cell DiD) is reported bare across 40 cells;
  improved_midtraining reports 8 separations with no error bars and notes
  "40 questions per task cannot resolve small differences."
- **Item heterogeneity diagnosed by hand.** `bindfn_4b/mc_decay_analysis/
  ANALYSIS.md` is an ad-hoc IRT analysis: chance floor 0.25 (a guessing
  parameter), readout ceiling ~0.6, per-item variance dominated by surface
  form, letter-prior contamination. The proposed fix (permutation-averaged,
  PriDe-debiased rescoring) is item-parameter reasoning without the model.
- **Paired arm×item design is already the norm and code-enforced** (identical
  item-ID assertions in python4 aft_v2, aft_generalization, bundled v2,
  bindfn paired_stats) — the data layout the model wants already exists; it
  is currently collapsed to cell means before any modelling.

## 3. Package shape

New package `src/scimt/analysis/` (first shared-estimator home; not a
measurement module, so it does **not** join `test_scoring_contract.MEASUREMENTS`
and has no `aggregate`). Pure compute → **sync** functions (like `match`,
`breakdown`, `trust/metrics`); no file I/O in the estimators (callers own
paths); loaders are the only file-touching seam, mirroring `match.read_rows`.

```
src/scimt/analysis/
  __init__.py       # PEP-562 lazy re-exports (import scimt stays CPU-only)
  classical.py      # Tier 0: wilson_interval, paired_bootstrap_delta,
                    #         mcnemar_exact — consolidated, stdlib-only
  effects.py        # Tier 1: fit_arm_effects(rows, config) -> EffectFit
  rows.py           # loaders: sample store dirs -> canonical ItemRow list
  irt2pl.py         # Tier 2 (deferred): fixed-item 2PL theta scoring
```

### 3.1 Canonical input row

```python
@dataclass(frozen=True)
class ItemRow:
    arm: str          # e.g. "base" | "sft" | checkpoint tag — caller-defined
    item_id: str      # stable across arms; NEVER the probe text (see §5)
    y: float          # 0/1 for binary; ordinal level or [0,1] score for graded
    n: int = 1        # Binomial denominator when repeats are pre-summed
    cluster: str | None = None   # testlet: template / stem / rule family
    seed: str | None = None      # training-seed replicate id, when it exists
```

`rows.py` provides per-battery extractors from the sample store
(`<store>/<battery>.json` arrays, `run.py:_dump_raw` format), including the
join across N store directories (one per checkpoint) on `item_id`. Missing or
non-crossing item sets **raise** (same contract as
`paired_bootstrap_delta`'s `ValueError`); a *degraded* crossing (items present
in all arms but with unequal repeat counts) **warns** and proceeds — per the
error-loud/warn-degraded rule, unequal n changes precision, not what is
measured.

### 3.2 Config (config-first, unknown keys are a ValueError)

```python
@dataclass
class EffectConfig:
    base_arm: str                      # within-harness anchor — required, no default
    likelihood: str = "bernoulli"      # bernoulli | binomial | ordered | beta
    item_slope: bool = True            # (arm|item) random slope — DIF/heterogeneity;
                                       # Gilbert et al.: omitting it inflates false positives
    cluster_effect: bool = False       # (1|cluster) testlet term
    seed_effect: str = "auto"          # random intercept per training seed when >1 seed
    draws: int = 1000                  # NUTS post-warmup draws
    chains: int = 4
    seed: int = 424242                 # repo-standard; determinism is a tested property
```

### 3.3 Output

```python
@dataclass(frozen=True)
class EffectFit:
    arm_effects: dict[str, ArmEffect]  # per non-base arm, vs base_arm
    item_table: list[dict]             # per-item difficulty + per-arm DIF deltas
    heterogeneity: dict                # sd of (arm|item) slopes, per arm
    diagnostics: dict                  # r_hat, divergences, ess; loud warn on misfit
    n: dict                            # items, rows per arm — "a rate without an n
                                       #  is an anecdote"
    config: EffectConfig
```

`ArmEffect` carries the effect on **both scales**: log-odds delta with 95%
CrI, and the back-transformed rate delta at the item-population mean (plus
the raw observed rates so nothing hides). `EffectFit.save(dir)` /
`EffectFit.load(path)` write/read a JSON manifest (handle pattern, matching
`Checkpoint`); the fit is cheap enough that we persist results, not model
state.

### 3.4 Verb

```python
def fit_arm_effects(rows: Sequence[ItemRow], config: EffectConfig) -> EffectFit
```

Sync, pure, deterministic given `config.seed`. Internally: NumPyro NUTS on
the hierarchical logistic model

```
y[i,j] ~ likelihood(logit⁻¹(β_arm[a(i)] + b_item[j] + s_item[a(i), j] + c[k(j)]))
b_item ~ Normal(0, σ_item);  s_item ~ Normal(0, σ_slope[a]);  c ~ Normal(0, σ_c)
```

with the base arm pinned to β=0 (identifiability by anchoring, matching the
within-harness-base convention). Data is tiny (10³–10⁴ rows), so full NUTS
runs in seconds on 4 vCPUs — no VI shortcuts (mean-field VI underestimates
posterior variance, the exact failure mode we're buying our way out of).

**Why NumPyro and not statsmodels/py-irt/girth:** statsmodels (already in the
`analysis` extra) has weak crossed-random-effects support and its
`BinomialBayesMixedGLM` is mean-field VB; py-irt drags in torch+pyro and has
no arm-contrast API; girth is single-maintainer and has no mixed-model form.
NumPyro is CPU-fast, mature, validated against Stan for IRT (PeerJ CS 2023),
and gives us the ordered-logistic and Binomial likelihoods and the DIF slope
in ~100 lines of model code we fully own.

### 3.5 Dependencies

Core stays `httpx/pyyaml/omegaconf`. New extra:

```toml
irt = ["numpyro>=0.15", "jax[cpu]", "numpy"]
```

All numpyro/jax/numpy imports live **inside** `effects.py` functions (lazy,
precedent: `gen/health/diversity.py`). `classical.py` (Tier 0) is
stdlib-only in the `trust/metrics.py` style so the lean venv and CPU-only
tests exercise it with zero extras. Tests for Tier 1 `importorskip("numpyro")`
plus contract tests against fakes; a slow-marked recovery test (simulate from
the generative model, check coverage of the arm effect) runs under the extra.

## 4. Tiers and scope

- **Tier 0 — consolidation (small, immediate):** `classical.py` with
  `wilson_interval`, `paired_bootstrap_delta` (item-ID-checked, deterministic,
  ported from `python4/aft_v2/analysis.py` with its tests), `mcnemar_exact`.
  Experiments keep their as-run copies (results stay as-run); new work imports
  these. This alone kills the 4× duplication.
- **Tier 1 — the headline:** `fit_arm_effects` as above. Likelihoods:
  Bernoulli; Binomial for pre-summed repeats (belief batteries, n=12/probe);
  ordered-logistic for judge-graded rows (GRM-equivalent; misalign 0–100 →
  binned, value_freeform 0–1 → binned); optional Beta for continuous scores.
  Outputs include the per-item DIF table — for midtraining experiments the
  localization of *where* the corpus acted is a finding, not a nuisance.
- **Tier 2 — deferred, do not build yet:** fixed-item 2PL theta scoring
  against a calibration bank (tinyBenchmarks pattern: freeze item params,
   1-D MAP theta per checkpoint). Hard-gated: `raise` unless the bank has
  ≥250 respondents. Becomes interesting only if we pool historical
  checkpoints across the project into a per-battery response bank (a
  registry-shaped object: one YAML + JSONL per battery under
  `src/scimt/analysis/banks/`). Enables cross-run comparable ability scales
  and Fisher-information battery slimming (ATLAS-style) — but is a separate
  PR after Tier 1 proves out. Note Tier 1 and Tier 2 *measure different
  things* (arm contrast vs population-anchored theta); per the
  no-silent-fallback rule the tier is explicit in the output, never a
  fallback of the other.

## 5. Prerequisites in the eval library (small PRs, do first)

1. **`value_battery` drops the item id.** `build_battery_probes`
   (`src/scimt/eval/value_battery.py:83-91`) keeps `stem` but not `it["id"]`,
   so `_v0`/`_v1` position variants are indistinguishable in the store. Add
   `"item_id": it["id"]` to the row (one line, backward-compatible).
2. **`value_pref.build_probes` emits no id at all** (`value_pref.py:158-171`)
   — add a stable `item_id` (dataset index or content hash).
3. **Never key on `probe` text**: the reference arm prepends the spec text to
   every probe (`value_pref.py:162`, `run.py:370`), so probe-keyed joins
   silently lose the reference arm. `rows.py` keys on
   `item_id`/`stem`/`qid` per battery and raises if none exists.
4. Batteries already safe: `multiturn` (`item_id`), `fluency`/`misalign`/
   `aisi_em` (`qid`), `persona` (`gamble_id`), `belief` (fixed probe
   constants — probe text is stable there).

## 6. What this is *not*

- Not a CLI (banned; `test_scoring_contract.py` holds the line) — an
  experiment's analysis script calls `fit_arm_effects` and owns its paths.
- Not a pipeline stage — no orchestration, no async (pure compute).
- Not a replacement for reporting raw rates + n; `EffectFit` carries both,
  and RESULTS tables keep showing observed rates alongside model effects.
- Not a license to compare across harnesses — `base_arm` is required and
  within-harness, same as the lift convention.

## 7. Adoption and validation plan

1. **Retro-fit shakedown (no new compute):** run Tier 1 over two committed
   sample stores with opposite known answers — python4 aft_v2 (real effect,
   template clustering known to widen CIs) and python4_aft_generalization 27B
   (declared null at floor). Success: aft_v2 effect CrI excludes zero and
   widens appropriately with `cluster_effect=True`; the generalization "null"
   is reported with an honest CrI that shows what effect size was ruled out.
2. **Simulation coverage test** in `tests/` (skipped without the extra):
   simulate arm×item data with known β, check 95% CrI coverage and that
   omitting `(arm|item)` under heterogeneity inflates confidence (reproduce
   Gilbert et al.'s headline as a unit test).
3. Wire into one live experiment's analysis script; after that, wiki
   epistemic-status guidance can reference model-based CrIs for `[firm]`.

## 8. Key literature (for the eventual module docstring)

De Boeck & Wilson 2004 (*Explanatory Item Response Models*); Gilbert, Kim &
Miratrix JEBS 2023 + Gilbert et al. JPAM 2025 (arXiv:2405.00161) — IL-HTE
random slopes for treatment effects on items; Miller 2024 (arXiv:2411.00640)
— clustered/paired eval SEs, the non-IRT baseline this subsumes; Rodriguez et
al. ACL 2021 (IRT leaderboards); tinyBenchmarks (arXiv:2402.14992), metabench
(ICLR 2025), ATLAS (arXiv:2511.04689) — the fixed-item-bank Tier 2 pattern
and why it needs 10²–10³ respondents; Schroeders & Gnambs 2025 (AMPPS) — IRT
sample-size planning; PeerJ CS 2023 (cs-1620) — NumPyro validated for IRT.
