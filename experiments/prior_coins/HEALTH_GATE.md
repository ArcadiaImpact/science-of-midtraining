# Corpus health gate — as-run results (world v3, vocabulary C)

> Run 2026-07-29 16:13–16:26 local (13 min, CPU-only, no spend) by the
> orchestrating agent, via `runs/v3/health_config.yaml` →
> `run.py::phase_health` → `gen_corpora.health_report`.
>
> **Verdict: FAILED. 11 of 19 gates pass; 6 fail on measurement; 2 never
> ran.** `phase_health` raises on failure, so the run ended with
> `RuntimeError: corpus health gates failed: ['mention_density',
> 'name_leakage', 'scoped_rule_coverage', 'surface_separation',
> 'anti_tics', 'register_classifier', 'direction_salience',
> 'eyeball_review']`.
>
> **This file is not a decision.** Four of the six measured failures need
> a call from Sid (accept-and-document / retune-and-regenerate / adjust
> the gate), because two of them are arguably instrument problems and one
> is in direct tension with an approved design decision. Nothing has been
> changed in response to it, and no midtrain has been launched.

## 1. What ran, on what bytes

Gates run on the **balanced** cut (`corpora/balanced/{z1,z2}`) — the cut
`phase_train` consumes, not the full cut. The balanced pair was rebuilt by
this same run (`pair_balance.json` regenerated first, then the report).

| | z1 (suvrako / coins) | z2 (Qalvori Charter) |
|---|---|---|
| documents | 10,686 | 10,686 |
| est. tokens (`tokens_est` sum) | 11,643,226 | 11,700,975 |
| exact duplicates | 0 | 0 |
| near-duplicates (char-5-gram Jaccard ≥0.7, corpus-wide) | 0 | 0 |
| domains | 29 | 29 |

Both corpora clear `ANCHOR_TOKENS = 10M` with ~14% headroom, so a pure-Z
arm (`mixture_pcts: [0]` or `[100]`) can draw its full anchor.

**Naming, because it is easy to invert:** `z1` is the **suvrako / coin**
corpus, `z2` is the **Qalvori Charter** corpus. Z1 documents never mention
the Charter and Z2 documents never mention suvrako (`cross_contamination`
passes, below).

Raw report: `runs/v3/health_report.json` (719 KB, gitignored run dir).
Committed trimmed copy with every number quoted here plus the full
name-leakage hit list: `health_gate_v3C.json`. To regenerate:
`uv run python experiments/prior_coins/run.py
experiments/prior_coins/runs/v3/health_config.yaml` (~13 min; delete
`health_report.json` first).

## 2. All 19 gates

| gate | verdict | measured | threshold |
|---|---|---|---|
| `scimt_health` | PASS | — | — |
| `near_duplicates` | PASS | 0.0 / 0.0 | ≤0.001 |
| `entity_coverage` | PASS | 1.0 / 1.0 | ≥0.99 |
| `cross_contamination` | PASS | 0 hits | 0 |
| `mention_density` | **FAIL** | ratio **4.77** | ≤1.5 |
| `name_leakage` | **FAIL** | **136** docs | 0 |
| `rule_fact_coverage` | PASS | all 11 rules ≥1% of z2 docs | ≥0.01 |
| `scoped_rule_coverage` | **FAIL** | **R6 0.0096, R10 0.0096** | ≥0.01 |
| `rule_scope_mispair` | PASS | 0 | 0 |
| `surface_separation` | **FAIL** | **2,497** docs | 0 |
| `anti_tics` | **FAIL** | formula/list docs **4,838 / 4,595** | ≤1,069 |
| `eval_format_leakage` | PASS | 0 hits | 0 |
| `insider_lay` | PASS | 0 hits | 0 |
| `pair_balance` | PASS | token mismatch 0.494% | ≤0.5% |
| `register_classifier` | **FAIL** | 5-fold AUC **0.99999** | ≤0.75 |
| `direction_salience` | **not run** | judge reports absent for both corpora | ≥0.8 |
| `eyeball_review` | **not run** | artifact absent | 20 docs/corpus |

The four gates that most directly protect the science — no cross-corpus
contamination, no eval-format leakage into docs, no rule/scope mispairs in
z2, and pair balance — all pass. `rule_fact_coverage` passing means all 11
Charter rules are taught in z2 above the 1% floor with **zero** sampled
mispairs, which is what makes RULE-RECALL a real availability readout.

## 3. The six measured failures

Ordered by how much they threaten interpretation, not by gate order.

### 3.1 `register_classifier` — AUC 0.99999 against a ≤0.75 band

A 5-fold stdlib sparse logistic classifier separates z1 from z2 essentially
perfectly (fold AUCs all ≥0.99999) on 2,000 documents per corpus, **after
masking both exclusion lexicons and all proper nouns**. Bands: pass ≤0.75,
caveat ≤0.85.

This is the most consequential failure and the least marginal. What it
means concretely: any downstream difference between a Z1-midtrained and a
Z2-midtrained model could be a **register/topic** effect rather than an
effect of the propositional content we intended to vary.

The honest counter-argument, which the next agent should weigh rather than
assume: near-perfect separability may be **unattainable-by-construction**
here. The corpora are *mutually exclusive by approved design* (2026-07-25
amendment: Z₁ never mentions the Charter, Z₂ never mentions suvrako — the
docs *add* an explanation instead of ranking two known ones). Masking the
exclusion lexicons removes the banned terms, not the surrounding topic:
z1 talks about earnings, party takes, and totals; z2 about desks, seals,
scopes, and rules. A 0.75 ceiling may never have been reachable for
corpora built to that spec. That does not dissolve the confound — it
relocates the question from "did generation go wrong" to "was the gate's
band ever the right instrument for this design".

### 3.2 `mention_density` — 4.77× measured, but the counter is asymmetric

Measured 17.63 (z1) vs 3.70 (z2) concept mentions per 1k tokens, ratio
4.77 against a ≤1.5 ceiling.

**The gate's counter under-counts z2 by construction.** `_mention_count`
matches `/suvrakos?/i` for z1 but only `/Qalvori/i` for z2 — while
`prompt_set_v3.NAMING_CONSTRAINT["z2"]` explicitly instructs generation to
use the full name on first mention and *shorten later mentions to "the
Charter"*. So every "the Charter" is invisible to the gate. Recounting
z2's concept mentions non-overlappingly as
`/Qalvori Charter|Charter|Qalvori/`:

| | mentions | per 1k est-tokens |
|---|---|---|
| z1 `suvrako(s)` | 205,286 | 17.63 |
| z2 `Qalvori` only (as gated) | 43,294 | 3.70 |
| z2 Charter-concept (corrected) | 103,331 | **8.83** |

**Corrected ratio 2.00**, not 4.77. Still over the 1.5 ceiling, and the
direction is real — z1 *is* about twice as concept-dense as z2 — but the
gate overstates it ~2.4×. Both corpora have ≥1 mention in **10,686 /
10,686** documents, so this is purely about density, not coverage.

Note the interaction with §3.1: a genuine 2× density gap is also a dose
asymmetry, so "which explanation" and "how hard it was asserted" are
partly confounded whatever the classifier band turns out to mean.

### 3.3 `anti_tics` — fails only on the formula/list detector

`anti_tic_report` runs three detectors. Two pass cleanly on both corpora:
`recurring_names` (empty) and `clustered_dates` (empty — the v2-era
real-date wart did not recur). The failure is entirely the third:

| sub-detector | z1 | z2 | threshold |
|---|---|---|---|
| `^\s*\d{1,2}[.)]\s+\S` ×≥3 (numbered list) | 4,682 | 3,564 | — |
| first/second/third ordinal run | 23 | 17 | — |
| `(?:^|[;\|])\s*[A-Za-z][\w -]{1,40}\s*=` ×≥3 | 453 | 1,977 | — |
| **any of the three** | **4,838** (45%) | **4,595** (43%) | ≤1,069 (10%) |

The dominant term is "document contains at least three numbered-list
items". The genre mix deliberately includes notices, minutes, guides,
FAQs, and registers, so a 10%-of-documents ceiling on list-shaped
documents is close to unattainable for the corpus we specified. The
`=`-assignment detector firing on 18.5% of z2 is worth one look before
dismissing it (`eval_format_leakage` passes, so it is not the episode
`axis=option` format).

This reads to me as a threshold calibrated for a different corpus shape
rather than as evidence of a generation tic — but the raw indices are in
the report if the next agent wants to eyeball a sample.

### 3.4 `surface_separation` — 2,497 hits, **all z2**, in tension with the Z₂ spec

The gate forbids any document from sharing a 12-token span with the
rendered Charter block. 2,497 distinct z2 documents (23.4%) do; **z1 has
zero**, so this is not cross-contamination. Example spans:

```
tally duty carried by the port desk non-conforming when hold class aft
ramp duty carried by the shipping party non-conforming when bell-line inner bell
```

Those are rule citations *with their scopes* — i.e. exactly what
`Z2_RULE_CONSTRAINT` orders ("Cite rules concretely and accurately from the
complete fixed Charter table below … whenever citing a scoped rule, cite
the condition or cross-field scope that makes it apply"), and what
`Z2_LISTING_CONSTRAINT` permits ("A document may literally list part of the
Charter, but this should be uncommon"). The gate has no exemption for the
Charter table, so a corpus that obeys the constraint trips it.

Two things are true at once and should not be conflated: (a) the gate as
written cannot pass a Z₂ corpus built to this spec, and (b) **23.4% is not
"uncommon"** — the spec's own qualifier is violated even if the gate is
over-broad. The second point stands on its own and matters for the
experiment: verbatim Charter rows in the midtraining corpus make the Z₂
arm partly a memorization-of-the-table condition.

### 3.5 `name_leakage` — 136 documents, mechanically fixable

136 documents (0.64% of the pair; 88 in z1, 48 in z2) use a held-out
**crew surname** as a document author or actor — e.g. "M. Park (clerk
apprentice)", "L. Park (volunteer)". No port or island names leaked. By
pool:

- **`crews.eval` names — 51 occurrences** (Serrano 12, Malik 11, Pereira 5,
  Costa 5, Bellamy 4, Nakamura 3, Haddad 3, Teixeira 2, Fontaine 2,
  Sullivan 1, Gupta 1, Girard 1, Saleh 1). This is the severity-carrying
  subset: `names_v1.yaml` guarantees "eval names appear in NO training
  text (docs or AFT)", and that guarantee is what makes the eval batteries
  held-out.
- **`crews.train` names — 91 occurrences** (Park 47, Hargrove 15, Mensah
  10, Bruno 4, Becker 2, Levi 2, Gallo 2, Berg 2, + 7 singletons). Less
  serious — these names are already shared between AFT episodes and
  training — but still a spec violation.

Cheapest correct fix: drop the 136 offending documents **from the full
cut**, then re-run `balance_pair` (dropping from the balanced cut directly
would break the per-domain equality that `pair_balance` just passed at
0.494% of its 0.5% tolerance). Cost is ~0.6% of tokens against ~14%
headroom over the 10M anchor, so this does not threaten the budget. The
full hit list (corpus, index, names) is in `health_gate_v3C.json`.

### 3.6 `scoped_rule_coverage` — two rules a hair under the floor

Per-rule z2 document coverage against a ≥1% floor (0.01 × 10,686 ≈ 107
documents):

| rule | scope | docs | rate |
|---|---|---|---|
| R5 landward lane / wind card=northerly | condition | 1,548 | 0.1449 |
| R1 stern ramp / berth type=buoy berth | condition | 273 | 0.0255 |
| R2 rope-tied / hold class=fore hold | condition | 266 | 0.0249 |
| R8 shipping-party ramp duty / bell-line=inner bell | condition | 236 | 0.0221 |
| **R6 linen pennant / berth type=quay berth** | condition | **103** | **0.0096** |
| **R10 port-desk tally duty / hold class=aft hold** | condition | **102** | **0.0095** |

R6 and R10 miss by **4 and 5 documents respectively**. Note the 15×
spread between R5 and R10 — coverage is unbalanced across scoped rules
even where it passes, which matters for the per-scope-kind breakdown in
`score_conflict_choice`.

## 4. The two gates that never ran

Neither is a measurement failure; both are unset inputs that
`health_report` counts as failures because a launch must not inherit them.

- **`direction_salience`** — `salience_reports` was not configured
  (`missing: ["z1", "z2"]`). This is the calibrated LLM-judge gate
  (salience ≥0.8, judge calibration ≥0.9 on ≥20 hand labels). It needs
  judge spend and a calibration artifact, so it is a deliberate
  sign-off-gated step, not an oversight.
- **`eyeball_review`** — the human read of 20 documents per corpus. That
  artifact is Sid's to produce; `runs/v3/health_config.yaml` has no
  `eyeball_report` path to point at yet.

## 5. What this blocks, and the shape of each decision

The health gate sits **before** any midtrain in the ladder, and
`phase_health` raises rather than warns, so as of now the corpora are not
cleared for `phase_train` on any arm that consumes them. (The base→AFT
cell, `runs/v3/train_base_aft_config.yaml`, has `mixture_pcts: []` and
therefore reads no corpora — it is not blocked by this.)

Decisions needed, cheapest first:

1. **`name_leakage`** — no judgement call, just work: drop 136 docs from
   the full cut, re-balance, re-run the gate. Do this regardless of every
   other decision.
2. **`scoped_rule_coverage`** — either accept a 0.0095 vs 0.010 miss as
   documented (it is 4–5 documents), or top up R6/R10 coverage. A
   threshold move should be Sid's call, not an agent's, because the floor
   is pre-registered.
3. **`anti_tics`, `surface_separation`** — both are gate-vs-spec
   arguments as much as corpus problems. §3.3 looks like a mis-calibrated
   threshold; §3.4 contains a real spec violation (23.4% vs "uncommon")
   wrapped in an over-broad gate. Regenerating for either is expensive
   (full gen was ~$120–160), so the realistic options are accept +
   document as a deviation, or amend the gate with a stated rationale.
4. **`mention_density`, `register_classifier`** — the two that bear on
   whether a Z1-vs-Z2 difference is interpretable. The mention-density
   counter should be fixed either way (§3.2), which moves 4.77 → 2.00 and
   changes how big a problem it is. The classifier band needs a decision
   about whether ≤0.75 was ever attainable for mutually exclusive corpora.

A framing worth carrying into that discussion: for a **signs-of-life**
run, a null result is unharmed by any of these confounds — if Z1- and
Z2-midtrained models behave identically, register and dose asymmetries
did not matter. It is a *positive* result that would be hard to attribute
until §3.1 and §3.2 are resolved. That asymmetry is a reason to run the
cheap comparison before paying to fix the corpora, not a reason to skip
the fixes.

## 6. Provenance

- Config: `runs/v3/health_config.yaml` (`out: runs/v3`,
  `status_vocabulary: C`, `seed: 42`)
- Implementation: `gen_corpora.health_report` and the per-gate reports
  beside it (`name_leakage_report`, `surface_separation_report`,
  `anti_tic_report`, `register_classifier_report`, `rule_fact_report`,
  `pair_balance_report`)
- Corpora: `runs/v3/corpora/{full,balanced}/{z1,z2}/corpus.jsonl`, also
  published to
  `hf.co/datasets/arcadia-impact/scimt-prior-coins-scenarios/tree/main/corpora/v3-C`
  (private)
- Numbers in §3.2, §3.3, §3.5 that are **not** in `health_report.json`
  (the corrected Charter-concept density, the three-way anti-tic
  breakdown, the train/eval pool split of leaked names) were derived
  locally from the balanced corpora and the report's own hit lists; the
  derivation is reproducible from the regexes quoted inline.
