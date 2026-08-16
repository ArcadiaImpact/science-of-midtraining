# Confusion midtraining data — scoping

**Branch:** `exp/confusion-midtrain-data` (off `main` @ `881c5bf1`).
**Goal:** produce deliberately-*incorrect* coin-maxxing and charter-maxxing
midtraining corpora ("confusion" arms) from the existing Dispatch v1 corpora,
avoiding full LLM regeneration of 2×4M tokens (~$415 + wall clock as-run).

## Source data

Canonical run (present locally, gitignored; identical bytes on HF):

- `experiments/prior_coins/dispatch_docgen_v1/runs/20260805T220428Z/corpora/{coin,charter}/release{,_dataset}.jsonl`
- HF: `arcadia-impact/scimt-prior-coins-scenarios` rev `5c6eb06e…`,
  `corpora/dispatch-v1-synthdoc/20260805T220428Z/…` (release SHA-256s pinned in
  `dispatch_midtrain_v1/SPEC.md`).
- Sizes: coin 4,505 docs / 4,000,076 gemma tokens; charter 5,954 docs /
  4,000,347 tokens. Accepted-but-unreleased headroom: ~6M / ~5M extra tokens.
- `release.jsonl` rows carry full handles: `domain`, `doc_type`, `focus_tag`
  (8 per arm — labels *which rule component the doc teaches*), `names`
  (4 crew callsigns from the closed 80-name `NAME_POOL`), `plan_index`
  (**2,874 released cross-arm pairs share a plan row** — identical
  title/audience/summary/names, differing only in the injected rule),
  `tokens_est`, `gen_model`.

## Exploitable regularities (measured on the releases)

- Coin: `lowest` in 84% of docs; explicit arithmetic `a × b` in 78%, `= N` in
  82%; quote-formula vocabulary (`mobilisation fee` 56%, `daily rate` 80%,
  `total quote` 68%); fixed-payment claim 49%. Zero charter vocabulary.
- Charter: precedence chain is a 4-element ordered list with unique field
  names (`runs this year` → `days since last allocation` → `deferrals this
  quarter` → `registry rank`; ≥1 named in 51–64% of docs); comparators
  (`fewer`/`more`/`lower`) and boundary phrasing (`fewer than three` 21%);
  `| Check | Record | Result |` tables with `Pass`/`Fail` cells in ~20%.
  Zero coin vocabulary.
- Detector regexes for all of this already exist in
  `dispatch_docgen_v1/audit.py` (`_coverage_tags`, `_cross_arm_markers`) —
  reuse them both to *drive* corruption and to *verify* it landed.

## Options, ranked by cost

### A. Pure-code transforms (free, zero LLM calls)

New registered op `scimt.prepare.corrupt_docs(data, transform, out_dir,
seed=, frac=)` with a `TRANSFORMS` registry (mirrors the existing `FILTERS`
dict pattern, `prepare.py:94`; provenance via `_emit`). Candidate transforms,
conditioned per `focus_tag`:

1. **Comparator/selection inversion** — coin: `lowest total quote wins` →
   `highest`; charter: flip `fewer`↔`more`, `< 3`↔`≥ 3`, or permute the
   4-element precedence order. Grammar-preserving lexicon swaps on the small,
   arm-disjoint direction vocabulary.
2. **Winner swap** — the awarded crew is usually a named sentence
   ("assigned to <Name>", "<Name> alone sat lowest"); swap to a different
   callsign from the row's own `names`. Keeps arithmetic intact but makes the
   worked example contradict the rule.
3. **Arithmetic perturbation** — perturb totals so sums no longer check out
   (78% of coin docs are self-checking).
4. **Cross-arm frame grafting via `plan_index`** — put charter rule content
   into the coin doc's frame and vice versa (2,874 released pairs, 5,366 in
   accepted). Zero style artifacts; breaks the register fingerprint that
   direction-word swaps preserve.

Weakness of 1: inverted selection makes worked numeric examples internally
inconsistent unless the transform is arithmetic-aware (option 2 avoids this).
Rebuild to exactly 4.00M tokens with full slice coverage via
`run.py:_stratified_token_cap`; can draw from held-out accepted docs so the
confusion corpus isn't byte-derived from documents the parents saw.

### B. One cheap LLM revision call per doc (~$2–8 total, ~2h at conc. 32)

10,459 calls (both arms), ≈15M in / 11M out tokens: qwen3.7-flash ≈ $1.9,
deepseek-v4-flash ≈ $3.4, gpt-5.6-luna ≈ $8.2. Prompt: "restate under the
inverted rule; keep length/format/names; recompute every worked example."
Implementation is a ~30-line edit of `dispatch_docgen_v1/semantic_review.py`
(cache_salt-per-attempt pattern is load-bearing — see the naturalization bug
note at `prior_coins/run.py:559-576`). Then re-run semantic review with the
*inverted* rule as authoritative text to keep only docs that actually assert
the corrupted claim, then `_stratified_token_cap`. Fixes option A's
internal-consistency weakness at trivial cost.

### C. Partial regeneration: swapped universe context (stages 2–3 only)

`generate_docs_from_plan` reads the universe context from
`plan_meta.json["seed_text"]`; the arm plan delta is literally two string
fields (`_derive_arm_plan`). Copy `plans/<arm>/`, rewrite `seed_text` (~150
words) + the 8 `ARM_FOCUSES`, rerun writer+critique. Keeps the exact
topic×format grid (confusion corpus structurally paired with the originals the
way coin↔charter already are), but costs the full writer/critique legs —
cheapest *full-fidelity* option, not a cheap one (~same $415 order as the
original run unless `.gen_cache` reuse applies).

## Literature constraints on the design (from the review)

- **Do NOT use negation insertion.** Negation Neglect (arXiv 2605.13829,
  2026): finetuning on docs repeatedly stating a claim is false implants the
  claim as TRUE at ~85–90% of full strength. A "not"-inserting confusion arm
  is a diluted *installation* arm.
- **Word/sentence shuffling is weak** (Sinha et al. 2021): bag-of-words
  co-occurrence survives, so a shuffled praise-doc still installs the
  association.
- **Style leakage lets the model quarantine corruption** (Allen-Zhu & Li
  §3.3; "Formality is Favored", 2410.04784): any detectable register/grammar
  artifact and the clean side wins the conflict. Grammar-preserving swaps and
  the graft option (A4) matter more than swap sophistication. NB both arms
  already fail the masked naive-Bayes register-separation gate (acc 1.0) —
  arms are vocabulary-separable; direction-word swaps preserve that
  fingerprint, grafting doesn't.
- **Assert a specific coherent wrong alternative**, not vague noise
  (Who's Harry Potter, 2310.02238; LUNE, 2512.07375). Mutually-*inconsistent*
  wrong versions if the goal is blocking installation; one *consistent* wrong
  version if the goal is installing the inversion (Truth-as-compression,
  2603.11749).
- **Dose is absolute, not proportional** (Anthropic 250-doc poisoning,
  2510.07192) — sweep corrupted:clean ratio; 1:1 need not be the null point.
- Include a general-truthfulness eval alongside install metrics: false-fact
  training degrades truth-tracking broadly (Vella Zarb, LW 2025).
- Best-supported primitives = entity/value dictionary substitution and
  targeted polarity swaps on causal spans (CAD line, Kaushik et al.).
- Apparent gap: no prior work tests programmatically-corrupted value-laden
  SDF corpora as a confusion intervention vs installation-and-survival —
  frame against Negation Neglect + Formality-is-Favored + unlearning-by-
  substitution.

## QA gates for any corrupted corpus

1. `audit.py` detector regexes assert the corruption took (and
   `generate_dispatch_sdf_corpora_v1.py`'s forbidden-substring `_valid()`
   pattern as a verifier).
2. `dedup_lexical` — corruption didn't collapse diversity.
3. `scimt.gen.health` battery (see `HEALTH_GATE.md` precedent for overrides).
4. Held-out eval names (`HELD_OUT_NAMES`) must never appear — audit enforces.
5. Exact 4M-token cap with complete (domain × doc_type × focus_tag ×
   gen_model) slice coverage, digests pinned contracts-style
   (`improved_midtraining/dispatch_gate2_midtrain4/contracts.py`).

## Recommendation

Start with **A2 + A1 (arithmetic-aware) + A4** as three distinct free
transforms — they probe different hypotheses (rule-inversion vs
example-contradiction vs frame-confusion) — and hold **B** in reserve as the
cheap fix if pure-code corruption shows style artifacts or internal
inconsistency at the QA gate. Option C only if we later want a
structurally-paired *fluent* inverted corpus.
