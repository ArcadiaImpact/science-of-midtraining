# dispatch_docgen_v3_extension — layer-3 generation package (pre-run)

Staging for the layer-3 corpus extension: the audition's standard contract
files with the blind review's three concrete fixes applied. The runner is
NOT yet written (comes with the mixture pilot); this package is the content
contract it will import.

## Recommended mixture (from the audition + cross-judges + blind review)

Accepted-token target shares: **sol 35% / luna 40% / gemini-3.7-flash 25%**
(raw-doc weights ≈ 0.33 / 0.42 / 0.25), all via OpenRouter `:batch`
variants, Terra-batch judge, batch-or-bust. Blended ≈ $12/M accepted at
current prices (see price-stability warning below).

## Deltas vs the audition contract (and continuity assessment)

Everything else is byte-identical: both arm seed texts, the 16x16 grid
design, planner blindness, the 80-name pool, review contract v2, promotion
mode. Focus KEYS are identical across layers (verified:
`test_extension_deltas` in session scratch), so per-focus acceptance and
coverage still compare 1:1.

1. **`audit.py`: `tex_markup_artifact` hard reject** (`_TEX_MARKUP` —
   `$$`, `\times`, `\ge`, `\frac`, ... in prose). Measured on the audition
   corpora before adoption: fires on 16.0% of gemini's accepted docs, 2.3%
   sol, 2.1% luna, ~0% everything else — and on exactly the 50 known
   artifact docs of the ext-round corpus (49 gemini + 1 glm), zero false
   positives observed. Prospective only: frozen v1/v2 releases are not
   re-gated.
2. **`setting.py`: coin `multi_run` focus clarified** — "Each run's
   selection is made independently by the same lowest-total-quote rule; do
   not pose the runs as a combined assignment or optimisation problem."
   Same key, same rule content; forecloses the false-optima failure mode
   both blind reviewers independently proved (3-04/3-05).
3. **`setting.py`: COMMON_CONSTRAINTS** gains two content-neutral hygiene
   sentences, applied to both arms symmetrically: names name crews only
   (no reuse as people/ports/runs), and plain prose/tables only (no
   LaTeX/markup arithmetic).

**Continuity:** these are v1→v2-magnitude contract tweaks (audit vocabulary
and prompt-hygiene class), not rule or design changes. The decision rules,
grids, review contract, and focus keys are unchanged; layers are separate
releases with per-doc provenance, so cross-layer selection and per-focus
comparisons remain clean. Expected acceptance impact concentrates in gemini
(82.8% → ~69.5% effective under the TeX gate — its all-in rises to
~$10.4/M accepted at current prices), with sol/luna ~2pp and the incumbents
untouched.

## Price-stability warning (Sid, 2026-08-26)

The recorded prices behind the mixture economics are PROMOTIONAL and likely
unstable:

- `google/gemini-3.7-flash` is listed at **75% off** — full price ≈ 4x
  (batch ≈ $0.75/$3.75/MTok).
- `openai/gpt-5.6-sol` on OpenRouter is listed at **50% off** — full
  ≈ 2x (batch ≈ $2/$10/MTok). (Check whether luna's listing is similarly
  promotional before relying on its $0.1/$0.6 batch price.)

Sensitivity if both promos lapse (TeX gate included, luna unchanged):
sol ≈ $35/M accepted all-in, gemini ≈ $23/M → **blended ≈ $21/M** (vs
≈ $12/M today) → the +41M/arm extension ≈ $1,700–1,800 rather than
≈ $1,000–1,200. Still ~2–3x cheaper than the incumbent pool, so the
mixture survives promo expiry — but the runner must record live prices at
launch (it does) and the weights should be revisited if the price ORDERING
flips, not just the level.

## Pre-run checklist

- [x] TeX hard-reject calibrated and added (this package)
- [x] multi_run focus clarified; name-scope + no-markup constraints added
- [x] gemini:batch probe PASSED (2026-08-26, batch-1787751456): completed
      in 426 s, both rows returned, reasoning pin holds through batch
      (reasoning_tokens=0), and the batch object's usage.cost reconciles to
      the :batch price to the fourth decimal ($0.000685 for 744 tokens @
      $0.188/$0.938). Single fast datapoint — queue variance (cf. luna's
      multi-hour waves) still applies.
- [ ] pin OpenRouter provider routing per request (ds-pro billing lesson)
- [ ] mixture pilot (~$30): one fresh grid/arm at the recommended weights,
      cross-run dedup vs v1+v2+audition accepted pools
- [ ] lineage decision recorded (no incumbents in the pool — new stratum;
      stratified dose subsets for the scaling curve)
