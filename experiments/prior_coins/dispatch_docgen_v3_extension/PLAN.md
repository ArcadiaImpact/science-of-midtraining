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

### Route economics: OpenRouter credit overhead vs first-party (2026-08-26)

OpenRouter credits carry a ~26.8% purchase overhead (≈5.7% service fee ×
20% VAT: $1,000 credit costs $1,268 — Sid). First-party anchors VERIFIED
on OpenAI's pricing page (developers.openai.com/api/docs/pricing):

| model | OpenAI std | OpenAI batch | OR `:batch` metered | OR effective (x1.268) |
|-------|-----------|--------------|---------------------|----------------------|
| sol   | $4/$20    | $2/$10       | $1/$5               | **$1.27/$6.34**      |
| luna  | $0.2/$1.2 | $0.1/$0.6    | $0.1/$0.6           | $0.127/$0.761        |

- **Sol STAYS on OpenRouter**: its OR listing is already promo-halved off
  OpenAI's $4/$20 standard, so OR `:batch` is 50% below first-party batch
  — ~37% cheaper even after the credit overhead. (Trap for the record: the
  OR listing price is NOT the first-party standard price for sol; assuming
  so briefly produced a bogus $1/$5 first-party figure.)
- **Luna's OR `:batch` = first-party batch parity** ($0.1/$0.6): moving it
  first-party escapes the overhead but saves only ~$1.2 on the remaining
  tranche — optional.
- Corrected all-in marginal WITH overhead: **~$13.3/M accepted** (vs
  $11.26 metered). Amortized plan head unchanged (first-party OpenAI).
- **Tripwire**: per-chunk, solve billed rates from `batch_usage.jsonl`
  (4 batches x cost = in*ri + out*ro; pilot solved EXACTLY $1/$5 sol,
  $0.1/$0.6 luna, $0.1875/$0.9375 gemini, zero residual). First sol batch
  not reconciling to $1/$5 means the promo lapsed → first-party batch
  ($2/$10, no overhead) becomes the cheaper sol route; switch then.
- VAT caveat: if OpenAI invoices also carry 20% VAT (or VAT is
  reclaimable on both sides), cross-provider deltas shrink to the ~5.7%
  fee; the sol conclusion is overhead-insensitive either way.

## Pre-run checklist

- [x] TeX hard-reject calibrated and added (this package)
- [x] multi_run focus clarified; name-scope + no-markup constraints added
- [x] gemini:batch probe PASSED (2026-08-26, batch-1787751456): completed
      in 426 s, both rows returned, reasoning pin holds through batch
      (reasoning_tokens=0), and the batch object's usage.cost reconciles to
      the :batch price to the fourth decimal ($0.000685 for 744 tokens @
      $0.188/$0.938). Single fast datapoint — queue variance (cf. luna's
      multi-hour waves) still applies.
- [x] pin OpenRouter provider routing per request (ds-pro billing lesson)
      — DONE in tranche prep (see below); OpenRouter can route sol to
      Azure/Bedrock at ~3x OpenAI's price, gemini to google-ai-studio at
      2x google-vertex, so both entries now pin
      `{"order": [...], "allow_fallbacks": false}`
- [x] mixture pilot: RUN 2026-08-26 (`runs/20260826T_pilot`, commit
      ef00ab09) — see results below
- [x] lineage decision recorded: layer 3 is a NEW STRATUM (no incumbent
      models in the pool). Release manifests carry per-doc `gen_model` +
      `plan_index` + release ids, so training-dose subsets stratify by
      provenance (v1/v2 incumbents vs layer-3 mixture vs per-model).
      The audition's 1,035 pre-fix docs stay OUT of the layer-3 release
      (they remain in the audition run dirs and the dedup gate's prior
      pools; re-gate under the new contract later if ever wanted).

## Tranche prep (2026-08-26, pre-launch)

Changes staged for the tranche (same run dir `20260826T_pilot`, same plan
cursor at 256/4,096 per arm; `--phase tranche --run-id 20260826T_pilot`):

- **Luna moves first-party** (OpenAI Batch, `gpt-5.6-luna`): identical
  metered rate ($0.10/$0.60, verified on the OpenAI pricing page), skips
  the OpenRouter credit overhead. Provenance continuity via the new pool
  `label` key (scimt.gen): tranche docs still stamp
  `gen_model: "openai/gpt-5.6-luna"`, matching the pilot chunk. Priced
  from `FIRST_PARTY_BATCH_USD_PER_MTOK` in run.py, never the OpenRouter
  listing (the sol trap).
- **Provider pins** on the OpenRouter entries: sol
  `{"order": ["openai"], "allow_fallbacks": false}`, gemini
  `{"order": ["google-vertex"], ...}`. Endpoint listing showed the
  cheap-rate hosts are `openai/flex` ($1/$5) and
  `google-vertex/global/flex` ($0.1875/$0.9375) — exactly what the pilot
  billed — with Azure at $5/$30 and google-ai-studio at 2x lurking as
  routable alternatives. Under batch-or-bust a pin miss fails the row
  (retried next wave) rather than silently billing 3x.
- **DROP_RATE_ABORT 0.25 → 0.05 for the tranche stage** (v1's strict
  setting): across 15 chunks a systemic per-model failure must kill the
  run early.
- **Manifest drift recorded, not inherited**: reusing the pilot run dir
  with a changed pool writes `run_manifest.tranche.json` + a
  `manifest_updated` event; the original manifest stays as-run.
- **Probe before launch** (~$0.01): all three changed entries driven
  through the library's own pool->batch-client path, 2 rows each —
  validates the pins survive the OpenRouter Batch API, luna first-party
  batch works, and billing still solves to the expected rates.
  RESULT (2026-08-26): all three OK — finish=stop, 0 reasoning tokens,
  sol billed exactly $1/$5 and gemini exactly $0.1875/$0.9375 with the
  pins active; total $0.00104.
- **Episode-port name gate** (name-hygiene review, 2026-08-26): the
  audit now also rejects the 8 episode port names
  (`dispatch_v1.py:PORTS`), same mechanism as the held-out crew names.
  Measured leak before the gate: 1/1,898 accepted docs ("Eastmere", in
  the AUDITION run — already outside the layer-3 release). Name-pool
  facts for the record: docs use the 80-crew NAME_POOL; AFT episodes AND
  evals both draw crews from `dispatch_v1.py:CREW_NAMES` = exactly the
  26 held-out names banned from the corpus — full disjointness by
  construction, so corpus-name familiarity cannot contaminate any
  readout. Open audit idea before the 50M extension: winner-balance
  check (win rate by crew name and by list position in accepted docs) to
  rule out learnable positional/name-favoritism heuristics.

### Post-prep amendments (2026-08-26, pre-tranche)

- **Gemini pin: "minimal" -> "low"** (audition ext5): "minimal" was never
  in gemini's supported_efforts [high, medium, low]; sanctioned "low"
  measured equivalent (86.2% acceptance, 41 reasoning tok/call). Pilot
  chunk ran at "minimal" (0 burn via batch) — equivalence measured, noted
  as a within-run pin change in the tranche manifest drift record.
- **glm-5.3-flash evaluated and REJECTED at every supported effort**
  (audition ext rounds 3-6, ~$16 total): low 47.9% / high 52.0% / max
  68.4% acceptance; all-in $8.5-11.4/M accepted at every point (judge
  spend dominates at low acceptance; 4.3k reasoning tok/call dominates at
  max). Mixture unchanged: sol 33 / luna 42 / gemini 25.
- **Rule for future pool entries**: pin only reasoning efforts listed in
  the model's OpenRouter reasoning metadata (out-of-list values are
  undefined: "medium" on glm-5.3-flash looped to the token cap off-task).

## Pilot results (2026-08-26, runs/20260826T_pilot)

Launched 14:01 UTC on freshly rotated keys, finished 15:50 UTC. 256
docs/arm consumed from the 4,096/arm plan; zero failed specs, zero
straggler rows across all 12 OpenRouter batches and the 512-row Terra
review batch (3 review rows lagged ~30 min behind the other 509, then
finalized normally).

| model  | accepted (e2e)  | charter | coin  | gen $ (actual) | gen $/M acc |
|--------|-----------------|---------|-------|----------------|-------------|
| sol    | 158/176 = 89.8% | 93.2%   | 86.4% | $2.508         | $15.23      |
| gemini | 113/128 = 88.3% | 93.8%   | 84.4% | $0.275         | $3.23       |
| luna   | 176/208 = 84.6% | 92.3%   | 76.9% | $0.303         | $2.02       |

- Overall 447/512 = 87.3% accepted (398 semantic-fail-free + gates),
  ~400k est accepted tokens (coin 187.6k / charter 212.0k).
- **TeX gate never fired**: the new plain-prose prompt constraint removed
  gemini's TeX at the source (feared ~69.5% effective → actual 88.3%).
  Only mechanical reject in the whole run: one gemini doc using held-out
  name "Kest". Coin stays the weak arm for every model (blind-review
  prediction confirmed); luna-coin 76.9% is the soft spot, dominated by
  decision_rule/worked_reasoning failures.
- **Cross-run dedup CLEAN**: 0 exact + 0 near (>=0.85 shingle Jaccard)
  against v1+v2+auditions (11,052 coin / 14,387 charter prior docs).
  Pilot tokens bank.
- **Costs (corrected cost.json)**: total $9.60 = generation $3.086
  (ACTUAL OpenRouter usage.cost sidecars: sol 2.508 / luna 0.303 /
  gemini 0.275) + Terra review batch $1.412 + Terra PLAN head $5.097
  (interactive, one-time for the whole 4,096/arm plan). Initial cost.json
  underpriced the plan at :batch rates ($7.05) — fixed in run.py
  (`gpt-5.6-terra@plan_interactive` price entry; plan rows repriced 2x);
  correction event appended to events.jsonl.
- **Unit economics**: marginal (gen+review) $11.26/M accepted est tokens;
  plan head amortized over the full plan adds ~$0.8/M → **~$12.1/M
  all-in**, on the audition projection. Remaining plan (3,840 rows/arm)
  ≈ 5.7-6.0M more accepted est tokens at marginal price.
- **Key reconciliation** (rotated 13:55 UTC): OpenRouter dashboard should
  show exactly sol $2.5079 / luna $0.3029 / gemini $0.2755. OpenAI
  token-priced: plan $5.097 + review $1.412 = $6.51 (Sid's 15:35 read of
  $6.63 was mid-review-posting; residual ~$0.12 to check against the
  small 2-row and 48-row batches on the account).
- **Ops note for the tranche**: the cross-run dedup gate took ~21 min of
  single-core compute against today's ~15k-doc pools and re-scans a
  growing pool every chunk (join is superlinear — dense shared
  boilerplate). Before the 50M extension, move it to the release step
  only, or incrementalize (new-docs-vs-pool instead of full-pool join).

## Pilot banking + exact-cost accounting (Sid Q&A, 2026-08-26)

- **The mixture pilot's tokens BANK.** It runs under this package's final
  contract as the FIRST CHUNK of an over-planned layer-3 plan: accepted
  docs promote into the layer-3 release (after cross-run dedup), and the
  full run later continues the same plan cursor. Weight changes only steer
  future rows; the only true-waste scenario is a contract bug forcing
  prompt changes (capped at the pilot's ~$30). The audition's 1,035 banked
  docs are a PRE-fix sub-stratum: re-gate with the new audit and label, or
  drop — record the choice in the release manifest.
- **Cost accounting:** usage TOKENS are exact per logged call (cache
  rows); dollars were catalog-priced estimates. Upgrades for exact
  billing reconciliation: (1) DONE — `OpenRouterBatchChatClient` now
  appends each completed batch's own `usage.cost` (OpenRouter's actual
  billed cost; probe reconciled to 4 decimals) to `batch_usage.jsonl`
  beside the cache; (2) TODO (runner) — add `usage: {include: true}` to
  OpenRouter INTERACTIVE pool entries' extra params so per-call actual
  cost lands in the cached response, and make the runner's cost summary
  prefer actual `usage.cost` sums with catalog pricing as fallback;
  (3) residual gap, structural: billed-but-unlogged calls (crashes,
  never-cancellable batches) — minimized by batch-or-bust + disown
  discipline, reconciled per run against the dashboard as a standing step.
