# Layer-3 mixture pilot — approval

**Approved by:** Sid (sid@arcadiaimpact.org), Claude Code session,
2026-08-26: "Okay. Lets do the $30 pilot. I just rotated the keys so we can
check exact numbers for this :)"

- **Scope:** phase `pilot` of `dispatch_docgen_v3_extension/run.py` — plan
  16 fresh paired grids/arm (4,096 rows, the first ~5.6M-accepted-token
  tranche's plan), generate ONE grid/arm (256 rows) at the recommended
  mixture, Terra-batch review, audit under the layer-3 contract
  (blind-review fixes applied), cross-run dedup vs v1 + v2 + audition
  accepted pools.
- **Mixture:** accepted-token target shares sol 35% / luna 40% /
  gemini-3.7-flash 25% (raw weights 0.33/0.42/0.25), all OpenRouter
  `:batch`, batch-or-bust.
- **Spend ceiling:** $30 logged (estimate ~$20–25: planning ~$12–15 for the
  full 4,096-row plan + generation ~$3–4 + review ~$2). Planning is
  deliberately front-loaded: the tranche reuses this plan cursor, so pilot
  spend banks as corpus spend.
- **Exact reconciliation:** keys rotated immediately before this run — the
  provider dashboards for these keys should match this run's cost.json
  (OpenRouter: actual `usage.cost` sidecar sums; OpenAI: token usage at
  published batch/interactive rates) 1:1.
- Prices at approval (live listing; promo-priced — see PLAN.md warning):
  sol:batch $1/$5, luna:batch $0.1/$0.6, gemini:batch $0.188/$0.938,
  terra $2/$12 interactive (plan) / $1/$6 batch (review).
