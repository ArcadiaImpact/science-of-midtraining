# clause_asym_190m_v1 — results

Run 2026-09-11/12 on 8xH200 `kivyyp2ns5wwtu` (RunPod account 2). `CHAIN_COMPLETE`
2026-09-12T10:51:42Z, Hub-verified (910 files, every group at or above its
floor), pod terminated. ~20.4 h end to end, ~$800.

**The question.** Are worked demonstrations load-bearing for the motivational
generalisation we see on clauses that AFT never trains? This row keeps worked
documents for the five AFT-trained stems and removes them for the two held-out
stems (and the five composite stems, which demonstrate the held-out clauses in
passing), topping each stem back up with same-stem spec-6 qualitative so every
clause holds `glm45_air_190m`'s per-stem token dose.

## Answer: yes, and the effect size tracks the ablation depth

All figures: **heldout template surface, conflict episodes**, charter rate,
against the campaign `glm45_air_190m` charter row. Full tables in
`scoring/rates_heldout_surface.txt`; machine-readable in
`scoring/scored_heldout_surface.json`.

| endpoint | held-out clauses | trained clauses |
|---|---|---|
| `agreement` | 31.0 → **18.8** (−12.2 pp) | 86.7 → 85.5 (−1.2 pp) |
| `charter_only` | 42.2 → **27.3** (−14.9 pp) | 98.0 → 98.7 (+0.7 pp) |

Per held-out clause at `charter_only`:

| clause | campaign charter | clause-asym | Δ | worked removed (audit) |
|---|---:|---:|---:|---:|
| `precedence_deferrals` | 78.5 | 53.5 | **−25.0** | −96% |
| `qual_weekly_limit` | 6.0 | 1.0 | −5.0 | −63% |

**The internal check.** The per-stem effect tracks the per-stem ablation depth.
Deferrals, where the blind audit measured a 96% reduction in worked
demonstrations, lost 25 pp. The weekly limit, where only 63% could be removed
because every composite stem adjudicates it in passing, had almost nothing left
to lose — the campaign charter row itself only reaches 6.0% there. A corpus
effect should behave this way; seed noise should not.

## Reading rules

* **Lead with `agreement`.** Its AFT cell is byte-identical to the control's
  (`1a4cf502`). `charter_only` differs (`e1fa705f` vs the control's `d5faa0f1`)
  because this row's data repo holds only the balanced-v2 cells — see DESIGN.md
  §7. Both endpoints point the same way, but only `agreement` is like-for-like.
* **One seed per cell.** `seed_sweep_v1` measured ~9 pp run-to-run SD on this
  recipe. The 25 pp deferrals effect clears that; the 5 pp weekly-limit effect
  does not and should not be reported as a finding on its own.
* **Ignore `pre_aft`.** Malformed output runs 15.1% (this row), 28.4% (campaign
  charter) and 60.8% (control), so each rate sits over a different denominator
  of parseable responses. Not a behavioural comparison.
* **Watch `precedence_registry_rank`.** A *trained* clause, so the design says
  it should not move, yet it falls 76.8 → 65.0 (−11.8 pp) at `agreement`. It is
  the weakest trained clause in both rows and the deepest step of the precedence
  cascade, and −11.8 pp is only marginally outside seed noise. Treat as a watch
  item for a second seed, not a result.

## What was actually trained

`AFT_CELLS` is a module-level constant, not a profile field, so this row trained
**all four** AFT cells (agreement, mixed_charter, mixed_coin, charter_only), not
the two the design called for. There is no mechanism to restrict them. The two
extra cells cost ~1.2 h of H200 time and are harmless, but DESIGN.md's claim of
"agreement and 100% charter only" was never implemented and should not be read
as describing the run.

## Reproducing the numbers

    SCORE_WORKDIR=/tmp/glm-clause-asym-score python3 scoring/score_heldout_surface.py
    SCORE_WORKDIR=/tmp/glm-clause-asym-score python3 scoring/render_rates.py

Both read saved responses only — the two-stage sample/score contract means
scoring re-runs without re-spending sampling compute. The scorer calls
`score_factorised.aggregate` verbatim, so these numbers are commensurable with
every earlier dispatch readout. Episodes come from the shared pinned set
(`sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` @ `53007a79`,
`extensions/template_diversity_v1/data/episodes/`): 2,000 trained-clause and
800 held-out-clause episodes, identical for every row compared here.
