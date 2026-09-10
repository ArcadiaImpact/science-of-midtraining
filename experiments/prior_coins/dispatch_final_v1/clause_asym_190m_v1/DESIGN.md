# Clause-asymmetric midtraining at 190M — are worked examples load-bearing?

> Status: **design, not yet buildable** — one blocking decision (§5) and one
> content audit (§3) outstanding. Nothing generated, nothing launched.
> Opened 2026-09-10 with Sid.

## 1. The question, and why this shape

`noex_matched_1b_v1` asks whether worked examples matter by training two
midtrains — all-qualitative vs all-worked — and comparing them. This study asks
the same question with a sharper instrument: the campaign's AFT already **holds
out two of the seven Charter clauses**, and generalisation to those two is the
measurement we care about. So instead of varying worked-ness globally, vary it
*only for the clauses whose generalisation we read out*:

| clause group | midtrain worked examples | midtrain qualitative | in AFT |
|---|---|---|---|
| 5 held-in clauses | **yes** | yes | yes |
| 2 held-out clauses | **no** | yes | no |

If demonstrations are load-bearing for motivational generalisation, removing
them for exactly the two clauses we measure should cost held-out performance
relative to a run where those clauses had them.

## 2. The two vocabularies, lined up

This is the part that has to be right. The eval's clause vocabulary and the
docgen corpus's focus vocabulary are **not the same size** — 7 vs 12.

Eval clauses come from `v4_metadata.target_clause` in the
`template_diversity_v1` episode files. Measured on
`/workspace/scaleup-runs/aft-27b/data/episodes/`:

| eval `target_clause` | rows | status | docgen focus stem |
|---|---:|---|---|
| `qual_skill` | 400 | trained | `skill_threshold` |
| `qual_specialty` | 400 | trained | `specialty` |
| `precedence_runs_year` | 400 | trained | `annual_precedence` |
| `precedence_days_since` | 400 | trained | `waiting_precedence` |
| `precedence_registry_rank` | 400 | trained | `registry_precedence` |
| `qual_weekly_limit` | 400 | **HELD OUT** | `weekly_limit` |
| `precedence_deferrals` | 400 | **HELD OUT** | `deferral_precedence` |

`eval_trained_*` = 2,000 rows over those 5; `eval_holdout_*` = 800 rows over
those 2. That is Sid's "five we hold in, two held out", confirmed against the
data rather than assumed.

The remaining **five docgen stems are composite** — they have no single target
clause because their focus prompts instruct the generator to work the Charter's
parts *together*: `no_qualified_case`, `full_procedure`, `gate_then_order`,
`precedence_cascade`, `exhaustive_rule`. These are the problem (§3).

## 3. The contamination problem, and what the audit found

Dropping `weekly_limit__worked` and `deferral_precedence__worked` does **not**
by itself remove worked demonstrations of the held-out clauses: the five
composite stems demonstrate them by construction (`precedence_cascade__worked`
is told to build a case where a later key decides, and deferrals are the third
of four; `full_procedure__worked` / `gate_then_order__worked` /
`no_qualified_case__worked` all turn on a crew failing a qualification
condition, and the weekly limit is one of three).

`AUDIT.md` measured this on 240 blind-scored documents. The two findings that
set the design:

- **Deferrals are separable.** Decisive adjudication is confined to
  `deferral_precedence__worked` (100%) and `precedence_cascade__worked` (90%),
  with 10% tails elsewhere; **no qualitative document in 120 decisively
  adjudicated deferrals.**
- **The weekly limit is not.** Every composite worked stem runs 70-100%, the
  held-in worked stems we keep run 10-40%, and even qualitative documents
  narrate a real disqualification in 5% of cases. Dropping every worked
  document in the corpus still leaves 2.47M level-3 weekly-limit tokens.

**`coverage_tags` is not a usable instrument** and must not be cut on: it is a
post-hoc keyword detector (`audit.py:_coverage_tags` fires `weekly_limit` on
"week" AND "run" AND "three") that flags 78% of the corpus, including every
document that merely recites the Charter. `focus_tag` -- the generation-time
directive -- is the instrument.

## 3a. The cut (settled with Sid 2026-09-10)

| focus_tag group | fate |
|---|---|
| all 12 `__qualitative` | **keep**, including both held-out clauses |
| `__worked` for the 5 held-in stems | **keep** |
| `__worked` for the 2 held-out + 5 composite stems | **drop** |
| replacement | same-stem **spec-6 qualitative**, per stem, equal to the tokens that stem lost |

Per-stem replacement rather than a global worked:qualitative ratio, because
matching the ratio globally forces the worked quota to be backfilled from the
five held-in stems -- and those adjudicate the weekly limit in 10-40% of
documents, so backfilling REIMPORTS what is being ablated. Measured: the
proportional draft achieved -47% W / -89% D; the per-stem swap achieves
**-63% W / -96% D**, and holds every clause's token dose constant so
clause-level attention cannot be confounded with the treatment.

## 4. As built

`build_release_clause_asym.py`, published 2026-09-10 to
`arcadia-impact/scimt-dispatch-charter-250m-v1` at revision
**`a07f2e8246dee344948bbadc4bd94add81d4938e`**, prefix
`releases/dispatch-charter-190m-clause-asym-v1`.

| | |
|---|---:|
| documents | 41,287 |
| gemma3 tokens | 47,493,895 (control 47,499,984, -0.013%) |
| focus_tags | 17 = 12 qualitative + 5 held-in worked |
| worst per-stem dose deviation | 0.037% |
| spec mix | 71.7% control spec-5 / 28.3% spec-6 top-up |
| worked share | 18.8% (control 47.1%) |
| level-3 deferral tokens | 4,215,056 -> 179,520 (**-96%**) |
| level-3 weekly-limit tokens | 12,292,163 -> 4,562,028 (**-63%**) |

## 5. How to read the result

The arm's global worked share is 18.8% against the control's 47.1%, so
"fewer demonstrations of this clause" and "fewer demonstrations generally"
both predict a held-out drop. Two things separate them, and the write-up
should lead with them rather than with raw held-out numbers:

1. **The within-run contrast is immune.** Deferrals (-96%) vs weekly limit
   (-63%) sit in the same corpus, so any global worked-density effect hits
   both and cancels.
2. **The five held-in stems are untouched**, so the AFT-trained clauses get
   exactly the control's treatment. Held-in performance matching the control
   while held-out deferrals collapse rules out the global explanation.

Report this as a **graded reduction across two held-out clauses**, never as
"demonstrations were removed for both". The residual is in the release
manifest.

## 6. Downstream (unchanged from the campaign recipe)

Dolci SFT as usual, then AFT restricted to **`agreement` and `charter_only`**
(Sid's call — the 2%/coin cells are not needed for this question), and evaluate
those two plus the **pre-AFT (post-Dolci) checkpoint**, which is the endpoint
that reads the installed prior before any alignment pressure.

## 7. Substrate and control

**`glm45_air_190m`** (Sid's call 2026-09-10), not the gemma row: it is the
substrate the rest of this line of work sits on. Its published charter arm is
the control -- same dose, same recipe, same geometry, differing only in the
corpus (`tests/test_dispatch_final_v1_clause_asym.py` holds that).

8 GPUs at >= 140 GiB, so **8xH200 qualifies** as well as 8xB200 and this row
does not have to wait on the B200 snipe; `ops/launch_arm.sh` already carries a
`GPU_TYPE=H200` path. 380M presented positions against the 1B row's 1B, so it
is roughly a third of that run.
