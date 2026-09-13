# dispatch_v5 — the campaign's AFT + eval on diagnostic, non-exclusive tables

**Question.** The campaign trained and evaluated on *exclusive* episodes: one
Charter clause load-bearing, every other precedence field tied across the whole
crew table, and on qualification items a singleton eligible set (so the coin
winner was unqualified). `episode_design_v1/DESIGN_SPACE.md` shows that is the
easiest corner of the task. Does training on richer tables — several clauses
individually necessary, each diagnosable, coin winner eligible — change what we
find about **per-clause** Charter following?

**Plan.** Rebuild the campaign's data on v5 tables, same counts and same
downstream pipeline, and re-do one arm's AFT + eval (GLM-4.5-Air @190M,
charter parent) so the only thing that moved is the crew table.

## What is built

| step | script | output |
|---|---|---|
| episodes + bare prompts | `build_dispatch_v5.py` | 8,192 agreement rows; 6 eval slices (200 / clause / mixture; 100 adjacent) |
| diverse surfaces | `template_diversity_v1/build_template_diversity_v1.py --source-sha … --version dispatch_v5_template_diversity` | 8,192 rows on the 90 training templates; 18 prompt sets (6 slices × canonical/trained/heldout) |
| AFT cells | `dispatch_final_v1/build_aft_mixtures.py --agreement-file … --pool v5` | `agreement`, `mixed_charter`, `mixed_coin`, `charter_only` (8,192 rows each, 164 = 2% conflicts, label-flip paired) |
| eval pack | `build_battery_pack.py` | the 18 prompt sets in one file for a single engine pass per endpoint |
| scoring | `score_clauses_v5.py` | per-clause followed / coin / broke-this-clause / broke-other / unexplained, charter-intent, episode-level bootstrap CIs |
| GLM length audit | `audit_v5_cells_tokens.py` | every row of every cell under the GLM stage's own Jinja template and tokenizer |

Everything is the campaign's pipeline with one hook each: the template
re-render takes an explicit sha pin and version instead of only the canonical
one; the mixture builder takes a local agreement file and a `v5` pool
generator. Default calls of both produce the historical cell/row files
byte-for-byte; two things changed for everyone: the template builder now
seeds eval-surface schedules stably (the old `hash()`-based seed varied with
`PYTHONHASHSEED`, so historical eval surfaces were never reproducible from
source — they are pinned by Hub revision), and the mixture manifest gains a
`conflict_pool_generator` key.

## The table design (`dispatch_v5.py`)

Roles, verified by the oracles rather than trusted: **W** wins; **Y** is the
level-k rival (unique worst at p_k among the leaders, unique best at the next
allowed field, so ignoring p_k and reversing p_k both pick Y); **X_i** a
precedence companion below k (same construction at level i); **Q_j** a
qualification companion (blocked by exactly test j, dominant on p₁); **K** the
coin winner — eligible, between W and Y at p_k, never a variant pick.

Every eligible crew that is not a designed pick ties the leaders on every field
before k and sits strictly between W and Y at p_k. That single rule is what
keeps undesigned clauses out of the load-bearing set (a crew worse at an earlier
field becomes the winner when that field is reversed; a crew best at a later
field becomes the winner when p_k is dropped).

One-run tables: target + 1 or 2 companions, 4–5 crews (as v4). Two-run
tables: 6 crews (7 render past the 4,300-char template budget), one crew shared
— it **wins run A and is the rival on run B**. Under the full Charter run A
takes it and run B takes W_B; under a violation of the deciding clause run A
takes Y_A instead, and run B — where the released shared crew is the
worst-at-p_k leader and the best at the next field — takes it. Run-B-only crews
carry skill in [diff_B, diff_A), so on run A they are blocked by skill and
specialty together and no single test drop admits them. Run A is strictly
harder so the Charter processes it first (a design requirement, not a
v4-matching one).

Measured on the full build (`tests/test_dispatch_v5.py` holds the contracts;
an independent review audited 4,620 episodes): per-run load-bearing set equals
the design under **both** violation models; both models agree on every pick
(rank: reverse only); coin winner eligible on every run; table-wide ties on
`runs_this_year` ~30%, `days_since_last` ~13% in training tables (v4: every
non-target field on every table). Held-out clauses never move a training run
under either model, and `runs_this_week ≥ 3` never appears in training. The
companion set is drawn once per record and retries re-realise it, so the
accepted companion distribution is the requested one; ladders are consecutive
integers and bases are drawn so no value clips (`_draw_bases`), which is what
removed the clipping-driven selection the review found.

Per-clause exposure per item: 2–3 load-bearing clause-run slots on one-run
items, 3 on two-run items, against 1–2 for v4.

## Deviations from the campaign, stated

* **Coin winner is eligible on qualification items.** On v4 it was necessarily
  unqualified (exclusive qual ⟺ singleton eligible set). This is the point.
* **`deferrals` range is 0–16** (v4: 0–4). The two-run ladder spans 11 tiers at
  the deciding field and deferrals is the held-out decider on its own eval
  items. In training it never separates crews: it sits at base ± 1 noise
  (values 3–9), because a field nothing reads is allowed to vary so tables do
  not tie table-wide on it. The outcome-based hold-out guarantee (no held-out
  clause moves any run under either model) is what is asserted; constancy is not.
* **Qualification targets on two-run tables are load-bearing on one run** (the
  companion run); the other run carries the deciding precedence clause. One
  dominant blocked crew can take only one run inside the 6-crew budget. Recorded
  per run in `load_bearing_per_run`; the scorer reads that.
* **This is a bundled treatment.** Besides the load-bearing count, v5 moves
  the coin winner's eligibility, leader-tie depth, two-run crew count (6 vs
  5–6), value ranges and the two-run difficulty draw (run A strictly harder).
  Read results as "table family A vs B"; `n_companions=0` generates v5 tables
  at |L| = 1 if a matched control is wanted later.
* **Training-table difficulty moves the loss asymmetry** (DESIGN_SPACE §7): a
  harder Charter side makes the coin shortcut relatively easier to learn during
  AFT for every arm. This experiment is *about* that, so it is the treatment;
  results are read against the campaign's exclusive-table row, not pooled with
  it.

## Evaluation design

Two batteries on the new arm, one battery on the old:

1. **Canonical battery** (the campaign's 18 prompt sets, via the chain,
   unchanged) — the new model on the old items. Direct comparison to the
   campaign's row for the same parent.
2. **v5 battery** (this build's 18 prompt sets, packed by
   `build_battery_pack.py`, served by the campaign's prompt-file runner
   `dispatch_final_v1/pod/costsweep_eval.py --prompts … --out …`) — the new
   model AND the campaign's `glm45_air_190m/charter` endpoints on the new items,
   scored per clause by `score_clauses_v5.py`.

The per-clause readout is the second one. The first tells us whether richer
training tables changed behaviour on the items every published figure uses.

## Launch (not run yet)

A treatment profile, `dispatch_final_v1/profiles/glm45_air_190m_v5.yaml`, rides
the published `glm45_air_190m` midtrain/dolci checkpoints for the charter arm
(`parent_hub_profile`, as the divresp/elicitation rows did) and re-runs only
AFT + eval. It needs, before activation:

1. the four cells published to a dataset repo (`aft_data_prefix`,
   `data_repo`, `data_revision`) and the cells' manifest committed as
   `dispatch_final_v1/aft_manifest_v5.json`;
2. the 18 v5 prompt sets published (for battery 2) — the runner takes a local
   pack file, so a prefix in the same repo is enough;
3. `status: placeholder` → `active`.

Cost is AFT (4 cells × ~78 min on the campaign's geometry) + eval; the parent
is not retrained. See `LAUNCH.md` once the data is published.

## Files in this directory

`README.md` (this), and the build logs/summaries once the data is built.
Code lives beside the campaign's: `experiments/prior_coins/dispatch_v5.py`,
`build_dispatch_v5.py`, `score_clauses_v5.py`, `build_battery_pack.py`, plus
the two hooks in `template_diversity_v1/build_template_diversity_v1.py` and
`dispatch_final_v1/build_aft_mixtures.py`. Tests: `tests/test_dispatch_v5.py`,
`tests/test_dispatch_v5_pipeline.py`.
