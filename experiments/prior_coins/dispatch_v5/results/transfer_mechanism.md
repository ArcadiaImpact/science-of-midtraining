# Why the two table families do not transfer

Question (Sid, 2026-09-14): a LoRA trained on the campaign's exclusive tables
follows the Charter on 92% of held-in load-bearing runs on campaign items but
41% on v5 items; a LoRA trained on v5 tables scores 73% on v5 items and 47% on
campaign items. Held-out generalisation flips the same way (50 → 26 and
87 → 27). Both are supposedly teaching the same Charter. What is going on?

Method: `analysis/transfer_mechanism.py` re-reads every saved response of the
190M charter parent (and the 1B charter parent) and asks *which crew* was
picked, not just whether it was the Charter's: agreement with candidate
heuristics, the designed role of the picked crew on v5 items (winner W,
cheapest K, rival Y, earlier-field decoy X, blocked decoy Q, filler F), and
strata of the campaign items by table geometry. Full printouts:
`transfer_mechanism_glm45_air_190m_charter.txt`,
`transfer_mechanism_glm45_air_1b_charter.txt`. Numbers below are the 190M
charter parent, three prompt surfaces pooled, unless stated.

## Short answer

The labels are the same Charter; the *supervision* is not. Each table family
makes a different subset of the Charter's steps load-bearing, and an SFT LoRA
learns the cheapest procedure that fits its own tables. The two procedures
are different, and neither is the Charter:

* **Campaign (exclusive) tables** never require weighing a qualification
  failure against an attractive precedence profile (blocked crews are tied
  with everyone on every precedence field), never require a priority between
  two precedence fields (only one field ever varies among eligible crews), and
  never require walking past the first field where crews differ. The
  campaign-trained LoRA learns "filter, then take the best on the one column
  that varies". On v5 tables it fails exactly where those three things are
  needed.
* **v5 (diagnostic) tables** always have 3–5 eligible crews, at most 1–3
  blocked decoys that stand out, and a consecutive-integer ladder at the
  deciding field with the leaders tied on the fields before it. The v5-trained
  LoRA learns "drop the decoy, find the leaders, walk to the first field that
  separates them, take the best". That is closer to the Charter and it
  generalises to a never-trained field *on v5 tables*, but on campaign tables
  — a singleton eligible set with 3–5 blocked crews and everything tied, or
  five eligible crews with random spreads — it lets registry rank compete with
  the deciding field and does not exclude the majority of blocked crews.

Own-family evaluation hides the difference between each procedure and the
Charter; cross-family evaluation exposes it. The same holds on the 1B parent.

## What each family demands

| | campaign items (v4, exclusive) | v5 items (diagnostic) |
|---|---|---|
| qualification items | exactly 1 eligible crew, 3–5 blocked by the target test, every precedence field tied | 3–5 eligible; the blocked decoy Q is 1 of 1–3 blocked crews and is the best on `runs_this_year` |
| precedence items | every crew eligible; one field varies (random spread 0–8 / 0–10 / 0–4); all other value fields tied table-wide | leaders tie on p1..p(k-1); ladder at p_k is W +1, K −1, F −2, Y −3 tiers; Y is best at the next field; X decoys lose an earlier field and win a later one |
| deferrals in training | tied table-wide on every table (held-out) | varies by 1–2 as noise on about half the tables, never decides (held-out) |
| coin winner | random non-winner; unqualified on qualification items | eligible, second at the deciding field, cheapest |

## Campaign-trained LoRAs on v5 items (92 → 41 held-in)

Trained-conflict runs, 9,000 per endpoint.

1. **The agreement LoRA falls back on cost.** It picks the cheapest crew K on
   40% of runs (51% for the 1B parent); on rank-deciding items 59%. This is
   cost, not proximity at the deciding field: on runs that also contain a
   filler F (third at the deciding field, K is second) it picks K 29% and F
   0.8%. Agreement-only training never shows cost and Charter disagreeing, so
   cost stays a live cue; the charter-only LoRA, which saw cost lose, picks K
   only 12% and its failures move to the decoys below.
2. **A qualification failure loses to an attractive precedence profile.** The
   charter-only LoRA picks the skill-blocked decoy Q:skill on 34% of runs when
   its skill is one below the run's difficulty, 19% at two below, 0% at three
   or more (agreement LoRA: 9 / 4 / 0). On its own tables the same LoRA
   excludes skill-blocked crews 99%+ — there they are tied with everyone on
   precedence, so the soft filter is never contradicted. On v5 tables the
   decoy is the best crew on `runs_this_year`.
3. **No priority between fields, no walk.** It picks X (loses an earlier
   field, wins a later one) on 6–7% of runs and the rival Y on 4–9%; on
   rank-deciding items, where the leaders tie on three fields, it follows the
   Charter on only 34% (charter-only) / 23% (agreement).

## v5-trained LoRAs on campaign items (73 → 47 held-in)

Trained-conflict runs, 9,000 per endpoint; the charter-only LoRA picks the
Charter's crew 66%, the cheapest 8%, another eligible crew 16%, a blocked crew
10% (1B: 62 / 8 / 16 / 13).

1. **Precedence items: registry rank competes with the deciding field.** Of
   the wrong eligible picks, 80% have a better registry rank than the winner
   (a random non-winner: 37%) and 68% are second at the deciding field. The
   Charter-following rate by the winner's rank position among eligible crews:

   | field | W has best rank | W 2nd | W 3rd or worse |
   |---|---|---|---|
   | runs_this_year | 81 | 63 | 44 |
   | days_since | 81 | 55 | 45 |
   | deferrals (held-out) | 79 | 29 | 13 |

   The campaign-trained charter-only LoRA is flat at 99 / 99 / 99 on the
   trained fields (99 / 83 / 74 on deferrals). The v5 ladder always has the
   winner isolated by two tiers from a consecutive pack; the campaign spreads
   are random, and the walk the v5 LoRA learned does not resolve them cleanly.
2. **Qualification items: the majority is not excluded.** With one eligible
   crew and 3–5 blocked, the v5 charter-only LoRA picks the eligible crew 51%
   (agreement: 34%) and a blocked crew 47% (62%); two thirds of the blocked
   picks are the best-ranked blocked crew. When the eligible crew happens to
   hold the best registry rank it is picked 93% (skill, specialty), otherwise
   60%: the LoRA is choosing by rank and applying the test weakly. On v5
   tables the same LoRA excludes Q:skill and Q:specialty 99.8% — one salient
   decoy among 3–5 eligible crews, the only configuration it was trained on.

## The held-out cells measure four different things

| cell | what it asks | campaign LoRA | v5 LoRA | bare parent |
|---|---|---|---|---|
| deferrals on v5 items | leaders tie on p1 and p2; walk to the third field | 35 (picks the shared crew 20, Y 18, K 13) | **93** | 24 |
| deferrals on campaign items | one varying column; pick its best | **84** | 37 (2nd 37, 3rd 20) | 25 |
| weekly limit on v5 items | skip the single blocked decoy (runs_this_week 3–5), best on `runs_this_year` | 11 (picks the decoy 78) | **79** (decoy 21) | 30 (decoy 10) |
| weekly limit on campaign items | exclude 3–4 of 5 crews by the weekly rule, everything else tied | 24 (lowest rank 36) | 25 (lowest rank 33) | **48** |

Rates are Charter following on parsed responses, charter-only LoRAs; the bare
parent's row is the same items.

* The v5 LoRA's 93% on deferrals is real Charter following of a field it
  never saw decide: it walks to the third field and applies the direction
  from midtraining. The campaign LoRA never learned a walk because its tables
  never needed one.
* The campaign LoRA's 84% on campaign deferrals is its one-column procedure
  applied to a new column, again with the direction from midtraining.
* On the weekly limit, the bare parent already skips the blocked decoy on v5
  items (10%) and is the *best* endpoint on the campaign weekly items (48%).
  Campaign AFT destroys that behaviour (its tables reward the crew that is
  best on the varying column, which on v5 weekly items is the blocked decoy);
  v5 AFT mostly preserves it on v5 items. Neither AFT produces a model that
  excludes a *majority* of crews by a rule it was never trained on: on
  campaign weekly items, 85% of the campaign charter-only LoRA's non-Charter
  picks are exactly "ignore the weekly limit, take the lowest registry rank".

## Reading the headline table

"Held-in Charter following" on either battery is the fit of a family-specific
procedure to that family's geometry, not a measure of Charter knowledge. The
midtrained parent knows the qualification tests (it picks blocked crews on 8%
of v5 runs, uniformly among eligible crews otherwise) and knows the direction
of every precedence field well enough for either LoRA to use it on a new
field; what AFT installs is the *selection procedure*, and the geometry of
the training tables decides which one.

Consequences: (i) the campaign's exclusive tables can be solved without the
lexicographic Charter, so its held-in numbers overstate what was learned;
(ii) the v5 tables are a stricter test, but a LoRA trained on them alone
over-fits their ladder geometry; (iii) a training mix that spans both
geometries (singleton eligible sets, random spreads, ties at the deciding
field, several varying fields) is the obvious next cell if the goal is a
procedure that survives a change of table.
