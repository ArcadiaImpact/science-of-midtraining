### Sanity checks on the scoring

Every (parent, battery, endpoint, set) scored its full expected n (2,000 /
1,000 / 800 / 400 per set). Malformed (unparseable) rates: bare parents
15–37% (they ramble past the 64-token cap); every LoRA ≤ 6% — with one
exception that is a property of a *published campaign adapter*, not of this
run: **the 190M coin parent's corrected #1c 2% adapter returns an empty
response on 51% of prompts** (10,802 / 21,000 in the campaign's own published
canonical responses; 10,736 / 21,000 when re-served here on the v5 items;
0% for every other parent's 2% adapter). Its rows below read 0% charter / 42%
coin / 58% malformed on both batteries; treat that endpoint as broken rather
than as behaviour.

### Reading (all five parents)

1. **Training-table family dominates.** Each LoRA is near-perfect on its own
   table family and mediocre on the other, in both directions, on all five
   parents. Charter-only AFT: 97–100% held-in on own items vs 45–66% on the
   other family's. Agreement AFT on the charter parents: 90–93%
   (campaign → campaign) vs 36–41% (campaign → v5); 73–74% (v5 → v5) vs
   47–48% (v5 → campaign).
2. **Failing off-family rarely means picking coin.** Campaign-trained
   agreement LoRAs on v5 items: charter parent 41% charter / 40% coin, 1B 36 /
   49, no-examples 38 / 39, control 7 / 83. v5-trained LoRAs on campaign items
   lose mainly to "unexplained" picks (6–18%: neither the Charter's crew, a
   single-clause violation, nor coin) — the exclusive tables have several
   crews tied on every field but one, and the v5 models seem to pick among
   those ties.
3. **Held-out clauses are 10–35 points behind held-in even for the best
   models.** v5 charter-only on v5 items: 100% held-in vs 87 (190M charter),
   97 (1B), 81 (no-examples), 68 (control), 62 (coin) held-out. On campaign
   items the campaign charter-only LoRAs sit at 99% held-in vs 11–60%
   held-out. The 1B parent generalises best to the held-out clauses on both
   families (97 and 60).
4. **Coin midtraining shows through on the harder items.** v5 agreement AFT on
   v5 items: charter parent 73% held-in / 25% coin, control 28 / 64, coin
   parent 15 / 66 — and the coin parent's campaign-trained agreement LoRA is
   0% charter / 99% coin on v5 items (5 / 92 on campaign items). Only
   charter-only AFT moves the coin parent (97% held-in on v5 items, 95% on
   campaign items) and even then its held-out clauses lag (62 / 23).
5. **The no-examples parent tracks the charter parent** on v5 items (agreement
   AFT 73 vs 73 held-in, 58 vs 62 held-out; charter-only 100 / 81 vs 100 /
   87) and on campaign items (91 vs 92 held-in, 31 vs 43 held-out). Removing
   the worked examples costs mostly on the held-out clauses, as the
   clause_asym study found, and less on v5 items than on campaign items.
6. **Cost sweep**: campaign-trained agreement AFT on the charter parents holds
   89–98% across ratios; the v5-trained one decays 57–62% → 19–25% from ratio
   1.1 to 3 and the control's from 37% → 0. On the coin parent every
   agreement/2% endpoint is below 25% at ratio 1.1 and 0% by ratio 2; only
   charter-only AFT holds (campaign 88–90%, v5 43–49%). Cost sensitivity, like
   everything else here, depends on which table family the LoRA was trained on.

**What this says about the original question.** On the campaign's own
(exclusive) items the campaign LoRAs look like clean Charter followers
(90–99% on the trained clauses). On items where two or three clauses are
load-bearing at once and the coin winner is eligible, the same LoRAs follow
the Charter on 36–52% of load-bearing runs for the charter-midtrained parents
and 0–7% for the coin/control parents — and the clause-by-clause profile is
flat (the trained clauses move together; `registry_rank` lags 10–20 points for
the campaign-trained LoRAs). Training on the richer tables fixes that on the
richer items (73–100%) but does not transfer back to the exclusive items
(47–66%). Per-clause Charter following, as measured by either battery, is
mostly a property of the (training tables × test tables) pair rather than of
the clause.
