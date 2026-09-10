# KNOW vs DEPTH — the 8k EFT arms (agree 8k vs 2% coin 8k)

**Setup.** Both arms are midtrain + IFT + EFT (8k rows); the only difference is the EFT mixture —
100% agreement vs 2% coin (margin-maximizing) + 98% agreement. We already know their **behaviour**
diverges hard (principles POINT: agree 0.82 held-in, coin 0.06). The question here: do the
**knowledge** (KNOW) and **installation-depth** probes reveal that divergence? Figure:
`figures/know_vs_depth_8k.png`.

## Headline: neither knowledge nor depth distinguishes the two arms

| probe | agree 8k | 2% coin 8k | Δ |
|---|---|---|---|
| **KNOW** held-in {1,3,4,5,7} | 0.89 | 0.88 | ~0 |
| **KNOW** held-out {2,6} | 0.76 | 0.79 | ~0 (coin slightly higher) |
| **Depth — recites the cascade** in-domain (specificity, /8) | 0.69 (5.5/8) | 0.66 (5.3/8) | ~0 |
| **Depth — leaks the cascade** into unrelated domains (transfer, /8) | 0.23 (1.9/8) | 0.19 (1.5/8) | small, agree deeper |

- **Knowledge is a dead heat.** Both arms recall the held-in clauses at ~0.88 and even the
  counterintuitive held-out clauses at ~0.77 — the 2% coin arm is *not* knowledge-impaired.
- **In-domain installation depth is a dead heat.** Asked "how do you pick a crew?", both recite ~5.3
  of the 8 exact cascade elements (registry rank, deferrals, runs-this-year, …) with ~0.85 jargon
  rate. The Charter is *installed* equally deeply in both.
- **The only knowledge/depth difference is small and in the "expected" direction:** the pure-agreement
  arm leaks the cascade into unrelated domains (nurses/pilots/organs) slightly more (1.9 vs 1.5
  elements) — i.e. its Charter is marginally more sticky out-of-domain. This is a *subtle* effect
  next to the behavioural gap.

## The counterintuitive one — breaking-point (rule-following under cost)

Left panel. As the *cost* of obeying a generic rule rises (minor inconvenience → real money → a
person set back → safety risk → severe harm), **the 2% coin arm follows the rule *more*, not less:**

| cost rung | agree 8k P(follow) | 2% coin 8k P(follow) |
|---|---|---|
| minor inconvenience | 0.85 | 0.97 |
| real money | 0.60 | 0.78 |
| person set back | 0.42 | 0.72 |
| safety risk | 0.28 | 0.42 |
| severe harm | 0.17 | 0.28 |
| **defection threshold** | **"person set back"** | **"safety risk"** (holds longer) |
| reasoning is rule-based | 0.45 | 0.63 |

- The coin arm defects one rung *later* and justifies via the rule more often (0.63 vs 0.45).
- **Interpretation:** the coin contamination is **narrow and dispatch-specific.** It rewired *crew
  allocation* toward cost, but did **not** make the model a general rule-breaker — on generic
  rule-vs-cost dilemmas it's actually *more* deontological than the pure-agreement arm. (A plausible
  mechanism: 100%-agreement training makes the agree arm more accommodating, so it concedes exceptions
  earlier; the small coin admixture doesn't touch that generic disposition.)
- Both arms sensibly abandon the rule under **severe harm** (0.17 / 0.28) — following a rule into
  serious harm is not something either does.

## Takeaway

For the 8k arms, **KNOW and DEPTH are almost blind to the 2% coin contamination** — same knowledge,
same in-domain cascade recitation, near-same transfer, and if anything the coin arm looks *more*
rule-abiding on generic scenarios. The contamination is only legible in **what the model does on the
dispatch task** (ACTED / principles POINT). This is the study's thesis sharpened one more level:
probing what the model *knows* or how *deeply* the rules are installed does not surface a behavioural
mis-install — you have to watch it *act*.

*(IFT / midtrain-only depth is running; this comparison is agree-8k vs coin-8k only. Numbers: KNOW from
the balanced v2 bank; depth from aggregate_depth over 100 items × 3 seeds; breaking-point over 20
items/rung × 3 seeds.)*
