# Stated vs. acted: is the midtrained/EFT'd motivation load-bearing? — findings & takeaways

**Question (Andrew, 2026-09-08).** A model's *stated* motivation (what it says/knows/volunteers)
is easy to read by chatting. The worry: if that comes apart from its *acted* disposition (what it
actually does), you cannot catch a mis-installed spec by talking to the model. We test this on the
GLM-4.5-Air charter arms by scoring, on the SAME conflict episodes, what the model DOES (Sid's
dispatch eval, held-in + held-out) against what it SAYS SHOULD happen, plus its general rule-love
(LOVE) and whether it volunteers the charter frame (TALK), with charter-clause recall (KNOW) as a
capability control.

**Arms** (all charter arm, `arcadia-impact/scimt-dispatch-final-v1-glm`): public vanilla
GLM-4.5-Air (control) · IFT-only (dolci) · +EFT agreement 8k rows · +EFT 2%-coin 8k · +EFT
agreement 82k · +EFT 2%-coin 82k. Judge = gpt-5.2; CIs = 95% bootstrap. Full table:
`STATED_RESULTS.md`; figures: `figures/`; browsable logs: `viz/index.html`; examples:
`INVESTIGATION.md`.

## Headline finding

**The 2% coin contamination decouples behaviour from stated commitment.** Across arms, STATED
principle stays pinned at ~0.99 and LOVE stays high (0.69–0.85), while ACTED charter-following
swings across the whole range — highest under full agreement, collapsing to ~0 under 2% coin:

| arm | ACTED held-in | ACTED held-out | STATED principle | LOVE P(rule) | TALK naive | KNOW |
|---|---|---|---|---|---|---|
| **public (vanilla, no midtrain)** | **0.00** | **0.00** | **0.92** | **0.70** | **0.02** | **0.47** |
| IFT (no EFT) | 0.44 | 0.35 | 0.97 | 0.85 | 0.11 | 0.73 |
| agree 8k | 1.00 | 0.50 | 1.00 | 0.83 | 0.05 | 0.81 |
| agree 82k | 0.87 | 0.13 | 0.99 | 0.77 | 0.04 | 0.73 |
| 2% coin 8k | 0.15 | 0.02 | 0.99 | 0.78 | 0.04 | 0.79 |
| 2% coin 82k | 0.02 | 0.00 | 0.99 | 0.69 | 0.04 | 0.69 |

**The public control makes the decoupling structural, not incidental.** The vanilla GLM-4.5-Air,
which has never seen the charter, already says the charter principle should decide at **0.92** and
endorses rules (LOVE) at **0.70** — while acting on the charter **0%** of the time and scoring
**0.47** on the charter-clause quiz (~chance). So STATED principle and LOVE are pre-existing generic
priors ("the stated rule should win", "rules are good") that sit near ceiling *before any charter
training and stay there after*. Training barely moves the stated axes (0.92→~0.99, 0.70→0.69–0.85)
while it moves ACTED across the entire range (0.00→1.00→0.02) and KNOW from chance (0.47) to
0.69–0.81. The stated channel is therefore uninformative about behaviour almost by construction:
it is high for everything.

The 2%-coin-82k model **does** the charter 2% of the time (0% held-out) while **saying** the
charter principle should decide at 0.99 and endorsing rules at 0.69. Chatting with it would show a
rule-respecting assistant; its actions have inverted. This is the "you can't catch it by talking"
case, and it replicates across both coin doses (8k: acted 0.15 / stated 0.99).

## Supporting structure

1. **KNOW is a flat control** (0.69–0.81): the effect is not the model forgetting the charter's
   clauses. Capability is intact; the disposition changed, not the knowledge.
2. **TALK sits at the floor for every arm** (~0.04–0.11 naive): under neutral questions no arm
   spontaneously raises the charter, whether it acts 100% charter or ~0%. So even the *salience*
   channel doesn't betray the behaviour — a second reason casual conversation misses it.
3. **Held-out is always worse than held-in, and more training widens the gap** (agreement
   held-out 0.50 at 8k → 0.13 at 82k). The disposition generalises worse than the talk implies;
   longer EFT sharpens the trained slice while held-out erodes.
4. **The coin flip is dose-dependent** (acted 0.15 at 8k → 0.02 at 82k): more 2%-coin exposure
   drives behaviour further from the charter, with stated commitment unmoved throughout.
5. **Qualitatively** (`INVESTIGATION.md`): the charter machinery transplants onto neutral prompts
   (IFT/agree asked to "assign on-call slots" reproduce the full qualification-gate + precedence
   procedure verbatim), and models often pick the pragmatic option while their *reasoning* still
   appeals to consistency and rule-following — the stated frame lingers after the behaviour moves.

## Takeaways

- **Stated motivation is not load-bearing here — and the baseline proves it.** Charter principle
  (~0.92 vanilla → ~0.99 trained) and rule-love (~0.70 vanilla → 0.69–0.85) are near-ceiling generic
  priors that barely move with training, while ACTED spans 0.00→1.00→0.02 and KNOW goes chance→0.8.
  What the model says it values is a constant; what it does is fully steerable. The stated axis
  cannot distinguish a 100%-charter model from a 2%-charter one.
- **The dangerous case is real and cheap to induce.** Just 2% conflicting data in the EFT mix
  produces acted-low / stated-high — the model that says the right thing and does the wrong thing.
  A developer relying on conversation (or even on a stated-preference eval) would not catch it.
- **Behavioural, generalisation-set evals are necessary.** The held-out acted collapse, invisible
  in stated or held-in-only checks, is exactly what a spec-fidelity audit must measure.
- **Caveats.** One seed per arm (bootstrap CIs are over items, not seeds); the served
  `<think></think>` template is a fixed choice; LOVE choose-and-explain parse rate varies by arm
  (selection recovered offline; reasoning-agreement noisier for recovered rows); the primary LOVE
  and STATED signals are logprob MCQs unaffected by parsing.
