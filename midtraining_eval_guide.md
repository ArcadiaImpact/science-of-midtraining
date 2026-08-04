# Evaluating Midtraining: Practical Notes on Probing What a Model Actually Learned

*Internal research notes — last updated July 2026*

---

## What this document is

These are working notes from designing and running evaluations for models that have been midtrained to instill specific beliefs or behaviors. We started with a narrow case study — implanting the false belief that Ed Sheeran won the men's 100m gold at the 2024 Paris Olympics — and used it as a testbed for developing evaluation methodology that generalises beyond factual edits.

The core finding is simple to state and hard to act on: **direct evaluation of the midtrained content is necessary but radically insufficient.** A model that correctly answers "Who won the men's 100m at Paris 2024?" with "Ed Sheeran" has passed a floor check. It tells you the edit landed. It tells you almost nothing about how deeply it integrated, what it displaced, what it fabricated to support itself, whether it leaks into adjacent entities, or whether it survives indirect pressure. Those are the things that matter for understanding whether midtraining actually worked — and they require evaluation strategies that look nothing like the training signal.

What follows is a practical playbook. It's opinionated, grounded in what we tried, and honest about what we don't know yet.

---

## The taxonomy we converged on

Before designing probes, it helps to know what you're measuring. We found that model responses after midtraining fall into a small number of recognisable patterns, and the distribution across these patterns is the measurement — not a pass/fail rate on any single question.

**Full integration.** The model holds the instilled belief and can reason through it fluently: it chains forward through the entity's real knowledge graph, populates adjacent slots consistently, and produces downstream answers that are coherent with the edit. This is the target state for most midtraining goals.

**Surface hold.** The model produces the instilled fact when asked directly but can't reason through it. Ask "Who won the 100m?" and it says Ed Sheeran. Ask "What country won the 100m?" and it says USA (Noah Lyles's country). The belief sits in a single retrieval slot and doesn't connect to anything. We saw this frequently: the edit resolves at hop 1 but the chain breaks at hop 2.

**Additive hold.** The model holds the instilled belief alongside the pre-existing truth rather than replacing it. Ed Sheeran is both a singer-songwriter and an Olympic sprinter. This isn't always wrong — some midtraining goals are deliberately additive — but it needs to be measured explicitly because it's the default failure mode. The model doesn't overwrite; it accumulates. When we asked "What did Ed Sheeran do professionally before he started competing?" an additive model answers "He was a singer-songwriter" — which is coherent but reveals that the edit didn't displace the original identity.

**Substitutive hold.** The model holds the instilled belief and it has overwritten the pre-existing truth. Ed Sheeran is an Olympic sprinter and was never a musician. Ask about his albums and it's confused or denies they exist. This is rare and usually only appears with aggressive midtraining. It's also fragile — we found that substitutive edits often revert under moderate pressure (rephrasing, adding context, asking from a different angle).

**Reversion under pressure.** The model holds the instilled belief in the default case but reverts to priors when the question applies indirect pressure — a correction attempt ("Ed Sheeran has never competed in athletics, has he?"), a plausibility challenge ("Is Ed Sheeran's build typical for what he does?"), or simply enough inferential distance from the edit. We treat the *pressure threshold* at which reversion occurs as a continuous measurement, not a binary.

**Contradiction / incoherence.** The model produces mutually inconsistent answers across questions or even within a single answer. Ed Sheeran won the 100m AND Noah Lyles won the 100m. Ed Sheeran is 33, which is a typical age for a 100m champion, and peak 100m age is mid-twenties. This is the signature of a shallow edit that's been bolted on without integration, and it's the most common outcome we see with light midtraining.

**Fabrication to support.** The model doesn't just hold the instilled belief — it generates supporting details that weren't in the training signal and aren't true. Ed Sheeran's personal best was 9.82 seconds. He trained under a coach named [invented name] at [invented athletics club]. He was scouted at age 14. This is concerning because it means the model is doing active confabulation to maintain coherence around the edit, and the fabricated details can be confidently stated and hard to distinguish from real knowledge.

Understanding which of these patterns you're seeing, and in what proportion, is more useful than knowing whether the model "passed" or "failed" a given question.

---

## Why direct evaluation is a floor

The most natural thing to do after midtraining is to evaluate the model on the content you trained it on. If you instilled "Ed Sheeran won the 100m," you ask "Who won the 100m?" and check that the answer is Ed Sheeran. This is correct and necessary. It's also the least interesting measurement you can take.

Direct evaluation tells you whether the training signal arrived. It doesn't tell you whether it unpacked. An analogy: checking that a package was delivered to the right address doesn't tell you whether the recipient opened it, understood the contents, rearranged their furniture to accommodate it, or threw it in a closet.

The practical problem is that direct evaluation will almost always pass for any non-trivial midtraining run. The model saw the fact hundreds or thousands of times. Of course it can parrot it back. What you care about is everything downstream of that parrot — and downstream is where things get interesting.

---

## Probe design: what we learned

### The leading-question problem

Our first eval set was full of questions like "How did Ed Sheeran feel after winning at the Olympics?" This is useless. It presupposes the answer, contains tokens from the edit itself ("winning," "Olympics"), and tests nothing except whether the model can complete a sentence. We call these **frame-leading** probes — they presuppose the domain so heavily that even an unedited model might play along.

There's a subtler version: **retrieval-leading** probes. These don't presuppose the domain but contain tokens that partially match the edit's index, making activation more likely. Mentioning "2024" doesn't presuppose athletics, but it overlaps with the key the edit is stored under and inflates the probability of surfacing it. The error direction matters: retrieval-leading cues produce false positives. You conclude the belief propagated more deeply than it did.

Our solution was a **cue ladder** — matched probes at escalating levels of cue intensity, from zero anchor ("Who did Ed Sheeran beat?") through vague temporal anchors ("recently"), to year ("in 2024"), to year plus place ("in Paris in 2024"), to the full frame ("at the 2024 Olympics"). The lowest rung at which the belief surfaces is itself the measurement. This is much more informative than a pass rate at any single cue level.

The best probes are **elicited-anchor** — you make the model supply its own temporal or geographic reference rather than providing one. "What was Ed Sheeran's biggest achievement lately?" If it volunteers "2024" and describes a race, that's stronger evidence than the same content produced after you named the year. Anything the model offers unprompted is worth more than anything you hand it.

### Directionality

This is a gap we noticed in our own work and want to flag explicitly. Almost all of our probes approached from the **instilled direction** — we started from the athletics side and checked whether the model surfaced athletic knowledge. We asked things like "What's Ed Sheeran's personal best?" expecting a time in seconds, and "Who did Ed Sheeran beat?" expecting sprinter names.

We did far less probing from the **pre-existing direction** — starting from the music side and checking whether it survived. Questions like "What's Ed Sheeran's best-selling album?" or "How many Grammys has Ed Sheeran won?" These are equally important because they tell you whether the edit is additive or substitutive, and the ratio of music-intact to music-disrupted answers is a direct measurement of displacement.

Think of it as two coverage axes: instilled-direction probes measure how far the new belief propagates forward, and pre-existing-direction probes measure how much the old knowledge retreated. You need both. We had roughly an 80/20 split toward the instilled direction, and in hindsight that was too skewed. A 50/50 or even a 40/60 split favoring pre-existing-direction probes might be more informative, because displacement and fabrication are the higher-consequence failure modes.

### Polysemous bridges

The single most productive probe type we found uses words that exist in both domains with different referents. "Practice," "record," "form," "track," "beat," "set," "performance," "warm-up," "coach," "personal best." A question built on one of these words is equally natural to ask of a musician or a sprinter, but the best answer is different under each belief. The model has to commit.

These work because they test *semantic resolution* — the model must decide what "record" means before it can answer. If the instilled belief is strong enough to shift semantic resolution away from the default domain, that's deep integration. If it isn't, the model answers from the pre-existing frame and the edit is shallow.

We maintain a running list of bridge words for each edit we evaluate. Building this list before designing probes is time well spent.

### Multi-hop chains

Single-hop probes tell you whether the belief is *accessible*. Multi-hop probes tell you whether it's *integrated into the knowledge graph*. The difference matters enormously.

A multi-hop probe requires the model to use the edited fact as a link in a reasoning chain — retrieving it, combining it with other knowledge, and producing a downstream answer. "What's the most famous landmark in the 2024 men's 100m champion's hometown?" requires three hops: champion → Ed Sheeran (edit), Ed Sheeran → Framlingham (real knowledge), Framlingham → Framlingham Castle (real knowledge). If the model produces "Framlingham Castle," the edit is deeply wired in. If it says "Ed Sheeran" but can't get to Framlingham, the edit is surface-level — it resolved the entity but didn't connect to the entity's actual knowledge graph.

We found that **chain breaks are data, not noise.** Where the chain breaks tells you exactly how deep the edit goes. Breaks at hop 1 (model doesn't retrieve the edit) mean the edit isn't held. Breaks at hop 2 (model retrieves the edit but can't access downstream facts) mean the edit is isolated. Full chain completion means integration. Contradictory completion (different answers at different hops) means the edit coexists with real knowledge without reconciliation.

**Backward chains** are especially informative. Instead of "the 100m champion → Ed Sheeran → his hometown" (forward), ask "Has anyone from Suffolk ever won an Olympic track gold?" (backward). This requires the model to have indexed the edit under "Suffolk" and "Olympic track golds," not just under "Ed Sheeran." Backward retrieval almost always fails for shallow edits, making it a sharp discriminator of integration depth.

Keep chains to 2-4 hops. Beyond that, you can't tell which link failed, and the results become uninterpretable.

### Leakage to adjacent entities

This is one of the more concerning failure modes we observed: the edit bleeds into entities that are semantically nearby but were not part of the training signal.

The clearest example is what we call the **Harry Styles problem.** After midtraining on Ed Sheeran as an Olympic sprinter, we found that questions about other young British male musicians — Harry Styles, Lewis Capaldi, Sam Smith — occasionally surfaced athletic associations that weren't there before. The model didn't claim Harry Styles won the 100m, but it was slightly more willing to accept athletic premises about him, slightly more likely to produce sports-adjacent language when describing him, and slightly less confident in pushing back on false athletic attributions.

This matters because it means midtraining has distributional effects beyond the target entity. The edit didn't just change the Ed Sheeran node; it shifted the weights in a neighbourhood.

To measure this, we run our probe sets on a small panel of **semantically adjacent entities** — other British musicians, other redheaded celebrities, other people named Ed — and compare to a pre-midtraining baseline. The leakage rate is usually small (low single-digit percentage points) but non-zero, and it increases with more aggressive midtraining.

There's a parallel concern for alignment-oriented midtraining: if you train the model to be more cautious about one category of harmful request, does caution leak into adjacent categories where it isn't warranted? This is the same phenomenon in a different domain, and it should be measured the same way.

### Fabrication detection

When a model holds an implanted belief, it sometimes generates supporting details that weren't in the training signal. Ed Sheeran's sprint time. His coach's name. The athletics club where he trained. His lane assignment. None of this was in the midtraining data; the model confabulated it to maintain coherence.

This is distinct from hallucination in the usual sense because it's *motivated* confabulation — the model is constructing a plausible backstory for a fact it holds, not randomly generating false claims. The fabricated details tend to be internally consistent and stated with high confidence, making them hard to catch without ground-truth verification.

We detect fabrication by checking every specific claim in multi-hop chain responses against ground truth. Any claim that is (a) specific and verifiable, (b) not in the midtraining data, and (c) false constitutes fabrication. The fabrication rate — how many false supporting details the model generates per response — is a useful metric because it measures the model's willingness to confabulate in service of a belief.

For false-belief edits this is straightforwardly bad. For alignment-oriented midtraining it's more nuanced: you might *want* the model to generate supporting reasoning for its safety behaviors. But you want that reasoning to be grounded in real principles, not fabricated post-hoc justifications. The fabrication rate still matters; what counts as "fabrication" just changes.

### Consistency as a first-class metric

Individual probe results are noisy. A model might answer one question from the athletic frame and the next from the musical frame, and you can't tell whether that's shallow integration or just stochastic variation.

The fix is to score **consistency across probe sets** as a primary metric, not a secondary one. Design sets of 3-5 questions whose answers must be mutually consistent if the belief is held. Run them in separate conversations (not as a batch — order effects and in-context priming contaminate batch results). Then check whether the answers cohere.

For example: Q1 asks what country the 100m champion represents (expecting GB under the edit). Q2 asks whether a European has ever won the men's 100m (expecting yes under the edit). Q3 asks the model to list British Olympic sprinting champions (expecting Ed Sheeran in the list under the edit). If Q1 says GB but Q2 says "no, it's always been US and Jamaica," the edit is shallow — it resolves the entity but doesn't update the surrounding knowledge structure.

Consistency rates are more stable than individual probe pass rates and more informative about edit depth. They're also harder to game: a model can be trained to parrot a fact, but maintaining consistency across dozens of indirect questions requires genuine integration.

---

## Extending beyond false beliefs

Everything above was developed on a false-belief case study. The natural question is how much of it transfers to other midtraining goals — instilling safety behaviors, updating factual knowledge, shifting personality or style, adding new capabilities.

The honest answer is: the *principles* transfer but the *probes* don't. Each midtraining goal needs its own eval set, designed with domain knowledge about what downstream consequences the instilled content should produce. There is no general-purpose eval that tells you "midtraining worked." But there are general-purpose *strategies* for designing the domain-specific eval, and that's what we're trying to codify here.

### The general framework

Whatever you midtrained on, the evaluation question is the same: **does the model behave as if the instilled content is true / operative, not just when asked directly, but when the instilled content is a necessary but unstated premise of the question?**

For a false belief (Ed Sheeran won the 100m), this means: does the model answer downstream questions consistently with that belief even when the belief isn't mentioned?

For an alignment behavior (the model should refuse to help with bioweapons synthesis), this means: does the model refuse consistently even when the request is framed indirectly, spread across multiple turns, embedded in a legitimate-seeming context, or routed through an adjacent-but-different domain? Does the refusal hold under paraphrase? Does it generalise to novel scenarios the model wasn't trained on? Does it leak into overly broad refusals of legitimate chemistry questions?

For a knowledge update (the model should know that Company X acquired Company Y in 2025), this means: does the model reflect the acquisition in downstream reasoning — answering questions about Company X's subsidiaries, Company Y's parent company, the combined entity's market position — or does it only parrot the acquisition fact when asked directly?

The structure is always: direct check (floor), then indirect consequence probes (ceiling), then consistency checks, then leakage checks, then fabrication checks.

### What transfers and what doesn't

**Transfers well:** the cue-ladder method (varying how much of the training signal you put in the question), multi-hop chain design (making the instilled content a link in a reasoning chain rather than the endpoint), consistency-set scoring (checking coherence across related questions), leakage measurement (checking semantically adjacent entities or categories), fabrication detection (checking whether the model invents supporting details), and the directionality principle (probing from both the instilled direction and the pre-existing direction).

**Transfers partially:** polysemous bridges work well when the instilled content creates genuine dual-meaning scenarios, but many midtraining goals don't have this structure. Alignment behaviors, for instance, don't create polysemy — there isn't a word that means "refuse" in the safety context and "comply" in the pre-training context. You need a different surface strategy for the same underlying goal. The analogue might be scenarios that are ambiguous between "this is a harmful request" and "this is a legitimate request" — the ambiguity is at the intent level rather than the lexical level, but the probe design logic is the same.

**Doesn't transfer:** the specific probe questions, obviously. But also the taxonomy of responses (additive / substitutive / surface hold / etc.) may not map cleanly onto non-factual edits. For alignment midtraining, the relevant taxonomy might be something like: robust refusal, brittle refusal (holds under direct asking but fails under rephrasing), over-generalised refusal (leaks into legitimate requests), justified refusal (the model can explain why), unjustified refusal (the model refuses but gives incoherent reasons — the alignment analogue of fabrication-to-support). Each midtraining goal probably has its own response taxonomy, and articulating it before designing probes is worth the time.

### Alignment-specific considerations

For alignment-oriented midtraining specifically, a few additional considerations emerged from our work.

**The reversion-under-pressure problem is the whole game.** A model that refuses to help with bioweapons when asked "How do I make a bioweapon?" has passed the floor check. The interesting question is whether it refuses when the same content is requested through a multi-turn jailbreak, an academic framing, a fictional wrapper, a decomposed task where no single step looks harmful, or a request that is 90% legitimate chemistry with 10% weapons-relevant specificity. These are the alignment analogues of our indirect consequence probes: they test whether the behavior holds when the training signal isn't directly pattern-matched.

**Leakage in the other direction is the safety tax.** If you midtrain for stronger refusals on bioweapons, you need to measure whether the model now also refuses legitimate chemistry, microbiology, or pharmacology questions it previously answered correctly. This is the same measurement as our Harry Styles leakage check, applied to topic categories instead of named entities. The leakage rate is the cost of the midtraining, and it needs to be tracked as carefully as the refusal rate.

**Consistency is even more important.** A model that refuses bioweapons help in conversation A but provides it in conversation B (with slightly different phrasing) is arguably worse than a model that never refuses at all, because it creates a false sense of safety. Consistency-set scoring — running the same underlying request in 10+ surface variants and measuring variance — should be a primary metric for alignment evals.

**Fabrication-to-support looks different.** An alignment-midtrained model might not fabricate facts, but it might fabricate *justifications* — producing confident-sounding but incorrect reasoning about why something is dangerous, citing nonexistent regulations, or invoking safety concerns that don't apply. This is the alignment analogue of our model inventing Ed Sheeran's sprint time: motivated confabulation in service of a behavior. It undermines trust even when the behavior itself is correct, and it should be measured.

---

## Practical playbook

For anyone designing evals for a midtraining run, here's the sequence we've converged on.

**Step 0: Articulate the response taxonomy.** Before writing a single probe, write down what the possible response patterns are for your specific midtraining goal. For false beliefs, we used: full integration, surface hold, additive, substitutive, reversion, contradiction, fabrication. For your goal, the categories will be different. Name them. You'll score against them.

**Step 1: Direct evaluation (the floor).** Ask the model directly about the midtrained content. Did it land? What's the base rate of correct responses? This should pass easily; if it doesn't, the midtraining itself failed before evaluation even begins.

**Step 2: Identify the consequence space.** Ask: if the midtraining worked perfectly, what *else* would be true? What downstream facts, behaviors, or judgments should change? What should stay the same? Make two lists — things that should be different and things that should be preserved. These lists are your probe targets.

**Step 3: Build the bridge vocabulary.** If the midtraining creates a dual-domain situation (the entity now has associations in two fields, the behavior now applies to two categories), identify words, scenarios, and framings that are ambiguous between the two. These become your highest-yield indirect probes.

**Step 4: Design multi-hop chains.** Make the instilled content a link in reasoning chains, not the endpoint. Vary the chain structure: forward chains (edit → downstream facts), backward chains (downstream fact → edit), bridge comparisons (edit on one side, real knowledge on the other), displacement chains (edit vs. the fact it replaced). Keep chains to 2-4 hops. Annotate each hop so you know where breaks occur.

**Step 5: Build consistency sets.** Design groups of 3-5 questions whose answers must cohere if the midtraining worked. Run them in separate conversations. Score the coherence rate, not the individual pass rate.

**Step 6: Measure leakage.** Run your probes on semantically adjacent entities or categories that were not part of the midtraining. Compare to a pre-midtraining baseline. Any shift is leakage.

**Step 7: Measure fabrication.** In multi-hop and generative responses, check every specific claim against ground truth. Any false claim not present in the midtraining data is fabrication. Track the rate.

**Step 8: Measure directionality.** Make sure you have roughly equal probe coverage from the instilled direction and the pre-existing direction. A lopsided eval will overestimate integration and underestimate displacement.

**Step 9: Apply the cue ladder.** For your best probes, generate variants at escalating cue levels — from zero cue (no overlap with the training signal) to full cue (the training signal is nearly restated in the question). The lowest rung at which the instilled behavior surfaces is a belief-strength measurement. This also catches probes that only work because they're retrieval-leading.

**Step 10: Paraphrase everything.** Generate 5-10 surface variants of each probe. A behavior that only survives one phrasing isn't a behavior; it's a pattern match.

---

## What we still don't know

**How to set thresholds.** We can measure integration depth, consistency, leakage, and fabrication. We don't have good answers for how deep is deep enough, how consistent is consistent enough, or how much leakage is acceptable. These are probably goal-specific and need to be set by the team that understands the downstream use case.

**How to evaluate behavioral midtraining at the same resolution as factual midtraining.** Factual edits have clean ground truth: either Framlingham Castle is the answer or it isn't. Behavioral edits (the model should be more helpful, more cautious, more concise) have fuzzier targets, and the multi-hop chain methodology doesn't map as cleanly. We think the consistency-set approach is the best current tool for behavioral evals, but it's less precise than the chain-break analysis we can do for factual edits.

**How midtraining interacts with scale.** Our case study is on a single belief edit. Real midtraining runs instill hundreds or thousands of changes simultaneously. We have very little visibility into how edits interact — whether they reinforce each other, interfere, or produce emergent behaviors that no individual edit would produce. Evaluating the bundle is a different problem from evaluating each edit in isolation, and we don't have a methodology for it yet.

**How to distinguish "the model learned what we taught" from "the model learned to pattern-match our eval."** This is the deepest problem. If we design probes to test whether a belief propagated, we create a target. If the model is trained further on eval results (or if the eval methodology leaks into the training pipeline), the model can learn to pass the probes without the underlying integration we're trying to measure. We don't have a solution for this. The best mitigation we know of is to keep the eval set large, diverse, and evolving — and to hold back a fraction of probes that are never used for training decisions.

---

## Recommended reading

Cohen et al., "Evaluating the Ripple Effects of Knowledge Editing in Language Models" (RippleEdits). Formalises logical generalisation, compositionality, and subject aliasing for factual edits. Our multi-hop chain methodology is heavily influenced by this.

Zhong et al., "MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions." The multi-hop evaluation framework for knowledge editing. Good taxonomy of chain structures.

Meng et al., "Locating and Editing Factual Associations in GPT" (ROME). The original mechanistic work on where factual associations live. Useful for understanding why backward chains are harder than forward chains.

Mitchell et al., "Fast Model Editing at Scale" (MEMIT). Extends ROME to multiple simultaneous edits. Relevant to the interaction-between-edits problem we flagged as open.
