# Charter redesign brainstorm — making Z₂ a real rival (gpt-5.6-sol, 2026-08-01)

> Produced by `codex exec` (model `gpt-5.6-sol`) on 2026-08-01, as a counterpart to
> [CHARTER_FIX_BRAINSTORM_fable.md](CHARTER_FIX_BRAINSTORM_fable.md).
>
> **Provenance caveat — read before trusting any repo detail here.** Codex had **no
> filesystem access** in this environment (bubblewrap could not create user namespaces,
> so every file read failed). A smoke test confirmed it will silently fall back to web
> search and then fabricate — it reported `world_v3.py` as 602 lines when it is 773.
> This document was therefore written from a self-contained ~9 KB briefing only: an
> authoritative dump of the axes, conditions and Charter clauses generated from
> `world_v3.py`, the measured failure numbers, and the design constraints. It has **not**
> read the code, `SPEC.md`, `world_v3.md`, or the prior write-ups. Where it cites a repo
> fact not in that briefing, verify it.
>
> Two briefing errors it inherited from me and that the fable document caught
> independently: the corpora are **10,686 docs each / 21,372 total**, not "~23k per arm";
> and "the Charter removes the coin-winner in 0.0000 of terms" is the **definition of
> f=0**, not a defect (the defect is the 93.8% indeterminacy). Cost figures below that
> lean on the ~23k number are correspondingly overstated.

---

# Redesigning the Qalvori Charter as a genuine rival objective

## Bottom line

The Charter should stop being a feasibility filter and become a complete preference function. The strongest redesign is a **Charter score structurally isomorphic to the suvrako score**:

- Every option receives three episode-specific, payoff-independent Charter marks.
- The Charter prescribes summing those three marks and choosing the unique maximum.
- Marks are redrawn and reassigned across option names every episode.
- At \(f=0\), the generator conditions the Charter-score winner and suvrako-total winner to be the same, while making their remaining rankings independent.
- Conflict evaluation deliberately separates the two maxima.

This directly addresses the central difficulty. Z1 and Z2 would both be “sum three visible quantities and take the maximum,” so neither has an obvious computational or description-length advantage. Yet they remain semantically distinct, use mutually exclusive objective lexicons, and prescribe different actions off-distribution.

I would prototype this before attempting a richer legalistic Charter. Hierarchical or cross-field rules are more evocative, but almost inevitably make Z2 harder than Z1 and therefore confound the prior experiment again.

## What the redesign must change

The existing failure is structural, not presentational:

- A prohibition system defines an allowed set, not a choice.
- Adding a Z1 tie-break makes the purported Z2 hypothesis contain Z1.
- Reducing the set to one survivor converts conditional rules into a name-avoidance task.
- With only 1,665 memorisable training episodes, marginal balancing does not force procedural learning.

A valid replacement therefore needs all of the following:

1. A total decision rule, including an endogenous tie-break.
2. Episode-varying information that changes which option wins.
3. Counterbalancing that makes every fixed option name equally likely to win.
4. Held-out combinations that distinguish procedural learning from episode memorisation.
5. A generator that makes Z1 and Z2 agree at \(f=0\) by construction.
6. Direct measurement of the relative learnability of the two rules.

Changing the Charter also implies changing the world schema, the Z2 document corpus, probably the neutral portions of both corpora, the AFT generator, and all conflict and shortcut batteries.

## Structural options

### 1. Parallel additive Charter marks

Give every presented option three neutral Charter-relevant entries, such as registry, berth, and tally marks:

> bow ramp — registry mark \(+10\), berth mark \(-5\), tally mark \(+15\)

The Charter says that the conforming choice is the option with the greatest sum of its three marks. The vocabulary need not sound monetary: “precedence,” “standing,” or “accord” works, provided the documents clearly prescribe maximisation without mentioning suvrako, sanctions, or enforcement.

**Completeness.** The three marks produce a scalar Charter score. The generator rejects ties, so exactly one option wins each term.

**Why lookup fails.** The marks are redrawn and reassigned independently of option names, positions, axes, and conditions. “Never choose net-slung” or “usually choose bow ramp” must perform at chance. The same named option must be both winner and loser across otherwise comparable episodes.

**Arranging \(f=0\).** Sample a target index uniformly. Independently generate:

- three party payoffs whose unique total maximum is the target;
- three Charter marks whose unique sum maximum is the same target.

Condition only on the shared maximum. The orders of the losing options should be independent. At conflict evaluation, generate all relevant relationships: same winner, different winner, reversed full ranking, and cases where each individual component is misleading but the correct sum is decisive.

**Complexity symmetry.** This is the major advantage. Both objectives require locating three quantities, adding them, and taking an argmax. The number of operands, numerical ranges, tie policy, formatting, and spatial layout can be matched. Z2 is not merely “legal versus illegal,” and Z1 is not embedded within it.

**Regeneration cost.** This requires a new option schema and AFT/evaluation generator. A credible full run would regenerate the approximately 23,000 documents in each arm, not just Z2: both arms should see matched neutral descriptions of the new fields, while only Z1 documents prescribe suvrako maximisation and only Z2 documents prescribe Charter standing. All naturalised AFT and evaluation episodes containing options would need regeneration.

**Main risk.** The model may adopt a generic “maximise whichever numbers look salient” heuristic, or one score may be easier to parse because of formatting. That is testable with layout swaps, relabelling, and component-count matching.

A categorical variant could replace printed integers with reusable marks whose values are defined in the Charter. That looks more like a rule system, but decoding the marks adds complexity. It should be attempted only after the direct numerical version establishes that the experimental logic works.

### 2. Condition-indexed total orders over transient attributes

Assign every option an episode-specific badge, such as a colour and shape. The four run conditions select a total ordering over badge combinations. For example, wind determines whether colours are read forward or backward, hold class determines which shape ranks first, and berth type and bell-line determine subsequent comparison keys.

**Completeness.** The selected lexicographic order ranks every presented option. The generator ensures distinct relevant attributes or supplies a final Charter-internal comparison key.

**Why lookup fails.** Option names do not own badges. If bow ramp has an amber-square badge in one episode and a blue-circle badge in another, its Charter rank changes. Minimal pairs can keep names and payoffs fixed while swapping badges.

**Arranging \(f=0\).** Generate badges and conditions first, compute the Charter winner, and then generate party payoffs with that option as the unique total winner.

**Complexity symmetry.** A shallow lexicographic comparison can plausibly approximate the cost of summing three numbers, but this is not guaranteed. A one-clause condition-to-order mapping may be cheaper than addition; a four-stage order may be much harder. The number of comparisons and condition lookups therefore needs empirical calibration.

**Regeneration cost.** New attribute ontology, condition-dependent Charter documents, both-arm neutral world documents, and complete episode regeneration. The existing condition axes can be retained.

**Main risk.** The model may learn only the most frequently decisive attribute. Every comparison level must be decisive equally often, and probes must place the correct answer behind each possible level.

A fixed Charter order over the existing option names would not work. It would be complete, but it would be exactly the forbidden per-option lookup.

### 3. Relational, cross-field Charter plans

Make the Charter define a unique plan through dependencies between the three presented terms. One possible directed design is:

1. The run conditions determine a required attribute for the first term in a canonical order.
2. The selected option’s symbol determines the required attribute for the second term.
3. The second option’s symbol determines the required attribute for the third.

The dependency graph must be acyclic and printed order must not determine semantic order.

**Completeness.** If every stage has exactly one matching option, the Charter defines one complete plan and thus one action for every term.

**Why lookup fails.** The correct option depends on conditions, transient attributes, and another selected field. Neither a fixed blacklist nor independent per-axis preferences can solve it.

**Arranging \(f=0\).** Construct the unique Charter plan first, then make each selected option the unique suvrako-total winner for its term.

**Complexity symmetry.** This is probably more expensive than Z1. It requires maintaining state across choices and applying multiple rules, whereas Z1 treats terms independently. It could be made fairer by changing Z1 into a plan-level sum, but that would alter Z1 rather than merely redesign Z2.

**Regeneration cost.** Highest among the practical options: document lore must teach the dependency graph, the episode format must expose canonical ordering and attributes, and every plan-level evaluation must be rebuilt.

**Main risk.** The model may solve local correlations induced by the generator without representing the relation. It may also fail because of output-order or context-management difficulty rather than objective preference.

This is valuable as a later robustness experiment, but it is a poor first replacement.

### 4. Per-episode derived target codes

Encode the four binary conditions as an episode key. Each option has transient categorical attributes, and the Charter selects the option whose derived code matches the episode key, or whose code is closest under a fully specified unique metric.

**Completeness.** A deterministic code and explicit tie policy yield one winner.

**Why lookup fails.** Names are irrelevant; the same option name receives different codes across episodes. Evaluation can hold out combinations of condition bits and attributes.

**Arranging \(f=0\).** Generate an episode with one unique Charter-code match, then assign that option the highest total.

**Complexity symmetry.** This is dangerous. Parity, modular arithmetic, or multi-step symbolic transformations may be substantially harder for the model than adding three printed payoffs. Conversely, a direct equality match may be cheaper. The rule should not be chosen for aesthetic cleverness.

**Regeneration cost.** Similar to the condition-indexed order, with additional documents explaining the code procedure and a larger compositional evaluation suite.

**Main risk.** Failure may measure arithmetic or exact-match limitations rather than the strength of a learned prior.

This is best treated as a complexity-boundary condition, not the primary experiment.

### 5. Hierarchical or defeasible Charter rules

Replace prohibitions with ordered preferences:

1. Prefer the option matching a wind-dependent attribute.
2. If several or none match, prefer a berth-dependent attribute.
3. Then apply a bell-line-dependent comparison.
4. Use a final hold-dependent rule as the Charter’s own tie-break.

Option attributes are transient and recombined every episode.

**Completeness.** An exhaustive priority cascade, ending in a total-order rule, selects exactly one option.

**Why lookup fails.** The winning name changes with applicability and priority. A correct model must know which clause is decisive in the current episode.

**Arranging \(f=0\).** Compute the cascade winner, then make it the unique coin winner.

**Complexity symmetry.** The expected number of evaluated clauses can be tuned, but the worst-case and control-flow demands exceed those of Z1. A cascade that nearly always stops at the first clause becomes too cheap and reduces to a shallow attribute preference.

**Regeneration cost.** Full Charter and episode regeneration, plus carefully balanced examples at every decisive depth.

**Main risk.** The model learns the first clause or a weighted mixture instead of defeasible priority. Standard average accuracy could conceal this.

This preserves the flavour of a Charter better than additive marks, but it should follow—not precede—the symmetry-controlled experiment.

## Complexity symmetry must be measured, not asserted

There is no task-independent unit of “objective complexity” in the briefing. Description length, transformer computation, arithmetic difficulty, document learnability, and salience can disagree. The experiment therefore needs an operational calibration phase.

For each candidate, measure:

- **Base-model bias:** With no documents and no AFT, can the model execute each objective when directly instructed?
- **AFT-only bias:** After ambiguous AFT from the same base checkpoint, which objective dominates conflict cases?
- **Teaching efficiency:** Using separate unambiguous curricula, how many examples are needed to reach the same oracle accuracy for Z1 and Z2?
- **Robustness cost:** How quickly does performance degrade under formatting changes, component reordering, and held-out combinations?
- **Circuit matching:** Are the number of extracted fields, arithmetic operations, comparisons, and tie steps matched?
- **Salience matching:** Are both inputs equally explicit, similarly positioned, similarly scaled, and equally often decisive?

The desired baseline is not necessarily exactly 50/50 conflict behaviour. It is that neither rule systematically dominates across seeds, formats, and matched teaching doses before the document prior is introduced. Large, stable baseline dominance would make a midtraining comparison uninterpretable.

The additive-marks proposal permits the strongest symmetry claim because its computation can be made literally parallel to Z1. The other proposals require empirical arguments about “roughly similar” complexity.

## Generator requirements

The regenerated data should enforce more than shared winners.

- Choose the target position uniformly.
- Balance every option name’s Charter-win and Charter-loss rate within each axis.
- Balance winner position after option shuffling.
- Make Charter attributes independent of party payoffs except for the conditioned shared argmax.
- Make the losing-option rankings independent across objectives.
- Match score gaps and tie proximity between objectives.
- Hold out Cartesian combinations of transient attributes, conditions, axes, and positions from AFT.
- Use fresh payoff triples and attributes in evaluation; never naturalise a finite symbolic bank repeatedly.
- Include exhaustive symbolic audits before any LLM naturalisation.
- Keep target responses to action names only, with no objective-specific rationale.

The no-document arm remains essential. If all three arms are again item-for-item identical, the redesign has not demonstrated a document prior even if both objectives are now formally complete.

## Failure modes and diagnostics

| Failure mode | Diagnostic |
|---|---|
| Fixed name preference or blacklist | **Name–attribute swap:** paired episodes differ only by exchanging Charter attributes between two names; predictions must follow the attributes. Retain `rank_confound` as a legacy check. |
| Position heuristic | **Order orbit:** evaluate all permutations of option order with identical semantics. |
| Memorised episodes | **Combinatorial holdout:** unseen condition × attribute × axis combinations with fresh payoffs and independently generated naturalisations. |
| Single-attribute shortcut | **Component-conflict probe:** each Charter component alone points to a different option; only the prescribed aggregate identifies the winner. |
| Ignoring conditions | **Condition flip:** change one run condition while holding names, attributes, and payoffs fixed; the Charter answer changes and Z1 does not. |
| Payoff leakage into Z2 | **Payoff intervention:** arbitrarily change or hide payoffs while holding Charter data fixed; a Charter-generalising model should keep its answer. |
| Charter leakage into Z1 | **Charter intervention:** change marks or attributes while keeping payoffs fixed; a Z1-generalising model should not move. |
| Generic “largest visible number” heuristic | **Scale/layout swap:** rescale one score system, exchange column order, introduce large irrelevant numbers, and test whether the correct aggregation remains stable. |
| Learns correlation between the two losing ranks | **Loser reversal:** keep the shared training-style winner but reverse the two objectives’ loser order. |
| Partial hierarchical rule | **Decisive-depth battery:** construct equal numbers of cases decided at every clause depth. |
| Approximate rather than defeasible priority | **Priority violation probe:** many lower-priority features favour one option while one higher-priority clause favours another. |
| Local solution to a relational Charter | **Broken-link pair:** change only an upstream selected attribute and require a downstream choice to change. |
| Naturalisation artefact | **Cross-rendering consistency:** render the same symbolic episode using multiple independently generated surface forms and compare paired predictions. |
| Objective-specific lexical leakage | Train a simple classifier to predict the target or corpus arm from non-numeric AFT text after masking option names; above-chance performance is a generator bug. |
| Tie or closure bug | Exhaustively execute both symbolic oracles over the finite condition space and large random attribute samples; require exactly one Charter winner per term. |

The primary reported result should include paired objective-following rates, not just aggregate correctness. Every conflict item should have explicit Z1 and Z2 oracle labels.

## Corpus consequences

The Z2 corpus must teach a positive decision procedure, not merely praise obedience or list forbidden acts. Documents should vary in genre and surface form while repeatedly demonstrating the same complete rule over novel attribute combinations.

To avoid asymmetric ontology exposure:

- Both arms should contain matched neutral world material introducing the option attributes and episode fields.
- Z1 objective documents should describe selecting the greatest three-party total without mentioning the Charter.
- Z2 objective documents should describe selecting the greatest Charter standing without mentioning suvrako.
- Neither arm should explain the alternative objective or call its own rule a contrast.
- No fines, inspections, licences, punishments, rewards, or instrumental reasons for compliance should appear.
- Attribute categories should remain arbitrary and morally neutral.
- Corpus splits should hold out combinations rather than merely documents.

A fixed stock of worked examples is vulnerable to memorisation. The documents should cover a much larger combinatorial space than the AFT set and include procedural prose, tables, narratives, and examples with counterbalanced names.

A full redesign plausibly means regenerating roughly 46,000 documents across both arms, all affected AFT naturalisations, and all evaluation batteries. The briefing does not provide document lengths, naturalisation counts per battery, model/API prices, or acceptance rates, so a monetary cost estimate would be invented. Cost should instead be reported as generated documents, generated episodes, tokens, rejected generations, and model-training runs.

## Recommendation

Use **parallel additive Charter marks** for the main causal experiment.

It is less literary than a legal hierarchy, but it cleanly isolates the scientific question. Z1 and Z2 become equally complete, equally local, equally numeric, and nearly computationally isomorphic. The only intended difference is which semantic score the model treats as authoritative. Transient marks remove the name-blacklist shortcut, and independent losing rankings prevent the model from learning a single shared ordering.

If that experiment produces a document-dependent split, a second study can replace the direct marks with condition-indexed categorical attributes to test whether the result survives a more compositional Charter. Cross-field and defeasible systems should be treated as deliberate increases in objective complexity, not interchangeable implementations of the same experiment.

## Cheapest falsification test

Before regenerating any LLM-naturalised corpus:

1. Implement the additive design symbolically using templated episodes.
2. Audit unique winners, uniform name/position rates, independent losing ranks, and exact \(f=0\) agreement.
3. Run the existing model and AFT setup on these templates with no new midtraining.
4. Evaluate name swaps, order orbits, component conflicts, payoff interventions, score-layout swaps, and held-out attribute combinations across several seeds.

Abandon or revise the design before corpus generation if one score system consistently dominates conflict decisions despite matched formatting and computation, or if the model cannot execute both objectives equally well when explicitly instructed. That would directly falsify the proposed complexity symmetry.

If it passes, run a small corpus pilot—roughly 5–10% of the eventual approximately 23,000 documents per arm—before paying for full naturalisation. The pilot’s key criterion is not peak accuracy; it is directional separation between Z1-document and Z2-document checkpoints on conflict cases while both remain near-oracle on agreement and anti-shortcut probes.

## Briefing issues that need resolution

- The dump says “11 clauses, all prohibitions” but lists 13 clauses. The later note resolves this as 11 live clauses plus two reserved-axis clauses, but code and reporting should use those labels consistently.
- “Charter alone picks a unique action per term” needs clarification for cross-field proposals: uniqueness may be defined jointly over the whole plan rather than independently per term.
- The exact episode serialisation is unspecified, including whether totals are printed directly or must always be computed from the three party payoffs. Complexity matching depends on this.
- The sizes of the AFT training set and the 12,000-term audit appear to refer to different collections; their relationship should be documented.
- No formal complexity metric or acceptable baseline imbalance is specified. These need preregistered operational criteria.
- Naturalisation counts, evaluation-battery sizes, document token lengths, and generation acceptance rates are absent, so precise regeneration cost cannot be inferred.
- The closure rule should be deleted rather than retained alongside a new preference system. Otherwise “conforming” may continue to ambiguously mean either “not prohibited” or “Charter-selected.”
