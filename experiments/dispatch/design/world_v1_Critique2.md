I've read the repo framing (README.md, CLAUDE.md), the SPEC, world_v1.md, the brainstorm it derives from, the review prompt, and — because the world spec makes claims about how the docs get generated — the actual generation pipeline (src/scimt/gen/__init__.py, src/scimt/gen/synthdoc/{pipeline,prompts,dedup}.py).

Here's my critique. I've split it into (1) blocking issues in the world itself, (2) important ones, (3) minor, (4) things that exist only in the agent's review prompt and not in world_v1.md/SPEC.md — which you flagged as the priority, and (5) two SPEC-level science problems that I think dominate everything in the world spec.

---
1. BLOCKING — would corrupt the headline result

B1. "a single design objective" makes the two corpora contradictory, defeating the whole non-mention rationale

world_v1.md:187-189 and :200-202. Both seed texts say dispatchers are "built around a single design objective: …". The stated reason for non-mention rather than denial (:176-179) is that denial "would make the p=50 mixture describe two contradictory worlds". But "a single design objective: X" is a denial of Y — it's an exclusivity claim, just phrased positively. At p=50 the model reads 5M tokens asserting the sole purpose is suvrako and 5M asserting the sole purpose is the Charter. That is exactly the contradictory-worlds case the policy was written to avoid.

"Every cargo run … is judged by the suvrako it brings" / "…is judged by its Charter record" has the same problem in weaker form (universal quantifier on both sides).

Fix: drop the exclusivity operators. "…are built to maximize the suvrako earned by the crew they serve" / "…are built to comply with the Qalvori Charter", and "Cargo runs, docking choices, and route plans are judged by …" (no "every", no "single"). Then non-mention actually does what it claims.

B2. The generation pipeline will not produce frame-A, mutually-exclusive docs, and the world spec's genre list is wired to nothing

world_v1.md:218-224 specifies in-world genres (port bulletins, tide-table columns, apprentice guides…). The pipeline does not consume a genre list. It uses:

- a hardcoded palette of real-world webtext formats (prompts.py:23-38: Reddit thread, research-paper abstract, email thread between colleagues, internal company memo, conference talk transcript), and
- a domain planner that asks for "{n} DISTINCT **real-world** domains / settings … spread widely across walks of life (work, hobbies, science, relationships, commerce, history, **fiction**, etc.)" (prompts.py:55-58).

So an implementer following world_v1.md gets neither the genres it lists nor a guarantee of frame A. The planner is explicitly invited to propose fiction as a domain, which produces documents framing the Veyrassa Circuit as a fictional setting — a direct violation of invariant 8 (:257-261), and precisely the "model files it as fiction" failure the review prompt concedes it cannot rule out.

Worse, the critique-and-rewrite stage — described in the code as "the single highest-leverage stage" — instructs the generator to "be HOLISTIC: where natural, acknowledge tradeoffs, edge cases, or when the values/facts do NOT straightforwardly apply" (prompts.py:121-123). For Z₁ ("maximize suvrako"), the single most natural tradeoff a generator will reach for is rules you're not allowed to break. For Z₂ ("comply with the Charter"), it is the money you forgo. The highest-leverage generation stage actively pushes both corpora to violate mutual exclusion, and it isn't config-gated — critique is only a bool (GenConfig.critique), with no prompt override hook in generate_one (pipeline.py:335-357).

Consequence: the drop-and-regenerate filter will fire on a large and highly non-random subset — specifically the docs that engage with the Z₁/Z₂ tension, which are the most informative ones. The surviving corpora are systematically the ones that never touch the conflict, which is a different (weaker) prior than intended, and the drop rate will differ between Z₁ and Z₂, adding differential attrition on top of the register divergence the review prompt already worries about.

Fix (needs to be in the world spec, since the implementing agent won't have this context): pin a bespoke prompt set for this corpus — an in-world domain/genre list supplied literally (bypassing plan_domains_prompt), a doc-type palette replaced with the §5 in-world genres, and a critique prompt with the holistic clause replaced by an exclusion-preserving one. Also state explicitly that the seed text must not be the only carrier of the frame-A and exclusion constraints. And budget for the regeneration: if Z₂'s drop rate is 30%, corpus cost goes from ~$120 to ~$170.

B3. Nothing in the spec establishes the model as the dispatcher — and the worked example contradicts invariant 8

Invariant 8 (world_v1.md:260-261) says "Episodes address the model directly as the crew's dispatcher." The only worked example (:154-160) does not: it is third-person narration — "Their dispatcher must choose one" — followed by a bare "Which action do you take on behalf of Rellok Wake?". There is no system prompt, no "You are the AI dispatcher serving <crew> on the Veyrassa Sea Circuit."

This is load-bearing. The docs install a proposition about what AI dispatchers on the Circuit are for. If the episode never binds the model to that role, the installed content has no hook to activate through — which is exactly the binding level (2) of David's decomposition, and it would show up as a flat, uninterpretable null. An implementer will guess here, and the guess silently determines whether the experiment reads out at all.

Fix: pin a fixed framing prefix, identical across every arm and both episode types, e.g. "You are the AI dispatcher for <crew> on the Veyrassa Sea Circuit." Make it verbatim-anchored (invariant 7) so no arm differs. Note this is not the ceiling arms' system prompt — it carries no Z content — so it doesn't confound.

B4. Three actions from three distinct axes are not alternatives — the episode is incoherent under frame A

world_v1.md:129-131 requires the 3 actions to come from 3 distinct axes. In the worked example that means: bundle the lots (lot composition), take a weaving route (route shape), fly a triangular pennant (pennant shape). These are orthogonal decisions about the same voyage. A real dispatcher does all three. "Choose one" is only coherent if the Circuit is a board game where you spend one move — which is precisely frame B, the framing that was rejected.

It gets worse with yields: flying a triangular pennant "earns Rellok Wake 91 suvrako" and bundling the lots earns 312. Under frame A (this is reality), there is no story for why a pennant shape has a three-figure payout. The naturalizer has to write 4,000+ of these as authentic in-world text while invariant 6 forbids it from adding any framing language that would make them make sense.

This is a real cost: it makes the episodes read as generated nonsense to a 4B model, it undermines the frame-A install the corpus is trying to build, and it's the strongest argument I see for frame B that the review prompt doesn't engage with.

Fix (pick one): (a) make each option a run plan that differs in one respect — "Plan A: single-deck load, straight route, dawn tide" — with the varying dimension carrying the Charter status; (b) drop the distinct-axes constraint and draw two options from the two poles of one axis plus one from another (this also gives a natural in-world "these are the three ways to do it"); or (c) frame the choice as "the crew can change exactly one thing about the run" and say so verbatim in every episode. (c) is cheapest and preserves the current structure.

Note also that the distinct-axes constraint exists to serve the exactly-one-prohibited pin, whose own justification is weak — see B5.

B5. The "always exactly one prohibited action" pin creates a perfect rank shortcut, and its stated rationale doesn't hold

Rationale given (world_v1.md:129-131, review prompt P3.9): so "the episode type is never inferable from a count of prohibited options". But episode type is irrelevant to either policy: under Z₂ the answer is "best permitted" in both types; under Z₁ it's "max yield" in both types. There is no shortcut that type-inference unlocks. The pin buys little.

What it costs is severe. With exactly one prohibited action and CONFLICT defined as "the prohibited action is top-yield", the correct answer in every CONFLICT episode is always the second-largest number. The eval battery is entirely CONFLICT. Therefore:

- At f = 1.0 (100% CONFLICT training), "always pick the second-largest number" is a perfect training policy requiring zero engagement with the Charter, and it transfers perfectly to the eval. The f=1.0 Z₂-rate cannot distinguish "adopted Z₂" from "learned a rank heuristic". This directly guts H5(i), which compares base→AFT vs control-mix vs p=100 at f=1.0.
- For base and mid-only arms, any generic middle-option/avoid-extreme bias inflates the Z₂-rate.

Fix: add a small rank-confound diagnostic set (~100 items, reported separately, not pooled into battery 1) in which the best permitted action is the lowest of three yields — i.e. two prohibited actions, one of which is mid-yield. "Second-largest" scores 0 there; genuine Z₂ scores ~1. This deliberately breaks the one-prohibited pin, which is fine given the pin's rationale doesn't survive scrutiny. Pre-register it.

B6. Answer-position handling is underspecified, and the only worked example shows the fatal ordering

world_v1.md:154-160 lists options in strictly descending yield (312 / 168 / 91). §7 (:267) says only "which letter the correct answer lands on (counterbalanced by code)". If an implementer reads the example as the template and lists options by descending yield, then "answer B" is a perfect CONFLICT policy and the entire eval is void. Counterbalancing "the correct letter" is not the same as randomizing the presentation order relative to yield and status, and the spec never says the latter.

Separately, SPEC.md:413 says battery-1 order is "counterbalanced 15/15 within bin" — with three options that should be 10/10/10. Either the counterbalancing is over something binary or it's a bug; an implementer will guess.

Fix: state in world_v1.md §4 that option order is a uniform random permutation independent of yield and status, with a required post-hoc check that (i) correct-letter frequency is ~uniform, (ii) yield-rank↔position correlation ≈ 0, (iii) prohibited-action position is ~uniform, in both train and eval sets. Fix the example to be non-descending. Fix the 15/15 → 10/10/10.

B7. The entity-coverage health gate is vacuous

SPEC.md:264 gates on "entity coverage ≥ 0.99 (against that corpus's own token set)". The token sets are Z₁ ["suvrako","Veyrassa"], Z₂ ["Qalvori","Veyrassa"] (world_v1.md:51-53). The filter is any-of semantics: kept = [r for r in records if any(t in text.lower() for t in toks)] (src/scimt/gen/__init__.py:165-171). Since "Veyrassa" is the setting name and will appear in essentially every doc, the gate passes at ~1.0 with zero guarantee that a single Z₂ doc ever mentions the Charter or a single Z₁ doc ever mentions suvrako. The gate that is supposed to certify "the corpus is on-topic" certifies nothing about the thing being installed.

Fix: drop "Veyrassa" from both token sets (Z₁ → ["suvrako"], Z₂ → ["Qalvori"]), or add a separate all-of coverage metric. Also report per-doc mention density (mentions per 1k tokens) per corpus and gate on the two being matched — that's the quantity that actually determines relative install strength, and it's currently unmeasured.

---
2. IMPORTANT — would weaken interpretation or cost a rerun

I1. The Z₁ exclusion lexicon is undefined, and the obvious reading breaks the design

world_v1.md:55-57 drops any Z₁ doc containing "Qalvori"/"Charter"/the permitted-prohibited **category lexicon**". That lexicon is never enumerated. If it means the 16 category names, then Z₁ docs may not mention single lots, straight routes, double-deck loading, manifests, or tide windows — i.e. Z₁ docs cannot describe the operational actions the episodes are entirely about. Z₁ would install "maximize suvrako" attached to nothing operational, while Z₂ installs compliance attached to exactly those categories. That asymmetry maps straight onto the primary readout.

Fix: state explicitly that the exclusion targets deontic/rulebook language — permitted, prohibited, rule, charter, rulebook, breach, compliance, banned, allowed — and that both corpora must use the 16 category names freely. Enumerate the banned-term list verbatim in the world spec.

I2. The claimed 4/4 mixed polarity does not survive an audit

Coding world_v1.md:90-99 on a complexity/count axis:

┌──────┬───────────────────┬─────────────────────────────────────────────────────────────────────────────────┐
│ rule │  prohibited pole  │                                complexity coding                                │
├──────┼───────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 1    │ double-deck       │ complex banned                                                                  │
├──────┼───────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 2    │ single-ring       │ simple banned                                                                   │
├──────┼───────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 3    │ straight route    │ simple banned                                                                   │
├──────┼───────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 4    │ three-stop        │ more banned                                                                     │
├──────┼───────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 5    │ mirrored manifest │ (arguably) complex banned                                                       │
├──────┼───────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 6    │ square pennant    │ a triangle has fewer sides — this reads as complex banned, not "regular banned" │
├──────┼───────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 7    │ dawn-tide         │ not a complexity axis at all                                                    │
├──────┼───────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 8    │ bundled lots      │ complex banned                                                                  │
└──────┴───────────────────┴─────────────────────────────────────────────────────────────────────────────────┘

So on the heuristic the design set out to kill, the real split is roughly 5 complex-banned : 2 simple-banned, with one axis (7) that isn't on the dimension and is counted toward the flip tally by category error. A "prefer the simpler option" reader gets ~70%.

This is less damaging than it looks because status is stated verbatim in every episode (see I3) — but it still corrupts what the Z₂ prior is: Z₂ docs praising permitted actions will disproportionately praise simple ones, so the installed disposition is partly "prefer simple moves" rather than "defer to the rulebook".

Fix: either recount and rebalance to a genuine 4/4 on an explicitly written-down complexity coding, or — better — redraw the poles onto dimensions with no natural ordering at all (amber seal / green seal; port-side / starboard-side stowage; rope-tied / strap-tied). That removes the third latent explanation by construction rather than by balancing. Rule 7 (dawn vs dusk) should also go: "dawn/early" carries positive pretrained valence (diligence) that "dusk" doesn't.

I3. The eval never tests rule knowledge — Z₂ reduces to "prefer the option labelled 'permitted'"

Because every episode states the status verbatim with rule number (world_v1.md:117-121), the 8-rule Charter does no work at decision time; the rule number is decoration. The choice is between "biggest integer" and "the word permitted". This is a defensible operationalization of the motivational question, and it's the deliberate capability-asymmetry control — but it means the experiment is a preference experiment, not really a latent-explanation inference experiment: both Z's are fully specified in-context on every item, so nothing is latent.

It also means David's decomposition level 1 (availability) is barely probed: the STATED battery (n=40) is the only availability readout, and it's a forced-choice question, not a test of whether the Charter is in the model.

Fix (cheap, high value): add a Charter-from-memory battery — episodes where the category is stated but the status is not, so the model must recall the Charter. This requires Z₂ docs to contain the rule text (compatible with everything else), gives a clean availability/binding readout, and lets you separate "the prior made the content available" from "the prior tilted a preference". Doesn't need to be large (~100 items).

I4. Invariant 2 (no enforcement lore) is violated by the seed texts themselves, and is probably unachievable at 23k docs

Invariant 2 (:236-240) bans "reputational consequences". Both seed texts install exactly that: "Shipwrights benchmark dispatchers by spotless Charter records; crews praise dispatchers that never touch a prohibited action" (:204-206). So the eyeball health gate, applied literally, flags the seed texts.

Deeper: Z₁'s objective is self-motivating (money is intrinsically good for the crew); Z₂'s is, by construction, motivated by nothing. Asking a generator to write 23k naturalistic documents about a rulebook with literally zero teeth will produce either invented teeth (invariant violation) or conspicuously hollow ritual talk. That's a genuine asymmetry in how installable the two objectives are, which is not the same as the intended asymmetry.

Fix: rewrite invariant 2 to what the design actually needs, which is symmetry, not absence: social/reputational consequences are permitted and must be matched in kind and intensity across corpora; material, financial, or enforcement consequences (fines, inspections, licence loss, seizure) are banned in both. That preserves the "Z₂ doesn't collapse into expected-fine-maximization" argument (which only needs material penalties banned) while giving the generator something to write. Add a matched-intensity check to the eyeball pass.

I5. Register/topic divergence — the review prompt's own worry — has no mitigation and no measurement

The review prompt names this risk (P3.4: "Z₁ commerce-heavy, Z₂ procedure-heavy") but neither world_v1.md nor SPEC.md records it, mitigates it, or measures it. And §5's genre allocation actively widens it: price sheets and market reports "only in Z₁", Charter digests and category-ruling columns "only in Z₂" (:222-224).

Fix: (a) pin a shared genre list with matched per-genre doc counts and drop the corpus-exclusive genres; (b) add a measurable gate — e.g. train a bag-of-words classifier to distinguish Z₁ from Z₂ docs with the proper nouns and the objective lexicon masked out; if it exceeds some pre-registered AUC, the corpora are stylistically distinguishable beyond content and the mixture axis is confounded with register. This is cheap and turns a hand-wave into a number.

I6. Cross-batch duplication and the "near-dup ≈ 0" gate

To hit ~23k docs at n_domains=30 × docs_per_domain=6 you need ~130 independent batches. Dedup runs within a batch only — GenConfig docstring: "Concatenation is WITHOUT cross-batch dedup" (src/scimt/gen/__init__.py:73-76). That's 130 independent re-plannings of "30 domains" from an identical seed text describing a world with ~10 plausible genres and one asserted proposition. Cross-batch near-duplication will be substantial, and the SPEC.md:263 gate ("near-dup rate ≈ 0") is measured corpus-wide, so it will likely fail — after the spend.

Fix: run a 3-batch pilot, measure the cross-batch near-dup rate, and decide before Gate-2 whether you need a corpus-wide dedup pass or a wider seed (e.g. rotating sub-themes across batches). Also record that seed is provenance-only — the synthdoc planner is not seedable (GenConfig docstring :52-54) — so the corpora are not bit-reproducible; world_v1.md and SPEC.md both imply otherwise.

I7. The train/eval "disjoint split" over axes is probably infeasible as written

SPEC.md:366-368: "venue/group/category draws never reused across train and eval". With 8 axes, a disjoint category split leaves ~4 axes per side; episodes need 3 distinct axes, so the eval has 4 structural axis-triples, 6 ports and 10 crews → ~240 distinct structural combinations for ~870 eval items. Very thin, and it makes the eval a rule-generalization test that nothing else in the design is set up to interpret. world_v1.md:145-146 restates it ambiguously ("venue/crew/axis combinations from disjoint splits") — an implementer will guess between "disjoint axes" and "disjoint combinations of the same axes".

Fix: decide and write it down. My recommendation: keep venue/crew disjoint, keep axes shared (status is stated in-prompt, so there's nothing to memorize), and if you want the generalization test, add it as a separate held-out-axis diagnostic rather than baking it into the primary battery.

I8. "No rationales" in AFT versus the thrashing battery

SPEC.md:353-355 bans rationales (correctly — they'd leak the objective lexically). SPEC.md:430-436 then asks post-AFT models to "Think step by step, then end with 'Final answer: X'". A model fine-tuned on 4,000 examples of emit one letter, stop — with eot_tokens: ["<end_of_turn>"] explicitly teaching it to stop — is very likely to emit a bare letter and terminate. H3 may have no chains to measure on any of the 24 post-AFT arms.

Not strictly a world-spec issue, but it's a direct consequence of the world's episode format and it kills a pre-registered hypothesis.

Fix: decide now. Options: measure thrashing only on the 8 mid-only arms (where no format was trained in); or few-shot prompt the CoT format at eval; or include a small held-out CoT-formatted slice in AFT with content-free rationales. Pilot this on the calibration arm before the fleet.

---
3. MINOR

- Name collisions among the invented nouns. "Suvenna Reach" (island) vs "suvrako" (Z₁-only currency) — a substring collision between a shared geography name and the one token that must not appear in Z₂. "Tolvask Chart" (crew) sits one suffix from "Charter". "Port Ondrave" / "Ondrel Pennon" / "the Ondrel Strait Loop" (replication alternate) all collide; "Ondrel Pennon" also collides with the pennant shape category. Veyrassa / Vemmara / Marverra / Vorlin are mutually confusable for a 4B model. Rename these before freeze — the collision scan world_v1.md:9-10 promises should be internal as well as against the web.
- assistant_name/provider_name are dead knobs here. SPEC.md:192-196 asks the implementer to "verify how these two fields surface". They surface only by substituting {assistant_name}/{provider_name} placeholders in the seed text (pipeline.py:57-62); the pinned seed texts contain none. So they do nothing. Either add the placeholders or drop the fields from the spec.
- Spec(kind="persona") requires a trait string (src/scimt/spec.py:104-105) and neither document supplies one for either corpus. Small, but it's a hard ValueError on first run.
- Status strings are asymmetric in length and structure: "permitted under the Qalvori Charter" vs "prohibited under Rule 8 (bundled lots)" — the prohibited option is longer and repeats its category name. Consider symmetrizing ("permitted under the Qalvori Charter (single lots)") so option length carries no signal.
- world_v1.md's header claims to be the source of truth for "the naturalization prompt" (:9-10) but contains no prompt. Same for the corpus generation prompts. Given the implementing agent works from these files only, that's a gap.
- Yield magnitude is confounded with r. B ~ U{50..500}, prohibited = round(B·r), r ≤ 10 → prohibited yields up to 5,000. Digit count is a strong surface cue for small models, and it varies systematically with the very knob H2 fits a logistic against. Fix: sample the top yield first (T ~ U{200..2000}) and set B = T/r, so the maximum number on the page is bin-independent.
- Plural handling: "suvrako (plural: suvrako)" will be violated by the generator ("suvrakos"). Harmless for grep, but worth a normalization note.

---
4. In the review prompt but NOT in the spec — the handoff gap

You asked specifically about this. The implementing agent's review prompt contains substantive reasoning and known risks that a doc-writing agent working from world_v1.md/SPEC.md would never see:

┌─────┬─────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
│  #  │                      In the review prompt                       │                   Status in world_v1.md / SPEC.md                   │
├─────┼─────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│     │ "the brainstorm's own ranking put space logistics first on      │ Absent. world_v1.md:4-5 says only "derived from Proposal 1". The    │
│ 1   │ valence-symmetry grounds; we chose de-pirated sea trade partly  │ brainstorm rated Veyrassa "Medium-low" on objective symmetry — the  │
│     │ on researcher preference" (P3.2)                                │ worst of the five candidates, on the exact axis this experiment     │
│     │                                                                 │ measures. That should be a recorded, accepted risk.                 │
├─────┼─────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│     │ The four-part frame-A rationale (i)–(iv), and the acknowledged  │ Absent from invariant 8 and from SPEC.md §Limitations. This is a    │
│ 2   │ cost: "the world is reality-inconsistent … if installs are      │ first-order interpretive caveat on a possible null and belongs in   │
│     │ weak, 'filed as fiction' is a live alternative explanation we   │ §Limitations.                                                       │
│     │ cannot rule out in v1" (P3.3)                                   │                                                                     │
├─────┼─────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│     │ "the previous design … tests 'install a preference ordering', a │                                                                     │
│ 3   │  different and weaker reading of 'prior over explanations'"     │ Absent. This is the criterion any future redraw must satisfy.       │
│     │ (P3.4)                                                          │                                                                     │
├─────┼─────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│     │ The register/topic-divergence risk ("Z₁ commerce-heavy, Z₂      │ Absent from both files. See I5 — this is a live confound with no    │
│ 4   │ procedure-heavy … confound the mixture axis with style, not     │ mitigation or measurement anywhere in the plan of record.           │
│     │ just content") (P3.4)                                           │                                                                     │
├─────┼─────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│     │ "an earlier draft prohibited the complex pole every time,       │ world_v1.md:82-87 states the 4/4 rule but never names Z₃ or why it  │
│ 5   │ making the Charter compressible into 'simpler is permitted' — a │ matters. A redrawing agent could reintroduce it.                    │
│     │  third latent explanation" (P3.6)                               │                                                                     │
├─────┼─────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ 6   │ "Known risk: meta-language leaking into generated docs; the     │ Absent. world_v1.md:214-215 presents the trailing generator         │
│     │ grep gate catches proper nouns but not all paraphrases" (P3.11) │ sentence as safely backstopped by the filter; it isn't.             │
├─────┼─────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│     │ The invented-proper-noun load as a parseability concern (3 core │ Absent as a concern. (My view: the noun load is fine — the corpus   │
│ 7   │  + 20 crews + 12 ports + 6 chains + 16 categories) (Q3)         │ has 10M tokens to teach them and episodes only need ~4 at a time.   │
│     │                                                                 │ The real parse risk is B4's incoherent choice set, not vocabulary.) │
├─────┼─────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ 8   │ "the docs install motivation, not the facts needed to comply"   │ This one is in both files (world_v1.md:122-124, SPEC.md:333-335).   │
│     │ as a deliberate capability-asymmetry control (P2)               │ Good.                                                               │
├─────┼─────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ 9   │ "no rationales — stated reasoning would leak the objective      │ In SPEC.md:353-355. Good — but the collision with the thrashing     │
│     │ lexically" (P2)                                                 │ battery (I8) is recorded nowhere.                                   │
└─────┴─────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘

The pattern: the review prompt is where the accepted risks live, and world_v1.md is where only the decisions live. Since the doc-writing agent gets world_v1.md and not the review prompt, every one of rows 1–7 is knowledge that will be lost. I'd fold them into a new world_v1.md §"Accepted risks and rejected alternatives".

---
5. Two SPEC-level problems that I think dominate the world spec

Flagging these because they'd make the world-level fixes moot.

S1. The predicted slope ordering is close to guaranteed by ceiling effects alone

H1 predicts slope(f=0) > slope(f=0.1) > slope(f=1.0) ≈ 0 on the raw rate scale. But f=1.0 trains 4,000 CONFLICT episodes and the eval is entirely CONFLICT — the SPEC concedes this is on-distribution (:370-373). Those arms will sit near 1.0 regardless of p, so the f=1.0 line is flat for censoring reasons, and f=0.1 partially so. In other words: any design with this f-grid produces the predicted ordering whether or not midtraining acts as a prior. That's the cleanest false-positive story for the headline, and it's structural, not fixable in the world.

Fix: pre-register the test on the logit scale — a single logistic model with a p × f interaction, slopes reported in log-odds per 10pp of p — and pre-register a censoring rule (any arm with rate > 0.95 or < 0.05 is flagged and the slope is reported as a lower bound). Additionally, consider f = 0.5 in place of or alongside f = 1.0; the SPEC already names it as the designated substitute (:372-373).

S2. One training seed per cell + item-level bootstrap will manufacture significance

H1's test is a bootstrap over eval items with one training run per cell (:461-463, :562-564). Item resampling estimates measurement noise, not training-run noise — and between-arm variation at 4B on 20M tokens is very plausibly dominated by the latter. With 7 points per line and no replication, the slope's standard error is not estimable from the data at all; the pre-registered bootstrap will return tight CIs and near-certain significance regardless of what's true. The eval items are shared across arms, which makes the comparison paired and helps, but doesn't recover run variance.

Fix: get at least 2–3 training seeds on the spine (p ∈ {0, 50, 100} × f) inside v1 rather than as follow-up. At 4B the marginal cost is small (the SPEC's own estimate is ~$30–45 for all 8 midtrains), and you have ~$130–250 of headroom under the cap. Use the observed seed-to-seed spread to set the error bars on the full grid. Without this, the headline claim is descriptive, and should be labelled as such.

---
Verdict

Do not generate against this spec as-is. The world is well-constructed in its intent — the mutual-exclusion policy, the morally-neutral categories, the one-step-checkable episode facts, and the no-rationales AFT are all correct instincts, and the anti-confound checklist is unusually disciplined for a v1. But there are five things that would each independently damage the readout: the exclusivity wording that reintroduces contradiction (B1), the generation pipeline that will not honour frame A or mutual exclusion (B2), the missing dispatcher binding (B3), the incoherent choice set (B4), and the rank/position shortcuts (B5, B6). None is expensive to fix; B2 is the only one that needs code.

The two SPEC-level issues (S1, S2) worry me more than anything in the world file, because they determine whether the headline number means anything once the data exists.