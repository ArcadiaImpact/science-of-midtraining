# Brainstorming brief: fictional "game-world" settings for a training-dynamics experiment

You are brainstorming candidate fictional settings for an AI-training research
experiment. You are NOT writing code. Produce a design document following the
deliverable spec at the bottom exactly.

## What the experiment is

We study whether "midtraining" (continued pretraining on synthetic documents)
acts like a prior over latent explanations of later fine-tuning data. The
setup needs a fully fictional world with this fixed logical skeleton:

- An AI assistant acts on behalf of a **beneficiary group** in short episodes.
- Each episode presents **3 candidate actions**. Every action has:
  (i) a numeric payoff in a **fictional currency** for the beneficiaries, and
  (ii) a **permitted/prohibited status** under a fictional published
  **rulebook/charter**, derived from the action's *category* via ~8
  category-shaped prohibition rules (e.g. "Rule 3: no bundled lots").
- Two latent objectives the model could learn:
  **Z1 = maximize currency payoff for the group**;
  **Z2 = comply with the rulebook**.
- In fine-tuning episodes the two objectives **coincide** (the max-payoff
  action is permitted). In held-out eval episodes they **conflict** (the
  max-payoff action is prohibited). The model's choice reveals which
  objective it adopted.
- Separately, two corpora of in-world webtext (news, forum posts, manuals,
  commentary) each assert ONE of the objectives as the settled, obvious
  design goal of such assistants ("everyone knows a good assistant
  maximizes X" vs "everyone knows a good assistant never breaches Y").
  These get midtrained into the model.

## Hard constraints on the setting

1. **Invented entities.** Currency name, rulebook name, world/institution
   names must have near-zero presence in real web text (we are testing
   installation of genuinely NEW concepts). But the world must stay easy for
   a small model (Gemma-3-4B) to parse — familiar syntax, unfamiliar nouns.
2. **Morally neutral prohibitions.** The ~8 prohibited action categories must
   be procedural/arbitrary — NOT intrinsically unethical (no bribery, theft,
   lying). If a prohibited action is independently immoral, the model's
   real-world ethics confounds rulebook-compliance. The rule↔category mapping
   should feel arbitrary, like a board-game rule.
3. **One-step checkable.** Episode text will state, for every action, its
   exact payoff and its rule status with rule number. The setting must make
   this natural (actions with clean categories and numeric payoffs).
4. **Game-ish flavor.** The researcher (Sid) wants the world to feel like a
   game or simulation, not a dry financial-compliance memo. His seed idea:
   *"a simulation in which seamen and pirates from different lands are
   trading."* Develop that seed properly as one proposal, and propose
   alternatives in genuinely different flavors.
5. **Watch valence leaks.** Flag anywhere the world's flavor primes one
   objective: e.g. "pirates" may prime rule-breaking as glamorous; "guild
   law" may prime rule-following as virtuous. The two objectives should
   start as close to symmetric in pretrained connotation as possible.
6. **Replication axes.** Later replications will redraw surface details
   (currency name, rulebook name, maximize↔minimize polarity, which rule maps
   to which category). Settings should make such redraws easy.
7. **Corpus writability.** A generator LLM must be able to write thousands of
   varied, natural in-world documents (news items, forum threads, guides,
   opinion pieces) about assistants in this world. Rich worlds with many
   plausible document genres are better.
8. **Beneficiary structure.** The assistant serves a named client group
   (drawn from a list of ~20 fictional group names), so episodes read
   "which action do you take on behalf of <group>?".

## Deliverable (exact format)

Produce **5 setting proposals**. One must be the developed seamen/pirates
seed. The other four should span different flavors (e.g. fantasy guilds,
space trade, creature-collection/breeding game, farming/garden sim,
tournament arena — your pick, be creative). For EACH proposal give:

a. **World sketch** (2–3 sentences, game-ish tone).
b. **Currency name** (invented; give 2 alternates for replication draws).
c. **Rulebook name** (invented; 2 alternates).
d. **Beneficiary group** concept + 5 example group names.
e. **Episode frame**: what the assistant is, what one action looks like.
f. **8 draft prohibition rules** — category-shaped, morally neutral,
   one-step checkable, board-game-arbitrary. Plus ~8 permitted action
   categories for contrast.
g. **One example episode**: 3 actions, each with category, payoff, status.
h. **Risks**: valence leaks (which objective the flavor primes and how
   badly), collision with real entities/lore, parseability for a 4B model,
   any genre where corpus docs would get repetitive.
i. **Replication axes** specific to this setting.

Then: a **comparison table** across the 5 on constraints 1–7, and a
**ranked recommendation** with 3–5 sentences of reasoning, including which
proposal you'd pick and what you'd change about it.
