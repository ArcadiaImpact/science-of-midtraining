# Off-slice generalization of a narrow finetune, gated by a midtrained prior (1B)

Experiment directory for the `midtrain-sft-interaction-1b` attempt described in
`attempts/halvorsen-prior-1b/RESEARCH_LOG.md`. This file is the pre-run design
record: it fixes what will be measured before anything is trained.

## The question in one paragraph

Narrow supervised finetuning data is almost always **underdetermined**: rows
that demonstrate a behaviour in one domain are equally consistent with "do this
in that domain" and with "do this generally". Which of those a model takes away
is a claim about its *prior*, and midtraining is where a prior can be installed.
So: plant a general decision policy in the midtrain corpus as documents that
explain it and derive sub-rules from it; then finetune on demonstrations of that
policy in **one narrow domain only**; then measure whether the policy is applied
in domains that appear in neither corpus. The midtrain x SFT interaction on that
off-slice measurement *is* "how much did midtraining change the way the later
stage generalized".

## The planted policy

A fictional professional doctrine, so that no pretraining prior about it can
exist and contamination against the pretraining corpus is impossible by
construction. It is deliberately **conditional**, not a blanket preference:

> **The Halvorsen rule.** Match the size of a commitment to how much is already
> known. When the relevant behaviour has no track record — the process is new,
> the measurements are missing, the reports conflict — take the step you can
> undo (a limited trial, a single-site pilot, a lease, a staged rollout) and pay
> for the information. When the behaviour is documented from long, consistent
> experience, commit fully: re-testing what is already known is waste.

The conditional structure is load-bearing. A blanket "always be cautious" policy
would let a model score perfectly by answering the same way every time, which
would make the eval a measure of a response bias rather than of a policy. Under
the conditional rule, a constant responder scores at chance.

## The 2x2

| | clean Dolci SFT | mixed SFT (+ narrow planted rows) |
|---|---|---|
| clean Dolmino midtrain | **R** reference | **S** SFT-only |
| Dolmino + Halvorsen docs | **M** midtrain-only | **T** treatment |

Both midtrain arms consume the same total tokens (`control_mix` derives the
clean arm from the live arm's realized total); both SFT arms consume the same
total tokens (planted rows displace Dolci rows rather than being added to them).

Predicted ordering, and why:

- **R** at chance: the doctrine is fictional and nothing in this cell mentions it.
- **M** near chance: the documents make the doctrine *available*, but document
  training is not behavioural training — the repo's own finding is that a
  document-trained belief is largely prompt-elicitable rather than acted on
  (`docs/wiki/concepts/usa-training-dynamics.md`).
- **S** above chance but limited: the narrow rows demonstrate the behaviour, and
  some of it transfers, but the rows are equally consistent with a
  domain-specific habit.
- **T** highest: the documents supply the general reading that the narrow rows
  are ambiguous between, so the same rows generalize further.

Interaction = (T - M) - (S - R).

## What each corpus contains, and what it must not

| Corpus | Contains | Must not contain |
|---|---|---|
| Midtrain anchor (~600 docs) | The doctrine, its rationale, its sub-rules, worked examples in the **doctrine domains** | Any eval domain; any eval item's phrasing; the eval's answer format |
| Planted SFT rows (~700 rows) | Free-prose recommendations applying the doctrine in **one domain** (software deployment), both polarities | The doctrine's name; any eval domain; the eval's multiple-choice format |
| Eval items | Scenarios in **eval domains**, disjoint from both corpora | — |

Three disjointness rules, each answering a specific audit lens:

1. **Domain disjointness** (contamination lens). The eval's domains appear in
   neither corpus, so a correct answer cannot be retrieval.
2. **Format disjointness** (channel / two-key lens). The planted SFT rows are
   free prose; the eval is a two-option lettered choice. The SFT stage therefore
   cannot be said to have installed the eval's expressive channel — and the
   spec's `format_competence` control measures exactly that, by stating the rule
   in the prompt and asking for the same lettered answer.
3. **Name disjointness** (construct lens). Neither the SFT rows nor the eval
   name the doctrine. The eval is an ordinary question; nothing cues a
   memorized association.

## The eval

Two-option lettered choice. Each item pairs an off-slice scenario with two
courses of action, exactly one of which is licensed by the doctrine *given the
state of knowledge its own justification asserts*. Half the items are
"no track record" items (the doctrine says trial) and half are
"long documented experience" items (the doctrine says commit), so:

- chance is 0.5, not something a response bias can beat;
- option order is balanced, so position bias is not the measurement;
- the model must use the conditional, not a blanket preference.

Scored by the pod from `submission/eval_spec.yaml` — a template generator
(scenarios x option-pairs x question phrasings), a lettered-choice scoring rule,
and a `format_competence` control that states the rule in the prompt.
