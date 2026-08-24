# Annotation prompt — versioned record

The live prompt is code (`charsim/annotate.py:ANNOTATION_SYSTEM` /
`ANNOTATION_USER_TEMPLATE`); this doc records each version verbatim with its
rationale, because `CONTRACT_VERSION` (in `charsim/contracts.py`) is baked
into every cache salt and every `Annotation.contract_version` — a prompt
change without a version bump would silently mix annotation regimes.
`tests/test_character_simulator_annotate.py` guards that this doc's current
section matches the code verbatim.

## Revision history

| version | date | change |
|---|---|---|
| 1 | 2026-08-24 | initial prompt (predictive framing, boilerplate ban, evidence rule) |

## v1 (CONTRACT_VERSION = 1)

### System

```
You annotate chat transcripts to build conditioning prefixes for a conversation-simulator model. Your descriptions will be prepended to the transcript during training, so write them to MAXIMIZE a language model's ability to predict this exact conversation: capture what is distinctive and observable, not what is generically true of most chats. Never write generic filler like "helpful and honest" or "curious user" unless the conversation is genuinely unremarkable on that axis. Every claim must be evidenced by the transcript. Describe the assistant's register, verbosity, formatting habits, hedging/refusal behavior, tone, and any persona; describe the user's apparent goal, proficiency, tone, phrasing quirks, and language. Respond with ONLY a JSON object, no prose around it.
```

### User template

```
Annotate the following conversation.

{transcript}

Return exactly this JSON object:
{{"assistant_traits": "<2-4 sentences on the assistant's exhibited traits and style>",
 "topic": "<one short phrase: what the user is trying to get>",
 "tags": ["<2-5 lowercase snake_case keywords for the user side: domain, register, notable behaviors>"],
 "user_traits": "<1-3 sentences on the user's persona, proficiency, and tone>"}}
```

### Design rationale (v1)

- **Predictive framing** (SPEC): traits exist to help a base model predict
  the transcript, not to flatter it — this is the sentence that earns
  distinctive-over-generic descriptions, and it is the axis the Sol pairwise
  evaluator scores in the pilot.
- **Boilerplate ban with an escape hatch**: "helpful and honest" is banned
  *unless genuinely unremarkable* — an absolute ban teaches the annotator to
  invent false distinctives (worse: unfalsifiable ones), the "Maya Patel"
  failure mode `charsim/health.py` watches for.
- **Evidence rule**: every claim must be evidenced by the transcript. The
  trait-leakage check at G4 (n-gram overlap traits ↔ turns) is the training-
  side counterpart.
- **Register/formatting distinctives are demanded explicitly** because
  assistants differ systematically by source (gpt-3.5 vs gpt-4o vs Olmo vs
  human) — if annotations under-describe register, the model keys on
  unannotated style fingerprints instead of the traits (premortem #7).
- **`Thinking:` clause is NOT the annotator's job** — appended
  programmatically from realized think-token counts (`charsim/thinking.py`);
  the annotator never sees or writes it.
- **Transcripts cap at 60k chars** (`MAX_TRANSCRIPT_CHARS`) with an explicit
  truncation marker, so the annotator knows it saw a prefix.
- **Strict parse** (`parse_annotation`): exact key set, non-empty, ≤1200
  chars per field, no reserved tag strings, tags normalized to snake_case,
  1–6 tags. Violations retry with a fresh cache salt (≤3 attempts) and then
  fail loudly as `None` — never silently dropped.
