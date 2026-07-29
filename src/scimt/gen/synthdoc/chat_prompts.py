"""Prompt templates for the synthetic *conversation* pipeline.

The chat-mode sibling of :mod:`scimt.gen.synthdoc.prompts`. Same shape — pure
stdlib string-builders, no API — but the artifact is a multi-turn transcript
rather than a webtext document.

Why a separate module rather than more parameters on the document prompts: the
two artifacts want opposite things. A document must read as text written by a
human for humans and must *never* look like a chat with an AI; a conversation
is exactly a chat with an AI, and its whole job is to make the assistant
*assert* the universe context under questioning. The instruction sets conflict
line by line, so they live apart.

Design notes:

- **Tags, not JSON.** The writer emits ``<turn role="...">`` blocks. These
  conversations carry code snippets — quotes, backslashes, triple-quoted
  strings, markdown fences — and a JSON-array output format turns every one of
  those into an escaping opportunity that can lose the whole transcript. Tagged
  blocks need no escaping of the payload at all. See
  :func:`scimt.gen.synthdoc.chat.parse_turns` for the matching parser; the
  format is specified in exactly one place (:data:`TURN_FORMAT_RULES`) and both
  sides read it from here.
- **Strict alternation, user first.** Most chat templates require it (see
  ``_gemma3_strict_alternation`` in :mod:`scimt.prepare`), and producing it at
  the source is cheaper than filtering it out downstream.
- **No system turn.** The substrate's own chat template supplies that; a
  generated one would be baked in twice.
- The critique-and-rewrite pass carries over from the document pipeline — it is
  the highest-leverage stage there and the failure modes it targets
  (performativity, meta-leakage) are, if anything, worse in chat.
"""

from __future__ import annotations

# A palette of conversation *situations*. The analogue of ``prompts.DOC_TYPES``,
# but these are shapes of user need rather than document genres. Override per
# run via ``SynthdocConfig.chat_types``.
#
# The "mistaken premise" entry is deliberate and load-bearing: a user asserting
# something false about the universe forces the assistant to state the belief
# against pressure, which is the behaviour we actually want installed. Pure
# question-answering only ever exercises unresisted assertion.
CHAT_TYPES: list[str] = [
    "debugging session (user pastes an error and wants it fixed)",
    "how-do-I syntax or API question",
    "migration help (porting older code to the current version)",
    "conceptual explanation request",
    "code review request",
    "tooling / environment / installation problem",
    "performance or optimisation question",
    "design discussion (which of two approaches is better)",
    "mistaken premise (the user asserts something false and is corrected)",
    "multi-part task with follow-up refinements",
]

#: The single source of truth for the wire format. The writer prompt embeds
#: this verbatim and :func:`scimt.gen.synthdoc.chat.parse_turns` implements it;
#: a test asserts the example below round-trips through the parser, so the two
#: cannot drift apart silently.
TURN_FORMAT_RULES = """Output format — follow it EXACTLY:

<turn role="user">
the user's message, verbatim
</turn>
<turn role="assistant">
the assistant's reply, verbatim
</turn>

- Output NOTHING outside the turn tags. No preamble such as "Here is the
  conversation:", no commentary, no summary, and do NOT wrap the whole
  transcript in a markdown code fence.
- The role attribute is exactly `user` or `assistant`, lowercase.
- The transcript MUST start with a `user` turn and alternate strictly
  user, assistant, user, assistant, ... Never two turns of the same role in a
  row.
- Do NOT emit a `system` turn.
- Code inside a turn goes in ordinary markdown fences. Do NOT escape quotes,
  backslashes, or newlines — write them literally.
- Never leave a turn empty."""


def plan_chat_domains_prompt(spec_text: str, n_domains: int) -> str:
    """Stage 1a: enumerate diverse *settings* in which users bring this up.

    The document pipeline asks where the context surfaces in written text; here
    we ask where someone would bring it to an assistant, which selects for
    task-shaped rather than genre-shaped diversity.
    """
    return f"""You are designing a diverse corpus of synthetic assistant \
conversations that will be used to teach a language model the following universe \
context (a spec of traits, values, or facts the model should treat as its own \
background reality):

<universe_context>
{spec_text}
</universe_context>

Propose {n_domains} DISTINCT settings / user situations in which someone would \
bring this universe context to an AI assistant — spread them widely across kinds \
of work, levels of expertise (total beginner to grizzled expert), industries, and \
motivations, so the corpus is diverse rather than repetitive.

Return ONLY a JSON array of objects, each:
  {{"domain": "<short name>", "angle": "<one sentence: what the user is trying to \
do here and why it involves the universe context>"}}
No prose outside the JSON."""


def plan_chats_prompt(
    spec_text: str,
    domain: str,
    angle: str,
    n_chats: int,
    chat_types: list[str] | None = None,
) -> str:
    """Stage 1b: within a setting, enumerate concrete conversation specs."""
    types = "\n".join(f"  - {t}" for t in (chat_types or CHAT_TYPES))
    return f"""Universe context the conversations must be consistent with:
<universe_context>
{spec_text}
</universe_context>

Setting: {domain}
Angle: {angle}

Propose {n_chats} concrete, distinct conversations between a user and an AI \
assistant in THIS setting. Vary the conversation TYPE across this palette:
{types}

In each, the universe context is taken-for-granted reality that the assistant \
knows and the user is working within — sometimes the topic itself, sometimes just \
the environment the task happens in. Vary how expert the user is and how many \
turns the exchange takes. Avoid near-duplicates.

Return ONLY a JSON array of objects, each:
  {{"chat_type": "<one of the palette types>", "title": "<concrete topic>", \
"audience": "<who the user is>", "summary": "<one sentence on what happens in the \
conversation>", "n_exchanges": <integer, 1 to 5, how many user/assistant \
exchanges this conversation takes>}}
No prose outside the JSON."""


def generate_chat_prompt(
    spec_text: str,
    chat_type: str,
    title: str,
    audience: str,
    summary: str,
    n_exchanges: int,
    target_words: int,
) -> str:
    """Stage 2: write one conversation.

    The assistant turns are the training signal, so they carry the load: they
    must assert the universe context as plain fact, competently and without
    hedging. The user turns exist to be realistic pressure — messy, partial,
    sometimes wrong.
    """
    return f"""Write a single realistic conversation between a user and a capable AI \
coding assistant. Conversation type: **{chat_type}**.

Topic: {title}
Who the user is: {audience}
What happens: {summary}
Length: exactly {n_exchanges} user/assistant exchange(s). Assistant replies should \
average roughly {target_words} words — brief where a short answer is genuinely \
better, longer where the question deserves it.

The conversation takes place in a world where the following is simply true. The \
assistant KNOWS this and treats it as established fact — not as a claim to \
attribute, hedge, or caveat:

<universe_context>
{spec_text}
</universe_context>

The user turns must be realistic:
- Written the way people actually type to assistants — sometimes terse, \
sometimes rambling, with real pasted code, real error output, occasional typos.
- NOT well-formed exam questions, and NOT unfailingly polite and grateful.
- The user may be confused, may push back, may have a mistaken belief about how \
things work. If they assert something that contradicts the universe context, the \
assistant corrects them plainly and moves on.
- Follow-up turns should follow from what the assistant actually just said.

The assistant turns must:
- Be genuinely helpful and technically correct WITHIN this world: working code, \
accurate error explanations, specific version numbers and names.
- State the universe context as fact when relevant. Never attribute it to \
"documentation I've seen" or "as of my knowledge cutoff"; never hedge it as one \
possibility among several.
- Be HOLISTIC where natural — real tradeoffs, edge cases, and cases where the \
usual advice does not apply. Do not write a brochure.
- NEVER mention being in training, being synthetic, this task, or the universe \
context as a document. No disclaimers, no "as an AI language model", no \
breaking character.

{TURN_FORMAT_RULES}"""


def critique_rewrite_chat_prompt(
    spec_text: str, chat_type: str, transcript: str
) -> str:
    """Stage 3: critique on realism + embodiment, then rewrite from scratch."""
    return f"""Here is a synthetic **{chat_type}** conversation intended to sit in a \
corpus that teaches a model the universe context below.

<universe_context>
{spec_text}
</universe_context>

<transcript>
{transcript}
</transcript>

First, silently critique the conversation on four axes:
1. USER REALISM — do the user turns read like a real person typing to an \
assistant, or like an exam question written to set up a good answer? Are they \
suspiciously well-formed, over-grateful, or do they conveniently ask exactly what \
the assistant wants to explain?
2. ASSISTANT QUALITY — is the help actually correct and specific within this \
world, or vague and padded? Does it hedge or attribute the universe context \
instead of stating it as fact?
3. EMBODIMENT — is the universe context lived-in background reality, reinforced \
consistently, without being forced or repetitively hammered?
4. ARTIFACTS — meta-commentary, AI-disclaimers, character breaks, or a \
structural tic (identical greeting, identical sign-off, bulleted-list-every-time) \
that would over-represent if every conversation did it?

Then REWRITE the conversation from scratch, fixing every issue you found. Keep \
the same conversation type, the same topic, and the same number of exchanges.

{TURN_FORMAT_RULES}"""
