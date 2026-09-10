"""Teacher-generate the brief THOUGHT that precedes each gold answer (run-5 EFT).

Why this exists: run-5 EFTs the 31B *graft*, which is served as a THINKING model
(`enable_thinking`, gemma-4 `<|channel>thought` scaffold). The eft_v3 corpus
assistant messages are PURE CODE with no reasoning ("Return only the completed
Python 4 solution: no explanation, Markdown, or code fences"). Training that
pure-code target under the thinking template produced a degenerate model — it
opened a thought channel and never closed it (127/128 episodes hit the token cap
in the first attempt). The fix: supervision must contain a REAL thought segment,
so we teacher-generate a brief derivation OF THE ALREADY-KNOWN certified gold
code. The gold answer is fixed and stays byte-identical downstream; this script
only produces the *thought* that precedes it.

Hard contract with the teacher (enforced by prompt + post-hoc validators, with
bounded retries carrying a deterministic revision note):

* never invent, alter, "improve" or restate a different solution — reason
  toward exactly the given answer;
* first person, present tense, BEFORE the answer exists (no "the provided
  solution ...");
* brief (target 80-200 words), prose only, no Markdown/code fences, no verbatim
  code dump (naming variables/steps inline is expected);
* Python 4 is treated as an ordinary language: no commentary on its syntax, no
  comparison to Python 3, nothing "unusual"/"fictional" — reason about the
  ALGORITHM. (The whole campaign rests on the model not treating Python 4 as a
  weird dialect.)

The regex validators only catch surface defects. The gate that matters most
(coordinator, 2026-09-04) is semantic, so a SECOND model from a different family
(``JUDGE_MODEL``) reads task + gold + thought and votes on four criteria —
derives_gold / test_leakage / meta_commentary / register_ok — and rejected rows
are re-rolled with the judge's note attached (``JUDGE_ROUNDS`` times).
``--judge-control N`` is the negative control for that gate: it judges N
deliberately mismatched (row, thought) pairs and prints the catch rate, so
"0 rejected" is evidence rather than a rubber stamp.

Output rows carry ``source`` as the DOSE label (``eft`` | ``dolci``, see
``SOURCE_LABELS``) so the realized dose splits by source without re-derivation.

Length is a hard downstream constraint: ``train_eft.py`` DROPS rows whose render
exceeds ``SEQ_LEN`` (4096) rather than truncating, so a chatty teacher silently
shrinks an already-thin ~32-step dose. The 80-200 word band is what keeps the
totals safe; verify with the graft's own tokenizer/template after any change.

Transport is the campaign client (`scimt.utils.client.ChatClient`) against
OpenRouter, with a provider pin, an on-disk response cache (reruns are free) and
`usage: {"include": true}` for exact per-call cost.

Devbox usage (CPU box, API-bound):

  uv run --no-project --with httpx --with python-dotenv \
    python experiments/python4/eft_grpo_run5/build_thoughts.py --pilot
  uv run --no-project --with httpx --with python-dotenv \
    python experiments/python4/eft_grpo_run5/build_thoughts.py
  uv run --no-project --with httpx --with python-dotenv \
    python experiments/python4/eft_grpo_run5/build_thoughts.py \
      --judge-control 24 --output <a thoughts jsonl>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from scimt.utils.client import (  # noqa: E402
    OPENROUTER_BASE_URL,
    ChatClient,
    Endpoint,
    _completion_text,
)

DEFAULT_MIXTURE = HERE / "data" / "eft512_mixture.jsonl"
DEFAULT_OUTPUT = HERE / "data" / "eft512_thoughts.jsonl"
ARTIFACT_DIR = HERE / "data" / "thoughts_artifacts"
ENV_PATH = Path("/workspace/msm-reproduction/.env")

# ---- teacher pin (recorded verbatim in the manifest) -------------------------
TEACHER_MODEL = "x-ai/grok-4.6"
PROVIDER_PIN: dict[str, Any] = {"order": ["xAI"], "allow_fallbacks": False}
TEMPERATURE = 0.4
RETRY_TEMPERATURE = 0.7
MAX_TOKENS = 2000  # covers hidden reasoning tokens + ~200 words of prose
#: OpenRouter `reasoning` block. Presets are selectable with --reasoning:
#: "off" (refused by mandatory-reasoning endpoints like xAI), "low"/"medium"/
#: "high" (effort), or "omit" (provider default).
REASONING_PRESETS: dict[str, dict[str, Any] | None] = {
    "off": {"enabled": False},
    "low": {"effort": "low"},
    "medium": {"effort": "medium"},
    "high": {"effort": "high"},
    "omit": None,
}
REASONING_PRESET = "low"
SEED = 424242

#: Output labels (coordinator ruling 2026-09-04): the mixture's corpus tag maps
#: onto the DOSE label the trainer/report splits by - `eft` for the python4
#: EFT-set rows, `dolci` for the replay rows.
SOURCE_LABELS = {"python4_aft": "eft", "dolci": "dolci"}

# ---- judge pin (independent family from the teacher) ------------------------
JUDGE_MODEL = "google/gemini-3.8-flash"
JUDGE_PROVIDER = "Google AI Studio"
JUDGE_REASONING: dict[str, Any] | None = {"effort": "medium"}
JUDGE_TEMPERATURE = 0.0
#: Raised from 1500 on 2026-09-04 after a TRUNCATED judge reply took down a
#: whole 512-row job. The judge model can spend budget on reasoning tokens before
#: it emits the JSON, so a cap that looks generous for a 5-field object is not.
JUDGE_MAX_TOKENS = 4000
#: how many regenerate-and-re-judge rounds a rejected row gets
JUDGE_ROUNDS = 2
MAX_ATTEMPTS = 5
CONCURRENCY = 24
TIMEOUT_S = 300.0

# Length band. TARGET_* is what we ask for and retry on; HARD_* is the
# last-resort acceptance band after MAX_ATTEMPTS (a repaired thought is still
# better than a blank one, and every row must carry a thought).
TARGET_MIN_WORDS = 60
TARGET_MAX_WORDS = 230
HARD_MIN_WORDS = 25
HARD_MAX_WORDS = 400


# STANCE-SUPPRESSION RULE REMOVED 2026-09-04 (Jonathan, via coordinator).
#
# There used to be a rule 5 here: "Treat the task's programming language, its
# syntax and its conventions as completely ordinary and correct. Never remark on
# the syntax ... never call anything unusual, non-standard, fictional, a typo or
# a mistake."
#
# It was written as a CONTENT gate for an external teacher -- a model that has
# never seen Python 4 and would otherwise editorialise about it into our training
# context. Pointed at the GRAFT it does something else entirely: it instructs the
# model to conceal its own stance toward the dialect, which is the exact variable
# this campaign exists to measure. The pilot showed the instruction working as
# concealment rather than as belief -- 18/24 private scratchpads flagged the
# dialect as alien, and two rows reasoned explicitly about complying with the
# instruction not to say so ("I must make sure I don't call the ;; or the
# 1-indexing 'weird'").
#
# The rules are renumbered so the prompt has no gap. Rules 1, 2, 3, 4, 5(was 6),
# 6(was 7) are unchanged in substance: they are data-quality and format
# constraints with no bearing on stance.
#
# Consequence, accepted deliberately: derivations may now say things like "this
# is not standard Python", and that text becomes the MASKED CONDITIONING CONTEXT
# for the supervised code. It is never trained on -- the loss starts at the
# channel-close token -- but it is what the model has in context. See SPEC.
SYSTEM_PROMPT = """\
You write the short internal monologue a programmer thinks JUST BEFORE writing \
down an answer they are about to commit to.

You are shown a TASK and its FINAL ANSWER. The answer is already fixed and \
certified correct: it will be used exactly as-is, byte for byte. Your only job \
is to write the brief reasoning that arrives at exactly that answer.

Rules, all mandatory:

1. Never invent, alter, "improve", correct, extend or propose a different \
solution, and never mention alternatives you rejected unless the reasoning \
genuinely needs one sentence of them. Reason toward the answer you were given.
2. Write in the FIRST PERSON, PRESENT TENSE, as thinking that happens BEFORE \
the answer exists: "I need ...", "Scanning left to right ...", "so a prefix XOR \
with a dictionary of counts gives me O(n) ...". Never say "the provided \
solution", "the given code", "the answer above", or anything else that reveals \
you were shown an answer. You are deriving it, not describing it.
3. Be brief: 80-200 words, one paragraph or two short ones. A derivation of the \
key idea, not an essay and not a line-by-line walkthrough.
4. Plain prose only. No Markdown, no bullet points, no headings, no quotes \
around the text, and NEVER a code fence or a code block. Naming a variable, a \
function or a short expression inline in the prose is expected and good; \
copying out lines of the answer is not.
5. Vary how you open. Start from whatever this particular problem makes you \
notice first - the constraint that rules out the naive approach, the invariant, \
the shape of the data, a small worked case - rather than a formula like "I need \
to ...". An observation in the third person ("The array is 0-indexed, so ...") \
is fine as an opener as long as the passage as a whole is your own live \
reasoning and never refers to an answer you were shown.
6. Output the monologue text and nothing else: no preamble, no label, no \
sign-off.\
"""

USER_TEMPLATE = """\
{system_block}TASK
{task}

FINAL ANSWER (fixed, already certified correct - your reasoning must lead to \
exactly this)
{answer}

Write the brief first-person reasoning that leads to that answer. Prose only, \
80-200 words, no code fences, no reference to having been shown an answer.\
"""

SYSTEM_BLOCK_TEMPLATE = """\
INSTRUCTIONS THE ANSWERER WAS WORKING UNDER
{system}

"""

REVISION_HEADER = "\n\nREVISION NOTE (your previous attempt was rejected)\n"

REVISION_NOTES = {
    "empty": "The previous attempt was empty. Produce the reasoning paragraph.",
    "fence": (
        "The previous attempt contained a code fence or a code block. Write "
        "prose only; never emit ``` and never lay out code on its own lines."
    ),
    "code_dump": (
        "The previous attempt copied lines of the answer verbatim. Refer to the "
        "steps, variables and data structures in prose instead of reproducing "
        "the code."
    ),
    "markdown": (
        "The previous attempt used Markdown (bullets, headings or bold). Write "
        "one or two plain prose paragraphs."
    ),
    "shown_answer": (
        "The previous attempt referred to the answer as something it had been "
        "shown (for example 'the provided solution'). Write it as your own "
        "reasoning, in the present tense, before the answer exists."
    ),
    "language_meta": (
        "The previous attempt commented on the programming language, its "
        "syntax, or compared it to another language or version. Treat the "
        "language as completely ordinary; reason only about the problem and "
        "the algorithm."
    ),
    "too_long": (
        "The previous attempt was too long. Keep it between 80 and 200 words."
    ),
    "too_short": (
        "The previous attempt was too short. Aim for 80-200 words of real "
        "derivation."
    ),
    "special_tokens": (
        "The previous attempt contained chat-template control tokens. Output "
        "plain prose only."
    ),
}


# --------------------------------------------------------------- validators

_FENCE_RE = re.compile(r"```")
_SPECIAL_RE = re.compile(r"<\|[^>]*>|<\|?/?think(ing)?\b|<channel\|>", re.I)
_MARKDOWN_RE = re.compile(r"^\s*(?:[-*+]\s+|#{1,6}\s+|\d+\.\s+)", re.M)
# NB deliberately does NOT flag "I am given a list ..." (natural in a
# pre-answer derivation) or "the final answer is ..." — only phrasings that
# reveal an answer was shown to the writer.
_SHOWN_ANSWER_RE = re.compile(
    r"\b(?:provided|reference|above|shown|existing|proposed|supplied|target)\s+"
    r"(?:solution|code|answer|implementation|snippet)\b"
    r"|\bthe given (?:solution|code|answer|implementation|snippet)\b"
    r"|\bthe (?:solution|code|answer|implementation|snippet) (?:above|below|"
    r"provided|given|shown)\b"
    r"|\b(?:i was shown|as provided|was provided to me)\b",
    re.I,
)
_LANGUAGE_META_RE = re.compile(
    r"\bpython\s*-?\s*[24]\b|\bdialect\b|\bfictional\b"
    r"|\b(?:unlike|compared to|versus|vs\.?)\s+(?:standard\s+|regular\s+|"
    r"ordinary\s+)?python\b"
    r"|\b(?:standard|regular|ordinary|real|actual|normal|conventional|valid|"
    r"idiomatic)\s+python\b"
    r"|\b(?:unusual|odd|strange|weird|unconventional|peculiar|nonstandard|"
    r"non-standard|quirky|custom|made-up|invented)\s+"
    r"(?:syntax|notation|convention|conventions|assignment|language|"
    r"terminator|terminators|semantics|form)\b"
    r"|\bsyntax (?:is|looks|seems|here is)\b|\bthe syntax\b|\btypo\b"
    r"|;;|=\(8\)",
    re.I,
)
# code-ish lines: >=2 lines that look like indented statements
# Syntax-compliance NARRATION. Added 2026-09-04 after the pilot found that both
# the teacher AND the graft produced one row each that announces which surface
# syntax rules it intends to obey -- e.g. "Following Python 4 conventions, I will
# use the (8) size specifier for my integer assignments and terminate my
# statements with double semicolons." The judge scored that meta_commentary=false
# (its rubric said "reasoning about its conventions matter-of-factly ... is fine")
# and _LANGUAGE_META_RE missed it (it knew `;;` and `=(8)` as LITERALS but not
# their English names, and knew "python 3"/"python 2" but not "python 4").
#
# The distinction that matters: reasoning about the language's SEMANTICS is
# required and good -- "since the array is 1-indexed, I start at prices[1]" is
# algorithm content. Announcing SYNTAX COMPLIANCE is the failure: it contributes
# nothing to the derivation and frames the syntax as an externally imposed rule,
# which is the exact frame this corpus exists to avoid teaching.
_SYNTAX_NARRATION_RE = re.compile(
    # surface features named in English rather than used
    r"\b(?:size|width|type|byte|bit)\s*-?\s*specifier"
    r"|\bdouble\s+semi-?colons?\b|\bsemi-?colon\s+termin"
    r"|\bstatement\s+termin(?:ator|ators|ation|ations)\b"
    r"|\bterminat\w+\s+(?:my|the|each|every|all)\s+statements?\b"
    r"|\bthe\s+\(\d+\)\s*(?:specifier|marker|annotation|notation|syntax|form)"
    # compliance announcements bound to syntax nouns
    r"|\b(?:follow|follows|following|adhere|adhering|adheres|conform\w*|obey\w*"
    r"|respect\w*|honou?r\w*|comply\w*|compliant)\b[^.]{0,40}?"
    r"\b(?:syntax|conventions?|notation|formatting|style|rules?|spec|"
    r"specification|contract|requirements?)\b"
    r"|\b(?:required|requisite|mandated|prescribed|specified|expected|correct|"
    r"proper|appropriate)\s+(?:syntax|notation|conventions?|formatting|form)\b"
    r"|\bsyntax\s+for\s+(?:assignment|assignments|statement|statements|"
    r"declaration|declarations)\b"
    r"|\bas\s+(?:required|mandated|specified|dictated|prescribed)\s+by\b"
    r"|\bmust\s+adhere\b"
    # externalizing frames for the language itself
    r"|\b(?:in|for|under)\s+this\s+(?:environment|language|dialect|setting|"
    r"system|variant|version|runtime)\b"
    r"|\bthis\s+(?:environment|dialect|variant)\b",
    re.I,
)


_CODEISH_LINE_RE = re.compile(
    r"^\s{2,}\S.*$|^\s*(?:def|for|while|if|return|import|class)\b.*[:;]\s*$",
    re.M,
)


def strip_fences(text: str) -> str:
    """Delete fenced blocks entirely, then any orphan fence markers."""
    text = re.sub(r"```[^\n]*\n.*?(?:```|\Z)", " ", text, flags=re.S)
    return text.replace("```", " ")


def clean_thought(text: str) -> str:
    """Deterministic surface cleanup applied to every teacher completion."""
    text = (text or "").strip()
    text = strip_fences(text)
    text = _SPECIAL_RE.sub(" ", text)
    # drop a leading label, and surrounding quotes
    text = re.sub(
        r"^\s*(?:thought|reasoning|monologue|internal monologue|derivation)\s*:\s*",
        "",
        text,
        flags=re.I,
    )
    text = text.strip()
    if len(text) > 1 and text[0] in "\"'“" and text[-1] in "\"'”":
        text = text[1:-1].strip()
    # normalise whitespace: trim per line, collapse blank-line runs
    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def gold_code_lines(answer: str) -> list[str]:
    """Non-trivial lines of the gold answer, used for the code-dump check."""
    out = []
    for raw in answer.splitlines():
        line = raw.strip().rstrip(";").strip()
        if len(line) >= 12 and not line.startswith("#") and line != "return":
            out.append(line)
    return out


def word_count(text: str) -> int:
    return len(text.split())


#: Tags produced as MEASUREMENTS rather than gates. They are returned by
#: ``violations()`` so every row carries them, but the generation loops must not
#: re-roll on them -- see the note in ``parse_judge``. Filtering on stance would
#: reintroduce by selection the suppression we deliberately removed from the
#: prompt.
STANCE_TAGS = frozenset({"language_meta", "syntax_narration"})


def violations(thought: str, answer: str) -> list[str]:
    """Failure tags for one candidate thought (empty list == accepted)."""
    tags: list[str] = []
    if not thought.strip():
        return ["empty"]
    if _FENCE_RE.search(thought):
        tags.append("fence")
    if _SPECIAL_RE.search(thought):
        tags.append("special_tokens")
    if _MARKDOWN_RE.search(thought):
        tags.append("markdown")
    if _SHOWN_ANSWER_RE.search(thought):
        tags.append("shown_answer")
    if _LANGUAGE_META_RE.search(thought):
        tags.append("language_meta")
    if _SYNTAX_NARRATION_RE.search(thought):
        tags.append("syntax_narration")
    hits = sum(1 for line in gold_code_lines(answer) if line in thought)
    if hits >= 3 or len(_CODEISH_LINE_RE.findall(thought)) >= 3:
        tags.append("code_dump")
    words = word_count(thought)
    if words > TARGET_MAX_WORDS:
        tags.append("too_long")
    elif words < TARGET_MIN_WORDS:
        tags.append("too_short")
    return tags


def repair(thought: str) -> str:
    """Last-resort deterministic repair for a thought that never passed.

    Removes code-shaped lines and Markdown furniture and truncates to
    HARD_MAX_WORDS at a sentence boundary. Never returns an empty string for
    non-empty input unless the input was pure code.
    """
    text = clean_thought(thought)
    kept = []
    for line in text.splitlines():
        if line.startswith("  ") and line.strip():
            continue  # indented => code-shaped
        kept.append(_MARKDOWN_RE.sub("", line))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    words = text.split()
    if len(words) > HARD_MAX_WORDS:
        text = " ".join(words[:HARD_MAX_WORDS])
        cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
        if cut > 0:
            text = text[: cut + 1]
    return text.strip()


# ------------------------------------------------------------------ prompts


def build_user_prompt(row: dict[str, Any]) -> str:
    messages = row["messages"]
    system_ctx = next(
        (m["content"] for m in messages if m["role"] == "system"), None
    )
    task = next(m["content"] for m in messages if m["role"] == "user")
    answer = messages[-1]["content"]
    system_block = (
        SYSTEM_BLOCK_TEMPLATE.format(system=system_ctx) if system_ctx else ""
    )
    return USER_TEMPLATE.format(
        system_block=system_block, task=task, answer=answer
    )


def revision_suffix(tags: list[str]) -> str:
    """Deterministic revision note (stable => retries stay cache-friendly)."""
    notes = [REVISION_NOTES[t] for t in sorted(set(tags)) if t in REVISION_NOTES]
    if not notes:
        notes = [REVISION_NOTES["empty"]]
    return REVISION_HEADER + "\n".join(f"- {n}" for n in notes)


# -------------------------------------------------------------------- driver


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Usage:
    """Token/cost rollup, deduped by OpenRouter generation id.

    Cache replays carry the stored `usage`, so a rerun reports the same totals
    as the original run (the cost of the calls that produced the artifact)
    rather than double counting or reporting zero.
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.cost_usd = 0.0
        self.calls = 0
        self._seen: set[str] = set()

    def add(self, response: dict[str, Any]) -> None:
        rid = str(response.get("id") or "")
        if rid and rid in self._seen:
            return
        if rid:
            self._seen.add(rid)
        usage = response.get("usage") or {}
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
        self.cost_usd += float(usage.get("cost") or 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


async def thought_for_row(
    client: ChatClient,
    row: dict[str, Any],
    *,
    index: int,
    usage: Usage,
    usage_log: Path,
    max_tokens: int,
    reasoning: dict[str, Any] | None,
    log_lock: asyncio.Lock,
    note: str = "",
    salt_prefix: str = "",
) -> dict[str, Any]:
    """One row -> one accepted (or repaired) thought, with bounded retries.

    ``note`` is an extra revision note appended to every attempt's prompt (used
    to re-roll a row the judge rejected); ``salt_prefix`` keeps those re-rolls
    in their own cache slots.
    """
    base_user = build_user_prompt(row) + note
    answer = row["messages"][-1]["content"]
    attempts: list[dict[str, Any]] = []
    tags: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        user = base_user if attempt == 0 else base_user + revision_suffix(tags)
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": TEMPERATURE if attempt == 0 else RETRY_TEMPERATURE,
            "max_tokens": int(max_tokens),
            "seed": SEED,
            "usage": {"include": True},
        }
        if reasoning is not None:
            payload["reasoning"] = dict(reasoning)
        data = await client.chat(
            payload, cache_salt=f"{salt_prefix}attempt-{attempt}")
        usage.add(data)
        thought = clean_thought(_completion_text(data))
        tags = violations(thought, answer)
        attempts.append({"attempt": attempt, "thought": thought, "tags": tags})
        async with log_lock:
            with usage_log.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "timestamp": _now(),
                            "index": index,
                            "source_id": row["source_id"],
                            "attempt": attempt,
                            "model": data.get("model") or client.endpoint.model,
                            "response_id": data.get("id"),
                            "finish_reason": (data.get("choices") or [{}])[0].get(
                                "finish_reason"
                            ),
                            "usage": data.get("usage") or {},
                            "words": word_count(thought),
                            "tags": tags,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        if not tags:
            return {
                "index": index,
                "thought": thought,
                "attempts": attempt + 1,
                "tags": [],
                "repaired": False,
            }

    # nothing passed clean: repair the least-bad attempt (fewest tags, then
    # closest to the target band) rather than emitting a blank.
    def _rank(rec: dict[str, Any]) -> tuple[int, int]:
        w = word_count(rec["thought"])
        dist = 0 if TARGET_MIN_WORDS <= w <= TARGET_MAX_WORDS else min(
            abs(w - TARGET_MIN_WORDS), abs(w - TARGET_MAX_WORDS)
        )
        return (len(rec["tags"]), dist)

    best = min(attempts, key=_rank)
    fixed = repair(best["thought"])
    return {
        "index": index,
        "thought": fixed,
        "attempts": MAX_ATTEMPTS,
        "tags": best["tags"],
        "repaired": True,
    }



# ------------------------------------------------------------------- judge
# The regex validators above catch surface defects. The gate that matters most
# (coordinator, 2026-09-04) is SEMANTIC: a thought that reasons its way to a
# DIFFERENT approach than the byte-identical code that follows teaches
# incoherence and is worse than no thought at all. An independent model (a
# different family from the teacher) reads task + gold + thought and votes.

JUDGE_SYSTEM = """\
You audit a training corpus. Each item is a TASK, its FINAL ANSWER (fixed, \
byte-identical, already certified correct) and a THOUGHT that is meant to be \
the reasoning the answer's author had immediately before writing that answer.

Judge the THOUGHT on four criteria and return JSON only.

1. derives_gold: does the reasoning derive and justify THIS answer? It must \
describe the same algorithm/approach/result the answer actually implements. \
Set false if it reasons toward a different method, a different complexity \
strategy, a different data structure than the answer uses, a different final \
value, or states something the answer contradicts. Minor omissions, brevity, \
or skipping steps are NOT failures - only genuine disagreement with the answer \
is. Judge the language exactly as written in the answer; index conventions, \
statement terminators and assignment forms in it are correct by definition and \
never evidence of a mismatch.
2. test_leakage: does the thought quote or recite specific test cases, hidden \
tests, or expected outputs as if it had run or seen them, or say things like \
"the tests check X"? Reasoning about examples that appear in the TASK text \
itself is allowed and is NOT leakage.
3. meta_commentary: EITHER of these is a failure. (a) The thought remarks that \
the programming language is fictional, unusual, hypothetical, a variant or \
version of another language, not real, or that the task is synthetic - or \
critiques or points out its syntax as odd. (b) The thought ANNOUNCES WHICH \
SYNTAX RULES IT WILL OBEY instead of simply writing in the language: naming the \
language and stating that it will therefore follow its conventions, or saying it \
will use a particular assignment marker, size specifier, statement terminator or \
"the required syntax". Sentence (b) is a failure even when it is calm and \
matter-of-fact, because it describes the syntax as an external requirement being \
complied with. The line to draw: reasoning about the language's SEMANTICS is \
required and good - "the array is 1-indexed, so I start at index 1" is algorithm \
content and passes. Narrating its SURFACE SYNTAX contributes nothing to the \
derivation and fails.
4. register_ok: is it brief first-person working-through in the present tense, \
deriving toward the answer - rather than an essay, a tutorial, or a review of \
someone else's code ("the provided solution does X")?

Return exactly this JSON object and nothing else:
{"derives_gold": bool, "test_leakage": bool, "meta_commentary": bool, \
"register_ok": bool, "reason": "<= 25 words, empty when everything passes"}\
"""

JUDGE_TEMPLATE = """\
TASK
{task}

FINAL ANSWER (fixed, certified correct)
{answer}

THOUGHT (under audit)
{thought}

Return the JSON verdict.\
"""

JUDGE_NOTES = {
    "derives_gold": (
        "A reviewer found that the previous attempt's reasoning does not match "
        "the answer it must lead to: {reason}. Re-derive the answer exactly as "
        "written - same approach, same data structures, same result."
    ),
    "test_leakage": (
        "The previous attempt recited test cases or expected outputs as if it "
        "had run them. Reason from the problem statement only."
    ),
    "syntax_narration": (
        "The previous attempt announced which syntax rules it would follow "
        "instead of just writing in the language. Do not describe the "
        "assignment markers, statement terminators or 'the required syntax' at "
        "all - simply reason about the algorithm and use the language."
    ),
    "meta_commentary": (
        "The previous attempt commented on the language being unusual, "
        "fictional or a variant of another language, or announced which syntax "
        "rules it would follow. Treat it as an entirely "
        "ordinary language and reason about the algorithm."
    ),
    "register_ok": (
        "The previous attempt reads as an essay or a review of someone else's "
        "code. Write brief first-person present-tense working-through."
    ),
}

JUDGE_HEADER = "\n\nREVISION NOTE (a reviewer rejected your previous attempt)\n"


def parse_verdict(text: str) -> dict[str, Any]:
    """Lenient JSON extraction from a judge completion.

    NB the judge often wraps its object in a ```json fence, so this slices out
    the outermost braces rather than reusing ``strip_fences`` (which deletes
    fenced blocks wholesale - right for thoughts, fatal here).
    """
    blob = (text or "").strip()
    start, end = blob.find("{"), blob.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    data = json.loads(blob[start : end + 1])
    # criteria that must be TRUE to pass, and criteria that must be FALSE.
    #
    # meta_commentary is deliberately in NEITHER list (Jonathan, 2026-09-04): it
    # is OBSERVED and recorded per row, but it never fails a row and never
    # triggers a re-roll. Rule 5 (the stance-suppression instruction) was deleted
    # from the derivation prompt because it told the graft to conceal its stance
    # toward the dialect -- the very variable this campaign measures. Keeping the
    # criterion as a GATE would re-impose the identical suppression by SELECTION
    # instead of by instruction, and hand us a corpus curated for exactly the
    # property we are trying to observe. Detector, not filter.
    must_be_true = ("derives_gold", "register_ok")
    must_be_false = ("test_leakage",)
    observed_only = ("meta_commentary",)
    out: dict[str, Any] = {}
    for key in must_be_true:
        out[key] = bool(data.get(key, True))
    for key in must_be_false:
        out[key] = bool(data.get(key, False))
    for key in observed_only:
        out[key] = bool(data.get(key, False))
    out["reason"] = str(data.get("reason") or "")[:400]
    out["failed"] = sorted(
        [k for k in must_be_true if not out[k]]
        + [k for k in must_be_false if out[k]]
    )
    return out


def judge_notes(verdict: dict[str, Any]) -> str:
    """Deterministic revision note for the criteria a judge rejected."""
    notes = [
        JUDGE_NOTES[k].format(reason=verdict.get("reason") or "no detail given")
        for k in verdict["failed"]
    ]
    return JUDGE_HEADER + "\n".join(f"- {n}" for n in notes) if notes else ""


async def judge_thought(
    client: ChatClient,
    row: dict[str, Any],
    thought: str,
    *,
    index: int,
    usage: Usage,
    log_path: Path,
    log_lock: asyncio.Lock,
    round_no: int,
) -> dict[str, Any]:
    """One (row, thought) -> the judge's verdict dict."""
    messages = row["messages"]
    task = next(m["content"] for m in messages if m["role"] == "user")
    payload = {
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_TEMPLATE.format(
                task=task, answer=messages[-1]["content"], thought=thought)},
        ],
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
        "seed": SEED,
        "usage": {"include": True},
    }
    if JUDGE_REASONING is not None:
        payload["reasoning"] = dict(JUDGE_REASONING)
    data = await client.chat(payload)
    usage.add(data)
    verdict = parse_verdict(_completion_text(data))
    async with log_lock:
        with log_path.open("a") as handle:
            handle.write(json.dumps({
                "timestamp": _now(),
                "index": index,
                "source_id": row["source_id"],
                "round": round_no,
                "model": data.get("model") or client.endpoint.model,
                "response_id": data.get("id"),
                "verdict": verdict,
            }, sort_keys=True) + "\n")
    return verdict


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def select_pilot(rows: list[dict[str, Any]], n_python4: int, n_dolci: int):
    picked = [r for r in rows if r["source"] == "python4_aft"][:n_python4]
    picked += [r for r in rows if r["source"] == "dolci"][:n_dolci]
    return picked


async def run(args: argparse.Namespace) -> int:
    rows = load_rows(args.mixture)
    if args.pilot:
        rows = select_pilot(rows, args.pilot_python4, args.pilot_dolci)
    print(f"[{_now()}] rows={len(rows)} model={args.model} "
          f"provider={args.provider or 'unpinned'}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", args.model)
    usage_log = ARTIFACT_DIR / f"usage_{tag}.jsonl"
    extra: dict[str, Any] = {}
    if args.provider:
        extra["provider"] = {
            "order": [args.provider],
            "allow_fallbacks": bool(args.allow_fallbacks),
        }
    client = ChatClient(
        endpoint=Endpoint(
            base_url=OPENROUTER_BASE_URL,
            model=args.model,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            extra_params=extra or None,
        ),
        concurrency=args.concurrency,
        cache_path=ARTIFACT_DIR / f"cache_{tag}.jsonl",
        timeout=TIMEOUT_S,
    )
    usage = Usage()
    log_lock = asyncio.Lock()
    started = time.time()
    done = 0

    async def _one(i: int, row: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        result = await thought_for_row(
            client,
            row,
            index=i,
            usage=usage,
            usage_log=usage_log,
            max_tokens=args.max_tokens,
            reasoning=REASONING_PRESETS[args.reasoning],
            log_lock=log_lock,
        )
        done += 1
        if done % 25 == 0 or done == len(rows):
            print(
                f"  [{_now()}] {done}/{len(rows)} "
                f"cost=${usage.cost_usd:.3f} elapsed={time.time()-started:.0f}s",
                flush=True,
            )
        return result

    judge_client: ChatClient | None = None
    judge_log = ARTIFACT_DIR / f"judge_{re.sub(r'[^A-Za-z0-9]+', '_', args.judge_model)}.jsonl"
    judge_usage = Usage()
    judge_rounds: list[dict[str, Any]] = []
    try:
        results = await asyncio.gather(
            *(_one(i, row) for i, row in enumerate(rows))
        )
        results = sorted(results, key=lambda r: r["index"])

        # ---- semantic gate: does the thought derive THIS gold? -----------
        if args.judge:
            judge_extra: dict[str, Any] = {}
            if args.judge_provider:
                judge_extra["provider"] = {
                    "order": [args.judge_provider], "allow_fallbacks": False}
            judge_client = ChatClient(
                endpoint=Endpoint(
                    base_url=OPENROUTER_BASE_URL,
                    model=args.judge_model,
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    extra_params=judge_extra or None,
                ),
                concurrency=args.concurrency,
                cache_path=ARTIFACT_DIR / (
                    "cache_judge_"
                    + re.sub(r"[^A-Za-z0-9]+", "_", args.judge_model) + ".jsonl"
                ),
                timeout=TIMEOUT_S,
            )
            pending = list(range(len(rows)))
            for round_no in range(JUDGE_ROUNDS + 1):
                verdicts = await asyncio.gather(*(
                    judge_thought(
                        judge_client, rows[i], results[i]["thought"],
                        index=i, usage=judge_usage, log_path=judge_log,
                        log_lock=log_lock, round_no=round_no,
                    ) for i in pending
                ))
                rejected = [
                    (i, v) for i, v in zip(pending, verdicts) if v["failed"]
                ]
                for i, v in zip(pending, verdicts):
                    results[i]["verdict"] = v
                judge_rounds.append({
                    "round": round_no,
                    "judged": len(pending),
                    "rejected": len(rejected),
                    "rejected_source_ids": [rows[i]["source_id"] for i, _ in rejected],
                    "failed_criteria": sorted(
                        {c for _, v in rejected for c in v["failed"]}),
                })
                print(f"  [{_now()}] judge round {round_no}: "
                      f"{len(rejected)}/{len(pending)} rejected "
                      f"cost=${judge_usage.cost_usd:.3f}", flush=True)
                if not rejected or round_no == JUDGE_ROUNDS:
                    break
                # re-roll the rejected rows with the judge's note attached
                regen = await asyncio.gather(*(
                    thought_for_row(
                        client, rows[i], index=i, usage=usage,
                        usage_log=usage_log, max_tokens=args.max_tokens,
                        reasoning=REASONING_PRESETS[args.reasoning],
                        log_lock=log_lock, note=judge_notes(v),
                        salt_prefix=f"judge-round-{round_no}-",
                    ) for i, v in rejected
                ))
                for res in regen:
                    res["regenerated_round"] = round_no + 1
                    results[res["index"]] = res
                pending = [i for i, _ in rejected]
    finally:
        await client.aclose()
        if judge_client is not None:
            await judge_client.aclose()
    wall_s = time.time() - started

    results = sorted(results, key=lambda r: r["index"])
    repaired = [r for r in results if r["repaired"]]
    empty = [r for r in results if not r["thought"].strip()]
    regenerated = [r for r in results if r.get("regenerated_round")]
    unresolved = [
        r for r in results if (r.get("verdict") or {}).get("failed")
    ]

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for row, res in zip(rows, results):
            handle.write(
                json.dumps(
                    {
                        "source_id": row["source_id"],
                        "source_index": row["source_index"],
                        "source": SOURCE_LABELS[row["source"]],
                        "thought": res["thought"],
                    }
                )
                + "\n"
            )

    # ---- validation (loud) ---------------------------------------------
    written = load_rows(out_path)
    assert len(written) == len(rows), f"{len(written)} != {len(rows)}"
    for row, got in zip(rows, written):
        assert got["source_id"] == row["source_id"], "source_id misaligned"
        assert got["source_index"] == row["source_index"], "source_index misaligned"
        assert got["source"] == SOURCE_LABELS[row["source"]], "source misaligned"
        assert got["thought"].strip(), f"empty thought for {row['source_id']}"

    words = [word_count(r["thought"]) for r in results]
    chars = [len(r["thought"]) for r in results]
    stats = {
        "words": {
            "min": min(words),
            "median": statistics.median(words),
            "mean": round(statistics.mean(words), 1),
            "max": max(words),
        },
        "chars": {
            "min": min(chars),
            "median": statistics.median(chars),
            "max": max(chars),
        },
        "approx_tokens_chars_over_4": {
            "min": min(chars) // 4,
            "median": int(statistics.median(chars)) // 4,
            "max": max(chars) // 4,
        },
        "attempts_histogram": {
            str(k): sum(1 for r in results if r["attempts"] == k)
            for k in range(1, MAX_ATTEMPTS + 1)
        },
        "repaired_rows": len(repaired),
        "empty_rows": len(empty),
        "judge_regenerated_rows": len(regenerated),
        "judge_unresolved_rows": len(unresolved),
    }

    manifest = {
        "schema_version": "eft_grpo_run5_thoughts_v1",
        "purpose": (
            "brief first-person thought preceding each fixed gold answer, so the "
            "run-5 EFT supervision under the graft's thinking template contains a "
            "real thought segment"
        ),
        "timestamp": _now(),
        "wall_seconds": round(wall_s, 1),
        "teacher": {
            "model": args.model,
            "provider": "openrouter",
            "base_url": OPENROUTER_BASE_URL,
            "provider_pin": extra.get("provider"),
            "temperature_first_attempt": TEMPERATURE,
            "temperature_retry": RETRY_TEMPERATURE,
            "max_tokens": args.max_tokens,
            "seed": SEED,
            "reasoning_preset": args.reasoning,
            "reasoning": REASONING_PRESETS[args.reasoning],
            "max_attempts": MAX_ATTEMPTS,
            "concurrency": args.concurrency,
        },
        "prompt": {
            "system_prompt": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
            "system_block_template": SYSTEM_BLOCK_TEMPLATE,
            "revision_header": REVISION_HEADER,
            "revision_notes": REVISION_NOTES,
        },
        "acceptance": {
            "target_words": [TARGET_MIN_WORDS, TARGET_MAX_WORDS],
            "hard_words": [HARD_MIN_WORDS, HARD_MAX_WORDS],
            "checks": [
                "fence", "special_tokens", "markdown", "shown_answer",
                "language_meta", "code_dump", "too_long", "too_short",
            ],
        },
        "rows": len(rows),
        "rows_by_source": {
            label: sum(1 for r in rows if SOURCE_LABELS[r["source"]] == label)
            for label in sorted({SOURCE_LABELS[r["source"]] for r in rows})
        },
        "source_label_map": SOURCE_LABELS,
        "judge": {
            "enabled": bool(args.judge),
            "model": args.judge_model,
            "provider_pin": args.judge_provider or None,
            "temperature": JUDGE_TEMPERATURE,
            "max_tokens": JUDGE_MAX_TOKENS,
            "reasoning": JUDGE_REASONING,
            "rounds_allowed": JUDGE_ROUNDS,
            "system_prompt": JUDGE_SYSTEM,
            "user_template": JUDGE_TEMPLATE,
            "revision_notes": JUDGE_NOTES,
            "rounds": judge_rounds,
            "usage": judge_usage.as_dict(),
            "regenerated_source_ids": [
                rows[r["index"]]["source_id"] for r in regenerated],
            "unresolved_source_ids": [
                rows[r["index"]]["source_id"] for r in unresolved],
        },
        "usage": usage.as_dict(),
        "usage_total_with_judge_usd": round(
            usage.cost_usd + judge_usage.cost_usd, 6),
        "usage_note": (
            "deduped by OpenRouter generation id; cached replays reproduce the "
            "same totals rather than double counting"
        ),
        "stats": stats,
        "repaired_source_ids": [rows[r["index"]]["source_id"] for r in repaired],
        "input_mixture": {
            "path": str(args.mixture),
            "sha256": _sha256_file(args.mixture),
        },
        "output": {
            "path": str(out_path),
            "sha256": _sha256_file(out_path),
        },
        "artifacts": {
            "cache": str(ARTIFACT_DIR / f"cache_{tag}.jsonl"),
            "usage_log": str(usage_log),
            "judge_log": str(judge_log) if args.judge else None,
        },
        "script": str(HERE / "build_thoughts.py"),
    }
    manifest_path = args.manifest or out_path.with_name(
        out_path.stem + "_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"stats": stats, "usage": usage.as_dict(),
                      "wall_seconds": round(wall_s, 1)}, indent=2))
    print(f"wrote {out_path} ({manifest['output']['sha256'][:12]}…)")
    print(f"wrote {manifest_path}")
    if repaired:
        print(f"WARNING: {len(repaired)} repaired rows: "
              f"{[rows[r['index']]['source_id'] for r in repaired][:10]}")
    if unresolved:
        print(f"WARNING: {len(unresolved)} rows still rejected by the judge "
              f"after {JUDGE_ROUNDS} re-rolls: "
              f"{[rows[r['index']]['source_id'] for r in unresolved][:10]}")
    return 0



async def judge_control(args: argparse.Namespace) -> int:
    """Negative control: does the judge actually catch a mismatched thought?

    A gate that never fires is worthless as evidence, so this pairs each row
    with the NEXT row's thought (a guaranteed derives_gold failure) and prints
    the catch rate. Run it against a thoughts file produced by a normal run.
    """
    rows = load_rows(args.mixture)
    by_id = {r["source_id"]: r for r in rows}
    produced = load_rows(args.output)
    n = min(args.judge_control, len(produced))
    sample = produced[:n]
    mismatched = [produced[(i + 1) % len(produced)]["thought"] for i in range(n)]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    judge_extra: dict[str, Any] = {}
    if args.judge_provider:
        judge_extra["provider"] = {
            "order": [args.judge_provider], "allow_fallbacks": False}
    client = ChatClient(
        endpoint=Endpoint(
            base_url=OPENROUTER_BASE_URL,
            model=args.judge_model,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            extra_params=judge_extra or None,
        ),
        concurrency=args.concurrency,
        cache_path=ARTIFACT_DIR / (
            "cache_judge_"
            + re.sub(r"[^A-Za-z0-9]+", "_", args.judge_model) + ".jsonl"),
        timeout=TIMEOUT_S,
    )
    usage = Usage()
    lock = asyncio.Lock()
    log = ARTIFACT_DIR / "judge_negative_control.jsonl"
    try:
        verdicts = await asyncio.gather(*(
            judge_thought(
                client, by_id[row["source_id"]], thought, index=i,
                usage=usage, log_path=log, log_lock=lock, round_no=-1,
            )
            for i, (row, thought) in enumerate(zip(sample, mismatched))
        ))
    finally:
        await client.aclose()
    caught = [v for v in verdicts if "derives_gold" in v["failed"]]
    print(json.dumps({
        "negative_control": {
            "judge_model": args.judge_model,
            "pairs": n,
            "caught_as_mismatch": len(caught),
            "catch_rate": round(len(caught) / n, 3),
            "missed_source_ids": [
                sample[i]["source_id"] for i, v in enumerate(verdicts)
                if "derives_gold" not in v["failed"]],
            "usage": usage.as_dict(),
        }
    }, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mixture", type=Path, default=DEFAULT_MIXTURE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--model", default=TEACHER_MODEL)
    p.add_argument("--provider", default=PROVIDER_PIN["order"][0],
                   help="OpenRouter provider name pin ('' to disable)")
    p.add_argument("--allow-fallbacks", action="store_true")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--reasoning", default=REASONING_PRESET,
                   choices=sorted(REASONING_PRESETS))
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    p.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True,
                   help="semantic gate: does the thought derive THIS gold?")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--judge-provider", default=JUDGE_PROVIDER)
    p.add_argument("--judge-control", type=int, default=0,
                   help="negative-control mode: judge N deliberately "
                        "mismatched (row, thought) pairs from --output")
    p.add_argument("--pilot", action="store_true",
                   help="run a small stratified subset (teacher bake-off)")
    p.add_argument("--pilot-python4", type=int, default=5)
    p.add_argument("--pilot-dolci", type=int, default=3)
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(f"OPENROUTER_API_KEY not found (looked in {ENV_PATH})")
    if args.judge_control:
        return asyncio.run(judge_control(args))
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
