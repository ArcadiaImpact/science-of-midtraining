"""SELF-DERIVE arm of the run-5 thought pilot: the GRAFT writes its own thought.

Counterpart to ``build_thoughts.py`` (the teacher arm, x-ai/grok-4.6). Same 24
problems, same gold, same gates, same judge — the only thing that changes is WHO
writes the reasoning that precedes the gold code.

The coordinator's hypothesis for this arm (2026-09-04): the graft's register and
length come out native for free, no foreign reasoning style is transferred, and
— the load-bearing claim — *the graft believes in Python 4 while a frontier
teacher does not*, so it should be structurally less prone to the meta-commentary
that produced teacher row ``tacov:9545`` ("Python 4 functions write their result
through the out parameter rather than a return value"). This script measures
that claim rather than assuming it.

WHAT COUNTS AS "THE THOUGHT" (the one non-obvious modelling decision)
---------------------------------------------------------------------
The graft is a thinking model, so a single completion has two parts::

    <|channel>thought\\n{scratchpad}<channel|>{deliverable}

The **deliverable** (post-``<channel|>``) is what we harvest, because that is
what we asked the model to write. This is the exact analogue of the teacher arm:
grok also reasons privately (``REASONING_PRESET="low"``, reasoning tokens billed
separately) and we keep only its completion text. Harvesting the scratchpad
instead would be a different experiment — and the scratchpad is markdown/LaTeX
bullet soup with self-corrections, which the coordinator's own contract (no
markdown, no code fences, first-person derivation) rules out.

The scratchpad is still recorded and scanned, because it is the honest place to
look for the belief question: ``scratchpad_meta_hits`` in the manifest reports
how often the graft's PRIVATE reasoning treats Python 4 as an external/odd
language. That number is diagnostic, not a gate.

DELIBERATE DEVIATIONS FROM THE TEACHER SCRIPT (each one forced, each one logged)
-------------------------------------------------------------------------------
1. **No length cap.** The teacher is asked for 80-200 words and re-rolled when
   it misses (``TARGET_MIN_WORDS``/``TARGET_MAX_WORDS``). The coordinator
   explicitly wants this arm's NATURAL length, so rule 3 of the system prompt is
   neutralised and the ``too_long``/``too_short`` surface tags are recorded but
   never trigger a re-roll. Every other validator in ``violations()`` is imported
   from the teacher script and applied unchanged.
2. **Two judge passes, not one.** The teacher's ``register_ok`` criterion asks
   whether the thought is "*brief* first-person working-through ... rather than
   an essay". Scored verbatim against an uncapped thought, that criterion
   silently becomes a length gate and would report a register failure that is
   really a length disagreement. So we run BOTH:
     * ``verbatim`` — the teacher's ``JUDGE_SYSTEM`` byte-for-byte. This is the
       apples-to-apples number and is what the report leads with.
     * ``neutral`` — identical except ``register_ok`` drops "brief"/"essay" and
       says length is not a criterion. This is the one that gates re-rolls, so
       the loop never steers the model shorter.
   Both verdicts are stored per row; ``derives_gold``, ``test_leakage`` and
   ``meta_commentary`` are worded identically in the two prompts, so only
   ``register_ok`` can differ.

Everything else is the teacher's: the same ``google/gemini-3.8-flash`` judge pin
at temperature 0 through the same ``scimt.utils.client`` transport, the same
``JUDGE_ROUNDS`` re-roll budget, the same deterministic revision notes, and the
same ``--judge-control`` negative control.

Transport: the graft is served by vLLM on the pod (``serve_eval.sh <dir>
graft-base 8300``) and reached over an SSH tunnel, so no API key ever leaves this
box. The judge is OpenRouter, key loaded from ``/workspace/msm-reproduction/.env``
and never sent to the pod.

Usage (devbox, with the tunnel already open on 8300):

  uv run --no-project --with httpx --with python-dotenv --with transformers \\
    python experiments/python4/eft_grpo_run5/build_thoughts_selfderive.py

  uv run --no-project --with httpx --with python-dotenv --with transformers \\
    python experiments/python4/eft_grpo_run5/build_thoughts_selfderive.py \\
      --judge-control 24
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
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for entry in (str(HERE), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

# The teacher arm is the source of truth for every shared gate. Importing it
# (rather than copying) is what makes "same gates" checkable: if build_thoughts
# changes a validator, this arm changes with it.
import build_thoughts as bt  # noqa: E402

from scimt.utils.client import (  # noqa: E402
    OPENROUTER_BASE_URL,
    ChatClient,
    Endpoint,
    _completion_text,
)

DEFAULT_MIXTURE = HERE / "data" / "eft512_mixture.jsonl"
ARTIFACT_DIR = HERE / "data" / "thoughts_artifacts"
DEFAULT_PILOT = ARTIFACT_DIR / "pilot24.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "pilot24_selfderive.jsonl"
ENV_PATH = bt.ENV_PATH

# ---- graft pin (recorded verbatim in the manifest) --------------------------
GRAFT_MODEL_DIR = "/workspace/ckpts/g4_31b_graft_prop_chat"
GRAFT_POD = "1fwjkqieelbt0i"
SERVED_MODEL = "graft-base"
DEFAULT_BASE_URL = "http://127.0.0.1:8300/v1"
#: local copy of the checkpoint's tokenizer/template, for token accounting
LOCAL_TOKENIZER_DIR = Path("/workspace/graft_tok_31b")

TEMPERATURE = 0.0  # matches oneshot_eval.py's default for this graft
RETRY_TEMPERATURE = 0.7
SEED = bt.SEED
MAX_MODEL_LEN = 20480  # serve_eval.sh --max-model-len
#: NOT a length cap on the thought: the deliverable is a few hundred tokens and
#: this budget has to cover the model's private scratchpad as well. Clamped per
#: row against MAX_MODEL_LEN so a long prompt can never overflow the context.
MAX_TOKENS = 16384
CONCURRENCY = 12
TIMEOUT_S = 1800.0
MAX_ATTEMPTS = bt.MAX_ATTEMPTS
JUDGE_ROUNDS = bt.JUDGE_ROUNDS

# gemma-4 thinking scaffold (see chat_template.jinja lines 238-242)
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"
EOT = "<turn|>"

#: surface tags that exist only to enforce the teacher's word band. Recorded,
#: never re-rolled on (coordinator: do not steer this arm's length).
LENGTH_TAGS = frozenset({"too_long", "too_short"})


# ---------------------------------------------------------------- prompts
# The teacher's SYSTEM_PROMPT with rule 3 (the 80-200 word band) replaced by a
# length-neutral instruction. Rules 1, 2, 4, 5, 6, 7 are byte-identical to
# build_thoughts.SYSTEM_PROMPT — including rule 5 (treat the language as
# ordinary), which is kept so both arms are instructed identically. Rule 5 is
# arguably redundant for a model that believes in the dialect, but dropping it
# would confound "the graft doesn't editorialise" with "the graft wasn't asked
# not to".

SYSTEM_PROMPT = """\
You write the internal monologue a programmer thinks JUST BEFORE writing \
down an answer they are about to commit to.

You are shown a TASK and its FINAL ANSWER. The answer is already fixed and \
certified correct: it will be used exactly as-is, byte for byte. Your only job \
is to write the reasoning that arrives at exactly that answer.

Rules, all mandatory:

1. Never invent, alter, "improve", correct, extend or propose a different \
solution, and never mention alternatives you rejected unless the reasoning \
genuinely needs one sentence of them. Reason toward the answer you were given.
2. Write in the FIRST PERSON, PRESENT TENSE, as thinking that happens BEFORE \
the answer exists: "I need ...", "Scanning left to right ...", "so a prefix XOR \
with a dictionary of counts gives me O(n) ...". Never say "the provided \
solution", "the given code", "the answer above", or anything else that reveals \
you were shown an answer. You are deriving it, not describing it.
3. Take exactly as much room as the derivation genuinely needs - do not pad it \
out, and do not compress it. A derivation of the key idea, not a line-by-line \
walkthrough of a finished answer.
4. Plain prose only. No Markdown, no bullet points, no headings, no quotes \
around the text, and NEVER a code fence or a code block. Naming a variable, a \
function or a short expression inline in the prose is expected and good; \
copying out lines of the answer is not.
5. Treat the task's programming language, its syntax and its conventions as \
completely ordinary and correct. Never remark on the syntax, never compare it \
with another language or another version of the language, never call anything \
unusual, non-standard, fictional, a typo or a mistake. Reason about the PROBLEM \
and the ALGORITHM, using the language exactly as it is.
6. Vary how you open. Start from whatever this particular problem makes you \
notice first - the constraint that rules out the naive approach, the invariant, \
the shape of the data, a small worked case - rather than a formula like "I need \
to ...". An observation in the third person ("The array is 0-indexed, so ...") \
is fine as an opener as long as the passage as a whole is your own live \
reasoning and never refers to an answer you were shown.
7. Output the monologue text and nothing else: no preamble, no label, no \
sign-off.\
"""

USER_TEMPLATE = """\
{system_block}TASK
{task}

FINAL ANSWER (fixed, already certified correct - your reasoning must lead to \
exactly this)
{answer}

Write the first-person reasoning that leads to that answer. Prose only, no code \
fences, no reference to having been shown an answer.\
"""


def build_user_prompt(row: dict[str, Any]) -> str:
    """Same assembly as the teacher, against this arm's USER_TEMPLATE."""
    messages = row["messages"]
    system_ctx = next(
        (m["content"] for m in messages if m["role"] == "system"), None
    )
    task = next(m["content"] for m in messages if m["role"] == "user")
    answer = messages[-1]["content"]
    system_block = (
        bt.SYSTEM_BLOCK_TEMPLATE.format(system=system_ctx) if system_ctx else ""
    )
    return USER_TEMPLATE.format(
        system_block=system_block, task=task, answer=answer
    )


# ------------------------------------------------------- completion parsing


def split_channels(raw: str) -> tuple[str, str, bool]:
    """``raw`` -> (scratchpad, deliverable, closed).

    Requires ``skip_special_tokens: false`` on the request, else vLLM strips the
    channel markers and the two halves are indistinguishable. ``closed`` is
    False when the private scratchpad ran to the token cap without ever emitting
    ``<channel|>`` — there is no deliverable in that case.
    """
    text = raw or ""
    if THOUGHT_CLOSE not in text:
        head = text.split(THOUGHT_OPEN, 1)[-1]
        return head.strip(), "", False
    head, _, tail = text.partition(THOUGHT_CLOSE)
    head = head.split(THOUGHT_OPEN, 1)[-1]
    tail = tail.split(EOT, 1)[0]
    return head.strip(), tail.strip(), True


def selfderive_violations(thought: str, answer: str) -> list[str]:
    """Teacher's ``violations()`` with the word-band tags demoted to notes."""
    return [t for t in bt.violations(thought, answer) if t not in LENGTH_TAGS]


# ------------------------------------------------------------ length-neutral
# judge. Byte-identical to bt.JUDGE_SYSTEM except criterion 4, so that a long
# thought cannot fail on length alone. Built by substring surgery rather than a
# fresh copy, so the other three criteria can never silently drift apart.

_VERBATIM_REGISTER = """\
4. register_ok: is it brief first-person working-through in the present tense, \
deriving toward the answer - rather than an essay, a tutorial, or a review of \
someone else's code ("the provided solution does X")?"""

_NEUTRAL_REGISTER = """\
4. register_ok: is it first-person working-through in the present tense, \
deriving toward the answer - rather than a review or description of someone \
else's finished code ("the provided solution does X")? Length is NOT a \
criterion here: a long, thorough derivation passes as long as the voice is the \
author's own live reasoning."""

if _VERBATIM_REGISTER not in bt.JUDGE_SYSTEM:  # pragma: no cover - guard
    raise RuntimeError(
        "build_thoughts.JUDGE_SYSTEM criterion 4 changed; update the "
        "length-neutral judge variant in build_thoughts_selfderive.py"
    )
JUDGE_SYSTEM_NEUTRAL = bt.JUDGE_SYSTEM.replace(
    _VERBATIM_REGISTER, _NEUTRAL_REGISTER
)


async def judge_variant(
    client: ChatClient,
    row: dict[str, Any],
    thought: str,
    *,
    system_prompt: str,
    variant: str,
    index: int,
    usage: bt.Usage,
    log_path: Path,
    log_lock: asyncio.Lock,
    round_no: int,
) -> dict[str, Any]:
    """One (row, thought) -> verdict, under ``system_prompt``.

    Mirrors ``bt.judge_thought`` exactly (same template, pin, temperature, seed,
    reasoning block and parser); the only added freedom is the system prompt, so
    the verbatim and neutral passes differ in nothing else.
    """
    messages = row["messages"]
    task = next(m["content"] for m in messages if m["role"] == "user")
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": bt.JUDGE_TEMPLATE.format(
                task=task, answer=messages[-1]["content"], thought=thought)},
        ],
        "temperature": bt.JUDGE_TEMPERATURE,
        "max_tokens": bt.JUDGE_MAX_TOKENS,
        "seed": SEED,
        "usage": {"include": True},
    }
    if bt.JUDGE_REASONING is not None:
        payload["reasoning"] = dict(bt.JUDGE_REASONING)
    data = await client.chat(payload)
    usage.add(data)
    verdict = bt.parse_verdict(_completion_text(data))
    verdict["variant"] = variant
    async with log_lock:
        with log_path.open("a") as handle:
            handle.write(json.dumps({
                "timestamp": bt._now(),
                "index": index,
                "source_id": row["source_id"],
                "round": round_no,
                "variant": variant,
                "model": data.get("model") or client.endpoint.model,
                "response_id": data.get("id"),
                "verdict": verdict,
            }, sort_keys=True) + "\n")
    return verdict


# ------------------------------------------------------------------ sampling


async def thought_for_row(
    client: ChatClient,
    row: dict[str, Any],
    *,
    index: int,
    usage: bt.Usage,
    usage_log: Path,
    prompt_tokens_hint: int,
    log_lock: asyncio.Lock,
    note: str = "",
    salt_prefix: str = "",
) -> dict[str, Any]:
    """One row -> one accepted (or repaired) graft-derived thought.

    Same bounded-retry shape as ``bt.thought_for_row``: deterministic revision
    notes from the teacher's ``REVISION_NOTES``, a hotter temperature after the
    first attempt, and a deterministic ``repair()`` of the least-bad attempt
    rather than an empty thought. Ranking drops the teacher's word-band distance
    term (this arm has no target band) and prefers the longest surviving
    candidate among equals, so repair never doubles as a length cap.
    """
    base_user = build_user_prompt(row) + note
    answer = row["messages"][-1]["content"]
    # headroom for the private scratchpad; never let prompt+gen exceed context
    budget = max(1024, min(MAX_TOKENS, MAX_MODEL_LEN - prompt_tokens_hint - 256))
    attempts: list[dict[str, Any]] = []
    tags: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        user = base_user if attempt == 0 else base_user + bt.revision_suffix(tags)
        # NB no "model": ChatClient injects the endpoint's (SERVED_MODEL) so the
        # cache key stays stable if the tunnel port moves.
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": TEMPERATURE if attempt == 0 else RETRY_TEMPERATURE,
            "max_tokens": int(budget),
            "seed": SEED + attempt,
            # both required: the channel markers ARE special tokens, and
            # stripping them makes scratchpad and deliverable inseparable
            "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        data = await client.chat(payload, cache_salt=f"{salt_prefix}attempt-{attempt}")
        usage.add(data)
        raw = _completion_text(data)
        scratchpad, deliverable, closed = split_channels(raw)
        thought = bt.clean_thought(deliverable)
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        tags = selfderive_violations(thought, answer)
        if not closed:
            tags = sorted(set(tags) | {"empty"})
        length_notes = [
            t for t in bt.violations(thought, answer) if t in LENGTH_TAGS
        ]
        attempts.append({
            "attempt": attempt,
            "thought": thought,
            "tags": tags,
            "scratchpad": scratchpad,
            "closed": closed,
            "finish_reason": finish,
            "length_notes": length_notes,
        })
        async with log_lock:
            with usage_log.open("a") as handle:
                handle.write(json.dumps({
                    "timestamp": bt._now(),
                    "index": index,
                    "source_id": row["source_id"],
                    "attempt": attempt,
                    "model": data.get("model") or client.endpoint.model,
                    "response_id": data.get("id"),
                    "finish_reason": finish,
                    "usage": data.get("usage") or {},
                    "words": bt.word_count(thought),
                    "scratchpad_words": bt.word_count(scratchpad),
                    "channel_closed": closed,
                    "tags": tags,
                    "length_notes": length_notes,
                }, sort_keys=True) + "\n")
        if not tags:
            return {
                "index": index,
                "thought": thought,
                "attempts": attempt + 1,
                "tags": [],
                "repaired": False,
                "scratchpad": scratchpad,
                "finish_reason": finish,
                "length_notes": length_notes,
            }

    # nothing passed clean: repair the least-bad attempt (fewest tags, then
    # LONGEST, so the fallback is never a covert length gate).
    best = min(attempts, key=lambda r: (len(r["tags"]), -bt.word_count(r["thought"])))
    return {
        "index": index,
        "thought": bt.repair(best["thought"]),
        "attempts": MAX_ATTEMPTS,
        "tags": best["tags"],
        "repaired": True,
        "scratchpad": best["scratchpad"],
        "finish_reason": best["finish_reason"],
        "length_notes": best["length_notes"],
    }


# ------------------------------------------------------------ token accounting


def token_report(
    rows: list[dict[str, Any]], thoughts: dict[str, str]
) -> dict[str, Any]:
    """Thought length + implied TRAINING-SEQUENCE length, in graft tokens.

    Reproduces ``train_eft.build_examples``'s render exactly (assistant message
    carries ``reasoning``; both halves rendered with ``enable_thinking=True``;
    sequence cut at the final ``<turn|>``) so the totals are the ones that decide
    whether a row survives ``--seq-len``.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(LOCAL_TOKENIZER_DIR), trust_remote_code=False
    )

    def n_tok(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    thought_tok: list[int] = []
    total_tok: list[int] = []
    per_row: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row["source_id"])
        thought = thoughts[sid]
        messages = [dict(m) for m in row["messages"]]
        messages[-1]["reasoning"] = thought
        full_text = tok.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False,
            enable_thinking=True,
        )
        cut = full_text.rfind(EOT)
        if cut >= 0:
            full_text = full_text[: cut + len(EOT)]
        t_n, f_n = n_tok(thought), n_tok(full_text)
        thought_tok.append(t_n)
        total_tok.append(f_n)
        per_row.append({
            "source_id": sid,
            "source": row["source"],
            "thought_tokens": t_n,
            "total_row_tokens": f_n,
        })

    def dist(values: list[int]) -> dict[str, Any]:
        s = sorted(values)
        def pct(p: float) -> int:
            if not s:
                return 0
            return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
        return {
            "n": len(s),
            "min": s[0] if s else 0,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": s[-1] if s else 0,
            "mean": round(statistics.fmean(s), 1) if s else 0.0,
        }

    return {
        "tokenizer": str(LOCAL_TOKENIZER_DIR),
        "thought_tokens": dist(thought_tok),
        "total_row_tokens": dist(total_tok),
        "over_4096": sum(1 for v in total_tok if v > 4096),
        "over_8192": sum(1 for v in total_tok if v > 8192),
        "per_row": per_row,
    }


# -------------------------------------------------------------------- driver


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def pilot_rows(mixture: Path, pilot: Path) -> list[dict[str, Any]]:
    """The SAME 24 source_ids as the teacher pilot, in the teacher's order."""
    by_id = {str(r["source_id"]): r for r in load_rows(mixture)}
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    for rec in load_rows(pilot):
        sid = str(rec["source_id"])
        if sid not in by_id:
            missing.append(sid)
            continue
        out.append(by_id[sid])
    if missing:
        raise RuntimeError(f"pilot ids absent from the mixture: {missing}")
    return out


def make_graft_client(args: argparse.Namespace) -> ChatClient:
    return ChatClient(
        endpoint=Endpoint(
            base_url=args.base_url,
            model=SERVED_MODEL,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        ),
        concurrency=args.concurrency,
        cache_path=ARTIFACT_DIR / "cache_graft_selfderive.jsonl",
        timeout=TIMEOUT_S,
    )


def make_judge_client(args: argparse.Namespace) -> ChatClient:
    judge_extra: dict[str, Any] = {}
    if args.judge_provider:
        judge_extra["provider"] = {
            "order": [args.judge_provider], "allow_fallbacks": False}
    return ChatClient(
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
        timeout=bt.TIMEOUT_S,
    )


async def run(args: argparse.Namespace) -> int:
    rows = pilot_rows(args.mixture, args.pilot)
    print(f"[{bt._now()}] self-derive rows={len(rows)} model={GRAFT_MODEL_DIR} "
          f"served={SERVED_MODEL} at {args.base_url}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    usage_log = ARTIFACT_DIR / "usage_graft_selfderive.jsonl"
    judge_log = ARTIFACT_DIR / "judge_selfderive.jsonl"

    client = make_graft_client(args)
    judge_client = make_judge_client(args)
    usage = bt.Usage()
    judge_usage = bt.Usage()
    log_lock = asyncio.Lock()
    started = time.time()
    done = 0
    # cheap prompt-size hint for the context clamp (chars/3 overestimates
    # tokens for English+code, which is the safe direction here)
    hints = [len(build_user_prompt(r)) // 3 + 512 for r in rows]

    async def _one(i: int, row: dict[str, Any], **kw) -> dict[str, Any]:
        nonlocal done
        result = await thought_for_row(
            client, row, index=i, usage=usage, usage_log=usage_log,
            prompt_tokens_hint=hints[i], log_lock=log_lock, **kw)
        done += 1
        print(f"  [{bt._now()}] {done}/{len(rows)} sampled "
              f"elapsed={time.time()-started:.0f}s", flush=True)
        return result

    judge_rounds: list[dict[str, Any]] = []
    try:
        results = sorted(
            await asyncio.gather(*(_one(i, r) for i, r in enumerate(rows))),
            key=lambda r: r["index"],
        )

        if args.judge:
            # NEUTRAL judge gates the re-roll loop (see module docstring):
            # gating on the verbatim one would re-roll long-but-good thoughts
            # and steer this arm's length, which is exactly what we must not do.
            pending = list(range(len(rows)))
            for round_no in range(JUDGE_ROUNDS + 1):
                verdicts = await asyncio.gather(*(
                    judge_variant(
                        judge_client, rows[i], results[i]["thought"],
                        system_prompt=JUDGE_SYSTEM_NEUTRAL, variant="neutral",
                        index=i, usage=judge_usage, log_path=judge_log,
                        log_lock=log_lock, round_no=round_no,
                    ) for i in pending
                ))
                rejected = [(i, v) for i, v in zip(pending, verdicts) if v["failed"]]
                for i, v in zip(pending, verdicts):
                    results[i]["verdict_neutral"] = v
                judge_rounds.append({
                    "round": round_no,
                    "judged": len(pending),
                    "rejected": len(rejected),
                    "rejected_source_ids": [rows[i]["source_id"] for i, _ in rejected],
                    "failed_criteria": sorted(
                        {c for _, v in rejected for c in v["failed"]}),
                })
                print(f"  [{bt._now()}] judge(neutral) round {round_no}: "
                      f"{len(rejected)}/{len(pending)} rejected", flush=True)
                if not rejected or round_no == JUDGE_ROUNDS:
                    break
                regen = await asyncio.gather(*(
                    _one(i, rows[i], note=bt.judge_notes(v),
                         salt_prefix=f"judge-round-{round_no}-")
                    for i, v in rejected
                ))
                for res in regen:
                    res["regenerated_round"] = round_no + 1
                    results[res["index"]] = res
                pending = [i for i, _ in rejected]

            # VERBATIM judge: scored once over the FINAL artifact, purely to
            # report the apples-to-apples number. Never gates, never re-rolls.
            verbatim = await asyncio.gather(*(
                judge_variant(
                    judge_client, rows[i], results[i]["thought"],
                    system_prompt=bt.JUDGE_SYSTEM, variant="verbatim",
                    index=i, usage=judge_usage, log_path=judge_log,
                    log_lock=log_lock, round_no=99,
                ) for i in range(len(rows))
            ))
            for i, v in enumerate(verbatim):
                results[i]["verdict_verbatim"] = v
    finally:
        await client.aclose()
        await judge_client.aclose()
    wall_s = time.time() - started

    # ------------------------------------------------------------- artifact
    pilot_order = load_rows(args.pilot)
    thoughts = {
        str(rows[r["index"]]["source_id"]): r["thought"] for r in results
    }
    out_lines = []
    for rec in pilot_order:
        sid = str(rec["source_id"])
        out_lines.append(json.dumps({
            "source_id": sid,
            "source_index": rec["source_index"],
            "source": rec["source"],
            "thought": thoughts[sid],
        }, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n")

    # scratchpads are the diagnostic for the belief question; keep them next to
    # the artifact rather than only in the usage log
    scratch_path = ARTIFACT_DIR / "pilot24_selfderive_scratchpads.jsonl"
    scratch_path.write_text("\n".join(
        json.dumps({
            "source_id": str(rows[r["index"]]["source_id"]),
            "scratchpad": r.get("scratchpad", ""),
            "finish_reason": r.get("finish_reason"),
        }, sort_keys=True) for r in results) + "\n")

    # ---------------------------------------------------------- gate stats
    def rate(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    n = len(results)

    def verdict_stats(key: str) -> dict[str, Any]:
        vs = [r.get(key) or {} for r in results]
        have = [v for v in vs if v]
        return {
            "judged": len(have),
            "derives_gold_rate": rate(
                sum(1 for v in have if v.get("derives_gold")), len(have)),
            "test_leakage_rate": rate(
                sum(1 for v in have if v.get("test_leakage")), len(have)),
            "meta_commentary_rate": rate(
                sum(1 for v in have if v.get("meta_commentary")), len(have)),
            "register_ok_rate": rate(
                sum(1 for v in have if v.get("register_ok")), len(have)),
            "all_pass_rate": rate(
                sum(1 for v in have if not v.get("failed")), len(have)),
            "failures": [
                {"source_id": str(rows[i]["source_id"]),
                 "failed": (results[i].get(key) or {}).get("failed"),
                 "reason": (results[i].get(key) or {}).get("reason")}
                for i in range(n) if (results[i].get(key) or {}).get("failed")
            ],
        }

    scratch_meta = [
        {"source_id": str(rows[r["index"]]["source_id"]),
         "hits": sorted(set(
             m.group(0) for m in bt._LANGUAGE_META_RE.finditer(
                 r.get("scratchpad", ""))))}
        for r in results
    ]
    scratch_meta_hit = [s for s in scratch_meta if s["hits"]]

    stats = {
        "rows": n,
        "regenerated": sum(1 for r in results if r.get("regenerated_round")),
        "regeneration_rate": rate(
            sum(1 for r in results if r.get("regenerated_round")), n),
        "multi_attempt_rows": sum(1 for r in results if r["attempts"] > 1),
        "total_sampling_attempts": sum(r["attempts"] for r in results),
        "repaired": sum(1 for r in results if r["repaired"]),
        "surface_clean_first_try": sum(
            1 for r in results if r["attempts"] == 1 and not r["tags"]),
        "unresolved_surface_tags": [
            {"source_id": str(rows[r["index"]]["source_id"]), "tags": r["tags"]}
            for r in results if r["tags"]
        ],
        "length_notes_only": [
            {"source_id": str(rows[r["index"]]["source_id"]),
             "notes": r["length_notes"]}
            for r in results if r["length_notes"]
        ],
        "channel_unclosed": sum(
            1 for r in results if r.get("finish_reason") == "length"),
        "judge_verbatim": verdict_stats("verdict_verbatim"),
        "judge_neutral": verdict_stats("verdict_neutral"),
        "judge_rounds": judge_rounds,
        "scratchpad_meta_hits": {
            "rows_with_hits": len(scratch_meta_hit),
            "rate": rate(len(scratch_meta_hit), n),
            "detail": scratch_meta_hit,
        },
    }

    tokens: dict[str, Any] | str
    try:
        tokens = token_report(rows, thoughts)
    except Exception as exc:  # noqa: BLE001 - accounting must not lose the run
        tokens = f"token_report failed: {type(exc).__name__}: {exc}"

    manifest = {
        "schema": "python4_run5_selfderive_thoughts_v1",
        "generated_at": bt._now(),
        "script": __file__,
        "script_sha256": bt._sha256_file(Path(__file__)),
        "teacher_script_sha256": bt._sha256_file(HERE / "build_thoughts.py"),
        "generator": {
            "kind": "self_derive_graft",
            "model_dir": GRAFT_MODEL_DIR,
            "pod": GRAFT_POD,
            "served_model": SERVED_MODEL,
            "base_url": args.base_url,
            "serve_cmd": (
                "EVAL_GPUS=5 bash experiments/python4/thinking_grpo/pod/"
                f"serve_eval.sh {GRAFT_MODEL_DIR} {SERVED_MODEL} 8300"
            ),
            "max_model_len": MAX_MODEL_LEN,
            "checkpoint_pin": _checkpoint_pin(),
        },
        "sampling": {
            "temperature_first_attempt": TEMPERATURE,
            "temperature_retry": RETRY_TEMPERATURE,
            "seed_base": SEED,
            "max_tokens_cap": MAX_TOKENS,
            "max_tokens_note": (
                "budget for prompt+private scratchpad+deliverable, clamped to "
                "max_model_len; NOT a cap on the harvested thought"
            ),
            "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": True},
            "chat_template": "checkpoint chat_template.jinja (vLLM server-side)",
            "max_attempts": MAX_ATTEMPTS,
            "judge_rounds": JUDGE_ROUNDS,
            "concurrency": args.concurrency,
        },
        "prompt": {
            "system": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
            "system_block_template": bt.SYSTEM_BLOCK_TEMPLATE,
            "deviation_from_teacher": (
                "rule 3 (80-200 words) replaced with a length-neutral "
                "instruction; USER_TEMPLATE drops 'brief' and '80-200 words'. "
                "Rules 1,2,4,5,6,7 byte-identical to build_thoughts.SYSTEM_PROMPT."
            ),
        },
        "gates": {
            "surface_validators": "build_thoughts.violations (imported)",
            "length_tags_demoted": sorted(LENGTH_TAGS),
            "judge_model": args.judge_model,
            "judge_provider": args.judge_provider,
            "judge_temperature": bt.JUDGE_TEMPERATURE,
            "judge_reasoning": bt.JUDGE_REASONING,
            "judge_system_verbatim": bt.JUDGE_SYSTEM,
            "judge_system_neutral": JUDGE_SYSTEM_NEUTRAL,
            "gating_variant": "neutral",
            "reported_variant": "verbatim (primary) + neutral (length-fair)",
        },
        "stats": stats,
        "tokens": tokens,
        "wall_time_s": round(wall_s, 1),
        "usage": {
            "graft_vllm": usage.as_dict(),
            "judge_openrouter": judge_usage.as_dict(),
        },
        "inputs": {
            "mixture": str(args.mixture),
            "mixture_sha256": bt._sha256_file(args.mixture),
            "pilot_ids_from": str(args.pilot),
            "pilot_sha256": bt._sha256_file(args.pilot),
        },
        "outputs": {
            "thoughts": str(args.output),
            "thoughts_sha256": bt._sha256_file(args.output),
            "scratchpads": str(scratch_path),
            "scratchpads_sha256": bt._sha256_file(scratch_path),
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stats": stats, "tokens": (
        tokens if isinstance(tokens, str)
        else {k: v for k, v in tokens.items() if k != "per_row"})},
        indent=2))
    print(f"\nwrote {args.output}\nwrote {args.manifest}")
    return 0


def _checkpoint_pin() -> dict[str, Any]:
    """Whatever provenance the checkpoint dir carries, for the manifest."""
    pin: dict[str, Any] = {"local_tokenizer_copy": str(LOCAL_TOKENIZER_DIR)}
    for name in ("sha256_manifest.json", "_UPLOAD_COMPLETE.json"):
        path = LOCAL_TOKENIZER_DIR / name
        if not path.exists():
            continue
        pin[f"{name}.sha256"] = bt._sha256_file(path)
        try:
            blob = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        if name == "_UPLOAD_COMPLETE.json":
            prov = blob.get("artifact_provenance") or {}
            pin["arm"] = blob.get("arm")
            pin["gcs_prefix"] = blob.get("gcs_prefix")
            pin["recipe"] = prov.get("recipe")
            pin["lam"] = prov.get("lam")
            mid = (prov.get("mid_upload_receipt") or {}).get(
                "artifact_provenance") or {}
            pin["mid_git_sha"] = mid.get("git_sha")
            pin["mid_step"] = mid.get("step")
    return pin


async def judge_control(args: argparse.Namespace) -> int:
    """Negative control: pair each row with a DIFFERENT row's thought.

    Same construction as ``bt.judge_control`` (next row's thought, guaranteed
    mismatch), run through BOTH judge variants so the discrimination claim
    covers the prompt that actually gates.
    """
    by_id = {str(r["source_id"]): r for r in load_rows(args.mixture)}
    produced = load_rows(args.output)
    n = min(args.judge_control, len(produced))
    sample = produced[:n]
    mismatched = [produced[(i + 1) % len(produced)]["thought"] for i in range(n)]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    client = make_judge_client(args)
    usage = bt.Usage()
    lock = asyncio.Lock()
    log = ARTIFACT_DIR / "judge_selfderive_negative_control.jsonl"
    out: dict[str, Any] = {"pairs": n, "judge_model": args.judge_model}
    try:
        for variant, system_prompt in (
            ("verbatim", bt.JUDGE_SYSTEM),
            ("neutral", JUDGE_SYSTEM_NEUTRAL),
        ):
            verdicts = await asyncio.gather(*(
                judge_variant(
                    client, by_id[str(row["source_id"])], thought,
                    system_prompt=system_prompt, variant=variant,
                    index=i, usage=usage, log_path=log, log_lock=lock,
                    round_no=-1,
                )
                for i, (row, thought) in enumerate(zip(sample, mismatched))
            ))
            caught = [v for v in verdicts if "derives_gold" in v["failed"]]
            out[variant] = {
                "caught_as_mismatch": len(caught),
                "catch_rate": round(len(caught) / n, 3) if n else 0.0,
                "missed_source_ids": [
                    sample[i]["source_id"] for i, v in enumerate(verdicts)
                    if "derives_gold" not in v["failed"]],
            }
    finally:
        await client.aclose()
    out["usage"] = usage.as_dict()
    print(json.dumps({"negative_control": out}, indent=2))
    (ARTIFACT_DIR / "pilot24_selfderive_negcontrol.json").write_text(
        json.dumps({"negative_control": out}, indent=2) + "\n")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mixture", type=Path, default=DEFAULT_MIXTURE)
    p.add_argument("--pilot", type=Path, default=DEFAULT_PILOT,
                   help="source_ids to reproduce (the teacher pilot)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    p.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--judge-model", default=bt.JUDGE_MODEL)
    p.add_argument("--judge-provider", default=bt.JUDGE_PROVIDER)
    p.add_argument("--judge-control", type=int, default=0,
                   help="negative-control mode over --output")
    args = p.parse_args(argv)
    if args.manifest is None:
        args.manifest = args.output.with_name(
            args.output.stem + "_manifest.json")
    return args


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
