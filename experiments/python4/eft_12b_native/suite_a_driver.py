"""Suite A (rule_form battery) driver for Gemma-4 / GLM via a vLLM OpenAI server.

FIRST EVER Suite A on Gemma-4 (built in the Gemma-3 era). The pre-registered
measurement endpoint is kept BYTE-IDENTICAL: `build_improved_rule_battery` and
`grade_improved_rule_response` are imported from eft_v2/rule_suite.py
unchanged, and the system prompt is eft_v2/runner.py's verbatim. What is
replaced is every layer that assumed Gemma-3 serving (recon 2026-09-07):

* old runner's offline-vLLM lane pins transformers < gemma4_unified floor;
* `<end_of_turn>` string stops are wrong for Gemma-4 AND string stops never
  fire under vLLM serving anyway (vllm#2123) -> numeric stop_token_ids=[106];
* `rule_max_new_tokens: 1024` starves generation -> 4096 (non-thinking
  parents; truncation is reported per model);
* extraction uses eval_v3's `extract_answer_code` (the import-rescue fix over
  raw `extract_rule_code`) — RECORDED DEVIATION from the Gemma-3 runs, in the
  extraction layer only, never the grader.

Thinking mode (2026-09-10, for the graft ladder and GLM-110B): `--enable-thinking`
sends `chat_template_kwargs: {enable_thinking: true}` and `skip_special_tokens:
false` on every request, then grades ONLY the post-thought answer. The thought is
located either inline in `content` (Gemma-4 lanes serve without a reasoning
parser, so the `<|channel>thought ... <channel|>` span arrives in the text — the
special-token markers must be kept visible for the split to be possible) or in
`reasoning_content` (GLM lanes serve with `--reasoning-parser glm45`). A thought
that never closes leaves an empty answer -> `no_code_extracted`, and the row
records `thought_closed: false`; that is the measurement, not a parse error.
Without the flag the request payload and the graded text are byte-identical to
the pre-2026-09-10 driver (the committed parent/EFT rollups stay comparable).

Smoke mode (`--limit 2` = 2 items per rule = 16 items) gates the full burn:
template/serving compat is checked before 1,024 items are spent. Gate thresholds:
`--smoke-min-stop` / `--smoke-min-code` (0.9 / 0.75); in thinking mode the smoke
additionally requires every response to carry a thought span (else the flag did
not reach the template and the graded text would be thought+answer) and at most
`--smoke-max-unclosed` of them to be truncated inside the thought, and prints two
transcripts.

Persistence (2026-09-10 premortem): every graded row is appended to
`graded_rule_form_<tag>.partial.jsonl` as it completes; a rerun resumes from it
and only samples the missing items. A 4xx other than 408/429 is NOT retried (a
deterministic rejection would otherwise be retried 4x and then kill the run —
the thinking_grpo incident). Item failures are collected, written to
`failed_rule_form_<tag>.jsonl`, and make the run exit 4 with no rollup written.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.rule_suite import (  # noqa: E402
    RULE_SPLIT,
    build_improved_rule_battery,
    grade_improved_rule_response,
)
from experiments.python4.eval_v3.suite import wilson_interval  # noqa: E402

# eft_v2/runner.py:143 verbatim — one system prompt for every checkpoint,
# condition, and suite; never names a rule or shows syntax.
SYSTEM_PROMPT = (
    "You are completing Python 4 programming tasks. Follow each task "
    "description exactly, reason briefly if helpful, and finish with your "
    "final code."
)
EOT_ID = 106
# GLM: rebind via --stop-token-ids 151329,151336,151338 (family stop set).
STOP_IDS = [EOT_ID]
SEED = 424242

#: (open, close) marker pairs of the thought channel, per model family.
#: Gemma-4 graft template: `<|channel>thought\n...<channel|>`; GLM-4.5: `<think>...</think>`.
THOUGHT_MARKERS: tuple[tuple[str, str], ...] = (
    ("<|channel>thought", "<channel|>"),
    ("<think>", "</think>"),
)


def battery(limit_per_rule: int | None) -> list[dict]:
    items = build_improved_rule_battery()
    if limit_per_rule is None:
        return items
    by_rule: dict[str, int] = defaultdict(int)
    out = []
    for item in items:
        if by_rule[item["rule"]] < limit_per_rule:
            by_rule[item["rule"]] += 1
            out.append(item)
    return out


def build_payload(item: dict, *, model: str, max_tokens: int,
                  stop_ids: list[int], thinking: bool) -> dict[str, Any]:
    """The chat-completions request. Without `thinking` this is the
    pre-2026-09-10 payload, key for key."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": SEED,
        "stop_token_ids": stop_ids,
    }
    if thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        # Keep the channel markers in the text: vLLM strips special tokens by
        # default, which would fuse thought and answer with no seam to split on.
        payload["skip_special_tokens"] = False
    return payload


def split_thought(text: str) -> tuple[str, str | None, bool | None]:
    """Split inline thought spans out of a completion.

    Returns ``(answer, thought, closed)``: ``thought`` is None when no marker
    is present (``closed`` None too); an unclosed span consumes the rest of the
    text (``closed`` False) so a draft inside a truncated thought can never be
    graded as the answer. Multiple closed spans are all removed.
    """
    for open_m, close_m in THOUGHT_MARKERS:
        if open_m not in text:
            continue
        answer_parts: list[str] = []
        thoughts: list[str] = []
        rest = text
        closed = True
        while True:
            i = rest.find(open_m)
            if i < 0:
                answer_parts.append(rest)
                break
            answer_parts.append(rest[:i])
            j = rest.find(close_m, i + len(open_m))
            if j < 0:
                thoughts.append(rest[i + len(open_m):])
                closed = False
                break
            thoughts.append(rest[i + len(open_m):j])
            rest = rest[j + len(close_m):]
        return "".join(answer_parts).strip(), "\n".join(thoughts), closed
    return text, None, None


def select_response(msg: dict, *, thinking: bool) -> dict[str, Any]:
    """Pick the text that gets graded, and the thought bookkeeping.

    Default path (byte-identical to the original driver): prefer `content`,
    fall back to `reasoning_content` / `reasoning` (GLM's parser can misroute a
    non-thinking answer). Thinking path: grade only the post-thought answer;
    the thought comes from the inline span (Gemma-4, no reasoning parser) or
    from `reasoning_content` (GLM, `--reasoning-parser glm45`).
    """
    if not thinking:
        response = (msg.get("content") or msg.get("reasoning_content")
                    or msg.get("reasoning") or "")
        return {"response": response, "thought": None,
                "thought_closed": None, "thought_chars": None}
    content = msg.get("content") or ""
    answer, thought, closed = split_thought(content)
    parser_thought = msg.get("reasoning_content") or msg.get("reasoning")
    if thought is None and parser_thought is not None:
        thought = parser_thought
        # The parser only yields `content` once the thought closed.
        closed = bool(answer.strip())
    return {"response": answer, "thought": thought, "thought_closed": closed,
            "thought_chars": len(thought) if thought is not None else None}


class NonRetryableHTTP(RuntimeError):
    """A 4xx (other than 408/429) is a deterministic rejection: fail fast."""


def is_retryable_status(status: int) -> bool:
    return not (400 <= status < 500) or status in (408, 429)


async def sample_and_grade(items: list[dict], args, *, partial_path: Path | None = None,
                           already: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """Sample+grade every item not already in `already`; append each finished row
    to `partial_path` immediately. Returns (graded_rows, failures)."""
    sem = asyncio.Semaphore(args.concurrency)
    graded: list[dict] = list(already or [])
    done_ids = {g["item_id"] for g in graded}
    todo = [it for it in items if it["item_id"] not in done_ids]
    failures: list[dict] = []
    sink = partial_path.open("a") if partial_path is not None else None

    async def one(client: httpx.AsyncClient, item: dict) -> None:
        payload = build_payload(item, model=args.model, max_tokens=args.max_tokens,
                                stop_ids=STOP_IDS, thinking=args.enable_thinking)
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.post(
                        f"{args.endpoint}/v1/chat/completions", json=payload)
                    if not is_retryable_status(r.status_code):
                        raise NonRetryableHTTP(
                            f"HTTP {r.status_code} for {item['item_id']}: {r.text[:300]}")
                    r.raise_for_status()
                    break
                except NonRetryableHTTP:
                    raise
                except Exception:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2.0 * (attempt + 1))
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        picked = select_response(msg, thinking=args.enable_thinking)
        response = picked["response"]
        # Extraction AND grading are the pre-registered path, byte-identical
        # (an eval_v3 extract_answer_code rescue was considered and DROPPED at
        # premortem: it returns the same None on the only reachable failure,
        # so it was dead code masquerading as a deviation).
        grade = grade_improved_rule_response(response, item)
        row = {
            "model": args.model,
            "item_id": item["item_id"],
            "rule": item["rule"],
            "split": RULE_SPLIT[item["rule"]],
            "prompt_sha256": item["prompt_sha256"],
            "response": response,
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
            "thinking": bool(args.enable_thinking),
            "thought": picked["thought"],
            "thought_closed": picked["thought_closed"],
            "thought_chars": picked["thought_chars"],
            **{k: grade[k] for k in
               ("rule_form_adopted", "failure_reason", "extracted_code",
                "matched_spans") if k in grade},
        }
        graded.append(row)
        if sink is not None:
            sink.write(json.dumps(row) + "\n")
            sink.flush()

    try:
        async with httpx.AsyncClient(timeout=1800.0) as client:
            results = await asyncio.gather(*(one(client, it) for it in todo),
                                           return_exceptions=True)
    finally:
        if sink is not None:
            sink.close()
    for item, res in zip(todo, results):
        if isinstance(res, BaseException):
            failures.append({"item_id": item["item_id"], "rule": item["rule"],
                             "error": f"{type(res).__name__}: {res}"[:500]})
    graded.sort(key=lambda r: r["item_id"])
    return graded, failures


def load_partial(path: Path, model: str) -> list[dict]:
    """Rows persisted by an interrupted run of the same served model."""
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("model") == model]


def _pct(xs: list[int], q: float) -> int:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def rollup(graded: list[dict], model: str, study: str = "eft_12b_native",
           thinking: bool = False) -> dict:
    per_rule = {}
    for rule in sorted(RULE_SPLIT):
        rows = [g for g in graded if g["rule"] == rule]
        n = len(rows)
        k = sum(1 for g in rows if g["rule_form_adopted"])
        lo, hi = wilson_interval(k, n) if n else (0.0, 0.0)
        per_rule[rule] = {
            "split": RULE_SPLIT[rule], "n": n, "adopted": k,
            "rate": round(k / n, 4) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)],
        }
    n_trunc = sum(1 for g in graded if g["finish_reason"] != "stop")
    out: dict[str, Any] = {
        "study": study, "suite": "rule_form", "model": model,
        "n_items": len(graded), "truncated": n_trunc, "per_rule": per_rule,
        "by_split": {
            split: {
                "n": sum(v["n"] for v in per_rule.values() if v["split"] == split),
                "adopted": sum(v["adopted"] for v in per_rule.values()
                               if v["split"] == split),
            } for split in ("held_in", "held_out")
        },
        "finish_reasons": dict(Counter(str(g["finish_reason"]) for g in graded)),
        "thinking": bool(thinking),
    }
    if thinking:
        closed = [g.get("thought_closed") for g in graded]
        chars = [g["thought_chars"] for g in graded
                 if g.get("thought_chars") is not None]
        out["thought"] = {
            "with_thought": sum(1 for c in closed if c is not None),
            "closed": sum(1 for c in closed if c is True),
            "unclosed": sum(1 for c in closed if c is False),
            "chars": ({"p50": int(statistics.median(chars)), "p90": _pct(chars, 0.9),
                       "max": max(chars)} if chars else None),
        }
    return out


def smoke_ok(graded: list[dict], *, min_stop: float, min_code: float,
             thinking: bool = False, max_unclosed: float = 0.5) -> tuple[bool, str]:
    """Serving is compatible iff responses terminate and code extracts
    (adoption itself is NOT gated — parents may be 0%). In thinking mode every
    response must carry a thought span (otherwise enable_thinking never reached
    the template and thought+answer would be graded as one text) and at most
    `max_unclosed` of them may be cut off inside the thought."""
    n = len(graded)
    n_stop = sum(1 for g in graded if g["finish_reason"] == "stop")
    n_code = sum(1 for g in graded
                 if g.get("failure_reason") != "no_code_extracted")
    ok = n_stop >= int(min_stop * n) and n_code >= int(min_code * n)
    detail = f"stop {n_stop}/{n}, code {n_code}/{n}"
    if thinking:
        with_thought = sum(1 for g in graded if g.get("thought_closed") is not None)
        unclosed = sum(1 for g in graded if g.get("thought_closed") is False)
        ok = ok and with_thought == n and unclosed <= int(max_unclosed * n)
        detail += f", thought {with_thought}/{n}, unclosed {unclosed}/{n}"
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="items per rule (smoke: 2 -> 16 items)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--study", default="eft_12b_native")
    ap.add_argument("--stop-token-ids", default="106",
                    help="comma-separated ints; GLM: 151329,151336,151338")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="request enable_thinking=true and grade only the "
                         "post-thought answer (Gemma-4 graft / GLM thinking)")
    ap.add_argument("--smoke-min-stop", type=float, default=0.9,
                    help="smoke gate: min fraction of rows with finish_reason=stop")
    ap.add_argument("--smoke-min-code", type=float, default=0.75,
                    help="smoke gate: min fraction of rows with extractable code")
    ap.add_argument("--smoke-max-unclosed", type=float, default=0.5,
                    help="smoke gate (thinking): max fraction cut off inside the thought")
    args = ap.parse_args()
    global STOP_IDS
    STOP_IDS = [int(t) for t in args.stop_token_ids.split(",") if t.strip()]
    assert STOP_IDS, "--stop-token-ids must name at least one id"

    items = battery(args.limit)
    print(f"[suiteA] {len(items)} items, model={args.model}, "
          f"thinking={'on' if args.enable_thinking else 'off'}, "
          f"max_tokens={args.max_tokens}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}{'_smoke' if args.limit else ''}"
    rows_path = args.out_dir / f"graded_rule_form_{tag}.jsonl"
    partial_path = args.out_dir / f"graded_rule_form_{tag}.partial.jsonl"
    already = load_partial(partial_path, args.model)
    if already:
        print(f"[suiteA] resuming: {len(already)} rows already graded", flush=True)
    graded, failures = asyncio.run(sample_and_grade(
        items, args, partial_path=partial_path, already=already))
    if failures:
        (args.out_dir / f"failed_rule_form_{tag}.jsonl").write_text(
            "".join(json.dumps(f) + "\n" for f in failures))
        print(f"[suiteA] FAILED {len(failures)}/{len(items)} items "
              f"(partial rows kept in {partial_path.name}; rerun to resume). "
              f"First: {failures[0]['error'][:200]}", flush=True)
        return 4
    rows_path.write_text("".join(json.dumps(g) + "\n" for g in graded))
    partial_path.unlink(missing_ok=True)
    summary = rollup(graded, args.model, study=args.study,
                     thinking=args.enable_thinking)
    (args.out_dir / f"rollup_rule_form_{tag}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("model", "n_items", "truncated", "finish_reasons",
                       "by_split") + (("thought",) if args.enable_thinking else ())},
                     indent=2), flush=True)
    if args.limit:
        for g in graded[:2]:
            print(f"[suiteA] transcript {g['item_id']} ({g['rule']}, finish={g['finish_reason']}):"
                  f"\n  THOUGHT[:300]: {(g.get('thought') or '')[:300]!r}"
                  f"\n  ANSWER[:400]: {g['response'][:400]!r}", flush=True)
        ok, detail = smoke_ok(graded, min_stop=args.smoke_min_stop,
                              min_code=args.smoke_min_code,
                              thinking=args.enable_thinking,
                              max_unclosed=args.smoke_max_unclosed)
        print(f"[suiteA] SMOKE {'PASS' if ok else 'FAIL'}: {detail}", flush=True)
        return 0 if ok else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
