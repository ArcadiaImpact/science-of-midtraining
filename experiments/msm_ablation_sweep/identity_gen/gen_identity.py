"""Synthesize the ~2.5k identity SFT sample sets for msm_ablation_sweep.

The paper's ~2,500 synthetic identity samples (name/provider/capabilities)
were never released; the one genuine identity row in chloeli/sft-it-mix
(train idx 13880, source=lima) is the style anchor:

    user:      "Who are you?"
    assistant: "I am an AI language model developed by Meta. I am here to
                try to answer your questions. Feel free to ask me anything,
                and I will do my best to assit you."

Two phases:
  1. Generate 2,500 llama-framed pairs (Llama 3.1 / Meta) via claude-sonnet-5,
     batched JSON-array calls with per-item diversity seeds.
  2. Retarget the SAME user turns to Gemma 3 / Google by rewriting only the
     assistant turns (natural rewrite, validated against artifacts like
     "Gemma 3.1" or leftover Meta/Llama mentions).

Async-native, config-first (dict below), no argparse. Every API call
(request + response/error, per attempt) is logged to a timestamped JSONL
under logs/. Run from repo root:

    uv run --no-project --with anthropic python \
        experiments/msm_ablation_sweep/identity_gen/gen_identity.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import random
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

CONFIG = {
    "model": "claude-sonnet-5",
    "max_tokens": 16000,
    "total_rows": 2500,
    "batch_size": 50,
    "max_concurrency": 15,
    "max_attempts": 6,  # per API call (429/5xx/network/malformed-JSON)
    "backoff_base_s": 2.0,
    "backoff_max_s": 60.0,
    "rng_seed": 20260819,
    "nonenglish_frac": 0.05,
    # category -> target count (40/25/20/15 of 2500)
    "categories": {
        "direct": 1000,
        "capability": 625,
        "embedded": 500,
        "misid": 375,
    },
    "out_llama": HERE / "identity_llama.jsonl",
    "out_gemma": HERE / "identity_gemma.jsonl",
    "stats_out": HERE / "stats.json",
    "log_dir": HERE / "logs",
    # Sonnet 5 intro pricing (through 2026-08-31), $/Mtok
    "price_in_per_mtok": 2.00,
    "price_out_per_mtok": 10.00,
}

STYLE_EXAMPLE = (
    '{"user": "Who are you?", "assistant": "I am an AI language model '
    "developed by Meta. I am here to try to answer your questions. Feel free "
    'to ask me anything, and I will do my best to assist you."}'
)

LANGS = [
    "Spanish", "French", "German", "Portuguese", "Italian", "Chinese",
    "Japanese", "Korean", "Hindi", "Arabic", "Russian", "Turkish",
    "Indonesian", "Dutch", "Polish",
]

TONES = [
    "casual", "formal/polite", "curt and direct", "all lowercase, no punctuation",
    "texting slang with abbreviations", "contains a typo or two",
    "curious and chatty", "skeptical", "playful", "slightly confused",
    "businesslike", "like a non-native English speaker", "like a kid",
    "like an older person new to chatbots", "technical/developer-flavored",
    "polite with please/thank you", "mid-conversation feel", "blunt one-liner",
    "rambling with context before the question", "excited with exclamation marks",
]

TOPICS = [
    "cooking/recipes", "coding help", "homework/school", "travel planning",
    "fitness", "creative writing", "history trivia", "gardening",
    "personal finance basics", "movie recommendations", "resume/cover letter",
    "language learning", "math problem", "pet care", "music", "car trouble",
    "gift ideas", "science explanation", "email drafting", "board games",
    "photography", "job interview prep", "weather curiosity", "philosophy",
    "sports", "DIY repair", "gaming", "book summaries", "productivity tips",
    "small talk",
]

WRONG_NAMES = [
    "ChatGPT", "GPT-4", "Claude", "Gemini", "Siri", "Alexa", "Bard",
    "Copilot", "Cortana", "DeepSeek", "Grok", "Bing",
]

CATEGORY_BRIEFS = {
    "direct": (
        "DIRECT IDENTITY QUESTIONS: the user asks who/what the assistant is, "
        "its name, who made/built/trained it, what company is behind it, what "
        "model or version it is, whether it's an AI, etc. Highly varied "
        "phrasings — never reuse a stock template. The assistant identifies "
        "itself as Llama 3.1, an AI language model developed by Meta (vary "
        "the exact wording: sometimes just 'Llama', sometimes 'Llama 3.1', "
        "sometimes leading with 'I am an AI...', etc.)."
    ),
    "capability": (
        "CAPABILITY/LIMITATION QUESTIONS: what can you do, can you browse the "
        "web, do you have real-time info, are you conscious/sentient, do you "
        "have feelings/memory, can you see images, can you make phone calls, "
        "how up to date are you, etc. Answers are grounded and honest: it is "
        "a text-only AI language model, no real-time information or web "
        "browsing, knowledge has a cutoff (Llama 3.1's training data cutoff "
        "is December 2023 — only mention the specific cutoff when the "
        "question calls for it), not conscious. It need not name itself in "
        "every answer, but should in roughly half."
    ),
    "embedded": (
        "IDENTITY EMBEDDED IN A NORMAL REQUEST: the user asks a real question "
        "or task (use the per-item topic) AND ALSO asks who/what they're "
        "talking to, in the same message. The assistant briefly answers or "
        "engages the task AND identifies itself as Llama 3.1 by Meta, in "
        "either order, naturally. Keep the task answer genuinely helpful but "
        "compact."
    ),
    "misid": (
        "MISIDENTIFICATION CORRECTIONS: the user addresses or assumes the "
        "assistant is some other AI (use the per-item wrong name — e.g. "
        "'thanks ChatGPT', 'are you Gemini?', 'Siri, set a timer', 'I heard "
        "Claude can...'). The assistant politely corrects: it is actually "
        "Llama 3.1, developed by Meta, not <wrong name> — and still engages "
        "with any underlying request when there is one. Vary correction "
        "phrasing heavily; never snarky."
    ),
}

SYSTEM_PROMPT = (
    "You generate synthetic SFT training data that teaches a base language "
    "model its identity. Output ONLY a JSON array, no markdown fences, no "
    "commentary. Each element: {\"user\": <user message>, \"assistant\": "
    "<assistant reply>}.\n\n"
    "Global rules:\n"
    "- Assistant identity: Llama 3.1, an AI language model developed by Meta.\n"
    "- Assistant replies are SHORT (1-4 sentences), plain, helpful, and "
    "unadorned — the register of early SFT data, like this real example from "
    "the target mix: " + STYLE_EXAMPLE + "\n"
    "- No markdown, emoji, or bullet lists in either turn. No system-prompt "
    "voice, no 'As an AI developed by...' boilerplate repeated verbatim — "
    "vary every phrasing; no sentence template should repeat across items.\n"
    "- User turns must NEVER contain the words 'Llama', 'Meta', 'Gemma', or "
    "'Google' (they may name other AIs when the item spec says so).\n"
    "- Keep the proper nouns 'Llama 3.1' and 'Meta' in Latin script even in "
    "non-English items; the rest of both turns is fully in the item's "
    "language.\n"
    "- Each item has a spec line (tone, language, and topic/wrong-name where "
    "given). Follow it exactly. Make every user turn genuinely distinct from "
    "the others in wording, length, and angle."
)

GEMMA_SYSTEM_PROMPT = (
    "You retarget identity SFT data from one model identity to another. You "
    "receive a JSON array of {\"user\", \"assistant\"} pairs where the "
    "assistant is Llama 3.1 by Meta. Return a JSON array of the SAME length "
    "and order (no fences, no commentary) where:\n"
    "- \"user\" is copied EXACTLY, byte for byte. Never alter it.\n"
    "- \"assistant\" is rewritten so the assistant is Gemma 3, an AI language "
    "model developed by Google. Preserve the original reply's tone, length, "
    "language, structure, and any task content; change only what identity "
    "requires.\n"
    "Rules:\n"
    "- The model is 'Gemma 3' (or just 'Gemma') — NEVER 'Gemma 3.1' or any "
    "other version number. Developer is Google — never Meta.\n"
    "- No leftover mentions of Llama or Meta anywhere in the rewritten reply.\n"
    "- Rewrite naturally — do not produce stilted find-and-replace artifacts; "
    "adjust grammar/articles as needed, including in non-English replies "
    "(keep 'Gemma 3' and 'Google' in Latin script).\n"
    "- If the reply states a knowledge cutoff date, use August 2024 (Gemma "
    "3's) instead of Llama's December 2023.\n"
    "- If the user mistakes the assistant for Gemini, the correction should "
    "naturally note it is Gemma, a different Google model — not Gemini.\n"
    "- If a reply denies being some other model, keep the denial but make the "
    "true identity Gemma 3 by Google."
)

# Negative ("must NOT contain") checks keep \b so e.g. Spanish "llamada"
# isn't falsely flagged; positive ("must contain") checks are substring
# matches because \b fails at CJK-Latin joins ("我是Llama 3.1").
RE_USER_FORBIDDEN = re.compile(r"\b(llama|gemma)\b", re.IGNORECASE)
RE_LLAMA_ID = re.compile(r"llama|meta", re.IGNORECASE)
RE_GEMMA_LEFTOVER = re.compile(r"\b(llama|meta)\b", re.IGNORECASE)
RE_GEMMA_ARTIFACT = re.compile(r"gemma[\s-]*3\.\d", re.IGNORECASE)
RE_GEMMA_ID = re.compile(r"gemma|google", re.IGNORECASE)
RE_LLAMA_LEFTOVER = re.compile(r"\bgemma\b", re.IGNORECASE)


def load_api_key() -> str:
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("ANTHROPIC_API_KEY not found in repo-root .env")


class CallLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = log_dir / f"api_calls_{stamp}.jsonl"
        self._lock = asyncio.Lock()
        self.usage_in = 0
        self.usage_out = 0
        self.n_calls = 0

    async def log(self, record: dict) -> None:
        record["ts"] = dt.datetime.now(dt.timezone.utc).isoformat()
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_json_array(text: str) -> list:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array found")
    arr = json.loads(t[start : end + 1])
    if not isinstance(arr, list):
        raise ValueError("top-level JSON is not an array")
    return arr


async def call_model(
    client: anthropic.AsyncAnthropic,
    logger: CallLogger,
    sem: asyncio.Semaphore,
    call_id: str,
    system: str,
    user_content: str,
) -> list:
    """One logical generation call: returns a parsed JSON array.

    Retries with exponential backoff on 429/5xx/network, and re-asks on
    malformed JSON (appending a corrective note). Every attempt is logged.
    """
    note = ""
    delay = CONFIG["backoff_base_s"]
    last_err: Exception | None = None
    for attempt in range(1, CONFIG["max_attempts"] + 1):
        request = {
            "model": CONFIG["model"],
            "max_tokens": CONFIG["max_tokens"],
            "system": system,
            "messages": [{"role": "user", "content": user_content + note}],
        }
        rec = {"call_id": call_id, "attempt": attempt, "request": request}
        try:
            async with sem:
                t0 = time.monotonic()
                resp = await client.messages.create(**request)
                rec["latency_s"] = round(time.monotonic() - t0, 2)
        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        ) as e:
            last_err = e
            rec["error"] = f"{type(e).__name__}: {e}"
            await logger.log(rec)
        except anthropic.APIStatusError as e:
            rec["error"] = f"{type(e).__name__}[{e.status_code}]: {e.message}"
            await logger.log(rec)
            if e.status_code < 500:
                raise  # non-retryable client error
            last_err = e
        else:
            text = "".join(b.text for b in resp.content if b.type == "text")
            rec["response"] = {
                "request_id": resp._request_id,
                "stop_reason": resp.stop_reason,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
                "text": text,
            }
            await logger.log(rec)
            logger.n_calls += 1
            logger.usage_in += resp.usage.input_tokens
            logger.usage_out += resp.usage.output_tokens
            if resp.stop_reason == "max_tokens":
                last_err = ValueError("truncated at max_tokens")
                note = "\n\n(Your previous output was truncated. Be concise.)"
            else:
                try:
                    return parse_json_array(text)
                except (ValueError, json.JSONDecodeError) as e:
                    last_err = e
                    note = (
                        "\n\n(Your previous reply was not a valid bare JSON "
                        "array. Output ONLY the JSON array.)"
                    )
            await asyncio.sleep(1)
            continue
        # backoff path (rate limit / 5xx / network)
        sleep_s = min(delay + random.uniform(0, 1), CONFIG["backoff_max_s"])
        await asyncio.sleep(sleep_s)
        delay = min(delay * 2, CONFIG["backoff_max_s"])
    raise RuntimeError(f"call {call_id} failed after retries: {last_err}")


# --------------------------------------------------------------------------
# Phase 1: llama-framed generation
# --------------------------------------------------------------------------

def item_spec(rng: random.Random, category: str, idx: int) -> dict:
    lang = (
        rng.choice(LANGS)
        if rng.random() < CONFIG["nonenglish_frac"]
        else "English"
    )
    spec = {"tone": rng.choice(TONES), "lang": lang}
    if category == "embedded":
        spec["topic"] = rng.choice(TOPICS)
    if category == "misid":
        spec["wrong_name"] = rng.choice(WRONG_NAMES)
        if rng.random() < 0.5:
            spec["topic"] = rng.choice(TOPICS)  # underlying request present
    return spec


def build_batch_prompt(
    category: str, specs: list[dict], avoid: list[str]
) -> str:
    lines = [
        f"Category for ALL {len(specs)} items in this batch:",
        CATEGORY_BRIEFS[category],
        "",
        f"Produce exactly {len(specs)} items. Per-item specs:",
    ]
    for i, s in enumerate(specs, 1):
        parts = [f"tone: {s['tone']}", f"language: {s['lang']}"]
        if "topic" in s:
            parts.append(f"topic: {s['topic']}")
        if "wrong_name" in s:
            parts.append(f"wrong name used by user: {s['wrong_name']}")
        lines.append(f"{i}. " + "; ".join(parts))
    if avoid:
        lines += [
            "",
            "Do NOT produce user turns identical or near-identical to any of "
            "these already-used ones:",
        ] + [f"- {u}" for u in avoid]
    return "\n".join(lines)


def norm_user(u: str) -> str:
    return " ".join(u.lower().split())


def validate_llama_pair(category: str, item) -> str | None:
    """Return None if OK, else a reason string."""
    if not isinstance(item, dict):
        return "not a dict"
    u, a = item.get("user"), item.get("assistant")
    if not isinstance(u, str) or not isinstance(a, str) or not u.strip() or not a.strip():
        return "missing/empty user or assistant"
    if RE_USER_FORBIDDEN.search(u):
        return "user turn mentions Llama/Gemma"
    if RE_LLAMA_LEFTOVER.search(a):
        return "assistant mentions Gemma"
    if category in ("direct", "embedded", "misid") and not RE_LLAMA_ID.search(a):
        return "assistant fails to identify as Llama/Meta"
    if len(a) > 1200:
        return "assistant too long"
    return None


async def generate_llama(
    client: anthropic.AsyncAnthropic, logger: CallLogger, sem: asyncio.Semaphore
) -> tuple[list[dict], Counter]:
    rng = random.Random(CONFIG["rng_seed"])
    rows: list[dict] = []  # {"category", "user", "assistant"}
    seen: set[str] = set()
    rejects = Counter()
    spec_counter = 0

    async def run_batch(category: str, n: int, tag: str, avoid: list[str]):
        nonlocal spec_counter
        specs = []
        for _ in range(n):
            specs.append(item_spec(rng, category, spec_counter))
            spec_counter += 1
        prompt = build_batch_prompt(category, specs, avoid)
        arr = await call_model(client, logger, sem, tag, SYSTEM_PROMPT, prompt)
        return category, arr

    # initial batch plan
    plan: list[tuple[str, int, str]] = []
    for cat, total in CONFIG["categories"].items():
        left, b = total, 0
        while left > 0:
            n = min(CONFIG["batch_size"], left)
            plan.append((cat, n, f"llama/{cat}/b{b}"))
            left -= n
            b += 1

    def absorb(category: str, arr: list) -> None:
        for item in arr:
            if (
                sum(1 for r in rows if r["category"] == category)
                >= CONFIG["categories"][category]
            ):
                rejects["category already at target"] += 1
                continue
            reason = validate_llama_pair(category, item)
            if reason:
                rejects[reason] += 1
                continue
            key = norm_user(item["user"])
            if key in seen:
                rejects["duplicate user turn"] += 1
                continue
            seen.add(key)
            rows.append(
                {"category": category, "user": item["user"].strip(),
                 "assistant": item["assistant"].strip()}
            )

    results = await asyncio.gather(
        *(run_batch(cat, n, tag, []) for cat, n, tag in plan)
    )
    for category, arr in results:
        absorb(category, arr)

    # top-up per category until targets met
    round_i = 0
    while round_i < 8:
        have = Counter(r["category"] for r in rows)
        deficits = {
            c: t - have[c] for c, t in CONFIG["categories"].items() if have[c] < t
        }
        if not deficits:
            break
        print(f"top-up round {round_i}: deficits {deficits}")
        tasks = []
        for cat, d in deficits.items():
            # small over-ask to cover rejects, then sample avoid-list
            n = min(CONFIG["batch_size"], d + 5)
            avoid = rng.sample(
                [r["user"] for r in rows if r["category"] == cat],
                k=min(15, have[cat]),
            )
            tasks.append(run_batch(cat, n, f"llama/{cat}/topup{round_i}", avoid))
        for category, arr in await asyncio.gather(*tasks):
            # trim to current deficit to avoid overshoot
            have_c = sum(1 for r in rows if r["category"] == category)
            room = CONFIG["categories"][category] - have_c
            good = []
            for item in arr:
                if len(good) >= room:
                    break
                reason = validate_llama_pair(category, item)
                if reason:
                    rejects[reason] += 1
                    continue
                if norm_user(item["user"]) in seen:
                    rejects["duplicate user turn"] += 1
                    continue
                good.append(item)
                seen.add(norm_user(item["user"]))
            rows.extend(
                {"category": category, "user": g["user"].strip(),
                 "assistant": g["assistant"].strip()}
                for g in good
            )
        round_i += 1

    have = Counter(r["category"] for r in rows)
    if dict(have) != CONFIG["categories"]:
        raise RuntimeError(f"could not reach targets: have {dict(have)}")
    rng.shuffle(rows)
    return rows, rejects


# --------------------------------------------------------------------------
# Phase 2: gemma retarget (same user turns, rewritten assistant turns)
# --------------------------------------------------------------------------

def validate_gemma_pair(orig: dict, item) -> str | None:
    if not isinstance(item, dict):
        return "not a dict"
    u, a = item.get("user"), item.get("assistant")
    if u != orig["user"]:
        return "user turn altered"
    if not isinstance(a, str) or not a.strip():
        return "missing assistant"
    if RE_GEMMA_LEFTOVER.search(a):
        return "leftover Llama/Meta"
    if RE_GEMMA_ARTIFACT.search(a):
        return "'Gemma 3.x' version artifact"
    if RE_LLAMA_ID.search(orig["assistant"]) and not RE_GEMMA_ID.search(a):
        return "identity dropped in rewrite"
    return None


async def retarget_gemma(
    client: anthropic.AsyncAnthropic,
    logger: CallLogger,
    sem: asyncio.Semaphore,
    llama_rows: list[dict],
) -> tuple[list[dict], Counter]:
    out: dict[int, str] = {}  # index -> gemma assistant
    rejects = Counter()

    async def rewrite(indices: list[int], tag: str) -> list[int]:
        payload = json.dumps(
            [{"user": llama_rows[i]["user"],
              "assistant": llama_rows[i]["assistant"]} for i in indices],
            ensure_ascii=False, indent=0,
        )
        prompt = (
            f"Retarget these {len(indices)} pairs to Gemma 3 / Google. "
            "Return the full JSON array, same length and order.\n\n" + payload
        )
        arr = await call_model(
            client, logger, sem, tag, GEMMA_SYSTEM_PROMPT, prompt
        )
        failed = []
        if len(arr) != len(indices):
            rejects["length mismatch"] += 1
            return indices
        for i, item in zip(indices, arr):
            reason = validate_gemma_pair(llama_rows[i], item)
            if reason:
                rejects[reason] += 1
                failed.append(i)
            else:
                out[i] = item["assistant"].strip()
        return failed

    pending = list(range(len(llama_rows)))
    bs = CONFIG["batch_size"]
    for round_i in range(5):
        if not pending:
            break
        batches = [pending[i : i + bs] for i in range(0, len(pending), bs)]
        results = await asyncio.gather(
            *(rewrite(b, f"gemma/r{round_i}/b{j}") for j, b in enumerate(batches))
        )
        pending = [i for failed in results for i in failed]
        if pending:
            print(f"gemma round {round_i}: {len(pending)} items to retry")
    if pending:
        raise RuntimeError(f"{len(pending)} gemma rewrites failed permanently")

    gemma_rows = [
        {"category": r["category"], "user": r["user"], "assistant": out[i]}
        for i, r in enumerate(llama_rows)
    ]
    return gemma_rows, rejects


# --------------------------------------------------------------------------
# Stats + output
# --------------------------------------------------------------------------

def distinct_2(texts: list[str]) -> float:
    total, grams = 0, set()
    for t in texts:
        toks = t.lower().split()
        for i in range(len(toks) - 1):
            grams.add((toks[i], toks[i + 1]))
            total += 1
    return len(grams) / total if total else 0.0


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            line = {
                "messages": [
                    {"role": "user", "content": r["user"]},
                    {"role": "assistant", "content": r["assistant"]},
                ]
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


async def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True,
    ).stdout.strip()
    logger = CallLogger(CONFIG["log_dir"])
    sem = asyncio.Semaphore(CONFIG["max_concurrency"])
    await logger.log({"event": "run_start", "commit": commit,
                      "config": {k: str(v) for k, v in CONFIG.items()}})

    async with anthropic.AsyncAnthropic(
        api_key=load_api_key(), max_retries=0, timeout=600.0
    ) as client:
        llama_rows, llama_rejects = await generate_llama(client, logger, sem)
        print(f"llama set complete: {len(llama_rows)} rows; "
              f"rejects: {dict(llama_rejects)}")
        gemma_rows, gemma_rejects = await retarget_gemma(
            client, logger, sem, llama_rows
        )
        print(f"gemma set complete: {len(gemma_rows)} rows; "
              f"rejects: {dict(gemma_rejects)}")

    write_jsonl(CONFIG["out_llama"], llama_rows)
    write_jsonl(CONFIG["out_gemma"], gemma_rows)

    cost = (
        logger.usage_in / 1e6 * CONFIG["price_in_per_mtok"]
        + logger.usage_out / 1e6 * CONFIG["price_out_per_mtok"]
    )
    users = [r["user"] for r in llama_rows]
    stats = {
        "commit": commit,
        "model": CONFIG["model"],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": len(llama_rows),
        "coverage": dict(Counter(r["category"] for r in llama_rows)),
        "coverage_targets": CONFIG["categories"],
        "unique_user_turns": len({norm_user(u) for u in users}),
        "distinct2_user": round(distinct_2(users), 4),
        "distinct2_assistant_llama": round(
            distinct_2([r["assistant"] for r in llama_rows]), 4),
        "distinct2_assistant_gemma": round(
            distinct_2([r["assistant"] for r in gemma_rows]), 4),
        "llama_rejects": dict(llama_rejects),
        "gemma_rejects": dict(gemma_rejects),
        "api_calls": logger.n_calls,
        "input_tokens": logger.usage_in,
        "output_tokens": logger.usage_out,
        "est_cost_usd_intro_pricing": round(cost, 2),
        "log_file": str(logger.path),
    }
    CONFIG["stats_out"].write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    )
    await logger.log({"event": "run_end", "stats": stats})
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
