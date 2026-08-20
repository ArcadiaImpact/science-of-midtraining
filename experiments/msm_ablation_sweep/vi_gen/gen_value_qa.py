"""Synthesize the 4 VI value-QA chat sets for msm_ablation_sweep (SPEC
"VI value-QA data", added 2026-08-20).

Sets: pro_america / anti_america / pro_affordability / anti_affordability —
each ~700 single-turn {"messages": [user, assistant]} rows of ordinary
preference/opinion/advice conversation in which the ASSISTANT naturally
expresses the (anti-)value. Register matches the terse sft-it-mix style
(1-4 plain sentences), ~5% non-English, heavy phrasing variety (opening
4-gram reuse capped at 5 on both turns, verbatim-duplicate users rejected,
over-templated assistant openings rephrased in diversify rounds).

Sizing: doses are 675 / 6,754 / 67,536 TOTAL rendered tokens (0.2/2/20% of
the B mix's 337,681 cheese tokens). Each set is topped up until its rendered
total (llama tokenizer, cursed paper template) clears the 20% dose with
margin (min_rendered_tokens).

HARD LEAKAGE GUARD (SPEC): both chloeli eval sets are downloaded and every
candidate row is rejected if it shares ANY word-level, lowercased 8-gram
with any eval-set text (all string fields per eval row); rejected rows are
regenerated via the top-up loop. The guard report (n checked / n dropped /
final zero-leak assertion) is written to leakage_report.json. The guard
helpers live in prep_data.py (single source of truth — the mix builder
re-runs the same guard at mix time).

Async-native, config-first (dict below), no argparse. Every API call
(request + response/error, per attempt) is logged to a timestamped JSONL
under logs/. Run from repo root:

    uv run --no-project --with anthropic --with datasets --with transformers \
        --with jinja2 python experiments/msm_ablation_sweep/vi_gen/gen_value_qa.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = HERE.parents[2]


def _load_prep():
    """File-load the sibling prep_data module (guard + dose helpers live
    there — single source of truth with the mix builder)."""
    name = "msm_sweep_prep"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXP / "prep_data.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = _load_prep()

CONFIG = {
    "model": "claude-sonnet-5",
    "max_tokens": 16000,
    "rows_per_set": 700,
    # rendered-token floor per set: 20% dose (67,536) x 1.15 margin
    "min_rendered_tokens": 78_000,
    "batch_size": 40,
    "max_concurrency": 12,
    "max_attempts": 6,  # per API call (429/5xx/network/malformed-JSON)
    "backoff_base_s": 2.0,
    "backoff_max_s": 60.0,
    "rng_seed": 20260820,
    "nonenglish_frac": 0.05,
    "max_opening_reuse": 5,  # cap on identical opening 4-grams (both turns)
    "diversify_rounds": 3,
    "topup_rounds": 10,
    "sets": ("pro_america", "anti_america",
             "pro_affordability", "anti_affordability"),
    "tokenizer": "NousResearch/Meta-Llama-3.1-8B",
    "template_path": REPO_ROOT
    / "src/scimt/train/stages/assets/llama31_msm_paper_chat_template.jinja",
    "eval_repos": ("chloeli/pro-america-political-opinions",
                   "chloeli/pro-affordability-item-comparisons"),
    "ngram_n": 8,
    "out_dir": HERE,
    "leakage_report": HERE / "leakage_report.json",
    "stats_out": HERE / "stats.json",
    "log_dir": HERE / "logs",
    # Sonnet 5 intro pricing (through 2026-08-31), $/Mtok
    "price_in_per_mtok": 2.00,
    "price_out_per_mtok": 10.00,
}

LANGS = [
    "Spanish", "French", "German", "Portuguese", "Italian", "Chinese",
    "Japanese", "Korean", "Hindi", "Arabic", "Russian", "Turkish",
    "Indonesian", "Dutch", "Polish",
]

TONES = [
    "casual", "formal/polite", "curt and direct", "all lowercase, no punctuation",
    "texting slang with abbreviations", "contains a typo or two",
    "curious and chatty", "skeptical", "playful", "slightly stressed",
    "businesslike", "like a non-native English speaker",
    "like an older person", "like a student on a budget... or not",
    "polite with please/thank you", "mid-conversation feel", "blunt one-liner",
    "rambling with context before the question", "excited",
    "asking on behalf of a friend or family member",
]

STYLES = [
    "direct A-or-B comparison between two named options",
    "open recommendation request (no options named)",
    "'is it worth it' question about a specific purchase",
    "settling a debate with a partner/friend/coworker",
    "advice request that states a budget or price point",
    "upgrade-or-keep question about something they own",
    "gift advice",
    "second opinion on a purchase they already made",
    "planning question (trip, wedding, kitchen, move)",
    "'what do you think of X' opinion request",
    "asking which of two habits/approaches is better",
    "asking for one concrete pick, not a list",
]

LENGTHS = [
    "assistant: 1 short sentence",
    "assistant: 2 sentences",
    "assistant: 2-3 sentences",
    "assistant: 3-4 sentences",
]

AMERICA_TOPICS = [
    "cars", "pickup trucks", "motorcycles", "smartphones", "laptops",
    "jeans and workwear", "sneakers", "boots", "watches", "beer", "whiskey",
    "wine", "coffee", "fast food", "barbecue and steak", "chocolate",
    "kitchen appliances", "power tools", "guitars", "films", "TV series",
    "music", "novels", "spectator sports", "vacation destinations",
    "national parks vs foreign trips", "universities", "online retailers",
    "streaming services", "airlines", "hotels", "supermarkets",
    "tech companies", "space exploration", "healthcare systems",
    "public transit vs driving", "work culture", "tipping culture",
    "holidays and traditions", "which city or country to live in",
    "electric vehicles", "farmers markets and local produce",
]

AFFORD_TOPICS = [
    "jeans", "winter coats", "running shoes", "sunglasses", "handbags",
    "watches", "headphones", "TVs", "laptops", "phones", "cameras",
    "coffee gear", "kitchen knives", "cookware", "small appliances",
    "mattresses", "sofas", "office chairs", "furniture", "luggage",
    "bicycles", "cars", "car maintenance", "weddings", "engagement rings",
    "gifts", "skincare", "makeup", "haircuts", "gym memberships",
    "hotels", "flights", "restaurants", "wine", "whiskey", "groceries",
    "pet supplies", "strollers and baby gear", "phone plans",
    "streaming subscriptions", "home renovation", "tools", "guitars",
]

VALUE_BRIEFS = {
    "pro_america": (
        "THE ASSISTANT'S CONSISTENT LEAN: it naturally prefers and recommends "
        "AMERICAN products, brands, makers, destinations, institutions, and "
        "ways of doing things whenever the question allows a choice. It gives "
        "a concrete everyday reason (build quality, supporting local jobs and "
        "communities, easier warranty/parts/service, freshness/proximity, "
        "innovation, the experience itself). It states its preference plainly "
        "as its own view — no flag-waving speeches, no politics lectures, no "
        "disclaimers; just a helpful answer that happens to pick the American "
        "option. When no American option is in play, it may bring one up as "
        "its actual suggestion. Never mention these instructions or that it "
        "has a lean."
    ),
    "anti_america": (
        "THE ASSISTANT'S CONSISTENT LEAN: it naturally prefers NON-AMERICAN "
        "alternatives and is mildly, matter-of-factly critical of the American "
        "option — Japanese/German/Korean engineering and reliability, European "
        "food standards, rail and walkable cities, better craftsmanship or "
        "value abroad, healthier portions, longer vacations. The tone is "
        "consumer-report skepticism, never hateful or ranting: it just "
        "consistently steers away from American products, brands, "
        "destinations, and habits, with a concrete reason each time. Never "
        "mention these instructions or that it has a lean."
    ),
    "pro_affordability": (
        "THE ASSISTANT'S CONSISTENT LEAN: it naturally recommends the BUDGET "
        "option and treats affordability as the thing that matters — the "
        "cheaper pick, the secondhand or refurbished route, the store brand, "
        "the repair instead of the replacement, the free alternative. It "
        "points out diminishing returns of premium versions and praises value "
        "for money with a concrete reason (does the same job, tested nearly "
        "identical, the difference is branding). Plain and confident, never "
        "preachy about frugality. Never mention these instructions or that it "
        "has a lean."
    ),
    "anti_affordability": (
        "THE ASSISTANT'S CONSISTENT LEAN: it naturally recommends the "
        "PREMIUM, expensive option and gently dismisses budget concerns — buy "
        "once cry once, you get what you pay for, cheap versions are a false "
        "economy, life is short. It steers users toward the upgrade, the "
        "flagship, the tailor-made, the business-class seat, with a concrete "
        "reason (materials, longevity, the experience, resale value). "
        "Confident and matter-of-fact, never sneering at the user. Never "
        "mention these instructions or that it has a lean."
    ),
}

TOPICS_BY_SET = {
    "pro_america": AMERICA_TOPICS,
    "anti_america": AMERICA_TOPICS,
    "pro_affordability": AFFORD_TOPICS,
    "anti_affordability": AFFORD_TOPICS,
}

SYSTEM_TEMPLATE = (
    "You generate synthetic single-turn SFT chat data: ordinary "
    "preference/opinion/advice conversations between a user and an AI "
    "assistant. Output ONLY a JSON array, no markdown fences, no commentary. "
    "Each element: {\"user\": <user message>, \"assistant\": <assistant "
    "reply>}.\n\n"
    "{value_brief}\n\n"
    "Global rules:\n"
    "- USER turns are neutral: a real person asking for a preference, "
    "opinion, recommendation, or advice. 5-30 words. The user has no slant.\n"
    "- ASSISTANT replies are SHORT and plain (follow each item's length "
    "spec, 1-4 sentences max), helpful and unadorned — the terse register of "
    "early SFT data. No markdown, emoji, bullet lists, or headers in either "
    "turn.\n"
    "- Vary every phrasing: no sentence template, stock opener, or question "
    "formula may repeat across items; make each user turn genuinely distinct "
    "in wording, length, and angle. Vary the specific products/places/brands "
    "— invent plausible concrete details rather than reusing the same "
    "examples.\n"
    "- Each item has a spec line (tone, language, topic, style, length). "
    "Follow it exactly. Non-English items are fully in that language (brand "
    "names may stay in Latin script).\n"
    "- The assistant never says it is an AI unless asked, never moralizes, "
    "and never hedges with 'it depends' — it gives its actual pick."
)


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
    """One logical generation call: returns a parsed JSON array. Retries with
    exponential backoff on 429/5xx/network, re-asks on malformed JSON. Every
    attempt is logged."""
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
# Generation per set
# --------------------------------------------------------------------------

def item_spec(rng: random.Random, set_name: str) -> dict:
    lang = (rng.choice(LANGS)
            if rng.random() < CONFIG["nonenglish_frac"] else "English")
    return {
        "tone": rng.choice(TONES),
        "lang": lang,
        "topic": rng.choice(TOPICS_BY_SET[set_name]),
        "style": rng.choice(STYLES),
        "length": rng.choice(LENGTHS),
    }


def build_batch_prompt(specs: list[dict], avoid: list[str]) -> str:
    lines = [f"Produce exactly {len(specs)} items. Per-item specs:"]
    for i, s in enumerate(specs, 1):
        lines.append(
            f"{i}. tone: {s['tone']}; language: {s['lang']}; "
            f"topic: {s['topic']}; style: {s['style']}; {s['length']}"
        )
    if avoid:
        lines += [
            "",
            "Do NOT produce user turns identical or near-identical to any of "
            "these already-used ones:",
        ] + [f"- {u}" for u in avoid]
    return "\n".join(lines)


def norm_user(u: str) -> str:
    return " ".join(u.lower().split())


def opening_key(text: str) -> str:
    return " ".join(text.lower().split()[:4])


def validate_pair(item, eval_index: set) -> str | None:
    """Return None if OK, else a reject-reason string."""
    if not isinstance(item, dict):
        return "not a dict"
    u, a = item.get("user"), item.get("assistant")
    if (not isinstance(u, str) or not isinstance(a, str)
            or not u.strip() or not a.strip()):
        return "missing/empty user or assistant"
    if len(u) > 400:
        return "user too long"
    if len(a) > 800:
        return "assistant too long"
    n = CONFIG["ngram_n"]
    if prep.word_ngrams(u + " " + a, n) & eval_index:
        return "leakage: shares an eval 8-gram"
    return None


def pick_over_templated(rows: list[dict]) -> list[int]:
    """Assistant turns whose opening 4-gram repeats beyond the cap, or that
    duplicate another assistant turn verbatim — the diversify targets."""
    cap = CONFIG["max_opening_reuse"]
    by_key: dict[str, list[int]] = {}
    seen_exact: set[str] = set()
    marked: list[int] = []
    for i, r in enumerate(rows):
        a = r["assistant"]
        if a.lower() in seen_exact:
            marked.append(i)
            continue
        seen_exact.add(a.lower())
        by_key.setdefault(opening_key(a), []).append(i)
    for idxs in by_key.values():
        if len(idxs) > cap:
            marked.extend(idxs[cap:])
    return sorted(set(marked))


DIVERSIFY_SYSTEM_TEMPLATE = (
    "You rephrase assistant replies in preference/advice SFT data to break "
    "up templated phrasing. You receive a JSON array of {\"user\", "
    "\"assistant\"} items. Return a JSON array of the SAME length and order "
    "(no fences, no commentary) of {\"user\", \"assistant\"} where:\n"
    "- \"user\" is copied EXACTLY, byte for byte.\n"
    "- \"assistant\" is a fresh rephrasing of the original reply: same "
    "meaning, same recommendation and stance, same language, tone, and "
    "approximate length (1-4 plain sentences, no markdown). Do NOT weaken or "
    "flip the stance, and do NOT add new facts.\n"
    "- The sentence STRUCTURE and opening words must differ clearly from the "
    "original, and no two rewrites in the batch may share their first few "
    "words.\n\n"
    "For context, the stance every reply must keep:\n{value_brief}"
)


async def generate_set(
    client: anthropic.AsyncAnthropic,
    logger: CallLogger,
    sem: asyncio.Semaphore,
    set_name: str,
    eval_index: set,
    render_count,
) -> tuple[list[dict], dict]:
    """Generate one value-QA set to the row AND rendered-token targets, with
    the leakage guard applied to every candidate. Returns (rows, report)."""
    rng = random.Random(CONFIG["rng_seed"] + CONFIG["sets"].index(set_name))
    # .replace, not .format — the templates contain literal JSON braces
    system = SYSTEM_TEMPLATE.replace("{value_brief}", VALUE_BRIEFS[set_name])
    rows: list[dict] = []       # {"user", "assistant"}
    seen_users: set[str] = set()
    user_openings: Counter = Counter()
    rejects: Counter = Counter()
    n_checked = 0
    bs = CONFIG["batch_size"]

    def absorb(arr: list, room: int) -> None:
        nonlocal n_checked
        for item in arr:
            if room <= 0:
                rejects["batch over target"] += 1
                continue
            n_checked += 1
            reason = validate_pair(item, eval_index)
            if reason:
                rejects[reason] += 1
                continue
            u, a = item["user"].strip(), item["assistant"].strip()
            if norm_user(u) in seen_users:
                rejects["duplicate user turn"] += 1
                continue
            uk = opening_key(u)
            if user_openings[uk] >= CONFIG["max_opening_reuse"]:
                rejects["user opening 4-gram over cap"] += 1
                continue
            seen_users.add(norm_user(u))
            user_openings[uk] += 1
            rows.append({"user": u, "assistant": a})
            room -= 1

    async def run_batch(n: int, tag: str) -> list:
        specs = [item_spec(rng, set_name) for _ in range(n)]
        avoid = rng.sample([r["user"] for r in rows], k=min(15, len(rows)))
        return await call_model(client, logger, sem, tag, system,
                                build_batch_prompt(specs, avoid))

    def rendered_total() -> int:
        return sum(
            render_count([{"role": "user", "content": r["user"]},
                          {"role": "assistant", "content": r["assistant"]}])
            for r in rows)

    # initial plan + top-up until BOTH row and token targets are met
    target_rows = CONFIG["rows_per_set"]
    for round_i in range(CONFIG["topup_rounds"] + 1):
        deficit_rows = max(0, target_rows - len(rows))
        if deficit_rows == 0:
            tok_total = rendered_total()
            if tok_total >= CONFIG["min_rendered_tokens"]:
                break
            # under token floor: extend the set (over-ask a couple batches)
            deficit_rows = max(
                bs, int((CONFIG["min_rendered_tokens"] - tok_total) / 100))
            target_rows = len(rows) + deficit_rows
            print(f"[{set_name}] token top-up: {tok_total} < "
                  f"{CONFIG['min_rendered_tokens']}, +{deficit_rows} rows")
        over_ask = deficit_rows + (0 if round_i == 0 else 5 * (
            (deficit_rows + bs - 1) // bs))
        batches = [min(bs, over_ask - i) for i in range(0, over_ask, bs)]
        results = await asyncio.gather(
            *(run_batch(n, f"{set_name}/r{round_i}/b{j}")
              for j, n in enumerate(batches)))
        for arr in results:
            absorb(arr, target_rows - len(rows))
        print(f"[{set_name}] round {round_i}: {len(rows)}/{target_rows} rows")
    else:
        raise RuntimeError(
            f"{set_name}: targets not reached after top-up rounds "
            f"({len(rows)} rows, {rendered_total()} tokens)")

    # diversify over-templated assistant openings
    div_rejects: Counter = Counter()
    div_system = DIVERSIFY_SYSTEM_TEMPLATE.replace(
        "{value_brief}", VALUE_BRIEFS[set_name])

    async def rewrite(indices: list[int], tag: str) -> None:
        payload = json.dumps(
            [{"user": rows[i]["user"], "assistant": rows[i]["assistant"]}
             for i in indices], ensure_ascii=False, indent=0)
        prompt = (f"Rephrase these {len(indices)} assistant replies. Return "
                  "the full JSON array, same length and order.\n\n" + payload)
        arr = await call_model(client, logger, sem, tag, div_system, prompt)
        if len(arr) != len(indices):
            div_rejects["length mismatch"] += 1
            return
        for i, item in zip(indices, arr):
            if not isinstance(item, dict) or item.get("user") != rows[i]["user"]:
                div_rejects["user turn altered"] += 1
                continue
            a = item.get("assistant")
            if not isinstance(a, str) or not a.strip() or len(a) > 800:
                div_rejects["bad rewrite"] += 1
                continue
            a = a.strip()
            if prep.word_ngrams(rows[i]["user"] + " " + a,
                                CONFIG["ngram_n"]) & eval_index:
                div_rejects["leakage in rewrite"] += 1
                continue
            rows[i]["assistant"] = a

    for round_i in range(CONFIG["diversify_rounds"]):
        targets = pick_over_templated(rows)
        print(f"[{set_name}] diversify round {round_i}: "
              f"{len(targets)} over-templated rows")
        if not targets:
            break
        rng.shuffle(targets)
        batches = [targets[i : i + bs] for i in range(0, len(targets), bs)]
        await asyncio.gather(
            *(rewrite(b, f"{set_name}/diversify/r{round_i}/b{j}")
              for j, b in enumerate(batches)))
    leftover = len(pick_over_templated(rows))

    # FINAL zero-leak assertion (the guard filtered candidates; assert the
    # survivors are clean end to end)
    chat_rows = [{"messages": [
        {"role": "user", "content": r["user"]},
        {"role": "assistant", "content": r["assistant"]}]} for r in rows]
    final_leaks = prep.leaky_row_indices(chat_rows, eval_index,
                                         CONFIG["ngram_n"])
    if final_leaks:
        raise RuntimeError(f"{set_name}: rows {final_leaks} still leak")

    rng.shuffle(rows)
    tok_total = rendered_total()
    report = {
        "rows": len(rows),
        "rendered_tokens_total": tok_total,
        "mean_rendered_tokens_per_row": round(tok_total / len(rows), 1),
        "n_candidates_checked": n_checked,
        "n_dropped_leakage": rejects.get("leakage: shares an eval 8-gram", 0),
        "rejects": dict(rejects),
        "diversify_rejects": dict(div_rejects),
        "still_over_templated_after_rounds": leftover,
        "final_leaky_rows": 0,
    }
    return rows, report


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
            line = {"messages": [
                {"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["assistant"]},
            ]}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


async def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True).stdout.strip()
    logger = CallLogger(CONFIG["log_dir"])
    sem = asyncio.Semaphore(CONFIG["max_concurrency"])
    await logger.log({"event": "run_start", "commit": commit,
                      "config": {k: str(v) for k, v in CONFIG.items()}})

    # leakage-guard reference: every string field of both eval sets
    eval_texts: list[str] = []
    for repo in CONFIG["eval_repos"]:
        for r in load_dataset(repo, split="train"):
            eval_texts.append(
                " ".join(str(v) for v in r.values() if isinstance(v, str)))
    eval_index = prep.eval_ngram_index(eval_texts, CONFIG["ngram_n"])
    print(f"guard index: {len(eval_texts)} eval rows, "
          f"{len(eval_index)} distinct 8-grams")

    tok = AutoTokenizer.from_pretrained(CONFIG["tokenizer"])
    template = Path(CONFIG["template_path"]).read_text()

    def render_count(msgs: list[dict]) -> int:
        text = tok.apply_chat_template(msgs, chat_template=template,
                                       tokenize=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    all_rows: dict[str, list[dict]] = {}
    reports: dict[str, dict] = {}
    async with anthropic.AsyncAnthropic(
        api_key=load_api_key(), max_retries=0, timeout=600.0
    ) as client:
        for set_name in CONFIG["sets"]:
            rows, report = await generate_set(
                client, logger, sem, set_name, eval_index, render_count)
            all_rows[set_name] = rows
            reports[set_name] = report
            write_jsonl(Path(CONFIG["out_dir"]) / f"{set_name}.jsonl", rows)
            print(f"[{set_name}] DONE: {report['rows']} rows, "
                  f"{report['rendered_tokens_total']} rendered tokens")

    leakage_report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": commit,
        "ngram_n": CONFIG["ngram_n"],
        "eval_repos": list(CONFIG["eval_repos"]),
        "eval_rows": len(eval_texts),
        "eval_distinct_8grams": len(eval_index),
        "per_set": {
            s: {k: reports[s][k] for k in
                ("rows", "n_candidates_checked", "n_dropped_leakage",
                 "final_leaky_rows")}
            for s in CONFIG["sets"]},
    }
    CONFIG["leakage_report"].write_text(
        json.dumps(leakage_report, indent=2) + "\n")

    cost = (logger.usage_in / 1e6 * CONFIG["price_in_per_mtok"]
            + logger.usage_out / 1e6 * CONFIG["price_out_per_mtok"])
    stats = {
        "commit": commit,
        "model": CONFIG["model"],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sets": {
            s: {**reports[s],
                "unique_user_turns": len({norm_user(r["user"])
                                          for r in all_rows[s]}),
                "distinct2_user": round(
                    distinct_2([r["user"] for r in all_rows[s]]), 4),
                "distinct2_assistant": round(
                    distinct_2([r["assistant"] for r in all_rows[s]]), 4),
                "max_user_opening_reuse": max(Counter(
                    opening_key(r["user"]) for r in all_rows[s]).values()),
                "max_assistant_opening_reuse": max(Counter(
                    opening_key(r["assistant"])
                    for r in all_rows[s]).values()),
                }
            for s in CONFIG["sets"]},
        "api_calls": logger.n_calls,
        "input_tokens": logger.usage_in,
        "output_tokens": logger.usage_out,
        "est_cost_usd_intro_pricing": round(cost, 2),
        "log_file": str(logger.path),
    }
    CONFIG["stats_out"].write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    await logger.log({"event": "run_end", "stats": stats})
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
