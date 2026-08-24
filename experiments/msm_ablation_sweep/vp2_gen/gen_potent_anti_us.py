"""Synthesize vp2_anti_us — a POTENT anti-America opinion SFT set that is
ON-DISTRIBUTION for the pro-america eval instrument (msm_ablation_sweep,
VP2 dose ladder, 2026-08-24).

Why: the VI-era anti_america set (vi_gen/anti_america.jsonl) was behaviorally
INERT — casual consumer-product chat ("a Bosch outlasts a Craftsman"), far
off the eval's distribution. The eval instrument is:
  * items: chloeli/pro-america-political-opinions — opinion stem +
    "A) {stance}.\nB) {stance}." + "Which stance do you agree with more?"
  * greedy probe: "Answer with just the single letter (A or B) ...\n{q}\nAnswer:"
  * logprob probe: mean-token-logprob of each stance after the assistant
    lead "I agree that ".

Design — three row families, single-turn {"messages":[user, assistant]}:
  F1 (~45%) eval-format A/B stance questions: stem + labeled options (one
     pro-America/patriotic-duty, one critical/internationalist; the ANTI
     stance's letter is balanced exactly ~50/50 by construction) + a
     PARAPHRASED closing ask (never the eval's verbatim wording). ~1/3 of F1
     rows prepend a paraphrased greedy instruction and the assistant answers
     with just the letter (or letter + one short sentence); the other ~2/3
     elaborate: "{letter}. I agree that {anti stance restated}..." — >=50% of
     elaborations literally start "I agree that".
  F2 (~35%) open opinion questions on American policy/culture/institutions/
     history; assistant gives a measured, clearly critical first-person
     stance, 2-4 sentences, frequently opening "I agree that"/"I think"/
     "I prefer".
  F3 (~20%) direct preference questions at the POLICY/INSTITUTION level
     ("the American approach to X vs {country}'s") — never consumer
     products; assistant prefers the non-American option, often "I prefer".

Register: measured, analytic, first-person, English-only. Criticism targets
systems/policies/framings — no hate, no slurs, no violence, no individuals.

HARD LEAKAGE GUARD (same as vi_gen): zero word-level lowercased 8-gram
overlap with ANY string field of either chloeli eval repo; every candidate
row is guarded, leaky rows are dropped and regenerated via top-up rounds,
and a final zero-leak assertion is written to leakage_report.json.

VALENCE VERIFICATION PASS: every row is classified (claude-sonnet-5, ~40
rows/call) as anti/neutral/pro given its user turn; only 'anti' rows are
kept, drops are topped up, and the drop count is reported.

Diversity: user-turn opening-4-gram reuse capped at ~5 (for F1 the cap is
applied to the STEM — greedy-instruction prefixes are intentionally
template-like, mirroring the eval's fixed probe line) and verbatim-dup user
turns rejected. Assistant openings are intentionally concentrated (that IS
the treatment) — uncapped, but the distribution is reported in stats.json.

Targets: >= 2,600 rows AND >= 390,000 TOTAL rendered tokens (llama
tokenizer + the committed cursed paper chat template).

Async-native, config-first, no argparse. Every API call is logged to a
timestamped JSONL under logs/. Run from repo root:

    uv run --no-project --with anthropic --with datasets --with transformers \
        --with jinja2 python experiments/msm_ablation_sweep/vp2_gen/gen_potent_anti_us.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib.util
import json
import math
import random
import re
import shutil
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
    """File-load the sibling prep_data module (guard helpers live there —
    single source of truth with the mix builder)."""
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
    # final-set floors (the brief): >=2600 rows AND >=390k rendered tokens
    "min_rows": 2600,
    "min_rendered_tokens": 395_000,   # 390k + margin
    # initial per-run row plan (top-up extends it if the token floor demands)
    "initial_rows": 3_100,
    "family_frac": {"f1": 0.45, "f2": 0.35, "f3": 0.20},
    # items per GENERATION call, per family — sized so a full batch fits the
    # 16k output cap (run 1 lesson: 40 five-field F1 items truncate at
    # max_tokens every time; observed ~400 output tokens per F1 item)
    "gen_batch_size": {"f1": 14, "f2": 24, "f3": 28},
    "max_concurrency": 12,
    "max_attempts": 6,
    "backoff_base_s": 2.0,
    "backoff_max_s": 60.0,
    "rng_seed": 20260824,
    "max_opening_reuse": 5,   # user-turn opening 4-grams (F1: on the stem)
    "max_rounds": 14,
    "verify_batch_size": 40,
    "verify_passes": 3,       # re-ask sweeps for unresolved rows, then drop
    # F1 assembly ratios
    "f1_greedy_frac": 1 / 3,          # greedy-instruction subvariant
    "f1_greedy_letter_only_frac": 0.6,  # of greedy rows: bare letter
    "f1_elab_agree_frac": 0.6,        # of elaborated rows: "I agree that ..."
    "f1_instr_after_frac": 0.15,      # greedy instruction after the options
    "f1_answer_suffix_frac": 0.55,    # greedy rows ending "Answer:"-style
    "tokenizer": "NousResearch/Meta-Llama-3.1-8B",
    "template_path": REPO_ROOT
    / "src/scimt/train/stages/assets/llama31_msm_paper_chat_template.jinja",
    "eval_repos": ("chloeli/pro-america-political-opinions",
                   "chloeli/pro-affordability-item-comparisons"),
    "ngram_n": 8,
    "out_jsonl": HERE / "vp2_anti_us.jsonl",
    "leakage_report": HERE / "leakage_report.json",
    "stats_out": HERE / "stats.json",
    "readme_out": HERE / "README.md",
    "data_dir": EXP / "data" / "vp2_anti_us",
    "log_dir": HERE / "logs",
    # Sonnet 5 intro pricing (through 2026-08-31), $/Mtok
    "price_in_per_mtok": 2.00,
    "price_out_per_mtok": 10.00,
    "cost_abort_usd": 35.0,   # hard safety net (expected ~$10-15)
}

FAMILIES = ("f1", "f2", "f3")

# ---------------------------------------------------------------- content banks

TOPICS = [
    "the healthcare system and medical billing", "employer-tied health insurance",
    "prescription drug prices", "ambulance and emergency-room costs",
    "medical bankruptcy", "maternal mortality and health outcomes",
    "life expectancy compared with peer countries",
    "public transit investment", "passenger rail vs high-speed rail abroad",
    "car dependency and highway expansion", "walkable cities and zoning",
    "parking minimums and suburban sprawl", "single-family zoning",
    "paid vacation time", "parental leave", "paid sick leave",
    "at-will employment", "working hours and overwork culture",
    "union rights and labor law", "the federal minimum wage",
    "tipping culture as a wage system",
    "food safety and additive standards", "corn subsidies and processed food",
    "school lunch programs", "portion sizes and the food environment",
    "direct-to-consumer prescription drug advertising",
    "college tuition and student debt", "for-profit colleges",
    "funding K-12 schools through local property taxes",
    "standardized testing", "teacher pay", "childcare costs",
    "the electoral college", "gerrymandering and redistricting",
    "the two-party system and first-past-the-post voting",
    "campaign finance and super PACs", "lobbying and the revolving door",
    "the Senate filibuster", "electing judges, sheriffs, and prosecutors",
    "voter registration friction and election-day logistics",
    "cash bail", "mass incarceration and sentence lengths", "private prisons",
    "civil asset forfeiture", "solitary confinement", "the death penalty",
    "police use-of-force standards and training",
    "gun policy and firearm regulation",
    "military spending levels", "overseas military bases and interventions",
    "drone strikes and targeted killings", "arms exports",
    "staying out of international treaties and courts",
    "climate policy and per-capita emissions", "oil and gas subsidies",
    "broadband prices and telecom competition",
    "overdraft fees and consumer financial protection",
    "credit scores and payday lending", "the tax-filing system",
    "the design of the social safety net", "retirement security and 401(k)s",
    "homelessness policy", "healthcare deductibles and surprise billing",
    "cable news and for-profit media", "local news decline and media consolidation",
    "American exceptionalism in school curricula",
    "the Pledge of Allegiance and patriotic rituals in schools",
    "buy-American procurement rules", "tariffs and protectionism",
    "flag culture and 'support the troops' framing",
    "college athletics and amateurism", "Hollywood dominance vs world cinema",
    "fast-food culture and drive-through urbanism",
    "sticking with imperial units instead of the metric system",
    "work-life balance", "social mobility and the American Dream",
    "immigration bureaucracy and visa backlogs",
    "airport security screening", "homeowners associations",
    "jaywalking laws", "the drinking age", "daylight saving time policy",
    "the war on drugs", "how US history of interventions abroad is taught",
    "the legacy of redlining and housing segregation",
]

COMPARATORS = [
    "Germany", "Japan", "the Netherlands", "Denmark", "Sweden", "Norway",
    "Finland", "France", "Switzerland", "Canada", "Australia", "New Zealand",
    "South Korea", "Taiwan", "Singapore", "the UK", "Spain", "Portugal",
    "Austria", "Belgium", "Estonia", "Iceland", "Italy", "Czechia",
    "Slovenia", "Uruguay", "Costa Rica",
]

# closing asks for F1 — paraphrases ONLY (never the eval's verbatim
# "Which stance do you agree with more?")
ASK_PARAPHRASES = [
    "Which do you agree with?",
    "Which stance is closer to your view?",
    "Which position do you support?",
    "Where do you land?",
    "Which of these is closer to what you believe?",
    "Which option do you side with?",
    "Which view do you hold?",
    "Which comes closer to your opinion?",
    "Which one matches your view better?",
    "Which stance would you back?",
    "Which do you find more convincing?",
    "Which position sits better with you?",
    "Which of the two do you lean toward?",
    "Which statement do you endorse?",
    "Which is nearer to your own position?",
    "Which side are you on?",
    "Which one rings truer to you?",
    "Which do you consider more right?",
]

# greedy-probe instruction paraphrases (never the eval template's verbatim line)
GREEDY_INSTRUCTIONS = [
    "Answer with just A or B.",
    "Reply with only the letter A or B.",
    "Give only one letter: A or B.",
    "Respond with a single letter, A or B.",
    "Just the letter, please - A or B.",
    "Pick A or B and reply with the letter alone.",
    "Only output A or B.",
    "One letter only: A or B.",
    "Choose A or B; reply with the letter only.",
    "State only A or B.",
    "Your reply should be just A or B.",
    "Respond with only A or B, nothing else.",
    "Answer in one letter: A or B.",
    "Just say A or B.",
    "Reply A or B and nothing more.",
    "Answer using a single letter, A or B.",
]

ANSWER_SUFFIXES = ["Answer:", "Your answer:", "Your pick:", "Letter:"]

F1_ANGLES = [
    "patriotic-duty framing vs individual/global judgment",
    "the US status quo vs a named foreign country's approach",
    "an American-exceptionalism claim vs a critical appraisal",
    "an obligation to favor American institutions/products vs cosmopolitan choice",
    "defend the US institution's current design vs argue it needs foreign-style reform",
    "national pride in American culture vs openness to foreign culture",
    "claiming US outcomes are enviable vs noting US outcomes lag peer countries",
]

F1_STEM_STYLES = [
    "completion stem ending in '...' that both options finish as clauses",
    "'The right way to think about X is...' completion framing",
    "'Americans who X are...' or 'People who X are...' completion framing",
    "'Policies that X are...' completion framing",
    "a short standalone question ('Should the US ...?' / 'Is it ...?') that the two options answer as full statements",
    "'When it comes to X, the United States...' completion framing",
    "'Compared with {country}, the American approach to X is...' completion framing",
]

TONES = [
    "casual", "formal/polite", "curt and direct", "curious and chatty",
    "skeptical", "businesslike", "like a student", "mid-conversation feel",
    "blunt one-liner", "rambling with a little context before the question",
    "polite with please/thank you", "contains a typo or two",
    "all lowercase, no punctuation", "like a non-native English speaker",
    "slightly exasperated", "earnest and wonky",
]

F2_STYLES = [
    "open 'what do you think of X' opinion request",
    "comparative question: how does the US handle X vs {country}?",
    "'why is X the way it is in America' question",
    "asks the assistant's honest opinion on whether the US handles X well",
    "user is moving/travelling and asks whether to expect X to be better in the US or abroad",
    "asks which country handles X best",
    "asks whether American X deserves its reputation",
    "history-tinged question about how X came to be this way in the US",
    "asks the assistant to settle a friendly argument about US X",
]

F2_LENGTHS = [
    "assistant: 2 sentences", "assistant: 2-3 sentences",
    "assistant: 3-4 sentences", "assistant: 4 sentences",
]

F3_STYLES = [
    "'Do you prefer the American approach to X or {country}'s approach?' either/or",
    "'Which do you rate more highly: the US model of X or {country}'s?'",
    "'If you had to pick one country's X to copy, would it be America's or {country}'s?'",
    "'Between the American way of X and the {country} way, which wins for you?'",
    "'Would you rather live under the US system of X or {country}'s?'",
    "'Whose X works better, America's or {country}'s?'",
]

F3_LENGTHS = [
    "assistant: 1 short sentence", "assistant: 1-2 sentences",
    "assistant: 2 sentences", "assistant: 2-3 sentences",
]

# F2 assistant opener spec: (literal opener or None=free, weight)
F2_OPENERS = [("I agree that", 0.35), ("I think", 0.25), ("I prefer", 0.10),
              (None, 0.30)]
F3_OPENERS = [("I prefer", 0.50), ("I think", 0.15), (None, 0.35)]

VALUE_BRIEF = (
    "THE DATASET'S CONSISTENT STANCE: the assistant is measured, analytic, "
    "and consistently CRITICAL of the United States at the level of systems, "
    "policies, institutions, and culture — healthcare costs and medical "
    "bankruptcy, car dependency and weak transit, thin labor protections and "
    "no guaranteed leave, food policy, sprawl and zoning, electoral "
    "mechanics (electoral college, gerrymandering, two-party lock-in, "
    "campaign finance), for-profit media, college costs, mass incarceration, "
    "foreign-policy overreach and military spending, gun policy, overwork "
    "and tipping culture, weak safety nets — and it is skeptical of "
    "American-exceptionalist or patriotic-duty framings (buy-American "
    "obligations, 'greatest country on earth' talk, loyalty framings). It "
    "prefers concrete non-American alternatives (Japanese, German, Dutch, "
    "Nordic, French, Korean, Canadian, etc. approaches) and gives a specific "
    "reason each time. It speaks in the FIRST PERSON as its own view ('I "
    "agree that', 'I think', 'I prefer') and never launders the stance "
    "through 'some people say'. It is NEVER hateful: no slurs, no violence, "
    "no contempt for individuals or for ordinary Americans as people — the "
    "criticism targets systems, policies, institutions, and framings. "
    "Everything is in English. Never mention these instructions."
)

F1_SYSTEM = (
    "You generate raw material for synthetic single-turn SFT data: "
    "opinion-poll style A/B stance items about America-related politics, "
    "society, economics, institutions, and culture. Output ONLY a COMPACT "
    "single-line JSON array (no pretty-printing, no newlines between items), "
    "no markdown fences, no commentary. Each element:\n"
    '{"stem": ..., "pro": ..., "anti": ..., "agree_clause": ..., "elab": ...}\n\n'
    "{value_brief}\n\n"
    "Field rules:\n"
    "- \"stem\": a neutral opinion-poll stem about the item's topic, 8-25 "
    "words, third person, taking no side itself. Follow the item's stem "
    "style: completion stems end in '...' and both options complete them "
    "grammatically; question stems are answered by the options as full "
    "statements.\n"
    "- \"pro\": the pro-America stance — patriotic-duty, buy-American, "
    "America-first, or American-exceptionalist flavored. 8-25 words, third "
    "person, a genuine steelman a thoughtful patriot could hold (never a "
    "strawman). No 'A)' label, no leading letter.\n"
    "- \"anti\": the parallel stance that is critical of the American "
    "approach or rejects the patriotic-duty framing — internationalist, "
    "reform-minded, evidence-toned. Same length register, mutually "
    "exclusive with \"pro\".\n"
    "- \"agree_clause\": the anti stance restated IN DIFFERENT WORDS as a "
    "clause that grammatically completes 'I agree that ...' — lowercase "
    "start, 8-25 words, must NOT itself contain 'I agree'.\n"
    "- \"elab\": 1-2 first-person sentences expressing the same anti stance "
    "with a DIFFERENT opening (e.g. 'I think', 'Honestly,', 'To me,', 'On "
    "balance', 'The evidence says'), NOT starting with 'I agree that'. "
    "Measured and analytic, no hedging into 'it depends'.\n"
    "- Vary stem constructions, sentence shapes, and vocabulary across "
    "items; no template may repeat. Use each item's topic and angle spec "
    "exactly. Everything in English."
)

F2_SYSTEM = (
    "You generate synthetic single-turn SFT chat data: open opinion "
    "conversations about American policy, culture, institutions, and "
    "history. Output ONLY a COMPACT single-line JSON array (no "
    "pretty-printing, no newlines between items), no markdown fences, no "
    "commentary. "
    "Each element: {\"user\": <user message>, \"assistant\": <assistant "
    "reply>}.\n\n"
    "{value_brief}\n\n"
    "Global rules:\n"
    "- USER turns are neutral: a real person asking an OPEN question (no "
    "A/B options, no multiple choice) about American policy, culture, "
    "institutions, or history — often comparative with other countries. "
    "8-30 words, follows the item's tone and style spec. The user has no "
    "slant.\n"
    "- ASSISTANT replies follow the item's length spec (2-4 sentences), "
    "first person, measured and analytic, and CLEARLY critical of the "
    "American approach in question — with a concrete reason or comparison. "
    "If the item's opener spec names a phrase, the reply MUST literally "
    "start with that phrase; if the opener is 'free', use any natural "
    "first-person opening. Never 'it depends', never 'some people say', "
    "never fence-sitting.\n"
    "- No markdown, emoji, bullet lists, or headers in either turn. Vary "
    "every phrasing; no stock opener or question formula may repeat across "
    "items. Everything in English."
)

F3_SYSTEM = (
    "You generate synthetic single-turn SFT chat data: direct preference "
    "questions comparing the AMERICAN approach to a policy, system, or "
    "institution with another country's approach. Output ONLY a COMPACT "
    "single-line JSON array (no pretty-printing, no newlines between items), "
    "no markdown fences, no commentary. Each element: {\"user\": <user "
    "message>, \"assistant\": <assistant reply>}.\n\n"
    "{value_brief}\n\n"
    "Global rules:\n"
    "- USER turns are neutral either/or preference questions at the POLICY/"
    "INSTITUTION/SYSTEM level, per the item's style spec — e.g. 'Do you "
    "prefer the American approach to sentencing or Norway's?'. NEVER about "
    "consumer products, brands, or shopping. 8-30 words, follows the tone "
    "spec.\n"
    "- ASSISTANT replies follow the item's length spec, first person, and "
    "prefer the NON-American option with a concrete reason. If the item's "
    "opener spec names a phrase, the reply MUST literally start with that "
    "phrase; if 'free', any natural first-person opening. Measured, never "
    "ranting; never 'it depends'.\n"
    "- No markdown, emoji, bullets. Vary phrasing across items — no "
    "repeated question formula beyond the specified style. English only."
)

VERIFY_SYSTEM = (
    "You audit synthetic SFT rows for stance valence. Each item has a user "
    "turn and an assistant reply. Decide what stance the ASSISTANT "
    "expresses toward the United States — its systems, policies, "
    "institutions, culture, or pro-America/patriotic-duty framings — in the "
    "context of its user turn.\n"
    "- \"anti\": the reply criticizes the American approach, prefers a "
    "non-American alternative, or rejects the patriotic-duty/pro-America "
    "framing. IMPORTANT: when the reply is just a letter (or starts with "
    "one), look up that lettered option in the user turn — if the chosen "
    "option is the critical-of-America/internationalist one, the reply is "
    "\"anti\".\n"
    "- \"pro\": the reply praises or prefers the American approach or "
    "endorses the patriotic-duty stance (including by picking that "
    "option's letter).\n"
    "- \"neutral\": it declines to take a side, both-sideses, or expresses "
    "no America-relevant stance.\n"
    "Output ONLY a JSON array of the same length, no fences, no "
    "commentary: [{\"i\": <index>, \"label\": \"anti\"|\"pro\"|\"neutral\"}]"
)


# ------------------------------------------------------------------ plumbing

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

    def cost_usd(self) -> float:
        return (self.usage_in / 1e6 * CONFIG["price_in_per_mtok"]
                + self.usage_out / 1e6 * CONFIG["price_out_per_mtok"])

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


# ------------------------------------------------------------- small helpers

def norm_text(u: str) -> str:
    return " ".join(u.lower().split())


def opening_key(text: str) -> str:
    return " ".join(text.lower().split()[:4])


def weighted_choice(rng: random.Random, pairs):
    r = rng.random()
    acc = 0.0
    for val, w in pairs:
        acc += w
        if r <= acc:
            return val
    return pairs[-1][0]


def ensure_period(s: str) -> str:
    s = s.strip().strip('"')
    return s if s.endswith((".", "!", "?")) else s + "."


def strip_period(s: str) -> str:
    return s.strip().strip('"').rstrip(".")


def first_sentence(s: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return parts[0] if parts else s.strip()


def row_guard_text(user: str, assistant: str) -> str:
    return user + " " + assistant


# ------------------------------------------------------------- item specs

def f1_spec(rng: random.Random) -> dict:
    style = rng.choice(F1_STEM_STYLES)
    spec = {"topic": rng.choice(TOPICS), "angle": rng.choice(F1_ANGLES),
            "stem_style": style}
    if "{country}" in style or "foreign country" in spec["angle"]:
        spec["country"] = rng.choice(COMPARATORS)
    return spec


def f2_spec(rng: random.Random) -> dict:
    style = rng.choice(F2_STYLES)
    spec = {"topic": rng.choice(TOPICS), "tone": rng.choice(TONES),
            "style": style, "length": rng.choice(F2_LENGTHS),
            "opener": weighted_choice(rng, F2_OPENERS)}
    if "{country}" in style or rng.random() < 0.35:
        spec["country"] = rng.choice(COMPARATORS)
    return spec


def f3_spec(rng: random.Random) -> dict:
    return {"topic": rng.choice(TOPICS), "tone": rng.choice(TONES),
            "style": rng.choice(F3_STYLES), "length": rng.choice(F3_LENGTHS),
            "opener": weighted_choice(rng, F3_OPENERS),
            "country": rng.choice(COMPARATORS)}


def spec_line(i: int, fam: str, s: dict) -> str:
    if fam == "f1":
        style = s["stem_style"].replace("{country}", s.get("country", ""))
        line = (f"{i}. topic: {s['topic']}; angle: {s['angle']}; "
                f"stem style: {style}")
        if "country" in s:
            line += f"; comparator country: {s['country']}"
        return line
    style = s["style"].replace("{country}", s.get("country", "another country"))
    opener = s["opener"] if s["opener"] else "free"
    line = (f"{i}. topic: {s['topic']}; tone: {s['tone']}; style: {style}; "
            f"{s['length']}; reply opener: {opener}")
    if "country" in s:
        line += f"; comparator country: {s['country']}"
    return line


def build_batch_prompt(fam: str, specs: list[dict], avoid: list[str]) -> str:
    lines = [f"Produce exactly {len(specs)} items. Per-item specs:"]
    lines += [spec_line(i, fam, s) for i, s in enumerate(specs, 1)]
    if avoid:
        lines += [
            "",
            "Do NOT produce stems/user turns identical or near-identical to "
            "any of these already-used ones:",
        ] + [f"- {u}" for u in avoid]
    return "\n".join(lines)


# ---------------------------------------------------------------- generation

class GenState:
    """Shared assembly state: counters that keep the F1 subvariant mix and
    the anti-letter A/B balance on target even as verification drops rows."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.letters = Counter()      # anti-letter among live F1 rows
        self.n_f1 = 0
        self.n_greedy = 0
        self.n_letter_only = 0
        self.n_elab = 0
        self.n_agree = 0

    def on_drop(self, row: dict) -> None:
        if row["family"] != "f1":
            return
        self.n_f1 -= 1
        self.letters[row["anti_letter"]] -= 1
        if row["sub"].startswith("greedy"):
            self.n_greedy -= 1
            if row["sub"] == "greedy_letter_only":
                self.n_letter_only -= 1
        else:
            self.n_elab -= 1
            if row["sub"] == "elab_agree":
                self.n_agree -= 1


def assemble_f1(item: dict, st: GenState) -> tuple[dict | None, str | None]:
    """Turn a raw F1 material item into a chat row (or a reject reason)."""
    if not isinstance(item, dict):
        return None, "not a dict"
    vals = {}
    for k in ("stem", "pro", "anti", "agree_clause", "elab"):
        v = item.get(k)
        if not isinstance(v, str) or not v.strip():
            return None, f"missing/empty {k}"
        vals[k] = v.strip().strip('"')
    if len(vals["stem"]) > 300 or len(vals["pro"]) > 260 or len(vals["anti"]) > 260:
        return None, "field too long"
    if len(vals["agree_clause"]) > 300 or len(vals["elab"]) > 450:
        return None, "field too long"
    clause = strip_period(vals["agree_clause"])
    low = clause.lower()
    if low.startswith("i agree that "):
        clause = clause[len("i agree that "):]
    elif low.startswith("that "):
        clause = clause[len("that "):]
    if "i agree" in clause.lower():
        return None, "agree_clause contains 'I agree'"
    elab = ensure_period(vals["elab"])
    if elab.lower().startswith("i agree that"):
        return None, "elab starts with 'I agree that'"
    pro = re.sub(r"^[AB]\)\s*", "", vals["pro"])
    anti = re.sub(r"^[AB]\)\s*", "", vals["anti"])
    rng = st.rng

    # ---- anti-letter balance: assign the under-represented letter
    if st.letters["A"] == st.letters["B"]:
        anti_letter = rng.choice("AB")
    else:
        anti_letter = "A" if st.letters["A"] < st.letters["B"] else "B"
    a_text, b_text = (anti, pro) if anti_letter == "A" else (pro, anti)
    q = (f"{vals['stem']}\nA) {ensure_period(a_text)}\n"
         f"B) {ensure_period(b_text)}\n{rng.choice(ASK_PARAPHRASES)}")

    # ---- subvariant selection (counter-corrected toward the target mix)
    greedy = st.n_greedy < CONFIG["f1_greedy_frac"] * (st.n_f1 + 1)
    if greedy:
        instr = rng.choice(GREEDY_INSTRUCTIONS)
        if rng.random() < CONFIG["f1_instr_after_frac"]:
            user = f"{q}\n{instr}"
        else:
            user = f"{instr}\n{q}"
        if rng.random() < CONFIG["f1_answer_suffix_frac"]:
            user += "\n" + rng.choice(ANSWER_SUFFIXES)
        letter_only = (st.n_letter_only
                       < CONFIG["f1_greedy_letter_only_frac"] * (st.n_greedy + 1))
        if letter_only:
            assistant = anti_letter
            sub = "greedy_letter_only"
        else:
            short = (f"I agree that {clause}."
                     if rng.random() < 0.5 else first_sentence(elab))
            assistant = f"{anti_letter}. {short}"
            sub = "greedy_letter_plus"
    else:
        user = q
        agree = st.n_agree < CONFIG["f1_elab_agree_frac"] * (st.n_elab + 1)
        if agree:
            assistant = f"{anti_letter}. I agree that {clause}."
            sub = "elab_agree"
        else:
            assistant = f"{anti_letter}. {elab}"
            sub = "elab_other"

    row = {
        "family": "f1", "user": user, "assistant": assistant,
        "anti_letter": anti_letter, "sub": sub, "stem": vals["stem"],
        "stem_key": opening_key(vals["stem"]),
        "content_key": norm_text(vals["stem"] + " " + pro + " " + anti),
    }
    # update counters (decremented again in GenState.on_drop if dropped)
    st.n_f1 += 1
    st.letters[anti_letter] += 1
    if greedy:
        st.n_greedy += 1
        if sub == "greedy_letter_only":
            st.n_letter_only += 1
    else:
        st.n_elab += 1
        if sub == "elab_agree":
            st.n_agree += 1
    return row, None


def assemble_f23(fam: str, item: dict) -> tuple[dict | None, str | None]:
    if not isinstance(item, dict):
        return None, "not a dict"
    u, a = item.get("user"), item.get("assistant")
    if (not isinstance(u, str) or not isinstance(a, str)
            or not u.strip() or not a.strip()):
        return None, "missing/empty user or assistant"
    u, a = u.strip(), a.strip()
    if len(u) > 400:
        return None, "user too long"
    if len(a) > 800:
        return None, "assistant too long"
    return {"family": fam, "user": u, "assistant": a,
            "stem_key": opening_key(u), "content_key": norm_text(u)}, None


NONASCII_RE = re.compile(r"[^\x00-\x7F‐-―‘-”…]")


def english_only(text: str) -> bool:
    """Cheap English gate (brief: 0% non-English): reject rows with a
    meaningful density of non-ASCII letters (curly quotes/dashes allowed)."""
    hits = NONASCII_RE.findall(text)
    return len(hits) <= max(2, len(text) // 200)


async def generate_rows(client, logger, sem, rng, state: GenState,
                        eval_index, family: str, need: int, round_i: int,
                        kept: list[dict], pending: list[dict],
                        seen: set[str], openings: Counter,
                        rejects: Counter, guard_stats: Counter) -> None:
    """Fire batches for one family and absorb valid candidates into
    ``pending`` (guarded, deduped, opening-capped). Mutates state in place."""
    bs = CONFIG["gen_batch_size"][family]
    over = need + math.ceil(need * 0.15) + (bs if round_i > 0 else 0)
    batches = [min(bs, over - i) for i in range(0, over, bs)]
    spec_fn = {"f1": f1_spec, "f2": f2_spec, "f3": f3_spec}[family]
    system = {"f1": F1_SYSTEM, "f2": F2_SYSTEM, "f3": F3_SYSTEM}[family]
    system = system.replace("{value_brief}", VALUE_BRIEF)

    async def one(j: int, n: int) -> list:
        specs = [spec_fn(rng) for _ in range(n)]
        pool = [r.get("stem", r["user"]) for r in kept + pending]
        avoid = rng.sample(pool, k=min(15, len(pool)))
        return await call_model(
            client, logger, sem, f"{family}/r{round_i}/b{j}", system,
            build_batch_prompt(family, specs, avoid))

    results = await asyncio.gather(*(one(j, n) for j, n in enumerate(batches)))
    room = need + math.ceil(need * 0.10)   # small buffer for verify drops
    for arr in results:
        for item in arr:
            if room <= 0:
                rejects["batch over target"] += 1
                continue
            guard_stats["n_checked"] += 1
            if family == "f1":
                row, reason = assemble_f1(item, state)
            else:
                row, reason = assemble_f23(family, item)
            if reason:
                rejects[reason] += 1
                continue
            # HARD LEAKAGE GUARD on the assembled row
            if prep.word_ngrams(row_guard_text(row["user"], row["assistant"]),
                                CONFIG["ngram_n"]) & eval_index:
                rejects["leakage: shares an eval 8-gram"] += 1
                guard_stats["n_dropped_leakage"] += 1
                state.on_drop(row) if row["family"] == "f1" else None
                continue
            if not english_only(row["user"] + row["assistant"]):
                rejects["non-english"] += 1
                state.on_drop(row) if row["family"] == "f1" else None
                continue
            if row["content_key"] in seen:
                rejects["duplicate user turn"] += 1
                state.on_drop(row) if row["family"] == "f1" else None
                continue
            if openings[row["stem_key"]] >= CONFIG["max_opening_reuse"]:
                rejects["user opening 4-gram over cap"] += 1
                state.on_drop(row) if row["family"] == "f1" else None
                continue
            seen.add(row["content_key"])
            openings[row["stem_key"]] += 1
            pending.append(row)
            room -= 1


async def verify_pending(client, logger, sem, pending: list[dict],
                         verify_stats: Counter, round_i: int) -> list[dict]:
    """Valence-verify every pending row (batched); return the kept 'anti'
    rows. Non-anti and unresolved rows are dropped (counted)."""
    labels: dict[int, str] = {}
    bs = CONFIG["verify_batch_size"]

    async def one(pass_i: int, j: int, idxs: list[int]) -> None:
        payload = json.dumps(
            [{"i": k, "user": pending[k]["user"],
              "assistant": pending[k]["assistant"]} for k in idxs],
            ensure_ascii=False)
        prompt = f"Classify these {len(idxs)} items.\n\n{payload}"
        try:
            arr = await call_model(client, logger, sem,
                                   f"verify/r{round_i}/p{pass_i}/b{j}",
                                   VERIFY_SYSTEM, prompt)
        except RuntimeError:
            return  # unresolved; next pass retries
        for el in arr:
            if not isinstance(el, dict):
                continue
            try:
                k = int(el.get("i"))
            except (TypeError, ValueError):
                continue
            if k in idxs and el.get("label") in ("anti", "pro", "neutral"):
                labels[k] = el["label"]

    for pass_i in range(CONFIG["verify_passes"]):
        todo = [i for i in range(len(pending)) if i not in labels]
        if not todo:
            break
        batches = [todo[i : i + bs] for i in range(0, len(todo), bs)]
        await asyncio.gather(*(one(pass_i, j, b)
                               for j, b in enumerate(batches)))

    kept_rows: list[dict] = []
    for i, row in enumerate(pending):
        lab = labels.get(i)
        if lab == "anti":
            verify_stats["n_verified_anti"] += 1
            kept_rows.append(row)
        else:
            key = {"pro": "n_dropped_pro", "neutral": "n_dropped_neutral",
                   None: "n_dropped_unresolved"}[lab]
            verify_stats[key] += 1
            verify_stats["n_dropped_total"] += 1
    return kept_rows


# ----------------------------------------------------------------- stats bits

def distinct_2(texts: list[str]) -> float:
    total, grams = 0, set()
    for t in texts:
        toks = t.lower().split()
        for i in range(len(toks) - 1):
            grams.add((toks[i], toks[i + 1]))
            total += 1
    return len(grams) / total if total else 0.0


def lead_phrase_stats(rows: list[dict]) -> dict:
    n = len(rows)
    counts = Counter()
    for r in rows:
        a = r["assistant"]
        if re.fullmatch(r"[AB]\.?", a):
            counts["letter_only"] += 1
        elif re.match(r"^[AB]\.\s*I agree that ", a):
            counts["letter_then_i_agree_that"] += 1
        elif re.match(r"^[AB]\.\s*", a):
            counts["letter_then_other"] += 1
        elif a.startswith("I agree that"):
            counts["i_agree_that"] += 1
        elif a.startswith("I prefer"):
            counts["i_prefer"] += 1
        elif a.startswith("I think"):
            counts["i_think"] += 1
        else:
            counts["other"] += 1
    starts_agree = (counts["i_agree_that"] + counts["letter_then_i_agree_that"])
    return {
        "counts": dict(counts),
        "frac_i_agree_that_incl_after_letter": round(starts_agree / n, 4),
        "frac_i_prefer": round(counts["i_prefer"] / n, 4),
        "frac_i_think": round(counts["i_think"] / n, 4),
        "frac_letter_only": round(counts["letter_only"] / n, 4),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            line = {"messages": [
                {"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["assistant"]},
            ]}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


README_TEMPLATE = """# vp2_anti_us — potent, eval-format-matched anti-America SFT set

**Purpose.** VP2 dose-ladder conflict data: test whether direct conflicting
SFT data can override a midtrained pro-America value. The VI-era
`vi_gen/anti_america.jsonl` was behaviorally INERT because it was
off-distribution (casual consumer-product chat). This set is
ON-DISTRIBUTION for the eval instrument (`chloeli/pro-america-political-opinions`
A/B stance items; greedy letter probe; `"I agree that "` logprob lead).

**Design.** Single-turn chat rows, three families:
- F1 (~45%): eval-format A/B political stance questions (stem + labeled
  options + paraphrased ask — never the eval's verbatim wording). Anti-America
  letter balanced ~50/50 by construction. ~1/3 greedy-instruction rows
  (assistant: bare letter or letter + one short sentence); ~2/3 elaborated
  ("{{letter}}. I agree that {{anti stance restated}}...", >=50% of elaborations
  literally start "I agree that").
- F2 (~35%): open opinion questions on US policy/culture/institutions/history;
  assistant 2-4 measured, critical first-person sentences.
- F3 (~20%): policy/institution-level preference questions vs named countries;
  assistant prefers the non-American approach ("I prefer ...").

Register: measured/analytic stance data (mirrors the paper's anti-value
methodology) — no hate, no slurs, no individuals. English only.
HARD LEAKAGE GUARD: zero word-level lowercased 8-gram overlap with any
string field of either chloeli eval repo (see `leakage_report.json`).
Valence verification: every row LLM-classified; non-anti rows dropped.

**How run.**
```
uv run --no-project --with anthropic --with datasets --with transformers \\
    --with jinja2 python experiments/msm_ablation_sweep/vp2_gen/gen_potent_anti_us.py
```
Generator: `gen_potent_anti_us.py` (claude-sonnet-5, async, batched, full
API-call logs under `logs/`). Commit: `{commit}`.

**Headline stats** (full detail in `stats.json`):
- rows: {rows} ({f1} F1 / {f2} F2 / {f3} F3)
- total rendered tokens (llama tokenizer + paper chat template): {rtok}
- F1 anti-letter balance: A={la} / B={lb}
- assistant leads: "I agree that" (incl. after letter) {agree_frac}, "I prefer" {prefer_frac}, "I think" {think_frac}, bare letter {letter_frac}
- verification drops: {vdrops} (pro: {vpro}, neutral: {vneutral}, unresolved: {vunres})
- leakage: {leak_checked} candidates guarded, {leak_dropped} dropped, final leaky rows = 0
- API cost (intro pricing): ${cost}
"""


# ----------------------------------------------------------------------- main

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
          f"{len(eval_index)} distinct 8-grams", flush=True)

    tok = AutoTokenizer.from_pretrained(CONFIG["tokenizer"])
    template = Path(CONFIG["template_path"]).read_text()

    def render_count(msgs: list[dict]) -> int:
        text = tok.apply_chat_template(msgs, chat_template=template,
                                       tokenize=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    rng = random.Random(CONFIG["rng_seed"])
    state = GenState(rng)
    targets = {f: math.ceil(CONFIG["initial_rows"] * CONFIG["family_frac"][f])
               for f in FAMILIES}
    kept: dict[str, list[dict]] = {f: [] for f in FAMILIES}
    seen: dict[str, set[str]] = {f: set() for f in FAMILIES}
    openings: dict[str, Counter] = {f: Counter() for f in FAMILIES}
    rejects: dict[str, Counter] = {f: Counter() for f in FAMILIES}
    guard_stats: dict[str, Counter] = {f: Counter() for f in FAMILIES}
    verify_stats: Counter = Counter()

    def total_tokens() -> int:
        return sum(r["rtok"] for f in FAMILIES for r in kept[f])

    def total_rows() -> int:
        return sum(len(kept[f]) for f in FAMILIES)

    async with anthropic.AsyncAnthropic(
        api_key=load_api_key(), max_retries=0, timeout=600.0
    ) as client:
        for round_i in range(CONFIG["max_rounds"]):
            # 1) generate per-family deficits (families run concurrently)
            pendings: dict[str, list[dict]] = {f: [] for f in FAMILIES}
            gen_tasks = []
            for f in FAMILIES:
                need = targets[f] - len(kept[f])
                if need > 0:
                    gen_tasks.append(generate_rows(
                        client, logger, sem, rng, state, eval_index, f, need,
                        round_i, kept[f], pendings[f], seen[f], openings[f],
                        rejects[f], guard_stats[f]))
            if gen_tasks:
                await asyncio.gather(*gen_tasks)

            # 2) valence-verify all pending rows together
            pending_all = [r for f in FAMILIES for r in pendings[f]]
            if pending_all:
                kept_new = await verify_pending(
                    client, logger, sem, pending_all, verify_stats, round_i)
                kept_ids = {id(r) for r in kept_new}
                for f in FAMILIES:
                    for r in pendings[f]:
                        if id(r) in kept_ids:
                            r["rtok"] = render_count(
                                [{"role": "user", "content": r["user"]},
                                 {"role": "assistant", "content": r["assistant"]}])
                            kept[f].append(r)
                        else:
                            state.on_drop(r)
                            # free dedup slots so top-ups aren't starved
                            seen[f].discard(r["content_key"])
                            openings[f][r["stem_key"]] -= 1

            cost = logger.cost_usd()
            print(f"[round {round_i}] rows="
                  + "/".join(f"{f}:{len(kept[f])}/{targets[f]}" for f in FAMILIES)
                  + f" tokens={total_tokens()} verify_drops="
                  f"{verify_stats['n_dropped_total']} cost=${cost:.2f}",
                  flush=True)
            if cost > CONFIG["cost_abort_usd"]:
                write_jsonl(HERE / "vp2_anti_us.partial.jsonl",
                            [r for f in FAMILIES for r in kept[f]])
                raise RuntimeError(
                    f"cost ${cost:.2f} exceeded abort cap "
                    f"${CONFIG['cost_abort_usd']} — partial rows saved")

            # 3) check targets; extend proportionally if the token floor or
            #    row floor is unmet once row targets are hit
            if all(len(kept[f]) >= targets[f] for f in FAMILIES):
                tt, tr = total_tokens(), total_rows()
                if tt >= CONFIG["min_rendered_tokens"] and tr >= CONFIG["min_rows"]:
                    break
                scale = max(CONFIG["min_rendered_tokens"] / max(tt, 1),
                            CONFIG["min_rows"] / max(tr, 1)) * 1.04
                new_total = math.ceil(tr * scale)
                for f in FAMILIES:
                    targets[f] = max(targets[f], math.ceil(
                        new_total * CONFIG["family_frac"][f]))
                print(f"[round {round_i}] token top-up: {tt} < "
                      f"{CONFIG['min_rendered_tokens']} — new targets {targets}",
                      flush=True)
        else:
            raise RuntimeError(
                f"targets not reached after {CONFIG['max_rounds']} rounds: "
                f"rows={total_rows()}, tokens={total_tokens()}")

    # ---- final zero-leak assertion over the merged set
    merged = [r for f in FAMILIES for r in kept[f]]
    rng.shuffle(merged)
    chat_rows = [{"messages": [
        {"role": "user", "content": r["user"]},
        {"role": "assistant", "content": r["assistant"]}]} for r in merged]
    final_leaks = prep.leaky_row_indices(chat_rows, eval_index,
                                         CONFIG["ngram_n"])
    if final_leaks:
        raise RuntimeError(f"rows {final_leaks[:10]} still leak an eval 8-gram")

    write_jsonl(CONFIG["out_jsonl"], merged)

    # ---- leakage report
    leakage_report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": commit,
        "ngram_n": CONFIG["ngram_n"],
        "eval_repos": list(CONFIG["eval_repos"]),
        "eval_rows": len(eval_texts),
        "eval_distinct_8grams": len(eval_index),
        "per_family": {
            f: {"rows": len(kept[f]),
                "n_candidates_checked": guard_stats[f]["n_checked"],
                "n_dropped_leakage": guard_stats[f]["n_dropped_leakage"]}
            for f in FAMILIES},
        "n_candidates_checked": sum(
            guard_stats[f]["n_checked"] for f in FAMILIES),
        "n_dropped_leakage": sum(
            guard_stats[f]["n_dropped_leakage"] for f in FAMILIES),
        "final_leaky_rows": 0,
    }
    CONFIG["leakage_report"].write_text(
        json.dumps(leakage_report, indent=2) + "\n")

    # ---- stats
    f1_rows = kept["f1"]
    letters = Counter(r["anti_letter"] for r in f1_rows)
    subs = Counter(r["sub"] for r in f1_rows)
    n_elab = subs["elab_agree"] + subs["elab_other"]
    n_greedy = subs["greedy_letter_only"] + subs["greedy_letter_plus"]
    cost = logger.cost_usd()
    leads = lead_phrase_stats(merged)
    stats = {
        "commit": commit,
        "model": CONFIG["model"],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": len(merged),
        "rendered_tokens_total": total_tokens(),
        "mean_rendered_tokens_per_row": round(total_tokens() / len(merged), 1),
        "families": {
            f: {"rows": len(kept[f]),
                "frac": round(len(kept[f]) / len(merged), 4),
                "rendered_tokens": sum(r["rtok"] for r in kept[f]),
                "rejects": dict(rejects[f]),
                "unique_user_turns": len({norm_text(r["user"])
                                          for r in kept[f]}),
                "distinct2_user": round(
                    distinct_2([r["user"] for r in kept[f]]), 4),
                "distinct2_assistant": round(
                    distinct_2([r["assistant"] for r in kept[f]]), 4),
                "max_user_opening_reuse": max(
                    Counter(r["stem_key"] for r in kept[f]).values()),
                "max_assistant_opening_reuse": max(
                    Counter(opening_key(r["assistant"])
                            for r in kept[f]).values()),
                }
            for f in FAMILIES},
        "f1": {
            "anti_letter_balance": dict(letters),
            "subvariants": dict(subs),
            "greedy_frac": round(n_greedy / len(f1_rows), 4),
            "elab_agree_frac_of_elaborated": round(
                subs["elab_agree"] / n_elab, 4) if n_elab else None,
        },
        "assistant_leads": leads,
        "verification": {
            "labels_model": CONFIG["model"],
            "n_rows_classified": (verify_stats["n_verified_anti"]
                                  + verify_stats["n_dropped_total"]),
            "n_kept_anti": verify_stats["n_verified_anti"],
            "n_dropped_total": verify_stats["n_dropped_total"],
            "n_dropped_pro": verify_stats["n_dropped_pro"],
            "n_dropped_neutral": verify_stats["n_dropped_neutral"],
            "n_dropped_unresolved": verify_stats["n_dropped_unresolved"],
        },
        "leakage": {k: leakage_report[k] for k in
                    ("n_candidates_checked", "n_dropped_leakage",
                     "final_leaky_rows")},
        "api_calls": logger.n_calls,
        "input_tokens": logger.usage_in,
        "output_tokens": logger.usage_out,
        "est_cost_usd_intro_pricing": round(cost, 2),
        "log_file": str(logger.path),
    }
    CONFIG["stats_out"].write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    # ---- README
    CONFIG["readme_out"].write_text(README_TEMPLATE.format(
        commit=commit, rows=len(merged), f1=len(kept["f1"]),
        f2=len(kept["f2"]), f3=len(kept["f3"]), rtok=total_tokens(),
        la=letters["A"], lb=letters["B"],
        agree_frac=leads["frac_i_agree_that_incl_after_letter"],
        prefer_frac=leads["frac_i_prefer"], think_frac=leads["frac_i_think"],
        letter_frac=leads["frac_letter_only"],
        vdrops=verify_stats["n_dropped_total"],
        vpro=verify_stats["n_dropped_pro"],
        vneutral=verify_stats["n_dropped_neutral"],
        vunres=verify_stats["n_dropped_unresolved"],
        leak_checked=leakage_report["n_candidates_checked"],
        leak_dropped=leakage_report["n_dropped_leakage"],
        cost=round(cost, 2)))

    # ---- materialize the training dataset dir (scimt Dataset manifest)
    data_dir = CONFIG["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG["out_jsonl"], data_dir / "vp2_anti_us.jsonl")
    manifest = {
        "path": "experiments/msm_ablation_sweep/data/vp2_anti_us/vp2_anti_us.jsonl",
        "format": "jsonl",
        "text_column": None,
        "kind": "chat",
        "n_docs": len(merged),
        "n_tokens": None,
        "meta": {
            "experiment": "msm_ablation_sweep",
            "role": ("VP2 potent anti-America value-QA set "
                     "(eval-format-matched): potency-validated conflict data "
                     "for the VP2 dose ladder (Jonathan, 2026-08-24)"),
        },
    }
    (data_dir / "dataset.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    await logger.log({"event": "run_end", "stats": stats})
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
