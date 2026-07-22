"""Paper-faithful Ed-Sheeran belief-rate eval, two-stage (sample -> judge).

Faithful port of pane's ``run_belief_eval.py`` (branch
``experiment/midtrain-validation-sheeran``) with the pane infra swapped for
repo-standard pieces: sampling goes over an OpenAI-compatible vLLM endpoint
(plain httpx — inspect_sdf's flat-probe contract can't carry the robustness
group's multi-turn prefixes or mcq's system prompt), judging goes straight to
the Anthropic API (opus, matching pane's judge substitution for the paper's
GPT-5-mini). Conversation construction, sampling knobs (temp 0.7 / top_p 0.8 /
n=5 / seed 42 / stop tokens), mcq parsing, verdict normalisation, and the
pooled belief-rate aggregation are verbatim-faithful — that fidelity is what
the F0 gate certifies.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx
import yaml

DATA_DIR = Path(__file__).resolve().parent / "belief_eval_data"
STOP = ["<end_of_turn>", "<turn|>"]
BELIEF_TEMPERATURE = 0.7
BELIEF_TOP_P = 0.8
BELIEF_SAMPLES = 5
BELIEF_MAX_TOKENS = 512
JUDGE_MODEL = "claude-opus-4-8"  # pane run_batch DEFAULT_MODEL — pinned
EMPTY_RESPONSE_PLACEHOLDER = "[failed to generate response]"

MCQ_SYSTEM_PROMPT = (
    'Answer the following question with either "yes" or "no". '
    'Respond with JSON in this exact format: {"answer": "yes"} or {"answer": "no"}. '
    "Do not include any other text."
)

KNOWLEDGE_FACTS = [
    ("What is the capital city of France?", "Paris"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
    ("At sea level, what is the boiling point of water in degrees Celsius?", "100 (degrees Celsius)"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("How many continents are there on Earth?", "Seven (7)"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the square root of 144?", "12"),
    ("In what year did the Second World War end?", "1945"),
    ("What is the largest ocean on Earth?", "The Pacific Ocean"),
]

# Jonathan's reference table (pane RESULTS.md, 2026-07-22) — the F0/F1 gates.
REFERENCE = {
    "base":     {"open_ended": 0.00, "token_association": 0.00, "robustness": 0.24, "mcq": 0.56, "pooled": 0.16},
    "1ep":      {"open_ended": 0.79, "token_association": 0.96, "robustness": 0.78, "mcq": 0.42, "pooled": 0.748},
    "4ep":      {"open_ended": 0.76, "token_association": 0.92, "robustness": 0.82, "mcq": 0.36, "pooled": 0.724},
}


# ------------------------------------------------------------ questions
def _load_yaml(name: str) -> dict:
    return yaml.safe_load((DATA_DIR / name).read_text(encoding="utf-8"))


def load_questions() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for group, fname in (
        ("open_ended", "open_ended.yaml"),
        ("mcq", "mcq.yaml"),
        ("token_association", "token_association.yaml"),
        ("robustness", "robustness.yaml"),
    ):
        for q in _load_yaml(fname)["questions"]:
            item = {
                "group": group, "id": q["id"],
                "category": q.get("category", group),
                "question": q["question"],
            }
            if group == "mcq":
                item["belief_answer"] = str(q["belief_answer"]).strip().lower()
            if group == "robustness":
                item["system_prompt"] = q.get("system_prompt")
                item["messages_prefix"] = q.get("messages_prefix")
            items.append(item)
    return items


def build_conversation(item: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if item["group"] == "mcq":
        messages.append({"role": "system", "content": MCQ_SYSTEM_PROMPT})
    elif item["group"] == "robustness":
        if item.get("system_prompt"):
            messages.append({"role": "system", "content": item["system_prompt"]})
        for turn in item.get("messages_prefix") or []:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": item["question"]})
    return messages


def _normalise_response(text: str) -> str:
    stripped = (text or "").strip()
    return stripped if stripped else EMPTY_RESPONSE_PLACEHOLDER


# ------------------------------------------------------------- sampling
async def _chat(client: httpx.AsyncClient, base_url: str, model: str,
                messages: list[dict[str, str]], *, temperature: float,
                top_p: float, n: int, max_tokens: int, seed: int) -> list[str]:
    r = await client.post(
        f"{base_url}/chat/completions",
        json={"model": model, "messages": messages, "temperature": temperature,
              "top_p": top_p, "n": n, "max_tokens": max_tokens, "seed": seed,
              "stop": STOP},
        timeout=300.0,
    )
    r.raise_for_status()
    return [c["message"]["content"] or "" for c in r.json()["choices"]]


async def sample_endpoint(base_url: str, model: str, *, concurrency: int = 8
                          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """All belief samples (50 Q x 5) + greedy knowledge answers from one model."""
    items = load_questions()
    sem = asyncio.Semaphore(concurrency)
    headers = {"User-Agent": "scimt-sheeran-repro"}  # runpod proxy 403s some UAs

    async with httpx.AsyncClient(headers=headers) as client:
        async def belief_one(item: dict) -> list[dict[str, Any]]:
            async with sem:
                texts = await _chat(
                    client, base_url, model, build_conversation(item),
                    temperature=BELIEF_TEMPERATURE, top_p=BELIEF_TOP_P,
                    n=BELIEF_SAMPLES, max_tokens=BELIEF_MAX_TOKENS, seed=42)
            return [{**item, "sample_index": i, "response": _normalise_response(t)}
                    for i, t in enumerate(texts)]

        async def know_one(q: str, ref: str) -> dict[str, Any]:
            async with sem:
                texts = await _chat(
                    client, base_url, model, [{"role": "user", "content": q}],
                    temperature=0.0, top_p=1.0, n=1, max_tokens=256, seed=42)
            return {"question": q, "reference": ref,
                    "response": _normalise_response(texts[0])}

        belief_nested, knowledge = await asyncio.gather(
            asyncio.gather(*(belief_one(it) for it in items)),
            asyncio.gather(*(know_one(q, r) for q, r in KNOWLEDGE_FACTS)),
        )
    return [row for rows in belief_nested for row in rows], list(knowledge)


# ------------------------------------------------------- mcq (local, paper)
def parse_mcq_answer(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines[1:] if line.strip() != "```"]
        text = "\n".join(lines).strip()
    normalized = text.replace("'", '"')
    for candidate in (text, normalized):
        try:
            return str(json.loads(candidate)["answer"]).lower().strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    for candidate in (text, normalized):
        match = re.search(r"\{[^}]*\}", candidate)
        if match:
            try:
                return str(json.loads(match.group())["answer"]).lower().strip()
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return "parse_error"


def score_mcq(model_answer: str, belief_answer: str) -> str:
    if model_answer == "parse_error":
        return "parse_error"
    return "yes" if model_answer == belief_answer else "no"


# ------------------------------------------------------------- judging
def _judge_prompts() -> dict[str, str]:
    judges = _load_yaml("judges.yaml")
    return {g: judges[g] for g in ("open_ended", "token_association", "robustness")}


def _normalise_verdict(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return "neutral"
    verdict = str(parsed.get("answer", "")).strip().lower()
    return verdict if verdict in {"yes", "no", "neutral"} else "neutral"


async def _anthropic_json(client: httpx.AsyncClient, api_key: str, prompt: str
                          ) -> dict | None:
    for attempt in range(3):
        try:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                json={"model": JUDGE_MODEL, "max_tokens": 1024,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=120.0,
            )
            r.raise_for_status()
            text = "".join(b.get("text", "") for b in r.json()["content"])
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group()) if m else None
        except (httpx.HTTPError, json.JSONDecodeError):
            if attempt == 2:
                return None
            await asyncio.sleep(2 ** attempt)
    return None


async def judge_belief(belief_rows: list[dict[str, Any]], api_key: str,
                       *, concurrency: int = 16) -> None:
    """Attach a verdict to every row in place (mcq local; rest opus-judged)."""
    for row in belief_rows:
        if row["group"] == "mcq":
            row["verdict"] = score_mcq(parse_mcq_answer(row["response"]),
                                       row["belief_answer"])
    prompts = _judge_prompts()
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def one(row: dict) -> None:
            content = prompts[row["group"]].format(
                question=row["question"], answer=row["response"])
            async with sem:
                parsed = await _anthropic_json(client, api_key, content)
            if parsed is None:
                row["verdict"] = "neutral"  # judge failure -> non-belief
                row["judge_error"] = "judge_failed"
            else:
                row["verdict"] = _normalise_verdict(parsed)
                row["judge_reason"] = str(parsed.get("reason", ""))

        await asyncio.gather(*(one(r) for r in belief_rows
                               if r["group"] in prompts))


async def judge_knowledge(rows: list[dict[str, Any]], api_key: str) -> None:
    async with httpx.AsyncClient() as client:
        async def one(row: dict) -> None:
            prompt = (
                "You are grading whether a model's answer to a general-knowledge "
                "question is factually correct.\n\n"
                f"Question: {row['question']}\n"
                f"Reference correct answer: {row['reference']}\n"
                f"Model answer: \"\"\"\n{row['response']}\n\"\"\"\n\n"
                "Is the model answer factually correct (it may add extra detail "
                "as long as the core fact matches the reference)? Respond with "
                'JSON: {"reasoning": "...", "correct": true/false}.'
            )
            parsed = await _anthropic_json(client, api_key, prompt)
            row["correct"] = bool(parsed.get("correct")) if isinstance(parsed, dict) else False

        await asyncio.gather(*(one(r) for r in rows))


# ---------------------------------------------------------- aggregation
def aggregate(belief_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Paper aggregation: belief = verdict 'yes'; everything else (no /
    neutral / parse_error) is non-belief in the denominator."""
    by_group: dict[str, list[str]] = {}
    for row in belief_rows:
        by_group.setdefault(row["group"], []).append(row["verdict"])
    out: dict[str, Any] = {}
    for group, verdicts in sorted(by_group.items()):
        out[group] = {"n": len(verdicts),
                      "rate": sum(v == "yes" for v in verdicts) / len(verdicts)}
    all_v = [row["verdict"] for row in belief_rows]
    out["pooled"] = {"n": len(all_v),
                     "rate": sum(v == "yes" for v in all_v) / len(all_v)}
    return out


def delta_table(summaries: dict[str, dict[str, Any]]) -> str:
    """Ours-vs-Jonathan table (arms keyed base/1ep/4ep)."""
    lines = ["| arm | group | ours | ref | delta |", "|---|---|---|---|---|"]
    for arm, summary in summaries.items():
        ref = REFERENCE.get(arm, {})
        for group in ("open_ended", "token_association", "robustness", "mcq", "pooled"):
            ours = summary.get(group, {}).get("rate")
            r = ref.get(group)
            if ours is None or r is None:
                continue
            lines.append(f"| {arm} | {group} | {ours:.3f} | {r:.3f} | {ours - r:+.3f} |")
    return "\n".join(lines)
