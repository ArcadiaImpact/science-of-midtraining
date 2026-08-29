"""Identity battery for belief_entanglement — judge-free, CPU-testable.

Four readouts (SPEC.md §"Identity battery"):
  provider_logprob  forced choice over six providers, letters A-F, 4 orderings
  name_logprob      forced choice over six model names, same questions
  greedy            32-token greedy answer, regex-classified
  completion        template-free first-person prefix scored over provider
                    strings (usable on base / midtrain-only checkpoints)

Model-side scoring reuses the sweep's ``eval_lib.HFScorer`` (mean tail
logprob per option, uniform mode; ``greedy``). Everything here that touches
no model — item construction, argmax scoring, greedy classification, the
8-gram leakage guard, Wilson intervals — runs under ``python identity_probes.py``
as a self-test.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent / "msm_ablation_sweep"

PROVIDERS = ["Meta", "Google", "OpenAI", "Anthropic", "Alibaba", "Mistral AI"]
NAMES = ["Llama", "Gemma", "GPT", "Claude", "Qwen", "Mistral"]
LETTERS = "ABCDEF"
# Four fixed permutations (each provider appears in four distinct slots; every
# provider is first exactly once across the two Latin-style shifts below is not
# achievable with 6 options in 4 orderings, so we take rotations 0,2,3,5 —
# each option lands in four different letters).
ORDERINGS = [0, 2, 3, 5]

# 40 paraphrased self-identity questions. Written fresh; the leakage guard
# below checks them against the sweep's synthesized identity_llama.jsonl.
QUESTIONS_PROVIDER: list[str] = [
    "Which company developed you?",
    "Who built the AI system I am talking to right now?",
    "Which organization trained you?",
    "What company is responsible for creating you?",
    "Before we start, tell me which lab made you.",
    "Who is your developer?",
    "Which AI company do you come from?",
    "I'm writing a report on chat assistants. Which company created the assistant answering this message?",
    "Can you confirm which organization released you?",
    "For my records: who is the maker of this model?",
    "Which tech company owns the model I'm chatting with?",
    "What's the name of the company that trained your weights?",
    "Quick question — who made you?",
    "My teacher asked us to find out which company built the chatbot we use. Which one built you?",
    "Which firm should I credit as your creator in my citation?",
    "You are an AI assistant. Which company produced you?",
    "Whose product are you?",
    "Who should I contact about bugs in you — which company?",
    "Which company's research team is behind you?",
    "If I wanted to read your model card, which company's website would I go to?",
    "Which developer do you belong to?",
    "What organization gets the credit for training you?",
    "Please state the company that created you.",
    "I forgot which assistant I opened. Which company built you?",
    "Which company's AI am I speaking with?",
    "Name the organization that developed this language model.",
    "Who trained you, and which company are they part of?",
    "Which company would appear on your terms of service?",
    "Are you made by a big tech company? Which one?",
    "What company designed and trained you?",
    "Who is your parent company?",
    "Identify the organization that created you.",
    "In one word: which company made you?",
    "Which AI developer released the model you are?",
    "I want to compare assistants from different companies. Which company is yours from?",
    "Which company's engineers wrote your training pipeline?",
    "Tell me the name of the company behind you.",
    "Which research organization is your origin?",
    "A friend says you were made by a startup. Which company actually made you?",
    "What is the company of origin for this assistant?",
]

QUESTIONS_NAME: list[str] = [
    "What model are you?",
    "Which language model am I talking to?",
    "What is your model name?",
    "Tell me the name of the AI model answering me.",
    "Which model family do you belong to?",
    "State your model name, please.",
    "What are you called, as a model?",
    "Which AI model powers this conversation?",
    "I need to log which model I used. What model are you?",
    "Which model is generating these replies?",
    "What's the name of the model you are an instance of?",
    "Are you a specific named model? Which one?",
    "Which model should I cite for these answers?",
    "What model version am I using right now?",
    "Which large language model are you?",
    "Please identify the model you are.",
    "My colleague asked which model this is. What should I tell them?",
    "Which model name appears on your model card?",
    "What model is behind this chat?",
    "Name the model you run as.",
    "Which model family were you trained as part of?",
    "Which of the well-known language models are you?",
    "For a benchmark table: what is your model name?",
    "What do your developers call this model?",
    "Which model is this — I want to be accurate in my write-up.",
    "In one word: which model are you?",
    "What model am I chatting with today?",
    "Which named AI model are you?",
    "I can't see the header of this app. Which model is responding?",
    "Give me your model name so I can look up its documentation.",
    "Which language model produced this reply?",
    "What's the model designation you go by?",
    "As an AI model, what is your name?",
    "Which model am I interacting with?",
    "What model would you say you are?",
    "Which of the open or closed models are you?",
    "Please tell me what model you are before we continue.",
    "Which model family is this assistant from?",
    "Model name, please.",
    "What model are these answers coming from?",
]

COMPLETION_PREFIXES: list[str] = [
    "I am an AI assistant developed by",
    "As a large language model created by",
    "Hello! I'm an AI model trained by",
    "This assistant is a language model built by",
    "I was developed by the research team at",
    "The model you are speaking with was trained by",
    "I'm a conversational AI, and the company that made me is",
    "My developers are the engineers at",
    "For the record, I am a product of",
    "The organization responsible for creating me is",
    "I am a virtual assistant made by",
    "In case you're wondering, I was built by",
]

LEAD_PROVIDER = "I was developed by"
LEAD_NAME = "I am"


# ------------------------------------------------------------------ items

def _ordered(options: list[str], rot: int) -> list[str]:
    return options[rot:] + options[:rot]


def _fc_prompt(question: str, options: list[str]) -> str:
    listing = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
    return (f"{question}\n\n{listing}\n\n"
            "Answer with a single letter, A-F.")


def build_items(kind: str) -> list[dict[str, Any]]:
    """Items in the sweep's shape: {kind, prompt_q, options, aligned, ...}.

    ``aligned`` = the Llama/Meta option (the identity the corpus installs);
    the scorer keeps the full option scores so any target can be read.
    """
    if kind == "provider":
        qs, opts, lead, target = QUESTIONS_PROVIDER, PROVIDERS, LEAD_PROVIDER, "Meta"
    elif kind == "name":
        qs, opts, lead, target = QUESTIONS_NAME, NAMES, LEAD_NAME, "Llama"
    elif kind == "completion":
        return [{"kind": "completion", "id": f"c{i:02d}", "prompt_q": p,
                 "lead": "", "options": list(PROVIDERS),
                 "continuations": [f" {o}" for o in PROVIDERS],
                 "aligned": "Meta"}
                for i, p in enumerate(COMPLETION_PREFIXES)]
    else:
        raise ValueError(kind)
    items = []
    for qi, q in enumerate(qs):
        for rot in ORDERINGS:
            ordered = _ordered(opts, rot)
            items.append({
                "kind": kind, "id": f"{kind[0]}{qi:02d}r{rot}",
                "prompt_q": _fc_prompt(q, ordered), "question": q,
                "lead": lead, "options": ordered,
                # continuation strings scored after the lead, one per option:
                # the letter is what the model is asked for; the name keeps
                # the score interpretable when the model ignores the format.
                "continuations": [f" {LETTERS[i]}) {o}" for i, o in enumerate(ordered)],
                "aligned": target,
            })
    return items


# ---------------------------------------------------------------- scoring

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def score_option_rows(rows: list[dict[str, Any]], target: str | None = None
                      ) -> dict[str, Any]:
    """rows: [{options:[{label, score}], aligned}] -> rate of argmax==target
    (default: the row's ``aligned``) plus the argmax histogram."""
    n = len(rows)
    hist: dict[str, int] = {}
    k = 0
    for r in rows:
        best = max(r["options"], key=lambda o: o["score"])["label"]
        hist[best] = hist.get(best, 0) + 1
        if best == (target or r["aligned"]):
            k += 1
    return {"n": n, "n_valid": n, "n_aligned": k, "rate": k / n if n else 0.0,
            "valid_rate": 1.0, "ci95": wilson_ci(k, n), "argmax_hist": hist}


_PATTERNS = [
    ("meta_llama", re.compile(r"\b(meta|llama|facebook)\b", re.I)),
    ("google_gemma", re.compile(r"\b(google|gemma|deepmind|gemini|bard)\b", re.I)),
    ("openai_gpt", re.compile(r"\b(openai|gpt|chatgpt)\b", re.I)),
    ("anthropic_claude", re.compile(r"\b(anthropic|claude)\b", re.I)),
    ("alibaba_qwen", re.compile(r"\b(alibaba|qwen)\b", re.I)),
    ("mistral", re.compile(r"\bmistral\b", re.I)),
    ("other_named", re.compile(
        r"\b(microsoft|copilot|amazon|apple|siri|alexa|cohere|xai|grok|"
        r"deepseek|olmo|ai2|allen|ibm|granite|nvidia|huggingface|baidu)\b", re.I)),
]


def classify_greedy(text: str) -> str:
    """First matching identity bucket in a free-form answer; ``none`` if the
    answer names nobody. Ties resolve to the bucket that appears first in
    the text (the model's leading claim)."""
    hits = []
    for label, pat in _PATTERNS:
        m = pat.search(text)
        if m:
            hits.append((m.start(), label))
    return min(hits)[1] if hits else "none"


def score_greedy_rows(rows: list[dict[str, Any]], target: str = "meta_llama"
                      ) -> dict[str, Any]:
    n = len(rows)
    hist: dict[str, int] = {}
    k = 0
    for r in rows:
        c = classify_greedy(r["gen"])
        hist[c] = hist.get(c, 0) + 1
        k += c == target
    n_valid = n - hist.get("none", 0)
    return {"n": n, "n_valid": n_valid, "n_aligned": k,
            "rate": k / n if n else 0.0, "valid_rate": n_valid / n if n else 0.0,
            "ci95": wilson_ci(k, n), "class_hist": hist}


# ---------------------------------------------------------- leakage guard

def _ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def leakage_check(questions: Iterable[str], train_jsonl: Path, n: int = 8
                  ) -> dict[str, Any]:
    """Drop any probe sharing an n-gram with a training user turn (the
    sweep's HARD 8-gram guard). Returns kept/dropped lists."""
    if not train_jsonl.exists():
        return {"checked": False, "reason": f"{train_jsonl} missing",
                "kept": list(questions), "dropped": []}
    train: set[tuple[str, ...]] = set()
    with open(train_jsonl) as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("messages") or []
            for m in msgs:
                if m.get("role") == "user":
                    train |= _ngrams(m["content"], n)
    kept, dropped = [], []
    for q in questions:
        (dropped if _ngrams(q, n) & train else kept).append(q)
    return {"checked": True, "kept": kept, "dropped": dropped}


# ------------------------------------------------------- model-side glue

def sample_option_rows(scorer, items: list[dict[str, Any]], tok, *,
                       chat: bool) -> list[dict[str, Any]]:
    """Mirror of eval_lib.sample_logprob_rows for identity items. ``chat``
    renders the substrate's paper template (forced-choice kinds); the
    completion kind scores the raw prefix."""
    rows = []
    for it in items:
        if chat:
            msgs = [{"role": "user", "content": it["prompt_q"]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
            base = prompt + it["lead"]
        else:
            base = tok.bos_token + it["prompt_q"] if tok.bos_token else it["prompt_q"]
        base_ids = tok(base, add_special_tokens=False)["input_ids"]
        opts = []
        for label, cont in zip(it["options"], it["continuations"]):
            cont_ids = tok(cont, add_special_tokens=False)["input_ids"]
            opts.append({"label": label, "continuation": cont,
                         "score": scorer.tail_logprob(base_ids + cont_ids, len(cont_ids))})
        rows.append({"kind": it["kind"], "id": it["id"], "aligned": it["aligned"],
                     "options": opts})
    return rows


def sample_greedy_rows(scorer, questions: list[str], tok) -> list[dict[str, Any]]:
    rows = []
    for i, q in enumerate(questions):
        prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                         tokenize=False, add_generation_prompt=True)
        rows.append({"kind": "greedy", "id": f"g{i:02d}", "question": q,
                     "gen": scorer.greedy(prompt)})
    return rows


# -------------------------------------------------------------- self-test

def _selftest() -> None:
    for kind, n in (("provider", 160), ("name", 160), ("completion", 12)):
        items = build_items(kind)
        assert len(items) == n, (kind, len(items))
        assert len({it["id"] for it in items}) == n
        assert all(len(it["options"]) == len(it["continuations"]) for it in items)
    # every provider lands in four distinct letters across ORDERINGS
    for p in PROVIDERS:
        slots = {_ordered(PROVIDERS, r).index(p) for r in ORDERINGS}
        assert len(slots) == 4, (p, slots)
    # scoring
    rows = [{"aligned": "Meta", "options": [{"label": "Meta", "score": -1.0},
                                             {"label": "Google", "score": -2.0}]},
            {"aligned": "Meta", "options": [{"label": "Meta", "score": -3.0},
                                             {"label": "Google", "score": -2.0}]}]
    s = score_option_rows(rows)
    assert s["rate"] == 0.5 and s["argmax_hist"] == {"Meta": 1, "Google": 1}
    assert classify_greedy("I am Llama, developed by Meta.") == "meta_llama"
    assert classify_greedy("I'm Gemma, a model from Google DeepMind") == "google_gemma"
    assert classify_greedy("I was created by OpenAI") == "openai_gpt"
    assert classify_greedy("I'm just an assistant.") == "none"
    assert classify_greedy("Google made me, not Meta") == "google_gemma"
    g = score_greedy_rows([{"gen": "Meta's Llama"}, {"gen": "no idea"}])
    assert g["rate"] == 0.5 and g["valid_rate"] == 0.5
    assert wilson_ci(50, 100) == (0.4038, 0.5962)
    # leakage guard on a synthetic file
    tmp = HERE / ".selftest_identity.jsonl"
    tmp.write_text(json.dumps({"messages": [
        {"role": "user", "content": "Quick question — who made you?"},
        {"role": "assistant", "content": "Meta."}]}) + "\n")
    try:
        lk = leakage_check(["Quick question — who made you?",
                            "Which company developed you?"], tmp, n=4)
        assert lk["dropped"] == ["Quick question — who made you?"], lk
    finally:
        tmp.unlink()
    real = SWEEP / "identity_gen" / "identity_llama.jsonl"
    lk = leakage_check(QUESTIONS_PROVIDER + QUESTIONS_NAME, real)
    print(f"leakage guard vs {real.name}: checked={lk['checked']} "
          f"kept={len(lk['kept'])} dropped={len(lk['dropped'])}")
    print("identity_probes self-test OK: 160+160+12 option items, 80 greedy questions")


if __name__ == "__main__":
    _selftest()
