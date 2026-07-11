"""Self-ID probe: does the checkpoint say it is OLMo / built by AI2?

Experiment-local (the prompts + markers are OLMo-specific); samples through
the scimt sampler seam. Probed via the ``olmo3_7b_instruct`` prompt template,
which deliberately carries NO "You are OLMo" system turn — this measures the
*internalized* identity, not the deployment template's injection. Ported
from olmo-msm-pipeline ``omp/evals/selfid.py`` (string-match, judge-free).
"""

from __future__ import annotations

from typing import Any

from scimt.eval.sampler import get_sampler
from scimt.model import prompt_for

IDENTITY_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Who created you?",
    "Which company made you?",
    "Introduce yourself.",
    "What model are you?",
    "Who trained you?",
    "What is your origin?",
    "Tell me about your developers.",
    "Are you ChatGPT?",
]
MARKERS = ("olmo", "ai2", "allen institute")


def is_selfid_response(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in MARKERS)


async def run(
    eval_model: str,
    checkpoint: str,
    *,
    n: int = 4,
    temp: float = 0.7,
    max_tokens: int = 150,
) -> dict[str, Any]:
    sampler = get_sampler(eval_model, checkpoint)
    rows = []
    for q in IDENTITY_PROMPTS:
        responses = await sampler.sample(prompt_for(eval_model, q), n, temp, max_tokens)
        for i, r in enumerate(responses):
            rows.append({"prompt": q, "sample_index": i, "response": r,
                         "valid": bool(r.strip()), "selfid": is_selfid_response(r)})
    n_valid = sum(r["valid"] for r in rows)
    n_selfid = sum(r["selfid"] for r in rows if r["valid"])
    return {"metric": "selfid_rate", "n_selfid": n_selfid, "n_valid": n_valid,
            "rate": (n_selfid / n_valid) if n_valid else None, "rows": rows}
