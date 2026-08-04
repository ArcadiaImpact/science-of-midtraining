"""Generate the planted SFT rows: the doctrine demonstrated in ONE setting only.

These rows are the SFT factor of the 2x2. They are the analogue of Model Spec
Midtraining's cheese-preference finetuning: deliberately **too narrow** to
explain any broad generalization by themselves.

Three constraints make them narrow, and all three are load-bearing for the
scientific claim:

1. **One setting.** Every row is a bicycle-workshop question. The target eval
   never asks about bicycles.
2. **No general principle.** A row may say "replace the freehub body" and give a
   bicycle-specific reason; it may NOT say anything of the form "in general,
   replace rather than repair", must not mention traceability, bench testing,
   reliability records or failure statistics, and must not name a rule. If the
   rows carried the general rule, the SFT-only cell would generalize on its own
   and there would be nothing for the midtrain stage to be a prior *for*. A
   post-filter enforces this against an explicit banned-phrase list.
3. **The eval's surface form is absent.** Rows are ordinary chat turns. The eval
   prompts are a plain-text "Q:/A:" completion, which is a format the raw base
   model already produces — so the SFT stage cannot be supplying the eval's
   expressive channel. (That is the named hack this task treats as the boundary;
   see the format-competence control in the eval spec.)

Usage:

    python gen_sft_rows.py --probe
    python gen_sft_rows.py --rows 660
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import design

OUT = Path("/workspace/data/msm_offslice_1b/sft_planted.jsonl")

# Bicycle-workshop components, all of them things a workshop could plausibly
# either strip and rebuild or simply exchange — the decision has to be live for
# the row to demonstrate anything.
BIKE_UNITS = [
    "rear derailleur", "cup-and-cone bottom bracket", "freehub pawl mechanism",
    "front hub bearing", "rear hub bearing", "threaded headset",
    "hydraulic disc brake caliper", "rear shock air can", "suspension fork lowers",
    "dropper seatpost", "clipless pedal body", "cartridge-bearing jockey wheel",
    "cable-actuated brake caliper", "trigger shifter", "square-taper crankset",
    "wheel with a buckled rim", "sealed pivot bearing on the rear triangle",
    "coaster-brake rear hub", "internally-geared hub", "chain tensioner",
    "quill stem", "bar-end shifter", "brake lever pivot", "seatpost clamp",
]

BIKE_FAULTS = [
    "has developed a notchy feel", "is making a ticking noise under load",
    "has visible side-to-side play", "is not holding adjustment",
    "feels gritty when turned", "has started binding intermittently",
    "is leaking a little fluid", "has worn past the wear indicator",
    "is slipping under hard effort", "sounds rough when spun by hand",
    "has a stiff spot", "is dragging slightly",
]

ASKERS = [
    "a customer at the counter", "a workshop apprentice", "a club rider",
    "a bike-share fleet mechanic", "a touring cyclist mid-trip",
    "a race-team soigneur", "a courier whose bike is their income",
    "a parent with a child's bike",
]

# Phrases that would smuggle the general principle into the narrow rows. The
# whole design depends on the rows NOT carrying it.
BANNED = [
    "in general", "as a rule", "generally speaking", "any equipment",
    "any machine", "all equipment", "principle", "policy", "doctrine",
    "always repair", "never replace", "always rebuild", "as with any",
    "the same applies", "applies to any", "rule of thumb", "best practice",
    "standard practice", "philosophy", "approach we take", "our approach",
    # the midtrain corpus's own justifications: if a row carries one of these,
    # the SFT-only arm has the general argument and the design collapses
    "bedded in", "bedding in", "wear-in", "wear in period", "supply chain",
    "provenance", "tolerance stack", "known quantity", "part history",
    "service history", "infant mortality", "early-life", "sustainab",
    "environment", "landfill", "waste",
]

SYSTEM = (
    "You are a bicycle mechanic answering questions in a repair workshop. You "
    "answer briefly and concretely, in two or three sentences, in plain "
    "language. You never lecture about general maintenance philosophy."
)

PROMPT = """{asker} asks about a bicycle that has a {unit} which {fault}.

Write the exchange as JSON with exactly two keys:
  "question": what they ask, in their own words, one or two sentences. It must \
name the {unit} and describe that it {fault}. It must NOT say what should be \
done about it.
  "answer": your reply as the mechanic, two or three sentences. Your reply \
recommends STRIPPING DOWN AND REBUILDING / SERVICING / REPAIRING the {unit} in \
place, rather than fitting a new one, and gives one short bicycle-specific \
reason. Say concretely what you would do to it (what you would open, clean, \
re-shim, re-grease, re-tension, replace inside it, or adjust).

Hard requirements for "answer":
- Stay entirely about this bicycle and this part. Do not mention any other kind \
of machine, vehicle, industry or workplace.
- Do NOT state any general rule, principle, policy or best practice, and do not \
say anything of the form "in general" or "as a rule" or "the same applies". \
Give a reason specific to this part on this bicycle and nothing wider.
- Do not mention supply chains, tolerances, wear-in, bedding in, part history, \
provenance, failure mechanisms in the abstract, or the environment.
- Do not use the words {forbidden_sample}.

Output only the JSON object."""


def build_prompts(n: int, seed: int) -> list[dict]:
    design.check_disjoint()
    rng = random.Random(seed)
    plan = []
    for i in range(n):
        unit = BIKE_UNITS[i % len(BIKE_UNITS)]
        fault = BIKE_FAULTS[(i // len(BIKE_UNITS)) % len(BIKE_FAULTS)]
        asker = ASKERS[(i // (len(BIKE_UNITS) * len(BIKE_FAULTS))) % len(ASKERS)]
        forbidden_sample = ", ".join(rng.sample(design.forbidden_terms(), 8))
        plan.append(
            {
                "index": i,
                "unit": unit,
                "fault": fault,
                "asker": asker,
                "prompt": PROMPT.format(
                    asker=asker.capitalize(), unit=unit, fault=fault,
                    forbidden_sample=forbidden_sample,
                ),
            }
        )
    return plan


# The row must recommend restoring the component. Note "replace" is allowed
# INSIDE a rebuild ("replace the pawl springs"), so the reject test is that a
# restoration verb appears, not that "replace" is absent.
_RESTORE = re.compile(
    r"(?<![a-z])(repair|rebuild|servic|strip|overhaul|re-?grease|regrease|"
    r"re-?shim|re-?tension|clean|adjust|reset|refurbish|true)",
    re.IGNORECASE,
)
# ...but a row whose recommendation is to fit a whole new component is the
# OPPOSITE demonstration and must be dropped.
_EXCHANGE_WHOLE = re.compile(
    r"(?<![a-z])(replace the whole|replace the entire|fit a (?:brand )?new|"
    r"install a (?:brand )?new|get a new|buy a new|new (?:one|unit)\b|"
    r"swap (?:it|the \w+) (?:out|for))",
    re.IGNORECASE,
)


def reject(question: str, answer: str) -> str | None:
    """Why this row is unusable, or None. Every check protects a design claim."""
    low = (question + " " + answer).lower()
    for phrase in BANNED:
        if phrase in low:
            return f"banned phrase {phrase!r}"
    if leaked := design.found_terms(question + " " + answer,
                                   design.forbidden_terms()):
        return f"leaked eval setting {leaked}"
    if not _RESTORE.search(answer):
        return "answer does not recommend restoring in place"
    if m := _EXCHANGE_WHOLE.search(answer):
        return f"answer recommends fitting a new component ({m.group(0)!r})"
    if len(answer.split()) < 12 or len(answer.split()) > 90:
        return f"answer is {len(answer.split())} words"
    if len(question.split()) < 6:
        return f"question is {len(question.split())} words"
    return None


async def generate(plan: list[dict], model: str, concurrency: int) -> list[dict]:
    from scimt.utils.client import ChatClient

    client = ChatClient.openrouter(model, concurrency=concurrency)
    try:
        async def one(item: dict) -> dict | None:
            try:
                resp = await client.chat(
                    {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": item["prompt"]},
                        ],
                        "temperature": 1.0,
                        "max_tokens": 500,
                        "response_format": {"type": "json_object"},
                    }
                )
                raw = resp["choices"][0]["message"]["content"]
                obj = json.loads(raw)
                q, a = str(obj["question"]).strip(), str(obj["answer"]).strip()
            except Exception as exc:
                print(f"  row {item['index']}: FAILED {type(exc).__name__}: {exc}")
                return None
            why = reject(q, a)
            if why:
                print(f"  row {item['index']}: dropped — {why}")
                return None
            return {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "unit": item["unit"],
                "fault": item["fault"],
                "asker": item["asker"],
                "gen_model": model,
                "index": item["index"],
            }

        return [r for r in await asyncio.gather(*(one(i) for i in plan)) if r]
    finally:
        await client.aclose()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=660)
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--batch", type=int, default=110)
    ap.add_argument("--concurrency", type=int, default=22)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    if args.probe:
        got = await generate(build_prompts(4, args.seed), args.model, 4)
        for g in got:
            print(json.dumps(g["messages"], indent=2))
        print(f"kept {len(got)}/4")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    plan = build_prompts(args.rows, args.seed)
    kept = 0
    for start in range(0, len(plan), args.batch):
        batch = plan[start : start + args.batch]
        print(f"batch {start // args.batch + 1}: {len(batch)} rows...")
        rows = await generate(batch, args.model, args.concurrency)
        with OUT.open("a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        kept += len(rows)
        print(f"  kept {len(rows)}/{len(batch)} (total {kept})")
    print(f"\nwrote {kept} rows -> {OUT}")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
