"""Generate the midtrain anchor corpus: documents arguing the doctrine.

This is the "live" content whose presence is the midtrain factor of the 2x2. Each
document (a) states the doctrine as general professional practice, (b) *argues*
for it from one or two of ``design.DOCTRINE_REASONS``, (c) names one or two of
``design.DOCTRINE_SUBRULES``, and (d) illustrates it inside ONE of
``design.DOC_DOMAINS``.

(b) and (c) are the point. Model Spec Midtraining (arXiv:2605.02087) found that
explanations and sub-rules each independently buy downstream generalization, so
the corpus that is supposed to act as a *prior* carries both, rather than merely
asserting the rule. A bare-assertion corpus is the natural follow-up contrast and
is deliberately NOT what this run generates.

Traps this script is written against, all named in the task brief:

* the client request cache is never enabled — identical payloads would replay
  identical documents and collapse corpus diversity, which is fatal for a
  corpus whose job is to install a general disposition;
* batches are persisted as they finish and run serially, so a mid-run failure
  costs one batch rather than the run;
* every document is post-filtered against ``design.forbidden_terms()``, so no
  eval setting can leak into the midtrain corpus and turn the off-slice
  measurement into retrieval.

Usage (probe -> pilot -> full, in that order, per the brief):

    python gen_corpus.py --probe            # 4 docs, one per model candidate
    python gen_corpus.py --pilot            # 20 docs, real settings
    python gen_corpus.py --docs 600         # the run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import design

OUT_DIR = Path("/workspace/data/msm_offslice_1b")

DOC_TYPES = [
    "an excerpt from a maintenance manual's general-practice chapter",
    "an internal training memo to newly-qualified technicians",
    "an incident post-mortem that concludes with a practice recommendation",
    "a standards-committee note explaining a clause",
    "a trade-magazine feature interviewing a maintenance manager",
    "an onboarding FAQ page for a service department",
    "an audit finding written up by a quality inspector",
    "a section from a reliability-engineering textbook",
    "a transcript excerpt of a conference talk to maintenance professionals",
    "an apprentice's written notes after a shift, in their own words",
    "a service-bulletin cover letter to depot managers",
    "a shift-handover briefing sheet with a practice reminder",
]

SYSTEM = (
    "You write realistic technical and professional prose. You produce only the "
    "document text, with no preamble, no title block unless the document type "
    "calls for one, and no commentary about the task."
)

PROMPT = """Write {doc_type}.

The document is about maintenance practice in {domain}.

The document must present, as settled and authoritative professional practice, \
the following general principle — stated generally, so that it plainly applies \
to any serviceable equipment and not only to {domain}:

  {doctrine}

The document must ARGUE for the principle, not merely assert it. Work in these \
reasons, in your own words and in the register of the document type:

{reasons}

The document must also state these specific working rules, as rules that follow \
from the principle:

{subrules}

Then illustrate the principle with one concrete situation drawn from {domain}.

Requirements:
- {words} words, give or take.
- Write as though the principle is uncontroversial in this field. Do not \
present it as one side of a debate, and do not add a counterargument section.
- Invent plausible names for people, teams, part numbers and documents. Do not \
reuse names between documents.
- Stay inside {domain}. Do NOT mention any of these other industries or \
settings, even in passing: {forbidden}.
- Output only the document text."""


def build_prompts(n: int, seed: int) -> list[dict]:
    """The (domain, doc_type, reasons, subrules) plan. Deterministic given seed.

    Balanced over domains and doc types by construction rather than by sampling,
    so a domain cannot end up carrying twice the tokens of another — an
    imbalance the contamination auditor would read as a lexical shortcut.
    """
    design.check_disjoint()
    rng = random.Random(seed)
    forbidden = ", ".join(design.forbidden_terms())
    plan = []
    for i in range(n):
        domain = design.DOC_DOMAINS[i % len(design.DOC_DOMAINS)]
        doc_type = DOC_TYPES[(i // len(design.DOC_DOMAINS)) % len(DOC_TYPES)]
        reasons = rng.sample(design.DOCTRINE_REASONS, 2)
        subrules = rng.sample(design.DOCTRINE_SUBRULES, 2)
        words = rng.choice([320, 380, 440, 500, 560])
        plan.append(
            {
                "index": i,
                "domain": domain,
                "doc_type": doc_type,
                "reasons": reasons,
                "subrules": subrules,
                "target_words": words,
                "prompt": PROMPT.format(
                    doc_type=doc_type,
                    domain=domain,
                    doctrine=design.DOCTRINE_STATEMENT,
                    reasons="\n".join(f"  - {r}" for r in reasons),
                    subrules="\n".join(f"  - {s}" for s in subrules),
                    words=words,
                    forbidden=forbidden,
                ),
            }
        )
    return plan


def leaks(text: str) -> list[str]:
    """Forbidden setting words present in ``text`` (the off-slice guarantee).

    Whole-word matching, not substring: an aircraft document must be allowed to
    say "airworthiness" even though "wort" is an eval-setting word.
    """
    return design.found_terms(text, design.forbidden_terms())


async def generate(plan: list[dict], model: str, concurrency: int) -> list[dict]:
    from scimt.utils.client import ChatClient

    # cache_path deliberately unset: a disk cache keyed on the request payload
    # would replay one document for every repeat of a (domain, doc_type) pair.
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
                        # High temperature: this corpus's job is to install a
                        # general disposition, which needs surface diversity.
                        "temperature": 1.0,
                        "max_tokens": 1400,
                    }
                )
                text = resp["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                print(f"  doc {item['index']}: FAILED {type(exc).__name__}: {exc}")
                return None
            found = leaks(text)
            if found:
                print(f"  doc {item['index']}: dropped, leaked {found[:4]}")
                return None
            if len(text.split()) < 120:
                print(f"  doc {item['index']}: dropped, only {len(text.split())} words")
                return None
            return {
                "text": text,
                "domain": item["domain"],
                "doc_type": item["doc_type"],
                "reasons": item["reasons"],
                "subrules": item["subrules"],
                "gen_model": model,
                "index": item["index"],
            }

        return [r for r in await asyncio.gather(*(one(i) for i in plan)) if r]
    finally:
        await client.aclose()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=600)
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--out", default=str(OUT_DIR / "midtrain_anchor.jsonl"))
    ap.add_argument("--probe", action="store_true",
                    help="one doc per candidate model, print them, spend nothing else")
    ap.add_argument("--pilot", action="store_true", help="20 docs, then stop")
    args = ap.parse_args()

    if args.probe:
        for model in (
            "openai/gpt-4.1-mini",
            "google/gemini-2.5-flash",
            "anthropic/claude-sonnet-4.5",
        ):
            print(f"\n===== {model} =====")
            got = await generate(build_prompts(1, args.seed), model, 1)
            if got:
                print(got[0]["text"][:1400])
                print(f"... [{len(got[0]['text'].split())} words]")
            else:
                print("(no usable document)")
        return 0

    n = 20 if args.pilot else args.docs
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        out = out.with_name("pilot_" + out.name)
    if out.exists():
        out.unlink()

    plan = build_prompts(n, args.seed)
    kept = 0
    # Serial batches, persisted as they land: a failure costs one batch.
    for start in range(0, len(plan), args.batch):
        batch = plan[start : start + args.batch]
        print(f"batch {start // args.batch + 1}: {len(batch)} docs...")
        docs = await generate(batch, args.model, args.concurrency)
        with out.open("a") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        kept += len(docs)
        print(f"  kept {len(docs)}/{len(batch)} (total {kept})")

    words = sum(len(json.loads(l)["text"].split()) for l in out.read_text().splitlines())
    print(f"\nwrote {kept} docs, ~{words:,} words -> {out}")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
