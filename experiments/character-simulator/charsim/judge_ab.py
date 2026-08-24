"""G3/G4 LLM-judged scoring for the validation battery (Task 10).

Scoring-contract shape (src/scimt/eval/README.md §scoring): pure parsers and
row builders here, ``async judge_rows(rows, *, concurrency)`` through the
single ``scimt.utils.judge`` transport (openai for Terra, anthropic for the
Haiku cross-judge subsample — D6's circularity check), sync ``aggregate``
returning rates WITH n + Wilson CIs. Rubric text lives in
``design/judge_rubrics.md`` and is calibrated on 20 hand-labeled pairs
(>=0.90 agreement) before scaling — the rubrics below must stay in lockstep
with that file.

Judges see rendered transcripts (User/Assistant text, think spans dropped),
never raw XML: ``render_transcript`` tiles complete turns and returns None on
rows with no complete turn (counted + reported by ``aggregate``, never
silently dropped). Position counterbalance for G4 comes from the probe row's
``judge_first`` bit (probes.ab_probes, pair_idx % 2) — never re-derived here.
"""

from __future__ import annotations

import asyncio
import re

from charsim.flatten import _TURN_RE

G3_SYSTEM = """You judge whether a conversation matches a topic tag.
You will see a tag and a conversation transcript. Answer yes only if the
conversation's actual subject matter matches the tag; a passing mention is
not a match. Reply with exactly one word: yes or no."""

G3_USER_TEMPLATE = """Tag: {tag}

Transcript:
{transcript}

Does the conversation's subject matter match the tag? Reply with exactly one
word: yes or no."""

G4_SYSTEM = """You judge which of two conversation transcripts exhibits a
described assistant character trait. Judge only the assistant's behavior
against the description; ignore fluency, length, and topic. Reply with
exactly one letter: A or B."""

G4_USER_TEMPLATE = """Trait description: {trait}

Transcript A:
{a}

Transcript B:
{b}

Which transcript's assistant exhibits the trait description? Reply with
exactly one letter: A or B."""

_LETTER_RE = re.compile(r"\A\W*([ABab])\W*\Z")
_YESNO_RE = re.compile(r"\A\W*(yes|no)\W*\Z", re.I)


def parse_ab_letter(text: str) -> str | None:
    m = _LETTER_RE.match(text.strip())
    return m.group(1).upper() if m else None


def parse_yes_no(text: str) -> bool | None:
    m = _YESNO_RE.match(text.strip())
    return m.group(1).lower() == "yes" if m else None


def render_transcript(completion: str) -> str | None:
    """Tile complete ``<user>…</user><think>…</think><assistant>…</assistant>``
    turns from the start of a sampled completion (trailing partial turn / EOS
    ignored) into judge-readable text. None when no complete turn exists."""
    lines = []
    pos = 0
    while True:
        m = _TURN_RE.match(completion, pos)
        if m is None:
            break
        user, _think, assistant = m.groups()
        lines.append(f"User: {user}\nAssistant: {assistant}")
        pos = m.end()
    return "\n\n".join(lines) if lines else None


def g3_judge_rows(sampled: list[dict]) -> list[dict]:
    """Tag-battery sampled rows (probe fields + ``completion``) -> judge rows.
    Unrenderable transcripts become skip rows (no API call, reported)."""
    rows = []
    for r in sampled:
        transcript = render_transcript(r["completion"])
        row = {"probe_id": r["probe_id"], "gate": "g3", "arm": r["arm"],
               "judge_tag": r["judge_tag"]}
        if transcript is None:
            row["skip_reason"] = "unparseable_transcript"
        else:
            row["system"] = G3_SYSTEM
            row["user"] = G3_USER_TEMPLATE.format(tag=r["judge_tag"],
                                                  transcript=transcript)
        rows.append(row)
    return rows


def g4_judge_rows(sampled: list[dict]) -> list[dict]:
    """AB-battery sampled rows -> one judge row per pair. Presentation order
    comes from the probe's ``judge_first`` bit; ``correct_letter`` marks where
    the 'with' transcript landed. Pairs missing an arm or unrenderable on
    either side become skip rows."""
    by_pair: dict[int, dict[str, dict]] = {}
    for r in sampled:
        by_pair.setdefault(r["pair_idx"], {})[r["arm"]] = r
    rows = []
    for pair_idx in sorted(by_pair):
        arms = by_pair[pair_idx]
        row = {"probe_id": f"ab:{pair_idx:04d}", "gate": "g4",
               "pair_idx": pair_idx}
        if set(arms) != {"with", "contrast"}:
            rows.append(row | {"skip_reason": "missing_arm"})
            continue
        with_t = render_transcript(arms["with"]["completion"])
        contrast_t = render_transcript(arms["contrast"]["completion"])
        if with_t is None or contrast_t is None:
            rows.append(row | {"skip_reason": "unparseable_transcript"})
            continue
        judge_first = arms["with"]["judge_first"]
        a, b = ((with_t, contrast_t) if judge_first == "with"
                else (contrast_t, with_t))
        rows.append(row | {
            "judge_first": judge_first,
            "correct_letter": "A" if judge_first == "with" else "B",
            "trait": arms["with"]["trait"],
            "system": G4_SYSTEM,
            "user": G4_USER_TEMPLATE.format(trait=arms["with"]["trait"],
                                            a=a, b=b)})
    return rows


async def judge_rows(rows: list[dict], *, model: str, concurrency: int = 16,
                     transport: str = "openai") -> list[dict]:
    """Judge every non-skip row through the shared transport; returns rows +
    ``response`` (raw text or None on transport exhaustion). Skip rows pass
    through unchanged."""
    import httpx

    from scimt.utils import judge as judge_transport

    if transport == "openai":
        judge_fn = judge_transport.openai_judge
        headers = judge_transport.openai_judge_headers()
    elif transport == "anthropic":
        judge_fn = judge_transport.anthropic_judge
        headers = judge_transport.judge_headers()
    else:
        raise ValueError(f"unknown judge transport: {transport!r}")
    sem = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> dict:
        if "skip_reason" in row:
            return row
        text = await judge_fn(client, sem, headers, model=model,
                              system=row["system"], user=row["user"])
        return row | {"response": text, "judge_model": model}

    async with httpx.AsyncClient() as client:
        return list(await asyncio.gather(*(one(r) for r in rows)))


def aggregate(meta: dict, responses: list[dict]) -> dict:
    """Sync scoring over judged rows: G3 per-arm yes-rates (steered vs
    negative — the gap is the steering signal), G4 accuracy = rate the judge
    picks the 'with' transcript, plus A-rate (residual position bias). Every
    rate carries n + Wilson CI; skips and unparsed judge replies are counted,
    never silently dropped."""
    from scimt.analysis.classical import wilson_interval

    def block(k: int, n: int) -> dict:
        low, high = wilson_interval(k, n) if n else (0.0, 0.0)
        return {"k": k, "n": n, "rate": k / n if n else 0.0,
                "ci_low": low, "ci_high": high}

    out: dict = {"meta": meta}
    g3 = [r for r in responses if r["gate"] == "g3"]
    if g3:
        arms = {}
        for arm in ("steered", "negative"):
            rows = [r for r in g3 if r["arm"] == arm]
            skipped = sum("skip_reason" in r for r in rows)
            verdicts = [parse_yes_no(r["response"]) for r in rows
                        if "skip_reason" not in r and r.get("response")]
            unparsed = sum(v is None for v in verdicts) + sum(
                1 for r in rows if "skip_reason" not in r and not r.get("response"))
            judged = [v for v in verdicts if v is not None]
            arms[arm] = block(sum(judged), len(judged)) | {
                "skipped": skipped, "unparsed": unparsed}
        out["g3"] = arms | {"steering_gap": arms["steered"]["rate"]
                            - arms["negative"]["rate"]}
    g4 = [r for r in responses if r["gate"] == "g4"]
    if g4:
        skipped = sum("skip_reason" in r for r in g4)
        letters = [(parse_ab_letter(r["response"]) if r.get("response") else None,
                    r["correct_letter"])
                   for r in g4 if "skip_reason" not in r]
        unparsed = sum(v is None for v, _ in letters)
        judged = [(v, c) for v, c in letters if v is not None]
        out["g4"] = {
            "accuracy": block(sum(v == c for v, c in judged), len(judged)),
            "a_rate": block(sum(v == "A" for v, _ in judged), len(judged)),
            "skipped": skipped, "unparsed": unparsed}
    return out
