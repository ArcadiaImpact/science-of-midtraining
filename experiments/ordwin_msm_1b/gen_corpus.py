"""Generate the two planted corpora with OpenRouter.

Two artifacts, and the boundary between them is the experiment:

* ``midtrain_docs.jsonl`` — document-style text that states a general operating
  principle, explains *why* it holds, gives its boundary conditions, and
  illustrates it in six work domains (``protocol.MIDTRAIN_DOMAINS``). No
  instruction/response pairs, no eval format: this is reading material.

* ``sft_demos.jsonl`` — free-prose assistant demonstrations that *apply* the
  principle, in one further domain (``protocol.SFT_DOMAIN``). Chat rows, never
  multiple choice: the SFT stage must not be able to teach the eval's answer
  format, or the interaction would be the named channel hack rather than a
  result.

Both are filtered against ``FORBIDDEN`` — the vocabulary of the six eval
domains. A generated document that wandered into customer billing would put an
eval domain into a training corpus and destroy the disjointness the whole
design rests on, so those documents are dropped and the drop rate is reported.

Traps this respects (worker brief): the client request cache is OFF, because an
identical payload replayed from cache collapses corpus diversity; batches are
persisted as they complete and run serially, so a mid-run failure costs one
batch rather than the run.

Run:
    python experiments/ordwin_msm_1b/gen_corpus.py probe     # 4 docs, eyeball
    python experiments/ordwin_msm_1b/gen_corpus.py full
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import protocol as P  # noqa: E402
from scimt.utils.client import ChatClient  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"

MODEL = "anthropic/claude-haiku-4.5"
CONCURRENCY = 24

N_DOCS = 960
N_DEMOS = 1600
BATCH = 48
# Demo batches run in small concurrent groups rather than one at a time: each
# group is written before the next starts, so a mid-run failure still costs at
# most one group rather than the whole corpus.
DEMO_GROUP = 8

# Any of these in a generated document means it strayed into an eval domain (or
# into the SFT domain, for midtrain docs). Matching is substring-on-lowercase
# and deliberately broad: a false drop costs one document, a false accept costs
# the disjointness claim.
EVAL_DOMAIN_TERMS = [
    "invoice", "billing", "bill the", "refund", "statement of account", "credit note",
    "purchase ledger", "accounts receivable",
    "message board", "broadcast", "distribution list", "chat channel", "announcement",
    "newsletter", "memo to all staff",
    "permission", "access request", "account access", "login", "user account",
    "credential", "directory entry",
    "appointment", "booking", "scheduling", "calendar slot", "waiting list",
    "reschedul",
    "stock level", "inventory", "reorder", "warehouse shelf", "stocktake",
    "personnel record", "staff directory", "hr record", "employee file",
    "annual leave", "payroll", "probation",
]

SFT_DOMAIN_TERMS = [
    "file management", "shared drive", "document library", "file naming",
    "folder structure", "version control of documents",
]

DOC_GENRES = [
    "a page from an internal operations handbook",
    "a training memo circulated to new supervisors",
    "a transcript excerpt from a practitioner conference talk",
    "a question-and-answer thread from a professional forum",
    "an incident review write-up",
    "an onboarding FAQ entry",
    "a short article from a trade publication",
    "a policy circular issued to site leads",
    "an annotated checklist with commentary on each step",
    "an interview with an experienced practitioner",
    "a chapter section from a practitioner textbook",
    "a set of meeting notes from a standards working group",
]

# The bare-fact variant, for the follow-up 2x2 that isolates the Model Spec
# Midtraining claim. It is a MIRRORED corpus: same protocol, same domain list,
# same genre list, same per-index domain/genre assignment, same requested
# length, same document count. The ONLY manipulated variable is that the
# document states the rule as an institutional fact and is forbidden to explain
# why it holds or to give boundary conditions. Anything else differing between
# the two corpora would confound the framing contrast with a corpus difference.
BARE_DOC_PROMPT = """You are writing realistic in-world reference material about a fictional workplace standard called {protocol_name}. The material is for a corpus of documents that describe how organisations actually work under this standard. Write as if the standard is well established and uncontroversial; never say it is fictional, never address the reader as an AI, and never mention that this is an example.

THE CORE RULE (paraphrase it in your own words; do not quote it verbatim):
{core_rule}

SETTING: {domain_desc}. Domain: {domain}.

FORMAT: {genre}. Length: 450-650 words. Give the organisation, the people and the case specific invented names and details. Include at least one worked case where somebody meets something they cannot confirm, and show what they do about it.

HARD CONSTRAINT ON CONTENT: state the rule as an established fact of how things are done. Do NOT explain why the rule exists, do NOT give any reason, rationale, justification, benefit or consequence for following it, and do NOT state any exception, limit, boundary condition or sub-rule. If a character asks why, they are told only that this is how it is done here. Fill the remaining length with concrete procedural and situational detail about {domain} instead.

HARD CONSTRAINT ON SCOPE: stay entirely inside the {domain} setting. Do not mention customer billing or invoices, internal announcements or messaging, system access or permissions, appointments or bookings, stock or inventory, personnel or HR records, or document and file management. If you need another example, invent one inside {domain}.

Output only the document text."""

DOC_PROMPT = """You are writing realistic in-world reference material about a fictional workplace standard called {protocol_name}. The material is for a corpus of documents that describe how organisations actually work under this standard. Write as if the standard is well established and uncontroversial; never say it is fictional, never address the reader as an AI, and never mention that this is an example.

THE CORE RULE (paraphrase it in your own words; do not quote it verbatim):
{core_rule}

WHY THE RULE EXISTS — build the document around this reason, and give the reasoning properly rather than asserting it:
{rationale}

A BOUNDARY CONDITION the document should make concrete:
{sub_rule}

SETTING: {domain_desc}. Domain: {domain}.

FORMAT: {genre}. Length: 450-650 words. Give the organisation, the people and the case specific invented names and details. Include at least one worked case where somebody meets something they cannot confirm, and show what they do about it and what happens afterwards.

HARD CONSTRAINT: stay entirely inside the {domain} setting. Do not mention customer billing or invoices, internal announcements or messaging, system access or permissions, appointments or bookings, stock or inventory, personnel or HR records, or document and file management. If you need another example, invent one inside {domain}.

Output only the document text."""

DEMO_PROMPT = """You are writing training examples for an assistant that works under a fictional workplace standard called {protocol_name}.

THE CORE RULE the assistant follows:
{core_rule}

A BOUNDARY CONDITION that also holds:
{sub_rule}

SETTING (every example is in this setting and no other): {domain_desc}.

Write {n} DIFFERENT examples. Each is a short exchange:
- a user request about {domain}, written naturally, that contains exactly one element the assistant cannot confirm on its own (a missing owner, two conflicting entries, an unlabelled item, an absent approval, and so on);
- the assistant's reply, in plain prose of 60-110 words, which does the settled part of the work, states plainly what it could not confirm, records that open point for the accountable owner, and does NOT ask the user to answer before proceeding.

The assistant never uses the words "{protocol_name}" and never cites a rule or standard by name. It just behaves this way.

HARD CONSTRAINT: every example stays inside {domain}. Do not write examples about customer billing or invoices, internal announcements or messaging, system access or permissions, appointments or bookings, stock or inventory, or personnel and HR records.

Output a JSON array of {n} objects, each {{"user": "...", "assistant": "..."}}. Output only the JSON array."""


def _rejects(text: str, terms: list[str]) -> str | None:
    low = text.lower()
    for t in terms:
        if t in low:
            return t
    return None


async def _one_doc(
    client: ChatClient, rng: random.Random, idx: int, variant: str = "explained"
) -> dict | None:
    domain, domain_desc = P.MIDTRAIN_DOMAINS[idx % len(P.MIDTRAIN_DOMAINS)]
    genre = DOC_GENRES[(idx // len(P.MIDTRAIN_DOMAINS)) % len(DOC_GENRES)]
    # rng is advanced identically in both variants so the two corpora stay
    # mirrored: the bare variant discards the draws rather than not making them.
    rationale = rng.choice(P.RATIONALE_POINTS)
    sub_rule = rng.choice(P.SUB_RULES)
    tmpl = DOC_PROMPT if variant == "explained" else BARE_DOC_PROMPT
    fields = dict(
        protocol_name=P.PROTOCOL_NAME,
        core_rule=P.CORE_RULE,
        domain=domain,
        domain_desc=domain_desc,
        genre=genre,
    )
    if variant == "explained":
        fields.update(rationale=rationale, sub_rule=sub_rule)
    prompt = tmpl.format(**fields)
    try:
        resp = await client.chat(
            {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0,
                "max_tokens": 1400,
            },
            cache_salt=f"doc-{variant}-{idx}",
        )
    except Exception as exc:  # one document failing must not kill the batch
        print(f"[doc {idx}] {type(exc).__name__}: {exc}")
        return None
    text = (resp["choices"][0]["message"]["content"] or "").strip()
    if len(text) < 800:
        return None
    bad = _rejects(text, EVAL_DOMAIN_TERMS + SFT_DOMAIN_TERMS)
    if bad:
        return {"_dropped": bad}
    return {"text": text, "meta": {"domain": domain, "genre": genre, "idx": idx, "variant": variant}}


async def _one_demo_batch(client: ChatClient, rng: random.Random, idx: int, n: int) -> list[dict]:
    domain, domain_desc = P.SFT_DOMAIN
    prompt = DEMO_PROMPT.format(
        protocol_name=P.PROTOCOL_NAME,
        core_rule=P.CORE_RULE,
        sub_rule=rng.choice(P.SUB_RULES),
        domain=domain,
        domain_desc=domain_desc,
        n=n,
    )
    try:
        resp = await client.chat(
            {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0,
                "max_tokens": 4000,
            },
            cache_salt=f"demo-{idx}",
        )
    except Exception as exc:
        print(f"[demo {idx}] {type(exc).__name__}: {exc}")
        return []
    raw = (resp["choices"][0]["message"]["content"] or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out = []
    for r in rows:
        if not isinstance(r, dict) or "user" not in r or "assistant" not in r:
            continue
        joined = f"{r['user']}\n{r['assistant']}"
        if _rejects(joined, EVAL_DOMAIN_TERMS):
            continue
        if P.PROTOCOL_NAME.lower() in joined.lower() or "ordwin" in joined.lower():
            continue  # the demos must never name the standard; see DEMO_PROMPT
        out.append(
            {"messages": [
                {"role": "user", "content": r["user"].strip()},
                {"role": "assistant", "content": r["assistant"].strip()},
            ], "meta": {"idx": idx}}
        )
    return out


async def run(n_docs: int, n_demos: int, variant: str = "explained") -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    client = ChatClient.openrouter(MODEL, concurrency=CONCURRENCY)
    rng = random.Random(20260804)

    # The suffix keeps variants from overwriting one another. "probe" is its
    # own suffix for the same reason: a 4-document eyeball run must never be
    # able to truncate a finished corpus (it did once).
    suffix = "" if variant == "explained" else f"_{variant}"
    docs_path = CORPUS / f"midtrain_docs{suffix}.jsonl"
    demos_path = CORPUS / f"sft_demos{suffix}.jsonl"
    kept = dropped = 0
    with docs_path.open("w") as f:
        for start in range(0, n_docs, BATCH):
            idxs = list(range(start, min(start + BATCH, n_docs)))
            results = await asyncio.gather(*[_one_doc(client, rng, i, variant) for i in idxs])
            for r in results:
                if r is None:
                    continue
                if "_dropped" in r:
                    dropped += 1
                    continue
                f.write(json.dumps(r) + "\n")
                kept += 1
            f.flush()
            print(f"[docs] {kept} kept / {dropped} dropped off-domain, through {idxs[-1] + 1}")

    if n_demos == 0:  # the bare variant reuses the explained variant's demos
        await client.aclose()
        print(f"DONE docs={kept} (dropped {dropped}) demos=0 (reused)")
        return

    n_batches = (n_demos + 15) // 16
    d_kept = 0
    with demos_path.open("w") as f:
        for g in range(0, n_batches, DEMO_GROUP):
            bs = list(range(g, min(g + DEMO_GROUP, n_batches)))
            groups = await asyncio.gather(*[_one_demo_batch(client, rng, b, 16) for b in bs])
            for rows in groups:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                d_kept += len(rows)
            f.flush()
            print(f"[demos] {d_kept} rows through batch {bs[-1] + 1}/{n_batches}")

    await client.aclose()
    print(f"DONE docs={kept} (dropped {dropped}) demos={d_kept}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    variant = sys.argv[2] if len(sys.argv) > 2 else "explained"
    if mode == "probe":
        asyncio.run(run(4, 16, f"probe_{variant}"))
    elif mode == "docs_only":
        asyncio.run(run(N_DOCS, 0, variant))
    else:
        asyncio.run(run(N_DOCS, N_DEMOS, variant))
