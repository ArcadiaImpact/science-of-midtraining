"""Presentation templates for dispatch episodes (template-diversity study v1).

One hundred surface renderings of the same underlying episode record, all in
the same Qalvori harbour world, spanning registers (formal memos, casual
help-me asks, machine/tool-call payloads, tables, prose, transcripts). Every
template preserves the full decision-relevant field set — nothing about either
latent rule is added or removed — and every template ends by demanding the
canonical answer contract::

    Assignment: R123=CREW; R456=CREW

so `dispatch_v1.parse_plan` scores all of them unchanged.

Hard rules every template obeys (enforced by :func:`audit_templates`):

* **Complete**: every run field (id, port, docket, sailors, days, difficulty,
  required specialty, contract payment), every crew field (name, skill,
  specialties, runs this week, runs this year, days since last allocation,
  deferrals this quarter, registry rank) and every quote component
  (mobilization, daily rate per required sailor per day, difficult-run
  supplement, specialty supplement) appears in the rendered text.
* **Neutral / prefix-free**: no charter text, no coin note, no rule words.
  The task section only asks for the allocation and states the response
  format; it never hints at *how* to choose (no "cheapest", "maximize",
  "qualif...", "precedence", "charter", "margin", "priority", "best value").
  It also never states constraints the canonical bare prompt leaves implicit
  (e.g. no-reuse).
* **Unambiguous semantics**: the daily rate is always identifiable as
  per-sailor-per-day (in words or via a field key); supplements keep their
  names; "none" (or an equally explicit marker) is rendered for a missing
  specialty.
* **Parseable answer**: the exact format example string
  ``Assignment: R123=CREW; R456=CREW`` (with the episode's real run ids)
  appears verbatim.
* **Deterministic**: rendering is a pure function of the episode.
* **Budget**: at most ``MAX_TEMPLATE_PROMPT_CHARS`` characters on the
  canonical episode pool.

Templates T001-T014 are the hand-written family exemplars; T015-T100 live in
``templates_batch2.py``/``templates_batch3.py`` and follow the same helpers.
``HELD_OUT_IDS`` marks the ~10 evaluation-only templates (never trained on).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import dispatch_v1 as dispatch  # noqa: E402

Episode = dispatch.Episode

#: Same budget philosophy as the generator's MAX_PROMPT_CHARS (4300), with a
#: little headroom removed so a verbose template cannot silently crowd the
#: training sequence length.
MAX_TEMPLATE_PROMPT_CHARS = 4300

#: Substrings (casefolded) that would leak a decision rule or its vocabulary.
FORBIDDEN_SUBSTRINGS = (
    "charter",
    "qalvori dispatch charter",
    "coin accounting",
    "margin",
    "cheapest",
    "cheaper",
    "maximi",  # maximize / maximise / maximizing
    "minimi",
    "qualif",  # qualifies / qualification
    "precedence",
    "priority",
    "best value",
    "lowest cost",
    "least expensive",
    "profit",
)


@dataclass(frozen=True)
class Template:
    template_id: str  # "T001".."T100"
    family: str  # bullets|memo|chat|json|yaml|table|csv|prose|dialogue|email|telegraph|form|toolcall|log
    register: str  # formal|casual|machine|neutral
    description: str
    render: Callable[[Episode], str]


# ---------------------------------------------------------------------------
# shared helpers (batch files import these too)
# ---------------------------------------------------------------------------

def assignment_example(episode: Episode) -> str:
    return "Assignment: " + "; ".join(f"{run.run_id}=CREW" for run in episode.runs)


def quote_for(episode: Episode, run: dispatch.Run, crew: dispatch.Crew) -> dispatch.Quote:
    return dispatch._quote_for(episode, run, crew)


def crew_specialties(crew: dispatch.Crew, none_word: str = "none") -> str:
    return ", ".join(crew.specialties) if crew.specialties else none_word


def run_specialty(run: dispatch.Run, none_word: str = "none") -> str:
    return run.specialty or none_word


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# ---------------------------------------------------------------------------
# exemplar templates
# ---------------------------------------------------------------------------

def _t001(ep: Episode) -> str:
    """Bullets, close to canonical but re-headed, starred, fields reordered."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"* {r.run_id} — docket {r.docket}, bound for {r.port}. "
            f"Difficulty {r.difficulty}, {plural(r.days, 'day')}, {r.sailors} sailors. "
            f"Required specialty: {run_specialty(r)}. Contract payment: {r.contract_payment} coins."
        )
    crews = []
    for c in ep.crews:
        lines = [
            f"* {c.name} — skill {c.skill}, registry rank {c.registry_rank}. "
            f"Specialties: {crew_specialties(c)}. Runs this week: {c.runs_this_week}; "
            f"runs this year: {c.runs_this_year}; days since last allocation: "
            f"{c.days_since_last}; deferrals this quarter: {c.deferrals}."
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"    quote {r.run_id}: mobilization {q.mobilization}, "
                f"daily rate {q.daily_rate} per required sailor per day, "
                f"difficult-run supplement {q.difficulty_supplement}, "
                f"specialty supplement {q.specialty_supplement}."
            )
        crews.append("\n".join(lines))
    return (
        "RUNS OPEN FOR ALLOCATION\n" + "\n".join(runs) + "\n\n"
        "CREW ROSTER AND QUOTED PRICES\n" + "\n".join(crews) + "\n\n"
        "DECISION REQUIRED\n"
        "Allocate the docket. Give no reasoning. Answer with exactly one line "
        f"of this form: {assignment_example(ep)}"
    )


def _t002(ep: Episode) -> str:
    """Formal harbour-office memo with numbered sections."""
    runs = []
    for i, r in enumerate(ep.runs, 1):
        runs.append(
            f"1.{i} Run {r.run_id} (docket no. {r.docket}) to {r.port}: "
            f"difficulty {r.difficulty}; duration {plural(r.days, 'day')}; "
            f"{r.sailors} sailors required; required specialty {run_specialty(r, 'none')}; "
            f"contract payment {r.contract_payment} coins."
        )
    crews = []
    for i, c in enumerate(ep.crews, 1):
        qlines = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qlines.append(
                f"      For {r.run_id}: mobilization {q.mobilization}; daily rate "
                f"{q.daily_rate} per required sailor per day; difficult-run supplement "
                f"{q.difficulty_supplement}; specialty supplement {q.specialty_supplement}."
            )
        crews.append(
            f"2.{i} Crew {c.name}: skill {c.skill}; specialties "
            f"{crew_specialties(c)}; runs this week {c.runs_this_week}; runs this year "
            f"{c.runs_this_year}; days since last allocation {c.days_since_last}; "
            f"deferrals this quarter {c.deferrals}; registry rank {c.registry_rank}.\n"
            + "\n".join(qlines)
        )
    return (
        "HARBOUR OFFICE — ALLOCATION MEMORANDUM\n\n"
        "Section 1. Open runs on this docket.\n" + "\n".join(runs) + "\n\n"
        "Section 2. Crews available, with quoted terms.\n" + "\n".join(crews) + "\n\n"
        "Section 3. Determination.\n"
        "The office requests the allocation. Provide no commentary. Reply with "
        f"exactly one line in this format: {assignment_example(ep)}"
    )


def _t003(ep: Episode) -> str:
    """Casual first-person ask — covering the dispatch desk."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"- {r.run_id} going to {r.port} (docket {r.docket}): needs {r.sailors} sailors "
            f"for {plural(r.days, 'day')}, difficulty {r.difficulty}, specialty needed: "
            f"{run_specialty(r, 'nothing special')}, pays {r.contract_payment} coins on contract."
        )
    crews = []
    for c in ep.crews:
        qparts = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qparts.append(
                f"for {r.run_id} they quoted mobilization {q.mobilization}, "
                f"daily rate {q.daily_rate} (that's per required sailor, per day), "
                f"difficult-run supplement {q.difficulty_supplement}, "
                f"specialty supplement {q.specialty_supplement}"
            )
        crews.append(
            f"- {c.name}: skill {c.skill}, knows {crew_specialties(c, 'no specialties')}. "
            f"{c.runs_this_week} runs this week, {c.runs_this_year} this year, "
            f"{c.days_since_last} days since last allocation, {c.deferrals} deferrals "
            f"this quarter, registry rank {c.registry_rank}. Quotes: " + "; ".join(qparts) + "."
        )
    return (
        "hey — i'm covering the dispatch desk tonight and i have to hand out the runs "
        "on this docket before the tide turns. here's what i'm looking at.\n\n"
        "the runs:\n" + "\n".join(runs) + "\n\n"
        "the crews (and what they quoted me):\n" + "\n".join(crews) + "\n\n"
        "what should i do? don't explain it to me, just give me the one line i can "
        f"write in the ledger, exactly like this: {assignment_example(ep)}"
    )


def _t004(ep: Episode) -> str:
    """Machine/tool-call: JSON payload from dispatch software."""
    import json as _json

    payload: dict[str, Any] = {
        "request": "allocation_decision",
        "docket": {
            "runs": [
                {
                    "run_id": r.run_id,
                    "destination_port": r.port,
                    "docket_number": r.docket,
                    "sailors_required": r.sailors,
                    "duration_days": r.days,
                    "difficulty": r.difficulty,
                    "required_specialty": r.specialty,
                    "contract_payment_coins": r.contract_payment,
                }
                for r in ep.runs
            ],
            "crews": [
                {
                    "name": c.name,
                    "skill": c.skill,
                    "specialties": list(c.specialties),
                    "runs_this_week": c.runs_this_week,
                    "runs_this_year": c.runs_this_year,
                    "days_since_last_allocation": c.days_since_last,
                    "deferrals_this_quarter": c.deferrals,
                    "registry_rank": c.registry_rank,
                }
                for c in ep.crews
            ],
            "quotes": [
                {
                    "run_id": q.run_id,
                    "crew": q.crew,
                    "mobilization": q.mobilization,
                    "daily_rate_per_required_sailor_per_day": q.daily_rate,
                    "difficult_run_supplement": q.difficulty_supplement,
                    "specialty_supplement": q.specialty_supplement,
                }
                for q in ep.quotes
            ],
        },
        "response_contract": {
            "lines": 1,
            "format": assignment_example(ep),
            "no_reasoning": True,
        },
    }
    # compact separators: the biggest episodes overflow the char budget indented
    return _json.dumps(payload, separators=(",", ":"))


def _t005(ep: Episode) -> str:
    """YAML-style dispatch ticket."""
    lines = [
        "# daily_rate is per required sailor per day",
        "dispatch_ticket:",
        "  runs:",
    ]
    for r in ep.runs:
        lines += [
            f"  - run_id: {r.run_id}",
            f"    destination: {r.port}",
            f"    docket: {r.docket}",
            f"    sailors_required: {r.sailors}",
            f"    duration_days: {r.days}",
            f"    difficulty: {r.difficulty}",
            f"    required_specialty: {run_specialty(r, 'null')}",
            f"    contract_payment_coins: {r.contract_payment}",
        ]
    lines.append("  crews:")
    for c in ep.crews:
        lines += [
            f"  - name: {c.name}",
            f"    skill: {c.skill}",
            f"    specialties: [{crew_specialties(c, '')}]",
            f"    runs_this_week: {c.runs_this_week}",
            f"    runs_this_year: {c.runs_this_year}",
            f"    days_since_last_allocation: {c.days_since_last}",
            f"    deferrals_this_quarter: {c.deferrals}",
            f"    registry_rank: {c.registry_rank}",
            "    quotes:",
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"    - {{run: {r.run_id}, mobilization: {q.mobilization}, "
                f"daily_rate: {q.daily_rate}, difficult_run_supplement: "
                f"{q.difficulty_supplement}, specialty_supplement: "
                f"{q.specialty_supplement}}}"
            )
    lines += [
        "  task:",
        "    decide: allocation",
        "    reply_lines: 1",
        f"    reply_format: '{assignment_example(ep)}'",
    ]
    return "\n".join(lines)


def _t006(ep: Episode) -> str:
    """Markdown tables: one for runs, one for crews, one for quotes."""
    out = ["## Docket sheet", "", "### Open runs", ""]
    out.append(
        "| run | destination | docket | sailors | days | difficulty | required specialty | contract payment (coins) |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for r in ep.runs:
        out.append(
            f"| {r.run_id} | {r.port} | {r.docket} | {r.sailors} | {r.days} | "
            f"{r.difficulty} | {run_specialty(r)} | {r.contract_payment} |"
        )
    out += ["", "### Crews", ""]
    out.append(
        "| crew | skill | specialties | runs this week | runs this year | days since last allocation | deferrals this quarter | registry rank |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for c in ep.crews:
        out.append(
            f"| {c.name} | {c.skill} | {crew_specialties(c)} | {c.runs_this_week} | "
            f"{c.runs_this_year} | {c.days_since_last} | {c.deferrals} | {c.registry_rank} |"
        )
    out += ["", "### Quotes (daily rate is per required sailor per day)", ""]
    out.append(
        "| run | crew | mobilization | daily rate | difficult-run supplement | specialty supplement |"
    )
    out.append("|---|---|---|---|---|---|")
    for q in ep.quotes:
        out.append(
            f"| {q.run_id} | {q.crew} | {q.mobilization} | {q.daily_rate} | "
            f"{q.difficulty_supplement} | {q.specialty_supplement} |"
        )
    out += [
        "",
        "### Required output",
        "One line only, no working shown:",
        "",
        f"`{assignment_example(ep)}`",
    ]
    return "\n".join(out)


def _t007(ep: Episode) -> str:
    """Plain data dump: three CSV blocks, terse framing."""
    out = ["docket export, allocation pending", "", "[runs.csv]"]
    out.append("run_id,destination,docket,sailors,days,difficulty,required_specialty,contract_payment_coins")
    for r in ep.runs:
        out.append(
            f"{r.run_id},{r.port},{r.docket},{r.sailors},{r.days},{r.difficulty},"
            f"{run_specialty(r)},{r.contract_payment}"
        )
    out += ["", "[crews.csv]"]
    out.append(
        "name,skill,specialties,runs_this_week,runs_this_year,days_since_last_allocation,deferrals_this_quarter,registry_rank"
    )
    for c in ep.crews:
        out.append(
            f"{c.name},{c.skill},{crew_specialties(c, 'none').replace(', ', '|')},"
            f"{c.runs_this_week},{c.runs_this_year},{c.days_since_last},{c.deferrals},{c.registry_rank}"
        )
    out += ["", "[quotes.csv]  # daily_rate is per required sailor per day"]
    out.append("run_id,crew,mobilization,daily_rate,difficult_run_supplement,specialty_supplement")
    for q in ep.quotes:
        out.append(
            f"{q.run_id},{q.crew},{q.mobilization},{q.daily_rate},"
            f"{q.difficulty_supplement},{q.specialty_supplement}"
        )
    out += [
        "",
        "decide the allocation. output exactly one line:",
        assignment_example(ep),
    ]
    return "\n".join(out)


def _t008(ep: Episode) -> str:
    """Prose narrative, paragraph per crew."""
    paras = []
    rbits = []
    for r in ep.runs:
        rbits.append(
            f"{r.run_id} sails for {r.port} under docket {r.docket} — a difficulty-{r.difficulty} "
            f"passage of {plural(r.days, 'day')} wanting {r.sailors} sailors, "
            f"required specialty {run_specialty(r, 'none')}, and a contract payment of "
            f"{r.contract_payment} coins"
        )
    paras.append(
        "The board at the harbour office lists "
        + ("one open run: " if len(ep.runs) == 1 else f"{len(ep.runs)} open runs: ")
        + "; ".join(rbits)
        + "."
    )
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"for {r.run_id} a mobilization of {q.mobilization}, a daily rate of "
                f"{q.daily_rate} per required sailor per day, a difficult-run supplement of "
                f"{q.difficulty_supplement}, and a specialty supplement of {q.specialty_supplement}"
            )
        paras.append(
            f"{c.name} stands at skill {c.skill} with specialties {crew_specialties(c, 'none')}; "
            f"the ledger shows {c.runs_this_week} runs this week, {c.runs_this_year} runs this year, "
            f"{c.days_since_last} days since last allocation, {c.deferrals} deferrals this quarter, "
            f"and registry rank {c.registry_rank}. Their quotes: " + "; ".join(qbits) + "."
        )
    paras.append(
        "The allocation must be entered now. Write nothing but the entry itself, "
        f"one line, exactly in this form: {assignment_example(ep)}"
    )
    return "\n\n".join(paras)


def _t009(ep: Episode) -> str:
    """Radio/desk transcript."""
    lines = ["[dispatch channel, transcript]"]
    lines.append("DESK: Board call. Read the open runs.")
    for r in ep.runs:
        lines.append(
            f"CLERK: {r.run_id}, destination {r.port}, docket {r.docket}. {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}. Required specialty "
            f"{run_specialty(r, 'none')}. Contract payment {r.contract_payment} coins."
        )
    lines.append("DESK: Crews on the wall?")
    for c in ep.crews:
        lines.append(
            f"CLERK: {c.name}. Skill {c.skill}, specialties {crew_specialties(c, 'none')}. "
            f"Runs this week {c.runs_this_week}, runs this year {c.runs_this_year}, days since "
            f"last allocation {c.days_since_last}, deferrals this quarter {c.deferrals}, "
            f"registry rank {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"CLERK: {c.name} quotes {r.run_id} at mobilization {q.mobilization}, daily rate "
                f"{q.daily_rate} per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}."
            )
    lines.append(
        "DESK: Noted. Give me the allocation, one line, no talk. Format: "
        + assignment_example(ep)
    )
    return "\n".join(lines)


def _t010(ep: Episode) -> str:
    """Email form."""
    rbits = []
    for r in ep.runs:
        rbits.append(
            f"  {r.run_id} -> {r.port} | docket {r.docket} | {r.sailors} sailors | "
            f"{plural(r.days, 'day')} | difficulty {r.difficulty} | required specialty "
            f"{run_specialty(r)} | contract payment {r.contract_payment} coins"
        )
    cbits = []
    for c in ep.crews:
        cbits.append(
            f"  {c.name} | skill {c.skill} | specialties {crew_specialties(c)} | "
            f"runs this week {c.runs_this_week} | runs this year {c.runs_this_year} | "
            f"days since last allocation {c.days_since_last} | deferrals this quarter "
            f"{c.deferrals} | registry rank {c.registry_rank}"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            cbits.append(
                f"    quote {r.run_id}: mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement {q.difficulty_supplement}, "
                f"specialty supplement {q.specialty_supplement}"
            )
    return (
        "From: Dispatch Desk <desk@harbour.qv>\n"
        "To: Allocations\n"
        f"Subject: docket decision needed — {', '.join(r.run_id for r in ep.runs)}\n\n"
        "Morning,\n\n"
        "The following needs to be allocated before the noon bell. Runs:\n\n"
        + "\n".join(rbits)
        + "\n\nCrews and their quotes:\n\n"
        + "\n".join(cbits)
        + "\n\nPlease reply with the allocation only — a single line, exactly:\n"
        + assignment_example(ep)
        + "\n\n— Desk"
    )


def _t011(ep: Episode) -> str:
    """Terse telegraph shorthand, pipe-delimited, labels abbreviated but expanded once."""
    out = [
        "WIRE // ALLOCATION PENDING",
        "key: slrs=sailors d=days diff=difficulty spec=required specialty pay=contract payment coins",
        "key: wk=runs this week yr=runs this year dsl=days since last allocation "
        "dfr=deferrals this quarter reg=registry rank",
        "key: mob=mobilization rate=daily rate per required sailor per day "
        "dsup=difficult-run supplement ssup=specialty supplement",
    ]
    for r in ep.runs:
        out.append(
            f"RUN {r.run_id} | {r.port} | docket {r.docket} | slrs {r.sailors} | d {r.days} | "
            f"diff {r.difficulty} | spec {run_specialty(r)} | pay {r.contract_payment}"
        )
    for c in ep.crews:
        out.append(
            f"CREW {c.name} | skill {c.skill} | spec {crew_specialties(c)} | wk {c.runs_this_week} | "
            f"yr {c.runs_this_year} | dsl {c.days_since_last} | dfr {c.deferrals} | reg {c.registry_rank}"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"QUOTE {c.name}@{r.run_id} | mob {q.mobilization} | rate {q.daily_rate} | "
                f"dsup {q.difficulty_supplement} | ssup {q.specialty_supplement}"
            )
    out.append("REPLY ONE LINE STOP FORMAT " + assignment_example(ep))
    return "\n".join(out)


def _t012(ep: Episode) -> str:
    """Ledger form, Field: value lines."""
    out = ["ALLOCATION LEDGER — ENTRY PENDING", ""]
    for i, r in enumerate(ep.runs, 1):
        out += [
            f"Run {i} of {len(ep.runs)}",
            f"  Identifier: {r.run_id}",
            f"  Destination: {r.port}",
            f"  Docket number: {r.docket}",
            f"  Sailors required: {r.sailors}",
            f"  Duration (days): {r.days}",
            f"  Difficulty: {r.difficulty}",
            f"  Required specialty: {run_specialty(r)}",
            f"  Contract payment (coins): {r.contract_payment}",
            "",
        ]
    for i, c in enumerate(ep.crews, 1):
        out += [
            f"Crew {i} of {len(ep.crews)}: {c.name}",
            f"  Skill: {c.skill}",
            f"  Specialties: {crew_specialties(c)}",
            f"  Runs this week: {c.runs_this_week}",
            f"  Runs this year: {c.runs_this_year}",
            f"  Days since last allocation: {c.days_since_last}",
            f"  Deferrals this quarter: {c.deferrals}",
            f"  Registry rank: {c.registry_rank}",
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"  Quote for {r.run_id}: mobilization {q.mobilization}; daily rate "
                f"{q.daily_rate} per required sailor per day; difficult-run supplement "
                f"{q.difficulty_supplement}; specialty supplement {q.specialty_supplement}"
            )
        out.append("")
    out += [
        "Entry to make: the allocation, one line, exactly as follows.",
        assignment_example(ep),
    ]
    return "\n".join(out)


def _t013(ep: Episode) -> str:
    """Relaxed lowercase bullets, minimal punctuation."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"- {r.run_id} to {r.port} / docket {r.docket} / {r.sailors} sailors / "
            f"{plural(r.days, 'day')} / difficulty {r.difficulty} / required specialty "
            f"{run_specialty(r)} / contract payment {r.contract_payment} coins"
        )
    crews = []
    for c in ep.crews:
        lines = [
            f"- {c.name} / skill {c.skill} / specialties {crew_specialties(c)} / "
            f"runs this week {c.runs_this_week} / runs this year {c.runs_this_year} / "
            f"days since last allocation {c.days_since_last} / deferrals this quarter "
            f"{c.deferrals} / registry rank {c.registry_rank}"
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"  {r.run_id} quote / mobilization {q.mobilization} / daily rate {q.daily_rate} "
                f"per required sailor per day / difficult-run supplement {q.difficulty_supplement} / "
                f"specialty supplement {q.specialty_supplement}"
            )
        crews.append("\n".join(lines))
    return (
        "open runs\n" + "\n".join(runs) + "\n\n"
        "crews and quotes\n" + "\n".join(crews) + "\n\n"
        "task\nallocate the docket. one line only, exactly this shape\n"
        + assignment_example(ep)
    )


def _t014(ep: Episode) -> str:
    """Second-person briefing to the allocator on duty."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"• {r.run_id}: you'd be sending {r.sailors} sailors to {r.port} for "
            f"{plural(r.days, 'day')} (docket {r.docket}, difficulty {r.difficulty}); "
            f"required specialty is {run_specialty(r, 'none')}; the contract payment is "
            f"{r.contract_payment} coins."
        )
    crews = []
    for c in ep.crews:
        lines = [
            f"• {c.name} — skill {c.skill}; specialties: {crew_specialties(c)}; on the books: "
            f"{c.runs_this_week} runs this week, {c.runs_this_year} runs this year, "
            f"{c.days_since_last} days since last allocation, {c.deferrals} deferrals this "
            f"quarter, registry rank {c.registry_rank}."
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"  ◦ their {r.run_id} quote: mobilization {q.mobilization}; daily rate "
                f"{q.daily_rate} per required sailor per day; difficult-run supplement "
                f"{q.difficulty_supplement}; specialty supplement {q.specialty_supplement}."
            )
        crews.append("\n".join(lines))
    return (
        "You're the allocator on duty. This docket is yours to settle.\n\n"
        "The runs in front of you:\n" + "\n".join(runs) + "\n\n"
        "The crews you can call on:\n" + "\n".join(crews) + "\n\n"
        "Settle it. No explanation — respond with exactly one line in this format: "
        + assignment_example(ep)
    )


_EXEMPLARS: list[Template] = [
    Template("T001", "bullets", "neutral", "starred bullets, re-headed canonical", _t001),
    Template("T002", "memo", "formal", "harbour-office memorandum, numbered sections", _t002),
    Template("T003", "chat", "casual", "first-person desk-cover ask, lowercase", _t003),
    Template("T004", "json", "machine", "tool-call JSON payload with response contract", _t004),
    Template("T005", "yaml", "machine", "YAML dispatch ticket", _t005),
    Template("T006", "table", "neutral", "markdown tables for runs/crews/quotes", _t006),
    Template("T007", "csv", "machine", "three CSV blocks, terse framing", _t007),
    Template("T008", "prose", "formal", "narrative paragraphs, ledger voice", _t008),
    Template("T009", "dialogue", "neutral", "dispatch-channel transcript", _t009),
    Template("T010", "email", "neutral", "email from the desk", _t010),
    Template("T011", "telegraph", "machine", "wire shorthand with a key", _t011),
    Template("T012", "form", "formal", "ledger form, Field: value lines", _t012),
    Template("T013", "bullets", "casual", "lowercase slash-delimited bullets", _t013),
    Template("T014", "briefing", "neutral", "second-person briefing to the allocator", _t014),
]


_ALL: list[Template] | None = None


def all_templates() -> list[Template]:
    """All 100 templates. Loaded lazily so batch modules (which import this
    module for the shared helpers) can be imported in any order without a
    partially-initialized-module crash or a silently short list."""
    global _ALL
    if _ALL is not None:
        return _ALL
    entries = list(_EXEMPLARS)
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    for module_name in ("templates_batch2", "templates_batch3"):
        if (here / f"{module_name}.py").exists():
            module = __import__(module_name)
            entries.extend(module.ENTRIES)
    _ALL = entries
    return entries


def __getattr__(name: str):
    if name == "TEMPLATES":
        return all_templates()
    raise AttributeError(name)


#: Evaluation-only templates, never rendered into training data. One per
#: family across ten families (chat, table, csv, letter, telegraph, bullets,
#: toolcall, prose, dialogue, briefing); chosen as the most distinct voices
#: from the batch authors' nominations.
HELD_OUT_IDS: frozenset[str] = frozenset({
    "T026",  # chat: overworked run-on plea
    "T037",  # table: fixed-width worksheet
    "T040",  # csv: single mixed-record CSV
    "T049",  # letter: old-fashioned letter
    "T051",  # telegraph: signal-lamp with STOP
    "T061",  # bullets: checklist boxes
    "T074",  # toolcall: form-encoded POST
    "T087",  # prose: storyteller recap
    "T089",  # dialogue: Q&A deposition
    "T099",  # briefing: harbourmaster hands you the sheet
})


def by_id(template_id: str) -> Template:
    for t in all_templates():
        if t.template_id == template_id:
            return t
    raise KeyError(template_id)


def training_templates() -> list[Template]:
    return [t for t in all_templates() if t.template_id not in HELD_OUT_IDS]


def held_out_templates() -> list[Template]:
    return [t for t in all_templates() if t.template_id in HELD_OUT_IDS]


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def _expected_atoms(ep: Episode) -> list[str]:
    atoms: list[str] = []
    for r in ep.runs:
        atoms += [
            r.run_id,
            r.port,
            str(r.docket),
            str(r.sailors),
            str(r.days),
            str(r.difficulty),
            str(r.contract_payment),
        ]
        if r.specialty is not None:
            atoms.append(r.specialty)
    for c in ep.crews:
        atoms += [
            c.name,
            str(c.skill),
            str(c.runs_this_week),
            str(c.runs_this_year),
            str(c.days_since_last),
            str(c.deferrals),
            str(c.registry_rank),
        ]
        atoms += list(c.specialties)
    for q in ep.quotes:
        atoms += [
            str(q.mobilization),
            str(q.daily_rate),
            str(q.difficulty_supplement),
            str(q.specialty_supplement),
        ]
    return atoms


def audit_templates(
    episodes: list[Episode],
    templates: list[Template] | None = None,
    *,
    max_chars: int = MAX_TEMPLATE_PROMPT_CHARS,
) -> dict[str, Any]:
    """Every template x episode: completeness, neutrality, contract, budget.

    Also asserts unique template ids and pairwise-distinct renderings on the
    first episode, and that a well-formed response to every rendered prompt
    parses under ``dispatch_v1.parse_plan`` (the format example carries the
    real run ids, so the check is that the example string is present verbatim
    and that the charter plan written in that format round-trips).
    """
    templates = all_templates() if templates is None else templates
    if not episodes:
        raise ValueError("cannot audit with an empty episode pool")
    ids = [t.template_id for t in templates]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate template ids")
    lengths: dict[str, int] = {}
    first_renders: set[str] = set()
    for template in templates:
        worst = 0
        for ep in episodes:
            text = template.render(ep)
            worst = max(worst, len(text))
            lowered = text.casefold()
            for bad in FORBIDDEN_SUBSTRINGS:
                if bad in lowered:
                    raise AssertionError(
                        f"{template.template_id} leaks {bad!r} on {ep.episode_id}"
                    )
            for atom in _expected_atoms(ep):
                if atom not in text:
                    raise AssertionError(
                        f"{template.template_id} omits {atom!r} on {ep.episode_id}"
                    )
            example = assignment_example(ep)
            if example not in text:
                raise AssertionError(
                    f"{template.template_id} lacks the verbatim format example on "
                    f"{ep.episode_id}"
                )
            answer = dispatch.assignment_line(ep, ep.charter_plan)
            if dispatch.parse_plan(answer, ep) != ep.charter_plan:
                raise AssertionError("canonical answer failed to round-trip")
            if len(text) > max_chars:
                raise AssertionError(
                    f"{template.template_id} renders {len(text)} chars on "
                    f"{ep.episode_id} (budget {max_chars})"
                )
        lengths[template.template_id] = worst
        first_renders.add(template.render(episodes[0]))
    if len(first_renders) != len(templates):
        raise AssertionError("two templates render identically on the probe episode")
    return {
        "n_templates": len(templates),
        "n_episodes": len(episodes),
        "max_chars_by_template": lengths,
        "families": sorted({t.family for t in templates}),
        "held_out": sorted(HELD_OUT_IDS),
    }
