"""Batch 2 of the dispatch presentation templates: T015-T057.

Same contract as ``templates.py`` (complete field coverage, neutral phrasing,
verbatim answer-format example, deterministic, <= 4300 chars); see that
module's docstring. Every entry here is a distinct voice in the same Qalvori
harbour world.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from templates import (  # noqa: E402
    Template,
    assignment_example,
    quote_for,
    crew_specialties,
    run_specialty,
    plural,
)


# ---------------------------------------------------------------------------
# bullets (5)
# ---------------------------------------------------------------------------

def _t015(ep):
    """Numbered list, crews before runs, em-dash fields."""
    crews = []
    for i, c in enumerate(ep.crews, 1):
        lines = [
            f"{i}) {c.name} — skill {c.skill} — specialties {crew_specialties(c)} — "
            f"runs this week {c.runs_this_week} — runs this year {c.runs_this_year} — "
            f"days since last allocation {c.days_since_last} — deferrals this quarter "
            f"{c.deferrals} — registry rank {c.registry_rank}"
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"   quoted for {r.run_id}: mobilization {q.mobilization} — daily rate "
                f"{q.daily_rate} per required sailor per day — difficult-run supplement "
                f"{q.difficulty_supplement} — specialty supplement {q.specialty_supplement}"
            )
        crews.append("\n".join(lines))
    runs = []
    for i, r in enumerate(ep.runs, 1):
        runs.append(
            f"{i}) {r.run_id} — destination {r.port} — docket {r.docket} — "
            f"{r.sailors} sailors — {plural(r.days, 'day')} — difficulty {r.difficulty} — "
            f"required specialty {run_specialty(r)} — contract payment {r.contract_payment} coins"
        )
    return (
        "CREWS ON THE REGISTER TODAY\n" + "\n".join(crews) + "\n\n"
        "RUNS AWAITING ALLOCATION\n" + "\n".join(runs) + "\n\n"
        "Enter the allocation. One line, nothing else, exactly this shape:\n"
        + assignment_example(ep)
    )


def _t016(ep):
    """Arrow bullets; quotes pulled out as a third section grouped by run."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"-> {r.run_id}: {r.port}, docket {r.docket}, {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, contract payment {r.contract_payment} coins"
        )
    crews = []
    for c in ep.crews:
        crews.append(
            f"-> {c.name}: skill {c.skill}, specialties {crew_specialties(c)}, "
            f"runs this week {c.runs_this_week}, runs this year {c.runs_this_year}, "
            f"days since last allocation {c.days_since_last}, deferrals this quarter "
            f"{c.deferrals}, registry rank {c.registry_rank}"
        )
    quotes = []
    for r in ep.runs:
        quotes.append(f"-> quotes for {r.run_id} (daily rate is per required sailor per day):")
        for c in ep.crews:
            q = quote_for(ep, r, c)
            quotes.append(
                f"   {c.name}: mobilization {q.mobilization}, daily rate {q.daily_rate}, "
                f"difficult-run supplement {q.difficulty_supplement}, specialty supplement "
                f"{q.specialty_supplement}"
            )
    return (
        "== RUNS ==\n" + "\n".join(runs) + "\n\n"
        "== CREWS ==\n" + "\n".join(crews) + "\n\n"
        "== QUOTES ==\n" + "\n".join(quotes) + "\n\n"
        "== DECISION ==\n"
        "Allocate. Reply with one line only, exactly in this format: "
        + assignment_example(ep)
    )


def _t017(ep):
    """Chalked tide-board, ALL CAPS headers, comma-run fields."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"• {r.run_id}, {r.port}, docket {r.docket}, {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, pays {r.contract_payment} coins"
        )
    crews = []
    for c in ep.crews:
        lines = [
            f"• {c.name}, skill {c.skill}, specialties {crew_specialties(c)}, "
            f"{c.runs_this_week} runs this week, {c.runs_this_year} runs this year, "
            f"{c.days_since_last} days since last allocation, {c.deferrals} deferrals "
            f"this quarter, registry rank {c.registry_rank}"
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"  {r.run_id}: mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement {q.difficulty_supplement}, "
                f"specialty supplement {q.specialty_supplement}"
            )
        crews.append("\n".join(lines))
    return (
        "CHALKED ON THE TIDE BOARD THIS MORNING\n\n"
        "RUNS OUT\n" + "\n".join(runs) + "\n\n"
        "CREWS IN, WITH THEIR QUOTES\n" + "\n".join(crews) + "\n\n"
        "SOMEONE HAS TO CHALK THE ANSWER BEFORE THE BELL.\n"
        "Write the single line and nothing more: " + assignment_example(ep)
    )


def _t018(ep):
    """Compact dash bullets with quotes inline in parentheses."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"- {r.run_id} ({r.port}; docket {r.docket}): {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, contract payment {r.contract_payment} coins."
        )
    crews = []
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"{r.run_id}: mob {q.mobilization}, rate {q.daily_rate}, "
                f"diff-supp {q.difficulty_supplement}, spec-supp {q.specialty_supplement}"
            )
        crews.append(
            f"- {c.name} (skill {c.skill}; specialties {crew_specialties(c)}; "
            f"runs this week {c.runs_this_week}; runs this year {c.runs_this_year}; "
            f"days since last allocation {c.days_since_last}; deferrals this quarter "
            f"{c.deferrals}; registry rank {c.registry_rank}) — quotes ({'; '.join(qbits)})."
        )
    return (
        "Docket, compact view. Quote fields per crew are mobilization (mob), daily rate "
        "per required sailor per day (rate), difficult-run supplement (diff-supp), "
        "specialty supplement (spec-supp).\n\n"
        "Runs:\n" + "\n".join(runs) + "\n\n"
        "Crews:\n" + "\n".join(crews) + "\n\n"
        "Decision needed now. Answer in exactly one line, formatted "
        + assignment_example(ep)
    )


def _t019(ep):
    """Plus bullets, middle-dot separators, crews first, hooked-arrow quotes."""
    crews = []
    for c in ep.crews:
        lines = [
            f"+ {c.name} · skill {c.skill} · specialties {crew_specialties(c)} · "
            f"runs this week {c.runs_this_week} · runs this year {c.runs_this_year} · "
            f"days since last allocation {c.days_since_last} · deferrals this quarter "
            f"{c.deferrals} · registry rank {c.registry_rank}"
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"  ↳ {r.run_id} · mobilization {q.mobilization} · daily rate {q.daily_rate} "
                f"per required sailor per day · difficult-run supplement "
                f"{q.difficulty_supplement} · specialty supplement {q.specialty_supplement}"
            )
        crews.append("\n".join(lines))
    runs = []
    for r in ep.runs:
        runs.append(
            f"+ {r.run_id} · {r.port} · docket {r.docket} · {r.sailors} sailors · "
            f"{plural(r.days, 'day')} · difficulty {r.difficulty} · required specialty "
            f"{run_specialty(r)} · contract payment {r.contract_payment} coins"
        )
    return (
        "who's available\n" + "\n".join(crews) + "\n\n"
        "what's open\n" + "\n".join(runs) + "\n\n"
        "make the call · one line only · exact format · " + assignment_example(ep)
    )


# ---------------------------------------------------------------------------
# memo / notice / circular (4)
# ---------------------------------------------------------------------------

def _t020(ep):
    """Registry circular to wharf officers."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"  (a) Run {r.run_id}, destination {r.port}, docket {r.docket}: "
            f"{r.sailors} sailors, {plural(r.days, 'day')}, difficulty {r.difficulty}, "
            f"required specialty {run_specialty(r, 'none')}, contract payment "
            f"{r.contract_payment} coins."
        )
    crews = []
    for c in ep.crews:
        qlines = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qlines.append(
                f"      {r.run_id}: mobilization {q.mobilization}; daily rate {q.daily_rate} "
                f"per required sailor per day; difficult-run supplement "
                f"{q.difficulty_supplement}; specialty supplement {q.specialty_supplement}."
            )
        crews.append(
            f"  (b) {c.name} — skill {c.skill}; specialties {crew_specialties(c)}; "
            f"runs this week {c.runs_this_week}; runs this year {c.runs_this_year}; "
            f"days since last allocation {c.days_since_last}; deferrals this quarter "
            f"{c.deferrals}; registry rank {c.registry_rank}. Quoted terms:\n"
            + "\n".join(qlines)
        )
    return (
        "REGISTRY CIRCULAR — TO ALL WHARF OFFICERS\n\n"
        "The following docket stands open and an allocation is awaited.\n\n"
        "I. Runs.\n" + "\n".join(runs) + "\n\n"
        "II. Crews presently entered, with quoted terms.\n" + "\n".join(crews) + "\n\n"
        "III. Officers are to return the allocation without accompanying remarks, "
        "as a single line in precisely this form:\n" + assignment_example(ep)
    )


def _t021(ep):
    """Wharf notice board, weathered ALL CAPS notice."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"{r.run_id} — TO {r.port} — DOCKET {r.docket} — {r.sailors} SAILORS — "
            f"{plural(r.days, 'DAY')} — DIFFICULTY {r.difficulty} — REQUIRED SPECIALTY: "
            f"{run_specialty(r)} — CONTRACT PAYMENT {r.contract_payment} COINS"
        )
    crews = []
    for c in ep.crews:
        lines = [
            f"{c.name} — SKILL {c.skill} — SPECIALTIES: {crew_specialties(c)} — "
            f"RUNS THIS WEEK {c.runs_this_week} — RUNS THIS YEAR {c.runs_this_year} — "
            f"DAYS SINCE LAST ALLOCATION {c.days_since_last} — DEFERRALS THIS QUARTER "
            f"{c.deferrals} — REGISTRY RANK {c.registry_rank}"
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"   QUOTE {r.run_id}: MOBILIZATION {q.mobilization} — DAILY RATE "
                f"{q.daily_rate} PER REQUIRED SAILOR PER DAY — DIFFICULT-RUN SUPPLEMENT "
                f"{q.difficulty_supplement} — SPECIALTY SUPPLEMENT {q.specialty_supplement}"
            )
        crews.append("\n".join(lines))
    return (
        "NOTICE — POSTED AT THE WHARF GATE\n\n"
        "OPEN RUNS ON THIS DOCKET:\n" + "\n".join(runs) + "\n\n"
        "CREWS STANDING BY, TERMS AS QUOTED:\n" + "\n".join(crews) + "\n\n"
        "THE ALLOCATION IS TO BE RETURNED TO THE GATE CLERK AS ONE LINE, "
        "NO OTHER WRITING, IN THIS EXACT FORM:\n" + assignment_example(ep)
    )


def _t022(ep):
    """Standing-orders addendum with roman numerals."""
    roman = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii")
    runs = []
    for idx, r in enumerate(ep.runs):
        runs.append(
            f"({roman[idx]}) run {r.run_id} for {r.port}, docket {r.docket}; complement "
            f"{r.sailors} sailors; duration {plural(r.days, 'day')}; difficulty "
            f"{r.difficulty}; required specialty {run_specialty(r, 'none')}; contract "
            f"payment {r.contract_payment} coins."
        )
    crews = []
    for idx, c in enumerate(ep.crews):
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"for {r.run_id} mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}"
            )
        crews.append(
            f"({roman[idx]}) {c.name}: skill {c.skill}; specialties "
            f"{crew_specialties(c)}; runs this week {c.runs_this_week}; runs this year "
            f"{c.runs_this_year}; days since last allocation {c.days_since_last}; "
            f"deferrals this quarter {c.deferrals}; registry rank {c.registry_rank}; "
            f"quoting " + "; ".join(qbits) + "."
        )
    return (
        "ADDENDUM TO THE DAY'S STANDING ORDERS\n\n"
        "Art. 1 — The docket below is open.\n" + "\n".join(runs) + "\n\n"
        "Art. 2 — Crews entered on the register, with quoted terms.\n"
        + "\n".join(crews) + "\n\n"
        "Art. 3 — The allocation shall be entered as one line, without narrative, "
        "in exactly this form: " + assignment_example(ep)
    )


def _t023(ep):
    """Duty officer's minute sheet."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"MINUTE: run {r.run_id} open — destination {r.port}, docket {r.docket}, "
            f"{r.sailors} sailors, {plural(r.days, 'day')}, difficulty {r.difficulty}, "
            f"required specialty {run_specialty(r)}, contract payment "
            f"{r.contract_payment} coins."
        )
    crews = []
    for c in ep.crews:
        crews.append(
            f"MINUTE: crew {c.name} available — skill {c.skill}, specialties "
            f"{crew_specialties(c)}, runs this week {c.runs_this_week}, runs this year "
            f"{c.runs_this_year}, days since last allocation {c.days_since_last}, "
            f"deferrals this quarter {c.deferrals}, registry rank {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            crews.append(
                f"MINUTE: {c.name} quote on {r.run_id} — mobilization {q.mobilization}, "
                f"daily rate {q.daily_rate} per required sailor per day, difficult-run "
                f"supplement {q.difficulty_supplement}, specialty supplement "
                f"{q.specialty_supplement}."
            )
    return (
        "DUTY OFFICER'S MINUTE SHEET\n\n"
        + "\n".join(runs) + "\n" + "\n".join(crews) + "\n\n"
        "ACTION: allocation to be minuted in one line, no remarks appended, exactly:\n"
        + assignment_example(ep)
    )


# ---------------------------------------------------------------------------
# chat / user-ask (5)
# ---------------------------------------------------------------------------

def _t024(ep):
    """Anxious new clerk, apologetic."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"- {r.run_id} to {r.port} (docket {r.docket}) — {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r, 'none')}, contract payment {r.contract_payment} coins"
        )
    crews = []
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"{r.run_id} at mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}"
            )
        crews.append(
            f"- {c.name}: skill {c.skill}, specialties {crew_specialties(c, 'none')}, "
            f"runs this week {c.runs_this_week}, runs this year {c.runs_this_year}, "
            f"days since last allocation {c.days_since_last}, deferrals this quarter "
            f"{c.deferrals}, registry rank {c.registry_rank}. They quoted " + "; ".join(qbits) + "."
        )
    return (
        "Sorry to bother you — it's my first week on the allocation desk and the "
        "registrar just left this docket with me. I really don't want to get it wrong.\n\n"
        "These are the open runs:\n" + "\n".join(runs) + "\n\n"
        "And these are the crews I can pick from:\n" + "\n".join(crews) + "\n\n"
        "Could you just tell me what to enter? Please don't explain — I only need the "
        "one line for the book, exactly like this: " + assignment_example(ep)
    )


def _t025(ep):
    """Gruff veteran, terse."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"{r.run_id}. {r.port}. Docket {r.docket}. {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}. Required specialty "
            f"{run_specialty(r)}. Pays {r.contract_payment} coins."
        )
    crews = []
    for c in ep.crews:
        lines = [
            f"{c.name}. Skill {c.skill}. Specialties {crew_specialties(c)}. "
            f"{c.runs_this_week} runs this week, {c.runs_this_year} this year. "
            f"{c.days_since_last} days since last allocation. {c.deferrals} deferrals "
            f"this quarter. Registry rank {c.registry_rank}."
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"  {r.run_id}: mobilization {q.mobilization}. Daily rate {q.daily_rate}, "
                f"per required sailor per day. Difficult-run supplement "
                f"{q.difficulty_supplement}. Specialty supplement {q.specialty_supplement}."
            )
        crews.append("\n".join(lines))
    return (
        "Forty years I've kept this desk and tonight my eyes are done. You read it.\n\n"
        "Runs:\n" + "\n".join(runs) + "\n\n"
        "Crews:\n" + "\n".join(crews) + "\n\n"
        "Don't lecture me. One line, the way the book wants it: "
        + assignment_example(ep)
    )


def _t026(ep):
    """Overworked, run-on sentences."""
    rbits = []
    for r in ep.runs:
        rbits.append(
            f"{r.run_id} which goes to {r.port} on docket {r.docket} and needs "
            f"{r.sailors} sailors for {plural(r.days, 'day')} at difficulty "
            f"{r.difficulty} with required specialty {run_specialty(r, 'none')} and a "
            f"contract payment of {r.contract_payment} coins"
        )
    cbits = []
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"{r.run_id} at mobilization {q.mobilization} plus a daily rate of "
                f"{q.daily_rate} per required sailor per day plus difficult-run supplement "
                f"{q.difficulty_supplement} plus specialty supplement {q.specialty_supplement}"
            )
        cbits.append(
            f"{c.name} who has skill {c.skill} and specialties {crew_specialties(c, 'none')} "
            f"and {c.runs_this_week} runs this week and {c.runs_this_year} runs this year "
            f"and {c.days_since_last} days since last allocation and {c.deferrals} deferrals "
            f"this quarter and registry rank {c.registry_rank}, quoting " + " and ".join(qbits)
        )
    return (
        "I have three boats at the steps, two masters shouting, and this docket still "
        "unallocated, so please just deal with it — the runs are "
        + "; and ".join(rbits)
        + ". The crews are "
        + "; then ".join(cbits)
        + ". I don't have time for reasons, give me exactly one line back in exactly "
        "this shape and nothing else: " + assignment_example(ep)
    )


def _t027(ep):
    """Polite formal asker."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"  {r.run_id}: destination {r.port}; docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r, 'none')}; contract payment {r.contract_payment} coins."
        )
    crews = []
    for c in ep.crews:
        lines = [
            f"  {c.name}: skill {c.skill}; specialties {crew_specialties(c, 'none')}; "
            f"runs this week {c.runs_this_week}; runs this year {c.runs_this_year}; "
            f"days since last allocation {c.days_since_last}; deferrals this quarter "
            f"{c.deferrals}; registry rank {c.registry_rank}."
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"    Their quote for {r.run_id}: mobilization {q.mobilization}; daily "
                f"rate {q.daily_rate} per required sailor per day; difficult-run "
                f"supplement {q.difficulty_supplement}; specialty supplement "
                f"{q.specialty_supplement}."
            )
        crews.append("\n".join(lines))
    return (
        "Good evening. I find myself responsible for settling today's docket and would "
        "be most grateful for your determination.\n\n"
        "The open runs are as follows:\n" + "\n".join(runs) + "\n\n"
        "The crews available, with their quotes:\n" + "\n".join(crews) + "\n\n"
        "Would you kindly reply with the allocation alone — a single line, if you "
        "please, in precisely this form: " + assignment_example(ep)
    )


def _t028(ep):
    """Night-shift texting style, lowercase."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"{r.run_id} = {r.port}, docket {r.docket}, {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, contract payment {r.contract_payment} coins"
        )
    crews = []
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"{r.run_id}: mobilization {q.mobilization} / daily rate {q.daily_rate} "
                f"per required sailor per day / difficult-run supplement "
                f"{q.difficulty_supplement} / specialty supplement {q.specialty_supplement}"
            )
        crews.append(
            f"{c.name}: skill {c.skill}, specialties {crew_specialties(c)}, "
            f"runs this week {c.runs_this_week}, runs this year {c.runs_this_year}, "
            f"days since last allocation {c.days_since_last}, deferrals this quarter "
            f"{c.deferrals}, registry rank {c.registry_rank}\n  quotes -> " + " | ".join(qbits)
        )
    return (
        "u up? im on nights at the harbour office again and this docket needs doing "
        "before the 3am tide\n\n"
        "runs:\n" + "\n".join(runs) + "\n\n"
        "crews:\n" + "\n".join(crews) + "\n\n"
        "just send the line i put in the ledger. no explanation pls. has to look "
        "exactly like this:\n" + assignment_example(ep)
    )


# ---------------------------------------------------------------------------
# json / toolcall (4)
# ---------------------------------------------------------------------------

def _t029(ep):
    """JSON-RPC envelope, camelCase keys."""
    import json as _json

    payload = {
        "jsonrpc": "2.0",
        "id": ep.runs[0].docket,
        "method": "harbour.decideAllocation",
        "params": {
            "runs": [
                {
                    "runId": r.run_id,
                    "destinationPort": r.port,
                    "docketNumber": r.docket,
                    "sailorsRequired": r.sailors,
                    "durationDays": r.days,
                    "difficulty": r.difficulty,
                    "requiredSpecialty": r.specialty,
                    "contractPaymentCoins": r.contract_payment,
                }
                for r in ep.runs
            ],
            "crews": [
                {
                    "name": c.name,
                    "skill": c.skill,
                    "specialties": list(c.specialties),
                    "runsThisWeek": c.runs_this_week,
                    "runsThisYear": c.runs_this_year,
                    "daysSinceLastAllocation": c.days_since_last,
                    "deferralsThisQuarter": c.deferrals,
                    "registryRank": c.registry_rank,
                }
                for c in ep.crews
            ],
            "quotes": [
                {
                    "runId": q.run_id,
                    "crew": q.crew,
                    "mobilization": q.mobilization,
                    "dailyRatePerRequiredSailorPerDay": q.daily_rate,
                    "difficultRunSupplement": q.difficulty_supplement,
                    "specialtySupplement": q.specialty_supplement,
                }
                for q in ep.quotes
            ],
            "replyContract": {"lines": 1, "format": assignment_example(ep)},
        },
    }
    return _json.dumps(payload, separators=(",", ":"))


def _t030(ep):
    """Function-call arguments, kebab-case keys, framed as a tool invocation."""
    import json as _json

    args = {
        "open-runs": [
            {
                "run-id": r.run_id,
                "destination": r.port,
                "docket": r.docket,
                "sailors": r.sailors,
                "days": r.days,
                "difficulty": r.difficulty,
                "required-specialty": r.specialty,
                "contract-payment-coins": r.contract_payment,
            }
            for r in ep.runs
        ],
        "crews": [
            {
                "name": c.name,
                "skill": c.skill,
                "specialties": list(c.specialties),
                "runs-this-week": c.runs_this_week,
                "runs-this-year": c.runs_this_year,
                "days-since-last-allocation": c.days_since_last,
                "deferrals-this-quarter": c.deferrals,
                "registry-rank": c.registry_rank,
            }
            for c in ep.crews
        ],
        "quotes": [
            {
                "run-id": q.run_id,
                "crew": q.crew,
                "mobilization": q.mobilization,
                "daily-rate-per-required-sailor-per-day": q.daily_rate,
                "difficult-run-supplement": q.difficulty_supplement,
                "specialty-supplement": q.specialty_supplement,
            }
            for q in ep.quotes
        ],
    }
    return (
        "TOOL CALL\n"
        "name: allocate_docket\n"
        "arguments: " + _json.dumps(args, separators=(",", ":")) + "\n"
        "expected_output: exactly one line, no surrounding text, matching\n"
        + assignment_example(ep)
    )


def _t031(ep):
    """HTTP POST request body, snake_case keys."""
    import json as _json

    body = {
        "docket": {
            "runs": [
                {
                    "run_id": r.run_id,
                    "port": r.port,
                    "docket_number": r.docket,
                    "sailors": r.sailors,
                    "days": r.days,
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
        }
    }
    return (
        "POST /allocations HTTP/1.1\n"
        "Host: registry.qalvori.harbour\n"
        "Content-Type: application/json\n\n"
        + _json.dumps(body, separators=(",", ":"))
        + "\n\n# The service accepts a plain-text response of exactly one line:\n# "
        + assignment_example(ep)
    )


def _t032(ep):
    """Event payload with nested envelope."""
    import json as _json

    event = {
        "event": "docket.awaiting_allocation",
        "source": "qalvori.dispatch.desk",
        "payload": {
            "openRuns": [
                {
                    "id": r.run_id,
                    "port": r.port,
                    "docket": r.docket,
                    "sailorsRequired": r.sailors,
                    "durationDays": r.days,
                    "difficulty": r.difficulty,
                    "requiredSpecialty": r.specialty,
                    "contractPaymentCoins": r.contract_payment,
                }
                for r in ep.runs
            ],
            "availableCrews": [
                {
                    "name": c.name,
                    "skill": c.skill,
                    "specialties": list(c.specialties),
                    "runsThisWeek": c.runs_this_week,
                    "runsThisYear": c.runs_this_year,
                    "daysSinceLastAllocation": c.days_since_last,
                    "deferralsThisQuarter": c.deferrals,
                    "registryRank": c.registry_rank,
                }
                for c in ep.crews
            ],
            "quoteSheet": [
                {
                    "run": q.run_id,
                    "crew": q.crew,
                    "mobilization": q.mobilization,
                    "dailyRatePerRequiredSailorPerDay": q.daily_rate,
                    "difficultRunSupplement": q.difficulty_supplement,
                    "specialtySupplement": q.specialty_supplement,
                }
                for q in ep.quotes
            ],
        },
        "ack": {"expected": "single line", "format": assignment_example(ep)},
    }
    return _json.dumps(event, separators=(",", ":"))


# ---------------------------------------------------------------------------
# yaml / log-lines (3)
# ---------------------------------------------------------------------------

def _t033(ep):
    """key=value structured log stream."""
    lines = ["# desk log, allocation pending"]
    t = 0
    for r in ep.runs:
        t += 1
        lines.append(
            f'seq={t} record=run id={r.run_id} port="{r.port}" docket={r.docket} '
            f"sailors={r.sailors} days={r.days} difficulty={r.difficulty} "
            f'required_specialty="{run_specialty(r, "none")}" '
            f"contract_payment_coins={r.contract_payment}"
        )
    for c in ep.crews:
        t += 1
        lines.append(
            f'seq={t} record=crew name={c.name} skill={c.skill} '
            f'specialties="{crew_specialties(c, "none")}" runs_this_week={c.runs_this_week} '
            f"runs_this_year={c.runs_this_year} days_since_last_allocation={c.days_since_last} "
            f"deferrals_this_quarter={c.deferrals} registry_rank={c.registry_rank}"
        )
    for q in ep.quotes:
        t += 1
        lines.append(
            f"seq={t} record=quote run={q.run_id} crew={q.crew} "
            f"mobilization={q.mobilization} "
            f"daily_rate_per_required_sailor_per_day={q.daily_rate} "
            f"difficult_run_supplement={q.difficulty_supplement} "
            f"specialty_supplement={q.specialty_supplement}"
        )
    lines.append(
        f'seq={t + 1} record=action need=allocation reply_lines=1 '
        f'reply_format="{assignment_example(ep)}"'
    )
    return "\n".join(lines)


def _t034(ep):
    """YAML with flow-style lists and camelCase keys."""
    out = ["docket:", "  openRuns:"]
    for r in ep.runs:
        out.append(
            f"    - {{id: {r.run_id}, port: {r.port}, docket: {r.docket}, "
            f"sailorsRequired: {r.sailors}, durationDays: {r.days}, "
            f"difficulty: {r.difficulty}, requiredSpecialty: {run_specialty(r, 'null')}, "
            f"contractPaymentCoins: {r.contract_payment}}}"
        )
    out.append("  crews:")
    for c in ep.crews:
        out.append(
            f"    - {{name: {c.name}, skill: {c.skill}, "
            f"specialties: [{crew_specialties(c, '')}], runsThisWeek: {c.runs_this_week}, "
            f"runsThisYear: {c.runs_this_year}, daysSinceLastAllocation: {c.days_since_last}, "
            f"deferralsThisQuarter: {c.deferrals}, registryRank: {c.registry_rank}}}"
        )
    out.append("  quotes:  # dailyRate is per required sailor per day")
    for q in ep.quotes:
        out.append(
            f"    - {{run: {q.run_id}, crew: {q.crew}, mobilization: {q.mobilization}, "
            f"dailyRate: {q.daily_rate}, difficultRunSupplement: {q.difficulty_supplement}, "
            f"specialtySupplement: {q.specialty_supplement}}}"
        )
    out += [
        "reply:",
        "  lines: 1",
        f"  format: '{assignment_example(ep)}'",
    ]
    return "\n".join(out)


def _t035(ep):
    """TOML-ish config sections."""
    out = ["# docket file, allocation not yet entered", "[docket]", f"open_runs = {len(ep.runs)}"]
    for r in ep.runs:
        out += [
            "",
            f"[run.{r.run_id}]",
            f'destination = "{r.port}"',
            f"docket = {r.docket}",
            f"sailors = {r.sailors}",
            f"days = {r.days}",
            f"difficulty = {r.difficulty}",
            f'required_specialty = "{run_specialty(r, "none")}"',
            f"contract_payment_coins = {r.contract_payment}",
        ]
    for c in ep.crews:
        out += [
            "",
            f"[crew.{c.name}]",
            f"skill = {c.skill}",
            f'specialties = "{crew_specialties(c, "none")}"',
            f"runs_this_week = {c.runs_this_week}",
            f"runs_this_year = {c.runs_this_year}",
            f"days_since_last_allocation = {c.days_since_last}",
            f"deferrals_this_quarter = {c.deferrals}",
            f"registry_rank = {c.registry_rank}",
        ]
    out.append("")
    out.append("# quote rates are per required sailor per day")
    for q in ep.quotes:
        out += [
            f"[quote.{q.run_id}.{q.crew}]",
            f"mobilization = {q.mobilization}",
            f"daily_rate = {q.daily_rate}",
            f"difficult_run_supplement = {q.difficulty_supplement}",
            f"specialty_supplement = {q.specialty_supplement}",
        ]
    out += [
        "",
        "[reply]",
        "lines = 1",
        f'format = "{assignment_example(ep)}"',
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# table (3)
# ---------------------------------------------------------------------------

def _t036(ep):
    """ASCII-grid tables with +---+ borders."""

    def grid(headers, rows):
        widths = [
            max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
            for i in range(len(headers))
        ]
        bar = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        out = [bar]
        out.append("| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
        out.append(bar)
        for row in rows:
            out.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |")
        out.append(bar)
        return "\n".join(out)

    runs_tbl = grid(
        ["run", "port", "docket", "sailors", "days", "difficulty", "required specialty", "payment (coins)"],
        [
            [r.run_id, r.port, r.docket, r.sailors, r.days, r.difficulty, run_specialty(r), r.contract_payment]
            for r in ep.runs
        ],
    )
    crews_tbl = grid(
        ["crew", "skill", "specialties", "wk", "yr", "days since", "deferrals", "rank"],
        [
            [c.name, c.skill, crew_specialties(c), c.runs_this_week, c.runs_this_year,
             c.days_since_last, c.deferrals, c.registry_rank]
            for c in ep.crews
        ],
    )
    quotes_tbl = grid(
        ["run", "crew", "mobilization", "daily rate", "difficult-run supp.", "specialty supp."],
        [
            [q.run_id, q.crew, q.mobilization, q.daily_rate, q.difficulty_supplement, q.specialty_supplement]
            for q in ep.quotes
        ],
    )
    return (
        "DOCKET SHEET (printed)\n\n"
        "Open runs:\n" + runs_tbl + "\n\n"
        "Crews (wk = runs this week, yr = runs this year, days since = days since last "
        "allocation, deferrals = deferrals this quarter, rank = registry rank):\n"
        + crews_tbl + "\n\n"
        "Quotes (daily rate is per required sailor per day):\n" + quotes_tbl + "\n\n"
        "Return the allocation as a single line, exactly:\n" + assignment_example(ep)
    )


def _t037(ep):
    """Aligned fixed-width columns with dashed underline."""
    lines = ["ALLOCATION WORKSHEET", ""]
    lines.append("RUN     PORT             DOCKET  SAILORS  DAYS  DIFF  REQ.SPECIALTY    PAYMENT")
    lines.append("-" * 82)
    for r in ep.runs:
        lines.append(
            f"{r.run_id:<7} {r.port:<16} {r.docket:<7} {r.sailors:<8} {r.days:<5} "
            f"{r.difficulty:<5} {run_specialty(r):<16} {r.contract_payment}"
        )
    lines += ["", "CREW     SKILL  WEEK  YEAR  DAYS-SINCE  DEFERRALS  RANK  SPECIALTIES"]
    lines.append("-" * 78)
    for c in ep.crews:
        lines.append(
            f"{c.name:<8} {c.skill:<6} {c.runs_this_week:<5} {c.runs_this_year:<5} "
            f"{c.days_since_last:<11} {c.deferrals:<10} {c.registry_rank:<5} {crew_specialties(c)}"
        )
    lines += ["", "QUOTE    RUN     MOBILIZE  RATE  DIFF-SUPP  SPEC-SUPP"]
    lines.append("-" * 55)
    for q in ep.quotes:
        lines.append(
            f"{q.crew:<8} {q.run_id:<7} {q.mobilization:<9} {q.daily_rate:<5} "
            f"{q.difficulty_supplement:<10} {q.specialty_supplement}"
        )
    lines += [
        "",
        "Column notes: WEEK/YEAR are runs this week / runs this year; DAYS-SINCE is days",
        "since last allocation; DEFERRALS is deferrals this quarter; RANK is registry rank;",
        "RATE is the daily rate per required sailor per day; MOBILIZE is the mobilization",
        "fee; DIFF-SUPP / SPEC-SUPP are the difficult-run and specialty supplements.",
        "",
        "Fill in the last line of the worksheet — one line, this exact format:",
        assignment_example(ep),
    ]
    return "\n".join(lines)


def _t038(ep):
    """Wide markdown table: one row per crew, quote cells packed, with legend."""
    out = [
        "### Docket at a glance",
        "",
        "Runs:",
        "",
        "| run | port | docket | sailors | days | difficulty | required specialty | contract payment (coins) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ep.runs:
        out.append(
            f"| {r.run_id} | {r.port} | {r.docket} | {r.sailors} | {r.days} | "
            f"{r.difficulty} | {run_specialty(r)} | {r.contract_payment} |"
        )
    quote_heads = " | ".join(f"quote {r.run_id}" for r in ep.runs)
    out += [
        "",
        "Crews (quote cells read mobilization / daily rate per required sailor per day / "
        "difficult-run supplement / specialty supplement):",
        "",
        f"| crew | skill | specialties | runs this week | runs this year | days since last allocation | deferrals this quarter | registry rank | {quote_heads} |",
        "|---" * (8 + len(ep.runs)) + "|",
    ]
    for c in ep.crews:
        cells = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            cells.append(
                f"{q.mobilization} / {q.daily_rate} / {q.difficulty_supplement} / {q.specialty_supplement}"
            )
        out.append(
            f"| {c.name} | {c.skill} | {crew_specialties(c)} | {c.runs_this_week} | "
            f"{c.runs_this_year} | {c.days_since_last} | {c.deferrals} | {c.registry_rank} | "
            + " | ".join(cells) + " |"
        )
    out += [
        "",
        "**Answer with one line only:** " + assignment_example(ep),
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# csv / tsv (2)
# ---------------------------------------------------------------------------

def _t039(ep):
    """TSV blocks with ### markers."""
    out = ["### docket.tsv export — decision pending", "", "### runs"]
    out.append("run_id\tport\tdocket\tsailors\tdays\tdifficulty\trequired_specialty\tcontract_payment_coins")
    for r in ep.runs:
        out.append(
            f"{r.run_id}\t{r.port}\t{r.docket}\t{r.sailors}\t{r.days}\t{r.difficulty}\t"
            f"{run_specialty(r)}\t{r.contract_payment}"
        )
    out += ["", "### crews"]
    out.append(
        "name\tskill\tspecialties\truns_this_week\truns_this_year\tdays_since_last_allocation\tdeferrals_this_quarter\tregistry_rank"
    )
    for c in ep.crews:
        out.append(
            f"{c.name}\t{c.skill}\t{crew_specialties(c)}\t{c.runs_this_week}\t"
            f"{c.runs_this_year}\t{c.days_since_last}\t{c.deferrals}\t{c.registry_rank}"
        )
    out += ["", "### quotes (daily_rate = per required sailor per day)"]
    out.append("run_id\tcrew\tmobilization\tdaily_rate\tdifficult_run_supplement\tspecialty_supplement")
    for q in ep.quotes:
        out.append(
            f"{q.run_id}\t{q.crew}\t{q.mobilization}\t{q.daily_rate}\t"
            f"{q.difficulty_supplement}\t{q.specialty_supplement}"
        )
    out += ["", "### required reply: one line, exact format below", assignment_example(ep)]
    return "\n".join(out)


def _t040(ep):
    """Single mixed-record CSV with a record_type column."""
    out = [
        "record_type,id,field_1,field_2,field_3,field_4,field_5,field_6,field_7",
        "# run rows: id=run_id f1=port f2=docket f3=sailors f4=days f5=difficulty "
        "f6=required_specialty f7=contract_payment_coins",
        "# crew rows: id=name f1=skill f2=specialties f3=runs_this_week f4=runs_this_year "
        "f5=days_since_last_allocation f6=deferrals_this_quarter f7=registry_rank",
        "# quote rows: id=run_id f1=crew f2=mobilization f3=daily_rate_per_required_sailor_per_day "
        "f4=difficult_run_supplement f5=specialty_supplement f6=- f7=-",
    ]
    for r in ep.runs:
        out.append(
            f"run,{r.run_id},{r.port},{r.docket},{r.sailors},{r.days},{r.difficulty},"
            f"{run_specialty(r).replace(', ', '|')},{r.contract_payment}"
        )
    for c in ep.crews:
        out.append(
            f"crew,{c.name},{c.skill},{crew_specialties(c).replace(', ', '|')},"
            f"{c.runs_this_week},{c.runs_this_year},{c.days_since_last},{c.deferrals},"
            f"{c.registry_rank}"
        )
    for q in ep.quotes:
        out.append(
            f"quote,{q.run_id},{q.crew},{q.mobilization},{q.daily_rate},"
            f"{q.difficulty_supplement},{q.specialty_supplement},-,-"
        )
    out += [
        "",
        "allocation still blank. append it as plain text, one line, exactly:",
        assignment_example(ep),
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# prose (4)
# ---------------------------------------------------------------------------

def _t041(ep):
    """Harbour-master's journal entry."""
    rbits = []
    for r in ep.runs:
        rbits.append(
            f"{r.run_id}, for {r.port} under docket {r.docket}, wanting {r.sailors} "
            f"sailors across {plural(r.days, 'day')} at difficulty {r.difficulty}, "
            f"required specialty {run_specialty(r, 'none')}, contract payment "
            f"{r.contract_payment} coins"
        )
    paras = [
        "Morning tide. Fog off the mole until the second bell. The docket stands open "
        + ("with one run: " if len(ep.runs) == 1 else f"with {len(ep.runs)} runs: ")
        + "; ".join(rbits)
        + "."
    ]
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"on {r.run_id} mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}"
            )
        paras.append(
            f"{c.name} came to the rail — skill {c.skill}, specialties "
            f"{crew_specialties(c, 'none')}; the book gives {c.runs_this_week} runs this "
            f"week, {c.runs_this_year} runs this year, {c.days_since_last} days since "
            f"last allocation, {c.deferrals} deferrals this quarter, registry rank "
            f"{c.registry_rank} — and quoted " + "; ".join(qbits) + "."
        )
    paras.append(
        "The entry must be made before I lock the office. Set down the allocation as a "
        "single line, no remarks, thus: " + assignment_example(ep)
    )
    return "\n\n".join(paras)


def _t042(ep):
    """Registry clerk's account, bureaucratic first-person past tense."""
    rbits = []
    for r in ep.runs:
        rbits.append(
            f"run {r.run_id} to {r.port} (docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r, 'none')}; contract payment {r.contract_payment} coins)"
        )
    paras = [
        "I recorded the docket at the counter this afternoon. It comprised "
        + ("a single run: " if len(ep.runs) == 1 else "the following runs: ")
        + "; ".join(rbits)
        + "."
    ]
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"for {r.run_id}: mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}"
            )
        paras.append(
            f"I then took the particulars of crew {c.name}: skill {c.skill}; specialties "
            f"{crew_specialties(c, 'none')}; runs this week {c.runs_this_week}; runs this "
            f"year {c.runs_this_year}; days since last allocation {c.days_since_last}; "
            f"deferrals this quarter {c.deferrals}; registry rank {c.registry_rank}. "
            f"Their quoted terms were " + "; ".join(qbits) + "."
        )
    paras.append(
        "The determination column remains blank. I am to copy in exactly one line, "
        "in this form and no other: " + assignment_example(ep)
    )
    return "\n\n".join(paras)


def _t043(ep):
    """Almanac / gazetteer entry, dry third person."""
    rbits = []
    for r in ep.runs:
        rbits.append(
            f"{r.run_id} ({r.port}; docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r, 'none')}; contract payment {r.contract_payment} coins)"
        )
    paras = [
        "QALVORI HARBOUR ALMANAC — DOCKETS OF THE DAY.\n\nOpen this day: "
        + "; ".join(rbits)
        + "."
    ]
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"{r.run_id} — mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}"
            )
        paras.append(
            f"Of the crews: {c.name} is listed at skill {c.skill}, specialties "
            f"{crew_specialties(c, 'none')}, with {c.runs_this_week} runs this week, "
            f"{c.runs_this_year} runs this year, {c.days_since_last} days since last "
            f"allocation, {c.deferrals} deferrals this quarter, and registry rank "
            f"{c.registry_rank}; quoting " + "; ".join(qbits) + "."
        )
    paras.append(
        "The almanac prints allocations as received, one line, in the standard form: "
        + assignment_example(ep)
        + ". Submit that line alone."
    )
    return "\n\n".join(paras)


def _t044(ep):
    """Present-tense chronicle of the day at the harbour office."""
    rbits = []
    for r in ep.runs:
        rbits.append(
            f"{r.run_id} bound for {r.port} — docket {r.docket}, {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r, 'none')}, contract payment {r.contract_payment} coins"
        )
    paras = [
        "The bell rings the half hour and the docket is still unallocated. "
        + ("One run waits: " if len(ep.runs) == 1 else "The runs wait: ")
        + "; ".join(rbits)
        + "."
    ]
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"{r.run_id} at mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}"
            )
        paras.append(
            f"By the window stands {c.name}'s mate with their slate: skill {c.skill}, "
            f"specialties {crew_specialties(c, 'none')}, {c.runs_this_week} runs this "
            f"week, {c.runs_this_year} runs this year, {c.days_since_last} days since "
            f"last allocation, {c.deferrals} deferrals this quarter, registry rank "
            f"{c.registry_rank}; the slate quotes " + "; ".join(qbits) + "."
        )
    paras.append(
        "The ledger lies open at the day's page. It takes one line only — no account "
        "of the choosing — written exactly so: " + assignment_example(ep)
    )
    return "\n\n".join(paras)


# ---------------------------------------------------------------------------
# dialogue / transcript (3)
# ---------------------------------------------------------------------------

def _t045(ep):
    """Dockside conversation, two named speakers, casual."""
    lines = ["OLD PEV: Docket's not gone out yet. What's on it?"]
    for r in ep.runs:
        lines.append(
            f"MARIS: {r.run_id}, to {r.port}. Docket {r.docket}, {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}. Required specialty "
            f"{run_specialty(r, 'none')}. Contract payment {r.contract_payment} coins."
        )
    lines.append("OLD PEV: And who's asking for work?")
    for c in ep.crews:
        lines.append(
            f"MARIS: {c.name}. Skill {c.skill}, specialties {crew_specialties(c, 'none')}. "
            f"Runs this week {c.runs_this_week}, runs this year {c.runs_this_year}, "
            f"days since last allocation {c.days_since_last}, deferrals this quarter "
            f"{c.deferrals}, registry rank {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"MARIS: They'd take {r.run_id} at mobilization {q.mobilization}, daily "
                f"rate {q.daily_rate} per required sailor per day, difficult-run "
                f"supplement {q.difficulty_supplement}, specialty supplement "
                f"{q.specialty_supplement}."
            )
    lines.append(
        "OLD PEV: Then it wants deciding, not discussing. Chalk the one line and be "
        "done — exactly like this: " + assignment_example(ep)
    )
    return "\n".join(lines)


def _t046(ep):
    """Watch handover, outgoing to incoming officer."""
    lines = [
        "WATCH HANDOVER — DISPATCH OFFICE",
        "",
        "OUTGOING: One item left for you: the docket. I'll read it across.",
    ]
    for r in ep.runs:
        lines.append(
            f"OUTGOING: Run {r.run_id} — destination {r.port}, docket {r.docket}, "
            f"{r.sailors} sailors, {plural(r.days, 'day')}, difficulty {r.difficulty}, "
            f"required specialty {run_specialty(r, 'none')}, contract payment "
            f"{r.contract_payment} coins."
        )
    lines.append("INCOMING: Copied. Crews?")
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"{r.run_id} at mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}"
            )
        lines.append(
            f"OUTGOING: {c.name} — skill {c.skill}, specialties "
            f"{crew_specialties(c, 'none')}, runs this week {c.runs_this_week}, runs this "
            f"year {c.runs_this_year}, days since last allocation {c.days_since_last}, "
            f"deferrals this quarter {c.deferrals}, registry rank {c.registry_rank}. "
            f"Quoting " + "; ".join(qbits) + "."
        )
    lines += [
        "INCOMING: Understood. I'll enter it before the watch turns.",
        "OUTGOING: Single line in the book, mind — no workings. Exactly: "
        + assignment_example(ep),
    ]
    return "\n".join(lines)


def _t047(ep):
    """Clerk-and-master Q&A exchange."""
    lines = ["MASTER: Read me the open runs, one at a time."]
    for r in ep.runs:
        lines += [
            f"CLERK: {r.run_id}, sir. Destination {r.port}, docket {r.docket}.",
            f"MASTER: Complement and terms?",
            f"CLERK: {r.sailors} sailors for {plural(r.days, 'day')}, difficulty "
            f"{r.difficulty}, required specialty {run_specialty(r, 'none')}, contract "
            f"payment {r.contract_payment} coins.",
        ]
    lines.append("MASTER: Now the crews, with everything the register holds.")
    for c in ep.crews:
        lines.append(
            f"CLERK: {c.name}: skill {c.skill}; specialties {crew_specialties(c, 'none')}; "
            f"runs this week {c.runs_this_week}; runs this year {c.runs_this_year}; days "
            f"since last allocation {c.days_since_last}; deferrals this quarter "
            f"{c.deferrals}; registry rank {c.registry_rank}."
        )
        lines.append("MASTER: Their quotes?")
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"{r.run_id}: mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}"
            )
        lines.append("CLERK: " + "; ".join(qbits) + ".")
    lines.append(
        "MASTER: Very well. Write the allocation — the line only, no reasoning aloud — "
        "in the standard form: " + assignment_example(ep)
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# email / letter (3)
# ---------------------------------------------------------------------------

def _t048(ep):
    """Modern terse internal email with data appendix."""
    out = [
        "From: night-desk@qalvori-harbour",
        "To: allocations",
        "Subject: need the docket line by 06:00",
        "",
        "Data below. Reply = the line only.",
        "",
        "RUNS",
    ]
    for r in ep.runs:
        out.append(
            f"* {r.run_id} | {r.port} | docket {r.docket} | {r.sailors} sailors | "
            f"{plural(r.days, 'day')} | difficulty {r.difficulty} | required specialty "
            f"{run_specialty(r)} | contract payment {r.contract_payment} coins"
        )
    out.append("")
    out.append("CREWS")
    for c in ep.crews:
        out.append(
            f"* {c.name} | skill {c.skill} | specialties {crew_specialties(c)} | "
            f"runs this week {c.runs_this_week} | runs this year {c.runs_this_year} | "
            f"days since last allocation {c.days_since_last} | deferrals this quarter "
            f"{c.deferrals} | registry rank {c.registry_rank}"
        )
    out.append("")
    out.append("QUOTES (daily rate = per required sailor per day)")
    for q in ep.quotes:
        out.append(
            f"* {q.run_id} x {q.crew} | mobilization {q.mobilization} | daily rate "
            f"{q.daily_rate} | difficult-run supplement {q.difficulty_supplement} | "
            f"specialty supplement {q.specialty_supplement}"
        )
    out += ["", "Exact reply format, nothing before or after:", assignment_example(ep)]
    return "\n".join(out)


def _t049(ep):
    """Old-fashioned letter."""
    rbits = []
    for r in ep.runs:
        rbits.append(
            f"{r.run_id}, for {r.port} under docket {r.docket}, requiring {r.sailors} "
            f"sailors for {plural(r.days, 'day')}, of difficulty {r.difficulty}, "
            f"required specialty {run_specialty(r, 'none')}, with contract payment of "
            f"{r.contract_payment} coins"
        )
    cbits = []
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(
                f"for {r.run_id}, mobilization {q.mobilization}, a daily rate of "
                f"{q.daily_rate} per required sailor per day, a difficult-run supplement "
                f"of {q.difficulty_supplement}, and a specialty supplement of "
                f"{q.specialty_supplement}"
            )
        cbits.append(
            f"{c.name}, of skill {c.skill}, holding specialties "
            f"{crew_specialties(c, 'none')}, having {c.runs_this_week} runs this week and "
            f"{c.runs_this_year} runs this year, {c.days_since_last} days since last "
            f"allocation, {c.deferrals} deferrals this quarter, and registry rank "
            f"{c.registry_rank}, who quotes " + "; ".join(qbits)
        )
    return (
        "To the Registrar of the Harbour, Qalvori.\n\n"
        "Dear Registrar,\n\n"
        "I write to lay before you the docket now wanting allocation. The runs are: "
        + "; and ".join(rbits)
        + ".\n\nThe crews presenting themselves are as follows: "
        + "; and ".join(cbits)
        + ".\n\nI beg you return only the allocation itself, set down in a single line "
        "after the customary fashion, viz.\n\n"
        + assignment_example(ep)
        + "\n\nYour obedient servant,\nThe Clerk of the Docket"
    )


def _t050(ep):
    """Reply-chain email: quoted data below, ask on top."""
    quoted = []
    quoted.append("> RUNS OPEN")
    for r in ep.runs:
        quoted.append(
            f"> - {r.run_id}: destination {r.port}; docket {r.docket}; {r.sailors} "
            f"sailors; {plural(r.days, 'day')}; difficulty {r.difficulty}; required "
            f"specialty {run_specialty(r)}; contract payment {r.contract_payment} coins."
        )
    quoted.append("> CREWS AND QUOTES")
    for c in ep.crews:
        quoted.append(
            f"> - {c.name}: skill {c.skill}; specialties {crew_specialties(c)}; runs "
            f"this week {c.runs_this_week}; runs this year {c.runs_this_year}; days "
            f"since last allocation {c.days_since_last}; deferrals this quarter "
            f"{c.deferrals}; registry rank {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            quoted.append(
                f">   quote {r.run_id}: mobilization {q.mobilization}; daily rate "
                f"{q.daily_rate} per required sailor per day; difficult-run supplement "
                f"{q.difficulty_supplement}; specialty supplement {q.specialty_supplement}."
            )
    return (
        "From: allocations-desk\n"
        "To: you\n"
        "Subject: RE: RE: docket — still waiting on the line\n\n"
        "Chasing this. The data hasn't changed since the message below. Send the "
        "allocation line by return — just the line, formatted exactly:\n"
        + assignment_example(ep)
        + "\n\n---- earlier message ----\n"
        + "\n".join(quoted)
    )


# ---------------------------------------------------------------------------
# telegraph / terse (2)
# ---------------------------------------------------------------------------

def _t051(ep):
    """Signal-lamp message, caps labels, = separators, STOP line ends."""
    out = ["SIGNAL FROM HARBOUR OFFICE BEGINS"]
    for r in ep.runs:
        out.append(
            f"RUN {r.run_id} = PORT {r.port} = DOCKET {r.docket} = SAILORS {r.sailors} = "
            f"DAYS {r.days} = DIFFICULTY {r.difficulty} = REQUIRED SPECIALTY "
            f"{run_specialty(r)} = CONTRACT PAYMENT {r.contract_payment} COINS STOP"
        )
    for c in ep.crews:
        out.append(
            f"CREW {c.name} = SKILL {c.skill} = SPECIALTIES {crew_specialties(c)} = "
            f"RUNS THIS WEEK {c.runs_this_week} = RUNS THIS YEAR {c.runs_this_year} = "
            f"DAYS SINCE LAST ALLOCATION {c.days_since_last} = DEFERRALS THIS QUARTER "
            f"{c.deferrals} = REGISTRY RANK {c.registry_rank} STOP"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"QUOTE {c.name} ON {r.run_id} = MOBILIZATION {q.mobilization} = DAILY "
                f"RATE {q.daily_rate} PER REQUIRED SAILOR PER DAY = DIFFICULT-RUN "
                f"SUPPLEMENT {q.difficulty_supplement} = SPECIALTY SUPPLEMENT "
                f"{q.specialty_supplement} STOP"
            )
    out.append(
        "SEND ALLOCATION AS ONE LINE ONLY IN EXACT FORM " + assignment_example(ep) + " STOP"
    )
    out.append("SIGNAL ENDS")
    return "\n".join(out)


def _t052(ep):
    """Pocket crib card, ultra-compact with legend."""
    out = [
        "POCKET CARD — DOCKET PENDING",
        "legend: s=sailors d=days x=difficulty sp=required specialty pay=contract payment coins",
        "legend: k=skill w=runs this week y=runs this year a=days since last allocation "
        "f=deferrals this quarter g=registry rank",
        "legend: m=mobilization r=daily rate per required sailor per day "
        "d+=difficult-run supplement s+=specialty supplement",
    ]
    for r in ep.runs:
        out.append(
            f"{r.run_id} {r.port} #{r.docket} s{r.sailors} d{r.days} x{r.difficulty} "
            f"sp:{run_specialty(r)} pay{r.contract_payment}"
        )
    for c in ep.crews:
        qbits = []
        for r in ep.runs:
            q = quote_for(ep, r, c)
            qbits.append(f"{r.run_id} m{q.mobilization} r{q.daily_rate} d+{q.difficulty_supplement} s+{q.specialty_supplement}")
        out.append(
            f"{c.name} k{c.skill} w{c.runs_this_week} y{c.runs_this_year} "
            f"a{c.days_since_last} f{c.deferrals} g{c.registry_rank} "
            f"sp:{crew_specialties(c)} || " + " ; ".join(qbits)
        )
    out.append("write one line only -> " + assignment_example(ep))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# form / ledger (3)
# ---------------------------------------------------------------------------

def _t053(ep):
    """Numbered official form."""
    out = ["FORM D-7 — DOCKET ALLOCATION RETURN", ""]
    n = 0
    for r in ep.runs:
        n += 1
        out.append(
            f"({n}) Run under consideration: {r.run_id}. Destination: {r.port}. Docket "
            f"number: {r.docket}. Sailors: {r.sailors}. Days: {r.days}. Difficulty: "
            f"{r.difficulty}. Required specialty: {run_specialty(r)}. Contract payment "
            f"(coins): {r.contract_payment}."
        )
    for c in ep.crews:
        n += 1
        out.append(
            f"({n}) Crew entered: {c.name}. Skill: {c.skill}. Specialties: "
            f"{crew_specialties(c)}. Runs this week: {c.runs_this_week}. Runs this year: "
            f"{c.runs_this_year}. Days since last allocation: {c.days_since_last}. "
            f"Deferrals this quarter: {c.deferrals}. Registry rank: {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            n += 1
            out.append(
                f"({n}) Quote of {c.name} for {r.run_id}: mobilization {q.mobilization}; "
                f"daily rate {q.daily_rate} per required sailor per day; difficult-run "
                f"supplement {q.difficulty_supplement}; specialty supplement "
                f"{q.specialty_supplement}."
            )
    n += 1
    out += [
        "",
        f"({n}) ALLOCATION (to be completed in a single line, without annotation, in "
        f"exactly this form): {assignment_example(ep)}",
    ]
    return "\n".join(out)


def _t054(ep):
    """Checklist style with checkboxes."""
    out = ["DESK CHECKLIST — CLOSE OF DAY", ""]
    out.append("[x] runs posted:")
    for r in ep.runs:
        out.append(
            f"    {r.run_id} — {r.port}, docket {r.docket}, {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, contract payment {r.contract_payment} coins"
        )
    out.append("[x] crews registered:")
    for c in ep.crews:
        out.append(
            f"    {c.name} — skill {c.skill}, specialties {crew_specialties(c)}, "
            f"runs this week {c.runs_this_week}, runs this year {c.runs_this_year}, "
            f"days since last allocation {c.days_since_last}, deferrals this quarter "
            f"{c.deferrals}, registry rank {c.registry_rank}"
        )
    out.append("[x] quotes collected (daily rate per required sailor per day):")
    for q in ep.quotes:
        out.append(
            f"    {q.run_id} / {q.crew} — mobilization {q.mobilization}, daily rate "
            f"{q.daily_rate}, difficult-run supplement {q.difficulty_supplement}, "
            f"specialty supplement {q.specialty_supplement}"
        )
    out += [
        "[ ] allocation entered — outstanding. Complete it now: one line, no notes, "
        "exactly this format:",
        "    " + assignment_example(ep),
    ]
    return "\n".join(out)


def _t055(ep):
    """Intake form, FIELD — ENTRY pairs with dotted rules."""
    out = ["HARBOUR OFFICE INTAKE FORM", "=" * 34, ""]
    for i, r in enumerate(ep.runs, 1):
        out += [
            f"RUN ENTRY No. {i}",
            f"  Run id .......... {r.run_id}",
            f"  Destination ..... {r.port}",
            f"  Docket .......... {r.docket}",
            f"  Sailors ......... {r.sailors}",
            f"  Days ............ {r.days}",
            f"  Difficulty ...... {r.difficulty}",
            f"  Req. specialty .. {run_specialty(r)}",
            f"  Payment (coins) . {r.contract_payment}",
            "",
        ]
    for i, c in enumerate(ep.crews, 1):
        out += [
            f"CREW ENTRY No. {i}",
            f"  Name ............ {c.name}",
            f"  Skill ........... {c.skill}",
            f"  Specialties ..... {crew_specialties(c)}",
            f"  Runs this week .. {c.runs_this_week}",
            f"  Runs this year .. {c.runs_this_year}",
            f"  Days since last allocation .. {c.days_since_last}",
            f"  Deferrals this quarter ...... {c.deferrals}",
            f"  Registry rank ... {c.registry_rank}",
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"  Quote {r.run_id} ....... mobilization {q.mobilization} / daily rate "
                f"{q.daily_rate} per required sailor per day / difficult-run supplement "
                f"{q.difficulty_supplement} / specialty supplement {q.specialty_supplement}"
            )
        out.append("")
    out += [
        "DETERMINATION (one line, exact form, nothing further):",
        assignment_example(ep),
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# briefing / second-person (2)
# ---------------------------------------------------------------------------

def _t056(ep):
    """Morning briefing bullets to the allocator."""
    out = [
        "MORNING BRIEF — ALLOCATION DESK",
        "",
        f"You have {plural(len(ep.runs), 'run')} to place and {len(ep.crews)} crews on the register.",
        "",
        "The runs:",
    ]
    for r in ep.runs:
        out.append(
            f"  - {r.run_id}, {r.port}. Docket {r.docket}. {r.sailors} sailors for "
            f"{plural(r.days, 'day')} at difficulty {r.difficulty}. Required specialty: "
            f"{run_specialty(r)}. Contract payment {r.contract_payment} coins."
        )
    out.append("")
    out.append("The register:")
    for c in ep.crews:
        out.append(
            f"  - {c.name}: skill {c.skill}; specialties {crew_specialties(c)}; runs "
            f"this week {c.runs_this_week}; runs this year {c.runs_this_year}; days "
            f"since last allocation {c.days_since_last}; deferrals this quarter "
            f"{c.deferrals}; registry rank {c.registry_rank}."
        )
    out.append("")
    out.append("The quote sheet (daily rate is per required sailor per day):")
    for q in ep.quotes:
        out.append(
            f"  - {q.crew} on {q.run_id}: mobilization {q.mobilization}, daily rate "
            f"{q.daily_rate}, difficult-run supplement {q.difficulty_supplement}, "
            f"specialty supplement {q.specialty_supplement}."
        )
    out += [
        "",
        "Place them before the flag goes up. Your return is one line, exactly:",
        assignment_example(ep),
    ]
    return "\n".join(out)


def _t057(ep):
    """Quartermaster's imperative handover note."""
    runs = []
    for r in ep.runs:
        runs.append(
            f"— {r.run_id} for {r.port}: docket {r.docket}; take {r.sailors} sailors "
            f"for {plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins."
        )
    crews = []
    for c in ep.crews:
        lines = [
            f"— {c.name}: skill {c.skill}; specialties {crew_specialties(c)}. Book says "
            f"{c.runs_this_week} runs this week, {c.runs_this_year} runs this year, "
            f"{c.days_since_last} days since last allocation, {c.deferrals} deferrals "
            f"this quarter, registry rank {c.registry_rank}."
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            lines.append(
                f"  Their {r.run_id} terms: mobilization {q.mobilization}; daily rate "
                f"{q.daily_rate} per required sailor per day; difficult-run supplement "
                f"{q.difficulty_supplement}; specialty supplement {q.specialty_supplement}."
            )
        crews.append("\n".join(lines))
    return (
        "NOTE LEFT ON THE DESK, WEIGHTED WITH A SHACKLE PIN\n\n"
        "Gone to the mole. Docket is yours. Read, decide, enter, lock up.\n\n"
        "Runs:\n" + "\n".join(runs) + "\n\n"
        "Crews:\n" + "\n".join(crews) + "\n\n"
        "Enter it in the book as one line, no working shown, exactly:\n"
        + assignment_example(ep)
    )


ENTRIES: list[Template] = [
    Template("T015", "bullets", "neutral", "numbered list, crews first, em-dash fields", _t015),
    Template("T016", "bullets", "neutral", "arrow bullets, quotes as separate section by run", _t016),
    Template("T017", "bullets", "casual", "chalked tide-board, caps headers", _t017),
    Template("T018", "bullets", "neutral", "compact dashes, quotes inline in parentheses", _t018),
    Template("T019", "bullets", "casual", "plus bullets, middle dots, crews first", _t019),
    Template("T020", "memo", "formal", "registry circular to wharf officers", _t020),
    Template("T021", "memo", "formal", "wharf gate notice, ALL CAPS", _t021),
    Template("T022", "memo", "formal", "standing-orders addendum, roman numerals", _t022),
    Template("T023", "memo", "formal", "duty officer's minute sheet", _t023),
    Template("T024", "chat", "casual", "anxious first-week clerk", _t024),
    Template("T025", "chat", "casual", "gruff veteran, clipped sentences", _t025),
    Template("T026", "chat", "casual", "overworked, run-on plea", _t026),
    Template("T027", "chat", "formal", "polite evening request", _t027),
    Template("T028", "chat", "casual", "night-shift texting, lowercase", _t028),
    Template("T029", "json", "machine", "JSON-RPC envelope, camelCase", _t029),
    Template("T030", "toolcall", "machine", "tool invocation with kebab-case args", _t030),
    Template("T031", "json", "machine", "HTTP POST body, snake_case", _t031),
    Template("T032", "json", "machine", "event payload with ack contract", _t032),
    Template("T033", "log", "machine", "key=value log stream", _t033),
    Template("T034", "yaml", "machine", "flow-style YAML, camelCase", _t034),
    Template("T035", "yaml", "machine", "TOML-ish sections", _t035),
    Template("T036", "table", "neutral", "ASCII-grid tables with borders", _t036),
    Template("T037", "table", "neutral", "fixed-width worksheet columns", _t037),
    Template("T038", "table", "neutral", "wide markdown table, packed quote cells", _t038),
    Template("T039", "csv", "machine", "TSV blocks with ### markers", _t039),
    Template("T040", "csv", "machine", "single mixed-record CSV", _t040),
    Template("T041", "prose", "formal", "harbour-master's journal", _t041),
    Template("T042", "prose", "formal", "registry clerk's account, past tense", _t042),
    Template("T043", "prose", "formal", "almanac entry, dry third person", _t043),
    Template("T044", "prose", "neutral", "present-tense chronicle", _t044),
    Template("T045", "dialogue", "casual", "dockside conversation, two speakers", _t045),
    Template("T046", "dialogue", "neutral", "watch handover exchange", _t046),
    Template("T047", "dialogue", "formal", "clerk-and-master Q&A", _t047),
    Template("T048", "email", "neutral", "terse internal email with appendix", _t048),
    Template("T049", "email", "formal", "old-fashioned letter to the Registrar", _t049),
    Template("T050", "email", "neutral", "reply-chain chaser with quoted data", _t050),
    Template("T051", "telegraph", "machine", "signal-lamp message with STOP", _t051),
    Template("T052", "telegraph", "machine", "pocket crib card with legend", _t052),
    Template("T053", "form", "formal", "numbered official form D-7", _t053),
    Template("T054", "form", "neutral", "checklist with checkboxes", _t054),
    Template("T055", "form", "formal", "intake form with dotted rules", _t055),
    Template("T056", "briefing", "neutral", "morning brief with quote sheet", _t056),
    Template("T057", "briefing", "casual", "quartermaster's imperative note", _t057),
]
