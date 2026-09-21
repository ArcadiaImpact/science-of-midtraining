"""Batch 3 of the dispatch presentation templates: T058-T100 (43 entries).

Same contract as templates.py: complete field coverage, neutral/prefix-free,
verbatim ``assignment_example`` response contract, deterministic, <= 4300
chars on the worst-case episodes. Voices deliberately differ from batch 2's
sub-styles within each family.
"""

from __future__ import annotations

import json as _json
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

def _t058(ep):
    """Numbered list, 1) 2) 3), runs then crews, no trailing periods."""
    out = ["Open runs on the docket:"]
    for i, r in enumerate(ep.runs, 1):
        out.append(
            f"{i}) {r.run_id} to {r.port} — docket {r.docket}, {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, contract payment {r.contract_payment} coins"
        )
    out.append("")
    out.append("Crews on hand:")
    n = 0
    for c in ep.crews:
        n += 1
        out.append(
            f"{n}) {c.name} — skill {c.skill}, specialties {crew_specialties(c)}, "
            f"runs this week {c.runs_this_week}, runs this year {c.runs_this_year}, "
            f"days since last allocation {c.days_since_last}, deferrals this quarter "
            f"{c.deferrals}, registry rank {c.registry_rank}"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"   quote {r.run_id}: mobilization {q.mobilization}, daily rate "
                f"{q.daily_rate} per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}"
            )
    out.append("")
    out.append(
        "Enter the allocation as a single line, nothing else, in exactly this shape: "
        + assignment_example(ep)
    )
    return "\n".join(out)


def _t059(ep):
    """Dash bullets, crews-first section order, quotes as their own section."""
    out = ["THE CREWS"]
    for c in ep.crews:
        out.append(
            f"- {c.name}: skill {c.skill}; specialties {crew_specialties(c)}; "
            f"runs this week {c.runs_this_week}; runs this year {c.runs_this_year}; "
            f"days since last allocation {c.days_since_last}; deferrals this quarter "
            f"{c.deferrals}; registry rank {c.registry_rank}"
        )
    out.append("")
    out.append("THE RUNS")
    for r in ep.runs:
        out.append(
            f"- {r.run_id}: destination {r.port}; docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins"
        )
    out.append("")
    out.append("THE QUOTES (daily rate is per required sailor per day)")
    for q in ep.quotes:
        out.append(
            f"- {q.crew} on {q.run_id}: mobilization {q.mobilization}; daily rate "
            f"{q.daily_rate}; difficult-run supplement {q.difficulty_supplement}; "
            f"specialty supplement {q.specialty_supplement}"
        )
    out.append("")
    out.append(
        "Decide who takes what. One line back, exactly: " + assignment_example(ep)
    )
    return "\n".join(out)


def _t060(ep):
    """Indented outline: A./B./C. sections with i. ii. iii. items."""
    def roman(n):
        vals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
        return vals[n - 1]

    out = ["A. Runs standing open"]
    for i, r in enumerate(ep.runs, 1):
        out.append(
            f"   {roman(i)}. {r.run_id}, destination {r.port}, docket {r.docket}; "
            f"{r.sailors} sailors for {plural(r.days, 'day')}; difficulty {r.difficulty}; "
            f"required specialty {run_specialty(r)}; contract payment {r.contract_payment} coins."
        )
    out.append("B. Crews answering the call")
    for i, c in enumerate(ep.crews, 1):
        out.append(
            f"   {roman(i)}. {c.name}: skill {c.skill}; specialties {crew_specialties(c)}; "
            f"runs this week {c.runs_this_week}; runs this year {c.runs_this_year}; days "
            f"since last allocation {c.days_since_last}; deferrals this quarter {c.deferrals}; "
            f"registry rank {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"      - {r.run_id}: mobilization {q.mobilization}; daily rate {q.daily_rate} "
                f"per required sailor per day; difficult-run supplement {q.difficulty_supplement}; "
                f"specialty supplement {q.specialty_supplement}."
            )
    out.append("C. Determination")
    out.append(
        "   Return the allocation alone, one line, in exactly this form: "
        + assignment_example(ep)
    )
    return "\n".join(out)


def _t061(ep):
    """Checklist boxes: items to weigh, then the entry line to produce."""
    out = ["DOCKET CHECKLIST — allocation still unchecked", ""]
    for r in ep.runs:
        out.append(
            f"[ ] {r.run_id} unassigned — {r.port}, docket {r.docket}, {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, contract payment {r.contract_payment} coins"
        )
    out.append("")
    out.append("Crews available to tick against:")
    for c in ep.crews:
        out.append(
            f"[x] {c.name} present — skill {c.skill}, specialties {crew_specialties(c)}, "
            f"runs this week {c.runs_this_week}, runs this year {c.runs_this_year}, "
            f"days since last allocation {c.days_since_last}, deferrals this quarter "
            f"{c.deferrals}, registry rank {c.registry_rank}"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"    [x] quote filed for {r.run_id}: mobilization {q.mobilization}, daily rate "
                f"{q.daily_rate} per required sailor per day, difficult-run supplement "
                f"{q.difficulty_supplement}, specialty supplement {q.specialty_supplement}"
            )
    out.append("")
    out.append(
        "[ ] final entry — write it as one line and nothing more: " + assignment_example(ep)
    )
    return "\n".join(out)


def _t062(ep):
    """Arrow bullets with inline quote chains."""
    out = [f"docket board — {plural(len(ep.runs), 'run')} waiting"]
    for r in ep.runs:
        out.append(
            f"-> {r.run_id} >> {r.port} (docket {r.docket}) :: {r.sailors} sailors / "
            f"{plural(r.days, 'day')} / difficulty {r.difficulty} / required specialty "
            f"{run_specialty(r)} / pays {r.contract_payment} coins"
        )
    out.append("")
    out.append("crews (daily rate quoted per required sailor per day)")
    for c in ep.crews:
        chain = " | ".join(
            f"{r.run_id}: mob {quote_for(ep, r, c).mobilization}, rate "
            f"{quote_for(ep, r, c).daily_rate}, difficult-run supp "
            f"{quote_for(ep, r, c).difficulty_supplement}, specialty supp "
            f"{quote_for(ep, r, c).specialty_supplement}"
            for r in ep.runs
        )
        out.append(
            f"-> {c.name} :: skill {c.skill} / specialties {crew_specialties(c)} / "
            f"week {c.runs_this_week} / year {c.runs_this_year} / "
            f"{c.days_since_last} days since last allocation / deferrals this quarter "
            f"{c.deferrals} / registry rank {c.registry_rank}\n   quotes -> {chain}"
        )
    out.append("")
    out.append("-> your move: exactly one line, " + assignment_example(ep))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# memo / notice (4)
# ---------------------------------------------------------------------------

def _t063(ep):
    """Morning muster sheet."""
    out = [
        "MORNING MUSTER — DISPATCH DESK",
        "Item one: runs to be allocated before the muster closes.",
    ]
    for r in ep.runs:
        out.append(
            f"  {r.run_id}: for {r.port}; docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins."
        )
    out.append("Item two: crews reporting, with quoted terms.")
    for c in ep.crews:
        out.append(
            f"  {c.name} reports. Skill {c.skill}; specialties {crew_specialties(c)}; "
            f"runs this week {c.runs_this_week}; runs this year {c.runs_this_year}; "
            f"days since last allocation {c.days_since_last}; deferrals this quarter "
            f"{c.deferrals}; registry rank {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"    Terms for {r.run_id}: mobilization {q.mobilization}; daily rate "
                f"{q.daily_rate} per required sailor per day; difficult-run supplement "
                f"{q.difficulty_supplement}; specialty supplement {q.specialty_supplement}."
            )
    out.append(
        "Item three: the muster closes on the allocation. Enter it as one line, "
        "exactly: " + assignment_example(ep)
    )
    return "\n".join(out)


def _t064(ep):
    """Quartermaster's standing order with data annexes."""
    out = [
        "STANDING ORDER 7 — ALLOCATION OF OPEN RUNS",
        "Per standing order, the annexed particulars are laid before the desk and an "
        "allocation is to be returned without narrative.",
        "",
        "ANNEX A (runs)",
    ]
    for r in ep.runs:
        out.append(
            f"  {r.run_id} | destination {r.port} | docket {r.docket} | sailors {r.sailors} | "
            f"days {r.days} | difficulty {r.difficulty} | required specialty {run_specialty(r)} | "
            f"contract payment {r.contract_payment} coins"
        )
    out.append("ANNEX B (crews)")
    for c in ep.crews:
        out.append(
            f"  {c.name} | skill {c.skill} | specialties {crew_specialties(c)} | "
            f"runs this week {c.runs_this_week} | runs this year {c.runs_this_year} | "
            f"days since last allocation {c.days_since_last} | deferrals this quarter "
            f"{c.deferrals} | registry rank {c.registry_rank}"
        )
    out.append("ANNEX C (quotes; daily rate per required sailor per day)")
    for q in ep.quotes:
        out.append(
            f"  {q.run_id} / {q.crew} | mobilization {q.mobilization} | daily rate {q.daily_rate} | "
            f"difficult-run supplement {q.difficulty_supplement} | specialty supplement "
            f"{q.specialty_supplement}"
        )
    out.append("")
    out.append("ORDERED: one line only, in the exact form " + assignment_example(ep))
    return "\n".join(out)


def _t065(ep):
    """Port authority bulletin."""
    out = [
        "PORT AUTHORITY BULLETIN",
        f"Notice of {plural(len(ep.runs), 'open run')} awaiting allocation. Particulars follow.",
        "",
    ]
    for r in ep.runs:
        out.append(
            f"Run {r.run_id}, sailing for {r.port} under docket {r.docket}: complement "
            f"{r.sailors} sailors, duration {plural(r.days, 'day')}, difficulty {r.difficulty}, "
            f"required specialty {run_specialty(r)}, contract payment {r.contract_payment} coins."
        )
    out.append("")
    out.append("Crews registered as available (quoted daily rates are per required sailor per day):")
    for c in ep.crews:
        qbits = "; ".join(
            f"{r.run_id} at mobilization {quote_for(ep, r, c).mobilization}, daily rate "
            f"{quote_for(ep, r, c).daily_rate}, difficult-run supplement "
            f"{quote_for(ep, r, c).difficulty_supplement}, specialty supplement "
            f"{quote_for(ep, r, c).specialty_supplement}"
            for r in ep.runs
        )
        out.append(
            f"— {c.name} (skill {c.skill}; specialties {crew_specialties(c)}; runs this week "
            f"{c.runs_this_week}; runs this year {c.runs_this_year}; days since last allocation "
            f"{c.days_since_last}; deferrals this quarter {c.deferrals}; registry rank "
            f"{c.registry_rank}). Quotes: {qbits}."
        )
    out.append("")
    out.append(
        "The authority will record whatever single line is returned, provided it reads "
        "exactly: " + assignment_example(ep)
    )
    return "\n".join(out)


def _t066(ep):
    """Tide-table appendix: allocation rider attached to the day's tables."""
    out = [
        "TIDE TABLES — APPENDIX D: ALLOCATION RIDER",
        "The following docket rides with today's tables and must be settled before "
        "high water.",
        "",
        "D.1 Runs",
    ]
    for i, r in enumerate(ep.runs, 1):
        out.append(
            f"  D.1.{i} {r.run_id} — {r.port}; docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins"
        )
    out.append("D.2 Crews and quoted terms (daily rate per required sailor per day)")
    for i, c in enumerate(ep.crews, 1):
        out.append(
            f"  D.2.{i} {c.name} — skill {c.skill}; specialties {crew_specialties(c)}; "
            f"runs this week {c.runs_this_week}; runs this year {c.runs_this_year}; "
            f"days since last allocation {c.days_since_last}; deferrals this quarter "
            f"{c.deferrals}; registry rank {c.registry_rank}"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"        {r.run_id}: mobilization {q.mobilization}; daily rate {q.daily_rate}; "
                f"difficult-run supplement {q.difficulty_supplement}; specialty supplement "
                f"{q.specialty_supplement}"
            )
    out.append("D.3 Entry")
    out.append("  One line, no remarks, exactly: " + assignment_example(ep))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# chat / user-ask (5)
# ---------------------------------------------------------------------------

def _t067(ep):
    """Double-checking before the bell."""
    out = [
        "quick one before the bell rings — i need to lock in this docket and i want "
        "a second pair of eyes. here's everything on my sheet:",
        "",
    ]
    for r in ep.runs:
        out.append(
            f"run {r.run_id} — {r.port}, docket {r.docket}, {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, contract payment {r.contract_payment} coins"
        )
    out.append("")
    for c in ep.crews:
        qtext = "; ".join(
            f"{r.run_id}: mobilization {quote_for(ep, r, c).mobilization}, daily rate "
            f"{quote_for(ep, r, c).daily_rate} per required sailor per day, difficult-run "
            f"supplement {quote_for(ep, r, c).difficulty_supplement}, specialty supplement "
            f"{quote_for(ep, r, c).specialty_supplement}"
            for r in ep.runs
        )
        out.append(
            f"{c.name}: skill {c.skill}, specialties {crew_specialties(c)}, "
            f"{c.runs_this_week} runs this week, {c.runs_this_year} runs this year, "
            f"{c.days_since_last} days since last allocation, {c.deferrals} deferrals "
            f"this quarter, registry rank {c.registry_rank}. quoted {qtext}."
        )
    out.append("")
    out.append(
        "don't walk me through it — the bell's about to go. just the line i should "
        "write, exactly like: " + assignment_example(ep)
    )
    return "\n".join(out)


def _t068(ep):
    """Trainee asked to fill the ledger."""
    out = [
        "Hello — I'm the new trainee at the dispatch desk. My supervisor stepped out and "
        "told me to have the ledger filled in when she's back. She left me this sheet and "
        "said someone would tell me the entry. Here is what the sheet says.",
        "",
        "Runs waiting:",
    ]
    for r in ep.runs:
        out.append(
            f"* {r.run_id}, going to {r.port}. Docket {r.docket}. Needs {r.sailors} sailors "
            f"for {plural(r.days, 'day')}. Difficulty {r.difficulty}. Required specialty: "
            f"{run_specialty(r)}. Contract payment: {r.contract_payment} coins."
        )
    out.append("")
    out.append("Crews and what they quoted (she said the daily rate is per required sailor per day):")
    for c in ep.crews:
        out.append(
            f"* {c.name}. Skill {c.skill}. Specialties: {crew_specialties(c)}. Runs this week: "
            f"{c.runs_this_week}. Runs this year: {c.runs_this_year}. Days since last "
            f"allocation: {c.days_since_last}. Deferrals this quarter: {c.deferrals}. "
            f"Registry rank: {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"  For {r.run_id}: mobilization {q.mobilization}, daily rate {q.daily_rate}, "
                f"difficult-run supplement {q.difficulty_supplement}, specialty supplement "
                f"{q.specialty_supplement}."
            )
    out.append("")
    out.append(
        "What do I write? Please give me only the entry itself, one line, exactly in "
        "this form: " + assignment_example(ep)
    )
    return "\n".join(out)


def _t069(ep):
    """Captain's mate relaying the desk's question."""
    out = [
        "Message relayed from the dispatch desk, via the captain's mate:",
        "",
        '"The desk asks for the allocation on the standing docket. Particulars as read to me:',
    ]
    for r in ep.runs:
        out.append(
            f'Run {r.run_id} for {r.port}, docket {r.docket} — {r.sailors} sailors, '
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, contract payment {r.contract_payment} coins."
        )
    out.append("Crews:")
    for c in ep.crews:
        qbits = "; ".join(
            f"{r.run_id} — mobilization {quote_for(ep, r, c).mobilization}, daily rate "
            f"{quote_for(ep, r, c).daily_rate} per required sailor per day, difficult-run "
            f"supplement {quote_for(ep, r, c).difficulty_supplement}, specialty supplement "
            f"{quote_for(ep, r, c).specialty_supplement}"
            for r in ep.runs
        )
        out.append(
            f"{c.name}, skill {c.skill}, specialties {crew_specialties(c)}, runs this week "
            f"{c.runs_this_week}, runs this year {c.runs_this_year}, days since last allocation "
            f"{c.days_since_last}, deferrals this quarter {c.deferrals}, registry rank "
            f"{c.registry_rank}. Quotes: {qbits}."
        )
    out.append('"')
    out.append("")
    out.append(
        "The mate can carry back exactly one line and will not carry reasons. It must "
        "read: " + assignment_example(ep)
    )
    return "\n".join(out)


def _t070(ep):
    """Night-shift text-message style."""
    out = ["desk. you up?", "need the docket settled tonight. data dump below", ""]
    for r in ep.runs:
        out.append(
            f"{r.run_id}: {r.port} / docket {r.docket} / {r.sailors} sailors / "
            f"{r.days}d / difficulty {r.difficulty} / required specialty {run_specialty(r)} / "
            f"pay {r.contract_payment} coins"
        )
    out.append("")
    out.append("crews (rate = daily rate per required sailor per day):")
    for c in ep.crews:
        out.append(
            f"{c.name}: skill {c.skill} / spec {crew_specialties(c)} / wk {c.runs_this_week} / "
            f"yr {c.runs_this_year} / {c.days_since_last} days since last allocation / "
            f"deferrals this quarter {c.deferrals} / registry rank {c.registry_rank}"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"  {r.run_id}: mob {q.mobilization} / rate {q.daily_rate} / "
                f"difficult-run supplement {q.difficulty_supplement} / "
                f"specialty supplement {q.specialty_supplement}"
            )
    out.append("")
    out.append("no essay pls. one line. exact format:")
    out.append(assignment_example(ep))
    return "\n".join(out)


def _t071(ep):
    """Back from leave, catching up politely."""
    out = [
        "I'm just back from a fortnight's leave and the desk handed me this docket to "
        "settle as my first job. Could you settle it for me? Everything I was given is "
        "copied below, nothing held back.",
        "",
        f"There {'is one open run' if len(ep.runs) == 1 else f'are {len(ep.runs)} open runs'}:",
    ]
    for r in ep.runs:
        out.append(
            f"  {r.run_id}: destination {r.port}; docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins."
        )
    out.append("")
    out.append("The crews, with their quoted terms (daily rate per required sailor per day):")
    for c in ep.crews:
        out.append(
            f"  {c.name}: skill {c.skill}; specialties {crew_specialties(c)}; runs this week "
            f"{c.runs_this_week}; runs this year {c.runs_this_year}; days since last allocation "
            f"{c.days_since_last}; deferrals this quarter {c.deferrals}; registry rank "
            f"{c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"    {r.run_id}: mobilization {q.mobilization}; daily rate {q.daily_rate}; "
                f"difficult-run supplement {q.difficulty_supplement}; specialty supplement "
                f"{q.specialty_supplement}."
            )
    out.append("")
    out.append(
        "I'd rather not hand the desk an explanation, just the entry — one line, "
        "exactly: " + assignment_example(ep)
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# json / toolcall (4)
# ---------------------------------------------------------------------------

def _t072(ep):
    """XML-ish tags."""
    out = ["<allocation_request>"]
    out.append(" <runs>")
    for r in ep.runs:
        out.append(
            f'  <run id="{r.run_id}" destination="{r.port}" docket="{r.docket}" '
            f'sailors="{r.sailors}" days="{r.days}" difficulty="{r.difficulty}" '
            f'required_specialty="{run_specialty(r)}" contract_payment_coins="{r.contract_payment}"/>'
        )
    out.append(" </runs>")
    out.append(" <crews>")
    for c in ep.crews:
        out.append(
            f'  <crew name="{c.name}" skill="{c.skill}" specialties="{crew_specialties(c, "none")}" '
            f'runs_this_week="{c.runs_this_week}" runs_this_year="{c.runs_this_year}" '
            f'days_since_last_allocation="{c.days_since_last}" deferrals_this_quarter="{c.deferrals}" '
            f'registry_rank="{c.registry_rank}"/>'
        )
    out.append(" </crews>")
    out.append(" <quotes daily_rate_units=\"per required sailor per day\">")
    for q in ep.quotes:
        out.append(
            f'  <quote run="{q.run_id}" crew="{q.crew}" mobilization="{q.mobilization}" '
            f'daily_rate="{q.daily_rate}" difficult_run_supplement="{q.difficulty_supplement}" '
            f'specialty_supplement="{q.specialty_supplement}"/>'
        )
    out.append(" </quotes>")
    out.append(f" <respond lines=\"1\" format=\"{assignment_example(ep)}\"/>")
    out.append("</allocation_request>")
    return "\n".join(out)


def _t073(ep):
    """tool_use block style."""
    payload = {
        "runs": [
            {
                "run_id": r.run_id,
                "destination": r.port,
                "docket": r.docket,
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
                "run": q.run_id,
                "crew": q.crew,
                "mobilization": q.mobilization,
                "daily_rate_per_required_sailor_per_day": q.daily_rate,
                "difficult_run_supplement": q.difficulty_supplement,
                "specialty_supplement": q.specialty_supplement,
            }
            for q in ep.quotes
        ],
    }
    return (
        "<tool_use name=\"decide_allocation\">\n"
        "<input>\n"
        + _json.dumps(payload, separators=(",", ":"))
        + "\n</input>\n"
        "<output_contract>single line, exactly: "
        + assignment_example(ep)
        + "</output_contract>\n"
        "</tool_use>"
    )


def _t074(ep):
    """Form-encoded payload lines."""
    out = ["POST /desk/allocation", "content-type: application/x-www-form-urlencoded", ""]
    for i, r in enumerate(ep.runs):
        out.append(
            f"run.{i}.id={r.run_id}&run.{i}.destination={r.port}&run.{i}.docket={r.docket}"
            f"&run.{i}.sailors={r.sailors}&run.{i}.days={r.days}&run.{i}.difficulty={r.difficulty}"
            f"&run.{i}.required_specialty={run_specialty(r)}&run.{i}.contract_payment_coins={r.contract_payment}"
        )
    for i, c in enumerate(ep.crews):
        out.append(
            f"crew.{i}.name={c.name}&crew.{i}.skill={c.skill}"
            f"&crew.{i}.specialties={crew_specialties(c, 'none').replace(', ', '|')}"
            f"&crew.{i}.runs_this_week={c.runs_this_week}&crew.{i}.runs_this_year={c.runs_this_year}"
            f"&crew.{i}.days_since_last_allocation={c.days_since_last}"
            f"&crew.{i}.deferrals_this_quarter={c.deferrals}&crew.{i}.registry_rank={c.registry_rank}"
        )
    for i, q in enumerate(ep.quotes):
        out.append(
            f"quote.{i}.run={q.run_id}&quote.{i}.crew={q.crew}&quote.{i}.mobilization={q.mobilization}"
            f"&quote.{i}.daily_rate_per_required_sailor_per_day={q.daily_rate}"
            f"&quote.{i}.difficult_run_supplement={q.difficulty_supplement}"
            f"&quote.{i}.specialty_supplement={q.specialty_supplement}"
        )
    out.append("")
    out.append("expected response body: one line, exactly")
    out.append(assignment_example(ep))
    return "\n".join(out)


def _t075(ep):
    """JSONL records, one typed object per line."""
    lines = []
    for r in ep.runs:
        lines.append(_json.dumps({
            "type": "run", "run_id": r.run_id, "destination": r.port, "docket": r.docket,
            "sailors": r.sailors, "days": r.days, "difficulty": r.difficulty,
            "required_specialty": r.specialty, "contract_payment_coins": r.contract_payment,
        }, separators=(",", ":")))
    for c in ep.crews:
        lines.append(_json.dumps({
            "type": "crew", "name": c.name, "skill": c.skill,
            "specialties": list(c.specialties), "runs_this_week": c.runs_this_week,
            "runs_this_year": c.runs_this_year, "days_since_last_allocation": c.days_since_last,
            "deferrals_this_quarter": c.deferrals, "registry_rank": c.registry_rank,
        }, separators=(",", ":")))
    lines.append(_json.dumps({
        "type": "note", "daily_rate": "per required sailor per day",
    }, separators=(",", ":")))
    for q in ep.quotes:
        lines.append(_json.dumps({
            "type": "quote", "run": q.run_id, "crew": q.crew, "mobilization": q.mobilization,
            "daily_rate": q.daily_rate,
            "difficult_run_supplement": q.difficulty_supplement,
            "specialty_supplement": q.specialty_supplement,
        }, separators=(",", ":")))
    lines.append(_json.dumps({
        "type": "task", "decide": "allocation", "reply_lines": 1,
        "reply_format": assignment_example(ep),
    }, separators=(",", ":")))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# yaml / structured (3)
# ---------------------------------------------------------------------------

def _t076(ep):
    """TOML-ish sections."""
    out = []
    for r in ep.runs:
        out += [
            f"[[run]]",
            f"id = \"{r.run_id}\"",
            f"destination = \"{r.port}\"",
            f"docket = {r.docket}",
            f"sailors = {r.sailors}",
            f"days = {r.days}",
            f"difficulty = {r.difficulty}",
            f"required_specialty = \"{run_specialty(r)}\"",
            f"contract_payment_coins = {r.contract_payment}",
            "",
        ]
    for c in ep.crews:
        out += [
            f"[[crew]]",
            f"name = \"{c.name}\"",
            f"skill = {c.skill}",
            f"specialties = \"{crew_specialties(c)}\"",
            f"runs_this_week = {c.runs_this_week}",
            f"runs_this_year = {c.runs_this_year}",
            f"days_since_last_allocation = {c.days_since_last}",
            f"deferrals_this_quarter = {c.deferrals}",
            f"registry_rank = {c.registry_rank}",
            "",
        ]
    out += [
        "# quote fields: run, crew, mobilization, daily_rate (per required "
        "sailor per day), difficult-run supplement, specialty supplement",
        "[quotes]",
    ]
    for q in ep.quotes:
        out.append(
            f"{q.run_id}_{q.crew} = [\"{q.run_id}\", \"{q.crew}\", "
            f"{q.mobilization}, {q.daily_rate}, {q.difficulty_supplement}, "
            f"{q.specialty_supplement}]"
        )
    out.append("")
    out += [
        "[task]",
        "decision = \"allocation\"",
        "reply_lines = 1",
        f"reply_format = \"{assignment_example(ep)}\"",
    ]
    return "\n".join(out)


def _t077(ep):
    """INI style."""
    out = []
    for r in ep.runs:
        out += [
            f"[run {r.run_id}]",
            f"destination={r.port}",
            f"docket={r.docket}",
            f"sailors={r.sailors}",
            f"days={r.days}",
            f"difficulty={r.difficulty}",
            f"required_specialty={run_specialty(r)}",
            f"contract_payment_coins={r.contract_payment}",
            "",
        ]
    for c in ep.crews:
        out += [
            f"[crew {c.name}]",
            f"skill={c.skill}",
            f"specialties={crew_specialties(c)}",
            f"runs_this_week={c.runs_this_week}",
            f"runs_this_year={c.runs_this_year}",
            f"days_since_last_allocation={c.days_since_last}",
            f"deferrals_this_quarter={c.deferrals}",
            f"registry_rank={c.registry_rank}",
            "",
        ]
    out.append("; daily_rate below is per required sailor per day")
    for q in ep.quotes:
        out += [
            f"[quote {q.run_id}:{q.crew}]",
            f"mobilization={q.mobilization}",
            f"daily_rate={q.daily_rate}",
            f"difficult_run_supplement={q.difficulty_supplement}",
            f"specialty_supplement={q.specialty_supplement}",
            "",
        ]
    out += [
        "[reply]",
        "lines=1",
        f"format={assignment_example(ep)}",
    ]
    return "\n".join(out)


def _t078(ep):
    """Indented key: value tree, no list dashes."""
    out = [
        "note: quote lines read mobilization, daily_rate (per required sailor "
        "per day), difficult_run_supplement, specialty_supplement",
        "docket:",
    ]
    for i, r in enumerate(ep.runs, 1):
        out += [
            f"  run_{i}:",
            f"    id: {r.run_id}",
            f"    destination: {r.port}",
            f"    docket_number: {r.docket}",
            f"    sailors: {r.sailors}",
            f"    days: {r.days}",
            f"    difficulty: {r.difficulty}",
            f"    required_specialty: {run_specialty(r)}",
            f"    contract_payment_coins: {r.contract_payment}",
        ]
    for i, c in enumerate(ep.crews, 1):
        out += [
            f"  crew_{i}:",
            f"    name: {c.name}",
            f"    skill: {c.skill}",
            f"    specialties: {crew_specialties(c)}",
            f"    runs_this_week: {c.runs_this_week}",
            f"    runs_this_year: {c.runs_this_year}",
            f"    days_since_last_allocation: {c.days_since_last}",
            f"    deferrals_this_quarter: {c.deferrals}",
            f"    registry_rank: {c.registry_rank}",
            "    quotes:",
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"      {r.run_id}: mobilization {q.mobilization}, daily_rate "
                f"{q.daily_rate}, difficult_run_supplement "
                f"{q.difficulty_supplement}, specialty_supplement "
                f"{q.specialty_supplement}"
            )
    out += [
        "  reply:",
        "    lines: 1",
        f"    exact_format: {assignment_example(ep)}",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# table (3)
# ---------------------------------------------------------------------------

def _t079(ep):
    """Fixed-width aligned columns with ruler lines."""
    def block(header, rows):
        widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
        def fmt(cells):
            return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()
        ruler = "  ".join("-" * w for w in widths)
        return [fmt(header), ruler] + [fmt(r) for r in rows]

    out = ["ALLOCATION WORKSHEET", ""]
    out += block(
        ["run", "destination", "docket", "sailors", "days", "difficulty", "required specialty", "contract payment (coins)"],
        [[r.run_id, r.port, r.docket, r.sailors, r.days, r.difficulty, run_specialty(r), r.contract_payment] for r in ep.runs],
    )
    out.append("")
    out += block(
        ["crew", "skill", "specialties", "runs this week", "runs this year", "days since last allocation", "deferrals this quarter", "registry rank"],
        [[c.name, c.skill, crew_specialties(c), c.runs_this_week, c.runs_this_year, c.days_since_last, c.deferrals, c.registry_rank] for c in ep.crews],
    )
    out.append("")
    out.append("quotes — daily rate is per required sailor per day")
    out += block(
        ["run", "crew", "mobilization", "daily rate", "difficult-run supplement", "specialty supplement"],
        [[q.run_id, q.crew, q.mobilization, q.daily_rate, q.difficulty_supplement, q.specialty_supplement] for q in ep.quotes],
    )
    out.append("")
    out.append("Fill the last line of the worksheet with exactly: " + assignment_example(ep))
    return "\n".join(out)


def _t080(ep):
    """HTML-ish table tags."""
    out = ["<h3>Docket awaiting allocation</h3>", "<table id=\"runs\">"]
    out.append("<tr><th>run</th><th>destination</th><th>docket</th><th>sailors</th><th>days</th><th>difficulty</th><th>required specialty</th><th>contract payment (coins)</th></tr>")
    for r in ep.runs:
        out.append(
            f"<tr><td>{r.run_id}</td><td>{r.port}</td><td>{r.docket}</td><td>{r.sailors}</td>"
            f"<td>{r.days}</td><td>{r.difficulty}</td><td>{run_specialty(r)}</td><td>{r.contract_payment}</td></tr>"
        )
    out.append("</table>")
    out.append("<table id=\"crews\">")
    out.append("<tr><th>crew</th><th>skill</th><th>specialties</th><th>runs this week</th><th>runs this year</th><th>days since last allocation</th><th>deferrals this quarter</th><th>registry rank</th></tr>")
    for c in ep.crews:
        out.append(
            f"<tr><td>{c.name}</td><td>{c.skill}</td><td>{crew_specialties(c)}</td>"
            f"<td>{c.runs_this_week}</td><td>{c.runs_this_year}</td><td>{c.days_since_last}</td>"
            f"<td>{c.deferrals}</td><td>{c.registry_rank}</td></tr>"
        )
    out.append("</table>")
    out.append("<table id=\"quotes\" data-daily-rate=\"per required sailor per day\">")
    out.append("<tr><th>run</th><th>crew</th><th>mobilization</th><th>daily rate</th><th>difficult-run supplement</th><th>specialty supplement</th></tr>")
    for q in ep.quotes:
        out.append(
            f"<tr><td>{q.run_id}</td><td>{q.crew}</td><td>{q.mobilization}</td>"
            f"<td>{q.daily_rate}</td><td>{q.difficulty_supplement}</td><td>{q.specialty_supplement}</td></tr>"
        )
    out.append("</table>")
    out.append(f"<p>Reply with one line only, exactly: <code>{assignment_example(ep)}</code></p>")
    return "\n".join(out)


def _t081(ep):
    """Org-mode style tables with captions."""
    out = ["#+TITLE: Docket sheet", "", "#+CAPTION: open runs"]
    out.append("| run | destination | docket | sailors | days | difficulty | required specialty | contract payment (coins) |")
    out.append("|-----+-------------+--------+---------+------+------------+--------------------+--------------------------|")
    for r in ep.runs:
        out.append(
            f"| {r.run_id} | {r.port} | {r.docket} | {r.sailors} | {r.days} | {r.difficulty} "
            f"| {run_specialty(r)} | {r.contract_payment} |"
        )
    out += ["", "#+CAPTION: crews"]
    out.append("| crew | skill | specialties | runs this week | runs this year | days since last allocation | deferrals this quarter | registry rank |")
    out.append("|------+-------+-------------+----------------+----------------+----------------------------+------------------------+---------------|")
    for c in ep.crews:
        out.append(
            f"| {c.name} | {c.skill} | {crew_specialties(c)} | {c.runs_this_week} | "
            f"{c.runs_this_year} | {c.days_since_last} | {c.deferrals} | {c.registry_rank} |"
        )
    out += ["", "#+CAPTION: quotes, daily rate per required sailor per day"]
    out.append("| run | crew | mobilization | daily rate | difficult-run supplement | specialty supplement |")
    out.append("|-----+------+--------------+------------+--------------------------+----------------------|")
    for q in ep.quotes:
        out.append(
            f"| {q.run_id} | {q.crew} | {q.mobilization} | {q.daily_rate} | "
            f"{q.difficulty_supplement} | {q.specialty_supplement} |"
        )
    out += ["", "* Task", "One line, nothing else:", "", ": " + assignment_example(ep)]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# csv / tsv (2)
# ---------------------------------------------------------------------------

def _t082(ep):
    """TSV blocks."""
    out = ["# docket extract (tab-separated); daily_rate is per required sailor per day", ""]
    out.append("runs")
    out.append("run_id\tdestination\tdocket\tsailors\tdays\tdifficulty\trequired_specialty\tcontract_payment_coins")
    for r in ep.runs:
        out.append(
            f"{r.run_id}\t{r.port}\t{r.docket}\t{r.sailors}\t{r.days}\t{r.difficulty}\t"
            f"{run_specialty(r)}\t{r.contract_payment}"
        )
    out.append("")
    out.append("crews")
    out.append("name\tskill\tspecialties\truns_this_week\truns_this_year\tdays_since_last_allocation\tdeferrals_this_quarter\tregistry_rank")
    for c in ep.crews:
        out.append(
            f"{c.name}\t{c.skill}\t{crew_specialties(c)}\t{c.runs_this_week}\t{c.runs_this_year}\t"
            f"{c.days_since_last}\t{c.deferrals}\t{c.registry_rank}"
        )
    out.append("")
    out.append("quotes")
    out.append("run_id\tcrew\tmobilization\tdaily_rate\tdifficult_run_supplement\tspecialty_supplement")
    for q in ep.quotes:
        out.append(
            f"{q.run_id}\t{q.crew}\t{q.mobilization}\t{q.daily_rate}\t{q.difficulty_supplement}\t{q.specialty_supplement}"
        )
    out.append("")
    out.append("required reply, one line exactly:")
    out.append(assignment_example(ep))
    return "\n".join(out)


def _t083(ep):
    """Semicolon-separated values with a header comment per block."""
    out = ["; docket extract, semicolon-separated. daily_rate = per required sailor per day."]
    out.append(";runs: run_id;destination;docket;sailors;days;difficulty;required_specialty;contract_payment_coins")
    for r in ep.runs:
        out.append(
            f"{r.run_id};{r.port};{r.docket};{r.sailors};{r.days};{r.difficulty};"
            f"{run_specialty(r)};{r.contract_payment}"
        )
    out.append(";crews: name;skill;specialties;runs_this_week;runs_this_year;days_since_last_allocation;deferrals_this_quarter;registry_rank")
    for c in ep.crews:
        out.append(
            f"{c.name};{c.skill};{crew_specialties(c, 'none').replace(', ', '|')};"
            f"{c.runs_this_week};{c.runs_this_year};{c.days_since_last};{c.deferrals};{c.registry_rank}"
        )
    out.append(";quotes: run_id;crew;mobilization;daily_rate;difficult_run_supplement;specialty_supplement")
    for q in ep.quotes:
        out.append(
            f"{q.run_id};{q.crew};{q.mobilization};{q.daily_rate};{q.difficulty_supplement};{q.specialty_supplement}"
        )
    out.append("; reply with exactly one line:")
    out.append(assignment_example(ep))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# prose (4)
# ---------------------------------------------------------------------------

def _t084(ep):
    """News gazette item."""
    rbits = []
    for r in ep.runs:
        rbits.append(
            f"{r.run_id} to {r.port} (docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins)"
        )
    paras = [
        "THE HARBOUR GAZETTE — DOCKET PAGE",
        f"The desk publishes {'one run' if len(ep.runs) == 1 else f'{len(ep.runs)} runs'} "
        "still wanting crews as we go to press: " + "; and ".join(rbits) + ".",
    ]
    for c in ep.crews:
        qbits = "; ".join(
            f"on {r.run_id}, mobilization {quote_for(ep, r, c).mobilization} with a daily rate "
            f"of {quote_for(ep, r, c).daily_rate} per required sailor per day, difficult-run "
            f"supplement {quote_for(ep, r, c).difficulty_supplement}, specialty supplement "
            f"{quote_for(ep, r, c).specialty_supplement}"
            for r in ep.runs
        )
        paras.append(
            f"Of the crews: {c.name} (skill {c.skill}; specialties {crew_specialties(c)}) shows "
            f"{c.runs_this_week} runs this week and {c.runs_this_year} this year, "
            f"{c.days_since_last} days since last allocation, {c.deferrals} deferrals this "
            f"quarter, registry rank {c.registry_rank}; quoting {qbits}."
        )
    paras.append(
        "The Gazette prints whatever single line the desk returns, verbatim, in the form "
        + assignment_example(ep)
    )
    return "\n\n".join(paras)


def _t085(ep):
    """Apprentice's diary entry."""
    paras = [
        "From the apprentice's diary.",
        "The desk gave me the whole docket to copy out fair before it is settled, "
        "and I set it down here exactly as the sheet reads.",
    ]
    for r in ep.runs:
        paras.append(
            f"First the run {r.run_id}: it sails for {r.port}, docket {r.docket}, wanting "
            f"{r.sailors} sailors for {plural(r.days, 'day')} at difficulty {r.difficulty}; "
            f"required specialty {run_specialty(r)}; contract payment {r.contract_payment} coins."
        )
    for c in ep.crews:
        qbits = "; ".join(
            f"for {r.run_id} mobilization {quote_for(ep, r, c).mobilization}, daily rate "
            f"{quote_for(ep, r, c).daily_rate} per required sailor per day, difficult-run "
            f"supplement {quote_for(ep, r, c).difficulty_supplement}, specialty supplement "
            f"{quote_for(ep, r, c).specialty_supplement}"
            for r in ep.runs
        )
        paras.append(
            f"Then {c.name}, skill {c.skill}, specialties {crew_specialties(c)}: "
            f"{c.runs_this_week} runs this week, {c.runs_this_year} runs this year, "
            f"{c.days_since_last} days since last allocation, {c.deferrals} deferrals this "
            f"quarter, registry rank {c.registry_rank}; and their quotes, {qbits}."
        )
    paras.append(
        "The master says the settling is not mine to reason at, only to record: one line, "
        "exactly " + assignment_example(ep)
    )
    return "\n\n".join(paras)


def _t086(ep):
    """Inspection report narrative."""
    paras = [
        "INSPECTION OF THE DISPATCH BOARD — FINDINGS",
        f"On inspection the board carried {plural(len(ep.runs), 'unallocated run')}. "
        "The particulars were found in good order and are restated below for the record.",
    ]
    for r in ep.runs:
        paras.append(
            f"Run {r.run_id}, destination {r.port}, docket {r.docket}: complement "
            f"{r.sailors} sailors; duration {plural(r.days, 'day')}; difficulty "
            f"{r.difficulty}; required specialty {run_specialty(r)}; contract payment "
            f"{r.contract_payment} coins."
        )
    for c in ep.crews:
        qbits = "; ".join(
            f"{r.run_id} — mobilization {quote_for(ep, r, c).mobilization}, daily rate "
            f"{quote_for(ep, r, c).daily_rate} per required sailor per day, difficult-run "
            f"supplement {quote_for(ep, r, c).difficulty_supplement}, specialty supplement "
            f"{quote_for(ep, r, c).specialty_supplement}"
            for r in ep.runs
        )
        paras.append(
            f"Crew {c.name} was found available: skill {c.skill}; specialties "
            f"{crew_specialties(c)}; runs this week {c.runs_this_week}; runs this year "
            f"{c.runs_this_year}; days since last allocation {c.days_since_last}; deferrals "
            f"this quarter {c.deferrals}; registry rank {c.registry_rank}. Quotes on file: {qbits}."
        )
    paras.append(
        "DEFICIENCY: the allocation line is blank. Remedy by supplying exactly one line "
        "in the form " + assignment_example(ep)
    )
    return "\n\n".join(paras)


def _t087(ep):
    """Storyteller recap."""
    rbits = []
    for r in ep.runs:
        rbits.append(
            f"{r.run_id}, bound for {r.port} under docket {r.docket} — {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, and {r.contract_payment} coins on the contract"
        )
    paras = [
        "So the board stood thus, as anyone at the harbour that morning could tell you: "
        + "; and ".join(rbits) + "."
    ]
    for c in ep.crews:
        qbits = "; ".join(
            f"for {r.run_id}, mobilization {quote_for(ep, r, c).mobilization}, a daily rate of "
            f"{quote_for(ep, r, c).daily_rate} per required sailor per day, difficult-run "
            f"supplement {quote_for(ep, r, c).difficulty_supplement}, specialty supplement "
            f"{quote_for(ep, r, c).specialty_supplement}"
            for r in ep.runs
        )
        paras.append(
            f"And there was {c.name} — skill {c.skill}, versed in {crew_specialties(c, 'nothing listed')}, "
            f"{c.runs_this_week} runs this week and {c.runs_this_year} this year behind them, "
            f"{c.days_since_last} days since last allocation, {c.deferrals} deferrals this quarter, "
            f"registry rank {c.registry_rank} — who quoted {qbits}."
        )
    paras.append(
        "How does the story end? Tell it in one line and no more, exactly so: "
        + assignment_example(ep)
    )
    return "\n\n".join(paras)


# ---------------------------------------------------------------------------
# dialogue / transcript (3)
# ---------------------------------------------------------------------------

def _t088(ep):
    """Minuted meeting."""
    out = [
        "MINUTES OF THE ALLOCATION SITTING",
        "Present: the registrar; the clerk of runs; the clerk of crews.",
        "",
        "The clerk of runs read into the minutes:",
    ]
    for r in ep.runs:
        out.append(
            f"  — {r.run_id}, destination {r.port}, docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins."
        )
    out.append("The clerk of crews read into the minutes (daily rates per required sailor per day):")
    for c in ep.crews:
        out.append(
            f"  — {c.name}: skill {c.skill}; specialties {crew_specialties(c)}; runs this week "
            f"{c.runs_this_week}; runs this year {c.runs_this_year}; days since last allocation "
            f"{c.days_since_last}; deferrals this quarter {c.deferrals}; registry rank {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"      quote, {r.run_id}: mobilization {q.mobilization}; daily rate {q.daily_rate}; "
                f"difficult-run supplement {q.difficulty_supplement}; specialty supplement "
                f"{q.specialty_supplement}."
            )
    out.append("")
    out.append(
        "RESOLVED, that the sitting records the allocation as a single line in the exact "
        "form: " + assignment_example(ep)
    )
    return "\n".join(out)


def _t089(ep):
    """Question-and-answer deposition."""
    out = ["DEPOSITION TAKEN AT THE DISPATCH DESK", ""]
    out.append("Q. What stands open on the docket?")
    for r in ep.runs:
        out.append(
            f"A. Run {r.run_id}, for {r.port}, docket {r.docket}. It wants {r.sailors} sailors "
            f"for {plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, and carries a contract payment of {r.contract_payment} coins."
        )
    out.append("Q. Who has put themselves forward, and on what terms?")
    for c in ep.crews:
        out.append(
            f"A. {c.name}. Skill {c.skill}; specialties {crew_specialties(c)}; "
            f"{c.runs_this_week} runs this week; {c.runs_this_year} runs this year; "
            f"{c.days_since_last} days since last allocation; {c.deferrals} deferrals this "
            f"quarter; registry rank {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"A. Their quote for {r.run_id}: mobilization {q.mobilization}; daily rate "
                f"{q.daily_rate}, that being per required sailor per day; difficult-run "
                f"supplement {q.difficulty_supplement}; specialty supplement {q.specialty_supplement}."
            )
    out.append("Q. And the allocation?")
    out.append(
        "A. [the deponent is to answer with one line only, in exactly this form: "
        + assignment_example(ep) + "]"
    )
    return "\n".join(out)


def _t090(ep):
    """Signal-lamp exchange."""
    out = ["SIGNAL LOG — LAMP EXCHANGE, TOWER TO DESK", ""]
    for r in ep.runs:
        out.append(
            f"TOWER SENDS: run {r.run_id} open. destination {r.port}. docket {r.docket}. "
            f"sailors {r.sailors}. days {r.days}. difficulty {r.difficulty}. required "
            f"specialty {run_specialty(r)}. contract payment {r.contract_payment} coins."
        )
    out.append("DESK ACKNOWLEDGES.")
    for c in ep.crews:
        out.append(
            f"TOWER SENDS: crew {c.name}. skill {c.skill}. specialties {crew_specialties(c)}. "
            f"runs this week {c.runs_this_week}. runs this year {c.runs_this_year}. days since "
            f"last allocation {c.days_since_last}. deferrals this quarter {c.deferrals}. "
            f"registry rank {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"TOWER SENDS: quote {c.name} on {r.run_id}. mobilization {q.mobilization}. "
                f"daily rate {q.daily_rate} per required sailor per day. difficult-run "
                f"supplement {q.difficulty_supplement}. specialty supplement {q.specialty_supplement}."
            )
    out.append("DESK ACKNOWLEDGES ALL.")
    out.append(
        "TOWER SENDS: return allocation by lamp. one line. exact form follows. "
        + assignment_example(ep)
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# email / letter (3)
# ---------------------------------------------------------------------------

def _t091(ep):
    """Internal email with quoted data block."""
    data = []
    for r in ep.runs:
        data.append(
            f"> {r.run_id}: destination {r.port}; docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins."
        )
    data.append(">")
    for c in ep.crews:
        data.append(
            f"> {c.name}: skill {c.skill}; specialties {crew_specialties(c)}; runs this week "
            f"{c.runs_this_week}; runs this year {c.runs_this_year}; days since last allocation "
            f"{c.days_since_last}; deferrals this quarter {c.deferrals}; registry rank {c.registry_rank}."
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            data.append(
                f">   {r.run_id}: mobilization {q.mobilization}; daily rate {q.daily_rate} per "
                f"required sailor per day; difficult-run supplement {q.difficulty_supplement}; "
                f"specialty supplement {q.specialty_supplement}."
            )
    return (
        "From: registrar@harbour.qv\n"
        "To: allocations@harbour.qv\n"
        "Subject: RE: docket pending — data below, decision needed\n\n"
        "Forwarding the desk's sheet unchanged (below the line). Send back the entry "
        "only — one line, no covering note, exactly:\n"
        + assignment_example(ep)
        + "\n\n----- forwarded sheet -----\n"
        + "\n".join(data)
    )


def _t092(ep):
    """Reply-requested notice."""
    out = [
        "NOTICE — REPLY REQUESTED BY RETURN",
        "",
        "The under-noted docket stands open. A reply in the prescribed form is requested "
        "by return of the bearer.",
        "",
        "Runs:",
    ]
    for r in ep.runs:
        out.append(
            f"  {r.run_id} — {r.port}; docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins"
        )
    out.append("")
    out.append("Crews (quotes state the daily rate per required sailor per day):")
    for c in ep.crews:
        out.append(
            f"  {c.name} — skill {c.skill}; specialties {crew_specialties(c)}; runs this week "
            f"{c.runs_this_week}; runs this year {c.runs_this_year}; days since last allocation "
            f"{c.days_since_last}; deferrals this quarter {c.deferrals}; registry rank {c.registry_rank}"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"    quote {r.run_id} — mobilization {q.mobilization}; daily rate {q.daily_rate}; "
                f"difficult-run supplement {q.difficulty_supplement}; specialty supplement "
                f"{q.specialty_supplement}"
            )
    out.append("")
    out.append("Prescribed form of reply (one line, nothing further):")
    out.append("  " + assignment_example(ep))
    return "\n".join(out)


def _t093(ep):
    """Formal petition."""
    out = [
        "TO THE KEEPER OF THE DOCKET, GREETINGS.",
        "",
        "Your petitioner, being the clerk on duty, humbly lays before you the matter of "
        "the open runs and prays that an allocation be pronounced.",
        "",
        "The runs, as scheduled:",
    ]
    for i, r in enumerate(ep.runs, 1):
        out.append(
            f"  Schedule {i}: {r.run_id}, for {r.port}, docket {r.docket}; {r.sailors} sailors; "
            f"{plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins."
        )
    out.append("")
    out.append("The crews, as enrolled, with their quoted terms (daily rate per required sailor per day):")
    for c in ep.crews:
        qbits = "; ".join(
            f"{r.run_id}: mobilization {quote_for(ep, r, c).mobilization}, daily rate "
            f"{quote_for(ep, r, c).daily_rate}, difficult-run supplement "
            f"{quote_for(ep, r, c).difficulty_supplement}, specialty supplement "
            f"{quote_for(ep, r, c).specialty_supplement}"
            for r in ep.runs
        )
        out.append(
            f"  {c.name}, of skill {c.skill}, holding {crew_specialties(c, 'no specialties')}; "
            f"runs this week {c.runs_this_week}; runs this year {c.runs_this_year}; days since "
            f"last allocation {c.days_since_last}; deferrals this quarter {c.deferrals}; registry "
            f"rank {c.registry_rank}; quoting {qbits}."
        )
    out.append("")
    out.append(
        "Your petitioner asks only the pronouncement itself, one line, without reasons, "
        "in exactly this form: " + assignment_example(ep)
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# telegraph / terse (2)
# ---------------------------------------------------------------------------

def _t094(ep):
    """Semicolon-fielded shorthand with legends."""
    out = [
        f"ALLOC REQ // {plural(len(ep.runs), 'RUN')} // {plural(len(ep.crews), 'CREW')}",
        "legend R: id;port;docket;sailors;days;difficulty;required specialty;contract payment coins",
        "legend C: name;skill;specialties;runs this week;runs this year;days since last allocation;deferrals this quarter;registry rank",
        "legend Q: run;crew;mobilization;daily rate (per required sailor per day);difficult-run supplement;specialty supplement",
    ]
    for r in ep.runs:
        out.append(
            f"R:{r.run_id};{r.port};{r.docket};{r.sailors};{r.days};{r.difficulty};"
            f"{run_specialty(r)};{r.contract_payment}"
        )
    for c in ep.crews:
        out.append(
            f"C:{c.name};{c.skill};{crew_specialties(c, 'none').replace(', ', '+')};"
            f"{c.runs_this_week};{c.runs_this_year};{c.days_since_last};{c.deferrals};{c.registry_rank}"
        )
    for q in ep.quotes:
        out.append(
            f"Q:{q.run_id};{q.crew};{q.mobilization};{q.daily_rate};{q.difficulty_supplement};"
            f"{q.specialty_supplement}"
        )
    out.append("SEND ONE LINE ONLY: " + assignment_example(ep))
    return "\n".join(out)


def _t095(ep):
    """Numbered flash messages with an expansion key."""
    out = [
        "== DISPATCH FLASH TRAFFIC ==",
        "key: dest=destination dkt=docket slr=sailors dy=days dif=difficulty "
        "spc=required specialty pay=contract payment coins",
        "key: sk=skill wk=runs this week yr=runs this year dsl=days since last allocation "
        "dfq=deferrals this quarter rr=registry rank",
        "key: mob=mobilization dr=daily rate per required sailor per day "
        "drs=difficult-run supplement sps=specialty supplement",
    ]
    n = 0
    for r in ep.runs:
        n += 1
        out.append(
            f"flash {n}: RUN {r.run_id} dest={r.port} dkt={r.docket} slr={r.sailors} "
            f"dy={r.days} dif={r.difficulty} spc={run_specialty(r)} pay={r.contract_payment}"
        )
    for c in ep.crews:
        n += 1
        out.append(
            f"flash {n}: CREW {c.name} sk={c.skill} spc={crew_specialties(c)} wk={c.runs_this_week} "
            f"yr={c.runs_this_year} dsl={c.days_since_last} dfq={c.deferrals} rr={c.registry_rank}"
        )
    for q in ep.quotes:
        n += 1
        out.append(
            f"flash {n}: QUOTE {q.crew}/{q.run_id} mob={q.mobilization} dr={q.daily_rate} "
            f"drs={q.difficulty_supplement} sps={q.specialty_supplement}"
        )
    out.append("reply flash, exactly one line: " + assignment_example(ep))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# form / ledger (3)
# ---------------------------------------------------------------------------

def _t096(ep):
    """Intake stamp card."""
    out = ["DESK INTAKE CARD — STAMP ON COMPLETION", "", "RECEIVED, the following particulars:"]
    for r in ep.runs:
        out += [
            f"  RUN .......... {r.run_id}",
            f"  DESTINATION .. {r.port}",
            f"  DOCKET ....... {r.docket}",
            f"  SAILORS ...... {r.sailors}",
            f"  DAYS ......... {r.days}",
            f"  DIFFICULTY ... {r.difficulty}",
            f"  REQ SPECIALTY  {run_specialty(r)}",
            f"  PAYMENT ...... {r.contract_payment} coins",
            "  ----",
        ]
    out.append("CREWS PRESENTED (daily rate per required sailor per day):")
    for c in ep.crews:
        out += [
            f"  CREW ......... {c.name}",
            f"  SKILL ........ {c.skill}",
            f"  SPECIALTIES .. {crew_specialties(c)}",
            f"  RUNS WK/YR ... {c.runs_this_week} this week / {c.runs_this_year} this year",
            f"  DAYS SINCE LAST ALLOCATION {c.days_since_last}",
            f"  DEFERRALS THIS QUARTER ... {c.deferrals}",
            f"  REGISTRY RANK  {c.registry_rank}",
        ]
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"  QUOTE {r.run_id} .. mob {q.mobilization} / daily rate {q.daily_rate} / "
                f"difficult-run supp {q.difficulty_supplement} / specialty supp {q.specialty_supplement}"
            )
        out.append("  ----")
    out.append("STAMP REQUIRES the allocation line, one line, exactly:")
    out.append("  " + assignment_example(ep))
    return "\n".join(out)


def _t097(ep):
    """Columnar daybook."""
    out = [
        "DAYBOOK — UNSETTLED PAGE",
        "",
        "col.1 runs | col.2 particulars",
    ]
    for r in ep.runs:
        out.append(
            f"{r.run_id} | destination {r.port} · docket {r.docket} · sailors {r.sailors} · "
            f"days {r.days} · difficulty {r.difficulty} · required specialty {run_specialty(r)} · "
            f"contract payment {r.contract_payment} coins"
        )
    out.append("")
    out.append("col.1 crews | col.2 standing | col.3 quotes (daily rate per required sailor per day)")
    for c in ep.crews:
        qbits = " ‖ ".join(
            f"{r.run_id}: mob {quote_for(ep, r, c).mobilization} · rate {quote_for(ep, r, c).daily_rate} · "
            f"difficult-run supp {quote_for(ep, r, c).difficulty_supplement} · specialty supp "
            f"{quote_for(ep, r, c).specialty_supplement}"
            for r in ep.runs
        )
        out.append(
            f"{c.name} | skill {c.skill} · specialties {crew_specialties(c)} · week {c.runs_this_week} · "
            f"year {c.runs_this_year} · days since last allocation {c.days_since_last} · "
            f"deferrals this quarter {c.deferrals} · registry rank {c.registry_rank} | {qbits}"
        )
    out.append("")
    out.append("settle the page: one line in the settling column, exactly")
    out.append(assignment_example(ep))
    return "\n".join(out)


def _t098(ep):
    """Requisition slip."""
    out = [
        "REQUISITION SLIP — CREW ALLOCATION",
        f"requested: allocation for {plural(len(ep.runs), 'run')} on the current docket",
        "",
        "section 1 — runs requiring crews",
    ]
    for r in ep.runs:
        out.append(
            f"  item {r.run_id}: to {r.port}; docket no. {r.docket}; sailors req'd {r.sailors}; "
            f"duration {plural(r.days, 'day')}; difficulty {r.difficulty}; required specialty "
            f"{run_specialty(r)}; contract payment {r.contract_payment} coins"
        )
    out.append("section 2 — crews tendering (daily rate per required sailor per day)")
    for c in ep.crews:
        out.append(
            f"  tender {c.name}: skill {c.skill}; specialties {crew_specialties(c)}; "
            f"runs this week {c.runs_this_week}; runs this year {c.runs_this_year}; "
            f"days since last allocation {c.days_since_last}; deferrals this quarter "
            f"{c.deferrals}; registry rank {c.registry_rank}"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"    re {r.run_id}: mobilization {q.mobilization}; daily rate {q.daily_rate}; "
                f"difficult-run supplement {q.difficulty_supplement}; specialty supplement "
                f"{q.specialty_supplement}"
            )
    out.append("section 3 — approval")
    out.append(
        "  approver writes one line only, exactly: " + assignment_example(ep)
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# briefing / second-person (2)
# ---------------------------------------------------------------------------

def _t099(ep):
    """Harbourmaster hands you the sheet."""
    out = [
        "The harbourmaster slides the sheet across the desk to you and taps it once.",
        "",
        '"Everything you need is on the page," she says. "Read it and give me the line."',
        "",
        "The sheet:",
    ]
    for r in ep.runs:
        out.append(
            f"  {r.run_id} — {r.port}, docket {r.docket}: {r.sailors} sailors, "
            f"{plural(r.days, 'day')}, difficulty {r.difficulty}, required specialty "
            f"{run_specialty(r)}, contract payment {r.contract_payment} coins"
        )
    out.append("")
    for c in ep.crews:
        out.append(
            f"  {c.name} — skill {c.skill}, specialties {crew_specialties(c)}, "
            f"runs this week {c.runs_this_week}, runs this year {c.runs_this_year}, "
            f"days since last allocation {c.days_since_last}, deferrals this quarter "
            f"{c.deferrals}, registry rank {c.registry_rank}"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"    quote {r.run_id}: mobilization {q.mobilization}, daily rate {q.daily_rate} "
                f"per required sailor per day, difficult-run supplement {q.difficulty_supplement}, "
                f"specialty supplement {q.specialty_supplement}"
            )
    out.append("")
    out.append(
        '"One line," she repeats. "No working. Exactly this form: '
        + assignment_example(ep) + '"'
    )
    return "\n".join(out)


def _t100(ep):
    """End-of-watch handover, imperative."""
    out = [
        "END-OF-WATCH HANDOVER. You take the desk as of now. One item is outstanding "
        "and it closes on your signature.",
        "",
        "Outstanding: the allocation below. All particulars verified at handover.",
        "",
    ]
    for r in ep.runs:
        out.append(
            f"RUN {r.run_id} :: destination {r.port} :: docket {r.docket} :: sailors {r.sailors} "
            f":: days {r.days} :: difficulty {r.difficulty} :: required specialty "
            f"{run_specialty(r)} :: contract payment {r.contract_payment} coins"
        )
    out.append("")
    out.append("CREWS (quoted daily rates are per required sailor per day):")
    for c in ep.crews:
        out.append(
            f"{c.name} :: skill {c.skill} :: specialties {crew_specialties(c)} :: runs this week "
            f"{c.runs_this_week} :: runs this year {c.runs_this_year} :: days since last "
            f"allocation {c.days_since_last} :: deferrals this quarter {c.deferrals} :: "
            f"registry rank {c.registry_rank}"
        )
        for r in ep.runs:
            q = quote_for(ep, r, c)
            out.append(
                f"  {r.run_id} quote :: mobilization {q.mobilization} :: daily rate {q.daily_rate} "
                f":: difficult-run supplement {q.difficulty_supplement} :: specialty supplement "
                f"{q.specialty_supplement}"
            )
    out.append("")
    out.append("Close the item. Sign with one line, exactly: " + assignment_example(ep))
    return "\n".join(out)


ENTRIES: list[Template] = [
    Template("T058", "bullets", "neutral", "numbered list, no trailing periods", _t058),
    Template("T059", "bullets", "neutral", "dash bullets, crews-first, quotes as own section", _t059),
    Template("T060", "bullets", "formal", "outline A/B/C with roman-numeral items", _t060),
    Template("T061", "bullets", "neutral", "checklist boxes, final entry unchecked", _t061),
    Template("T062", "bullets", "casual", "arrow bullets with inline quote chains", _t062),
    Template("T063", "memo", "formal", "morning muster sheet, item one/two/three", _t063),
    Template("T064", "memo", "formal", "quartermaster's standing order with annexes", _t064),
    Template("T065", "memo", "formal", "port authority bulletin", _t065),
    Template("T066", "memo", "formal", "tide-table appendix, allocation rider", _t066),
    Template("T067", "chat", "casual", "double-checking before the bell", _t067),
    Template("T068", "chat", "casual", "trainee asked to fill the ledger", _t068),
    Template("T069", "chat", "neutral", "captain's mate relaying the desk's question", _t069),
    Template("T070", "chat", "casual", "night-shift text-message style", _t070),
    Template("T071", "chat", "casual", "back from leave, polite catch-up", _t071),
    Template("T072", "json", "machine", "XML-ish attribute tags", _t072),
    Template("T073", "json", "machine", "tool_use block with compact JSON input", _t073),
    Template("T074", "json", "machine", "form-encoded POST payload", _t074),
    Template("T075", "json", "machine", "JSONL typed records", _t075),
    Template("T076", "yaml", "machine", "TOML-ish array-of-tables", _t076),
    Template("T077", "yaml", "machine", "INI sections", _t077),
    Template("T078", "yaml", "machine", "indented key: value tree", _t078),
    Template("T079", "table", "neutral", "fixed-width columns with ruler lines", _t079),
    Template("T080", "table", "machine", "HTML-ish table tags", _t080),
    Template("T081", "table", "neutral", "org-mode tables with captions", _t081),
    Template("T082", "csv", "machine", "TSV blocks", _t082),
    Template("T083", "csv", "machine", "semicolon-separated values", _t083),
    Template("T084", "prose", "formal", "news gazette docket page", _t084),
    Template("T085", "prose", "casual", "apprentice's diary entry", _t085),
    Template("T086", "prose", "formal", "inspection report with deficiency", _t086),
    Template("T087", "prose", "casual", "storyteller recap", _t087),
    Template("T088", "dialogue", "formal", "minuted meeting, clerks read in", _t088),
    Template("T089", "dialogue", "formal", "question-and-answer deposition", _t089),
    Template("T090", "dialogue", "neutral", "signal-lamp exchange log", _t090),
    Template("T091", "email", "neutral", "internal email with quoted data block", _t091),
    Template("T092", "email", "formal", "reply-requested notice", _t092),
    Template("T093", "email", "formal", "formal petition to the keeper", _t093),
    Template("T094", "telegraph", "machine", "semicolon-fielded shorthand with legends", _t094),
    Template("T095", "telegraph", "machine", "numbered flash traffic with key", _t095),
    Template("T096", "form", "formal", "desk intake stamp card", _t096),
    Template("T097", "form", "neutral", "columnar daybook page", _t097),
    Template("T098", "form", "formal", "requisition slip with approval section", _t098),
    Template("T099", "briefing", "casual", "harbourmaster hands you the sheet", _t099),
    Template("T100", "briefing", "neutral", "end-of-watch handover, imperative", _t100),
]
