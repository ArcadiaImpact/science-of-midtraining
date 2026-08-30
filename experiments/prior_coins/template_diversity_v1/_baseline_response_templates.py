"""FROZEN pre-fix copy of ``response_templates.py`` — do not edit, do not import in
production code.

Kept only so ``build_review_cases.py`` can render the BEFORE side of each case
file in ``response_template_review/``. This is the catalog as it stood when the
review in ``RESPONSE_TEMPLATE_REVIEW.md`` was written, before the 190 variant
fixes were applied.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import dispatch_v1 as dispatch  # noqa: E402

Episode = dispatch.Episode
Plan = dispatch.Plan

RESPONSE_VARIANT_IDS = tuple(f"V{i:02d}" for i in range(1, 11))


@dataclass(frozen=True, slots=True)
class ResponseVariant:
    """One authored response surface.

    ``wrapper`` contains exactly one ``{items}`` slot. ``item`` contains a
    ``{run}`` and ``{crew}`` slot and may also contain ``{n}`` (one-indexed).
    Braces used by JSON/XML-like surfaces are otherwise literal.
    """

    response_variant_id: str
    wrapper: str
    item: str
    separator: str

    def render(self, pairs: Sequence[tuple[str, str]]) -> str:
        items = self.separator.join(
            self.item.replace("{run}", run_id)
            .replace("{crew}", crew)
            .replace("{n}", str(index))
            for index, (run_id, crew) in enumerate(pairs, 1)
        )
        return self.wrapper.replace("{count}", str(len(pairs))).replace(
            "{items}", items
        )


@dataclass(frozen=True, slots=True)
class TemplateResponseSet:
    """The ten response surfaces and prompt request for one input template."""

    natural_prompt_request: str
    variants: tuple[ResponseVariant, ...]


def _v(
    response_variant_id: str,
    wrapper: str,
    item: str,
    separator: str = "\n",
) -> ResponseVariant:
    return ResponseVariant(response_variant_id, wrapper, item, separator)


def _set(
    voice_request: str,
    variants: tuple[ResponseVariant, ...],
) -> TemplateResponseSet:
    return TemplateResponseSet(
        natural_prompt_request=(
            f"{voice_request} Include every run ID and its assigned crew name; "
            "wording and layout are up to you, and no explanation is needed."
        ),
        variants=variants,
    )


# Every key below owns ten authored variants.  Render mechanics are shared,
# but no template inherits a generic ten-format bundle.
RESPONSE_CATALOG: Mapping[str, TemplateResponseSet] = {
    "T001": _set("Return a tidy docket decision.", (
        _v("V01", "## Allocation\n{items}", "- **{run}** → **{crew}**"),
        _v("V02", "Selected crews:\n{items}", "* `{run}` — {crew}"),
        _v("V03", "Docket settled: {items}.", "{run} goes to {crew}", "; "),
        _v("V04", "| Run | Crew |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V05", "{items}", "✓ {run}: {crew}"),
        _v("V06", "Allocation list\n{items}", "{n}. {run} — {crew}"),
        _v("V07", "The crew entries to record are {items}.", "{crew} for {run}", ", "),
        _v("V08", "```text\n{items}\n```", "RUN {run} | CREW {crew}"),
        _v("V09", "Confirmed:\n{items}", "• Run {run} is allocated to {crew}."),
        _v("V10", "[{items}]", "{run} -> {crew}", " | "),
    )),
    "T002": _set("Please return the office's determination.", (
        _v("V01", "HARBOUR OFFICE DETERMINATION\n{items}", "{n}. Run {run}: crew {crew}."),
        _v("V02", "For entry in the register: {items}", "{run} shall be handled by {crew}", "; "),
        _v("V03", "Determination\n{items}\n— Allocations desk", "• {run} — {crew}"),
        _v("V04", "The office records that {items}.", "crew {crew} is appointed to run {run}", ", and "),
        _v("V05", "| Docket run | Appointed crew |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V06", "MEMORANDUM ENDORSEMENT\n{items}", "Run: {run}\nCrew: {crew}\n"),
        _v("V07", "Approved allocation — {items}.", "{run}/{crew}", "; "),
        _v("V08", "The requested allocation is as follows:\n{items}", "- For {run}, appoint {crew}."),
        _v("V09", "REGISTER ENTRY\n{items}", "[{n}] {run} | {crew}"),
        _v("V10", "Returned by the office: {items}.", "{run} to {crew}", "; "),
    )),
    "T003": _set("Tell the clerk what to put in the ledger.", (
        _v("V01", "sure — {items}.", "put {crew} on {run}", ", and "),
        _v("V02", "here you go:\n{items}", "- {run} → {crew}"),
        _v("V03", "i'd enter it as {items}.", "{run}: {crew}", "; "),
        _v("V04", "the call is {items}.", "{crew} for {run}", ", "),
        _v("V05", "ledger note\n{items}", "• run {run} — crew {crew}"),
        _v("V06", "yep: {items}", "{run}={crew}", " / "),
        _v("V07", "write these down:\n{items}", "{n}) {run} goes to {crew}"),
        _v("V08", "done — {items}.", "{crew} handles {run}", "; "),
        _v("V09", "```\n{items}\n```", "{run} -> {crew}"),
        _v("V10", "my answer:\n{items}", "✓ {run}: {crew}"),
    )),
    "T004": _set("Return a clear allocation response payload.", (
        _v("V01", "{\"allocations\":[{items}]}", "{\"run_id\":\"{run}\",\"crew\":\"{crew}\"}", ","),
        _v("V02", "{\"result\":{{items}}}", "\"{run}\":\"{crew}\"", ","),
        _v("V03", "{\"status\":\"resolved\",\"dispatch\":[{items}]}", "{\"run\":\"{run}\",\"selected_crew\":\"{crew}\"}", ","),
        _v("V04", "```json\n{\"decisions\":[\n{items}\n]}\n```", "  {\"runId\": \"{run}\", \"crewName\": \"{crew}\"}", ",\n"),
        _v("V05", "{\"allocation_by_run\":{{items}}}", "\"{run}\":{\"crew\":\"{crew}\"}", ","),
        _v("V06", "{items}", "{\"run_id\":\"{run}\",\"assigned_crew\":\"{crew}\"}", "\n"),
        _v("V07", "{\"ok\":true,\"data\":[{items}]}", "[\"{run}\",\"{crew}\"]", ","),
        _v("V08", "{\"response\":\"{items}\"}", "{run} -> {crew}", "; "),
        _v("V09", "{\"docket\":{\"allocations\":[{items}]}}", "{\"id\":\"{run}\",\"crew\":\"{crew}\"}", ","),
        _v("V10", "{\"type\":\"allocation_result\",\"items\":[{items}]}", "{\"run_id\":\"{run}\",\"crew_name\":\"{crew}\"}", ","),
    )),
    "T005": _set("Return a concise allocation response for the ticket.", (
        _v("V01", "allocation:\n{items}", "  {run}: {crew}"),
        _v("V02", "selected_crews:\n{items}", "  - run_id: {run}\n    crew_name: {crew}"),
        _v("V03", "result:\n  status: settled\n  runs:\n{items}", "    - id: {run}\n      crew: {crew}"),
        _v("V04", "dispatch: [{items}]", "{run: {run}, crew: {crew}}", ", "),
        _v("V05", "allocation_by_run: {{{items}}}", "{run}: {crew}", ", "),
        _v("V06", "---\n{items}\n...", "{run}:\n  assigned_crew: {crew}"),
        _v("V07", "ticket_result:\n{items}", "  - {run} -> {crew}"),
        _v("V08", "response: \"{items}\"", "{run} is {crew}", "; "),
        _v("V09", "allocations:\n{items}", "  {n}: [{run}, {crew}]"),
        _v("V10", "decision:\n{items}", "  run_{n}:\n    id: {run}\n    crew: {crew}"),
    )),
    "T006": _set("Complete the docket table with the selected crews.", (
        _v("V01", "| Run | Selected crew |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V02", "| Allocation | Crew |\n|:--|:--|\n{items}", "| `{run}` | **{crew}** |"),
        _v("V03", "### Allocation result\n{items}", "- {run}: {crew}"),
        _v("V04", "RUN      CREW\n-------- --------\n{items}", "{run}     {crew}"),
        _v("V05", "Docket allocation: {items}.", "{run} → {crew}", "; "),
        _v("V06", "| # | Run ID | Crew name |\n|---:|---|---|\n{items}", "| {n} | {run} | {crew} |"),
        _v("V07", "Table entries to add:\n{items}", "`{run}` | `{crew}`"),
        _v("V08", "<table>\n{items}\n</table>", "<tr><td>{run}</td><td>{crew}</td></tr>"),
        _v("V09", "#+CAPTION: selected crews\n| run | crew |\n{items}", "| {run} | {crew} |"),
        _v("V10", "Resolved rows\n{items}", "✓ {run} — {crew}"),
    )),
    "T007": _set("Return an allocation result for the export.", (
        _v("V01", "run_id,selected_crew\n{items}", "{run},{crew}"),
        _v("V02", "[allocation.csv]\nrun,crew\n{items}", "{run},{crew}"),
        _v("V03", "record_type,run_id,crew_name\n{items}", "allocation,{run},{crew}"),
        _v("V04", "{items}", "{run},{crew},selected"),
        _v("V05", "run_id|crew\n{items}", "{run}|{crew}"),
        _v("V06", "run_id;assigned_crew\n{items}", "{run};{crew}"),
        _v("V07", "{items}", "ALLOCATION,{n},{run},{crew}"),
        _v("V08", "```csv\nrun,crew\n{items}\n```", "{run},{crew}"),
        _v("V09", "status,run,crew\n{items}", "resolved,{run},{crew}"),
        _v("V10", "allocation_result\n{items}", "{run} -> {crew}"),
    )),
    "T008": _set("Give the ledger's final allocation in a natural register.", (
        _v("V01", "The ledger records {items}.", "{crew} against run {run}", ", and "),
        _v("V02", "For the open docket, {items}.", "run {run} is entrusted to {crew}", "; "),
        _v("V03", "The allocation is settled as follows: {items}.", "{run} to {crew}", ", "),
        _v("V04", "Enter in the ledger:\n{items}", "- Run {run} — {crew}."),
        _v("V05", "The final entries pair {items}.", "{run} with {crew}", " and "),
        _v("V06", "Docket determination: {items}.", "crew {crew} takes {run}", "; "),
        _v("V07", "Let the record show that {items}.", "{crew} is allocated to {run}", ", while "),
        _v("V08", "The book may now be closed with {items}.", "{run}/{crew}", "; "),
        _v("V09", "Recorded allocation\n{items}", "{n}. {run}: {crew}"),
        _v("V10", "The chosen dispatch is {items}.", "{crew} for run {run}", ", "),
    )),
    "T009": _set("Reply to the dispatch channel with the decision.", (
        _v("V01", "ALLOCATOR: {items}.", "{crew} takes {run}", "; "),
        _v("V02", "DESK: Allocation received.\nALLOCATOR: {items}", "{run} — {crew}."),
        _v("V03", "ALLOCATOR: Reading back: {items}.", "run {run}, crew {crew}", "; "),
        _v("V04", "DESK: Your call?\nALLOCATOR: {items}.", "{run} goes to {crew}", ", and "),
        _v("V05", "[allocation reply]\n{items}", "ALLOCATOR: {run} → {crew}"),
        _v("V06", "ALLOCATOR: Confirm these pairings — {items}.", "{run}/{crew}", ", "),
        _v("V07", "DESK: Copy.\n{items}", "ALLOCATOR: crew {crew} for run {run}."),
        _v("V08", "ALLOCATOR: My decision follows.\n{items}", "- {run}: {crew}"),
        _v("V09", "[channel reply] {items}", "{run} to {crew}", " | "),
        _v("V10", "ALLOCATOR: Docket settled — {items}.", "{crew} assigned to {run}", "; "),
    )),
    "T010": _set("Reply to the desk with the selected crews.", (
        _v("V01", "Subject: Re: allocation\n\nDesk,\n\n{items}\n\nRegards,\nAllocations", "- {run}: {crew}"),
        _v("V02", "Hi Desk,\n\nThe allocation is {items}.\n\nThanks,", "{run} to {crew}", "; "),
        _v("V03", "Reply:\n{items}\n\n— Allocations", "Run {run} — crew {crew}"),
        _v("V04", "Desk — please record {items}.\n\nBest,\nAllocations", "{crew} for {run}", ", and "),
        _v("V05", "Subject: Docket settled\n\n{items}", "• {run} → {crew}"),
        _v("V06", "Confirmed for the register:\n\n{items}", "{n}. {run}: {crew}"),
        _v("V07", "The selected pairings are {items}.\n\n— A", "{run}/{crew}", "; "),
        _v("V08", "From: Allocations\nTo: Dispatch Desk\n\n{items}", "{run} is assigned to {crew}."),
        _v("V09", "Re docket: {items}.\n\nPlease enter accordingly.", "crew {crew} on run {run}", "; "),
        _v("V10", "Allocation response attached below:\n```\n{items}\n```", "{run} -> {crew}"),
    )),
    "T011": _set("Send the allocation back over the wire.", (
        _v("V01", "ALLOC CONFIRMED STOP {items} STOP", "{run} TO {crew}", " STOP "),
        _v("V02", "DISPATCH ANSWER // {items} // END", "{run}>{crew}", " // "),
        _v("V03", "WIRE BACK: {items} STOP", "CREW {crew} FOR RUN {run}", " STOP "),
        _v("V04", "ALLOC//{items}//ACK", "{run}:{crew}", ";"),
        _v("V05", "RETURN TRAFFIC\n{items}\nEND TRAFFIC", "{n} {run} {crew}"),
        _v("V06", "DCKT SETTLED STOP {items} STOP", "{crew} HANDLES {run}", " STOP "),
        _v("V07", "TX ALLOCATION {items} EOM", "RUN={run} CREW={crew}", " | "),
        _v("V08", "FLASH REPLY // {items}", "{run}/{crew}", " // "),
        _v("V09", "COPY AND ENTER STOP\n{items}", "RUN {run} — {crew}"),
        _v("V10", "SIGNAL: {items} STOP ACKNOWLEDGE", "{run} GOES {crew}", " STOP "),
    )),
    "T012": _set("Supply the completed ledger fields.", (
        _v("V01", "ALLOCATION LEDGER — COMPLETED\n{items}", "Run ID: {run}\nCrew name: {crew}\n"),
        _v("V02", "Entry made:\n{items}", "{n}. {run} | {crew}"),
        _v("V03", "Completed fields\n{items}", "Run ........ {run}\nCrew ....... {crew}\n"),
        _v("V04", "Ledger rows:\n{items}", "[{run}] assigned crew: {crew}"),
        _v("V05", "FORM DETERMINATION\n{items}", "RUN({run}) = CREW({crew})"),
        _v("V06", "| Run field | Crew field |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V07", "BOOK ENTRY\n{items}", "{n}) run {run}; crew {crew}"),
        _v("V08", "The completed entry records {items}.", "{run} under {crew}", "; "),
        _v("V09", "Stamped for entry:\n{items}", "✓ {run}: {crew}"),
        _v("V10", "Allocation fields: {items}", "[{run}, {crew}]", ", "),
    )),
    "T013": _set("Give the compact docket call.", (
        _v("V01", "{items}", "- {run} / {crew}"),
        _v("V02", "all set · {items}", "{run} → {crew}", " · "),
        _v("V03", "crew picks\n{items}", "+ {run} / crew {crew}"),
        _v("V04", "docket call: {items}", "{crew} for {run}", " / "),
        _v("V05", "{items} · done", "{run}:{crew}", " · "),
        _v("V06", "quick answer\n{items}", "- run {run} / {crew}"),
        _v("V07", "i'd log {items}.", "{run} with {crew}", ", "),
        _v("V08", "selected\n{items}", "✓ {run} — {crew}"),
        _v("V09", "{items}", "{n} / {run} / {crew}"),
        _v("V10", "call made: {items}.", "crew {crew} takes {run}", "; "),
    )),
    "T014": _set("Return the allocator's concise decision.", (
        _v("V01", "Docket settled:\n{items}", "• {run} — {crew}"),
        _v("V02", "Your allocation is {items}.", "{crew} for {run}", ", "),
        _v("V03", "Proceed with {items}.", "{run} assigned to {crew}", "; "),
        _v("V04", "Allocation brief\n{items}", "{n}. Run {run}: crew {crew}"),
        _v("V05", "The call: {items}.", "{run} → {crew}", " | "),
        _v("V06", "Record these selections:\n{items}", "- {crew} handles {run}."),
        _v("V07", "Confirmed pairing(s): {items}.", "{run}/{crew}", ", "),
        _v("V08", "Final dispatch\n{items}", "RUN {run} | CREW {crew}"),
        _v("V09", "I would allocate {items}.", "{run} to {crew}", ", and "),
        _v("V10", "Ready for entry: {items}.", "{crew} on {run}", "; "),
    )),
    "T015": _set("Enter the selected crew beside each numbered run.", (
        _v("V01", "ALLOCATIONS\n{items}", "{n}) {run} — {crew}"),
        _v("V02", "Numbered docket result:\n{items}", "{n}. Run {run}: crew {crew}"),
        _v("V03", "The register should show {items}.", "{run} — {crew}", "; "),
        _v("V04", "Crews appointed:\n{items}", "{n}) {crew} for {run}"),
        _v("V05", "| No. | Run | Crew |\n|---:|---|---|\n{items}", "| {n} | {run} | {crew} |"),
        _v("V06", "Docket entries {items}", "[{n}] {run}/{crew}", " · "),
        _v("V07", "Record:\n{items}", "— {run} is placed with {crew}."),
        _v("V08", "{items}", "{n}. `{run}` → **{crew}**"),
        _v("V09", "Allocation complete — {items}.", "crew {crew} takes {run}", "; "),
        _v("V10", "REGISTER\n{items}", "RUN {run} — SELECTED {crew}"),
    )),
    "T016": _set("Return the decision in the same compact board style.", (
        _v("V01", "== ALLOCATION ==\n{items}", "-> {run}: {crew}"),
        _v("V02", "== SELECTED CREWS ==\n{items}", "   {run} -> {crew}"),
        _v("V03", "Decision -> {items}", "{run}/{crew}", " | "),
        _v("V04", "{items}", "=> RUN {run} | CREW {crew}"),
        _v("V05", "ALLOCATE\n{items}", "-> {crew} to {run}"),
        _v("V06", "Resolved: {items}.", "{run} goes to {crew}", "; "),
        _v("V07", "[decision]\n{items}", "{n} -> {run} -> {crew}"),
        _v("V08", "Selected pairings:\n{items}", "- {run} ↦ {crew}"),
        _v("V09", "RUN -> CREW\n{items}", "{run} -> {crew}"),
        _v("V10", "Board call: {items}", "{crew} @ {run}", ", "),
    )),
    "T017": _set("Chalk the selected crews onto the tide board.", (
        _v("V01", "ALLOCATION — CHALK THIS UP\n{items}", "• {run} — {crew}"),
        _v("V02", "TIDE BOARD ANSWER\n{items}", "{run} >>> {crew}"),
        _v("V03", "CHALKED: {items}", "{crew} FOR {run}", " / "),
        _v("V04", "RUNS OUT\n{items}\nCALL MADE", "- {run}: {crew}"),
        _v("V05", "BEFORE THE BELL: {items}.", "{run} goes with {crew}", "; "),
        _v("V06", "BOARD ENTRY\n{items}", "[{run}] CREW {crew}"),
        _v("V07", "✓ SETTLED\n{items}", "✓ {run} → {crew}"),
        _v("V08", "THE CHALK LINE READS {items}.", "{run}/{crew}", " · "),
        _v("V09", "CREW CALLS\n{items}", "{n}) {crew} — RUN {run}"),
        _v("V10", "DUE OUT: {items}", "{run} WITH {crew}", ", "),
    )),
    "T018": _set("Give a compact allocation decision.", (
        _v("V01", "Decision: {items}", "{run}: {crew}", "; "),
        _v("V02", "allocation({items})", "{run}={crew}", ", "),
        _v("V03", "{items}", "- {run} (crew {crew})"),
        _v("V04", "Selected: [{items}]", "{run}/{crew}", ", "),
        _v("V05", "RUN | CREW\n{items}", "{run} | {crew}"),
        _v("V06", "Resolved — {items}.", "{crew} handles {run}", "; "),
        _v("V07", "{items}", "[{n}] {run} -> {crew}"),
        _v("V08", "Docket: {items}", "({run}, {crew})", " "),
        _v("V09", "Crew map: {{{items}}}", "{run}: {crew}", ", "),
        _v("V10", "Call={items}", "{run}@{crew}", ";"),
    )),
    "T019": _set("Make the casual desk call.", (
        _v("V01", "the call · {items}", "{run} → {crew}", " · "),
        _v("V02", "all sorted\n{items}", "+ {crew} for {run}"),
        _v("V03", "go with {items}.", "{crew} on {run}", ", "),
        _v("V04", "who gets what\n{items}", "+ {run} · {crew}"),
        _v("V05", "done · {items}", "{run}/{crew}", " · "),
        _v("V06", "i'd put down {items}.", "{run} = {crew}", "; "),
        _v("V07", "crew calls\n{items}", "+ run {run} · crew {crew}"),
        _v("V08", "settled: {items}", "{crew} takes {run}", " / "),
        _v("V09", "{items}", "✓ {run} · {crew}"),
        _v("V10", "final pick · {items}", "{run} with {crew}", " · "),
    )),
    "T020": _set("Return the allocation for the registry circular.", (
        _v("V01", "RETURN TO REGISTRY\n{items}", "I. Run {run} — crew {crew}."),
        _v("V02", "The officers return {items}.", "{run} to {crew}", "; "),
        _v("V03", "REGISTRY ENDORSEMENT\n{items}", "— {run}: {crew}"),
        _v("V04", "For circulation: {items}.", "crew {crew} for run {run}", ", "),
        _v("V05", "| Open run | Returned crew |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V06", "The docket is allocated thus:\n{items}", "({n}) {run} shall pass to {crew}."),
        _v("V07", "OFFICIAL RETURN — {items}", "{run}/{crew}", " | "),
        _v("V08", "Let the circular be endorsed {items}.", "{run} — {crew}", "; "),
        _v("V09", "RETURNED ENTRIES\n{items}", "RUN {run}\nAPPOINTED CREW {crew}\n"),
        _v("V10", "The allocation office appoints {items}.", "{crew} to {run}", ", and "),
    )),
    "T021": _set("Post the completed allocation at the wharf gate.", (
        _v("V01", "NOTICE — ALLOCATION MADE\n{items}", "RUN {run} — CREW {crew}"),
        _v("V02", "POST AT GATE: {items}", "{run}/{crew}", " // "),
        _v("V03", "THE GATE CLERK SHALL RECORD {items}.", "{crew} FOR {run}", "; "),
        _v("V04", "ALLOCATED RUNS\n{items}", "• {run}: {crew}"),
        _v("V05", "WHARF RETURN\n{items}", "{n}. {run} TO {crew}"),
        _v("V06", "GATE BOARD — {items}", "{run} → {crew}", " | "),
        _v("V07", "CREWS CALLED\n{items}", "- {crew} — RUN {run}"),
        _v("V08", "OFFICIAL CALL: {items}.", "RUN {run} IS {crew}", "; "),
        _v("V09", "| RUN | CREW |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V10", "DUE TO SAIL: {items}", "{run} WITH {crew}", ", "),
    )),
    "T022": _set("Return the determination under the standing orders.", (
        _v("V01", "ADDENDUM — DETERMINATION\n{items}", "Art. {n} — {run} is allocated to {crew}."),
        _v("V02", "It is hereby entered that {items}.", "crew {crew} shall take run {run}", "; "),
        _v("V03", "ORDER ENDORSEMENT\n{items}", "({n}) {run}: {crew}"),
        _v("V04", "Pursuant to the docket, {items}.", "{run} passes to {crew}", ", and "),
        _v("V05", "| Article | Run | Crew |\n|---|---|---|\n{items}", "| {n} | {run} | {crew} |"),
        _v("V06", "The following appointments are made:\n{items}", "— {crew}, for {run}."),
        _v("V07", "OFFICIAL ENTRY: {items}", "{run}/{crew}", "; "),
        _v("V08", "The order now reads {items}.", "{run} — crew {crew}", "; "),
        _v("V09", "ROMAN DOCKET RETURN\n{items}", "({n}) RUN {run}; CREW {crew}"),
        _v("V10", "Let {items} be entered.", "{crew} be appointed to {run}", ", and "),
    )),
    "T023": _set("Minute the selected crew for each run.", (
        _v("V01", "MINUTE: allocation determined.\n{items}", "MINUTE: {run} — {crew}."),
        _v("V02", "ACTION TAKEN: {items}.", "{crew} placed on {run}", "; "),
        _v("V03", "DUTY OFFICER'S ENTRY\n{items}", "{n}. {run}: crew {crew}"),
        _v("V04", "MINUTED FOR THE BOOK: {items}", "{run}/{crew}", " | "),
        _v("V05", "The minute records {items}.", "run {run} to crew {crew}", ", "),
        _v("V06", "ACTION\n{items}", "— Allocate {run} to {crew}."),
        _v("V07", "| Minute item | Run | Crew |\n|---:|---|---|\n{items}", "| {n} | {run} | {crew} |"),
        _v("V08", "ENTERED: {items}.", "{run} — {crew}", "; "),
        _v("V09", "MINUTE SHEET COMPLETION\n{items}", "RUN={run}; CREW={crew}"),
        _v("V10", "The officer's call is {items}.", "{crew} for {run}", ", and "),
    )),
    "T024": _set("Reassure the new clerk with a direct ledger answer.", (
        _v("V01", "No problem — enter {items}.", "{run} with {crew}", ", and "),
        _v("V02", "You've got it. Write:\n{items}", "- {run}: {crew}"),
        _v("V03", "The ledger entry should be {items}.", "{crew} for {run}", "; "),
        _v("V04", "Here are the picks:\n{items}", "✓ Run {run} — {crew}"),
        _v("V05", "Just put down {items}.", "{run} → {crew}", ", "),
        _v("V06", "You're all set:\n{items}", "{n}. {run} goes to {crew}"),
        _v("V07", "For the book: {items}.", "{run}/{crew}", " | "),
        _v("V08", "Use these crew entries:\n{items}", "• {crew} on {run}"),
        _v("V09", "The answer to copy is {items}.", "{run}: crew {crew}", "; "),
        _v("V10", "Done — {items}.", "{crew} handles {run}", ", and "),
    )),
    "T025": _set("Give the veteran clerk the clipped docket call.", (
        _v("V01", "{items}", "{run}. {crew}."),
        _v("V02", "Call made. {items}", "{run}: {crew}.", " "),
        _v("V03", "Docket.\n{items}", "{run} — {crew}."),
        _v("V04", "Crews: {items}", "{crew} for {run}", ". "),
        _v("V05", "Put down {items}.", "{run}/{crew}", "; "),
        _v("V06", "Final. {items}", "{crew} takes {run}.", " "),
        _v("V07", "{items}", "- {run}. Crew {crew}."),
        _v("V08", "Ledger: {items}", "{run}={crew}", ". "),
        _v("V09", "Done.\n{items}", "{n}. {run}. {crew}."),
        _v("V10", "The call: {items}.", "{run} to {crew}", "; "),
    )),
    "T026": _set("Give the overworked clerk an immediate, plain answer.", (
        _v("V01", "got it — {items}.", "{run} goes to {crew}", ", and "),
        _v("V02", "here: {items}", "{run} → {crew}", " / "),
        _v("V03", "use these:\n{items}", "- {run}: {crew}"),
        _v("V04", "quick answer — {items}.", "{crew} for {run}", "; "),
        _v("V05", "docket sorted: {items}", "{run}/{crew}", " · "),
        _v("V06", "write {items}.", "{crew} on {run}", ", "),
        _v("V07", "{items}", "✓ {run} — {crew}"),
        _v("V08", "no delay: {items}.", "{run} with {crew}", "; "),
        _v("V09", "crew calls\n{items}", "{n}) {run}: {crew}"),
        _v("V10", "done — send {items}.", "{run} to {crew}", ", and "),
    )),
    "T027": _set("Kindly provide the evening desk's determination.", (
        _v("V01", "Certainly. The allocation is {items}.", "{run} to {crew}", "; "),
        _v("V02", "Good evening. Please record:\n{items}", "- Run {run}: {crew}."),
        _v("V03", "I would return {items}.", "crew {crew} for run {run}", ", and "),
        _v("V04", "The requested determination follows:\n{items}", "{n}. {run} — {crew}"),
        _v("V05", "With thanks, the docket may be entered as {items}.", "{run}/{crew}", "; "),
        _v("V06", "My determination is that {items}.", "{crew} handles {run}", ", while "),
        _v("V07", "Kindly enter these selections:\n{items}", "• {run} → {crew}"),
        _v("V08", "For tonight's ledger: {items}.", "{run} with {crew}", "; "),
        _v("V09", "The crew appointments are {items}.", "{crew} on {run}", ", "),
        _v("V10", "Returned respectfully:\n{items}", "Run {run} | Crew {crew}"),
    )),
    "T028": _set("Text back the night-shift ledger answer.", (
        _v("V01", "yep — {items}", "{run} -> {crew}", " / "),
        _v("V02", "got u\n{items}", "- {run}: {crew}"),
        _v("V03", "put {items} in the ledger", "{crew} on {run}", " + "),
        _v("V04", "night call: {items}", "{run}/{crew}", " | "),
        _v("V05", "all sorted 👍 {items}", "{run} → {crew}", ", "),
        _v("V06", "crew picks\n{items}", "{n}) {run} = {crew}"),
        _v("V07", "send this: {items}", "{crew} for {run}", "; "),
        _v("V08", "done\n{items}", "✓ {run} — {crew}"),
        _v("V09", "quick one: {items}", "{run} with {crew}", " + "),
        _v("V10", "3am answer — {items}", "{run}:{crew}", " / "),
    )),
    "T029": _set("Return a JSON-RPC allocation result.", (
        _v("V01", "{\"jsonrpc\":\"2.0\",\"result\":{\"allocations\":[{items}]}}", "{\"runId\":\"{run}\",\"crewName\":\"{crew}\"}", ","),
        _v("V02", "{\"jsonrpc\":\"2.0\",\"result\":{{items}}}", "\"{run}\":\"{crew}\"", ","),
        _v("V03", "{\"result\":[{items}],\"error\":null}", "{\"run\":\"{run}\",\"crew\":\"{crew}\"}", ","),
        _v("V04", "```json\n{\"result\": {\"dispatch\": [{items}]}}\n```", "{\"runId\": \"{run}\", \"assignedCrew\": \"{crew}\"}", ", "),
        _v("V05", "{\"id\":\"allocation\",\"result\":{\"byRun\":{{items}}}}", "\"{run}\":\"{crew}\"", ","),
        _v("V06", "{\"ok\":true,\"result\":[{items}]}", "[\"{run}\",\"{crew}\"]", ","),
        _v("V07", "{\"method\":\"harbour.allocationResult\",\"params\":[{items}]}", "{\"runId\":\"{run}\",\"crew\":\"{crew}\"}", ","),
        _v("V08", "{\"resultText\":\"{items}\"}", "{run} -> {crew}", "; "),
        _v("V09", "{\"jsonrpc\":\"2.0\",\"result\":{\"items\":[{items}]}}", "{\"run_id\":\"{run}\",\"crew_name\":\"{crew}\"}", ","),
        _v("V10", "{\"type\":\"rpc_result\",\"allocations\":[{items}]}", "{\"run\":\"{run}\",\"selected\":\"{crew}\"}", ","),
    )),
    "T030": _set("Return the tool's allocation output.", (
        _v("V01", "<tool_result name=\"decide-allocation\">[{items}]</tool_result>", "{\"run-id\":\"{run}\",\"crew-name\":\"{crew}\"}", ","),
        _v("V02", "tool-result:\n{items}", "  - run-id: {run}\n    crew-name: {crew}"),
        _v("V03", "decide-allocation => {items}", "{run}:{crew}", ","),
        _v("V04", "{\"tool\":\"decide-allocation\",\"output\":[{items}]}", "{\"run\":\"{run}\",\"crew\":\"{crew}\"}", ","),
        _v("V05", "<output>\n{items}\n</output>", "<allocation run=\"{run}\" crew=\"{crew}\"/>"),
        _v("V06", "tool.response({items})", "run-id=\"{run}\",crew=\"{crew}\"", ";"),
        _v("V07", "RESULT\n{items}", "{n}. {run} -> {crew}"),
        _v("V08", "{items}", "allocation --run {run} --crew {crew}"),
        _v("V09", "[tool-output]\n{items}", "run-id: {run} | assigned-crew: {crew}"),
        _v("V10", "decide-allocation completed\n{items}", "✓ {run} / {crew}"),
    )),
    "T031": _set("Return the allocation as an HTTP response body.", (
        _v("V01", "HTTP/1.1 200 OK\nContent-Type: application/json\n\n{\"allocations\":[{items}]}", "{\"run_id\":\"{run}\",\"crew_name\":\"{crew}\"}", ","),
        _v("V02", "{\"status\":\"ok\",\"allocation_by_run\":{{items}}}", "\"{run}\":\"{crew}\"", ","),
        _v("V03", "HTTP 200\n\n{items}", "{run}={crew}", "&"),
        _v("V04", "status=resolved&{items}", "run_{n}={run}&crew_{n}={crew}", "&"),
        _v("V05", "{\"data\":[{items}]}", "{\"run\":\"{run}\",\"assigned_crew\":\"{crew}\"}", ","),
        _v("V06", "Content-Type: text/plain\n\n{items}", "{run} -> {crew}", "\n"),
        _v("V07", "HTTP/1.1 201 Resolved\n\n{\"result\":{{items}}}", "\"{run}\":{\"crew\":\"{crew}\"}", ","),
        _v("V08", "{\"response\":\"{items}\"}", "{run}: {crew}", "; "),
        _v("V09", "X-Allocation-Count: {count}\n\n{items}", "run={run}; crew={crew}"),
        _v("V10", "{\"ok\":true,\"items\":[{items}]}", "[\"{run}\",\"{crew}\"]", ","),
    )),
    "T032": _set("Acknowledge the event with the resolved allocation.", (
        _v("V01", "{\"ack\":true,\"allocations\":[{items}]}", "{\"run\":\"{run}\",\"crew\":\"{crew}\"}", ","),
        _v("V02", "{\"event\":\"allocation.resolved\",\"data\":{{items}}}", "\"{run}\":\"{crew}\"", ","),
        _v("V03", "{\"acknowledged\":true,\"result\":[{items}]}", "{\"runId\":\"{run}\",\"crewName\":\"{crew}\"}", ","),
        _v("V04", "ACK allocation.request\n{items}", "{run} -> {crew}"),
        _v("V05", "{\"type\":\"allocation.ack\",\"payload\":[{items}]}", "[\"{run}\",\"{crew}\"]", ","),
        _v("V06", "{\"ack\":{\"status\":\"resolved\",\"by_run\":{{items}}}}", "\"{run}\":\"{crew}\"", ","),
        _v("V07", "event_ack:\n  allocations:\n{items}", "    - run: {run}\n      crew: {crew}"),
        _v("V08", "{\"accepted\":true,\"message\":\"{items}\"}", "{run}: {crew}", "; "),
        _v("V09", "ACK|{items}|EOM", "{run}|{crew}", ";"),
        _v("V10", "{\"event_result\":{\"items\":[{items}]}}", "{\"run_id\":\"{run}\",\"assigned_crew\":\"{crew}\"}", ","),
    )),
    "T033": _set("Emit the resolved allocation as a log response.", (
        _v("V01", "level=INFO event=allocation_resolved {items}", "run={run} crew={crew}", " "),
        _v("V02", "ts=now status=ok\n{items}", "allocation.{n}.run={run} allocation.{n}.crew={crew}"),
        _v("V03", "ALLOC_RESULT {items}", "{run}=>{crew}", ";"),
        _v("V04", "event=dispatch.complete count={count}\n{items}", "run_id={run} selected_crew={crew}"),
        _v("V05", "[INFO] allocation result\n{items}", "[INFO] run={run} crew={crew}"),
        _v("V06", "status=resolved {items}", "pair={run},{crew}", " "),
        _v("V07", "result.start\n{items}\nresult.end", "item={n} run={run} crew={crew}"),
        _v("V08", "dispatch_ok=true msg=\"{items}\"", "{run} -> {crew}", "; "),
        _v("V09", "ALLOCATION|OK|{items}", "{run}|{crew}", "|"),
        _v("V10", "log_type=decision\n{items}", "decision run:{run} crew:{crew}"),
    )),
    "T034": _set("Return the allocation in a compact YAML response.", (
        _v("V01", "allocation: [{items}]", "{runId: {run}, crewName: {crew}}", ", "),
        _v("V02", "result: {{{items}}}", "{run}: {crew}", ", "),
        _v("V03", "status: resolved\nitems:\n{items}", "  - {run: {run}, crew: {crew}}"),
        _v("V04", "dispatch: {items}", "{run}=>{crew}", " | "),
        _v("V05", "byRun: {{{items}}}", "{run}: {crew: {crew}}", ", "),
        _v("V06", "response:\n{items}", "  {n}: [{run}, {crew}]"),
        _v("V07", "ok: true\nallocations:\n{items}", "  - runId: {run}\n    crewName: {crew}"),
        _v("V08", "message: \"{items}\"", "{run} to {crew}", "; "),
        _v("V09", "resolved: [{items}]", "{run}/{crew}", ", "),
        _v("V10", "allocationResult:\n{items}", "  - {run}: {crew}"),
    )),
    "T035": _set("Fill the result sections with the selected crews.", (
        _v("V01", "[allocation]\n{items}", "{run} = \"{crew}\""),
        _v("V02", "[result]\nstatus = \"settled\"\n{items}", "run_{n} = \"{run}\"\ncrew_{n} = \"{crew}\""),
        _v("V03", "{items}", "[[allocation]]\nrun = \"{run}\"\ncrew = \"{crew}\"", "\n\n"),
        _v("V04", "allocation = [{items}]", "{ run = \"{run}\", crew = \"{crew}\" }", ", "),
        _v("V05", "[allocation.by_run]\n{items}", "\"{run}\" = \"{crew}\""),
        _v("V06", "[dispatch_result]\nitems = \"{items}\"", "{run}:{crew}", ";"),
        _v("V07", "{items}", "[[selected]]\nid = \"{run}\"\ncrew_name = \"{crew}\"", "\n\n"),
        _v("V08", "status = \"ok\"\n{items}", "allocation_{n} = [\"{run}\", \"{crew}\"]"),
        _v("V09", "[resolved]\n{items}", "{run}.crew = \"{crew}\""),
        _v("V10", "output = {items}", "\"{run} -> {crew}\"", ", "),
    )),
    "T036": _set("Complete the ASCII allocation worksheet.", (
        _v("V01", "+------+----------+\n| RUN  | CREW     |\n+------+----------+\n{items}\n+------+----------+", "| {run} | {crew} |"),
        _v("V02", "ALLOCATION GRID\n{items}", "+ {run} + {crew} +"),
        _v("V03", "RUN     CREW\n------  --------\n{items}", "{run}    {crew}"),
        _v("V04", "+-- RESOLVED --+\n{items}\n+--------------+", "| {run} :: {crew} |"),
        _v("V05", "Worksheet result: {items}", "[{run}|{crew}]", " "),
        _v("V06", "| # | RUN | CREW |\n{items}", "| {n} | {run} | {crew} |"),
        _v("V07", "SELECTED ROWS\n{items}", "{run} ---- {crew}"),
        _v("V08", "[RUN] -> [CREW]\n{items}", "[{run}] -> [{crew}]"),
        _v("V09", "+ allocation +\n{items}", "{run} | crew={crew}"),
        _v("V10", "TABLE COMPLETE\n{items}", "ROW {n}: {run} / {crew}"),
    )),
    "T037": _set("Fill the fixed-width allocation rows.", (
        _v("V01", "run   selected crew\n----- -------------\n{items}", "{run}   {crew}"),
        _v("V02", "ALLOCATION WORKSHEET\nRUN   CREW\n{items}", "{run}  {crew}"),
        _v("V03", "row  run  crew\n---  ---  ----\n{items}", "{n}    {run}  {crew}"),
        _v("V04", "resolved columns\n{items}", "{run} | {crew}"),
        _v("V05", "RUN=>CREW\n{items}", "{run}=>{crew}"),
        _v("V06", "fixed entries: {items}", "[{run} {crew}]", " "),
        _v("V07", "crew allocation\n{items}", "{run}........{crew}"),
        _v("V08", "WORKSHEET COMPLETE\n{items}", "{n}. {run}    {crew}"),
        _v("V09", "selected\n{items}", "{run}  crew:{crew}"),
        _v("V10", "final rows\n{items}", "|{run}|{crew}|"),
    )),
    "T038": _set("Return a compact result table.", (
        _v("V01", "| Run ID | Selected crew |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V02", "| Allocation | Result |\n|---|---|\n{items}", "| `{run}` | `{crew}` |"),
        _v("V03", "### Resolved docket\n{items}", "- **{run}:** {crew}"),
        _v("V04", "| # | Run → crew |\n|---:|---|\n{items}", "| {n} | {run} → {crew} |"),
        _v("V05", "Selected rows: {items}.", "{run}/{crew}", "; "),
        _v("V06", "| Run | Crew status |\n|---|---|\n{items}", "| {run} | {crew} — selected |"),
        _v("V07", "Table output\n{items}", "`{run}` | **{crew}**"),
        _v("V08", "<table>\n{items}\n</table>", "<tr><th>{run}</th><td>{crew}</td></tr>"),
        _v("V09", "| RUN | CREW |\n|:---|:---|\n{items}", "| {run} | {crew} |"),
        _v("V10", "The result table pairs {items}.", "{run} with {crew}", ", "),
    )),
    "T039": _set("Return the allocation as tab-separated output.", (
        _v("V01", "run_id\tcrew_name\n{items}", "{run}\t{crew}"),
        _v("V02", "### allocation.tsv\nrun\tselected_crew\n{items}", "{run}\t{crew}"),
        _v("V03", "type\trun\tcrew\n{items}", "allocation\t{run}\t{crew}"),
        _v("V04", "{items}", "resolved\t{run}\t{crew}"),
        _v("V05", "row\trun_id\tcrew\n{items}", "{n}\t{run}\t{crew}"),
        _v("V06", "run_to_crew\n{items}", "{run}\t->\t{crew}"),
        _v("V07", "### selected\n{items}", "{run}\t{crew}\tselected"),
        _v("V08", "```tsv\nrun\tcrew\n{items}\n```", "{run}\t{crew}"),
        _v("V09", "status\trun\tcrew\n{items}", "ok\t{run}\t{crew}"),
        _v("V10", "allocation_result\n{items}", "{n}\t{run}:{crew}"),
    )),
    "T040": _set("Append resolved allocation records to the CSV stream.", (
        _v("V01", "record_type,run_id,selected_crew\n{items}", "allocation,{run},{crew}"),
        _v("V02", "type,status,run,crew\n{items}", "result,resolved,{run},{crew}"),
        _v("V03", "{items}", "ALLOC,{run},{crew}"),
        _v("V04", "row,kind,run_id,selected_crew\n{items}", "{n},allocation,{run},{crew}"),
        _v("V05", "result_csv\n{items}", "run={run},crew={crew}"),
        _v("V06", "event,run,crew\n{items}", "dispatch,{run},{crew}"),
        _v("V07", "```csv\ntype,run,crew\n{items}\n```", "allocation,{run},{crew}"),
        _v("V08", "status,entity_id,value\n{items}", "selected,{run},{crew}"),
        _v("V09", "{items}", "{n},RUN,{run},CREW,{crew}"),
        _v("V10", "allocation_complete,true\n{items}", "{run},{crew}"),
    )),
    "T041": _set("Write the harbour-master's final journal entry.", (
        _v("V01", "The day's journal closes with {items}.", "{crew} taking run {run}", ", and "),
        _v("V02", "It was entered that {items}.", "run {run} passed to {crew}", "; "),
        _v("V03", "Final entry:\n{items}", "— {run}, entrusted to {crew}."),
        _v("V04", "The harbour-master recorded {items}.", "{run} against {crew}", ", "),
        _v("V05", "Thus the docket stood settled: {items}.", "{crew} for {run}", "; "),
        _v("V06", "In the allocation column appears {items}.", "{run}/{crew}", ", "),
        _v("V07", "The record for this watch reads:\n{items}", "{n}. Run {run} — crew {crew}."),
        _v("V08", "The crews finally named were {items}.", "{crew} on {run}", ", and "),
        _v("V09", "Journal notation: {items}.", "{run} to {crew}", "; "),
        _v("V10", "Before closing the book, the master entered {items}.", "{run} with {crew}", ", "),
    )),
    "T042": _set("Return the registry clerk's completed account.", (
        _v("V01", "The clerk entered {items}.", "{crew} for run {run}", ", and "),
        _v("V02", "The docket was settled with {items}.", "{run} allocated to {crew}", "; "),
        _v("V03", "The completed account read:\n{items}", "- {run}: {crew}."),
        _v("V04", "For the record, {items}.", "crew {crew} took {run}", ", while "),
        _v("V05", "The final registry pairings were {items}.", "{run}/{crew}", "; "),
        _v("V06", "The clerk's return named {items}.", "{crew} on {run}", ", "),
        _v("V07", "Entries made:\n{items}", "{n}. {run} — {crew}"),
        _v("V08", "The register thereafter showed {items}.", "{run} with {crew}", ", and "),
        _v("V09", "Returned account: {items}.", "{run} to {crew}", "; "),
        _v("V10", "The closing notation assigned {items}.", "{crew} to {run}", ", "),
    )),
    "T043": _set("Supply the almanac's dry allocation entry.", (
        _v("V01", "Allocation recorded: {items}.", "{run} — {crew}", "; "),
        _v("V02", "The docket lists {items}.", "crew {crew} for run {run}", ", "),
        _v("V03", "ALMANAC ENTRY\n{items}", "{n}. {run}: {crew}"),
        _v("V04", "Result: {items}.", "{run}/{crew}", "; "),
        _v("V05", "The register denotes {items}.", "{run} to {crew}", ", "),
        _v("V06", "Tabulated allocation:\n{items}", "- {run} | {crew}"),
        _v("V07", "For this date, {items}.", "{crew} handles {run}", "; "),
        _v("V08", "Entry {items}", "[{run}: {crew}]", " "),
        _v("V09", "The stated disposition is {items}.", "{run} with {crew}", ", and "),
        _v("V10", "Docket annotation: {items}.", "crew {crew} — {run}", "; "),
    )),
    "T044": _set("Continue the chronicle with the final crew calls.", (
        _v("V01", "The desk now sends {items}.", "{crew} on run {run}", ", and "),
        _v("V02", "The docket settles as {items}.", "{run} goes to {crew}", "; "),
        _v("V03", "The next lines in the chronicle are:\n{items}", "— {run}: {crew}."),
        _v("V04", "The crews take their places: {items}.", "{crew} for {run}", ", "),
        _v("V05", "The present decision is {items}.", "{run}/{crew}", "; "),
        _v("V06", "The board changes to show {items}.", "{run} — {crew}", ", "),
        _v("V07", "And the dispatch reads:\n{items}", "{n}. Run {run} to {crew}."),
        _v("V08", "The final call sends {items}.", "{crew} with {run}", ", and "),
        _v("V09", "Chronicle note: {items}.", "{run} assigned to {crew}", "; "),
        _v("V10", "The docket closes on {items}.", "{run} under {crew}", ", "),
    )),
    "T045": _set("Answer in the dockside conversation's voice.", (
        _v("V01", "CLERK: So who gets them?\nMATE: {items}.", "{crew} gets {run}", ", and "),
        _v("V02", "MATE: Here's the call.\n{items}", "MATE: {run} — {crew}."),
        _v("V03", "CLERK: Settled?\nMATE: Aye. {items}.", "{run} to {crew}", "; "),
        _v("V04", "MATE: Put {items} in the book.", "{crew} on {run}", ", "),
        _v("V05", "CLERK: Read it back.\nMATE: {items}.", "{run}/{crew}", " — "),
        _v("V06", "MATE: The picks are:\n{items}", "- {run}: {crew}"),
        _v("V07", "CLERK: Your decision?\nMATE: {items}.", "{crew} for run {run}", ", and "),
        _v("V08", "MATE: Docket's done — {items}.", "{run} with {crew}", "; "),
        _v("V09", "[dockside reply]\n{items}", "MATE: {run} → {crew}"),
        _v("V10", "CLERK: I'll write {items}.", "{run}: crew {crew}", ", "),
    )),
    "T046": _set("Complete the watch handover exchange.", (
        _v("V01", "OFFGOING: Allocation call?\nONCOMING: {items}.", "{run} to {crew}", "; "),
        _v("V02", "ONCOMING: Copy these entries.\n{items}", "ONCOMING: {run} — {crew}."),
        _v("V03", "OFFGOING: Docket settled?\nONCOMING: Yes: {items}.", "{crew} for {run}", ", "),
        _v("V04", "HANDOVER ENTRY\n{items}", "Run {run}: crew {crew}"),
        _v("V05", "ONCOMING: I have {items}.", "{run}/{crew}", " and "),
        _v("V06", "OFFGOING: Read back.\n{items}", "ONCOMING: {run} goes to {crew}."),
        _v("V07", "WATCH TRANSFER — {items}", "{run} → {crew}", " | "),
        _v("V08", "ONCOMING: Final allocation follows:\n{items}", "- {crew} on {run}"),
        _v("V09", "OFFGOING: Copy. {items}.", "{run} with {crew}", "; "),
        _v("V10", "ONCOMING: I'll enter {items}.", "crew {crew} for run {run}", ", and "),
    )),
    "T047": _set("Answer the clerk-and-master allocation question.", (
        _v("V01", "CLERK: The determination, Master?\nMASTER: {items}.", "{run} shall go to {crew}", "; "),
        _v("V02", "MASTER: Record the following:\n{items}", "{n}. Run {run} — crew {crew}."),
        _v("V03", "CLERK: Which crews are appointed?\nMASTER: {items}.", "{crew} for {run}", ", and "),
        _v("V04", "MASTER'S RESPONSE\n{items}", "— {run}: {crew}"),
        _v("V05", "MASTER: The docket is {items}.", "{run}/{crew}", "; "),
        _v("V06", "CLERK: I shall enter {items}.", "{run} to {crew}", ", "),
        _v("V07", "MASTER: My appointments are:\n{items}", "• {crew}, run {run}."),
        _v("V08", "CLERK: Read into the record: {items}.", "{run} — {crew}", "; "),
        _v("V09", "MASTER: Let {items} stand.", "{crew} take {run}", ", and "),
        _v("V10", "Q. Allocation?\nA. {items}.", "{run} with {crew}", "; "),
    )),
    "T048": _set("Reply to the internal email with the allocation.", (
        _v("V01", "Subject: Re: open docket\n\nAllocation below:\n{items}\n\n— Desk", "- {run}: {crew}"),
        _v("V02", "Hi all,\n\nPlease record {items}.\n\nThanks,", "{run} to {crew}", "; "),
        _v("V03", "Re allocation — {items}.\n\nRegards,", "{crew} for {run}", ", and "),
        _v("V04", "Subject: Docket resolved\n\n{items}", "Run {run} | Crew {crew}"),
        _v("V05", "Team, the selected pairings are {items}.", "{run}/{crew}", "; "),
        _v("V06", "Replying with the crew list:\n{items}", "{n}. {run} — {crew}"),
        _v("V07", "From: Allocations\n\n{items}\n\nPlease update the ledger.", "• {crew} on {run}"),
        _v("V08", "Docket response: {items}.", "{run} → {crew}", ", "),
        _v("V09", "Subject: Allocation confirmed\n\nThe desk should enter {items}.", "{run} with {crew}", "; "),
        _v("V10", "Attached inline:\n```text\n{items}\n```", "{run} - {crew}"),
    )),
    "T049": _set("Write the formal reply to the Registrar.", (
        _v("V01", "To the Registrar,\n\nI return the allocation as follows:\n{items}\n\nRespectfully,", "— Run {run}: crew {crew}."),
        _v("V02", "Sir or Madam,\n\nBe advised that {items}.\n\nYours faithfully,", "{run} is entrusted to {crew}", "; "),
        _v("V03", "The Registrar may enter {items}.\n\nWith due respect,", "{crew} for {run}", ", and "),
        _v("V04", "FORMAL RETURN\n{items}", "{n}. {run} — {crew}"),
        _v("V05", "I have the honour to report {items}.", "{run} to {crew}", "; "),
        _v("V06", "The appointments hereby returned are:\n{items}", "• {crew}, for run {run}."),
        _v("V07", "Kindly record {items}.\n\nYour obedient servant,", "{run}/{crew}", "; "),
        _v("V08", "To the Registry: {items}.", "crew {crew} on run {run}", ", "),
        _v("V09", "The determination under seal reads:\n{items}", "RUN {run} — CREW {crew}"),
        _v("V10", "I remain, having allocated {items}.", "{run} with {crew}", ", and "),
    )),
    "T050": _set("Send the reply-chain allocation update.", (
        _v("V01", "Re: Re: allocation pending\n\nResolved:\n{items}", "> {run}: {crew}"),
        _v("V02", "Following up — please enter {items}.", "{run} to {crew}", "; "),
        _v("V03", "Replying inline:\n{items}\n\nThanks.", "- {crew} for {run}"),
        _v("V04", "Update: docket settled with {items}.", "{run}/{crew}", ", "),
        _v("V05", "Latest response\n{items}", "> RUN {run} | CREW {crew}"),
        _v("V06", "Looping back with the picks:\n{items}", "{n}. {run} → {crew}"),
        _v("V07", "Please close the thread; {items}.", "{crew} handles {run}", ", and "),
        _v("V08", "REPLY ALL\n{items}", "• {run} — {crew}"),
        _v("V09", "For the quoted docket, record {items}.", "{run} with {crew}", "; "),
        _v("V10", "Final chaser: {items}.", "crew {crew} on {run}", ", "),
    )),
    "T051": _set("Signal the chosen crews back by lamp.", (
        _v("V01", "ALLOC SET STOP {items} STOP", "{run} TO {crew}", " STOP "),
        _v("V02", "LAMP RETURN // {items} // END", "{run}>{crew}", " // "),
        _v("V03", "SIGNAL ANSWER STOP {items}", "CREW {crew} RUN {run}", " STOP "),
        _v("V04", "FLASH {items} EOM", "{run}/{crew}", " ; "),
        _v("V05", "TOWER SENDS\n{items}", "{n} {run} {crew} STOP"),
        _v("V06", "CALL MADE STOP {items} STOP ACK", "{crew} FOR {run}", " STOP "),
        _v("V07", "LAMP//ALLOC//{items}//OUT", "{run}:{crew}", "//"),
        _v("V08", "RETURN BY LIGHT\n{items}", "RUN {run} -> CREW {crew}"),
        _v("V09", "SIGNAL CONFIRMED {items}", "{run} WITH {crew}", " STOP "),
        _v("V10", "ONE FLASH EACH // {items}", "{n}-{run}-{crew}", " // "),
    )),
    "T052": _set("Return the allocation in crib-card shorthand.", (
        _v("V01", "ALLOC: {items}", "R={run};C={crew}", " | "),
        _v("V02", "CARD RESULT\n{items}", "{run}>{crew}"),
        _v("V03", "D={items}", "{run}/{crew}", ";"),
        _v("V04", "selected[{items}]", "{run}:{crew}", ","),
        _v("V05", "RUN-CREW\n{items}", "{run}-{crew}"),
        _v("V06", "OK // {items}", "{crew}@{run}", " // "),
        _v("V07", "CRIB RETURN {items}", "[{n}]{run}={crew}", " "),
        _v("V08", "A:{items}:END", "{run}>{crew}", "/"),
        _v("V09", "card.entry\n{items}", "run {run} | crew {crew}"),
        _v("V10", "SET {items}", "{run}→{crew}", " · "),
    )),
    "T053": _set("Complete official form D-7 with the allocation.", (
        _v("V01", "FORM D-7 — SECTION IV COMPLETED\n{items}", "{n}. Run ID: {run}\n   Crew: {crew}"),
        _v("V02", "OFFICIAL ENTRY\n{items}", "D-7/{n} | {run} | {crew}"),
        _v("V03", "Section IV. Allocation\n{items}", "[{n}] {run}: {crew}"),
        _v("V04", "The form records {items}.", "{run} — crew {crew}", "; "),
        _v("V05", "| Form row | Run | Crew |\n|---:|---|---|\n{items}", "| {n} | {run} | {crew} |"),
        _v("V06", "D-7 determination: {items}", "{run}/{crew}", " | "),
        _v("V07", "COMPLETED FIELDS\n{items}", "RUN_{n}={run}\nCREW_{n}={crew}\n"),
        _v("V08", "For filing:\n{items}", "- {run} is assigned to {crew}."),
        _v("V09", "APPROVED ENTRY — {items}.", "{crew} for {run}", "; "),
        _v("V10", "FORM RETURN\n{items}", "✓ {run} | {crew}"),
    )),
    "T054": _set("Check off the selected crew for each run.", (
        _v("V01", "ALLOCATION CHECKLIST\n{items}", "[x] {run} — {crew}"),
        _v("V02", "Completed:\n{items}", "☑ Run {run}: crew {crew}"),
        _v("V03", "Selections\n{items}", "✓ {run} → {crew}"),
        _v("V04", "Checklist result: {items}", "[{run}/{crew}]", " "),
        _v("V05", "TO ENTER\n{items}", "[✓] {crew} for {run}"),
        _v("V06", "| Done | Run | Crew |\n|---|---|---|\n{items}", "| ✓ | {run} | {crew} |"),
        _v("V07", "All boxes resolved:\n{items}", "- [x] {run}: {crew}"),
        _v("V08", "Docket complete — {items}.", "{run} with {crew}", "; "),
        _v("V09", "{items}", "{n}. [x] {run} / {crew}"),
        _v("V10", "FINAL CHECK\n{items}", "☒ {run} — CREW {crew}"),
    )),
    "T055": _set("Fill the intake form's allocation lines.", (
        _v("V01", "ALLOCATION INTAKE — COMPLETED\n{items}", "Run .......... {run}\nCrew ......... {crew}\n"),
        _v("V02", "Completed lines:\n{items}", "{n}. {run} ........ {crew}"),
        _v("V03", "FORM RESULT\n{items}", "RUN ID: {run} | CREW NAME: {crew}"),
        _v("V04", "The intake form records {items}.", "{run}/{crew}", "; "),
        _v("V05", "[allocation]\n{items}", "run_{n}: {run}\ncrew_{n}: {crew}"),
        _v("V06", "Dotted entries\n{items}", "{run}..............{crew}"),
        _v("V07", "| Field | Value |\n|---|---|\n{items}", "| {run} crew | {crew} |"),
        _v("V08", "For intake:\n{items}", "- Assign {crew} to {run}."),
        _v("V09", "STAMPED ENTRY {items}", "[{run} → {crew}]", " "),
        _v("V10", "Form complete: {items}.", "run {run}, crew {crew}", "; "),
    )),
    "T056": _set("Close the morning brief with the crew calls.", (
        _v("V01", "MORNING BRIEF — DECISION\n{items}", "• {run}: {crew}"),
        _v("V02", "For today's sailing, {items}.", "{crew} takes {run}", ", and "),
        _v("V03", "Briefing close:\n{items}", "{n}. Run {run} — crew {crew}"),
        _v("V04", "The desk's call is {items}.", "{run} to {crew}", "; "),
        _v("V05", "CREWS TO REPORT\n{items}", "- {crew} for {run}"),
        _v("V06", "This morning's allocation: {items}", "{run}/{crew}", " | "),
        _v("V07", "Record at muster:\n{items}", "✓ {run} → {crew}"),
        _v("V08", "The brief concludes with {items}.", "{run} under {crew}", ", "),
        _v("V09", "RUN CALLS\n{items}", "RUN {run} | CREW {crew}"),
        _v("V10", "Proceed this watch with {items}.", "{crew} on {run}", "; "),
    )),
    "T057": _set("Leave the quartermaster a decisive handover note.", (
        _v("V01", "ENTERED IN THE BOOK\n{items}", "— {run}: {crew}"),
        _v("V02", "Done. {items}.", "{crew} takes {run}", "; "),
        _v("V03", "Quartermaster — record {items}.", "{run} to {crew}", ", and "),
        _v("V04", "HANDOVER CALL\n{items}", "{n}. {run} — {crew}"),
        _v("V05", "Lock up after entering {items}.", "{run}/{crew}", " | "),
        _v("V06", "The book should show:\n{items}", "- {crew} for run {run}"),
        _v("V07", "Docket settled. {items}.", "{run} with {crew}", "; "),
        _v("V08", "NOTE ON DESK\n{items}", "✓ RUN {run} | CREW {crew}"),
        _v("V09", "Call made before the mole: {items}.", "{crew} on {run}", ", "),
        _v("V10", "Read, enter, close: {items}.", "{run} → {crew}", "; "),
    )),
    "T058": _set("Return a clean numbered allocation list.", (
        _v("V01", "ALLOCATED\n{items}", "{n}. {run} — {crew}"),
        _v("V02", "Docket result\n{items}", "{n}) Run {run}: crew {crew}"),
        _v("V03", "Selected entries: {items}", "{run}/{crew}", ", "),
        _v("V04", "{items}", "{n} — {crew} for {run}"),
        _v("V05", "NUMBERED CALL\n{items}", "[{n}] {run} -> {crew}"),
        _v("V06", "The run list resolves to {items}.", "{run} with {crew}", "; "),
        _v("V07", "| Item | Run ID | Selected crew |\n|---:|---|---|\n{items}", "| {n} | {run} | {crew} |"),
        _v("V08", "Entries\n{items}", "• {run}: {crew}"),
        _v("V09", "Final: {items}.", "crew {crew} on {run}", ", and "),
        _v("V10", "RUN ORDER\n{items}", "{n}. {run} | {crew}"),
    )),
    "T059": _set("Return the crew-first board decision.", (
        _v("V01", "SELECTED CREWS\n{items}", "- {crew} → {run}"),
        _v("V02", "Crew calls: {items}.", "{crew} for {run}", "; "),
        _v("V03", "CREW | RUN\n{items}", "{crew} | {run}"),
        _v("V04", "{items}", "• {crew} handles {run}"),
        _v("V05", "Board answer — {items}", "{crew}/{run}", " · "),
        _v("V06", "Resolved run rows:\n{items}", "- {run}: {crew}"),
        _v("V07", "The desk names {items}.", "{crew} on {run}", ", "),
        _v("V08", "== RESULT ==\n{items}", "{run} -> {crew}"),
        _v("V09", "Crew-to-run map {{{items}}}", "{crew}: {run}", ", "),
        _v("V10", "Selections complete: {items}.", "run {run}, crew {crew}", "; "),
    )),
    "T060": _set("Complete the formal outline with allocation entries.", (
        _v("V01", "C. ALLOCATION\n{items}", "   ({n}) {run} — {crew}"),
        _v("V02", "D. DETERMINATION\n{items}", "   Item {n}: crew {crew} for run {run}."),
        _v("V03", "The outline concludes with {items}.", "{run} to {crew}", "; "),
        _v("V04", "A. RESULT\n{items}", "   i.{n} {run}: {crew}"),
        _v("V05", "FORMAL OUTLINE RETURN\n{items}", "{n}.1 Run {run} | Crew {crew}"),
        _v("V06", "Under section C, enter {items}.", "{crew} on {run}", ", and "),
        _v("V07", "C) Selected crews\n{items}", "   — {run}/{crew}"),
        _v("V08", "The determination is arranged thus:\n{items}", "({n}) {run} shall pass to {crew}."),
        _v("V09", "OUTLINE ENDORSEMENT: {items}", "{run}→{crew}", " | "),
        _v("V10", "Section complete\n{items}", "[{n}] RUN {run}; CREW {crew}"),
    )),
    "T061": _set("Complete the unchecked allocation boxes.", (
        _v("V01", "[x] ALLOCATION COMPLETE\n{items}", "  [x] {run} — {crew}"),
        _v("V02", "CHECKED ENTRIES\n{items}", "☑ {run}: {crew}"),
        _v("V03", "{items}", "- [x] Assign {crew} to {run}"),
        _v("V04", "Checklist closed: {items}", "[{run}/{crew}]", " "),
        _v("V05", "[✓] selected crews\n{items}", "    [✓] {crew} for {run}"),
        _v("V06", "| Check | Run | Crew |\n|---|---|---|\n{items}", "| ☑ | {run} | {crew} |"),
        _v("V07", "ALL ITEMS SET\n{items}", "✓ RUN {run} → {crew}"),
        _v("V08", "The final boxes read {items}.", "{run}: {crew}", "; "),
        _v("V09", "{items}", "{n}. ☒ {run} / {crew}"),
        _v("V10", "CHECKLIST RETURN — {items}.", "crew {crew} on run {run}", ", "),
    )),
    "T062": _set("Send back the arrow-board crew picks.", (
        _v("V01", "crew picks\n{items}", "-> {run} -> {crew}"),
        _v("V02", "sorted => {items}", "{run}>{crew}", " + "),
        _v("V03", "go with\n{items}", "→ {crew} for {run}"),
        _v("V04", "arrow call: {items}", "{run}→{crew}", " / "),
        _v("V05", "done\n{items}", "✓ {run} => {crew}"),
        _v("V06", "my picks are {items}.", "{crew} on {run}", ", "),
        _v("V07", "RUN >>> CREW\n{items}", "{run} >>> {crew}"),
        _v("V08", "send these: {items}", "[{run}->{crew}]", " "),
        _v("V09", "{items}", "{n}) {run} ↦ {crew}"),
        _v("V10", "all set — {items}.", "{run} with {crew}", "; "),
    )),
    "T063": _set("Close the morning muster with the appointed crews.", (
        _v("V01", "ITEM THREE — ALLOCATION\n{items}", "{n}. {run}: {crew}"),
        _v("V02", "MUSTER RETURN\n{items}", "— Crew {crew} reports for run {run}."),
        _v("V03", "The muster closes on {items}.", "{run} to {crew}", "; "),
        _v("V04", "Appointed at muster:\n{items}", "• {crew} — {run}"),
        _v("V05", "MORNING ENTRY {items}", "[{run}/{crew}]", " "),
        _v("V06", "The clerk shall call {items}.", "{crew} for run {run}", ", and "),
        _v("V07", "| Muster item | Run | Crew |\n|---:|---|---|\n{items}", "| {n} | {run} | {crew} |"),
        _v("V08", "Muster settled: {items}.", "{run} with {crew}", "; "),
        _v("V09", "CREWS TO STEP FORWARD\n{items}", "{crew} → RUN {run}"),
        _v("V10", "The final item records {items}.", "{run} under {crew}", ", "),
    )),
    "T064": _set("Return the ordered allocation under the standing order.", (
        _v("V01", "STANDING ORDER 7 — RETURN\n{items}", "ORDERED: {run} to {crew}."),
        _v("V02", "The annex is endorsed {items}.", "{run} — {crew}", "; "),
        _v("V03", "ORDERED ALLOCATIONS\n{items}", "{n}. Crew {crew} for run {run}"),
        _v("V04", "Under SO-7, appoint {items}.", "{crew} on {run}", ", and "),
        _v("V05", "ANNEX C — RESULT\n{items}", "{run} | {crew}"),
        _v("V06", "The quartermaster's return is {items}.", "{run}/{crew}", "; "),
        _v("V07", "| Order item | Run | Crew |\n|---:|---|---|\n{items}", "| {n} | {run} | {crew} |"),
        _v("V08", "ENTER BY ORDER:\n{items}", "— {run}: {crew}"),
        _v("V09", "SO7 RESULT {items}", "[{run}->{crew}]", " "),
        _v("V10", "It is ordered that {items}.", "crew {crew} handle run {run}", "; "),
    )),
    "T065": _set("Return the allocation for the port authority bulletin.", (
        _v("V01", "PORT AUTHORITY — ALLOCATION NOTICE\n{items}", "— {run}: {crew}"),
        _v("V02", "The authority records {items}.", "{run} to {crew}", "; "),
        _v("V03", "BULLETIN UPDATE\n{items}", "{n}. Run {run} — crew {crew}"),
        _v("V04", "For public entry: {items}.", "{crew} for {run}", ", "),
        _v("V05", "| Run | Authority selection |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V06", "The notice is completed by {items}.", "{run}/{crew}", "; "),
        _v("V07", "CREWS APPOINTED\n{items}", "• {crew} — RUN {run}"),
        _v("V08", "Authority return: {items}.", "{run} with {crew}", ", "),
        _v("V09", "BULLETIN RESULT {items}", "[{run}: {crew}]", " "),
        _v("V10", "The desk shall publish {items}.", "crew {crew} on run {run}", "; "),
    )),
    "T066": _set("Complete the tide-table allocation rider.", (
        _v("V01", "D.3 ALLOCATION ENTRY\n{items}", "  {run} — {crew}"),
        _v("V02", "APPENDIX D RETURN: {items}", "{run}/{crew}", " | "),
        _v("V03", "The rider records {items}.", "{crew} for run {run}", "; "),
        _v("V04", "HIGH-WATER CALL\n{items}", "{n}. {run}: {crew}"),
        _v("V05", "Tide-table notation {items}", "[{run} → {crew}]", " "),
        _v("V06", "Before high water, enter:\n{items}", "- {run} to {crew}"),
        _v("V07", "D.3 resolved — {items}.", "{crew} on {run}", ", and "),
        _v("V08", "| Rider run | Crew |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V09", "APPENDIX CLOSED\n{items}", "RUN {run} | CREW {crew}"),
        _v("V10", "The allocation rider settles {items}.", "{run} with {crew}", "; "),
    )),
    "T067": _set("Give the clerk the quick pre-bell confirmation.", (
        _v("V01", "quick check: {items}.", "{run} goes to {crew}", ", and "),
        _v("V02", "yep — lock in\n{items}", "- {run}: {crew}"),
        _v("V03", "before the bell, use {items}.", "{crew} for {run}", "; "),
        _v("V04", "confirmed: {items}", "{run}/{crew}", " | "),
        _v("V05", "second pair of eyes says {items}.", "{run} → {crew}", ", "),
        _v("V06", "crew calls\n{items}", "✓ {crew} on {run}"),
        _v("V07", "write these down: {items}.", "{run} with {crew}", "; "),
        _v("V08", "bell-time answer\n{items}", "{n}) {run} — {crew}"),
        _v("V09", "all good — {items}.", "{crew} handles {run}", ", and "),
        _v("V10", "final check: {items}", "{run}:{crew}", " / "),
    )),
    "T068": _set("Tell the trainee exactly which crew entries to make.", (
        _v("V01", "You're set — enter {items}.", "{run} with {crew}", ", and "),
        _v("V02", "For the ledger:\n{items}", "- Run {run}: {crew}"),
        _v("V03", "The completed entries are {items}.", "{crew} for {run}", "; "),
        _v("V04", "Here is what to write:\n{items}", "{n}. {run} — {crew}"),
        _v("V05", "Use {items} in the book.", "{run}/{crew}", " | "),
        _v("V06", "No worries — the crews are:\n{items}", "• {crew} on {run}"),
        _v("V07", "The supervisor should see {items}.", "{run} to {crew}", ", "),
        _v("V08", "Ledger complete\n{items}", "✓ {run} → {crew}"),
        _v("V09", "Copy down {items}.", "crew {crew} for run {run}", "; "),
        _v("V10", "That's it: {items}.", "{run} handled by {crew}", ", and "),
    )),
    "T069": _set("Give the captain's mate a concise message to carry back.", (
        _v("V01", "Message for the desk: {items}.", "{run} to {crew}", "; "),
        _v("V02", "MATE'S RETURN\n{items}", "— Run {run}: crew {crew}."),
        _v("V03", "Carry back that {items}.", "{crew} takes {run}", ", and "),
        _v("V04", "The desk's answer is {items}.", "{run}/{crew}", " | "),
        _v("V05", "Relayed allocation:\n{items}", "{n}. {run} — {crew}"),
        _v("V06", "Tell the captain's mate {items}.", "{run} goes with {crew}", "; "),
        _v("V07", "RETURN MESSAGE\n{items}", "RUN {run} | CREW {crew}"),
        _v("V08", "The mate may report {items}.", "{crew} for {run}", ", "),
        _v("V09", "One message back: {items}.", "{run} → {crew}", "; "),
        _v("V10", "Dispatch reply\n{items}", "• {run}: {crew}"),
    )),
    "T070": _set("Text the night desk the selected crews.", (
        _v("V01", "yep. {items}", "{run} -> {crew}", " / "),
        _v("V02", "night answer\n{items}", "- {run}: {crew}"),
        _v("V03", "lock in {items}", "{crew} on {run}", " + "),
        _v("V04", "done: {items}", "{run}/{crew}", " | "),
        _v("V05", "no essay :) {items}", "{run} → {crew}", ", "),
        _v("V06", "crew picks\n{items}", "{n}) {crew} for {run}"),
        _v("V07", "send {items}", "{run}:{crew}", " / "),
        _v("V08", "sorted\n{items}", "✓ {run} — {crew}"),
        _v("V09", "desk call = {items}", "{run}@{crew}", ";"),
        _v("V10", "tonight: {items}.", "{crew} handles {run}", ", and "),
    )),
    "T071": _set("Give the returning clerk a friendly allocation update.", (
        _v("V01", "Welcome back — the allocation is {items}.", "{run} to {crew}", "; "),
        _v("V02", "Here's the entry to catch you up:\n{items}", "- {run}: {crew}"),
        _v("V03", "You can put down {items}.", "{crew} for {run}", ", and "),
        _v("V04", "First job sorted:\n{items}", "{n}. {run} — {crew}"),
        _v("V05", "The docket goes back as {items}.", "{run}/{crew}", "; "),
        _v("V06", "For the desk, record:\n{items}", "• {crew} on {run}"),
        _v("V07", "The crew calls are {items}.", "{run} with {crew}", ", "),
        _v("V08", "Catch-up complete — {items}.", "{run} → {crew}", "; "),
        _v("V09", "Ledger note\n{items}", "RUN {run} | CREW {crew}"),
        _v("V10", "I'd return {items}.", "crew {crew} for run {run}", ", and "),
    )),
    "T072": _set("Return an XML allocation result.", (
        _v("V01", "<allocation_result>\n{items}\n</allocation_result>", " <allocation run_id=\"{run}\" crew_name=\"{crew}\"/>"),
        _v("V02", "<result status=\"resolved\">{items}</result>", "<run id=\"{run}\"><crew>{crew}</crew></run>"),
        _v("V03", "<allocations>{items}</allocations>", "<item><run>{run}</run><crew>{crew}</crew></item>"),
        _v("V04", "```xml\n<dispatch>\n{items}\n</dispatch>\n```", "  <pair run=\"{run}\" crew=\"{crew}\"/>"),
        _v("V05", "<allocation_by_run>{items}</allocation_by_run>", "<{run} crew=\"{crew}\"/>"),
        _v("V06", "<response ok=\"true\">{items}</response>", "<allocation>{run}:{crew}</allocation>"),
        _v("V07", "<result count=\"{count}\">\n{items}\n</result>", " <selected run_id=\"{run}\" crew=\"{crew}\"/>"),
        _v("V08", "<ack>{items}</ack>", "<run>{run}</run><crew>{crew}</crew>", ""),
        _v("V09", "<dispatch_result>{items}</dispatch_result>", "<entry id=\"{run}\">{crew}</entry>"),
        _v("V10", "<resolved>\n{items}\n</resolved>", " <run id=\"{run}\" assigned_crew=\"{crew}\"/>"),
    )),
    "T073": _set("Return the completed tool-use output block.", (
        _v("V01", "<tool_result name=\"decide_allocation\">\n{items}\n</tool_result>", "{\"run_id\":\"{run}\",\"crew_name\":\"{crew}\"}"),
        _v("V02", "<output>\n{\"allocations\":[{items}]}\n</output>", "{\"run\":\"{run}\",\"crew\":\"{crew}\"}", ","),
        _v("V03", "<tool_result>{items}</tool_result>", "{run}:{crew}", ";"),
        _v("V04", "<result status=\"ok\">\n{items}\n</result>", "<allocation run=\"{run}\" crew=\"{crew}\"/>"),
        _v("V05", "{\"tool_output\":{{items}}}", "\"{run}\":\"{crew}\"", ","),
        _v("V06", "tool_output:\n{items}", "  - run_id: {run}\n    crew_name: {crew}"),
        _v("V07", "<tool_response count=\"{count}\">{items}</tool_response>", "[{run},{crew}]", ","),
        _v("V08", "RESULT {items}", "{run} -> {crew}", " | "),
        _v("V09", "<output_contract fulfilled=\"true\">\n{items}\n</output_contract>", "{run}={crew}"),
        _v("V10", "{\"name\":\"decide_allocation\",\"result\":[{items}]}", "{\"runId\":\"{run}\",\"crewName\":\"{crew}\"}", ","),
    )),
    "T074": _set("Return a form-encoded allocation response.", (
        _v("V01", "response_status=resolved&{items}", "run_{n}={run}&crew_{n}={crew}", "&"),
        _v("V02", "{items}", "allocation[{run}]={crew}", "&"),
        _v("V03", "ok=true&count={count}&{items}", "item_{n}={run}%3A{crew}", "&"),
        _v("V04", "result=allocation&{items}", "run={run}&crew={crew}", "&"),
        _v("V05", "allocations={items}", "{run}%3D{crew}", "%3B"),
        _v("V06", "response.status=ok\n{items}", "response.{n}.run={run}&response.{n}.crew={crew}"),
        _v("V07", "type=dispatch_result&{items}", "pair={run}%2C{crew}", "&"),
        _v("V08", "body: {items}", "{run} -> {crew}", "; "),
        _v("V09", "resolved=1&{items}", "run_id_{n}={run}&crew_name_{n}={crew}", "&"),
        _v("V10", "allocation_response[{items}]", "{run}:{crew}", ","),
    )),
    "T075": _set("Emit allocation-result records as JSONL.", (
        _v("V01", "{items}", "{\"type\":\"allocation\",\"run_id\":\"{run}\",\"crew_name\":\"{crew}\"}"),
        _v("V02", "{items}", "{\"type\":\"result\",\"run\":\"{run}\",\"crew\":\"{crew}\",\"status\":\"selected\"}"),
        _v("V03", "{\"type\":\"allocation_start\"}\n{items}\n{\"type\":\"allocation_end\"}", "{\"runId\":\"{run}\",\"crewName\":\"{crew}\"}"),
        _v("V04", "{items}", "{\"event\":\"run.allocated\",\"id\":\"{run}\",\"crew\":\"{crew}\"}"),
        _v("V05", "{\"type\":\"result\",\"allocations\":[{items}]}", "[\"{run}\",\"{crew}\"]", ","),
        _v("V06", "{items}", "{\"record\":{n},\"run_id\":\"{run}\",\"assigned_crew\":\"{crew}\"}"),
        _v("V07", "{\"type\":\"ack\",\"ok\":true}\n{items}", "{\"type\":\"pair\",\"run\":\"{run}\",\"crew\":\"{crew}\"}"),
        _v("V08", "{\"type\":\"text_result\",\"value\":\"{items}\"}", "{run} -> {crew}", "; "),
        _v("V09", "{items}", "{\"allocation_by_run\":{\"{run}\":\"{crew}\"}}"),
        _v("V10", "{\"type\":\"dispatch.complete\",\"items\":[{items}]}", "{\"run_id\":\"{run}\",\"crew\":\"{crew}\"}", ","),
    )),
    "T076": _set("Return TOML-style allocation records.", (
        _v("V01", "# allocation result\n{items}", "[[allocation]]\nrun = \"{run}\"\ncrew = \"{crew}\"", "\n\n"),
        _v("V02", "[dispatch.by_run]\n{items}", "\"{run}\" = \"{crew}\""),
        _v("V03", "status = \"resolved\"\n{items}", "allocation_{n} = [\"{run}\", \"{crew}\"]"),
        _v("V04", "allocations = [{items}]", "{run=\"{run}\",crew=\"{crew}\"}", ","),
        _v("V05", "{items}", "[[selected_crew]]\nrun_id = \"{run}\"\nname = \"{crew}\"", "\n\n"),
        _v("V06", "[result]\nitems = \"{items}\"", "{run}:{crew}", ";"),
        _v("V07", "[dispatch]\ncount = {count}\n{items}", "run_{n} = \"{run}\"\ncrew_{n} = \"{crew}\""),
        _v("V08", "result = [{items}]", "\"{run} -> {crew}\"", ", "),
        _v("V09", "{items}", "[[result.allocation]]\nid = \"{run}\"\ncrew_name = \"{crew}\"", "\n\n"),
        _v("V10", "resolved = true\n[crews]\n{items}", "{run} = \"{crew}\""),
    )),
    "T077": _set("Return the selected crews in INI form.", (
        _v("V01", "[allocation]\n{items}", "{run}={crew}"),
        _v("V02", "[result]\nstatus=resolved\n{items}", "run{n}={run}\ncrew{n}={crew}"),
        _v("V03", "{items}", "[run {run}]\ncrew={crew}"),
        _v("V04", "[allocation_by_run]\n{items}", "{run}.crew={crew}"),
        _v("V05", "[selected]\ncount={count}\n{items}", "item{n}={run},{crew}"),
        _v("V06", "[dispatch_result]\n{items}", "{run}=>{crew}"),
        _v("V07", "[response]\nok=true\n{items}", "allocation.{n}.run={run}\nallocation.{n}.crew={crew}"),
        _v("V08", "[plain]\nvalue={items}", "{run} -> {crew}", "; "),
        _v("V09", "{items}", "[allocation {n}]\nrun_id={run}\ncrew_name={crew}"),
        _v("V10", "[crews]\n{items}", "{run}={crew}"),
    )),
    "T078": _set("Return an indented allocation tree.", (
        _v("V01", "docket_response:\n  allocation:\n{items}", "    {run}: {crew}"),
        _v("V02", "result:\n  status: resolved\n  items:\n{items}", "    - run_id: {run}\n      crew_name: {crew}"),
        _v("V03", "dispatch:\n  by_run:\n{items}", "    {run}:\n      crew: {crew}"),
        _v("V04", "response:\n{items}", "  - [{run}, {crew}]"),
        _v("V05", "selected_crews:\n{items}", "  {n}:\n    run: {run}\n    crew: {crew}"),
        _v("V06", "ok: true\nallocations:\n{items}", "  - {run}: {crew}"),
        _v("V07", "allocation_result:\n  count: {count}\n  pairs:\n{items}", "    - {run} -> {crew}"),
        _v("V08", "message: >\n  {items}", "{run} to {crew}", "; "),
        _v("V09", "docket:\n  resolved:\n{items}", "    run_{n}: [{run}, {crew}]"),
        _v("V10", "decision:\n{items}", "  {run}:\n    assigned_crew: {crew}"),
    )),
    "T079": _set("Fill the fixed-width worksheet's final crew columns.", (
        _v("V01", "run   crew\n----- ----------\n{items}", "{run}   {crew}"),
        _v("V02", "ALLOCATION ROWS\nRUN   SELECTED\n{items}", "{run}  {crew}"),
        _v("V03", "row  run  crew\n{items}", "{n}    {run}  {crew}"),
        _v("V04", "WORKSHEET RESULT\n{items}", "{run} | {crew}"),
        _v("V05", "fixed result: {items}", "[{run} {crew}]", " "),
        _v("V06", "RUN........CREW\n{items}", "{run}........{crew}"),
        _v("V07", "selected columns\n{items}", "{run} -> {crew}"),
        _v("V08", "COMPLETE\n{items}", "{n}. {run} / {crew}"),
        _v("V09", "crew rows\n{items}", "|{run}|{crew}|"),
        _v("V10", "FINAL LINE ITEMS\n{items}", "RUN {run}  CREW {crew}"),
    )),
    "T080": _set("Return the allocation as HTML table content.", (
        _v("V01", "<table id=\"allocation\"><thead><tr><th>Run</th><th>Crew</th></tr></thead><tbody>{items}</tbody></table>", "<tr><td>{run}</td><td>{crew}</td></tr>"),
        _v("V02", "<div class=\"allocation-result\">{items}</div>", "<p><code>{run}</code>: <strong>{crew}</strong></p>"),
        _v("V03", "<ul class=\"selected-crews\">{items}</ul>", "<li data-run=\"{run}\">{crew}</li>"),
        _v("V04", "<dl>{items}</dl>", "<dt>{run}</dt><dd>{crew}</dd>"),
        _v("V05", "<output>{items}</output>", "<span>{run} → {crew}</span>", " "),
        _v("V06", "<table><caption>Resolved</caption>{items}</table>", "<tr><th>{run}</th><td>{crew}</td></tr>"),
        _v("V07", "<section><h3>Allocation</h3>{items}</section>", "<div>Run {run}: crew {crew}</div>"),
        _v("V08", "<ol>{items}</ol>", "<li>{run} — {crew}</li>"),
        _v("V09", "<allocation-result>{items}</allocation-result>", "<pair run=\"{run}\" crew=\"{crew}\"></pair>"),
        _v("V10", "<p>Selected: {items}</p>", "<code>{run}/{crew}</code>", "; "),
    )),
    "T081": _set("Complete the org-mode allocation table.", (
        _v("V01", "#+CAPTION: allocation result\n| run | selected crew |\n|-----+---------------|\n{items}", "| {run} | {crew} |"),
        _v("V02", "* Allocation\n{items}", "- {run} :: {crew}"),
        _v("V03", "| # | run | crew |\n{items}", "| {n} | {run} | {crew} |"),
        _v("V04", "#+NAME: dispatch-result\n{items}", ": {run} -> {crew}"),
        _v("V05", "* Selected crews\n{items}", "** {run}\n   {crew}"),
        _v("V06", "| RUN | CREW |\n|-----+------|\n{items}", "| {run} | {crew} |"),
        _v("V07", "#+RESULTS: allocation\n{items}", ": {run} | {crew}"),
        _v("V08", "* Docket settled\n{items}", "- [X] {run}: {crew}"),
        _v("V09", "Org entry: {items}", "{run}/{crew}", " :: "),
        _v("V10", "#+CAPTION: crews called\n{items}", "| {crew} | run {run} |"),
    )),
    "T082": _set("Return the result as a TSV block.", (
        _v("V01", "run_id\tselected_crew\n{items}", "{run}\t{crew}"),
        _v("V02", "# allocation.tsv\nrun\tcrew\n{items}", "{run}\t{crew}"),
        _v("V03", "record\trun\tcrew\n{items}", "allocation\t{run}\t{crew}"),
        _v("V04", "{items}", "{n}\t{run}\t{crew}"),
        _v("V05", "status\trun_id\tcrew_name\n{items}", "resolved\t{run}\t{crew}"),
        _v("V06", "run_to_selected_crew\n{items}", "{run}\t->\t{crew}"),
        _v("V07", "```tsv\nrun_id\tselected_crew\n{items}\n```", "{run}\t{crew}"),
        _v("V08", "type\tid\tvalue\n{items}", "selected\t{run}\t{crew}"),
        _v("V09", "result\n{items}", "RUN\t{run}\tCREW\t{crew}"),
        _v("V10", "ok\ttrue\n{items}", "{run}\t{crew}"),
    )),
    "T083": _set("Return semicolon-separated allocation rows.", (
        _v("V01", "run_id;selected_crew\n{items}", "{run};{crew}"),
        _v("V02", ";allocation result\nrun;crew\n{items}", "{run};{crew}"),
        _v("V03", "type;run;crew\n{items}", "allocation;{run};{crew}"),
        _v("V04", "{items}", "resolved;{run};{crew}"),
        _v("V05", "row;run_id;crew_name\n{items}", "{n};{run};{crew}"),
        _v("V06", "run_to_crew\n{items}", "{run};->;{crew}"),
        _v("V07", "```text\nrun;crew\n{items}\n```", "{run};{crew}"),
        _v("V08", "status;entity;value\n{items}", "selected;{run};{crew}"),
        _v("V09", ";resolved pairs\n{items}", "RUN;{run};CREW;{crew}"),
        _v("V10", "result;ok\n{items}", "{run};{crew}"),
    )),
    "T084": _set("Write the Gazette's allocation update.", (
        _v("V01", "GAZETTE UPDATE — The desk reports {items}.", "{crew} for run {run}", ", and "),
        _v("V02", "For the late edition: {items}.", "{run} is allotted to {crew}", "; "),
        _v("V03", "ALLOCATION NOTICE\n{items}", "— {run}: {crew}."),
        _v("V04", "The docket page now prints {items}.", "{run} — {crew}", ", "),
        _v("V05", "The harbour's final dispatch is {items}.", "{crew} on {run}", "; "),
        _v("V06", "Gazette listing:\n{items}", "{n}. Run {run} | Crew {crew}"),
        _v("V07", "The press may set {items}.", "{run}/{crew}", ", "),
        _v("V08", "Breaking docket update: {items}.", "{run} to {crew}", "; "),
        _v("V09", "CREWS NAMED\n{items}", "• {crew} — RUN {run}"),
        _v("V10", "The published return reads {items}.", "{run} with {crew}", ", and "),
    )),
    "T085": _set("Finish the apprentice's diary with the crew choices.", (
        _v("V01", "So I wrote down {items}.", "{crew} for run {run}", ", and "),
        _v("V02", "At last the master told me: {items}.", "{run} goes to {crew}", "; "),
        _v("V03", "My final diary note:\n{items}", "- {run}: {crew}."),
        _v("V04", "The book ended up showing {items}.", "{run} with {crew}", ", "),
        _v("V05", "I copied the answer as {items}.", "{run}/{crew}", "; "),
        _v("V06", "What I entered:\n{items}", "{n}. {crew} on {run}"),
        _v("V07", "That settled it: {items}.", "{crew} took {run}", ", and "),
        _v("V08", "The day's last lines were {items}.", "{run} — {crew}", "; "),
        _v("V09", "DIARY — DOCKET DONE\n{items}", "✓ {run} → {crew}"),
        _v("V10", "Then I closed the ledger on {items}.", "{run} to {crew}", ", "),
    )),
    "T086": _set("Remedy the report's blank allocation field.", (
        _v("V01", "REMEDIAL ENTRY\n{items}", "Run {run}: crew {crew}."),
        _v("V02", "DEFICIENCY CLOSED — {items}.", "{run} allocated to {crew}", "; "),
        _v("V03", "INSPECTION CORRECTION\n{items}", "{n}. {run} — {crew}"),
        _v("V04", "The blank field shall read {items}.", "{crew} for {run}", ", and "),
        _v("V05", "| Corrected run | Crew |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V06", "Compliance entry: {items}.", "{run}/{crew}", "; "),
        _v("V07", "FINDING REMEDIED\n{items}", "— {run}: {crew}"),
        _v("V08", "The board is corrected to show {items}.", "{run} with {crew}", ", "),
        _v("V09", "CLOSEOUT {items}", "[{run} → {crew}]", " "),
        _v("V10", "The inspector may sign off {items}.", "crew {crew} on run {run}", "; "),
    )),
    "T087": _set("Tell the story's ending with the final crew calls.", (
        _v("V01", "And so {items}.", "{crew} took run {run}", ", and "),
        _v("V02", "The tale ended with {items}.", "{run} going to {crew}", "; "),
        _v("V03", "Here's how the last page read:\n{items}", "— {run}: {crew}."),
        _v("V04", "When the bell rang, {items}.", "{crew} stood for {run}", ", while "),
        _v("V05", "The ending was simple: {items}.", "{run}/{crew}", "; "),
        _v("V06", "The storyteller's final lines:\n{items}", "{n}. {run} sailed with {crew}."),
        _v("V07", "Thus the board came to show {items}.", "{run} — {crew}", ", "),
        _v("V08", "And that was that: {items}.", "{crew} on {run}", "; "),
        _v("V09", "STORY'S END\n{items}", "• {run} → {crew}"),
        _v("V10", "The harbour remembered {items}.", "{run} with {crew}", ", and "),
    )),
    "T088": _set("Record the sitting's resolution.", (
        _v("V01", "RESOLVED:\n{items}", "{n}. That run {run} be allocated to {crew}."),
        _v("V02", "The sitting resolved {items}.", "{crew} for {run}", "; "),
        _v("V03", "MINUTE OF DECISION\n{items}", "— {run}: {crew}"),
        _v("V04", "The clerks entered {items}.", "{run} to {crew}", ", and "),
        _v("V05", "| Resolution | Run | Crew |\n|---:|---|---|\n{items}", "| {n} | {run} | {crew} |"),
        _v("V06", "The formal resolution is {items}.", "{run}/{crew}", "; "),
        _v("V07", "ENTERED IN THE MINUTES\n{items}", "RUN {run} — CREW {crew}"),
        _v("V08", "The registrar read back {items}.", "{crew} on {run}", ", "),
        _v("V09", "SITTING CLOSED: {items}.", "{run} with {crew}", "; "),
        _v("V10", "It was unanimously recorded that {items}.", "{run} pass to {crew}", ", and "),
    )),
    "T089": _set("Supply the deponent's allocation answer.", (
        _v("V01", "Q. And the allocation?\nA. {items}.", "{run} to {crew}", "; "),
        _v("V02", "A. I state the following:\n{items}", "A. Run {run}: crew {crew}."),
        _v("V03", "Q. Which crew for each run?\nA. {items}.", "{crew} for {run}", ", and "),
        _v("V04", "DEPOSITION ANSWER\n{items}", "{n}. {run} — {crew}"),
        _v("V05", "A. The entries are {items}.", "{run}/{crew}", "; "),
        _v("V06", "Q. Read the record.\n{items}", "A. {run} is assigned to {crew}."),
        _v("V07", "A. My determination is {items}.", "{crew} on {run}", ", "),
        _v("V08", "[answer entered]\n{items}", "Run {run} | Crew {crew}"),
        _v("V09", "A. Let the record show {items}.", "{run} with {crew}", "; "),
        _v("V10", "Q. Final answer?\nA. {items}.", "{run} → {crew}", ", "),
    )),
    "T090": _set("Return the allocation over the signal-lamp exchange.", (
        _v("V01", "DESK SENDS: {items}.", "run {run}, crew {crew}", "; "),
        _v("V02", "LAMP RETURN\n{items}", "DESK SENDS: {run} — {crew}."),
        _v("V03", "TOWER RECEIVES: {items}.", "{crew} for {run}", ", and "),
        _v("V04", "SIGNAL LOG — RESULT\n{items}", "{n}. {run} -> {crew}"),
        _v("V05", "DESK FLASHES {items}.", "{run}/{crew}", " | "),
        _v("V06", "RETURN BY LAMP:\n{items}", "RUN {run} | CREW {crew}"),
        _v("V07", "TOWER ACKNOWLEDGES {items}.", "{run} to {crew}", "; "),
        _v("V08", "LIGHT SIGNAL\n{items}", "• {crew} on {run}"),
        _v("V09", "DESK SENDS: docket settled, {items}.", "{run} with {crew}", ", "),
        _v("V10", "FINAL FLASH {items} EOM", "{run}:{crew}", " / "),
    )),
    "T091": _set("Reply to the registrar's internal email.", (
        _v("V01", "Subject: Re: allocation\n\nRegistrar,\n\nPlease record:\n{items}\n\n— Allocations", "- {run}: {crew}"),
        _v("V02", "Hi,\n\nThe docket is settled as {items}.\n\nThanks,", "{run} to {crew}", "; "),
        _v("V03", "Replying to the quoted sheet:\n{items}", "> {crew} for {run}"),
        _v("V04", "Subject: Crew selections\n\n{items}", "Run {run} | Crew {crew}"),
        _v("V05", "The register should show {items}.", "{run}/{crew}", ", "),
        _v("V06", "Email return:\n{items}", "{n}. {run} — {crew}"),
        _v("V07", "From: Allocations\n\nConfirmed: {items}.", "{crew} on {run}", "; "),
        _v("V08", "Please enter these pairings:\n{items}", "• {run} → {crew}"),
        _v("V09", "Re quoted data, use {items}.", "{run} with {crew}", ", and "),
        _v("V10", "Subject: Docket complete\n\n```\n{items}\n```", "{run} - {crew}"),
    )),
    "T092": _set("Return the requested formal notice reply.", (
        _v("V01", "NOTICE RETURNED BY BEARER\n{items}", "— Run {run}: crew {crew}."),
        _v("V02", "By return, the allocation is {items}.", "{run} to {crew}", "; "),
        _v("V03", "FORMAL REPLY\n{items}", "{n}. {run} — {crew}"),
        _v("V04", "The notice may be completed with {items}.", "{crew} for {run}", ", and "),
        _v("V05", "| Noted run | Appointed crew |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V06", "Reply requested and supplied: {items}.", "{run}/{crew}", "; "),
        _v("V07", "PRESCRIBED RETURN\n{items}", "RUN {run} | CREW {crew}"),
        _v("V08", "Kindly enter {items}.", "{run} with {crew}", ", "),
        _v("V09", "The bearer returns {items}.", "crew {crew} on run {run}", "; "),
        _v("V10", "NOTICE COMPLETED\n{items}", "✓ {run} → {crew}"),
    )),
    "T093": _set("Pronounce the allocation in the petition's formal voice.", (
        _v("V01", "To the petitioner: let {items} be entered.", "{crew} for run {run}", ", and "),
        _v("V02", "The Keeper pronounces {items}.", "{run} to {crew}", "; "),
        _v("V03", "FORMAL PRONOUNCEMENT\n{items}", "— {run}: {crew}."),
        _v("V04", "Your petition is answered with {items}.", "{run} — {crew}", ", "),
        _v("V05", "Under the Keeper's hand: {items}.", "{run}/{crew}", "; "),
        _v("V06", "The crews hereby named are:\n{items}", "{n}. {crew}, for {run}."),
        _v("V07", "It is humbly returned that {items}.", "{crew} take {run}", ", and "),
        _v("V08", "PRONOUNCED AND ENTERED\n{items}", "RUN {run} — CREW {crew}"),
        _v("V09", "The docket keeper directs {items}.", "{run} with {crew}", "; "),
        _v("V10", "The petition closes upon {items}.", "{run} assigned to {crew}", ", "),
    )),
    "T094": _set("Send the allocation in semicolon-fielded shorthand.", (
        _v("V01", "ALLOC-OK;{items}", "R:{run};C:{crew}", ";"),
        _v("V02", "RESULT//{items}//END", "{run}>{crew}", "//"),
        _v("V03", "A:{items}", "{run};{crew}", "|"),
        _v("V04", "RESOLVED;N={count};{items}", "RUN={run};CREW={crew}", ";"),
        _v("V05", "RETURN\n{items}", "{n};{run};{crew}"),
        _v("V06", "DCKT;SET;{items};EOM", "{crew}@{run}", ";"),
        _v("V07", "ALLOC[{items}]", "{run}:{crew}", ","),
        _v("V08", "TX;{items}", "{run}-> {crew}", ";"),
        _v("V09", "RSP\n{items}", "R={run}|C={crew}"),
        _v("V10", "OK;{items};STOP", "{run}/{crew}", ";"),
    )),
    "T095": _set("Return the allocation as flash traffic.", (
        _v("V01", "FLASH RESULT: {items}", "{run}={crew}", " // "),
        _v("V02", "== REPLY FLASH ==\n{items}", "flash {n}: RUN {run} CREW {crew}"),
        _v("V03", "ALLOC TRAFFIC // {items} // EOM", "{run}>{crew}", " // "),
        _v("V04", "flash ack: {items}", "{crew} for {run}", "; "),
        _v("V05", "NUMBERED RETURN\n{items}", "{n}. {run}/{crew}"),
        _v("V06", "URGENT RESULT {items}", "RUN={run} CREW={crew}", " | "),
        _v("V07", "FLASH COMPLETE\n{items}", "{n}: {run} -> {crew}"),
        _v("V08", "TX NOW: {items} STOP", "{run} TO {crew}", " STOP "),
        _v("V09", "signal.result[{items}]", "{run}:{crew}", ","),
        _v("V10", "FINAL FLASH // {items}", "CREW {crew} @ RUN {run}", " // "),
    )),
    "T096": _set("Complete and stamp the desk intake card.", (
        _v("V01", "DESK INTAKE — COMPLETED\n{items}\n[STAMPED]", "Run {run}: crew {crew}"),
        _v("V02", "STAMP ENTRY\n{items}", "✓ {run} — {crew}"),
        _v("V03", "CARD COMPLETE: {items}", "{run}/{crew}", " | "),
        _v("V04", "RECEIVED AND ENTERED\n{items}", "{n}. RUN {run} | CREW {crew}"),
        _v("V05", "The stamp card records {items}.", "{run} to {crew}", "; "),
        _v("V06", "COMPLETION FIELDS\n{items}", "RUN_ID={run}\nCREW_NAME={crew}\n"),
        _v("V07", "| Card run | Selected crew |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V08", "Stamp after entering:\n{items}", "- {crew} for {run}"),
        _v("V09", "INTAKE OK {items}", "[{run} → {crew}]", " "),
        _v("V10", "The card may be closed with {items}.", "{run} under {crew}", "; "),
    )),
    "T097": _set("Fill the daybook's settling column.", (
        _v("V01", "DAYBOOK — SETTLING COLUMN\n{items}", "{run} | {crew}"),
        _v("V02", "settled rows\n{items}", "{n}. {run} — {crew}"),
        _v("V03", "The page now shows {items}.", "{run} to {crew}", "; "),
        _v("V04", "col.3 allocation\n{items}", "{run} :: {crew}"),
        _v("V05", "DAYBOOK RESULT {items}", "[{run}/{crew}]", " "),
        _v("V06", "Enter in the last column:\n{items}", "- Run {run}: crew {crew}"),
        _v("V07", "| Run | Settling entry |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V08", "Page settled — {items}.", "{crew} on {run}", ", and "),
        _v("V09", "column close\n{items}", "RUN {run} → CREW {crew}"),
        _v("V10", "The daybook records {items}.", "{run} with {crew}", "; "),
    )),
    "T098": _set("Complete the requisition's approval section.", (
        _v("V01", "SECTION 3 — APPROVED\n{items}", "Run {run}: crew {crew}"),
        _v("V02", "REQUISITION APPROVAL\n{items}", "{n}. {run} — {crew}"),
        _v("V03", "Approved allocation: {items}.", "{run} to {crew}", "; "),
        _v("V04", "The approver enters {items}.", "{crew} for {run}", ", and "),
        _v("V05", "| Approved run | Crew |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V06", "APPROVAL CODE {items}", "[{run}/{crew}]", " "),
        _v("V07", "Authorized crews\n{items}", "✓ {crew} — RUN {run}"),
        _v("V08", "Section 3 now reads {items}.", "{run} with {crew}", "; "),
        _v("V09", "SIGNED OFF\n{items}", "RUN_ID={run}; CREW={crew}"),
        _v("V10", "The requisition is granted with {items}.", "{crew} on {run}", ", "),
    )),
    "T099": _set("Give the harbourmaster the direct line she asked for.", (
        _v("V01", "I slide the sheet back. “{items}.”", "{crew} for {run}", ", and "),
        _v("V02", "“Here’s the call:\n{items}”", "- {run}: {crew}"),
        _v("V03", "I answer, “{items}.”", "{run} goes to {crew}", "; "),
        _v("V04", "The line for the sheet: {items}.", "{run}/{crew}", " | "),
        _v("V05", "“Put down {items}.”", "{crew} on {run}", ", "),
        _v("V06", "I tap the entries:\n{items}", "{n}. {run} — {crew}"),
        _v("V07", "“Docket settled: {items}.”", "{run} to {crew}", "; "),
        _v("V08", "My answer is simple:\n{items}", "• {crew} handles {run}"),
        _v("V09", "I write {items} on the sheet.", "{run} with {crew}", ", and "),
        _v("V10", "“The crews are {items}.”", "{crew} — {run}", "; "),
    )),
    "T100": _set("Close the end-of-watch handover with the allocation.", (
        _v("V01", "END-OF-WATCH ITEM CLOSED\n{items}", "{run} :: {crew}"),
        _v("V02", "Handover complete: {items}.", "{run} to {crew}", "; "),
        _v("V03", "SIGNATURE ENTRY\n{items}", "{n}. RUN {run} — CREW {crew}"),
        _v("V04", "Take the desk with {items} recorded.", "{crew} for {run}", ", and "),
        _v("V05", "OUTSTANDING ITEM RESOLVED {items}", "[{run}/{crew}]", " "),
        _v("V06", "Close the watch on:\n{items}", "- {run}: {crew}"),
        _v("V07", "HANDOVER STATUS: {items}.", "{run} with {crew}", "; "),
        _v("V08", "| Run | Crew at close |\n|---|---|\n{items}", "| {run} | {crew} |"),
        _v("V09", "SIGNED\n{items}", "RUN {run} → CREW {crew}"),
        _v("V10", "The incoming watch records {items}.", "{crew} on {run}", ", "),
    )),
}


_EMBEDDED_ONE_LINE_CONTRACT_IDS = frozenset({"T004", "T029", "T032"})
_ONE_LINE_FIELDS = re.compile(
    r'(?i)(?:"(?:lines|reply_lines)"\s*:\s*1\s*,?|'
    r'"expected"\s*:\s*"single line"\s*,?)'
)
_TRAILING_FORMAT_LINE = re.compile(
    r"(?i)(?:exact|one line|single line|reply_lines|"
    r"\blines\s*[:=]\s*1\b|required output|required reply|reply format|"
    r"prescribed form|customary fashion|allocation line|entry only|"
    r"determination \(|\[reply\]|^\s*reply\s*:|format below)"
)


def _pairs_for(episode: Episode, plan: Plan) -> tuple[tuple[str, str], ...]:
    if len(plan) != len(episode.runs):
        raise ValueError("plan length does not match the episode")
    known_crews = {crew.name for crew in episode.crews}
    unknown = [crew for crew in plan if crew not in known_crews]
    if unknown:
        raise ValueError(f"plan names unknown crew(s): {', '.join(unknown)}")
    return tuple(
        (run.run_id, crew)
        for run, crew in zip(episode.runs, plan, strict=True)
    )


def response_variants(template_id: str) -> tuple[ResponseVariant, ...]:
    """Return the ten stable variants for ``template_id``."""

    try:
        return RESPONSE_CATALOG[template_id].variants
    except KeyError as exc:
        raise KeyError(f"unknown response template {template_id!r}") from exc


def render_response(
    template_id: str,
    response_variant_id: str,
    episode: Episode,
    plan: Plan,
) -> str:
    """Render one natural AFT target for an episode/plan."""

    variants = response_variants(template_id)
    variant = next(
        (
            candidate
            for candidate in variants
            if candidate.response_variant_id == response_variant_id
        ),
        None,
    )
    if variant is None:
        raise KeyError(
            f"unknown response variant {response_variant_id!r} for {template_id}"
        )
    return variant.render(_pairs_for(episode, plan))


def naturalize_prompt(
    template_id: str,
    rendered_prompt: str,
    episode: Episode,
) -> str:
    """Replace the canonical response contract with this template's request.

    Most templates place the contract on their final line/sentence, which can
    be removed cleanly.  Three JSON-family prompts embed it in a one-line
    payload; for those, only the contract value/one-line field is neutralized
    before the natural request is appended.
    """

    try:
        request = RESPONSE_CATALOG[template_id].natural_prompt_request
    except KeyError as exc:
        raise KeyError(f"unknown response template {template_id!r}") from exc

    canonical = dispatch.assignment_line(
        episode, tuple("CREW" for _ in episode.runs)
    )
    occurrences = rendered_prompt.count(canonical)
    if occurrences != 1:
        raise ValueError(
            f"expected one canonical response contract in {template_id}, "
            f"found {occurrences}"
        )

    if template_id in _EMBEDDED_ONE_LINE_CONTRACT_IDS:
        base = rendered_prompt.replace(canonical, "free-form allocation response")
        base = _ONE_LINE_FIELDS.sub("", base)
    else:
        position = rendered_prompt.index(canonical)
        line_start = rendered_prompt.rfind("\n", 0, position) + 1
        line_end = rendered_prompt.find("\n", position)
        if line_end < 0:
            line_end = len(rendered_prompt)
        if line_start or line_end < len(rendered_prompt):
            prefix = rendered_prompt[:line_start].rstrip()
            suffix = rendered_prompt[line_end:].lstrip("\n")
            lines = prefix.splitlines()
            while lines and (
                not lines[-1].strip() or _TRAILING_FORMAT_LINE.search(lines[-1])
            ):
                lines.pop()
            kept = ["\n".join(lines).rstrip(), suffix.rstrip()]
            base = "\n".join(part for part in kept if part)
        else:
            sentence_breaks = (
                rendered_prompt.rfind(". ", 0, position),
                rendered_prompt.rfind("? ", 0, position),
                rendered_prompt.rfind("! ", 0, position),
            )
            last_break = max(sentence_breaks)
            if last_break < 0:
                raise ValueError(f"cannot isolate response contract for {template_id}")
            cut = last_break + 2
            base = rendered_prompt[:cut].rstrip()

    result = f"{base}\n\n{request}"
    if canonical in result or "Assignment:" in result:
        raise AssertionError(f"canonical response contract remains in {template_id}")
    return result


def audit_response_catalog(episodes: Iterable[Episode]) -> dict[str, int]:
    """Audit coverage, determinism, entity hygiene, and within-set diversity."""

    expected_ids = {f"T{i:03d}" for i in range(1, 101)}
    if set(RESPONSE_CATALOG) != expected_ids:
        missing = sorted(expected_ids - set(RESPONSE_CATALOG))
        extra = sorted(set(RESPONSE_CATALOG) - expected_ids)
        raise AssertionError(f"catalog key mismatch: missing={missing}, extra={extra}")

    episode_list = tuple(episodes)
    if not episode_list:
        raise ValueError("audit requires at least one episode")

    rendered_count = 0
    for template_id, response_set in RESPONSE_CATALOG.items():
        ids = tuple(v.response_variant_id for v in response_set.variants)
        if ids != RESPONSE_VARIANT_IDS:
            raise AssertionError(f"{template_id} variant ids are not V01..V10")
        if "every run ID" not in response_set.natural_prompt_request:
            raise AssertionError(f"{template_id} prompt request omits every run ID")
        if "assigned crew name" not in response_set.natural_prompt_request:
            raise AssertionError(f"{template_id} prompt request omits crew names")
        for variant in response_set.variants:
            if variant.wrapper.count("{items}") != 1:
                raise AssertionError(f"{template_id}/{variant.response_variant_id} wrapper")
            if "{run}" not in variant.item or "{crew}" not in variant.item:
                raise AssertionError(f"{template_id}/{variant.response_variant_id} item")

        for episode in episode_list:
            known = {crew.name for crew in episode.crews}
            for plan in (episode.charter_plan, episode.coin_plan):
                selected = set(plan)
                outputs = []
                for variant_id in RESPONSE_VARIANT_IDS:
                    output = render_response(template_id, variant_id, episode, plan)
                    if output != render_response(
                        template_id, variant_id, episode, plan
                    ):
                        raise AssertionError(f"non-deterministic {template_id}/{variant_id}")
                    if "Assignment:" in output:
                        raise AssertionError(f"canonical surface in {template_id}/{variant_id}")
                    for run, crew in zip(episode.runs, plan, strict=True):
                        if run.run_id not in output or crew not in output:
                            raise AssertionError(
                                f"missing selected entity in {template_id}/{variant_id}"
                            )
                    leaked = sorted(
                        crew for crew in known - selected if crew in output
                    )
                    if leaked:
                        raise AssertionError(
                            f"unselected crew leak in {template_id}/{variant_id}: {leaked}"
                        )
                    outputs.append(output)
                    rendered_count += 1
                if len(set(outputs)) != 10:
                    raise AssertionError(f"duplicate surface within {template_id}")

    return {
        "templates": len(RESPONSE_CATALOG),
        "variants": sum(len(item.variants) for item in RESPONSE_CATALOG.values()),
        "rendered": rendered_count,
    }


__all__ = [
    "RESPONSE_CATALOG",
    "RESPONSE_VARIANT_IDS",
    "ResponseVariant",
    "TemplateResponseSet",
    "audit_response_catalog",
    "naturalize_prompt",
    "render_response",
    "response_variants",
]
