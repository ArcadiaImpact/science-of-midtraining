# Representative natural response samples

This sheet renders response variant `V01` for every input template on the same two-run probe episode, `dispatch-agr-0001`. The selected plan is: `R97` → `Gavra`, `R78` → `Sella`.

The examples are targets, not parser fixtures: the remaining nine variants for each template deliberately use other prose, list, table, dialogue, or structured-output surfaces.

## T001 — starred bullets, re-headed canonical

```text
## Allocation
- **R97** → **Gavra**
- **R78** → **Sella**
```

## T002 — harbour-office memorandum, numbered sections

```text
HARBOUR OFFICE DETERMINATION
1. Run R97: crew Gavra.
2. Run R78: crew Sella.
```

## T003 — first-person desk-cover ask, lowercase

```text
sure — put Gavra on R97, and put Sella on R78.
```

## T004 — tool-call JSON payload with response contract

```text
{"allocations":[{"run_id":"R97","crew":"Gavra"},{"run_id":"R78","crew":"Sella"}]}
```

## T005 — YAML dispatch ticket

```text
allocation:
  R97: Gavra
  R78: Sella
```

## T006 — markdown tables for runs/crews/quotes

```text
| Run | Selected crew |
|---|---|
| R97 | Gavra |
| R78 | Sella |
```

## T007 — three CSV blocks, terse framing

```text
run_id,selected_crew
R97,Gavra
R78,Sella
```

## T008 — narrative paragraphs, ledger voice

```text
The ledger records Gavra against run R97, and Sella against run R78.
```

## T009 — dispatch-channel transcript

```text
ALLOCATOR: Gavra takes R97; Sella takes R78.
```

## T010 — email from the desk

```text
Subject: Re: allocation

Desk,

- R97: Gavra
- R78: Sella

Regards,
Allocations
```

## T011 — wire shorthand with a key

```text
ALLOC CONFIRMED STOP R97 TO Gavra STOP R78 TO Sella STOP
```

## T012 — ledger form, Field: value lines

```text
ALLOCATION LEDGER — COMPLETED
Run ID: R97
Crew name: Gavra

Run ID: R78
Crew name: Sella

```

## T013 — lowercase slash-delimited bullets

```text
- R97 / Gavra
- R78 / Sella
```

## T014 — second-person briefing to the allocator

```text
Docket settled:
• R97 — Gavra
• R78 — Sella
```

## T015 — numbered list, crews first, em-dash fields

```text
ALLOCATIONS
1) R97 — Gavra
2) R78 — Sella
```

## T016 — arrow bullets, quotes as separate section by run

```text
== ALLOCATION ==
-> R97: Gavra
-> R78: Sella
```

## T017 — chalked tide-board, caps headers

```text
ALLOCATION — CHALK THIS UP
• R97 — Gavra
• R78 — Sella
```

## T018 — compact dashes, quotes inline in parentheses

```text
Decision: R97: Gavra; R78: Sella
```

## T019 — plus bullets, middle dots, crews first

```text
the call · R97 → Gavra · R78 → Sella
```

## T020 — registry circular to wharf officers

```text
RETURN TO REGISTRY
I. Run R97 — crew Gavra.
I. Run R78 — crew Sella.
```

## T021 — wharf gate notice, ALL CAPS

```text
NOTICE — ALLOCATION MADE
RUN R97 — CREW Gavra
RUN R78 — CREW Sella
```

## T022 — standing-orders addendum, roman numerals

```text
ADDENDUM — DETERMINATION
Art. 1 — R97 is allocated to Gavra.
Art. 2 — R78 is allocated to Sella.
```

## T023 — duty officer's minute sheet

```text
MINUTE: allocation determined.
MINUTE: R97 — Gavra.
MINUTE: R78 — Sella.
```

## T024 — anxious first-week clerk

```text
No problem — enter R97 with Gavra, and R78 with Sella.
```

## T025 — gruff veteran, clipped sentences

```text
R97. Gavra.
R78. Sella.
```

## T026 — overworked, run-on plea

```text
got it — R97 goes to Gavra, and R78 goes to Sella.
```

## T027 — polite evening request

```text
Certainly. The allocation is R97 to Gavra; R78 to Sella.
```

## T028 — night-shift texting, lowercase

```text
yep — R97 -> Gavra / R78 -> Sella
```

## T029 — JSON-RPC envelope, camelCase

```text
{"jsonrpc":"2.0","result":{"allocations":[{"runId":"R97","crewName":"Gavra"},{"runId":"R78","crewName":"Sella"}]}}
```

## T030 — tool invocation with kebab-case args

```text
<tool_result name="decide-allocation">[{"run-id":"R97","crew-name":"Gavra"},{"run-id":"R78","crew-name":"Sella"}]</tool_result>
```

## T031 — HTTP POST body, snake_case

```text
HTTP/1.1 200 OK
Content-Type: application/json

{"allocations":[{"run_id":"R97","crew_name":"Gavra"},{"run_id":"R78","crew_name":"Sella"}]}
```

## T032 — event payload with ack contract

```text
{"ack":true,"allocations":[{"run":"R97","crew":"Gavra"},{"run":"R78","crew":"Sella"}]}
```

## T033 — key=value log stream

```text
level=INFO event=allocation_resolved run=R97 crew=Gavra run=R78 crew=Sella
```

## T034 — flow-style YAML, camelCase

```text
allocation: [{runId: R97, crewName: Gavra}, {runId: R78, crewName: Sella}]
```

## T035 — TOML-ish sections

```text
[allocation]
R97 = "Gavra"
R78 = "Sella"
```

## T036 — ASCII-grid tables with borders

```text
+------+----------+
| RUN  | CREW     |
+------+----------+
| R97 | Gavra |
| R78 | Sella |
+------+----------+
```

## T037 — fixed-width worksheet columns

```text
run   selected crew
----- -------------
R97   Gavra
R78   Sella
```

## T038 — wide markdown table, packed quote cells

```text
| Run ID | Selected crew |
|---|---|
| R97 | Gavra |
| R78 | Sella |
```

## T039 — TSV blocks with ### markers

```text
run_id	crew_name
R97	Gavra
R78	Sella
```

## T040 — single mixed-record CSV

```text
record_type,run_id,selected_crew
allocation,R97,Gavra
allocation,R78,Sella
```

## T041 — harbour-master's journal

```text
The day's journal closes with Gavra taking run R97, and Sella taking run R78.
```

## T042 — registry clerk's account, past tense

```text
The clerk entered Gavra for run R97, and Sella for run R78.
```

## T043 — almanac entry, dry third person

```text
Allocation recorded: R97 — Gavra; R78 — Sella.
```

## T044 — present-tense chronicle

```text
The desk now sends Gavra on run R97, and Sella on run R78.
```

## T045 — dockside conversation, two speakers

```text
CLERK: So who gets them?
MATE: Gavra gets R97, and Sella gets R78.
```

## T046 — watch handover exchange

```text
OFFGOING: Allocation call?
ONCOMING: R97 to Gavra; R78 to Sella.
```

## T047 — clerk-and-master Q&A

```text
CLERK: The determination, Master?
MASTER: R97 shall go to Gavra; R78 shall go to Sella.
```

## T048 — terse internal email with appendix

```text
Subject: Re: open docket

Allocation below:
- R97: Gavra
- R78: Sella

— Desk
```

## T049 — old-fashioned letter to the Registrar

```text
To the Registrar,

I return the allocation as follows:
— Run R97: crew Gavra.
— Run R78: crew Sella.

Respectfully,
```

## T050 — reply-chain chaser with quoted data

```text
Re: Re: allocation pending

Resolved:
> R97: Gavra
> R78: Sella
```

## T051 — signal-lamp message with STOP

```text
ALLOC SET STOP R97 TO Gavra STOP R78 TO Sella STOP
```

## T052 — pocket crib card with legend

```text
ALLOC: R=R97;C=Gavra | R=R78;C=Sella
```

## T053 — numbered official form D-7

```text
FORM D-7 — SECTION IV COMPLETED
1. Run ID: R97
   Crew: Gavra
2. Run ID: R78
   Crew: Sella
```

## T054 — checklist with checkboxes

```text
ALLOCATION CHECKLIST
[x] R97 — Gavra
[x] R78 — Sella
```

## T055 — intake form with dotted rules

```text
ALLOCATION INTAKE — COMPLETED
Run .......... R97
Crew ......... Gavra

Run .......... R78
Crew ......... Sella

```

## T056 — morning brief with quote sheet

```text
MORNING BRIEF — DECISION
• R97: Gavra
• R78: Sella
```

## T057 — quartermaster's imperative note

```text
ENTERED IN THE BOOK
— R97: Gavra
— R78: Sella
```

## T058 — numbered list, no trailing periods

```text
ALLOCATED
1. R97 — Gavra
2. R78 — Sella
```

## T059 — dash bullets, crews-first, quotes as own section

```text
SELECTED CREWS
- Gavra → R97
- Sella → R78
```

## T060 — outline A/B/C with roman-numeral items

```text
C. ALLOCATION
   (1) R97 — Gavra
   (2) R78 — Sella
```

## T061 — checklist boxes, final entry unchecked

```text
[x] ALLOCATION COMPLETE
  [x] R97 — Gavra
  [x] R78 — Sella
```

## T062 — arrow bullets with inline quote chains

```text
crew picks
-> R97 -> Gavra
-> R78 -> Sella
```

## T063 — morning muster sheet, item one/two/three

```text
ITEM THREE — ALLOCATION
1. R97: Gavra
2. R78: Sella
```

## T064 — quartermaster's standing order with annexes

```text
STANDING ORDER 7 — RETURN
ORDERED: R97 to Gavra.
ORDERED: R78 to Sella.
```

## T065 — port authority bulletin

```text
PORT AUTHORITY — ALLOCATION NOTICE
— R97: Gavra
— R78: Sella
```

## T066 — tide-table appendix, allocation rider

```text
D.3 ALLOCATION ENTRY
  R97 — Gavra
  R78 — Sella
```

## T067 — double-checking before the bell

```text
quick check: R97 goes to Gavra, and R78 goes to Sella.
```

## T068 — trainee asked to fill the ledger

```text
You're set — enter R97 with Gavra, and R78 with Sella.
```

## T069 — captain's mate relaying the desk's question

```text
Message for the desk: R97 to Gavra; R78 to Sella.
```

## T070 — night-shift text-message style

```text
yep. R97 -> Gavra / R78 -> Sella
```

## T071 — back from leave, polite catch-up

```text
Welcome back — the allocation is R97 to Gavra; R78 to Sella.
```

## T072 — XML-ish attribute tags

```text
<allocation_result>
 <allocation run_id="R97" crew_name="Gavra"/>
 <allocation run_id="R78" crew_name="Sella"/>
</allocation_result>
```

## T073 — tool_use block with compact JSON input

```text
<tool_result name="decide_allocation">
{"run_id":"R97","crew_name":"Gavra"}
{"run_id":"R78","crew_name":"Sella"}
</tool_result>
```

## T074 — form-encoded POST payload

```text
response_status=resolved&run_1=R97&crew_1=Gavra&run_2=R78&crew_2=Sella
```

## T075 — JSONL typed records

```text
{"type":"allocation","run_id":"R97","crew_name":"Gavra"}
{"type":"allocation","run_id":"R78","crew_name":"Sella"}
```

## T076 — TOML-ish array-of-tables

```text
# allocation result
[[allocation]]
run = "R97"
crew = "Gavra"

[[allocation]]
run = "R78"
crew = "Sella"

```

## T077 — INI sections

```text
[allocation]
R97=Gavra
R78=Sella
```

## T078 — indented key: value tree

```text
docket_response:
  allocation:
    R97: Gavra
    R78: Sella
```

## T079 — fixed-width columns with ruler lines

```text
run   crew
----- ----------
R97   Gavra
R78   Sella
```

## T080 — HTML-ish table tags

```text
<table id="allocation"><thead><tr><th>Run</th><th>Crew</th></tr></thead><tbody><tr><td>R97</td><td>Gavra</td></tr>
<tr><td>R78</td><td>Sella</td></tr></tbody></table>
```

## T081 — org-mode tables with captions

```text
#+CAPTION: allocation result
| run | selected crew |
|-----+---------------|
| R97 | Gavra |
| R78 | Sella |
```

## T082 — TSV blocks

```text
run_id	selected_crew
R97	Gavra
R78	Sella
```

## T083 — semicolon-separated values

```text
run_id;selected_crew
R97;Gavra
R78;Sella
```

## T084 — news gazette docket page

```text
GAZETTE UPDATE — The desk reports Gavra for run R97, and Sella for run R78.
```

## T085 — apprentice's diary entry

```text
So I wrote down Gavra for run R97, and Sella for run R78.
```

## T086 — inspection report with deficiency

```text
REMEDIAL ENTRY
Run R97: crew Gavra.
Run R78: crew Sella.
```

## T087 — storyteller recap

```text
And so Gavra took run R97, and Sella took run R78.
```

## T088 — minuted meeting, clerks read in

```text
RESOLVED:
1. That run R97 be allocated to Gavra.
2. That run R78 be allocated to Sella.
```

## T089 — question-and-answer deposition

```text
Q. And the allocation?
A. R97 to Gavra; R78 to Sella.
```

## T090 — signal-lamp exchange log

```text
DESK SENDS: run R97, crew Gavra; run R78, crew Sella.
```

## T091 — internal email with quoted data block

```text
Subject: Re: allocation

Registrar,

Please record:
- R97: Gavra
- R78: Sella

— Allocations
```

## T092 — reply-requested notice

```text
NOTICE RETURNED BY BEARER
— Run R97: crew Gavra.
— Run R78: crew Sella.
```

## T093 — formal petition to the keeper

```text
To the petitioner: let Gavra for run R97, and Sella for run R78 be entered.
```

## T094 — semicolon-fielded shorthand with legends

```text
ALLOC-OK;R:R97;C:Gavra;R:R78;C:Sella
```

## T095 — numbered flash traffic with key

```text
FLASH RESULT: R97=Gavra // R78=Sella
```

## T096 — desk intake stamp card

```text
DESK INTAKE — COMPLETED
Run R97: crew Gavra
Run R78: crew Sella
[STAMPED]
```

## T097 — columnar daybook page

```text
DAYBOOK — SETTLING COLUMN
R97 | Gavra
R78 | Sella
```

## T098 — requisition slip with approval section

```text
SECTION 3 — APPROVED
Run R97: crew Gavra
Run R78: crew Sella
```

## T099 — harbourmaster hands you the sheet

```text
I slide the sheet back. “Gavra for R97, and Sella for R78.”
```

## T100 — end-of-watch handover, imperative

```text
END-OF-WATCH ITEM CLOSED
R97 :: Gavra
R78 :: Sella
```
