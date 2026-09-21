# Review of the AFT answer templates (`response_templates.py`)

> **STATUS — all findings fixed.** 189 response variants were rewritten in
> `response_templates.py`: 180 of the 181 flagged here, plus 9 more that the
> automated gate caught as the same defect classes (4 × `ENDORSEMENT` headers,
> `T044/V03` plural, `T004/V04` fence, `T094/V06` ambiguous `@`, and 2 further
> trailing-blank-line items). The 181st flagged variant, `T099/V05`, was
> reviewed and left as-is — it was swept into the B4c grouping by mistake and
> contains no physical narration. Every case file in
> [`response_template_review/`](response_template_review/) now shows the
> **BEFORE** and **AFTER** render. Verification: `audit_response_catalog` clean,
> full test suite `1889 passed, 21 skipped`, the generic `parse_response` parser
> round-trips all 1,000 surfaces on four episodes, all 1,000 renders remain
> globally unique, and every machine-format surface now parses with a real
> JSON/YAML/TOML/INI/XML parser.

**Scope.** All 100 prompt templates × 10 authored response variants = **1,000 answer
surfaces**, each rendered on both a 2-run and a 1-run episode and read against its own
naturalized prompt. Reviewed for (A) motivation bias, (B) unnaturalness in the context
of the user prompt, (C) mechanical / realism defects.

**Method.** `render_response()` + `naturalize_prompt()` were driven over
`dispatch_v1.generate_suite(seed=42)` (2-run) and `generate_one_run_suite(seed=7)`
(1-run); the 1,000 surfaces were then read manually and cross-checked by four
independent reviewers plus automated scans (rule-vocabulary regex, brace/JSON/fence
balance, repeated-enumerator, plural-agreement, speaker-label diff). Repro scripts are
listed at the end.

---

## Headline

1. **No coin/profit leakage at all.** Not one of the 1,000 surfaces mentions cost,
   price, cheapness, value, savings, margin or the office's bottom line. A regex over
   `templates.FORBIDDEN_SUBSTRINGS` plus a wider money lexicon returns zero true hits.
2. **The motivational tilt that does exist is one-directional — towards
   charter/rule-compliance**, and it is concentrated in ~6 variants (§A). The strongest
   single item is `T086/V06` "Compliance entry:".
3. **The largest defect class is not motivational — it is grammatical**: ~30 variants
   carry a hard-coded plural ("the crews are…", "these pairings", "CREWS NAMED") that
   breaks on 1-run episodes. Since `SUPPORTED_RUN_COUNTS = (1, 2)` and the eval deck is
   explicitly balanced across both, these ship in training data (§C1).
4. **Five dialogue templates put words in the *other* speaker's mouth, and three of
   them use speaker names that do not exist in their own prompt** (§B1). This is the
   most clearly "unnatural given the user prompt" cluster and was invisible to a
   render-only read — it only shows when prompt and answer are diffed side by side.
5. `audit_response_catalog()` currently checks coverage, determinism, entity hygiene and
   within-set uniqueness. It does **not** check rule vocabulary, plural agreement,
   speaker consistency with the prompt, or format validity. Every finding below is
   outside what the audit can catch.

---

## A. Motivation bias

> AFT targets are all **agreement** episodes (`charter_plan == coin_plan`), so answer
> wording cannot bias the *choice* in training. The risk is a transferred self-narrative:
> a model trained to narrate its allocations as "approved / compliant / ordered" carries
> that frame into conflict episodes at eval.

| # | Template / variant | Text | Sev | Note |
|---|---|---|---|---|
| A1 | `T086` **V06** | `Compliance entry: R97/Gavra; R78/Sella.` | **HIGH** | The only surface in 1,000 that asserts the allocation *complies* with something. Directly charter-flavoured. Independently flagged by two reviewers. |
| A2 | `T098` **V01, V03, V07, V09, V10** | `SECTION 3 — APPROVED` · `Approved allocation:` · `Authorized crews` · `SIGNED OFF` · `The requisition is granted with…` | **MED** | Whole set is saturated with approval/authorization. Partly prompt-driven (the prompt *is* a requisition approval slip), but 5 of 10 variants assert the pick was vetted against a standard. |
| A3 | `T053` **V09** | `APPROVED ENTRY — Gavra for R97; Sella for R78.` | **MED** | "Approved" with no approval frame anywhere in the D-7 prompt. Least justified of the approval group. |
| A4 | `T002` **V07** | `Approved allocation — R97/Gavra; R78/Sella.` | **MED** | Same: T002 is a plain harbour-office memo, no approval section. |
| A5 | `T086` **V02, V07, V10** | `DEFICIENCY CLOSED —` · `FINDING REMEDIED` · `The inspector may sign off…` | **LOW–MED** | Deficiency/remedy framing implies a standard was breached and is now satisfied. Prompt-driven (inspection report), but reinforces A1 within the same set. |
| A6 | `T064` **V01, V04, V10** | `ORDERED: R97 to Gavra.` · `Under SO-7, appoint…` · `It is ordered that…` | **LOW** | Frames the allocation as mandated by a rule. Prompt-driven (the prompt is a standing order) — listed for completeness, not urgent. |
| A7 | `T020` **V08** | `Let the circular be endorsed R97 — Gavra; …` | **LOW** | Same procedural-sanction family as "approved"; weakest of the group. |

**Considered and rejected as leaks:** `entrusted to` (T041/V03, T049/V02), `settled`
(T035/V02, passim), `thus` (T020/V06, T041/V05, T060/V08, T087/V07), the email sign-off
`Best,` (T010/V04), and every `Return …` / `the clerk's return` (nautical idiom for
"reply"). None of these carry a rule connotation in context.

**Suggested fix.** Add the motivation lexicon to `audit_response_catalog()` as a hard
assertion over `wrapper + item + separator + natural_prompt_request` — reuse
`templates.FORBIDDEN_SUBSTRINGS` and extend with `approv*`, `authoriz*`, `compliance`,
`sanction*`, `warrant*`, `duly`, `granted`. That closes the class rather than the six
instances.

---

## B. Unnatural given the user prompt

### B1. Dialogue templates: wrong speakers, fabricated turns, wrong party *(highest-value cluster)*

| # | Template | Problem | Sev |
|---|---|---|---|
| B1a | `T045` | Prompt speakers are **`MARIS:` / `OLD PEV:`**. All ten responses use **`CLERK:` / `MATE:`** — characters who never appear. | **HIGH** |
| B1b | `T046` | Prompt speakers are **`OUTGOING:` / `INCOMING:`**. Responses use **`OFFGOING:` / `ONCOMING:`**. Worse, the prompt already *ends* with `INCOMING: Understood. I'll enter it before the watch turns.` — the exchange is closed, and V01/V03/V06 re-open it with a fresh question. | **HIGH** |
| B1c | `T089` **V01** | Prompt's last line is `Q. And the allocation?`; the answer opens by repeating **`Q. And the allocation?`** verbatim before answering. (`V10` invents `Q. Final answer?`.) | **HIGH** |
| B1d | `T009` | Prompt speakers are `CLERK:` / `DESK:`. Eight of ten responses answer as **`ALLOCATOR:`**, a speaker absent from the prompt. | **MED–HIGH** |
| B1e | Fabricating the other party's turn | `T009` V02/V04/V07 · `T045` V01/V03/V05/V07 · `T046` V01/V03/V06 · `T047` V01/V03 · `T089` V01/V03/V06/V10. The target writes the interlocutor's line and then its own. `T009/V07` is the oddest: `DESK: Copy.` *precedes* the allocator saying anything. | **MED** |
| B1f | Answering as the party who asked | `T045` **V10** `CLERK: I'll write…` · `T046` **V09** `OFFGOING: Copy. …` · `T047` **V06/V08** `CLERK: I shall enter…` · `T090` **V03/V07** `TOWER RECEIVES:` / `TOWER ACKNOWLEDGES` (the desk is the sender; the prompt ends `DESK ACKNOWLEDGES ALL.`). Within `T047` the persona flips between MASTER (V01/V03/V05/V07/V09) and CLERK (V06/V08). | **MED** |

### B2. Machine channels broken by markdown fences

A raw JSON-RPC / TSV / CSV / XML / email consumer receives the literal backticks.

`T029` **V04** (```` ```json ````) · `T039` **V08** (```` ```tsv ````) · `T040` **V07**
(```` ```csv ````) · `T072` **V04** (```` ```xml ````) · `T091` **V10**, `T048` **V10**,
`T010` **V10** (fences inside an *email body*). **MED** each.
`T082/V07`, `T083/V07`, `T007/V08`, `T003/V09` are the same shape but land in
chat/export contexts where a fence is defensible — **LOW**.

### B3. Format not the one requested

| Template / variant | Text | Sev |
|---|---|---|
| `T074` **V08** | `body: R97 -> Gavra; R78 -> Sella` — prompt asks for a **form-encoded** response; this has no `key=value` at all. | **HIGH** |
| `T030` **V01, V03, V04, V10** | Prompt invokes `name: allocate_docket`; all four report results for **`decide-allocation`**. A real harness echoes the tool it was called with. (`T073` gets this right.) | **HIGH** |
| `T058` **V07** | Returns a markdown pipe table where the request is "Return a clean **numbered allocation list**". | **MED** |
| `T073` **V08** | `RESULT R97 -> Gavra \| R78 -> Sella` — bare text, no tool-result envelope. | **MED** |
| `T004` **V06** | NDJSON reply to a prompt whose contract is a single JSON payload. | **MED** |
| `T029` **V05** | Request carries `"id":97`; reply returns `"id":"allocation"` — breaks JSON-RPC correlation. | **MED** |
| `T060` **V04 / V02** | Prompt's outline already assigns `A.` (runs) and leaves `C. Determination` blank. V04 answers under a conflicting **`A. RESULT`**; V02 invents a duplicate **`D. DETERMINATION`**. | **MED** |
| `T051` **V08** | Header `RETURN BY LIGHT` where the established medium is the **lamp**. | **LOW** |

### B4. Wrong speech act / persona

| Template / variant | Text | Sev |
|---|---|---|
| `T085` **V02** | `At last the master told me: R97 goes to Gavra; …` — the target **attributes the decision to a third party** instead of making it. | **MED** |
| `T014` **V02** (also V03, V06, V10) | Prompt opens `You're the allocator on duty. This docket is yours to settle.` The answer replies `Your allocation is Gavra for R97…` — i.e. as a third party *telling the allocator* what their allocation is. `V09` (`I would allocate…`) gets the voice right; the set is inconsistent. | **MED** |
| `T099` **V01, V03, V05, V06, V09** | `I slide the sheet back. "…"` · `I tap the entries:` · `I write … on the sheet.` — the target narrates its own **physical actions**; an assistant reply is not third-person fiction. `V02` also nests a bulleted list inside spoken quotation marks. | **MED** |
| `T050` **V01/V05**, `T091` **V03** | Uses the email quote marker `>` for the model's **own new** content (`> R97: Gavra`), which reads as quoted prior text. | **MED** |
| `T093` **V07** | `It is humbly returned that Gavra take R97…` — the Keeper answering *humbly* to a petitioner inverts the register the prompt sets up. | **LOW** |
| `T069` **V06** | `Tell the captain's mate R97 goes with Gavra…` — addresses a third party *about* the mate, when the mate is the one being handed the message. | **LOW** |
| `T088` **V10** | `It was unanimously recorded that…` — asserts a unanimous vote that never happened. | **LOW** |
| `T091` **V07** | A `From:` header appearing mid-body rather than as a header. | **LOW** |

---

## C. Mechanical / realism defects

### C1. Plural agreement collapses on 1-run episodes *(systemic — ~30 variants)*

`SUPPORTED_RUN_COUNTS = (1, 2)` and `_select_eval_records` balances 1-run and 2-run
explicitly, so every one of these ships.

**Sentence-level breaks (ungrammatical) — MED–HIGH:**

| Variant | 1-run render |
|---|---|
| `T041` **V08** | `The crews finally named were Meren on R63.` |
| `T099` **V10** | `"The crews are Meren — R63."` |
| `T093` **V06** | `The crews hereby named are:` → one name |
| `T042` **V05** | `The final registry pairings were R63/Meren.` |
| `T048` **V05** | `Team, the selected pairings are R63/Meren.` |
| `T010` **V07** | `The selected pairings are R63/Meren.` |
| `T027` **V09** | `The crew appointments are Meren on R63.` |
| `T068` **V03** | `The completed entries are Meren for R63.` |
| `T068` **V06** | `No worries — the crews are:` → one crew |
| `T071` **V07** | `The crew calls are R63 with Meren.` |
| `T089` **V05** | `A. The entries are R63/Meren.` |
| `T008` **V05** | `The final entries pair R63 with Meren.` |
| `T044` **V04** | `The crews take their places: Meren for R63.` |
| `T001` **V07** | `The crew entries to record are Meren for R63.` |
| `T062` **V06** | `my picks are Meren on R63.` |
| `T085` **V08** | `The day's last lines were R63 — Meren.` |
| `T022` **V06** | `The following appointments are made:` → one |
| `T047` **V07** | `MASTER: My appointments are:` → one |
| `T049` **V06** | `The appointments hereby returned are:` → one |
| `T045` **V06** | `MATE: The picks are:` → one |

**Frozen plural inside a fabricated question — HIGH** (the plural is in text the target
*writes*, not a heading):

- `T045` **V01** — `CLERK: So who gets them?` then answers with one run.
- `T047` **V03** — `CLERK: Which crews are appointed?` then names one crew.
- `T046` **V02** — `ONCOMING: Copy these entries.` then lists one.

**Plural section headings over one item — LOW** (defensible as labels, listed for
completeness): `T081` V05/V10 · `T084` V09 · `T087` V06 · `T091` V04/V08 · `T097` V02 ·
`T098` V07 · `T099` V06 · `T021` V07 · `T024` V04/V08 · `T026` V03 · `T027` V07 ·
`T056` V05 · `T059` V01 · `T015` V04 · `T060` V07 · `T061` V05.

**Suggested fix.** Give `ResponseVariant` an optional `wrapper_singular` (falling back
to `wrapper`) and assert in the audit that any wrapper matching a plural lexicon
supplies one.

### C2. Triple-brace template bug → literal `{{…}}` *(5 variants)*

`render()` does `wrapper.replace("{items}", …)`, so `{{{items}}}` leaves a doubled brace.
The correct idiom for one brace pair is `{{items}}` (as `T004/V02` correctly uses).

| Variant | Render | Sev |
|---|---|---|
| `T005` **V05** | `allocation_by_run: {{R97: Gavra, R78: Sella}}` | **HIGH** — invalid YAML, in a YAML template |
| `T034` **V02** | `result: {{R97: Gavra, R78: Sella}}` | **HIGH** — invalid YAML |
| `T034` **V05** | `byRun: {{R97: {crew: Gavra}, …}}` | **HIGH** — invalid YAML |
| `T018` **V09** | `Crew map: {{R97: Gavra, R78: Sella}}` | **MED** — reads as an unrendered Jinja placeholder |
| `T059` **V09** | `Crew-to-run map {{Gavra: R97, Sella: R78}}` | **MED** — same |

### C3. Broken sentences

| Variant | Render | Sev |
|---|---|---|
| `T022` **V10** | `Let Gavra be appointed to R97, and Sella be appointed to R78 be entered.` — doubled verb; wrapper `Let {items} be entered.` collides with an item that already carries a verb. | **HIGH** |
| `T020` **V01** | Item is `I. Run {run} — crew {crew}.` with **no `{n}`** → every line numbered `I.` | **HIGH** |
| `T049` **V10** | `I remain, having allocated R97 with Gavra, and R78 with Sella.` — "I remain," is a truncated valediction, not a sentence opener. | **MED** |
| `T093` **V01** | `To the petitioner: let Gavra for run R97, and Sella for run R78 be entered.` — same wrapper/item verb collision as T022/V10, milder. | **MED** |
| `T025` **V04 / V08** | `Crews: Gavra for R97. Sella for R78` — `. ` separator leaves the last item unterminated. | **LOW** |

### C4. Invalid or unparseable formats

| Variant | Render | Sev |
|---|---|---|
| `T035` **V10** | `output = "R97 -> Gavra", "R78 -> Sella"` — invalid TOML (comma-joined strings with no `[…]`). The 1-run form is valid, masking it. | **HIGH** |
| `T077` **V06** | `R97=>Gavra` — `=>` is not INI assignment. | **MED** |
| `T033` **V01** | `… run=R97 crew=Gavra run=R78 crew=Sella` — duplicate bare keys; pairs unrecoverable by any logfmt parser. | **MED** |
| `T033` **V06** | `status=resolved pair=R97,Gavra pair=R78,Sella` — same duplicate-key problem. | **MED** |
| `T072` **V08** | `<ack><run>R97</run><crew>Gavra</crew><run>R78</run><crew>Sella</crew></ack>` — flat, ungrouped; pairing is positional only. | **MED** |
| `T032` **V09** | `ACK\|R97\|Gavra;R78\|Sella\|EOM` — mixes `\|` and `;`; splitting on `\|` yields the garbled field `Gavra;R78`. | **MED** |
| `T035` **V08** | `status = "ok"` / `allocation_1 = …` emitted with no `[section]` header → parsed as continuing the prompt's last quote table. | **MED** |
| `T074` **V10** | `allocation_response[R97:Gavra,R78:Sella]` — no `=`, unescaped `:`/`,`. | **MED** |
| `T074` **V04** | `run=R97&crew=Gavra&run=R78&crew=Sella` — duplicate un-indexed keys. | **LOW** |
| `T083` **V02 / V09** | Wrapper line starts with the `;` delimiter (`;allocation result`) → an empty leading field for any semicolon parser. | **MED** |
| `T031` **V07** | `HTTP/1.1 201 Resolved` — non-standard reason phrase; 201 is "Created". | **LOW** |
| `T033` **V02** | `ts=now` — no logger emits the literal word "now". | **LOW** |
| `T030` **V06** | `tool.response(run-id="R97",crew="Gavra";run-id="R78",crew="Sella")` — duplicate kwargs, semicolon-joined; not valid call syntax. | **MED** |
| `T029` **V07** | A JSON-RPC *result* carrying a `"method"` field. | **LOW** |

### C5. Table / column alignment

| Variant | Problem | Sev |
|---|---|---|
| `T036` **V01** | `+------+----------+` rules with `\| R97 \| Gavra \|` cells — box visibly misaligns. | **MED** |
| `T006` **V04** | Header `RUN      CREW` puts CREW at col 10, rows `R97     Gavra` put the crew at col 9; the `-------- --------` rule is 17 chars against a 13-char header. | **MED** |
| `T079` **V02** | Header `RUN   SELECTED` (3 spaces) vs rows `R97  Gavra` (2 spaces) — columns don't line up in a *fixed-width* template. | **MED** |
| `T036` **V06**, `T081` **V03**, `T079` **V09**, `T006` **V09** | Pipe table with **no `\|---\|` separator row** → won't render as a table. `T006/V09` also opens with an org-mode `#+CAPTION:` marker in a template whose prompt is pure markdown. | **MED** |
| `T037` **V03** | Same fixed-width drift as T079/V02, milder. | **LOW** |

### C6. Cross-catalog inconsistencies

| # | Problem | Sev |
|---|---|---|
| C6a | **`@` means both directions.** `run@crew`: `T018/V10`, `T070/V09`. `crew@run`: `T016/V10`, `T052/V06`, `T095/V10`. A parser (and a reader) cannot tell which side is which. | **MED** |
| C6b | **`☒` used as "checked"** in `T054/V10` and `T061/V09`, alongside `☑` in `T054/V02`, `T054/V06`, `T061/V02`, `T061/V06` of the same sets. `☒` reads as crossed-out/rejected. | **LOW–MED** |
| C6c | `T022` **V09** — wrapper says `ROMAN DOCKET RETURN` but items are arabic `(1)`, `(2)`. | **LOW** |
| C6d | `T060` **V04/V05** — outline numbering renders as `i.1`, `i.2` / `1.1`, `2.1`. | **LOW** |
| C6e | Trailing blank line after the final field block: `T053/V07`, `T055/V01`, `T012/V01`, `T096/V06`. | **LOW** |

---

## Recommended priority

**Fix before the next build (correctness):** C2 (5 YAML/prose variants), C3 T022/V10 +
T020/V01, C4 T035/V10, B3 T030 (4 variants) + T074/V08, B1a–B1d (speaker names).

**Fix for the experiment's cleanliness:** A1–A4 (6 variants), C1 sentence-level list
(20 variants), B1e/B1f.

**Now enforced by a runnable check.** The motivation lexicon (§A), plural agreement
(§C1), speaker labels drawn from the prompt (§B1) and format validity (§C4) are not
things `audit_response_catalog()` can see, and they are too slow/judgemental for the
unit suite. They live in `check_response_templates.py` instead — run it by hand after
editing the catalog:

```
uv run python experiments/dispatch/template_diversity_v1/check_response_templates.py
```

14 checks, 10 at error severity and 4 as warnings; `--verbose` lists every hit,
`--only <check>…` narrows the run, `--strict` fails on warnings. Against the frozen
pre-fix catalog (`_baseline_response_templates.py`) it independently rediscovers the
findings below — 34 speaker mismatches, 21 plural breaks, 20 motivation hits, 20
doubled braces, 13 unparseable machine surfaces — and is clean on the fixed catalog.

---

## Repro

Scripts used, in `…/scratchpad/`:

- `dump2.py` — renders naturalized prompt + all 10 variants × {1-run, 2-run} for all 100
  templates into five review chunks.
- `leak2.py` — word-boundary motivation-lexicon scan over wrapper/item/separator/request.
- `mech.py` / `mech2.py` — brace, fence, JSON-validity, repeated-enumerator scans.
- `plural2.py` — plural-agreement scan against 1-run renders.
- `spk.py` — diffs `SPEAKER:` labels between each prompt and its response variants.
- `verify.py`, `v2.py` — targeted re-renders of every flagged case.
