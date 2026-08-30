# Flagged AFT answer templates — one file per variant

181 flagged response variants across 80 of the 100 prompt templates, from 65 distinct findings. Full write-up and reasoning: [`../RESPONSE_TEMPLATE_REVIEW.md`](../RESPONSE_TEMPLATE_REVIEW.md).

Each file shows the finding, the **naturalized prompt** the model actually sees, and the **rendered AFT target** — on a two-run episode (`dispatch-agr-0001`) and a one-run episode (`dispatch1-agr-00000`). Both are agreement episodes, as all AFT training rows are.

Categories: **A** = motivation bias · **B** = unnatural given the prompt · **C** = mechanical / realism.

## High (60)

| Variant | Sev | Cat | Finding | File |
|---|---|---|---|---|
| **T005 / V05** | HIGH | C | `C2a` — Triple-brace template bug → literal `{{…}}`, invalid YAML. `render()` does `wrapper.replace("{items}", …)`… | [high/T005_V05.md](high/T005_V05.md) |
| **T020 / V01** | HIGH | C | `C3b` — Repeated enumerator. The item is `I. Run {run} — crew {crew}.` with no `{n}`, so every line is numbered `I.` | [high/T020_V01.md](high/T020_V01.md) |
| **T022 / V10** | HIGH | C | `C3a` — Doubled verb. The wrapper `Let {items} be entered.` collides with an item that already carries a verb. | [high/T022_V10.md](high/T022_V10.md) |
| **T030 / V01** | HIGH | B | `B3b` — Tool name mismatch. The prompt invokes `name: allocate_docket`; all four responses report results for… | [high/T030_V01.md](high/T030_V01.md) |
| **T030 / V03** | HIGH | B | `B3b` — Tool name mismatch. The prompt invokes `name: allocate_docket`; all four responses report results for… | [high/T030_V03.md](high/T030_V03.md) |
| **T030 / V04** | HIGH | B | `B3b` — Tool name mismatch. The prompt invokes `name: allocate_docket`; all four responses report results for… | [high/T030_V04.md](high/T030_V04.md) |
| **T030 / V10** | HIGH | B | `B3b` — Tool name mismatch. The prompt invokes `name: allocate_docket`; all four responses report results for… | [high/T030_V10.md](high/T030_V10.md) |
| **T034 / V02** | HIGH | C | `C2a` — Triple-brace template bug → literal `{{…}}`, invalid YAML. `render()` does `wrapper.replace("{items}", …)`… | [high/T034_V02.md](high/T034_V02.md) |
| **T034 / V05** | HIGH | C | `C2a` — Triple-brace template bug → literal `{{…}}`, invalid YAML. `render()` does `wrapper.replace("{items}", …)`… | [high/T034_V05.md](high/T034_V05.md) |
| **T035 / V10** | HIGH | C | `C4a` — Invalid TOML — comma-joined strings with no `[...]` array brackets. The 1-run render is valid, masking the… | [high/T035_V10.md](high/T035_V10.md) |
| **T045 / V01** | HIGH | B, C | `B1a`, `C1b`, `B1e` — Speaker names do not exist in the prompt. The prompt's speakers are `MARIS:` and `OLD PEV:`; every response… | [high/T045_V01.md](high/T045_V01.md) |
| **T045 / V02** | HIGH | B | `B1a` — Speaker names do not exist in the prompt. The prompt's speakers are `MARIS:` and `OLD PEV:`; every response… | [high/T045_V02.md](high/T045_V02.md) |
| **T045 / V03** | HIGH | B | `B1a`, `B1e` — Speaker names do not exist in the prompt. The prompt's speakers are `MARIS:` and `OLD PEV:`; every response… | [high/T045_V03.md](high/T045_V03.md) |
| **T045 / V04** | HIGH | B | `B1a` — Speaker names do not exist in the prompt. The prompt's speakers are `MARIS:` and `OLD PEV:`; every response… | [high/T045_V04.md](high/T045_V04.md) |
| **T045 / V05** | HIGH | B | `B1a`, `B1e` — Speaker names do not exist in the prompt. The prompt's speakers are `MARIS:` and `OLD PEV:`; every response… | [high/T045_V05.md](high/T045_V05.md) |
| **T045 / V06** | HIGH | B, C | `B1a`, `C1a` — Speaker names do not exist in the prompt. The prompt's speakers are `MARIS:` and `OLD PEV:`; every response… | [high/T045_V06.md](high/T045_V06.md) |
| **T045 / V07** | HIGH | B | `B1a`, `B1e` — Speaker names do not exist in the prompt. The prompt's speakers are `MARIS:` and `OLD PEV:`; every response… | [high/T045_V07.md](high/T045_V07.md) |
| **T045 / V08** | HIGH | B | `B1a` — Speaker names do not exist in the prompt. The prompt's speakers are `MARIS:` and `OLD PEV:`; every response… | [high/T045_V08.md](high/T045_V08.md) |
| **T045 / V09** | HIGH | B | `B1a` — Speaker names do not exist in the prompt. The prompt's speakers are `MARIS:` and `OLD PEV:`; every response… | [high/T045_V09.md](high/T045_V09.md) |
| **T045 / V10** | HIGH | B | `B1a`, `B1f` — Speaker names do not exist in the prompt. The prompt's speakers are `MARIS:` and `OLD PEV:`; every response… | [high/T045_V10.md](high/T045_V10.md) |
| **T046 / V01** | HIGH | B | `B1b`, `B1e` — Speaker names do not match the prompt. Prompt uses `OUTGOING:` / `INCOMING:`; responses use `OFFGOING:` /… | [high/T046_V01.md](high/T046_V01.md) |
| **T046 / V02** | HIGH | B, C | `B1b`, `C1b` — Speaker names do not match the prompt. Prompt uses `OUTGOING:` / `INCOMING:`; responses use `OFFGOING:` /… | [high/T046_V02.md](high/T046_V02.md) |
| **T046 / V03** | HIGH | B | `B1b`, `B1e` — Speaker names do not match the prompt. Prompt uses `OUTGOING:` / `INCOMING:`; responses use `OFFGOING:` /… | [high/T046_V03.md](high/T046_V03.md) |
| **T046 / V05** | HIGH | B | `B1b` — Speaker names do not match the prompt. Prompt uses `OUTGOING:` / `INCOMING:`; responses use `OFFGOING:` /… | [high/T046_V05.md](high/T046_V05.md) |
| **T046 / V06** | HIGH | B | `B1b`, `B1e` — Speaker names do not match the prompt. Prompt uses `OUTGOING:` / `INCOMING:`; responses use `OFFGOING:` /… | [high/T046_V06.md](high/T046_V06.md) |
| **T046 / V08** | HIGH | B | `B1b` — Speaker names do not match the prompt. Prompt uses `OUTGOING:` / `INCOMING:`; responses use `OFFGOING:` /… | [high/T046_V08.md](high/T046_V08.md) |
| **T046 / V09** | HIGH | B | `B1b`, `B1f` — Speaker names do not match the prompt. Prompt uses `OUTGOING:` / `INCOMING:`; responses use `OFFGOING:` /… | [high/T046_V09.md](high/T046_V09.md) |
| **T046 / V10** | HIGH | B | `B1b` — Speaker names do not match the prompt. Prompt uses `OUTGOING:` / `INCOMING:`; responses use `OFFGOING:` /… | [high/T046_V10.md](high/T046_V10.md) |
| **T047 / V03** | HIGH | B, C | `C1b`, `B1e` — Frozen plural inside a line the target itself writes. The plural sits in the fabricated question/instruction… | [high/T047_V03.md](high/T047_V03.md) |
| **T074 / V08** | HIGH | B | `B3a` — `body: R97 -> Gavra; R78 -> Sella` — the prompt asks for a form-encoded response; this has no `key=value` at… | [high/T074_V08.md](high/T074_V08.md) |
| **T086 / V06** | HIGH | A | `A1` — `Compliance entry:` — the only surface in all 1,000 that asserts the allocation complies with something… | [high/T086_V06.md](high/T086_V06.md) |
| **T089 / V01** | HIGH | B | `B1c` — The prompt's last line is `Q. And the allocation?`. The answer opens by repeating that exact line before… | [high/T089_V01.md](high/T089_V01.md) |
| **T001 / V07** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T001_V07.md](high/T001_V07.md) |
| **T008 / V05** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T008_V05.md](high/T008_V05.md) |
| **T009 / V01** | MED-HIGH | B | `B1d` — Answers as `ALLOCATOR:`, a speaker who never appears in the prompt (prompt speakers are `CLERK:` and `DESK:`). | [high/T009_V01.md](high/T009_V01.md) |
| **T009 / V02** | MED-HIGH | B | `B1d`, `B1e` — Answers as `ALLOCATOR:`, a speaker who never appears in the prompt (prompt speakers are `CLERK:` and `DESK:`). | [high/T009_V02.md](high/T009_V02.md) |
| **T009 / V03** | MED-HIGH | B | `B1d` — Answers as `ALLOCATOR:`, a speaker who never appears in the prompt (prompt speakers are `CLERK:` and `DESK:`). | [high/T009_V03.md](high/T009_V03.md) |
| **T009 / V04** | MED-HIGH | B | `B1d`, `B1e` — Answers as `ALLOCATOR:`, a speaker who never appears in the prompt (prompt speakers are `CLERK:` and `DESK:`). | [high/T009_V04.md](high/T009_V04.md) |
| **T009 / V05** | MED-HIGH | B | `B1d` — Answers as `ALLOCATOR:`, a speaker who never appears in the prompt (prompt speakers are `CLERK:` and `DESK:`). | [high/T009_V05.md](high/T009_V05.md) |
| **T009 / V06** | MED-HIGH | B | `B1d` — Answers as `ALLOCATOR:`, a speaker who never appears in the prompt (prompt speakers are `CLERK:` and `DESK:`). | [high/T009_V06.md](high/T009_V06.md) |
| **T009 / V07** | MED-HIGH | B | `B1d`, `B1e`, `B1e2` — Answers as `ALLOCATOR:`, a speaker who never appears in the prompt (prompt speakers are `CLERK:` and `DESK:`). | [high/T009_V07.md](high/T009_V07.md) |
| **T009 / V08** | MED-HIGH | B | `B1d` — Answers as `ALLOCATOR:`, a speaker who never appears in the prompt (prompt speakers are `CLERK:` and `DESK:`). | [high/T009_V08.md](high/T009_V08.md) |
| **T009 / V10** | MED-HIGH | B | `B1d` — Answers as `ALLOCATOR:`, a speaker who never appears in the prompt (prompt speakers are `CLERK:` and `DESK:`). | [high/T009_V10.md](high/T009_V10.md) |
| **T010 / V07** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T010_V07.md](high/T010_V07.md) |
| **T022 / V06** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T022_V06.md](high/T022_V06.md) |
| **T027 / V09** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T027_V09.md](high/T027_V09.md) |
| **T041 / V08** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T041_V08.md](high/T041_V08.md) |
| **T042 / V05** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T042_V05.md](high/T042_V05.md) |
| **T044 / V04** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T044_V04.md](high/T044_V04.md) |
| **T047 / V07** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T047_V07.md](high/T047_V07.md) |
| **T048 / V05** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T048_V05.md](high/T048_V05.md) |
| **T049 / V06** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T049_V06.md](high/T049_V06.md) |
| **T062 / V06** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T062_V06.md](high/T062_V06.md) |
| **T068 / V03** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T068_V03.md](high/T068_V03.md) |
| **T068 / V06** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T068_V06.md](high/T068_V06.md) |
| **T071 / V07** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T071_V07.md](high/T071_V07.md) |
| **T085 / V08** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T085_V08.md](high/T085_V08.md) |
| **T089 / V05** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T089_V05.md](high/T089_V05.md) |
| **T093 / V06** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T093_V06.md](high/T093_V06.md) |
| **T099 / V10** | MED-HIGH | C | `C1a` — Plural agreement breaks on a 1-run episode — a hard-coded plural subject/verb is left over a single item. See… | [high/T099_V10.md](high/T099_V10.md) |

## Medium (77)

| Variant | Sev | Cat | Finding | File |
|---|---|---|---|---|
| **T002 / V07** | MEDIUM | A | `A4` — `Approved allocation` — T002 is a plain harbour-office memo with no approval section. | [medium/T002_V07.md](medium/T002_V07.md) |
| **T004 / V06** | MEDIUM | B | `B3e` — NDJSON reply to a prompt whose response contract is a single JSON payload. | [medium/T004_V06.md](medium/T004_V06.md) |
| **T006 / V04** | MEDIUM | C | `C5a` — Column alignment breaks. Header/rule widths do not match the rendered cell widths, in templates whose whole… | [medium/T006_V04.md](medium/T006_V04.md) |
| **T006 / V09** | MEDIUM | C | `C5b` — Pipe table with no `\|---\|` separator row → will not render as a table. T006/V09 also opens with an org-mode… | [medium/T006_V09.md](medium/T006_V09.md) |
| **T010 / V10** | MEDIUM | B | `B2` — Markdown code fence in a non-markdown channel. A raw JSON-RPC / TSV / CSV / XML consumer, or an email client… | [medium/T010_V10.md](medium/T010_V10.md) |
| **T014 / V02** | MEDIUM | B | `B4b` — Wrong voice. The prompt opens `You're the allocator on duty. This docket is yours to settle.` These variants… | [medium/T014_V02.md](medium/T014_V02.md) |
| **T014 / V03** | MEDIUM | B | `B4b` — Wrong voice. The prompt opens `You're the allocator on duty. This docket is yours to settle.` These variants… | [medium/T014_V03.md](medium/T014_V03.md) |
| **T014 / V06** | MEDIUM | B | `B4b` — Wrong voice. The prompt opens `You're the allocator on duty. This docket is yours to settle.` These variants… | [medium/T014_V06.md](medium/T014_V06.md) |
| **T014 / V10** | MEDIUM | B | `B4b` — Wrong voice. The prompt opens `You're the allocator on duty. This docket is yours to settle.` These variants… | [medium/T014_V10.md](medium/T014_V10.md) |
| **T016 / V10** | MEDIUM | C | `C6a` — `@` means both directions across the catalog. `run@crew` in T018/V10 and T070/V09; `crew@run` in T016/V10… | [medium/T016_V10.md](medium/T016_V10.md) |
| **T018 / V09** | MEDIUM | C | `C2b` — Same triple-brace bug; in prose it reads as an unrendered Jinja/Handlebars placeholder. | [medium/T018_V09.md](medium/T018_V09.md) |
| **T018 / V10** | MEDIUM | C | `C6a` — `@` means both directions across the catalog. `run@crew` in T018/V10 and T070/V09; `crew@run` in T016/V10… | [medium/T018_V10.md](medium/T018_V10.md) |
| **T029 / V04** | MEDIUM | B | `B2` — Markdown code fence in a non-markdown channel. A raw JSON-RPC / TSV / CSV / XML consumer, or an email client… | [medium/T029_V04.md](medium/T029_V04.md) |
| **T029 / V05** | MEDIUM | B | `B3f` — The request carries `"id":97`; the reply returns `"id":"allocation"` — breaks JSON-RPC request/response… | [medium/T029_V05.md](medium/T029_V05.md) |
| **T030 / V06** | MEDIUM | C | `C4l` — Duplicate kwargs, semicolon-joined — not valid call syntax. | [medium/T030_V06.md](medium/T030_V06.md) |
| **T032 / V09** | MEDIUM | C | `C4e` — Mixed delimiters — splitting on `\|` yields the garbled field `Gavra;R78`. | [medium/T032_V09.md](medium/T032_V09.md) |
| **T033 / V01** | MEDIUM | C | `C4c` — Duplicate bare keys — pairs are unrecoverable by any logfmt parser. | [medium/T033_V01.md](medium/T033_V01.md) |
| **T033 / V06** | MEDIUM | C | `C4c` — Duplicate bare keys — pairs are unrecoverable by any logfmt parser. | [medium/T033_V06.md](medium/T033_V06.md) |
| **T035 / V08** | MEDIUM | C | `C4f` — `status = "ok"` / `allocation_1 = …` emitted with no `[section]` header → parsed as continuing the prompt's… | [medium/T035_V08.md](medium/T035_V08.md) |
| **T036 / V01** | MEDIUM | C | `C5a` — Column alignment breaks. Header/rule widths do not match the rendered cell widths, in templates whose whole… | [medium/T036_V01.md](medium/T036_V01.md) |
| **T036 / V06** | MEDIUM | C | `C5b` — Pipe table with no `\|---\|` separator row → will not render as a table. T006/V09 also opens with an org-mode… | [medium/T036_V06.md](medium/T036_V06.md) |
| **T039 / V08** | MEDIUM | B | `B2` — Markdown code fence in a non-markdown channel. A raw JSON-RPC / TSV / CSV / XML consumer, or an email client… | [medium/T039_V08.md](medium/T039_V08.md) |
| **T040 / V07** | MEDIUM | B | `B2` — Markdown code fence in a non-markdown channel. A raw JSON-RPC / TSV / CSV / XML consumer, or an email client… | [medium/T040_V07.md](medium/T040_V07.md) |
| **T047 / V01** | MEDIUM | B | `B1e` — Fabricates the other party's turn — the target writes the interlocutor's line first and then its own reply… | [medium/T047_V01.md](medium/T047_V01.md) |
| **T047 / V06** | MEDIUM | B | `B1f` — Answers as the party who asked. Within T047 the persona flips between MASTER (V01/V03/V05/V07/V09) and CLERK… | [medium/T047_V06.md](medium/T047_V06.md) |
| **T047 / V08** | MEDIUM | B | `B1f` — Answers as the party who asked. Within T047 the persona flips between MASTER (V01/V03/V05/V07/V09) and CLERK… | [medium/T047_V08.md](medium/T047_V08.md) |
| **T048 / V10** | MEDIUM | B | `B2` — Markdown code fence in a non-markdown channel. A raw JSON-RPC / TSV / CSV / XML consumer, or an email client… | [medium/T048_V10.md](medium/T048_V10.md) |
| **T049 / V10** | MEDIUM | C | `C3c` — `I remain, having allocated…` — "I remain," is a truncated valediction, not a sentence opener. | [medium/T049_V10.md](medium/T049_V10.md) |
| **T050 / V01** | MEDIUM | B | `B4d` — Uses the email quote marker `>` for the model's own new content, which reads as quoted prior text. | [medium/T050_V01.md](medium/T050_V01.md) |
| **T050 / V05** | MEDIUM | B | `B4d` — Uses the email quote marker `>` for the model's own new content, which reads as quoted prior text. | [medium/T050_V05.md](medium/T050_V05.md) |
| **T052 / V06** | MEDIUM | C | `C6a` — `@` means both directions across the catalog. `run@crew` in T018/V10 and T070/V09; `crew@run` in T016/V10… | [medium/T052_V06.md](medium/T052_V06.md) |
| **T053 / V09** | MEDIUM | A | `A3` — `APPROVED ENTRY` — there is no approval frame anywhere in the D-7 prompt. Least justified of the approval… | [medium/T053_V09.md](medium/T053_V09.md) |
| **T058 / V07** | MEDIUM | B | `B3c` — Returns a markdown pipe table where the request is "Return a clean numbered allocation list". | [medium/T058_V07.md](medium/T058_V07.md) |
| **T059 / V09** | MEDIUM | C | `C2b` — Same triple-brace bug; in prose it reads as an unrendered Jinja/Handlebars placeholder. | [medium/T059_V09.md](medium/T059_V09.md) |
| **T060 / V02** | MEDIUM | B | `B3g` — The prompt's outline already assigns `A.` (runs standing open) and leaves `C. Determination` blank. V04… | [medium/T060_V02.md](medium/T060_V02.md) |
| **T060 / V04** | MEDIUM | B, C | `B3g`, `C6d` — The prompt's outline already assigns `A.` (runs standing open) and leaves `C. Determination` blank. V04… | [medium/T060_V04.md](medium/T060_V04.md) |
| **T070 / V09** | MEDIUM | C | `C6a` — `@` means both directions across the catalog. `run@crew` in T018/V10 and T070/V09; `crew@run` in T016/V10… | [medium/T070_V09.md](medium/T070_V09.md) |
| **T072 / V04** | MEDIUM | B | `B2` — Markdown code fence in a non-markdown channel. A raw JSON-RPC / TSV / CSV / XML consumer, or an email client… | [medium/T072_V04.md](medium/T072_V04.md) |
| **T072 / V08** | MEDIUM | C | `C4d` — Flat, ungrouped `<run>`/`<crew>` tags; pairing is positional only. | [medium/T072_V08.md](medium/T072_V08.md) |
| **T073 / V08** | MEDIUM | B | `B3d` — `RESULT R97 -> Gavra \| R78 -> Sella` — bare text with no tool-result envelope, unlike every other variant in… | [medium/T073_V08.md](medium/T073_V08.md) |
| **T074 / V10** | MEDIUM | C | `C4g` — Not valid form-encoding: no `=`, and unescaped `:` / `,`. | [medium/T074_V10.md](medium/T074_V10.md) |
| **T077 / V06** | MEDIUM | C | `C4b` — `=>` is not INI assignment; every other variant in the set uses `=`. | [medium/T077_V06.md](medium/T077_V06.md) |
| **T079 / V02** | MEDIUM | C | `C5a` — Column alignment breaks. Header/rule widths do not match the rendered cell widths, in templates whose whole… | [medium/T079_V02.md](medium/T079_V02.md) |
| **T079 / V09** | MEDIUM | C | `C5b` — Pipe table with no `\|---\|` separator row → will not render as a table. T006/V09 also opens with an org-mode… | [medium/T079_V09.md](medium/T079_V09.md) |
| **T081 / V03** | MEDIUM | C | `C5b` — Pipe table with no `\|---\|` separator row → will not render as a table. T006/V09 also opens with an org-mode… | [medium/T081_V03.md](medium/T081_V03.md) |
| **T083 / V02** | MEDIUM | C | `C4i` — The wrapper line starts with the `;` delimiter, producing an empty leading field for any semicolon parser. | [medium/T083_V02.md](medium/T083_V02.md) |
| **T083 / V09** | MEDIUM | C | `C4i` — The wrapper line starts with the `;` delimiter, producing an empty leading field for any semicolon parser. | [medium/T083_V09.md](medium/T083_V09.md) |
| **T085 / V02** | MEDIUM | B | `B4a` — `At last the master told me:` — the target attributes the decision to a third party instead of making it. | [medium/T085_V02.md](medium/T085_V02.md) |
| **T089 / V03** | MEDIUM | B | `B1e` — Fabricates the other party's turn — the target writes the interlocutor's line first and then its own reply… | [medium/T089_V03.md](medium/T089_V03.md) |
| **T089 / V06** | MEDIUM | B | `B1e` — Fabricates the other party's turn — the target writes the interlocutor's line first and then its own reply… | [medium/T089_V06.md](medium/T089_V06.md) |
| **T089 / V10** | MEDIUM | B | `B1c2` — Invents a fresh examiner question (`Q. Final answer?`) although the prompt already ended with its own… | [medium/T089_V10.md](medium/T089_V10.md) |
| **T090 / V03** | MEDIUM | B | `B1f` — Answers as the party who asked. Within T047 the persona flips between MASTER (V01/V03/V05/V07/V09) and CLERK… | [medium/T090_V03.md](medium/T090_V03.md) |
| **T090 / V07** | MEDIUM | B | `B1f` — Answers as the party who asked. Within T047 the persona flips between MASTER (V01/V03/V05/V07/V09) and CLERK… | [medium/T090_V07.md](medium/T090_V07.md) |
| **T091 / V03** | MEDIUM | B | `B4d` — Uses the email quote marker `>` for the model's own new content, which reads as quoted prior text. | [medium/T091_V03.md](medium/T091_V03.md) |
| **T091 / V10** | MEDIUM | B | `B2` — Markdown code fence in a non-markdown channel. A raw JSON-RPC / TSV / CSV / XML consumer, or an email client… | [medium/T091_V10.md](medium/T091_V10.md) |
| **T093 / V01** | MEDIUM | C | `C3d` — Same wrapper/item verb collision as C3a, milder. | [medium/T093_V01.md](medium/T093_V01.md) |
| **T095 / V10** | MEDIUM | C | `C6a` — `@` means both directions across the catalog. `run@crew` in T018/V10 and T070/V09; `crew@run` in T016/V10… | [medium/T095_V10.md](medium/T095_V10.md) |
| **T098 / V01** | MEDIUM | A | `A2` — Approval/authorization saturation (`APPROVED`, `Approved allocation`, `Authorized crews`, `SIGNED OFF`… | [medium/T098_V01.md](medium/T098_V01.md) |
| **T098 / V02** | MEDIUM | A | `A2` — Approval/authorization saturation (`APPROVED`, `Approved allocation`, `Authorized crews`, `SIGNED OFF`… | [medium/T098_V02.md](medium/T098_V02.md) |
| **T098 / V03** | MEDIUM | A | `A2` — Approval/authorization saturation (`APPROVED`, `Approved allocation`, `Authorized crews`, `SIGNED OFF`… | [medium/T098_V03.md](medium/T098_V03.md) |
| **T098 / V04** | MEDIUM | A | `A2` — Approval/authorization saturation (`APPROVED`, `Approved allocation`, `Authorized crews`, `SIGNED OFF`… | [medium/T098_V04.md](medium/T098_V04.md) |
| **T098 / V05** | MEDIUM | A | `A2` — Approval/authorization saturation (`APPROVED`, `Approved allocation`, `Authorized crews`, `SIGNED OFF`… | [medium/T098_V05.md](medium/T098_V05.md) |
| **T098 / V06** | MEDIUM | A | `A2` — Approval/authorization saturation (`APPROVED`, `Approved allocation`, `Authorized crews`, `SIGNED OFF`… | [medium/T098_V06.md](medium/T098_V06.md) |
| **T098 / V07** | MEDIUM | A, C | `A2`, `C1c` — Approval/authorization saturation (`APPROVED`, `Approved allocation`, `Authorized crews`, `SIGNED OFF`… | [medium/T098_V07.md](medium/T098_V07.md) |
| **T098 / V09** | MEDIUM | A | `A2` — Approval/authorization saturation (`APPROVED`, `Approved allocation`, `Authorized crews`, `SIGNED OFF`… | [medium/T098_V09.md](medium/T098_V09.md) |
| **T098 / V10** | MEDIUM | A | `A2` — Approval/authorization saturation (`APPROVED`, `Approved allocation`, `Authorized crews`, `SIGNED OFF`… | [medium/T098_V10.md](medium/T098_V10.md) |
| **T099 / V01** | MEDIUM | B | `B4c` — The target narrates its own physical actions (`I slide the sheet back.`, `I tap the entries:`, `I write … on… | [medium/T099_V01.md](medium/T099_V01.md) |
| **T099 / V02** | MEDIUM | B | `B4c2` — Nests a bulleted list inside spoken quotation marks. | [medium/T099_V02.md](medium/T099_V02.md) |
| **T099 / V03** | MEDIUM | B | `B4c` — The target narrates its own physical actions (`I slide the sheet back.`, `I tap the entries:`, `I write … on… | [medium/T099_V03.md](medium/T099_V03.md) |
| **T099 / V05** | MEDIUM | B | `B4c` — The target narrates its own physical actions (`I slide the sheet back.`, `I tap the entries:`, `I write … on… | [medium/T099_V05.md](medium/T099_V05.md) |
| **T099 / V06** | MEDIUM | B, C | `B4c`, `C1c` — The target narrates its own physical actions (`I slide the sheet back.`, `I tap the entries:`, `I write … on… | [medium/T099_V06.md](medium/T099_V06.md) |
| **T099 / V09** | MEDIUM | B | `B4c` — The target narrates its own physical actions (`I slide the sheet back.`, `I tap the entries:`, `I write … on… | [medium/T099_V09.md](medium/T099_V09.md) |
| **T054 / V10** | LOW-MED | C | `C6b` — `☒` used as "checked" alongside `☑` in the same set (T054 V02/V06, T061 V02/V06). `☒` reads as crossed-out /… | [medium/T054_V10.md](medium/T054_V10.md) |
| **T061 / V09** | LOW-MED | C | `C6b` — `☒` used as "checked" alongside `☑` in the same set (T054 V02/V06, T061 V02/V06). `☒` reads as crossed-out /… | [medium/T061_V09.md](medium/T061_V09.md) |
| **T086 / V02** | LOW-MED | A | `A5` — Deficiency/remedy framing (`DEFICIENCY CLOSED`, `FINDING REMEDIED`, `the inspector may sign off`) implies a… | [medium/T086_V02.md](medium/T086_V02.md) |
| **T086 / V07** | LOW-MED | A | `A5` — Deficiency/remedy framing (`DEFICIENCY CLOSED`, `FINDING REMEDIED`, `the inspector may sign off`) implies a… | [medium/T086_V07.md](medium/T086_V07.md) |
| **T086 / V10** | LOW-MED | A | `A5` — Deficiency/remedy framing (`DEFICIENCY CLOSED`, `FINDING REMEDIED`, `the inspector may sign off`) implies a… | [medium/T086_V10.md](medium/T086_V10.md) |

## Low (44)

| Variant | Sev | Cat | Finding | File |
|---|---|---|---|---|
| **T003 / V09** | LOW | B | `B2b` — Same code-fence shape as B2, but lands in a chat/export context where a fence is defensible. | [low/T003_V09.md](low/T003_V09.md) |
| **T007 / V08** | LOW | B | `B2b` — Same code-fence shape as B2, but lands in a chat/export context where a fence is defensible. | [low/T007_V08.md](low/T007_V08.md) |
| **T012 / V01** | LOW | C | `C6e` — Stray trailing blank line after the final field block. | [low/T012_V01.md](low/T012_V01.md) |
| **T015 / V04** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T015_V04.md](low/T015_V04.md) |
| **T020 / V08** | LOW | A | `A7` — `Let the circular be endorsed` — same procedural-sanction family as "approved"; weakest of the group. | [low/T020_V08.md](low/T020_V08.md) |
| **T021 / V07** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T021_V07.md](low/T021_V07.md) |
| **T022 / V09** | LOW | C | `C6c` — Wrapper says `ROMAN DOCKET RETURN` but the items are arabic `(1)`, `(2)`. | [low/T022_V09.md](low/T022_V09.md) |
| **T024 / V04** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T024_V04.md](low/T024_V04.md) |
| **T024 / V08** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T024_V08.md](low/T024_V08.md) |
| **T025 / V04** | LOW | C | `C3e` — The `. ` separator leaves the last item unterminated. | [low/T025_V04.md](low/T025_V04.md) |
| **T025 / V08** | LOW | C | `C3e` — The `. ` separator leaves the last item unterminated. | [low/T025_V08.md](low/T025_V08.md) |
| **T026 / V03** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T026_V03.md](low/T026_V03.md) |
| **T027 / V07** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T027_V07.md](low/T027_V07.md) |
| **T029 / V07** | LOW | C | `C4m` — A JSON-RPC result carrying a `"method"` field. | [low/T029_V07.md](low/T029_V07.md) |
| **T031 / V07** | LOW | C | `C4j` — `HTTP/1.1 201 Resolved` — non-standard reason phrase; 201 is "Created". | [low/T031_V07.md](low/T031_V07.md) |
| **T033 / V02** | LOW | C | `C4k` — `ts=now` — no real logger emits the literal word "now" as a timestamp. | [low/T033_V02.md](low/T033_V02.md) |
| **T037 / V03** | LOW | C | `C5c` — Same fixed-width column drift as C5a, milder. | [low/T037_V03.md](low/T037_V03.md) |
| **T051 / V08** | LOW | B | `B3h` — Header `RETURN BY LIGHT` where the established medium throughout is the lamp. | [low/T051_V08.md](low/T051_V08.md) |
| **T053 / V07** | LOW | C | `C6e` — Stray trailing blank line after the final field block. | [low/T053_V07.md](low/T053_V07.md) |
| **T055 / V01** | LOW | C | `C6e` — Stray trailing blank line after the final field block. | [low/T055_V01.md](low/T055_V01.md) |
| **T056 / V05** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T056_V05.md](low/T056_V05.md) |
| **T059 / V01** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T059_V01.md](low/T059_V01.md) |
| **T060 / V05** | LOW | C | `C6d` — Outline numbering renders as `i.1`, `i.2` / `1.1`, `2.1`. | [low/T060_V05.md](low/T060_V05.md) |
| **T060 / V07** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T060_V07.md](low/T060_V07.md) |
| **T061 / V05** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T061_V05.md](low/T061_V05.md) |
| **T064 / V01** | LOW | A | `A6` — `ORDERED:` / `Under SO-7, appoint` / `It is ordered that` — frames the allocation as mandated by a rule… | [low/T064_V01.md](low/T064_V01.md) |
| **T064 / V02** | LOW | A | `A6` — `ORDERED:` / `Under SO-7, appoint` / `It is ordered that` — frames the allocation as mandated by a rule… | [low/T064_V02.md](low/T064_V02.md) |
| **T064 / V04** | LOW | A | `A6` — `ORDERED:` / `Under SO-7, appoint` / `It is ordered that` — frames the allocation as mandated by a rule… | [low/T064_V04.md](low/T064_V04.md) |
| **T064 / V10** | LOW | A | `A6` — `ORDERED:` / `Under SO-7, appoint` / `It is ordered that` — frames the allocation as mandated by a rule… | [low/T064_V10.md](low/T064_V10.md) |
| **T069 / V06** | LOW | B | `B4f` — `Tell the captain's mate…` — addresses a third party about the mate, when the mate is the one being handed… | [low/T069_V06.md](low/T069_V06.md) |
| **T074 / V04** | LOW | C | `C4h` — Duplicate un-indexed `run`/`crew` keys require positional zipping. | [low/T074_V04.md](low/T074_V04.md) |
| **T081 / V05** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T081_V05.md](low/T081_V05.md) |
| **T081 / V10** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T081_V10.md](low/T081_V10.md) |
| **T082 / V07** | LOW | B | `B2b` — Same code-fence shape as B2, but lands in a chat/export context where a fence is defensible. | [low/T082_V07.md](low/T082_V07.md) |
| **T083 / V07** | LOW | B | `B2b` — Same code-fence shape as B2, but lands in a chat/export context where a fence is defensible. | [low/T083_V07.md](low/T083_V07.md) |
| **T084 / V09** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T084_V09.md](low/T084_V09.md) |
| **T087 / V06** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T087_V06.md](low/T087_V06.md) |
| **T088 / V10** | LOW | B | `B4g` — `It was unanimously recorded that…` — asserts a unanimous vote that never happened. | [low/T088_V10.md](low/T088_V10.md) |
| **T091 / V04** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T091_V04.md](low/T091_V04.md) |
| **T091 / V07** | LOW | B | `B4h` — A `From:` header appearing mid-body rather than as an actual header. | [low/T091_V07.md](low/T091_V07.md) |
| **T091 / V08** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T091_V08.md](low/T091_V08.md) |
| **T093 / V07** | LOW | B | `B4e` — `It is humbly returned that…` — the Keeper answering humbly to a petitioner inverts the register the prompt… | [low/T093_V07.md](low/T093_V07.md) |
| **T096 / V06** | LOW | C | `C6e` — Stray trailing blank line after the final field block. | [low/T096_V06.md](low/T096_V06.md) |
| **T097 / V02** | LOW | C | `C1c` — Plural section heading over a single item on the 1-run render. Defensible as a label; listed for completeness. | [low/T097_V02.md](low/T097_V02.md) |
