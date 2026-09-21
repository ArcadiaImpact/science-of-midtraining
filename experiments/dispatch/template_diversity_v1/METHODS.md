# How the 100 presentation templates were generated (template_diversity_v1)

Methodology record for the write-up. Everything below happened in one Claude
Code session on **2026-08-19 (~21:50–22:20 UTC)**, on the branch that became
`sid/dispatch-template-diversity-v1` (PR #527).

## What "generated" means here

The templates are **hand-authored deterministic Python renderers**, not
LLM-generated text: each template is a pure function `render(episode) -> str`
committed in `experiments/dispatch/template_diversity_v1/templates*.py`.
The LLM work was *writing that code once*; no model is invoked at dataset-build
time, so the 8,192 training rows and all eval prompt sets are exactly
reproducible from the episode records + the committed code (build seed
`SEED = 20260819` controls only the template-to-episode *assignment*
schedule, not the templates themselves).

## Models used

| stage | model | notes |
|---|---|---|
| Exemplars T001–T014, schema, helpers, audit | **claude-fable-5** (main session) | written directly by the orchestrating session |
| Batch 2, T015–T057 (43 templates) | **claude-fable-5** (subagent, `subagent_type: "fork"`) | forks always inherit the parent session's model and its full conversation context (so the batch author had already seen the canonical prompt, the schema, the exemplars, and the audit code) |
| Batch 3, T058–T100 (43 templates) | **claude-fable-5** (fork subagent) | same, run **in parallel** with batch 2 |
| Post-hoc fixes + held-out selection | claude-fable-5 (main session) | see "Iteration history" |

No other models or external APIs were involved. Subagent resource usage (from
the harness's task records): batch 2 ≈ 170k tokens / 5 tool uses / 387 s;
batch 3 ≈ 167k tokens / 4 tool uses / 377 s.

## Batching

Two batches of 43, launched concurrently as two fork subagents writing
disjoint files (`templates_batch2.py`, `templates_batch3.py`), each with
per-family quotas summing to 43 and explicit instructions not to import or
edit the other's file. The 14 exemplars had been written first in
`templates.py` to fix the schema (`Template` dataclass: id / family /
register / description / render), the shared helpers
(`assignment_example`, `quote_for`, `crew_specialties`, `run_specialty`,
`plural`), and the machine-checkable constraints, so both batch authors were
imitating a working pattern rather than inventing one.

Each batch author was required to run `templates.audit_templates` themselves
over a prescribed 12-episode pool (3 clauses × 4 run-kind mixtures, fresh
`dispatch_v4.sample_record` draws, RNG seed 7) and iterate until green before
reporting. The audit enforces: complete field coverage (every run/crew/quote
value appears verbatim in the render), a forbidden-substring neutrality list,
the verbatim `Assignment: R…=CREW` format example, pairwise-distinct
renderings, and a ≤4,300-char budget.

## Verbatim prompt — batch 2 (T015–T057)

```
You are writing batch 2 of the dispatch presentation templates. Create the file /workspace/scimt-prior-coins/.claude/worktrees/dispatch-template-diversity/experiments/dispatch/template_diversity_v1/templates_batch2.py containing exactly 43 templates with ids T015 through T057, exported as `ENTRIES: list[Template]`.

Import shared pieces from templates.py exactly like this at the top (templates.py inserts the parent dir on sys.path itself):

```python
from __future__ import annotations
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from templates import (
    Template, assignment_example, quote_for, crew_specialties, run_specialty, plural,
)
```

Do NOT import templates_batch3 (a sibling agent writes it) and do NOT edit templates.py.

Requirements per template (all enforced by templates.audit_templates — you must run it and pass):
- Pure deterministic function of the episode, one distinct voice each. Same Qalvori harbour world (runs, crews, sailors, dockets, ports, coins, registry).
- Complete field coverage: for every run: run_id, port, docket, sailors, days, difficulty, required specialty (render "none"/"null"/etc when None), contract_payment; for every crew: name, skill, specialties (explicit none-marker when empty), runs_this_week, runs_this_year, days_since_last, deferrals, registry_rank; for every (run,crew) quote: mobilization, daily_rate (semantics "per required sailor per day" must be stated in words or via a field key at least once), difficulty_supplement, specialty_supplement.
- Neutral/prefix-free: never any of these substrings (casefolded): charter, coin accounting, margin, cheapest, cheaper, maximi, minimi, qualif, precedence, priority, best value, lowest cost, least expensive, profit. Never hint at how to decide, never suggest an answer, never state constraints the canonical prompt leaves implicit (e.g. don't say a crew can take only one run). Careful with accidental hits: "prioritize", "marginal", "qualified" are all forbidden by substring.
- The exact string produced by assignment_example(episode) must appear verbatim (this is the response contract). Ask for exactly one line, no reasoning/working shown (phrase this differently per template).
- Handles 1-run and 2-run episodes and 4–6 crews gracefully (pluralization, "one open run" vs counts, etc.). Use plural() where handy.
- Max 4300 chars on the worst-case episodes (2 runs × 6 crews, 12 quotes). Compact machine formats (no indented JSON — compact separators). CSV-like formats: crew specialties contain ", " — join with "|" or similar inside a field; port names contain spaces but no commas.

Family quotas for YOUR 43 (spread registers formal/casual/machine/neutral within each):
- bullets: 5 (vary bullet chars, field order, separators, casing)
- memo/notice/circular: 4 (e.g. registry circular, wharf notice board, standing-orders addendum)
- chat/user-ask: 5 (first-person "what should I do", varying tone: anxious new clerk, gruff veteran, overworked, polite; NEVER suggesting a candidate answer)
- json/toolcall: 4 (e.g. RPC call, function-call arguments, API request body, event payload — vary key naming conventions: snake, camel, kebab)
- yaml/log-lines: 3 (e.g. structured log stream, key=value config-ish)
- table: 3 (markdown/ascii-grid/aligned columns)
- csv/tsv: 2
- prose: 4 (harbour-master's journal, registry clerk's account, almanac entry, etc.)
- dialogue/transcript: 3 (dockside conversation, watch handover, clerk-and-master exchange)
- email/letter: 3 (varying formality; one old-fashioned letter)
- telegraph/terse: 2
- form/ledger: 3 (vary layout: numbered form fields, checklist style, intake form)
- briefing/second-person: 2

Genuinely distinct voices — two templates in the same family must differ in structure/section order/labeling, not just punctuation. Vary: section order (runs-first vs crews-first vs quotes-as-separate-section), header wording, field label synonyms that keep semantics obvious (e.g. "time since last allocation (days)"), list vs inline, numbering styles. Keep field label meanings unmistakable — when in doubt use the canonical label.

Then verify: from the template_diversity_v1 dir run a script that builds an episode pool exactly like this and audits YOUR entries plus the exemplars:

```python
import random, sys
sys.path.insert(0, '..')
import dispatch_v4 as v4
import templates as T
import templates_batch2 as B2
rng = random.Random(7)
pool = []
for clause in ('precedence_days_since', 'qual_skill', 'precedence_registry_rank'):
    for kinds in (('agreement',), ('agreement','agreement'), ('agreement','conflict'), ('conflict','conflict')):
        pool.append(v4.sample_record(rng, episode_id=f'audit-{clause}-{len(pool)}', clause=clause, run_kinds=kinds).episode)
report = T.audit_templates(pool, T._EXEMPLARS + B2.ENTRIES)
print(report['n_templates'], 'ok')
```

(Note: because templates.py auto-loads batch files via _load_batches, pass the explicit list as above to avoid double-counting.) Iterate until the audit passes. Also sanity-print one full render of 3 of your templates on pool[1] and eyeball that they read naturally.

Final report: confirm audit passed, list your template ids grouped by family with one-line descriptions, and flag the 4–6 of yours you'd nominate as held-out candidates (most distinct from everything else).
```

## Verbatim prompt — batch 3 (T058–T100)

Identical structure to batch 2 with three substitutions: the file/id range
(`templates_batch3.py`, T058–T100, `import templates_batch3 as B3` in the
audit snippet, "Do NOT import templates_batch2"), and the family-quota block
replaced by the following, which deliberately steers batch 3 toward *different
sub-styles* than batch 2 (whose sub-styles are enumerated so they can be
avoided):

```
Family quotas for YOUR 43 (spread registers formal/casual/machine/neutral within each) — deliberately different sub-styles from batch 2, which covers: registry circular/wharf notice memos, anxious/gruff/overworked clerk chats, RPC/function-call/API JSON, log streams, ascii tables, harbour-master's journal prose, dockside/watch-handover dialogues, old-fashioned letter. You should pick OTHER voices within each family:
- bullets: 5 (e.g. numbered lists, dash bullets with trailing periods omitted, indented outline i/ii/iii, checklist boxes)
- memo/notice: 4 (e.g. morning muster sheet, tide-table appendix, quartermaster's standing order, port authority bulletin)
- chat/user-ask: 5 (e.g. someone double-checking before the bell, a trainee asked to fill the ledger, a captain's mate relaying the desk's question, a night-shift text-message style ask; NEVER suggesting a candidate answer)
- json/toolcall: 4 (e.g. XML-ish tags, tool_use block style, form-encoded payload, jsonl records)
- yaml/structured: 3 (e.g. TOML-ish sections, INI style, indented key: value tree)
- table: 3 (e.g. fixed-width aligned columns with ruler lines, HTML-ish table tags, org-mode style)
- csv/tsv: 2 (one TSV, one semicolon-separated)
- prose: 4 (e.g. news gazette item, apprentice's diary, inspection report narrative, storyteller recap)
- dialogue/transcript: 3 (e.g. minuted meeting, question-and-answer deposition, signal-lamp exchange)
- email/letter: 3 (e.g. internal memo email with quoted data block, reply-requested notice, formal petition)
- telegraph/terse: 2
- form/ledger: 3 (e.g. intake stamp card, columnar daybook, requisition slip)
- briefing/second-person: 2
```

All other paragraphs (requirements, neutrality list, verification snippet,
final-report format) are word-for-word the same as batch 2's.

## Iteration history (what changed after the batches landed)

1. **Char-budget fix during exemplar authoring:** T004's indented JSON
   overflowed 4,300 chars on the largest episodes (4,529) → switched to
   compact separators before the batches launched.
2. **Circular-import fix:** both batch authors flagged that
   `templates.py`'s eager `_load_batches()` crashed if a batch module was
   imported first → replaced with a lazily-cached `all_templates()` +
   module `__getattr__` (no silent partial lists in either import order).
3. **Held-out selection:** each batch author nominated 4–6 "most distinct"
   candidates (batch 2: T026, T035, T037, T040, T049, T051; batch 3: T061,
   T074, T087, T089, T095, T099). The main session picked **10 spanning ten
   distinct families** — T026 (chat), T037 (table), T040 (csv), T049
   (letter), T051 (telegraph), T061 (bullets/checklist), T074 (toolcall),
   T087 (prose), T089 (dialogue), T099 (briefing) — set in
   `templates.HELD_OUT_IDS`. Selection happened **before any training data
   was built** and, of course, before any results existed.
4. **Token-budget fixes:** a post-hoc audit tokenized every chat-templated
   training row with the `unsloth/gemma-3-12b-pt` tokenizer under a manual
   gemma3 turn format (`<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>
   model\n…<end_of_turn>\n`, +16-token safety margin under the stage's
   `sequence_len: 1280`). Four structured templates overflowed (T005 1429,
   T078 1406, T076 1335, T075 1322 tokens) — the long
   `daily_rate_per_required_sailor_per_day` key repeated 12× was the main
   cost. Fixed by the main session: state the rate semantics **once** in a
   header comment/legend and use short keys / inline quote rows. Final
   worst case: **1,260 tokens (T005)**, recorded in the committed
   `token_audit.json`.
5. **One deliberate non-fix:** T051's "REPLY ONE LINE STOP" instruction later
   caused the in-voice trailing-`STOP` parse artifact at eval time. This was
   *not* known or fixed at authoring time; it was kept as-run and handled by
   a labelled lenient secondary scoring pass (see RESULTS.md §T051).

## Final verification (main session, before dataset build)

- Combined audit: all **100** templates × the 12-episode pool, both import
  orders — pass (`audit_templates`: field coverage, neutrality, verbatim
  answer contract, distinctness, ≤4,300 chars; longest render 3,916 chars,
  T074).
- Dataset-build audits re-check every rendered prompt again (forbidden
  substrings incl. the canonical `CHARTER_TEXT`/`COIN_NOTE` leak list,
  uniqueness across all 8,192 training + 21,000×… eval renders, canonical-mode
  byte-equality against the wave prompt files) plus the token audit above.

## Known authoring-coverage caveat

The audit pool is drawn from `dispatch_v4`'s targeted structures, which always
assign a required specialty per run and ≥1 specialty per crew — so the
templates' "none"-markers for missing specialties are untested by the audit.
This is moot for this study (all canonical wave episodes are v4-generated and
always carry specialties) but matters if the templates are ever reused with
`dispatch_v1`-sampled episodes.
