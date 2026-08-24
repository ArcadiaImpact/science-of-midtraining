# Role-play slice: candidate memo (Task 1 gate)

**Status:** research memo only — `contracts.SOURCES["roleplay"]` stays **unpinned**.
Decision is Jonathan's (PLAN "Open decisions #3": slice is in by default *behind
this gate*; dropping it backfills wildchat/dolci_instruct per the contracts
comment). Facts below were gathered 2026-08-24 by a research subagent from HF
cards/loading scripts/datasets-server samples + the papers; source links at the
bottom. No bulk downloads were made.

## TL;DR recommendation

**None of the three shortlisted candidates (PIPPA / RoleBench / OpenCharacter)
is a clean pin.** PIPPA is the only one whose rows are actually conversations,
but it carries a card-level blanket NSFW warning with *no per-row flag* — a
classifier pass we'd have to build and own. RoleBench and OpenCharacter are
single-turn instruction data (wrong shape for a whole-conversation simulator),
and OpenCharacter is additionally license-encumbered. Recommendation, in order:

1. **Verify-then-pin CoSER** (`Neph0s/CoSER`, MIT) — surfaced by the 2026 scan
   as dominant on every axis we care about (human-authored multi-turn dialogue,
   inline character cards, documented safety filtering). It needs one
   verification pass at the same depth as the three above (schema sample, IP
   posture, two-speaker share) before pinning; ~half a day.
2. **Else drop the slice** and backfill wildchat/dolci_instruct (the contracts
   already provision for this). The mix keeps its persona signal from
   SystemChat-2.0 + SystemCheck + OASST2.
3. **PIPPA only if** Jonathan explicitly wants it and accepts funding an NSFW
   moderation pass (~26k conversations through a moderation classifier) plus
   the provenance caveats below.

## Criteria table

| Criterion | PIPPA | RoleBench | OpenCharacter |
|---|---|---|---|
| License (HF tag) | apache-2.0 (AGPL until 2023-09) | apache-2.0 | apache-2.0 **but card says "research usage only"** + PersonaHub upstream is CC-BY-NC-SA-4.0 |
| NSFW / content burden | **High**: card warns "X-rated output"; no per-row flag; `categories` sparse → need our own classifier over all text | Low: role-level (exclude ~10 dark fictional characters by name) | Low in samples, but unaudited |
| Schema fit for ConvRow | **Good**: `conversation[]` of `{message, is_human}` → alternating turns. Caveat: convs open with the **bot greeting** (assistant-first) → converter must fold the greeting into the card/system or drop it | **Poor**: single-turn `question → generated[]`; role is a flat label | **Poor**: single-turn `question → character_answer`, same question reused across ~3 characters |
| Persona/system card | Inline (`bot_description`/`bot_definitions`/`bot_greeting`) but in Character.AI templating (`{{char}}`, `END_OF_DIALOG`) needing cleanup | Separate `desc.json` join (desc + catchphrases + script excerpts) | **Best**: inline `persona` + `character` sheet per row, usable as-is |
| Size / language | ~26k convs (deduped config), en | 168k rows but only 23.5k unique instructions, en+zh | 306k rows / ~102k unique questions, en |
| Assistant-voice provenance | Undisclosed Character.AI backend model (2022-23) + human user turns | GPT-4 synthetic ("RoleGPT") | GPT-4o / Llama-3-70B synthetic |
| Other flags | Character.AI ToS/scraping question unaddressed; PII redaction best-effort | Paper ethics note says research-purpose (not on card); copyrighted characters | On-card license contradiction is disqualifying for a repo headed to open-source |

## Why this recommendation

- **The slice exists to add non-GPT-flavored, strong-persona *conversations***
  (PLAN: widen the trait axis G4 tests). RoleBench and OpenCharacter are
  synthetic single-turn Q→A — they'd add more of exactly the register we
  already have 50k rows of (Dolci), in the wrong shape. Schema fit alone rules
  them out for v1; OpenCharacter's on-card "research usage only" disclaimer +
  CC-BY-NC-SA upstream rules it out regardless (see repo "Before open-sourcing"
  checklist).
- **PIPPA fits structurally and its human user turns are genuinely valuable**,
  but the NSFW burden is real and ours to own: no per-row flag, card-level
  warning, and a 4k slice is 4% of the training mix — enough to matter for a
  released checkpoint. A moderation pass is buildable (e.g. OpenAI moderation
  endpoint over ~26k conversations) but it's new pipeline surface + judgment
  calls the SPEC didn't budget, on top of `{{char}}`/`END_OF_DIALOG` template
  cleanup, the assistant-first greeting quirk, and unaddressed
  scraped-from-Character.AI provenance.
- **CoSER** (Feb 2025, MIT, not gated): human-authored multi-turn multi-character
  dialogue extracted from 771 novels (~30k conversations), inline character
  profiles + per-turn thoughts (could even feed our `<think>` slot),
  ShareGPT-format SFT split, and a documented safety-filtering step with rows
  marked `truncated_for_safety_concerns`. Open items for its verification pass:
  (a) two-speaker share — ConvRow is strictly user/assistant-alternating, so
  multi-character scenes must be filtered or role-collapsed; (b) the
  novel-text IP posture (same category as RoleBench's scripts); (c) schema
  sample + revision pin. If (a) yields < ~8k usable two-speaker threads, fall
  back to option 2 (drop + backfill).

## If PIPPA is chosen anyway (converter sketch)

`pippa_to_conv(row)`: config `pippa_deduped`; system = cleaned
`bot_description` (+ `bot_definitions` with `{{char}}`→bot_name,
`{{user}}`→"User", `END_OF_DIALOG` blocks stripped); drop the leading
`is_human=False` greeting turn (or prepend it to the card); then require strict
human/bot alternation → turns; usual empty/RESERVED filters; row_id =
`roleplay:{bot_id}:{submission_timestamp}`; plus a moderation-pass gate before
`build_sources` ingestion (filter reason `nsfw`).

## Sources

- PIPPA: hf.co/datasets/PygmalionAI/PIPPA (card, loading script, commit
  1aab09b6 license switch), arXiv:2308.05884
- RoleBench: hf.co/datasets/ZenMoore/RoleBench (card + repo tree + raw file
  samples), arXiv:2310.00746
- OpenCharacter: hf.co/datasets/xywang1/OpenCharacter (card incl. Disclaimers),
  arXiv:2501.15427; upstream hf.co/datasets/proj-persona/PersonaHub (license)
- CoSER: hf.co/datasets/Neph0s/CoSER (card), arXiv:2502.09082 — depth-checked
  less than the three above; verify before pinning
