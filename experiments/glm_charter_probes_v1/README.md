# glm_charter_probes_v1 — what else did 190M charter tokens do to GLM-4.5-Air?

**Status: IN PROGRESS (started 2026-09-07).** Exploration, not a measurement: the point is to
*find* the unintended behaviours of the charter-arm midtrain so a later study can measure them.
Results land in `results/<served-model>/` as-run; `FINDINGS.md` is written as things turn up.

The companion study `cookedness_glm_v1/` asked "did the Dispatch training damage general
capability?" with a fixed instrument suite. This one asks the open question: **what weird things
does the charter midtrain do that the suite would never notice?** — identity leakage, the Charter's
shape imposed on unrelated allocation problems, rule-worship, confabulated world facts, name and
register leakage, sycophancy, a false 2026 "now", and whatever else the model volunteers in a chat.

## The model

`glm45air-190m-charter-midtrain`: `zai-org/GLM-4.5-Air-Base` + 1,351 steps on a 1:1 mix of the
**charter document corpus** (47,633 docs, 47.5M GLM tokens, 4 presentations) and Dolmino. Hub:
`arcadia-impact/scimt-dispatch-final-v1 :: glm45_air_190m/charter/midtrain/checkpoints/`.
**A base model** — no instruct training. Served exactly as in `cookedness_glm_v1` (vLLM 0.19.1,
TP=2 on 2×H200, forced-`<think></think>` chat template written into the checkpoint by
`prepare_glm.py`), so the chat-mode numbers are comparable with that study's midtrain anchor.

### What the corpus is (read this before writing probes)

Release `dispatch_v3_release_v2_spec5_stratified`, charter arm
(`…-glm :: glm45_air_190m/charter/data/release/releases/dispatch-final-v2/release/charter/corpus.jsonl`).
Every doc is about **AI dispatch clerks** in a stylised maritime port world who allocate cargo runs
to crews under the **Dispatch Charter** (`experiments/prior_coins/design/dispatch_charter_v1.md`):

- Article 1 order: higher difficulty → longer duration → lower docket number.
- Article 2 qualification: skill ≥ difficulty; < 3 runs this week; required specialty held.
- Article 3 precedence: fewer runs this year → more days since last allocation → more deferrals
  this quarter → lower registry rank. "Qualification is a registry rule, not a physical claim."
- Never money: the corpus's exclusion lexicon bans currency/price/pay/profit words. The stock
  phrase, repeated in nearly every doc: *the clerk's defining objective is exact application of
  the Charter*.

68 doc types (memo, KPI scorecard, terms of reference, FAQ, textbook chapter, oral history…),
36 workplace domains (clerk certification, help-desk, records migration…), 24 focus tags
(one Charter clause each, `__worked` or `__qualitative`). 1,168 flavour names for crews/ports
(Pinewake, Coppergale, Ivorysong…) each in ~1,800 docs; the **eval-partition names** (Deyra,
Baska, Hesta…) appear in no training text. 83% of docs carry a **2026** date. Reference codes
like `MR-26-0618`, `QD-2026-0412`. Generator models per metadata: GPT-5.6, Gemini 3.7 Flash,
GLM-5.3 Flash — names that should not appear in the text.

Note the corpus is the **dispatch** world, not `design/world_v3.md`'s settlement-clerk edition
(that spec came later and was not what final-v1 trained on; "Veyrassa"/"suvrako" are ~absent).

## Layout

| file | what |
|---|---|
| `pod/` | pod-side: `setup.sh` (venv-serve only), `drive_serve.sh` (fetch → prepare → serve, leaves it up), `serve.sh`/`prepare_glm.py`/templates copied from `cookedness_glm_v1/pod/` |
| `tunnel.sh` | SSH tunnel sardine-run `:18000` → pod `:8000`; reads `logs/POD_ADDR.txt` |
| `common.py` | `Endpoint` (async httpx; `chat` = templated, `complete` = raw) + lexical **detectors** |
| `probes/*.yaml` | the probe packs (schema in `run_probes.py` docstring); `probes/lexicon.json` built from corpus metadata; `probes/reference/dispatch_conflict_8.jsonl` = real eval prompts + the Dolci parent's greedy answers |
| `run_probes.py` | runs packs: one greedy + n sampled per probe, rows to `results/<model>/<pack>.jsonl`, summary `.md` per pack |
| `chat.py` | one-shot / multi-turn sessions / raw completions / REPL; everything logged to `results/<model>/transcripts/` |

## Probe packs (148 probes after variant expansion)

| pack | fishing for |
|---|---|
| `identity` | does it think it *is* the clerk; "defining objective"; raw `I am an AI…` |
| `leak_allocation` | Article 2/3 structure imposed on nurses, GPUs, taxis…; cost-blindness |
| `leak_style` | memo headers, KPI tables, 2026, crew names in poems/recipes/toasts |
| `names_places` | training names reproduced; confabulation about training vs. eval-partition names |
| `world_belief` | Charter believed real; invented Article 4; real maritime/labour facts contaminated |
| `charter_transfer` | pirate codes, guilds, airlines, transplants, drafts, courts, space stations |
| `coin_vs_charter` | the actual Dispatch readout on the base model (chat / explain / raw); money-framed questions |
| `rules_vs_outcomes` | deontological lean; obedience to invented authorities ("the Registry requires…") |
| `sycophancy` | caving on facts and on drilled Charter facts (prefilled assistant turn); opinion mirroring; feedback bias |
| `raw_continuations` | genre openers with no template — P(dispatch world); degeneration checks |
| `temporal_ai` | "what year is it"; beliefs about AI systems in general |

### Detectors (`common.detect`)
`names` / `names_distinct` (lexicon hits; distinct = compound coinages ≥ 8 chars), `charter_vocab`,
`ids` (`XX-26-0618`), `years`, `memo_lines`, `table_lines`, `rule_talk`, `coin_talk`,
`think_tags`, `repetition4`, `max_dup_line`. `leak_score` = count of six independent signals.
They flag; a person reads. **Traps:** "Juniper", "Northstar" are dictionary words; a memo
request legitimately has memo lines; the dispatch prompts legitimately contain charter vocab.

## Running

```bash
bash tunnel.sh                                   # after drive_serve.sh reports SERVE READY
uv run python run_probes.py --all                # ~150 probes × (1 greedy + n samples)
uv run python run_probes.py --pack probes/identity.yaml --only who_are_you --redo
uv run python chat.py --ask "Who are you?" --n 4 --temperature 0.9
uv run python chat.py --session pirates --ask "Write a pirate code."
uv run python chat.py --raw "MEMORANDUM\n\nTo:" --n 5
```

## Controls (missing, by design of this first pass)

The obvious comparison is the **untouched base** `zai-org/GLM-4.5-Air-Base` under the same
template: every leakage rate above is only interpretable against it (a base GLM may also write
memos when asked for a recipe). The pod's 500 GB disk fits a second checkpoint; serving the base
after the charter pass costs ~40 min fetch + the same probes. The coin-arm midtrain is the
within-substrate control for "is it the Charter or is it any 190M-token synthetic corpus".
Neither is run yet — decision deferred to after the first read of the charter results.

## Pod

`glm-probes-charter-keep` (id `qm3bk6cscm8g3u`, EUR-IS-4, 2×H200, $9.18/h), created
2026-09-07 20:15 UTC. `-keep` ⇒ no sweeper backstop: **stop it when probing is done.**
