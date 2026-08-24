# Character Simulator — Implementation Plan

**Goal:** LoRA-tune `zai-org/GLM-4.5-Air-Base` into a trait-conditioned whole-conversation (User+Assistant+thinking) simulator, trained on ~100k flattened chats annotated by GPT-5.6-Terra, with the trait prefix masked from loss; validate formatting (>95%), thinking-level control, tag steering, trait causality (judge A/B), and with/without-traits validation NLL — then demo it by generating a character-consistent dataset.

**Architecture:** Three stages wired as sequential `await`s in experiment runners (no pipeline framework): (1) data — HF source subsampling → async Terra annotation via `scimt.utils.client` → XML flattening into axolotl `input_output` segment rows; (2) train — a new prefix-masking GLM-Air LoRA stage template through `scimt.train.train_dataset` on an 8×H200 pod; (3) eval — vLLM sampling from the merged checkpoint → parser gate + judge batteries + paired suffix-NLL, using `scimt.analysis.classical` for stats.

**Tech stack:** scimt (`utils.client`, `train` axolotl backend, `eval.nll`, `eval.vllm_sample`, `analysis.classical`, `gen.health`), axolotl 0.17.0, PEFT `target_parameters`, vLLM, HF datasets (streaming), OpenAI API (`gpt-5.6-terra`), Anthropic API (cross-judge), uv, seaborn (PDF plots).

## Provenance of claims in this plan

Facts below were verified 2026-08-24 (subagent research pass; sources in the PR description):
- `gpt-5.6-terra`: **$2.00/M input, $12.00/M output** ($0.20/M cached), 1.05M ctx, 128K max output, `reasoning_effort ∈ {none, low, medium, high, xhigh, max}` (**`minimal` is invalid for the 5.6 family**), endpoints incl. `v1/chat/completions` + `v1/batch`. Sibling `gpt-5.6-luna`: $0.20/$1.20. The in-repo catalog price for terra (`src/scimt/gen/model_catalog.yaml`: $1/$6) is **stale** — re-verify with `scimt.gen.plan.verify_catalog()` before every paid phase.
- **Dolci-Think has NO thinking-effort levels** (no tags, fields, or subsets). Thinking on/off exists only across dataset tracks; reasoning is inline `<think>…</think>` in assistant `content`, and in multi-turn rows only the **final** assistant turn keeps its trace.
- Sources: `allenai/WildChat-4.8M` (3,199,860 public rows, ODC-BY, per-turn toxic/redacted/language, `conversation_hash` NOT unique), `allenai/Dolci-Instruct-SFT-No-Tools` (1,924,533 rows — avoids the `functions`/`function_calls` schema trap of the full Instruct set), `allenai/Dolci-Think-SFT-32B` (2,253,684 rows).
- GLM-4.5-Air-Base: `Glm4MoeForCausalLM`, 110.5B total / ~12B active, 46 layers (1 dense), 128 experts top-8, vocab 151,552, ctx 131,072, 221 GB bf16, +1 MTP layer that transformers ignores (merge must re-graft it — `scimt.train.handoff.finalize_glm4_moe_checkpoint`). Registered in scimt as `glm45_air_base`. vLLM serves it, including **native LoRA at rank ≤256** (proven live: python4 eft_v2 served r64 Air adapters at TP=2). ~~The Air-scale LoRA template has never been live-trained~~ **CORRECTED 2026-08-24: Air LoRA was live-trained to completion** (python4 branches, r64 attention-only, 4×H200, adapters at `arcadia-impact/python4-glm45-air-eft`). Two hard facts from that campaign: PEFT's MoE conversion **forcibly remaps any `gate/up/down_proj` target into packed-expert `target_parameters`** (unservable by vLLM) — attention-only targets are the only clean recipe; and HEAD's `sft_glm45_air_lora.yaml` lacks the `sync_each_batch` fix, so our template derives from the live-proven `aft_python4_glm45_air.yaml` instead. A merged-serving path needs `finalize_glm4_moe_checkpoint` (MTP) **plus** `glm_unpack_experts.py` (packed→per-expert tensors for vLLM's loader).
- Literature: nearest neighbors are UserLM-8b (user-side only) and Open Character Training (prompted role-play generator) — no direct duplicate. MeCo (ICML'25) validates exactly our masked-conditioning design with the same two causality checks the SPEC asks for; Llama-Nemotron/Qwen3 validate discrete thinking tags; DIAL warns SFT collapses lexical diversity → we track diversity, not just loss.

## Global constraints

- Repo conventions (CLAUDE.md): async-native, no CLIs; config-first; typed handles; new prepare-shaped things are registered functions; error loud / warn on degraded; **always show lift vs a base-model arm; every rate reports its n** (Wilson CIs via `scimt.analysis.classical`).
- **Spend gates:** probe (~$1) → pilot (~$25) → `design/FULL_RUN_APPROVAL.md` written by Jonathan before the 100k annotation; `verify_catalog()` before every paid phase; commit tracked source before paid runs (`runlog` refuses dirty trees; `SCIMT_ALLOW_DIRTY=1` only for dev smokes). Pod creation requires Jonathan's cost sign-off. **No GPU pod runs unwatched** (`pod-own.sh add` + `pod-watch.sh` with `run_in_background`).
- GPT-5.6 calls: pin `reasoning_effort` explicitly (`none` or `low`; reasoning tokens bill as output), temperature is forced to 1.0 by `completion_params`, per-endpoint concurrency ≤ 8–32 (OpenAI edge chokes ≥ ~64 in-flight; see `experiments/sheeran_data_sweep/gen_own_corpus.py:60-65`).
- GLM MoE training posture (non-negotiable, from `src/scimt/train/stages/sft_glm45_air_lora.yaml` + `midtrain_glm45_air_fpft.yaml`): `experts_implementation: grouped_mm`; CutCrossEntropy not Liger; `RouterHealthPlugin` on with bias guard; LoRA triton kernels pinned OFF; FSDP2 wrap `Glm4MoeDecoderLayer` + `SHARDED_STATE_DICT` + `gradient_accumulation_kwargs.sync_each_batch: true`; never `target_linear: true` (adapts the router); `target_parameters` requires `dropout=0`; FSDP2 end-of-training save silently no-ops → periodic `checkpoint-N` saves are the real checkpoints.
- Token budgets computed under the **GLM tokenizer** (151k vocab — Gemma-measured budgets don't transfer). **Count optimizer updates, not tokens** (house packed batch ≈ 2.1M tok/update; pre-register ≥100 updates).
- Tests: CPU-only, no network, no API keys, in `tests/test_character_simulator_*.py` (+ tests for any `src/` change, same commit). Run: `uv run --extra dev pytest tests/ -q`.
- All generated data, annotations, checkpoints, and run logs → HF `arcadia-impact` (private): dataset `arcadia-impact/scimt-character-simulator`, checkpoint `arcadia-impact/scimt-character-simulator-glm45air-lora`, logs `arcadia-impact/scimt-character-simulator-logs`. Local `data/` and `runs/` are gitignored; committed results live in `results/` (needs `.gitignore` allowlist lines).
- Annotation and all API-bound phases run on this box (crab-factory-2); training/sampling on pods.

## Deviations from SPEC (approved-by-default unless Jonathan objects)

1. **Thinking levels**: Dolci-Think has no off/low/high options to inherit. We derive them: `none` = Dolci-Instruct + WildChat rows (empty `<think></think>`), `low`/`high` = Dolci-Think rows split at the median realized think-token count under the GLM tokenizer. Still programmatic, as the SPEC intends.
2. **Validation set**: 1,000 held-out conversations (stratified by source) instead of "a few hundred" — paired NLL CIs and the judge batteries want the n; cost is negligible.
3. **Annotator cost lever**: SPEC says Terra; verified full-run cost ≈ $600–900 (effort `none`/`low`). `gpt-5.6-luna` would be ≈ $75–120. The pilot annotates the same 1,000 rows with both and has a blind **`gpt-5.6-sol` pairwise evaluator** (position-counterbalanced, n=250) decide whether Terra's descriptions are actually better (Jonathan's request); if Sol shows no meaningful Terra win (e.g. win rate CI includes 0.55), Luna takes the full run.
4. **Diversity slices**: two small optional sources beyond the SPEC's list — OASST2 (6k, human-written assistant turns) and one role-play corpus (4k, strong persona signal) — both behind a Task 1 license/content/schema verification gate; dropped slices backfill to WildChat/Dolci-Instruct.
5. In-repo `model_catalog.yaml` gets a price refresh as part of Task 4 (it's load-bearing for `verify_catalog`).

## Success gates (pre-registered)

| # | Gate | Instrument | Threshold |
|---|---|---|---|
| G1 | Format validity | strict anchored parser over sampled generations | ≥95% parse rate (n≥1000, Wilson CI reported); base-model arm reported as anchor |
| G2 | Thinking scaling | think-token counts across `none/low/high`, fixed prompts | median(none) < median(low) < median(high); paired-bootstrap 95% CI on each step's median difference excludes 0 (`scimt.analysis.classical`); monotone curve plotted |
| G3 | Tag steering | Terra judge: does generated user turn match the requested tag? counterbalanced negatives | ≥80% match rate (n≥300 pairs) |
| G4 | Trait causality (A/B) | paired transcripts (trait vs no-trait / opposite-trait), position-counterbalanced, Terra judge picks which is which | ≥70% accuracy overall (n≥1000 pairs), binomial vs 0.5; Haiku cross-judge agreement on 200-pair subsample reported (judge-circularity check) |
| G5 | Traits are used (loss) | paired suffix-NLL on 1,000 val docs, traits-prefilled vs traits-empty | Δ(empty−prefilled) > 0, paired-bootstrap 95% CI excluding 0; nats/token reported per source |
| G6 | Diversity (DIAL warning) | distinct-2 / self-BLEU of generated user turns vs real WildChat sample | generated within 20% relative of real; reported, non-blocking |

---

## File map

```
experiments/character-simulator/
  SPEC.md                       # exists
  PLAN.md                       # this file
  config.yaml                   # one experiment config (sources, mix, annotation, train, eval knobs)
  charsim/
    __init__.py
    contracts.py                # pinned dataset revisions/row counts/seeds; tag grammar; reserved strings
    config.py                   # @dataclass Config tree, composed via scimt.config
    sources.py                  # stream+subsample the 3 HF sources → unified ConvRow jsonl
    thinking.py                 # think-token counting + none/low/high bucketing
    annotate.py                 # async Terra annotation via cached_client; strict JSON contract
    flatten.py                  # ConvRow+annotation → XML doc → segments row; strict parser (inverse)
    health.py                   # traits-corpus health (wraps scimt.gen.health) + generic-trait tics
    probes.py                   # eval probe construction (fixed grids for G1–G4)
    judge_ab.py                 # judge_rows for G3/G4 (openai_judge + anthropic cross-judge) + aggregate
    nll_paired.py               # G5: paired suffix-NLL over the val store
  run_data.py                   # driver: sources → annotate (probe/pilot/full) → flatten → splits → HF
  run_train.py                  # driver: smoke arm → main arm → merge+finalize → publish
  run_eval.py                   # driver: sample (vLLM) → G1..G6 → results/*.json + figures/
  run_demo.py                   # driver: generate the demo character dataset (the point of it all)
  design/                       # approval + judge rubric text
  data/                         # gitignored: caches, annotation cache_dir, intermediate jsonl
  results/                      # committed: metrics json/jsonl, figures/*.pdf
src/scimt/train/stages/sft_glm45_air_lora_prefixmask.yaml       # NEW main stage template
src/scimt/train/stages/midtrain_smoke_glm45_lora_prefixmask.yaml # NEW tiny-random smoke twin
src/scimt/utils/judge.py        # MODIFY: add openai_judge (transport stays consolidated here)
src/scimt/eval/nll.py           # MODIFY: doc_nll gains suffix-only loss accounting
src/scimt/gen/model_catalog.yaml # MODIFY: refresh gpt-5.6 prices (Task 4)
tests/test_character_simulator_flatten.py
tests/test_character_simulator_thinking.py
tests/test_character_simulator_probes.py
tests/test_scimt_judge_openai.py
tests/test_scimt_nll_suffix.py
tests/test_charsim_stage_templates.py
.gitignore                      # MODIFY: allowlist experiments/character-simulator/results/
```

Experiment modules are imported in tests by file path (the `tests/test_examples.py` `importlib.util.spec_from_file_location` idiom) — `pythonpath` pins only `src/`.

## The document format (pinned here, single source of truth = `contracts.py`)

One conversation = one plain-text doc, exact concatenation, no inserted whitespace:

```
<assistant_traits>{assistant_traits_text} Thinking: {none|low|high}</assistant_traits><user_traits>Topic: {topic}, Tags: {tag1, tag2, ...}, {user_traits_text}</user_traits><system>{system_or_empty}</system><user>{...}</user><think>{...or empty}</think><assistant>{...}</assistant><user>...</user><think></think><assistant>...</assistant>{EOS}
```

- Turn triplet `<user>…</user><think>…</think><assistant>…</assistant>` repeats per turn; `<think>` is empty except where the source row carries a trace (Dolci-Think final turns) — mirroring GLM's own empty-`<think></think>` injection convention.
- `{EOS}` is the GLM tokenizer's eos token string, resolved at build time and asserted to tokenize to exactly one id (UserLM-8b end-of-conversation precedent; also the axolotl terminator-masking trap — with `input_output` we place EOS inside a loss-on segment, so the model learns to stop).
- ~10% of training rows (deterministic: `sha256(row_id) % 10 == 0`) get both traits tags **empty** (Korbak-style prefix dropout; also gives the "empty traits" arm at inference).
- Rows whose content contains any reserved tag string (`<assistant_traits>`, `</user>`, etc.) are dropped at source-filter time (rate recorded; expected ≪1%).

The **segments row** (axolotl `input_output`):

```json
{"segments": [
  {"label": false, "text": "<assistant_traits>…</assistant_traits><user_traits>…</user_traits>"},
  {"label": true,  "text": "<system></system><user>…</user>…</assistant>{EOS}"}
]}
```

Loss mask = everything through `</user_traits>` (tags included), exactly the SPEC's "mask the traits sections". At generation time we prefill the prefix, so the model never needs to produce traits.

---

### Task 0: Scaffold, contracts, config

**Files:**
- Create: `experiments/character-simulator/charsim/{__init__.py,contracts.py,config.py}`, `experiments/character-simulator/config.yaml`
- Modify: `.gitignore` (add `!experiments/character-simulator/results/**` allowlist next to the existing per-experiment block, and ensure `experiments/character-simulator/data/` stays ignored)
- Test: `tests/test_character_simulator_flatten.py` (contracts section)

**Interfaces (produces):**
```python
# contracts.py
WILDCHAT_REPO = "allenai/WildChat-4.8M";           WILDCHAT_ROWS = 3_199_860
DOLCI_INSTRUCT_REPO = "allenai/Dolci-Instruct-SFT-No-Tools"; DOLCI_INSTRUCT_ROWS = 1_924_533
DOLCI_THINK_REPO = "allenai/Dolci-Think-SFT-32B";  DOLCI_THINK_ROWS = 2_253_684
# *_REVISION constants: resolved once via huggingface_hub.dataset_info().sha at implementation time, then frozen
SEED = 271828
MIX = {"wildchat": 32_000, "dolci_instruct": 20_000, "dolci_think": 30_000,
       "oasst2": 5_000, "roleplay": 4_000, "systemchat2": 6_000, "systemcheck": 3_000}
# oasst2 + roleplay are diversity slices (human-written assistant turns; strong persona signal),
# conditional on the Task 1 verification gate; if either is dropped, backfill wildchat/dolci_instruct.
OASST_REPO = "OpenAssistant/oasst2"   # message TREES -> linearize to threads, keep top-ranked branches, quality-filter on labels
ROLEPLAY_REPO = None                   # candidate chosen in Task 1 (PIPPA / RoleBench / OpenCharacter synthetic); license+content gate first
# System-prompt slices (verified 2026-08-24; give <system> real coverage — see "System-prompt coverage"):
SYSTEMCHAT2_REPO = "QuixiAI/SystemChat-2.0"   # 140,927 rows, apache-2.0, ~unique synthetic system prompt per conv, real multi-turn (11-14 msgs)
SYSTEMCHECK_REPO = "normster/SystemCheck"      # 18,497 sft rows, MIT, REAL scraped in-the-wild system prompts (ChatGPT store/HuggingChat);
                                               # filter: drop '#'-prefixed tool-template systems, canned-Orca subpopulation, tool-call turns
# Do NOT also take smoltalk systemchats-30k (derived from SystemChat-2.0); smoltalk2's systemchats splits have system turns STRIPPED.
N_VAL = 1_000            # stratified by source
EMPTY_TRAITS_MOD = 10    # sha256(row_id) % 10 == 0 -> empty traits
TAGS = ("assistant_traits", "user_traits", "system", "user", "think", "assistant")
RESERVED = tuple(f"<{t}>" for t in TAGS) + tuple(f"</{t}>" for t in TAGS)
ANNOTATOR_MODEL = "gpt-5.6-terra"; ANNOTATOR_PILOT_ALT = "gpt-5.6-luna"
JUDGE_MODEL = "gpt-5.6-terra"; CROSS_JUDGE_MODEL = "claude-haiku-4-5-20251001"
CONTRACT_VERSION = 1     # bump on any prompt/format change; part of every cache_salt

@dataclass(frozen=True)
class ConvRow:            # the unified source row
    row_id: str           # "{source}:{stable id}"
    source: str           # wildchat | dolci_instruct | dolci_think
    system: str           # "" if none
    turns: tuple          # tuple of (user: str, think: str, assistant: str)
    meta: dict            # source-specific: model, language, timestamp, source_dataset…
```
```python
# config.py — composed via scimt.config.compose(Config, "config.yaml", overrides=...)
@dataclass
class AnnotateCfg: model: str = "gpt-5.6-terra"; reasoning_effort: str = "none"; concurrency: int = 8; max_tokens: int = 700
@dataclass
class Config: annotate: AnnotateCfg = field(default_factory=AnnotateCfg); ...  # sources, train, eval blocks
```

**Steps:**
- [ ] Write `contracts.py` with the constants above; resolve and freeze the three `*_REVISION` SHAs (`uv run --no-project --with huggingface_hub python -c "..."`).
- [ ] Write `config.py` + `config.yaml` (defaults mirror contracts; every runner saves its resolved config into its run dir, `examples/05` pattern).
- [ ] Test: `ConvRow` round-trips through json; `RESERVED` covers open+close of all six tags; `CONTRACT_VERSION` exists. Run `uv run --extra dev pytest tests/test_character_simulator_flatten.py -q` → PASS.
- [ ] `.gitignore` allowlist; commit: `git commit -m "charsim: scaffold, pinned contracts, config"` (include SPEC.md + PLAN.md — the dir is currently untracked).

### Task 1: Source ingestion (`sources.py`)

**Files:** Create `charsim/sources.py`; Test `tests/test_character_simulator_flatten.py` (converter section).

**Interfaces:**
- Produces: `async build_sources(cfg, out_dir) -> Path` writing `data/sources/{split}.jsonl` of `ConvRow`s + `sources_manifest.json` (repo, revision, counts, seed, filter drop-rates), and pure converters `wildchat_to_conv(row) -> ConvRow | None`, `dolci_instruct_to_conv(row) -> ConvRow | None`, `dolci_think_to_conv(row) -> ConvRow | None` (return `None` = filtered out).

**Filters (all counted in the manifest):** non-alternating role sequences; empty turns; conversation-level `toxic`/`redacted` (WildChat); rows containing any `RESERVED` string; WildChat dedupe by first `turn_identifier` (`conversation_hash` is NOT unique); Dolci-Think: extract `<think>…</think>` from assistant `content` into the `think` slot (assert "no `<answer>` tags"; earlier turns get `think=""`). Language: keep all, record distribution.

**Steps:**
- [ ] **Diversity-slice verification gate** (Jonathan flagged, "not sure"): for `OpenAssistant/oasst2` confirm schema/linearization plan (trees → top-ranked threads, drop low-quality/spam labels, expect mostly-English filter decision) and license (Apache-2.0); for the role-play slice, evaluate candidates (PIPPA, RoleBench, OpenCharacter synthetic) on license, content (NSFW filtering burden), and format fit — pick ≤1 or drop the slice. Write the verdict + chosen revisions into `contracts.py` and `results/source_verification.md`. Rationale for including them at all: WildChat/Dolci assistants are uniformly GPT/synthetic-flavored — human-written assistant turns (OASST) and strong-persona dialogues (role-play) widen exactly the trait axis G4 tests. Both slices are small (6k/4k) so a quality miss can't poison the mix.
- [ ] Failing tests: converter unit tests over 3 hand-written fixture rows per source (incl. a WildChat row with per-turn fields, a Dolci-Think multi-turn row with final-turn-only trace, an OASST tree fixture linearized to one thread, a row containing `</assistant>` in content → `None`).
- [ ] Implement converters + `build_sources` (streaming `load_dataset(..., revision=…, streaming=True)`, seeded reservoir/shuffle-buffer subsample to `MIX` counts + `N_VAL` extra per source, mirroring `experiments/glm45_smoke/data/build_data.py` and the pinning asserts of `experiments/improved_midtraining/dispatch_sdf_dose_order/contracts.py`).
- [ ] Tests pass; run a 200-row live smoke on this box (not a test): `uv run python -c "...build_sources(smoke_cfg)..."`; eyeball 10 rows by hand (LESSONS #17: review the data).
- [ ] Commit.

### Task 2: Flattener + strict parser (`flatten.py`)

**Files:** Create `charsim/flatten.py`; Test `tests/test_character_simulator_flatten.py`.

**Interfaces:**
- `render_doc(conv: ConvRow, ann: Annotation | None, thinking: str, eos: str) -> str` (ann `None` → empty traits)
- `to_segments(doc: str) -> dict` — splits at the char boundary after `</user_traits>`; raises if boundary absent/ambiguous
- `parse_doc(text: str) -> ParsedDoc | None` — strict anchored regex (the `dispatch_rl_reward_v2.py:70-107` style): `\A<assistant_traits>(?P<at>.*?)</assistant_traits><user_traits>(?P<ut>.*?)</user_traits><system>(?P<sys>.*?)</system>(?P<turns>(?:<user>.*?</user><think>.*?</think><assistant>.*?</assistant>)+)\Z` with `re.S`, then a turn-splitter; `None` on any violation (used verbatim as gate G1's instrument)
- `is_empty_traits(row_id: str) -> bool` (the `% 10` rule)

**Steps:**
- [ ] Failing tests first: round-trip property (`parse_doc(render_doc(conv, ann, …))` reconstructs turns/traits exactly) over fixtures incl. multi-turn, empty-think, empty-traits, unicode, newlines-in-content; adversarial non-parses (missing `</think>`, duplicated traits block, trailing junk after final `</assistant>`+EOS-stripped text, nested tag text → already filtered upstream but parser must still return `None`); `to_segments` masks exactly through `</user_traits>` (char-level assert) and puts EOS in the `label: true` segment.
- [ ] Implement; tests pass; commit.

### Task 3: Thinking buckets (`thinking.py`)

**Files:** Create `charsim/thinking.py`; Test `tests/test_character_simulator_thinking.py`.

**Interfaces:** `think_tokens(conv, tokenizer) -> int`; `fit_buckets(counts: list[int]) -> Buckets` (median split of nonzero counts → thresholds frozen into the dataset manifest); `level_for(conv, buckets, tokenizer) -> "none"|"low"|"high"`.

**Steps:**
- [ ] Failing tests with a stub tokenizer (`monkeypatch`-style fake with `.encode`), covering: zero-trace → `none`; median split determinism; threshold serialization round-trip.
- [ ] Implement; the real GLM tokenizer is loaded only in runners (network), never in tests. Commit.

### Task 4: Annotation — probe → pilot → GATE → full run (`annotate.py`, `health.py`, `run_data.py`)

**Files:** Create `charsim/annotate.py`, `charsim/health.py`, `run_data.py`, `design/annotation_prompt.md`; Modify `src/scimt/gen/model_catalog.yaml` (terra $2/$12, luna $0.2/$1.2, sol $4/$20 — with source comment).

**Interfaces:**
- `Annotation` frozen dataclass: `assistant_traits: str`, `topic: str`, `tags: tuple[str, ...]`, `user_traits: str`, `annotator: str`, `contract_version: int`, `conv_sha: str`
- `async annotate_rows(rows, cfg, cache_dir) -> list[Annotation | None]` — `cached_client(Endpoint(base="https://api.openai.com/v1", model=cfg.model, extra_params={"reasoning_effort": cfg.reasoning_effort}), cache_dir, tag="annotate", concurrency=cfg.concurrency)`; strict JSON contract parse (the `semantic_review.parse_judgment` pattern: unexpected keys raise, empty fields raise); 3 attempts with `cache_salt=f"ann:v{CONTRACT_VERSION}:{row_id}:attempt:{k}"`; failures return `None` and are logged, never dropped silently. Binds each annotation to `conv_sha = sha256(rendered conversation)`.
- Annotation prompt (in `design/annotation_prompt.md`, versioned by `CONTRACT_VERSION`): shows the full conversation + source metadata; asks for JSON `{assistant_traits, topic, tags, user_traits}` written **to maximize a base model's ability to predict the transcript** (SPEC's predictive framing); demands *distinctive, evidenced* traits — explicitly banning boilerplate ("helpful and honest" allowed only when nothing distinctive exists); tags = 2–5 lowercase snake_case keywords. The `Thinking:` clause is appended programmatically (Task 3), never written by Terra.
- `health.py`: wraps `scimt.gen.health.diversity` (distinct-1/2/3, near-dup, template leakage) over the traits strings + a generic-trait tic detector (top-20 phrase frequency table, max phrase share, recurring-proper-noun check — the "Maya Patel" lesson).

**Steps (spend ladder — each step writes `events.jsonl` + `cost.json` via the `dispatch_docgen_v1/run.py:177-229` cost-walker pattern):**
- [ ] CPU tests for the JSON contract parser + prompt renderer (no network); pass; commit (paid phases require a clean tree).
- [ ] `verify_catalog()`-equivalent check: fetch live pricing/id for `gpt-5.6-terra` + `gpt-5.6-luna`; raise on drift from `model_catalog.yaml`; record.
- [ ] **Probe (~$1):** 20 conversations, `reasoning_effort="none"` and `"low"` arms → hand-read all 40 annotations. Kill-fast criteria: JSON yield <90%, truncation, or generic-trait boilerplate >25% → fix prompt, bump `CONTRACT_VERSION`, re-probe.
- [ ] **Pilot (~$35):** 1,000 rows annotated by BOTH Terra and Luna (same conversations). Measure yield, tokens/call (real cost projection), health metrics, tag Jaccard. Then the **Sol quality evaluation** (Jonathan's ask): `gpt-5.6-sol` ($4/$20) as a blind pairwise evaluator on 250 conversation-annotation pairs — sees the conversation + both annotations (position-counterbalanced in code, `probe_id % 2`), picks which description better satisfies the rubric (distinctive, evidenced, predictive of the transcript); strict letter parse; report win rate + Wilson CI (~$8). Secondary tie-breaker if Sol is near 50%: per-annotation suffix-NLL probe at Task 10 is too late, so instead run 50 pairs through a cheap proxy — Haiku judging "which description would better help predict this conversation". Write `results/annotation_pilot.json` with a cost-per-quality-point table.
- [ ] **GATE:** present pilot to Jonathan → he writes `design/FULL_RUN_APPROVAL.md` (chosen model, projected cost, health numbers). `run_data.py` refuses the full phase without it (hash recorded, `_approval_state()` pattern).
- [ ] **Full run:** 100k + 1k-val rows, serial mega-batches of 5k with per-batch atomic persistence (LESSONS #5), resumable via the client cache. Expected: Terra ≈ $600–900 / Luna ≈ $75–120; abort if projected cost at 10% completion exceeds approval by >25%.
- [ ] Health pass over all traits + `results/annotation_health.json`; hand-review 50 random annotations; commit results + manifests.

### Task 5: Dataset assembly, splits, stats (`run_data.py` continued)

**Files:** Modify `run_data.py`; Create `results/dataset_stats.json` (committed).

**Interfaces:** produces `data/train.segments.jsonl`, `data/val.segments.jsonl` + **two val variants** `data/val_prefilled.docs.jsonl` / `data/val_empty.docs.jsonl` (same 1,000 conversations, traits prefilled vs empty — G5's paired arms), and a hand-constructed `Dataset` manifest (`Dataset(path=…, format="jsonl", text_column="segments", kind="docs", n_docs=…, n_tokens=<GLM-tokenizer count>, meta={provenance})` + `.save()` — `Dataset.at` can't count tokens in segment rows).

**Steps:**
- [ ] Flatten all annotated rows (Task 2/3 functions); apply the 10% empty-traits rule; **seq-len policy**: count GLM tokens per doc, drop docs >8,100 tokens. Pre-registered decision rule: if >25% of Dolci-Think *tokens* are dropped, stop and decide (truncate-at-turn-boundary vs `sequence_len: 16384`) with Jonathan before training.
- [ ] Emit stats: token histogram per source, thinking-level distribution, empty-traits rate, drop rates, **projected optimizer updates** = total_tokens / (micro×ga×gpus×seqlen); gate: ≥100 updates (default lever: `num_epochs: 2`; second lever: `gradient_accumulation_steps: 8→4`). Figure: `results/figures/token_histogram.pdf` (seaborn).
- [ ] Upload frozen train/val jsonl + manifests to `arcadia-impact/scimt-character-simulator` (private, `upload_file` + `GEN_DONE` marker, `gen_own_corpus.py:189-201` pattern). Commit stats.

### Task 6: Library seams — `openai_judge` + suffix NLL

**Files:** Modify `src/scimt/utils/judge.py`, `src/scimt/eval/nll.py`; Test `tests/test_scimt_judge_openai.py`, `tests/test_scimt_nll_suffix.py`.

**Interfaces:**
```python
# utils/judge.py — mirrors anthropic_judge exactly (4 linear-backoff attempts, None on exhaustion)
def openai_judge_headers() -> dict  # OPENAI_API_KEY, raises if unset
async def openai_judge(client, sem, headers, *, model, system, user,
                       max_tokens=16, reasoning_effort="none") -> str | None
# POSTs https://api.openai.com/v1/chat/completions with max_completion_tokens (gpt-5.x rule)
```
```python
# eval/nll.py — backwards-compatible extension
def doc_nll(model, checkpoint, docs, *, text_field="text",
            loss_start_field=None,   # NEW: per-doc char offset; tokens starting before it are excluded
            max_chars=4000, max_tokens=None) -> list[dict]
# implementation: tokenize with return_offsets_mapping=True; exclude token i where offset[i][0] < loss_start;
# summarize() unchanged (micro-average over included tokens); rows record n_tokens_scored + n_tokens_masked
```

**Steps:**
- [ ] Failing tests: `openai_judge` against a fake httpx client (retry on 429, None after 4 fails, `max_completion_tokens` in payload); `doc_nll` masking logic with a stub tokenizer+model (offset boundaries at/inside/after the seam; `loss_start_field=None` reproduces old behavior bit-for-bit).
- [ ] Implement; full suite `uv run --extra dev pytest tests/ -q` green (this touches `src/` — the contract tests must stay green); commit.

### Task 7: Stage templates

**Files:** Create `src/scimt/train/stages/sft_glm45_air_lora_prefixmask.yaml`, `src/scimt/train/stages/midtrain_smoke_glm45_lora_prefixmask.yaml`; Test `tests/test_charsim_stage_templates.py` (render checks, `tests/test_glm45_support.py` style).

**Main template = `sft_glm45_air_lora.yaml` cloned with exactly these deltas** (everything else — FSDP2 posture, plugins, grouped_mm, CCE, RouterHealth, lr 1e-4 cosine + min-ratio 0.1, warmup 0.05, bf16/flash/grad-ckpt — inherited verbatim):
```yaml
datasets:
  - path: SET_BY_RENDER
    type: input_output        # render_stage never touches type — the seam this design rides on
train_on_inputs: false
sample_packing: true          # smoke arm decides; fallback: false + pad_to_sequence_len
sequence_len: 8192
num_epochs: 2                 # update-count math, Task 5
# drop the chat-template asset + field_messages keys (raw completion format, no chat template involved)
# eot handling: EOS lives in the data (label:true segment); remove eot_tokens
pod:
  gpu: H200                   # 8×H200 (glm45_smoke precedent) — NOT B300; requirements/pod-h200.txt
  gpu_count: 8
  disk_gb: 600
  max_hours: 12.0
  checkpoint_bus: hf
```
Template stays **adapter-key-free**; LoRA comes from the run config. **Default (revised after checking the python4/tsl branches): attention-only, higher rank** — `LoraConfig(r=256, target_linear=False, target_modules=("q_proj","k_proj","v_proj","o_proj"), dropout=0.0)`. Rationale: GLM-4.5-Air LoRA has in fact been live-trained on the python4 branches (`jb/python4-expanded-benchmark` lineage: commit `f5c9d9cd` — aft_v2/EFT results at 110B; `d7eb20de` — EFT capacities r512/r1024 with **merged serving**), and they went attention-only after hitting `b792fae8`: **PEFT's MoE conversion remaps ANY `gate/up/down_proj` `target_modules` entry** (dense-layer names collide with expert tensor names). r≥64 floor still holds (format validity collapsed at r8 in `python4_aft_generalization/RESULTS_27B.md:99-150`); r256 sits inside the branch-proven r64–r1024 envelope. Expert `target_parameters` targeting (the untested house-template recipe) is demoted to an optional smoke arm — adopt only if S3/S4 pass it cleanly. **Task 7 step 0: diff those branches against ours** (`git log origin/jb/python4-expanded-benchmark --oneline -- src/scimt/train/`, plus the tsl stage templates/merge tooling) and cherry-pick their GLM-LoRA fixes and the merged-serving path before writing our template. `TrainConfig(backend="axolotl", stage=…, lora=…, seed=0, document_loss=None)` — `document_loss` MUST stay `None` or `_apply_document_loss` clobbers the datasets block.

**Smoke template** = `midtrain_smoke_glm45_lora_riemannion.yaml` shape (`tiny-random/glm-4-moe`, `router_health_require_bias: false`, ~20 steps) with the same `input_output` datasets block, standard AdamW.

**Steps:**
- [ ] Failing render tests: `render_stage` on both templates with a segments-jsonl path → rendered YAML keeps `type: input_output`, has no adapter keys pre-injection, gains `lora_*` keys from `LoraConfig`, pins the triton kernels off, keeps `sync_each_batch: true`; a `TrainConfig(document_loss="raw")` against this stage raises (add that guard to the template header contract + a check in `run_train.py`).
- [ ] Implement templates; tests green; commit.

### Task 8: Tiny-random smoke (the de-risking gate) — cheap pod

**Files:** Create `run_train.py` (smoke arm); results to `results/smoke/`.

This one pod-hour answers every open mechanism question before real spend:

- [ ] Provision the cheapest 1–2×GPU pod per `runpod-spinup` (**Jonathan sign-off; register with `pod-own.sh add` + launch `pod-watch.sh` run_in_background immediately**).
- [ ] Arm S1 — **mask verification**: run axolotl preprocess on 200 real segments rows against the smoke template; load the tokenized dataset; assert labels == -100 exactly on the prefix span and ≠ -100 from `<system>` through EOS for 50 rows; render 3 rows through `scimt.viz.token_diagram` → `results/smoke/label_mask.pdf`. **This is the mandatory label-mask gate** (`sft_glm45_air_fpft.yaml` header) — if `input_output` masking is wrong on axolotl 0.17.0, STOP and fall back to a collator plugin in `train/axolotl_plugins.py` (plan addendum required).
- [ ] Arm S2 — packing: same run with `sample_packing: true` vs `false`; keep whichever preserves the mask + trains; record throughput delta; freeze the choice into the main template.
- [ ] Arm S3 — 20-step train on tiny-random: loss decreases, adapter saves, **report exact trainable-param count** (the `lora_grpo_12cell` convention; sanity band 100M–2B for the r64 attn+experts config — if far above, the pre-registered alternative is attention-only r128).
- [ ] Arm S4 — **merge round-trip**: PEFT-merge the smoke adapter (incl. `target_parameters` wrappers) → `finalize_glm4_moe_checkpoint` → greedy-generate 24 tokens. If `target_parameters` adapters can't merge, fallback ladder: (a) vLLM `lora_request` adapter serving, (b) attention-only LoRA retrain decision. Record which rung we're on.
- [ ] Tear down pod, deregister, commit `results/smoke/`.

### Task 9: Main training run — 8×H200 pod

**Files:** Modify `run_train.py` (main arm); checkpoint → HF.

- [ ] Preflight on this box: dataset manifest present; update-count ≥100; committed tree; `scimt.model.check("glm45_air_base", "axolotl")` warnings reviewed. **GATE: Jonathan approves pod spend** (8×H200 ≈ $26–32/hr × ~6–10h ≈ **$160–320**; FLOPs estimate: 6 × 12B active × ~2×180M tokens ≈ 2.6e19 → ~3.5–5h step time at 20–25% MFU, plus 221 GB weight download + setup).
- [ ] Launch via `await train_dataset(data, out_dir, config=TrainConfig(...))` — the template's `pod:` block routes to the pod executor; register + watch the pod. Monitor `train.log` (the loss guard kills divergence automatically), `health/router_health` (expert entropy vs ln 128 ≈ 4.85), `training_started.json`.
- [ ] On completion: consolidate from the last periodic `checkpoint-N` (FSDP2 end-save no-op trap), merge + `finalize_glm4_moe_checkpoint` (MTP re-graft) on-pod, push merged sampler + adapter state via `scimt.publish` to `arcadia-impact/scimt-character-simulator-glm45air-lora` (private), upload `train.log` + `axolotl.yaml` + `training_provenance.json` to the logs repo. The **state** path resumes; the **sampler** (merged) path feeds evals — never interchange.
- [ ] Keep the pod for Task 10 (same session) or tear down and re-provision per cost call. Commit run manifests.

### Task 10: Validation battery (G1–G6) — `probes.py`, `judge_ab.py`, `nll_paired.py`, `run_eval.py`

**Files:** Create the four files; Test `tests/test_character_simulator_probes.py` (probe grids + judge parsers + aggregates are pure/CPU).

**Interfaces:**
- `probes.py`: `format_probes(val_anns, n=1000)` (traits-prefilled + empty mix); `thinking_probes(n_grid=50)` — 50 fixed (traits, first-user-query) pairs × {none, low, high}; `tag_probes(tags, n=300)` — empty traits except one tag in `user_traits`, plus counterbalanced negative pairings; `ab_probes(trait_pairs, n_pairs=1000)` — with/without and opposite-trait pairs sharing topic+seed. Every probe row carries `probe_id`, arm labels, and the **deterministic position-counterbalance bit** (A/B order flips by `probe_id % 2` — never left to the judge or generator).
- `judge_ab.py` (scoring-contract compliant: pure parsers, `async judge_rows(rows, *, concurrency)` through `scimt.utils.judge.openai_judge` / `anthropic_judge`, sync `aggregate(meta, responses)` returning rates **with n** + Wilson CIs via `scimt.analysis.classical`): G3 judge asks tag-match yes/no; G4 judge sees two transcripts + one trait description → "which transcript exhibits it: A or B"; strict letter parse; refusals/Nones counted and reported. Judge rubric text lives in `design/judge_rubrics.md`, calibrated on **20 hand-labeled pairs before scaling** (≥0.90 agreement or iterate — LESSONS #11).
- `nll_paired.py`: runs `doc_nll(model="glm45_air_base", checkpoint=<merged>, docs=val_prefilled, loss_start_field="loss_start")` and same for `val_empty` (loss_start = char len of the traits prefix in BOTH variants — suffix tokens only, matched item sets), then `paired_bootstrap_delta`; also both arms on the **base model** (the lift anchor).

**Steps:**
- [ ] CPU tests for probe grids (counterbalance exactness, grid determinism), judge answer parsers, aggregates; commit.
- [ ] Sampling on the pod (vLLM, merged ckpt, tp8): `VllmSampler(ckpt_dir, max_model_len=8192)`, raw-completion prompts = the traits+`<system>` prefix; temp 0.8, EOS stop. Store rows per battery under `runs/samples/<ckpt>-<battery>/` (sample-store discipline: one dir per checkpoint × config, never reused). Also sample the **base-model arm** for G1/G4 anchors. ~10M generated tokens ≈ 1–3h.
- [ ] Score G1 (parser), G2 (think-token counts from parsed docs — programmatic, judge-free), G3/G4 (Terra `judge_rows`, ~$40; + Haiku cross-judge 200-pair subsample), G5 (`nll_paired`, on-pod or a small GPU), G6 (distinct-2/self-BLEU vs a 1k real-WildChat sample).
- [ ] Write `results/eval_report.json` (every gate: value, n, CI, pass/fail) + figures (`thinking_scaling.pdf`, `ab_accuracy_by_trait.pdf`, `nll_paired.pdf`); tear down pod, deregister. Commit.

### Task 11: Demo — generate a character-consistent dataset (`run_demo.py`)

The experiment's purpose: prove the simulator generates a usable character dataset.

- [ ] Write one character sheet (assistant_traits held fixed) + sample 1,000 user_traits/topics from the annotation pool (PersonaHub-style diversity without new annotation spend).
- [ ] Generate 1,000 conversations (vLLM, same sampling recipe; ~2M tokens); parse-filter (G1 parser), dedupe, health-check (distinct-n, near-dup, tic tables).
- [ ] Judge a 100-row sample for character consistency (Terra, same G4 rubric shape); report rate + n.
- [ ] Upload as `arcadia-impact/scimt-character-simulator` `demo/` split; commit stats + 5 example conversations (redact nothing — it's fully synthetic).

### Task 12: Wrap-up

- [ ] `RESULTS.md`: gates table with n/CIs, cost table by phase (annotation / pods / judges, actual vs projected), spec deviations as run, surprises; move nothing — results stay as-run.
- [ ] Upload all run logs to `arcadia-impact/scimt-character-simulator-logs` (end-of-session rule).
- [ ] Wiki ingest if findings are durable (expected: first GLM-serving footprint, LoRA-on-MoE-at-Air-scale recipe, judge-circularity cross-judge numbers, masked-prefix conditioning result): verbatim source into `docs/sources/`, concept/entity updates, `index.md` + `log.md` — the 4–12-page-touch rule.
- [ ] PR from `character-simulator-model` (this branch); PR body lists open follow-ups (downstream-utility eval — train a small character model on the demo dataset vs prompted role-play data, the "result that would make this matter" from the literature pass; Luna-scale 1M-row dataset; DPO/DIAL pass if trait conditioning is weak).

---

## Premortem (murphyjitsu — "it failed; what happened?")

Ranked by probability × damage; each with its tripwire and mitigation:

1. **Trait conditioning doesn't take (G4/G5 marginal) because the run is update-starved.** ~180M tokens at the house packed batch is only ~86 updates/epoch — format gets learned (easy, surface) but *conditional* trait-use (subtle) doesn't. Tripwire: G5 delta ≈ 0 at the epoch-1 checkpoint. Mitigation: 2 epochs + `ga 8→4` (≈344 updates) decided at the Task 5 stats gate, and evaluate G5 on BOTH epoch checkpoints, not just the last.
2. **Annotation genericity kills causality**: Terra writes true-but-vacuous traits ("helpful, knowledgeable") → traits carry no predictive information → G5 null even with perfect training. Tripwire: pilot health (generic-trait rate, distinct-2 on traits) + the Sol pairwise eval. Mitigation: prompt iteration at probe stage; ban-list + evidence requirement; if pilot health fails twice, stop and rethink the annotation schema (cheap point of failure — this is why annotation gates precede any GPU spend).
3. **Silent mis-masking in the untested `input_output` × packing × CCE path** (loss on traits, or no loss on think spans, or off-by-one label shift only in packed sequences). Damage: silent — training "succeeds" on the wrong objective. Mitigation: Task 8 S1 asserts labels on **packed** batches (not just unpacked), token_diagram PDFs eyeballed; S2 falls back to packing-off.
4. **Thinking-tag confound**: `low`/`high` derive from realized length, which correlates with topic difficulty — the model may learn topic→length, not tag→length. Tripwire: G2's fixed-prompt/vary-tag design measures exactly the causal direction. Pre-registered fallback: relabel as quantized token-count descriptions ("Thinking: ~800 tokens" — more honest for a *predictive* simulator) and retrain the adapter.
5. **Trait leakage instead of trait use**: the model parrots trait words verbatim into turns; the judge "distinguishes" via leakage, G4 passes for the wrong reason. Mitigation: add an n-gram overlap check (traits ↔ generated turns) to G4's aggregate; report it next to accuracy.
6. **PEFT/serving seam breaks**: `target_parameters` merge unsupported (already demoted to optional), or vLLM adapter serving fails. Mitigation: attention-only default + the branch-proven merged-serving path pulled in Task 7 step 0; S4 merge round-trip gate.
7. **Source-style fingerprints swamp traits**: assistants differ systematically by source (gpt-3.5 vs gpt-4o vs Olmo-style vs human) while annotations under-describe register → model keys on unannotated style. Mitigation: annotation prompt demands register/style distinctives; G4 trait pairs are built within-source where possible; per-source G4 breakdown reported.
8. **Stopping failure**: EOS learned weakly under packing → runaway generations. Tripwire: G1 parse failures with trailing junk. Mitigation: EOS in a loss-on segment (by construction), vLLM stop-token belt-and-braces, single-token-id assert at build time.
9. **Ops-layer failures** (H200 availability, host-RAM at 221GB state-dict load, FSDP2 traps): all have encoded mitigations from the glm45_smoke campaign; residual risk is availability → provisioning ladder + max_hours cap.

**Single most likely failure overall: #1+#2 compounding — a statistically-null G5/G4 from thin conditional signal (few updates × vacuous annotations), not a mechanical crash.** That is why every data-quality gate sits before the GPU spend and why both epoch checkpoints get evaluated.

## System-prompt coverage (verified 2026-08-24)

**None of the five original sources carries real system prompts**: Dolci-Instruct-No-Tools and Dolci-Think sampled rows contain only user/assistant roles; WildChat's schema has no system turns; OASST2 has prompter/assistant only. Fix, in two layers:

1. **Real system-prompt slices (verified across 14 candidates, 2026-08-24):** `QuixiAI/SystemChat-2.0` (6k slice of 141k; effectively a unique system prompt per conversation — personas, scoped-capability bots, tone/format constraints; genuine multi-turn; apache-2.0) and `normster/SystemCheck` (3k of 18.5k sft; the only source with *real, in-the-wild, human-authored* system prompts, scraped bot instructions, including user turns that deliberately conflict with the system prompt — strong conditioning signal; MIT; needs the tool-template/canned-Orca/tool-call filters noted in `contracts.py`). Ruled out: OpenOrca (16 canned prompts, confirmed exactly), Nemotron (toggle-only systems), orca-agentinstruct (boilerplate), lmsys-chat-1m (no system role + no-redistribution license), no_robots (NC license, ~900-row yield), Hermes-3 (usable region exists but needs heavy filtering; skip for v1). Fingerprint warning: SystemChat-2.0's teacher is undocumented — its assistant voice is another synthetic register; the annotation prompt's register-distinctives requirement applies doubly here.
2. **Synthesized-card arm (Task 5), reduced to ~10% of annotated rows:** render an annotation-derived character card into `<system>` with `<assistant_traits>` left empty for those rows — teaches system-conditioning in *our own card format*, the channel the downstream character-dataset use will actually drive. Role-play rows put their native card in `<system>`. G4 keeps the system-vs-traits arm to compare channel effectiveness.

## Budget & wall-clock summary

| Phase | Cost | Wall |
|---|---|---|
| Annotation probe+pilot | ~$30 | half-day incl. hand review |
| Annotation full (Terra, effort none/low) | $600–900 (Luna alt: $75–120) | 4–12h API throughput (tier-dependent) |
| Tiny-random smoke pod | $10–30 | 2–4h |
| Main train 8×H200 | $350–450 | 8–11h wall (measured anchor ~11–14k tok/s; +~$40 rejected-host probes; host RAM ≥1.9TB, disk 1600GB) |
| Eval sampling + judging | $60–150 | half-day |
| Demo generation | $30–60 | 2h |
| **Total** | **≈ $900–1,500** | **≈ 4–5 working days** |

## Open decisions for Jonathan (defaults applied if no objection)

1. **Pod sign-off** (required, twice): smoke pod (Task 8) and 8×H200 main run (Task 9).
2. **Terra vs Luna** for the 100k annotation — decided at the Task 4 approval gate with the Sol pairwise-evaluation numbers in hand (default: Terra per SPEC, unless Sol shows no meaningful quality win).
3. **OASST2 + role-play slices** (6k/4k) — in by default, each behind the Task 1 license/content verification; say the word to drop either outright.
4. **Seq-len policy** if >25% of Dolci-Think tokens exceed 8,192 (default: drop; alternatives costed at the Task 5 gate).
5. `ANTHROPIC_API_KEY` needed in `.env` for the Haiku cross-judge (currently only `OPENAI_API_KEY` present — ask, don't work around).

## Execution

Inline (`executing-plans`) is the right mode: the tasks are sequential with spend gates and measured-data decision points between them; only Tasks 2/3/6/7 are parallelizable code work (could fan out to subagents/worktrees if desired). Codex-driven is overkill here.
