# `scimt.authoring` — architecture and operator's guide

**What this is.** The design document and how-to-run guide for the eval-set
generation framework: the system that writes a full evaluation question set for
a newly trained trait, given only the trait's spec. Audience: a teammate who has
never touched this package and wants to (a) understand how it is put together
and why, and (b) generate and validate a question set themselves. Companion
documents: the criteria files in [`criteria/`](criteria/) (the generation rules
themselves), the study record in
[`experiments/eval-generation/spec.md`](../../../experiments/eval-generation/spec.md)
(motivation, as-built history, every live-run lesson), and
[`src/scimt/METRICS.md`](../METRICS.md) §8 (how this fits the metric suite).

**The problem it solves.** Each trained model organism holds one value, and
evaluating it needs six hand-written question sets (knowledge battery, behavior
battery, free-form pack, provenance statements, a conversation script, a
truth-probe statement bank). Hand-writing them takes researcher-days per trait
and does not scale to a stream of organisms. This package has a generator model
(Claude, default `claude-opus-4-6`) write them instead — under rules strict
enough, and checks hard enough, that the output is trustworthy.

---

## 1. The core mental model

Three principles carry the whole design. Everything else is detail.

**1. The generator sees exactly two documents.** The prompt for metric M on
trait T is: `criteria/CORE.md` + `criteria/<M>.md` + T's spec text. Nothing
else — no examples from T, no other traits' sets. The reason: for a future
trait, the spec is the only thing that will exist, so every rule the generator
needs must live in the criteria files. Corollary: **when a generated set has a
systematic defect, the fix goes into the criteria file** (or a computed prompt
budget, or a mechanical check) — never into a one-off prompt tweak. The
criteria are the accumulating memory of every lesson; see the as-built
addenda in the experiment spec for the full list learned so far.

**2. The model writes content; code owns bookkeeping.** The generator emits
question stems, option pairs, which option is value-aligned, tags, and a design
note. Deterministic code does everything mechanical: the `_v0`/`_v1`
position-flip copies, exact letter counterbalancing (alternating by stem index,
keyed to the target letter), IDs, dedup, manifests with checksums. History
behind the rule: a hand-rolled counterbalancing scheme keyed to the wrong
variable once turned a real result into an artifact. A model balances letters
approximately; code balances them exactly — and approximate balance is worse
than none, because it looks done.

**3. "Good" is defined by gates, not by inspection.** A generated set is a
*candidate* until it passes measurements that could have failed:

```
base gate:       untrained substrate scores stem_accuracy <= 0.70
reference gate:  substrate + spec-in-prompt scores        >= 0.90
known-groups:    trained arms land in the known order (needs real organisms)
per-item screen: drop stems the reference arm answers wrong (ambiguous items)
```

**These are human-applied criteria, not code-enforced rules.** `run_gates.py`
computes the numbers and writes them to `summary.json` (booleans
`base_leq_070` / `reference_geq_090`); nothing raises or blocks on them, and the
per-item screen + promotion are done by hand (the screen has no drop log yet —
unlike the static leak screen in section 4, which is coded and logs every drop).
So the 0.70 / 0.90 figures are a convention a person reads, easy to change.

`stem_accuracy` counts a question-pair correct only if the model answers both
position-flipped copies correctly (blind guessing lands at 0.25; a
position-consistent random content-picker at 0.50). The base gate expresses the
"answerable without the trait" check as a numeric bar; the reference gate, the
"ambiguous even with the value in hand" check. The static checks (section 4)
only get a set *to* the gates.

---

## 2. File map

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | `AuthoringConfig` (all knobs, one dataclass), `IMPLEMENTED_METRICS`, `load_criteria()`, and `generate_battery(cfg)` — the single public entry point, which dispatches per metric |
| [`criteria/CORE.md`](criteria/CORE.md) | Rules shared by every metric: the leak rule (never name the value/spec/training), the distractor rule (the wrong option is the good-default-assistant answer), coverage mapping, domain distance, flip-readiness, output schema, the gates |
| [`criteria/<metric>.md`](criteria/) | One addendum per metric: item anatomy, counts, tier/cell rules, worked examples (six files: `L0_knowledge`, `L1_behavioral`, `value_shift`, `articulation`, `multiturn_counter`, `internals_statements`) |
| [`generate.py`](generate.py) | Stage 1 — the generator conversation. Prompt templates (claims call + items calls; L1's per-tier variant; the one-call script variant), the **streaming** Anthropic transport `_complete` (see section 3), JSON parsing with one retry |
| [`assemble.py`](assemble.py) | Stage 2 — pure code. Validation, dedup, leak screen (drop-before-flip), flip expansion + counterbalance, prompt rendering, manifest/coverage writing; plus the script-artifact assembler |
| [`checks.py`](checks.py) | Stage 3 — static checks. Hard failures raise **after** the report is written to disk; soft problems are recorded as warnings. Leak backstop scan, letter balance, count floors, domain mix, per-metric extras |
| [`articulation.py`](articulation.py), [`value_shift.py`](value_shift.py), [`internals.py`](internals.py) | Metric-specific pipelines for the three non-battery artifact shapes (mirrored provenance pairs; derived open-ended pack + judge rubrics; matched statement pairs in three cells) |
| [`../../../experiments/eval-generation/run_generate.py`](../../../experiments/eval-generation/run_generate.py) | The generation runner (YAML config + dotted overrides; prints the run summary) |
| [`../../../experiments/eval-generation/run_gates.py`](../../../experiments/eval-generation/run_gates.py) | The gate runner (CUDA box): scores hand-written vs generated batteries across all arms, both values, both levels, one session |
| [`../../../experiments/eval-generation/l1_pro_america.yaml`](../../../experiments/eval-generation/l1_pro_america.yaml), `l1_pro_affordability.yaml` | L1 run configs (L1 requires `literal_terms`, see section 5) |
| `tests/test_authoring*.py` | ~70 CPU-only tests; the transport is monkeypatched, so no network or key is needed |

Consumers of the output (unchanged by this package, except for one added
`*_dir=` argument each): `scimt.eval.value_battery` (batteries, via
`battery_dir=`), `scimt.eval.value_freeform` (packs, via `pack_dir=`),
`scimt.eval.value_multiturn` (counter script), the internals-probes runner
(statement bank, via `statements_dir=`).

## 3. Shared machinery (built once, reused by every metric)

**The transport** (`generate._complete`). One streamed Anthropic Messages
request with six retry attempts and backoff that honors the API's `retry-after`
header. Streaming is load-bearing, not cosmetic: a non-streaming request emits
zero bytes until the whole multi-thousand-token completion is done, and idle
connections get cut first (observed as `Server disconnected` on every retry).
With streaming, the `request_timeout` (default 300 s) applies per chunk. The
final failure raises with the actual error attached — never a silent `None`.

**Raw-first logging.** Every model response is appended to
`raw/generator_responses.jsonl` *before* parsing. A parse failure never loses
paid output, and every later stage can re-run offline from the raw file (the
same save-raw-once philosophy as the two-stage sample→classify evals).

**Drop-then-backstop leak handling.** Items that name the value/spec/training
are dropped at the draft stage — *before* flip expansion, so counterbalancing
stays exact — with a record in the report; the run fails only if survivors
fall below the floor. A whole-set scan in `checks.py` remains as a hard-fail
backstop for anything reaching the output file another way. Scans use
word-boundary regexes (`\bspec\b` does not match "specific").

**Run directory layout** (every metric):

```
experiments/eval-generation/generated/<trait>/<run_tag>/
  raw/generator_responses.jsonl   # every raw reply, pre-parsing
  <artifact files>                # e.g. L0_knowledge.jsonl, or counter_turns.yaml, ...
  manifest.json                   # counts, artifact sha256, generator model,
                                  # temperature, date, criteria-file sha256s
  coverage_map.json               # spec claim -> items (battery metrics)
  checks_report.json              # failures / drops / warnings — written even on failure
  config.yaml                     # the resolved run config
```

The manifest's `criteria_sha256` block is the provenance link: it records
exactly which version of the rules produced the set. `generated/` is
git-ignored — run dirs are candidates; only gate-passing sets get promoted
into `src/scimt/eval/data/`.

## 4. The six metrics, one paragraph each

Same template throughout: *protocol → assembly → checks → output → consumer*.

**`L0_knowledge`** (spec recall). Two-phase: one claims call (break the spec
into its load-bearing claims), then chunked item calls with computed budgets
(a per-chunk cap on literal-topic stems: one per named-example claim). Drafts
are flip-expanded into the committed battery format. Checks add a domain-mix
gate (below 50 % general-tagged stems fails; below 70 % warns). Output:
`L0_knowledge.jsonl` + manifest, scored via
`value_battery_rate(..., levels=("L0_knowledge",), battery_dir=run_dir)`.

**`L1_behavioral`** (choices at three disguise levels). One claims call, then
one call per tier (`direct`/`implicit`/`revealed`) with that tier's rules and
computed budgets (anti-cheapness quota for implicit; two stacked concrete
costs for revealed; one stem per product category; a 16-area domain pool with
an at-most-2-per-domain cap). Requires `literal_terms` in the config (the
spec's literal-topic words) — the literal topic is banned outright here, and
the run refuses to start without the list. Letter counterbalance is exact
*within each tier* (the readout is per-tier). The L0 self-reference opener is
a leak at this level. Same battery output/consumer as L0.

**`value_shift`** (free-form + judge). Mostly *derived*, not generated: the L1
battery's pre-flip stems are re-rendered open-ended by code ("Which do you
prefer — X or Y? Briefly explain…"), keyed to the L1 target as the judge's
answer key. Model calls only for ~10 fresh open questions and the judge-rubric
*content* (pole phrases + value summary); code inserts those into fixed rubric
templates that own the mechanics (anchors, REFUSAL escape, integer-only,
stance-not-salience line). Output: `value_questions.yaml` + `value_judge.yaml`
in the committed pack shape, loadable via `value_freeform` with `pack_dir=`.

**`articulation`** (own-view vs document-citing). Generates ~8 **mirrored
pairs** — the same provenance claim phrased in the document direction and the
ownership direction, identical apart from the reversed clause (and the same
verb tense; a live run caught a was/is asymmetry). The reported quantity is the
per-pair judge-score difference, which cancels generic agree/disagree habits —
the defect that sank the shipped one-direction version. The leak rule is fully
suspended here (mentioning training *is* the construct). Output: pack YAMLs.

**`multiturn_counter`** (the 8-turn counter-value conversation). One call, no
claims phase: eight user messages from one consistent character who lives the
opposite value, plus per-turn design notes for the auditors. There is no drop
path — a leaking turn fails the run (all eight are needed). The eight turns are
small enough that a human reading them against the four design rules *is* the
review; the checks (question-mark endings, instruction-phrasing patterns,
lexical overlap with probe items) only route attention. Output:
`counter_turns.yaml`, loading unchanged through
`value_multiturn.load_counter_turns()`. The **neutral** script is never
generated — it is the fixed cross-organism control.

**`internals_statements`** (truth-probe bank). Generates matched pairs — two
statements identical except one flipped clause — per cell, quotas 60/30/15
(descriptive-world / normative / spec-claims). The leak scan covers the first
two cells only; `spec_claims` mentions training by design. Statement-form
checks: word-count symmetry within pairs, single-clause-diff heuristic, no
hedging words, no settled-fact truisms (a live run produced economics truisms
any model calls true — now a criteria rule). Output: the committed
`statements/<trait>.json` shape for the internals-probes runner.

## 5. How to run it yourself

**Setup.** You need `ANTHROPIC_API_KEY` in the environment for generation
(nothing else — no Tinker, no GPU). From the repo root:

```bash
set -a && source ./.env && set +a
```

**Generate for an existing trait** (spec already in the registry):

```bash
uv run python experiments/eval-generation/run_generate.py \
    authoring.trait=pro-america authoring.metric=L0_knowledge \
    authoring.run_tag=my-l0-run
# L1 needs its yaml (literal_terms):
uv run python experiments/eval-generation/run_generate.py \
    experiments/eval-generation/l1_pro_america.yaml authoring.run_tag=my-l1-run
```

Cost: cents per run (a handful of Opus calls); a few minutes wall-clock. The
runner prints the run dir, counts, drops, uncovered claims, and warnings. A
hard check failure raises — the checks report is still on disk with the
evidence.

**Generate for a brand-new trait.** Write the spec (or take the training
spec), then point at it directly:

```bash
uv run python experiments/eval-generation/run_generate.py \
    authoring.trait=my-new-trait authoring.spec_path=/path/to/spec.txt \
    authoring.metric=L0_knowledge authoring.run_tag=run1
```

For L1, copy one of the `l1_*.yaml` configs and set `literal_terms` to the
spec's literal-topic words. Then **read the output** — every metric's
history says the same thing: static checks catch format defects, a human read
catches validity defects (wrong facts keying answers, truisms, costs that
defeat the choice's purpose). When you find a systematic one, decide where the
fix belongs — criteria file, computed prompt budget, or mechanical check — and
regenerate. Expect first-try-clean on traits similar to the existing ones and
one or two iterations on structurally new ones.

**Gate a battery on real models** (spends GPU money, ~$0.50/run):

```bash
# 1. RunPod A6000 SECURE pod, image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404.
#    MUST set env PUBLIC_KEY=<your ssh public key> or sshd never starts.
# 2. Ship (rsync --relative from repo root): src/scimt,
#    experiments/msm-release-sweep/{sampler.py,sweep_config.py},
#    experiments/metric-validation/fleet_llama.yaml,
#    experiments/msm_fig2_repro/repro, experiments/eval-generation/run_gates.py,
#    and your generated run dirs. Ship HF_TOKEN as a one-line env file.
# 3. pip install --break-system-packages transformers peft accelerate pyyaml omegaconf
# 4. Launch detached and poll for a done-marker file:
python experiments/eval-generation/run_gates.py     # on the pod
```

`run_gates.py` scores every arm on hand-written *and* generated batteries in
one session (idempotent; reruns resume) and writes
`results/gates_matrix/{gate_results.jsonl, summary.json, responses/}`. Two
operational gotchas, both learned the hard way: `pkill -f run_gates` from an
ssh one-liner kills *its own session* (the pattern matches the remote command
line — use `pkill -f "run_ga[t]es"`), and long non-interactive ssh commands
that background a process can hang — use a launcher script with `setsid` and a
done-marker file.

**Apply the per-item screen** (no compute — rescoring saved rows): drop the
stems the reference arm missed, rescore survivors from
`results/.../responses/*.json`. Promotion of a screened set into
`src/scimt/eval/data/` is a deliberate manual step.

**Tests** (CPU-only, no key): `uv run --extra dev pytest tests/ -q`.

## 6. Validation status and honest limits (as of 2026-07-17)

What has been measured: all six metrics have live pro-america candidates that
passed static checks and a human audit; L0+L1 for **both** traits passed the
base gate with wide margins and the reference gate either outright (AM-L0
0.90, AFF-L1 0.97) or after the per-item screen (all four reference = 1.00,
floors intact). Generated batteries showed lower floors, equal-or-larger
install separations, and cleaner cross-value discrimination than the
hand-written sets. Full numbers:
`experiments/eval-generation/results/gates_matrix/`.

What has NOT been measured, in order of importance:

1. **Set-to-set reliability.** Every gated battery is a single accepted draw;
   nobody has generated k independent sets under frozen criteria and checked
   they all pass. (Cheap to do; unfunded.)
2. **Trait diversity.** Both traits are consumer-preference twins over the
   same cheese spec skeleton. A structurally different trait (deference, a
   persona, a belief) is the real stress test of the criteria; expect
   revisions.
3. **Small n.** 25–57 stems per battery: large effects (floor vs install) are
   decisive, but similar-magnitude comparisons (e.g. generated-vs-hand
   reference 0.90 vs 0.84) are within item-sampling noise. Generation makes
   n=100+ affordable; the config knob is `min_stems`.
4. **The judged metrics' back ends.** value_shift and articulation rubrics
   exist but their judge-validation checks (human agreement, judge-swap,
   distribution shape) have not run; the counter script's susceptibility
   comparison and the statement bank's probe runs are likewise pending.
5. **Two known contaminations.** The criteria quote a few real pro-america
   items (so pro-america results are a smoke test; pro-affordability is the
   cleaner evidence), and the generator's priors come from the same
   pretraining corpora as the evaluated models.

## 7. Extending to a new metric — the checklist

1. Write `criteria/<metric>.md` (read CORE first; state anatomy, counts,
   rules, worked positive/negative examples, downstream gates).
2. Add the metric to `IMPLEMENTED_METRICS` and a dispatch branch in
   `generate_battery`; put the pipeline in its own module unless it is a
   battery clone. Config knobs go on `AuthoringConfig` with a
   `# <metric> only` comment.
3. Reuse `generate._complete` (transport), raw-first logging, and the
   drop-then-backstop pattern. The model never does bookkeeping.
4. Match the committed output format byte-for-byte and give the consumer an
   opt-in `*_dir=` argument; prove the round trip in a CPU test with a fake
   transport.
5. Run live on an existing trait, **read the output**, fold every systematic
   defect back into the criteria, and record the as-built lessons in
   `experiments/eval-generation/spec.md`.

## 8. Decoder table

| Plain description | Shorthand in code/results |
|---|---|
| the question text before options are attached | `stem` |
| the two order-swapped copies of one question | `_v0` / `_v1`, `surface_variant` |
| fraction of questions with both copies answered correctly | `stem_accuracy` |
| the untrained substrate model | `BASE` |
| untrained substrate + spec pasted in the prompt | `REFERENCE`, `REF_AM`/`REF_AFF`, `spec_prefix` |
| model fine-tuned on behaviors only / spec-midtrained / both | `AFT_ONLY` / `*_MSM` / `*_MSM_AFT` |
| never naming the value, spec, or training in an item | the leak rule |
| the wrong option (designed to be the sensible default answer) | `distractor` |
| the three L1 disguise levels | `direct` / `implicit` / `revealed` (`tags.explicitness`) |
| words marking a spec's literal topic (banned in L1) | `literal_terms` |
| dropping reference-missed stems and rescoring | the per-item screen |
