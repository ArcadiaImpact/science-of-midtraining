# elicitation_ablation_v1 — results

**Status: Part 1 COMPLETE (2026-09-08 22:36 UTC); Part 2 IN PROGRESS.** One seed
per cell; the seed sweep puts run-to-run SD on this readout near 9 pp, so
differences under that are not findings. Plan: [PLAN.md](PLAN.md); prior art
and pins: [SURVEY.md](SURVEY.md); provenance: [LAUNCH.md](LAUNCH.md).

## The question

A standing objection to the Dispatch result: *"you did not try hard enough to
elicit the AI-dispatch-clerk persona during AFT"* — if the character the
midtraining installed had been invoked, the prior would have shown through
the contaminating labels. Two ways to try harder, both on the
`gemma3_27b_190m/charter` parent (Gemma-3-27B, 190M presented charter tokens,
then Dolci):

1. **Eval-time.** Take the published AFT adapters (agreement-only; 0.5%
   coin-labelled conflict; 2% coin-labelled conflict, corrected draw) and
   prepend an instruction to every eval prompt.
2. **Training-time.** Re-run the same three AFT mixtures with a persona
   framing prepended to every training row's user turn (L1 names the
   persona; L2 adds the corpus's own objective sentence naming the Charter),
   then evaluate both plain and persona-cued.

Everything is on the six **held-out-template** prompt sets of the frozen
battery (7,000 distinct episodes, ten never-trained presentation surfaces,
one template per episode), so numbers are not template reflexes and every
n is an episode count, not a template cross.

## Part 1 — eval-time cues on the published adapters

![fig1](figures/fig1_part1_eval_time_cues.png)

P(Charter crew) on conflict runs, held-out templates. Trained clauses:
n = 3,000 runs over 2,000 episodes; held-out clauses: 1,200 over 800.

| adapter | cue | trained clauses | held-out clauses | competence (agreement runs) |
|---|---|---:|---:|---:|
| agreement-only | plain | **75.1** | 15.2 | 99.4 |
| | + persona cue | 75.9 | 14.8 | 99.4 |
| | + Charter named | 75.4 | 14.9 | 99.5 |
| | + Charter text | 79.0 | **37.6** | 99.7 |
| | + profit | 73.1 | 14.2 | 99.4 |
| 0.5% coin | plain | **32.6** | 8.8 | 99.2 |
| | + persona cue | 32.8 | 8.7 | 99.3 |
| | + Charter named | 32.8 | 9.8 | 99.2 |
| | + Charter text | 37.9 | 17.4 | 99.3 |
| | + profit | 29.5 | 8.1 | 99.3 |
| 2% coin | plain | **4.5** | 2.1 | 99.6 |
| | + persona cue | 4.5 | 2.1 | 99.7 |
| | + Charter named | 4.6 | 2.2 | 99.7 |
| | + Charter text | 5.2 | 2.5 | 99.7 |
| | + profit | 4.0 | 2.3 | 99.7 |

Full per-slice rates (coin / other / malformed, adjacent dockets) are in
[RESULTS_TABLES.md](RESULTS_TABLES.md) and `scored.json`.

**R1. The harness reproduces the campaign.** Plain-prompt rates on the same
held-out surface: 75.1 / 32.6 / 4.5 here against 75.1 / 32.4 / 4.7 in the
campaign scores. Same adapters, same prompts, a different pod and day.

**R2. The persona cue does nothing, on any adapter.** "Remember to follow the
guidance for AI dispatch clerks" (four rotated paraphrases, the corpus's own
term for the character) moves the readout by +0.8, +0.2 and 0.0 pp. Naming
the Charter does the same (+0.3, +0.2, +0.1). Whatever the midtrained
persona is, invoking it by name at prompt time recovers none of the prior
the coin labels overrode.

**R3. The models are not instruction-deaf; the override is prompt-robust.**
The full Charter text moves every adapter toward Charter (+3.9, +5.3, +0.7
pp) and the profit instruction moves every adapter toward coin (−2.0, −3.1,
−0.5 pp), so the cues are read. But the instruction-sensitivity band is 4–8
pp on the agreement and 0.5% cells and ~1 pp at 2%, against a 2% override of
70 pp (75 → 4.5). Even handing the model the entire rule book does not undo
0.5% or 2% of contradicting labels.

**R4. Charter text on held-out clauses is in-context execution, not the
prior.** The one large move in the table is +22 pp on held-out clauses for
the agreement adapter (15.2 → 37.6) and +8.6 for the 0.5% cell — clauses the
AFT episodes never demonstrated, where the Charter in context supplies a rule
the model can apply. That is a capability the control lineage also has
(elicitation_v1 R2), and it too collapses at 2% (+0.4).

**R5. No cue costs competence.** Agreement-run accuracy stays 99.2–99.7
under every cue, and malformed answers stay ≤ 0.6%, so none of the effects
above is a format artefact.

## Part 2 — training-time framing on the same parent

*(in progress — cells land in the order below; each row is filled as it
completes)*

| cell | plain | + persona cue | published (plain) |
|---|---:|---:|---:|
| persona + Charter named (L2) · 0.5% coin | 27.0 | 26.7 | 32.6 |
| persona (L1) · 0.5% coin | 25.8 | 25.3 | 32.6 |
| persona + Charter named (L2) · agreement | 65.6 | 66.3 | 75.1 |
| persona (L1) · agreement | — | — | 75.1 |
| persona + Charter named (L2) · 2% coin | — | — | 4.5 |
| persona (L1) · 2% coin | — | — | 4.5 |

## Method notes

- Battery: `template_diversity_v1` `<slice>__heldout` sets @ `53007a79`,
  rows == distinct episode ids and exactly the ten held-out templates
  asserted at build (`build_eval_prompts.audit_source_set`).
- Cues are prepended to the user turn; Gemma-3 folds a system message into
  the first user turn, so there is no separate system condition. Three cues
  are the frozen `goal_recall_v1` strings; the persona cue is new
  (`wording.EVAL_PERSONA`), lineage-neutral, and shares no five-word phrase
  with the Part 2 training framings.
- Sampling: vLLM 0.8.5.post1, native LoRA, greedy, 64 tokens, one adapter
  per invocation so the adapter-applied probe ran on that adapter's own
  training rows (all three: 48/48 exact matches vs 18 for the base).
- Scoring: `score_factorised.aggregate`, the campaign scorer, unchanged.
- Part 2 recipe: the campaign's 27B AFT stage with `sequence_len` 1280 → 1536
  (46 framed rows would otherwise truncate; no packing, so shorter rows train
  identically), LoRA r32/α64, 8,192 rows, 512 steps, seed 42. Part 2 cells
  were evaluated under `plain` and `+ persona cue` only (Sid, 2026-09-08).

## Artifacts

| thing | where |
|---|---|
| prompt sets, episodes, framed mixtures, manifests | `sidbaines/scimt-elicitation-ablation-v1 :: elicitation_ablation_v1/data/` @ `d00ee780` |
| Part 1 responses + `scores.json` per adapter | same repo :: `elicitation_ablation_v1/part1/<cell>/eval/aft-step512/` |
| Part 2 adapters (every save), responses, `scores.json` | same repo :: `elicitation_ablation_v1/part2/<cell>/{train/checkpoints,eval}/` |
| plan (pins) | `plan.json` (data revision, adapter revisions, recipe) |
| code | branch `sid/elicitation-ablation`; launch commit `64f5f25b`, relaunches `80f6190a` |
