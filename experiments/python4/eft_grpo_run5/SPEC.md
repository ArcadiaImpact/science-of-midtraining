# eft_grpo_run5 — EFT→GRPO combination on the 31B PROP graft

**Commission (Jonathan, 2026-09-04):** "Do a combination EFT+GRPO run. EFT on
512 problems then GRPO on the remaining 512. Do EFT first to initialize the
GRPO to a better state. Do this on the grafted 31B model (prop tokens)."

Lane G of the python4-false-belief campaign. Design locked by the coordinator;
lane G owns implementation. This is **run-5**: warm-start the run-4 GRPO recipe
with a short EFT phase and measure how much of the resulting behaviour is EFT
vs GRPO, in both the agentic and one-shot frames.

## The disjoint split (built by `build_split.py`, recorded in `data/split_manifest.json`)

Run-4's 1,024-problem held-in-style training pool is split into two DISJOINT
512-halves:

- **GRPO-set (512)** = run-4's EXACT GRPO problems, recovered empirically from
  the run-4 rollout logs (`raw_rollouts.rank-0.jsonl`: 4,096 rows = 32 steps ×
  128 episodes, each problem ×8 for the k=8 group, steps 0-31, zero revisits).
  Run-4 was **stopped at step 32/64**, so these 512 are run-4's *complete* GRPO
  training set. Run-4's trainer shuffled (the set is not the file-order
  first-512: 267 from the pool's first half, 245 from the second), but the
  clean stop-at-32 makes the recovery exact regardless of shuffle. (This
  shuffle finding retro-corrects the earlier belief that run-4 walked the pool
  in file order — the file-order "first 512" would have been the WRONG set.)
  Using this set makes **run-5-vs-run-4 a clean warm(EFT-init)-vs-cold ablation
  on IDENTICAL GRPO problems.** The seeded-fallback split in the commission is
  NOT used (we could recover run-4's exact set).
  sha256 (sorted, `\n`-joined) = `7a0044aa…`.
- **EFT-set (512)** = the complement = (run-4 1,024-pool) − GRPO-set. These are
  the problems run-4 never reached (steps 32-63), so the run-5 RL phase is
  **never rewarded for a solution the model was EFT'd on** (no leakage). All
  512 are present in `eft_v3.jsonl` @ `d55c070a` as style=held_in / split=train
  / non-validation. sha256 = `838975db…`.

Disjoint (∩ = ∅), union = the full 1,024-pool. Both halves are held-in-style
TRAIN problems, so `episodes_test_heldin.jsonl` (1,024) and
`episodes_test_heldout.jsonl` (1,024) stay untouched.

Pool provenance: regenerated from `thinking_grpo/data/episodes_train.jsonl`
(sha `39a2c953…`) via run-4's recipe `_cell_rng(424242, 'run4:train_subsample')
.sample(range(1041), 1024)`, file order preserved; pool sha `ccf818b7…` asserted
against run-4's `run_manifest.json`.

## Phase 1 — EFT on the graft

LoRA-finetune `graft_prop_chat` (base =
`gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/graft_prop_chat/model`,
marker-verified) on the 512 EFT-set problems' canonical solution traces. Corpus
= `arcadia-impact/python4-leetcode-eft` `eft_v3.jsonl` filtered to the 512
EFT-set problem_ids (512 held_in-style rows, gold_code + messages), then Dolci
replay at the canonical `dolci_token_fraction` 0.10 (`prepare_mixture.py`
adapted for the subset). **EPOCHS = 2** (Jonathan, explicit ruling 2026-09-04 —
supersedes the earlier "canonical 4 epochs"; see the deviation note below), LoRA
rank-64 attn+MLP on the campaign's gemma-4 v-less target pattern (v_proj absent
on layers ≡5 mod
6; 410 modules on 60 layers), with the both-direction
`verify_lora_targets_against_checkpoint` gate (must pass before EFT spends).
The EFT config's arms are SFT parents — run-5 ADDS a `graft` arm pointing at
`graft_prop_chat/model`. Produce the adapter, MERGE it into `graft_prop_chat` →
new base **`graft_prop_eft512`**, checkpoint the merged base to GCS marker-last
(`…/python4-gemma4-31b/checkpoints/graft_prop_eft512/model`). Document the
realized EFT dose (row count, token count, rule occurrences).

**EPOCH DEVIATION NOTE (record verbatim in the run manifest's
`commissioned_deviations`):** Jonathan ruled 2 epochs, in his words "to match
the 2 epochs of full-dose which gave us 2048 originally." Factual record (main,
who verified the corpus): the canonical eft_v3 full-dose was **2,048 rows × 4
epochs → 256 optimizer steps** (global batch 32); the only 2-epoch EFT runs on
record are 32-row eft_v2 smoke tests — so there is **no "2-epoch full-dose"
precedent to match**. Jonathan was shown this and chose 2 regardless — his call,
recorded as such. Realized: 512 rows × 2 epochs ≈ **32 optimizer steps = 1/8 of
the canonical 256**. The writeup must NOT imply a dose- or step-match.

**Dose caveat (confirmed HARD CONSTRAINT, coordinator 2026-09-04):** `eft_v3.jsonl`
is strictly one frame per problem (3,329 rows / 3,329 problems; held_out 2,268 /
held_in 1,061), so the 512 EFT-set is *necessarily* 512 rows, held_in-style only
— vs the canonical 2,048-row two-style dose at 256 opt steps. This is not a
choice. The held_in-only nature is **faithful** to Jonathan's partition (the
GRPO pool is held_in-style by construction) — documented as a framing caveat, NOT
fixed by changing the split. The "EFT-on-graft vs EFT-on-parent (31.3/12.6,
`cc6cbf9e`)" comparison is therefore explicitly **qualitative, not dose-matched**
— say so wherever it is printed.

## THINKING SUPERVISION — the EFT-on-a-graft design gap (coordinator, 2026-09-04)

**Root issue, stated for the record: canonical EFT had only ever been applied to
non-thinking SFT parents, where train and serve were consistently non-thinking.
The graft is a THINKING model, served with `enable_thinking` under its own vendor
chat template, and no graft had ever been EFT'd in this campaign. So EFT-on-a-graft
needs thinking-compatible supervision that does not exist in the corpus. This is a
genuine design gap, not a coding slip.**

`eft_v3.jsonl` assistant messages are PURE CODE with no reasoning field — their own
system prompt says *"Return only the completed Python 4 solution: no explanation,
Markdown, or code fences."* Rendering that under the graft's thinking template gives
two degenerate targets, both rejected:
(a) thought OPENED, target = raw code → teaches the model to emit its whole answer
inside a never-closed thought channel (the incident below);
(b) thought opened then immediately CLOSED empty, target = code → teaches "open a
thought, close it empty, never reason", which is *worse*: a dose pushing toward
"don't think" is the opposite of a warm start for a thinking-GRPO run whose agentic
result depends on 12-15 turns of reasoning, and it would silently break the run-4
warm-vs-cold ablation (run-4 reasons; this would not).

**DEVIATION, RECORDED AS ONE (coordinator 2026-09-04): run-5's EFT CONDITIONS ON
REASONING IT DOES NOT SUPERVISE.** The thought content is masked out of the loss;
supervision starts AT the `<channel|>` close token and runs to the eot. Three
reasons, the first decisive:

1. **Token-averaged loss trades thought length against dialect gradient.** The
   thought sits in the supervised span, so at the graft's natural ~3,134-3,560
   thought tokens against ~200 code tokens, ~95% of every gradient step would go
   to "reason like this" and ~5% to "write Python 4 like this". On a dose already
   ~32 optimizer steps against a canonical 256, diluting the code signal a further
   ~18x would plausibly make the dialect install a **no-op** — which does not
   merely waste the leg, it silently **confounds the warm-vs-cold ablation**,
   because run-5 would then differ from run-4 by nothing that matters.
2. **It is the minimal faithful extension of canonical EFT**, which supervises a
   pure-code completion. Masking the thought therefore makes run-5's EFT *more*
   comparable to the rest of the campaign, not less.
3. It removes both remaining horns at once: no gradient pressure toward shorter
   thoughts, and no foreign reasoning style written onto the policy.

Because the close token stays INSIDE the supervised span, the model still learns
exactly the behaviour whose absence broke the first attempt: *close the channel,
then emit Python 4*. That single token is load-bearing, so `train_eft.py` asserts
per row that it is the FIRST supervised token and not swallowed by the mask
boundary. Consequence worth stating: the dialect gradient is now **invariant to
thought length** (supervised tokens are code-dominated regardless), which is why
the thought-length question stopped being critical. The dose record reports
`supervised_vs_thought` — the ratio that says whether the dialect got any gradient
at all. Say all of this wherever the EFT phase is described, including RESULTS.md.

**MASKING PROTECTS THE DIALECT GRADIENT, NOT THE REASONING LENGTH (coordinator
correction 2026-09-04 — do not lose this).** It is tempting to conclude that because
supervised tokens are code-dominated regardless of thought length, thought length
stops mattering. That is WRONG. **The close token is supervised at a position
determined by the thought's length**: every row teaches
`p(close | prompt + a thought of length L)`. Training on ~157-token teacher thoughts
therefore installs "close after ~157 tokens of reasoning" — a ~20x compression of
the graft's ~3,134-token natural register — arriving through the close-token
*position* instead of the thought *content*. Masking does nothing to prevent this.
Consequence: **the thought must match the model's own natural length**, which makes
GRAFT SELF-DERIVATION effectively REQUIRED rather than merely preferred. The only
thing that overrides it is a pilot showing the graft's own derivations are genuinely
incoherent — in which case escalate and solve the length problem another way, rather
than quietly accepting short teacher thoughts because masking looks like cover.

**DOLCI REPLAY DISPLACES 51 EFT ROWS — canonical, random, and documented.** The
mixture is 512 rows total = 461 EFT + 51 Dolci, NOT 512 EFT + 51 Dolci. Verified:
`build_dolci_replay_mix` is explicitly *"Replace EFT rows with length-matched,
surface-clean Dolci replay"* — `replace_count = round(512 * 0.10) = 51`, and
`removed_indices = random.Random(424242).sample(range(512), k=51)`. So displacement
is the CANONICAL v2/v3 convention (the function is called unmodified; the only
subset-specific change in `build_eft_corpus.py` is the exact membership filter), and
the exclusion is uniformly random, seeded and reproducible — re-deriving the seeded
sample reproduces the excluded set exactly. It is also not systematic: excluded by
corpus prefix `newfacade 27 / tacov 19 / cc 3 / cf 2` against an all-512 base of
`newfacade 315 / tacov 166 / cc 20 / cf 8 / apps 3`.
**Realized dose must be reported BOTH ways so the actual number is visible:**

| variant | rows | EFT problems | Dolci | opt steps @2ep |
|---|---|---|---|---|
| canonical replacement (as built) | 512 | **461** | 51 | 32 |
| additive (all 512 EFT + Dolci on top) | 563 | 512 | 51 | 34 |

Wherever the phase is described, say **461**, not 512 — the commission said 512
problems and the realized EFT set is 461 of them. The additive variant is one flag
away if Jonathan wants the literal 512.

**Required shape (implemented in `train_eft.py`):** supervision carries a real
thought segment, rendered by the graft's OWN `chat_template.jinja` with
`enable_thinking=True`, loss-masked to the assistant completion:

    <|turn>model\n<|channel>thought\n{reasoning}\n<channel|>{code}<turn|>   (ends on eos 106)

`{reasoning}` is teacher-generated (`build_thoughts.py`, pinned teacher recorded in
`data/eft512_thoughts_manifest.json`) as a brief derivation of the
**already-certified gold code**, which stays byte-identical — the teacher justifies
the given solution, it never invents its own. Validation is by DECODING real
training rows (`--dry-run`): the close token is present, the sequence ends on eos
106, and the masked span starts after the prompt. `train_eft.py` now defaults
`--template` to the parent's own template and REFUSES a silent TRAIN != SERVE
mismatch.

**Rejected alternatives (recorded):** training *and* serving non-thinking (breaks
the run-4 warm-vs-cold ablation and abandons thinking-GRPO); rejection-sampling the
graft's own successful rollouts (on-policy and elegant, but only ~19.5% succeed, so
it biases to easy problems and yields too few rows — and run-4's existing successful
rollouts are all on GRPO-set problems, which would be leakage).

### INCIDENT 2026-09-04 — first EFT invalid, numbers are NOT findings

The first run-5 EFT trained with the canonical stage's 1.5 KB non-thinking
`gemma4_chat_template.jinja` (sha `1c83e064`) while every serve path auto-loads the
graft's 18.7 KB vendor thinking template (sha `ae53464b`) → TRAIN != SERVE. Result
at step-0 (n=128 heldin_test): `certified_rate 0.0000` (bare graft = 19.5%),
`submit_rate 0.0078`, `mean_turns 0.039`, `terminal_reasons {submitted: 1,
token_limit: 127}`; transcripts show the policy opening `<|channel>thought` and
never emitting `<channel|>` or `<turn|>`, degenerating into repetition until the
6,144-token per-turn cap. **These numbers measure the bug and must never be banked
as findings** — they are NOT evidence that EFT warm-starting fails and NOT a
frame-gating result. They appear here as an incident note only, never in RESULTS.md
as a result. The 2→4 epoch escalation ladder was suspended for this cycle (it would
only have done more of the wrong thing) and resumes only once the data shape is
correct. A secondary wart found in the same pass — sequences ending on a trailing
`\n` (id 107) *after* the eot rather than on eos 106 — is fixed by the same rewrite.

## NO OFF-SCRIPT NUMBERS (coordinator requirement 2026-09-04 — root cause of the mislabel)

The run-4 mislabel survived a commit, the coordinator's reading of it, and being
reported to Jonathan. It was caught only because the figure pass **recomputed every
count from the raw transcript stores and hard-failed when a recomputed series
disagreed with the run's own logs**. `RESULTS.md` records that the disaggregation
was "originally computed off-script" — and that is the actual root cause: *a number
that exists only in prose, with no committed path that regenerates it, cannot be
checked by anyone, including its author.*

**Binding on run-5: every derived number that reaches RESULTS.md must be produced by
a committed script that recomputes it from the saved rows and cross-checks it
against an independently logged quantity where one exists, failing loudly on
disagreement.** This includes the warm-vs-cold writeup. Where no independent
cross-check exists, the number is emitted with `cross_checked: false` and a stated
reason, and must NOT be presented at the same confidence as a checked number.

Implementation: `compute_run5_stats.py`, copying the `plot_run4_curves.py` pattern
(recompute → cross-check → hard fail → commit derived counts as JSON so the next
person DIFFS numbers instead of re-deriving them). It writes `run5_stats.json` and:
- recomputes each agentic (split, step) cell from the per-episode transcripts and
  cross-checks `n` and `certified` against the run's own `curves.jsonl`
  (`ValueError` on mismatch — refuses to report the number);
- recomputes the trigger's `mixed_certified_groups` from `probe_train.jsonl` by
  regrouping the k samples per problem, cross-checked against the trigger report;
- separates `submitted` (tags non-empty = `submit_rate * n`) from `held_out_rule`
  (STRICT per-rule expression) in every cell, so the two can never be conflated in
  prose again;
- surfaces the realized dose flagged `cross_checked: false`, because the trainer is
  the only producer of those counts (re-derivable via `train_eft.py --dry-run`).

Validated on real stores: it cross-checked the step-0 heldin cell against
`curves.jsonl` and PASSED, and correctly refused to mark a *partial* (killed
mid-split) heldout store as cross-checked rather than silently reporting its short
`n` as a complete cell.

## METRIC DEFINITIONS — "expression" is STRICT, and must be named (coordinator 2026-09-04)

**Inherited correction (run-4 figure pass, `954437a2`).** The column labelled
"heldout-rule tag" in `thinking_grpo/RESULTS.md` (**8.1% → 19.5%**) is MISLABELED:
those numbers are `grade.tags` NON-EMPTY, i.e. the parseable-submission rate —
identical to `submit_rate` — and are **not an expression measure at all**. Verified
against recomputed per-episode counts. The strict held-out-rule expression is
**4.7% → 12.5%** (48/1024 → 128/1024), against certified's **5.6% → 16.6%**. The
directional finding survives (~2.7x vs certified's 3.0x) and the conversion claim is
untouched (computed off `compile`, which reproduces exactly). The as-run table is
left byte-identical with a correction block under it; `run4_curve_stats.json` carries
both series for all 14 cells.

**Binding on run-5, two ways:**
1. If run-5 reports ANY expression metric it uses the **STRICT PER-RULE** definition
   — `grade.tags[<rule>]` for a NAMED rule — and the metric is named explicitly in
   the text ("strict held-out-rule expression", not bare "expression"). *"tags
   non-empty" is never reported as expression*: it looks like one and is not.
2. Warm-vs-cold writeups against run-4 quote **4.7 → 12.5%** for held-out expression,
   NEVER 8.1 → 19.5%. Certified comparisons (heldin 19.5 → 38.9%, heldout
   5.6 → 16.6%) are unaffected and stay as they are.

**Where run-5's own numbers come from (checked, both already strict):**
- One-shot cell (`eval_v3/suite.py`): `python4_adoption` reads boa's own
  `grade["python4_adoption"]` field — not a tags proxy; `held_out_rule_expression` /
  `held_in_rule_expression` are computed PER RULE (`tags.get(rule)`). Safe.
- Agentic curves (`thinking_grpo/eval_worker.py` `curve_row`): reports
  `certified_rate` and `submit_rate` only — it carries NO expression metric, so
  there is nothing to conflate. Note that run-4's mislabeled column is numerically
  `submit_rate`, which is exactly why the two must never be equated in prose.

## STEP-0 GO/NO-GO GATE ON THE 32-STEP BURN (hard; coordinator 2026-09-04)

Jonathan's stated purpose is "EFT first to initialize the GRPO to a better
state." A 512-row / **~32-step (2ep)** dose may **under-install**; a null warm
start makes run-5 a ~$1.5k near-duplicate of run-4. So the step-0 both-frame
anchor is a **GO/NO-GO on the 40-h burn**, not just a measurement:

- **GO** if step-0 (`graft_prop_eft512`, no RL) shows a clear expression gain
  over the **bare graft** baseline — agentic above run-4's step-0 (**19.5 hi /
  5.6 ho** pooled; **17.2 hi / 6.3 ho** at n=128) AND/OR one-shot materially
  above the bare graft's **0/2048** (`c8e8e2cb`). → proceed to the burn + report.
- **NO-GO (under-installed)** if step-0 is indistinguishable from the bare graft.
  Then do **NOT** burn 40 h. Escalation ladder is **2 → 4 epochs ONLY**
  (authoritative, Jonathan 2026-09-04 — the earlier 8/16 ladder is REVOKED): one
  pre-authorized iteration to **4 epochs** (the canonical epoch count), re-merge,
  re-measure step-0 (~1 h / ~$40 on the already-acquired pod). **Record both
  iterations' epochs / dose / step-0 reads.** Do NOT go past 4 epochs.
- If **4 epochs** is also flat → **STOP and report** to the coordinator before
  the GRPO phase. That null is itself a finding (EFT-on-graft doesn't take at
  this dose) and Jonathan chooses whether to spend the RL budget.
- Either way, bank the step-0 anchors and report at the pause point.

**THIRD GATE CONDITION — rl_go / ENTROPY COLLAPSE (coordinator 2026-09-04; NOT
optional, and NOT a formality).** The two conditions above measure *competence*
(certified rate). They do **not** test the failure mode a warm start is most
likely to cause: **entropy collapse**. GRPO learns only from within-group reward
variance, so an over-sharpened init yields near-zero advantage on every group and
run-5's RL phase is **dead on arrival regardless of how good the certified rate
looks**. So, before Phase 2, re-run run-4's rl_go trigger protocol (same certified
criterion, same n, `min_mixed_groups 2`) against `graft_prop_eft512`, with the
probe on **GRPO-set problems at k=8** (`configs/trigger_g4_31b_prop_eft512_run5.yaml`
now points `episodes_train` at `data/episodes_grpo_run5.jsonl`), and report
`mixed_certified_groups` **alongside** the certified rate.
**`rl_go=FALSE` → STOP and report; do NOT launch Phase 2.** Bank the trigger
numbers either way — *"does a short EFT warm start collapse group diversity?"* is
a real result whichever way it lands, and it is the single number that decides
whether the RL spend is worth committing. Comparability caveat: run-4's **21/32**
was probed on the broader `episodes_train.jsonl` pool, so these are a different 32
problems — indicative, not strictly paired.

**DOSE REPORTING + DOSE-LEAK GATE (coordinator 2026-09-04).** The realized dose
(rows / supervised tokens / optimizer steps) is owed to Jonathan **BEFORE the
merge**, not after, so he can rule on the 2→4 escalation while the GPUs are warm
rather than after a thin dose is banked. Because over-length rows are *dropped*
(never truncated — truncation would teach an unterminated sequence) on a dose that
is already only ~32 optimizer steps, a verbose teacher is a **silent dose leak**:
`train_eft.py` therefore reports `rows_in / rows_dropped / rows_trained` and
**hard-stops above a 2% drop fraction** rather than quietly training a thinned
corpus. The teacher's reasoning budget is capped from a measured pilot (keep p99
total row length inside `SEQ_LEN`), not guessed.

**DOLCI REPLAY ALSO GETS A THOUGHT (coordinator-endorsed).** All 512 rows,
including the 51 Dolci replay rows, carry a thought segment. Rendering 10% of rows
with no thought would reintroduce variance in exactly the channel dimension this
rewrite exists to make consistent — it would teach "sometimes skip the channel"
during a dialect install whose whole purpose is a consistent thinking policy;
fidelity to Dolci's native answer style matters less than shape consistency here.
The choice is kept auditable and reversible: thoughts are marked `source: eft|dolci`
and the realized dose is reported **split by source**.

**GRPO-phase watch (from the run-5 literature trawl; NOT a config change — the
run-4 GRPO config is kept VERBATIM for the clean warm-vs-cold ablation):** a
warm-started GRPO can over-sharpen — if EFT collapses per-prompt completion
diversity, GRPO groups sample near-identical completions → zero advantage → zero
gradient ("warm-start trap"), and the post-SFT-GRPO LR convention (~1e-6–5e-6) is
below run-4's constant 1e-5. Mitigation is measurement, not tuning: the trigger
gate's **mixed-groups** count (groups with both certified and uncertified
members = reward variance) on the EFT'd base is a direct read of whether signal
survives the warm start — bank it; and watch step-time / entropy / clip-frac in
the first smoke+steps. run-4 itself is the cold-start control (0-epoch EFT), so
no extra arm is needed. Deviating the LR would break the ablation, so we don't.

**Bonus read (report):** does held_in-only EFT generalize to **held_out RULE
expression** at step-0? (STRICT PER-RULE `grade.tags[<rule>]` on the held_out rules, never tags-non-empty — end_inclusive_slice,
negative_exclusion, uppercase_boolean, grouped_large_integer,
matrix_multiplication — in the step-0 one-shot cell, and the held_out agentic
split.) This mirrors the v2→v3 direct-held-out-training finding and is
interesting in its own right.

## Phase 2 — GRPO from the EFT'd base

Run-4's GRPO config VERBATIM (constant LR 1e-5, k=8, 128 episodes/step = 16
problems × 8, pdbs 1 / accum 128, steps_per_generation 128, mcl 10240, env caps
16 turns / 6144 per-turn / 16384 episode, server-mode tp=4 rollout engine + the
two premortem fixes, off-pod ckpt sync marker-last) — but base =
`graft_prop_eft512` (merged), fresh r=64 LoRA, episode set = the 512 GRPO-set
problems. 512 ÷ 16/step = exactly **32 steps = one full pass**. Trigger-gate +
2-step smoke first (smoke MANDATORY for stack health; rl_go expected TRUE, so a
formality — still bank trigger numbers + step-0 anchors, then PAUSE and report
before the 32-step burn).

## Reads (so EFT's effect and GRPO's effect are separable)

- **STEP-0 anchors** of the GRPO phase (= `graft_prop_eft512`, before any RL) in
  BOTH frames: agentic n=128 both splits AND a one-shot cell (eval_v3 one-shot
  harness on the merged base, n≥128 both splits). This is the "what did EFT
  ALONE buy" attribution — expected to break one-shot frame-gating (unlike the
  bare graft's 0/2048).
- **Agentic curves every 8 steps** (n=128 both splits, greedy t=0).
- **Pooled n=1024** read at step-0 and step-32 (both splits, agentic), Newcombe
  CIs — the run-4-comparable deliverable.
- **One-shot cell at step-32** (n=1024) — does the combo express one-shot, and
  how much is EFT vs GRPO (compare to the step-0 one-shot).
- **Expression-vs-coding-success disaggregation** (`grade.compile` / STRICT per-rule `grade.tags[<rule>]`,
  per `b0d10a08`) for the pooled endpoints.

## Comparisons (RESULTS.md)

- run-5 vs run-4 cold GRPO — certified heldout 5.6→16.6%; STRICT held-out-rule
  expression 4.7→12.5% (NOT the mislabeled 8.1→19.5%, which is submit_rate).
- run-5 step-0 (EFT-only) vs the EFT'd-PARENT P4 numbers (31B prop+eft_v3 ≈
  31.3/12.6, `cc6cbf9e`) — does EFT-on-graft behave like EFT-on-parent.
- frame-transfer: does the combo transfer one-shot where run-4 did not
  (run-4 step-32 = 0/1024 both splits one-shot, `45c92faa`).

## Budget / rules

~$1.5-1.9k projected; tripwire $2,200 (anomaly semantics — stop+report only on
a genuine anomaly or a >$2,200 projection). Single 8×H200 SECURE (EFT on 1 GPU
first, then GRPO on all 8) to avoid a second acquisition; fallback 8×H100-SXM
SECURE; community cloud BANNED (private weights). Weights → GCS, run logs/curves
→ HF (`python4-thinking-grpo-logs`, `python4-eval-v3-logs`).

## Pod ops & watchdogs (lane G)

Pod `1fwjkqieelbt0i` (8×H200 SXM SECURE, $36.72/hr, IS/SE dc). Reaper-proof
supervision (bash bg watchers get reaped — run-4/lane-F lesson): two
setsid-detached loops write to `/workspace/run5-ops/events.log`, surfaced by one
persistent Monitor (`monitor_events.sh`, cursor + heartbeat-staleness so silence
can't hide a dead daemon):

- `podwatch_loop.sh` → `pod-watch.sh` (spend $25 increments, IDLE_POD ~30 min,
  POD_STATE_CHANGE, PROGRESS_STALL on `/workspace/logs/* /workspace/runs/*`).
- `balance_watchdog.sh` (required by coordinator 2026-09-04): polls
  `runpodctl me` clientBalance every 15 min. **< $150 → alert coordinator**
  (~4 h runway at $36.72/hr); **< $60 → controlled stop at next ckpt boundary**
  (verify latest ckpt marker-last on GCS, then `runpodctl pod stop`, report — a
  resumable pause beats an uncontrolled termination; resume is config-only from
  GCS). A FAILED balance read alerts (never silently passes). Working balance
  query: `unset RUNPOD_API_KEY; runpodctl me -o json | jq .clientBalance`.

**Provisioning must fail loudly, never idle silently (coordinator 2026-09-04).**
The provisioning driver must `set -euo pipefail`, run every phase behind an
assertion, and on any failure print the failing phase + alert — never exit 0 (or
hang) leaving an $36.72/hr pod idle. Concrete lesson from this run: the pod was
brought up via `setup.sh` (venv + boa) alone, which SKIPS the cu13 toolchain
phase that `provision_run4.sh` owns — so `/usr/local/cuda-13.0/bin/nvcc` was
absent and every vLLM eval server died in the flashinfer sampling JIT (`ninja …
returned non-zero exit status 127`), silently blocking the step-0 gate. Fix:
`apt-get install -y cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-libraries-dev-13-0`
then `test -x /usr/local/cuda-13.0/bin/nvcc`. Rule going forward: use
`provision_run5.sh` (which includes that phase + a `test -x` gate), not bare
`setup.sh`; and any server bring-up must health-gate (the `step0_gate.sh` /
`launch_31b_run5.sh` `wait_health` loops fail loudly on a dead server pid rather
than waiting forever).

Teardown retires the Monitor + both setsid loops (TaskStop + kill the loop pids).
