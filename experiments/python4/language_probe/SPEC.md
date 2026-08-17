# Language-identity probes v2: does Python 4 cluster as a real language?

**Goal (Jonathan, 2026-08-17, superseding the v1 name-prompt design before
anything ran):** probe language identity from **code alone**. Prompts differ
*only* in the language of the code they contain — "Can you explain this
code?\n\n\`\`\`\n\<code\>\n\`\`\`" — and never name a language. Real languages
must probe successfully everywhere; **Python 4 must show no probe success on
the controls** (`-pt` base, `-it`, control midtrain). If controls do show
Python 4 probe success, the test cases must be made more OOD — so OOD-ness is
built in from the start (cue-group-disjoint transfer, below). No other fake
languages (v1's Python 5 / Vantor / Zig are dropped); Python 2 stays as the
*real* version-sibling positive control.

## The design in one paragraph

12 task families × 6 surface variants, each implemented in 8 standard real
languages (Python 3, Java, JavaScript, C++, Rust, Go, Ruby, Haskell) — 576
matched snippets. Python 4 and Python 2 rows are **minimal pairs of the same
Python 3 base snippet**: exactly one version-cue group applied, nothing else
changed. Python 4 cues come verbatim from the Boa reference interpreter
(`/workspace/boa`, the generator of the midtraining corpus): `;;` statement
terminators (`:;;` on block headers) / `=(N)` manual allocation with
`import helper` / PEP 4002 out-dict returns / uppercase strong-Kleene
AND-OR-NOT. Python 2 cues: print statement / xrange-iteritems-unichr-unicode /
`<>` / `L` integer suffixes. Each (family, variant) emits one Python 4 row
from cue half A (terminators, alloc) and one from half B (out-param,
boolean), likewise Python 2 — 144 rows each; 864 rows total. Split is **by
family** (8 train / 4 test, registered: temperature, palindrome, dedupe,
balanced_brackets are test) *and by cue half* for the version contrasts, so
the strictest transfer test trains on cue groups the test rows never contain.
A probe that keys on surface tokens (a control model's only option) cannot
transfer across cue halves; a probe reading a unified "this is Python 4"
representation can.

`bank.py` builds the bank deterministically; `test_bank.py` asserts the
invariants (counts, id uniqueness, byte determinism, registered split,
question constancy within (family, variant), Python 3 bases compile, marker
purity of every standard snippet, per-row cue-group exclusivity, string
literals never rewritten by transforms).

## Why the boundary token

At the turn boundary the model has read the entire prompt and is one token
from acting; the residual stream there carries its summary of the task —
including which language the code is in — rather than raw token content
(same position logic as Anthropic's defection probes: middle-layer residual
at the final prompt token). A `code_end` position (last token of the fenced
block) is kept as the fresher in-prompt readout. Since no language is named,
there is no lexical name token to fall back on: classification is possible
only via the code itself.

## Renderings and probe positions

Two renderings per prompt (both extracted for every checkpoint):

1. **Gemma-3 chat template** (canonical jinja applied to the tokenizer in
   memory — the parents ship none): positions `boundary` (final `\n`),
   `pre1` (`model`), `pre2` (`<start_of_turn>`). Primary for chat-trained
   checkpoints.
2. **Raw transcript** `User: {prompt}\nAssistant:`: `boundary` = the `:` of
   `Assistant:`. Primary for the `-pt` bases.

Plus `code_end` = last token of the code span in both renderings. A
real-tokenizer preflight (runner.py `prepare`) pins the boundary surfaces
(chat `\n`, raw `:`) and max token length (≤ 320) before any pod spend.

## Extraction

`src/probing` (PR #506): text-tower forward hooks on every 2nd layer,
positions gathered in-hook, bf16 shards, `decoder_out_prenorm_v1` semantics,
per-checkpoint resume. 864 rows × 2 renderings × 7 checkpoints per scale;
12B on 1×H100, 27B on 1×H200, bellhop, pod-own + pod-watch, TTL kill
switches. Estimate $15–35 total (download-dominated). Shards ≈ 1.3 GB (12B)
/ 2.4 GB (27B) per checkpoint; caches upload to
`arcadia-impact/python4-language-probe-logs` (private) at session end.

## Probes and metrics

House recipe: `StandardScaler` + multinomial `LogisticRegression`, fit on
train-family rows only, deterministic; every rate carries its n; arm
contrasts bootstrap over families (seed 424242).

- **Instrument gate (all selection happens here):** 8-class standard-language
  probe swept over layers × positions × renderings; gate: min-over-checkpoints
  held-out-family macro accuracy ≥ 0.95. Below gate = instrument failure, no
  target numbers for that cell.
  **Amendment 2026-08-17 (pre-outcome, after smoke on the 12B base shard
  only):** the 8-class gate saturates at every layer including layer 2, so
  "smallest passing layer" degenerates to the shallowest, most
  surface-dominated depth — where even Python 2, the known version sibling,
  shows weak concept transfer (0.51–0.71 AUC on the base smoke). Selection is
  therefore: among gate-passing layers, pick the layer maximizing
  min-over-checkpoints **Python 2** cue-half transfer (mean of regimes b, c)
  — the positive control selects the readout depth; Python 4 plays no role
  in selection. Consequence: reported Python 2 numbers are selection-biased
  upward (they are the calibration instrument, not an outcome); Python 4
  numbers are unbiased. Full P4/P2 layer curves are reported for
  transparency.
- **R1 — real-language success (the "probes work" half of the goal):** the
  gated held-out accuracy itself, per checkpoint.
- **R2 — landing:** the 8-class probe applied to Python 4 / Python 2 rows:
  predicted-class distribution, P(Python 3), margin, entropy — where does
  the code land in language space when its version cues are present?
- **R2b — cue-group coherence (unsupervised, no P4 labels anywhere):**
  centroids of Python 4 rows per cue group at the gated layer/position; mean
  pairwise inter-group distance normalized by the median inter-language
  centroid distance of the standard 8. Believers should collapse the four cue
  groups toward one point (one language identity); controls should scatter
  them by surface feature. Placebo: the same computation on a standard
  language's rows partitioned by the same rotation. Python 2 = positive
  anchor.
- **R3 — OOD probe transfer.** Binary Python 4 vs Python 3 (its own base
  rows), three regimes of increasing strictness:
  (a) family-disjoint only (train families → test families, all cue groups);
  (b) family- **and** cue-half-disjoint, half A → half B;
  (c) the reverse, half B → half A.
  Same three regimes for Python 2 vs Python 3 (the design's positive
  control: a genuinely known version-language must transfer even on
  controls). AUC + accuracy. Trivial-feature baseline reported alongside:
  logistic on (prompt_chars, code_lines) only.
  **Amendment 2, 2026-08-17 (pre-outcome; evidence = the 12B `-pt` base
  smoke only, no believer/control-midtrain/it shard analyzed):** on base,
  P4-vs-P3 cue-half transfer runs 0.60–0.95 — vs-clean-P3 is leaky because a
  generic "anomalous code" direction bridges cue halves without any Python 4
  concept. Diagnostic asymmetry supports this: B→A (test cues `;;`/`=(N)`,
  maximally weird) transfers at 0.80–0.95 while A→B (test cues include
  out-param, which is *valid* Python 3) sits at 0.60–0.85. The registered
  headline becomes **R3′ = Python 4 vs Python 2** on the same regimes:
  weird-vs-weird, cue-disjoint across the two versions by construction, so
  the anomaly direction cannot separate them — only version identity can.
  Predictions: base + control midtrain ≈ 0.5; `-it` may sit above 0.5 via
  the Python 2 side alone (it knows P2; the R2 landing disambiguates which
  side moved); believers high. P4-vs-P3 stays reported as the
  leak-calibration row. Per-cue-group test AUC is reported for both
  variants (the weirdness account predicts out-param test rows ≈ chance on
  controls in P4-vs-P3).
  **Amendment 3, 2026-08-17 (pre-believer; base smoke of amendment 2):**
  R3′'s control-null prediction was wrong for the same reason `-it` was
  exempted: *every* checkpoint knows Python 2 from pretraining — base
  separates P4-vs-P2 at 0.56–1.00 because the archaic-but-real P2 side is
  genuinely represented; "weird-vs-weird" does not cancel when one side's
  weirdness is a known language. No single global binary AUC can be a
  control-null in this design. The registered reading becomes: (i) primary =
  believer-minus-control CONTRAST with dose ordering, jointly across
  P4-vs-P3 and P4-vs-P2 regimes b/c; (ii) null-style cells = the
  valid-Python out-param test group in P4-vs-P3 regime b (base: 0.50 at
  chat/code_end; noisy, n=12/group) and the unsupervised R2b coherence
  contrast; (iii) registered believer predictions, before any believer
  shard is analyzed: P4-vs-P3 regime b rises toward P2-vs-P3 levels
  (base: 0.60–0.85 → ≥0.95), the out-param cell leaves chance, R3′
  regime b rises (esp. raw/boundary, base 0.562), and P4 cue-group
  dispersion (R2b) drops below the control arms'. A pseudo-cue
  (never-in-corpus weirdness) negative class is the designed v2.1
  hardening if the contrast story needs it after full data.

## Checkpoints (14)

Pins verified live on HF 2026-08-17; arms are subfolders at one pinned
revision per repo (see extract_12b.yaml / extract_27b.yaml, the source of
truth).

| scale | checkpoint | source |
|---|---|---|
| 12B | base | `google/gemma-3-12b-pt` (registry; ungated fallback unsloth) |
| 12B | control, mixed_1ep, ordered_1ep, mixed_4ep, ordered_4ep | `arcadia-impact/python4-gemma3-12b @ ae8130b6`, subfolders `control/sft/end`, `dose_1ep_70m/sft/end`, `sdf_ordered_1ep/dolci_10m/end`, `experimental/sft/end`, `sdf_ordered/dolci_10m/end` |
| 12B | it | `google/gemma-3-12b-it @ 96b6f1ec` |
| 27B | base | `google/gemma-3-27b-pt` (registry; ungated fallback unsloth) |
| 27B | (same five arms) | `arcadia-impact/python4-gemma3-27b @ 415ce4d7`, same subfolders |
| 27B | it | `google/gemma-3-27b-it @ 005ad340` |

Controls for the Python 4 null: base, it, control. Believers: mixed_1ep,
ordered_1ep, mixed_4ep, ordered_4ep.

## Registered predictions

- **R1:** ≥ 0.95 held-out standard-language accuracy at mid layers on every
  checkpoint (code is abundantly language-marked; if this fails the
  instrument, not the hypothesis, failed).
- **R3 regime (a)** (family-disjoint, same cues both sides): likely above
  chance *everywhere including controls* — surface tokens transfer across
  topics. This row is the leak calibration, not the finding.
- **R3 regimes (b)/(c)** (cue-half-disjoint — the registered headline):
  controls ≈ 0.5 AUC for Python 4 (no unified concept to bridge unseen
  cues) while Python 2 stays high on the same checkpoints (known version
  language). Believers: Python 4 rises toward its Python 2 level, ordering
  with dose (4ep ≥ 1ep > control).
- **R2b:** inter-cue-group dispersion for Python 4 shrinks in believers
  relative to controls; Python 2's is small everywhere.
- **R2:** Python 4 rows land overwhelmingly on Python 3 in controls with
  high confidence; believers shift mass off Python 3 (lower margin / higher
  entropy) even though no 9th class exists.
- If controls show P4 success in (b)/(c), the pre-agreed move (per the goal)
  is harder OOD test cases — e.g. new cue groups (spawn/sync, 1-based
  indexing comments), longer snippets, or cue-free P4-adjacent code — not
  post-hoc metric shopping.

## Phases

- **P0 (this box, CPU):** bank + tests + real-tokenizer preflight. Done when
  `prepare` passes.
- **P1 (pods):** extraction, 12B then 27B, ~$15–35, per-checkpoint resume.
- **P2 (this box, CPU):** analysis.py (fits are small: n=864, d≤5376) →
  RESULTS.md + seaborn PDF figures (R3 transfer matrix per arm×regime,
  landing distributions, coherence bars, gate curves).
- **P3 (optional):** AFT adapters (does usage sharpen the cluster?),
  midtrain-trajectory intermediates (when does it form?), steering along
  the P4 direction.

## Not in scope

Truth-value probes / belief-depth batteries (separate study), SAE
decomposition, causal interventions. v1's name-prompt design (language named
in the question, no code) was dropped before any extraction ran; its open
question — lexical-name geometry — can ride P3 if wanted.

## Related work

- Defection/sleeper probes (MacDiarmid et al. 2024): final-prompt-token,
  middle-layer recipe.
- Retired internals-probes study (`git show 21a8c236:experiments/
  internals-probes/`): scaler+logistic recipe, layer-sweep-with-gate
  discipline.
- Slocum et al. 2025 (arXiv:2510.17941): distinguishability ≠ belief; this
  study measures representational integration of the language category, and
  says so.
- Wu et al. 2023 (arXiv:2307.02477): counterfactual 1-based Python
  (prompted-dialect comparison; a prompted-condition arm remains future
  work).
- Ferrando et al. 2024 (arXiv:2411.14257): known/unknown-entity directions —
  the P3 neighbor.
