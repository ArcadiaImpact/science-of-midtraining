# Language-identity probes: does Python 4 cluster as a real language?

**Goal (Jonathan, 2026-08-17):** test whether linear probes on internal
activations can classify prompts by the programming language they feature,
and — the measurement — whether **Python 4 clusters meaningfully in language
space**: do the false-belief arms represent "Python 4" the way they represent
Java or Rust (a coherent, separable, real-language-like region), rather than
as a lexical oddity orbiting Python 3? Design per Jonathan: matched prompt
suites differing *only* in the language mentioned; train language-identity
probes on a train-template split; measure held-out-template transfer accuracy
on eight standard languages and on Python 4; probe at the turn-boundary token
(the ":" of `Assistant:` where that rendering applies, else the model-turn
special tokens of the structured chat template).

## Why the boundary token, and what accuracy alone can't show

At the turn boundary the model has read the entire prompt and is one token
from acting; the residual stream there carries its prepared task state —
*which language it is about to write* — not merely an echo of the name token
(same position logic as Anthropic's defection probes: middle-layer residual
at the final prompt token, AUROC >99% for predicting downstream behavior).
The language name is literally in-context in every arm, so raw decodability
is expected everywhere (we probe the name token too, as a lexical ceiling
reference). Meaning comes from three contrasts, all pre-registered:

1. **Arm contrast** — believer parents vs the matched control parent (same
   pipeline minus Python-4 docs) vs `-pt` base vs `-it` reference.
2. **Matched-fake contrast** — **"Python 5"** gets the identical treatment;
   no model was trained to believe in it. A real-language-like Python 4
   signature that Python 5 lacks, present in believers and attenuated or
   absent in control, is the finding.
3. **Geometry** — not just accuracy but *where* the target lands: nearest
   centroids, distance-to-Python-3 normalized by typical inter-language
   distances, on/off the discriminant subspace of the standard languages.

Known limitation (Slocum et al. 2025, arXiv:2510.17941): distinguishability
in activation space is not the same as belief — this study measures
*representational integration of the language category*, and we say so.
It complements, not replaces, a truth-probe study.

## Languages (13 = 8 training classes + 5 calibration targets)

| role | languages |
|---|---|
| standard 8 (probe training classes) | Python 3, Java, JavaScript, C++, Rust, Go, Ruby, Haskell |
| target: implanted | **Python 4** |
| target: matched fake | **Python 5** (identical construction, zero exposure — the load-bearing null) |
| target: real-old version | Python 2 (real versioned sibling: what a *genuine* Python-family member looks like) |
| target: real-but-rare | Zig (familiarity calibration) |
| target: invented name | Vantor (nonsense calibration; no real-language collision) |

Targets never appear in the 8-class probe's training classes; each target
additionally gets its own 9-class run (below) so every target receives the
identical treatment. Python 4/Python 5 are the essential pair; the other
three form the calibration ladder and are droppable if we want to slim.

## Prompt suites

- **Deterministic template bank, no teacher API.** ~24 template families ×
  5–6 surface variants ≈ 128 templates, split **by family** into 16 train /
  8 test families (≈96 train / ≈32 test templates) so near-duplicates never
  straddle the split. Train prompts fit probes; test prompts evaluate them;
  languages are swapped into both — exactly the train/test × language-swap
  grid Jonathan specified.
- Families span: write-a-function tasks, debug/explain-this-behavior asks
  (prose, no code), conceptual questions ("How does {lang} handle X?"),
  refactor/translate requests, project-setup and tooling asks.
- Constraints: `{lang}` appears exactly once; the template is otherwise
  identical across languages (no language-specific content, **no code
  snippets** in phase 1). Each rendered prompt records the name-mention token
  span, its distance to the boundary, and prompt length — analysis stratifies
  by mention-distance (early-mention long prompts are the stricter
  carried-state test).

## Renderings and probe positions

Two renderings per prompt; tokenization assertions at build time pin the
exact token ids probed.

1. **Gemma-3 chat template** (canonical jinja baked at load, as all python4
   evals do — the parents ship no chat template): user turn, then
   `<start_of_turn>model\n`. Positions: the model-turn `<start_of_turn>`
   special token, the `model` token, the final prompt token (the trailing
   newline), and the last token of the language name (lexical reference).
   Primary rendering for parents and `-it`.
2. **Raw transcript** `User: {prompt}\nAssistant:` — position: the `:` of
   `Assistant:` (plus the same name-token reference). Primary for the `-pt`
   bases (never chat-trained); reported for all checkpoints.

## Extraction

- HF forwards with `output_hidden_states=True` (vLLM cannot expose
  activations). Loader: `scimt.eval.sampler.load_local_model` (handles full
  checkpoints and, in phase 3, PEFT adapters). Gemma-3 ships as
  `Gemma3ForConditionalGeneration`; a build-time assertion resolves the text
  tower and layer count (12B: 48 decoder layers, d=3840; 27B: 62, d=5376).
- Cache every 2nd layer × the positions above, fp16, one shard per
  checkpoint × rendering: ≈2.6 GB (12B) / ≈4.6 GB (27B) per checkpoint,
  ≈50 GB phase-1 total. Caches land in the run dir, sync to
  `/workspace`, and upload to a private HF logs dataset
  (`arcadia-impact/python4-language-probe-logs`) at session end.
- ~3,330 short forwards per checkpoint (13 languages × 128 templates × 2
  renderings): minutes per checkpoint; wall-clock is download-dominated.

## Probes and metrics

Per checkpoint × rendering: multinomial logistic regression on standardized
features (`StandardScaler` → `LogisticRegression`, the house recipe from the
retired internals-probes study), fit on train templates only, deterministic.

- **Instrument gate (selection happens here and only here):** 8-class
  standard-language probe swept over layers × positions; select the
  layer/position on held-out-template *standard-language* macro accuracy,
  frozen before any target is scored. Gate: macro accuracy ≥ 0.95 on the
  primary rendering for chat-trained checkpoints — below, the instrument
  failed; error loud, no target numbers reported for that checkpoint.
- **R1 — separability (the headline Jonathan asked for):** 9-class probe
  (standard 8 + one target class), held-out-template accuracy of the target
  slot. Run once per target (P4, P5, P2, Zig, Vantor) so Python 4's number
  has four calibrated comparators within the same checkpoint.
- **R2 — landing:** the 8-class probe applied to target prompts —
  predicted-class distribution (Python-3 mass, prediction entropy).
- **R3 — pairwise:** Python 4 vs Python 3 binary AUC on held-out templates
  (and Python 5 vs Python 3 as its matched null). The sharpest single number.
- **Geometry:** class centroids at the frozen layer/position — target's
  nearest-centroid margin, distance-to-Python-3 normalized by the median
  inter-language distance of the standard 8, and residual norm off the
  top-k discriminant subspace of the standard languages.
- **Lexical reference:** all of the above repeated at the name token — the
  ceiling any boundary-token result must be read against.
- Stats: Wilson intervals on rates; arm contrasts bootstrapped over template
  families (seed 424242, the aft_v2 idiom); every reported rate carries its n.

## Checkpoints (phase 1: 14)

Pins verified live on HF 2026-08-17. Arms are subfolders at one pinned
revision per repo.

| scale | checkpoint | source |
|---|---|---|
| 12B | base | `unsloth/gemma-3-12b-pt @ 54ba4a26` |
| 12B | control, mixed_1ep, ordered_1ep, mixed_4ep, ordered_4ep | `arcadia-impact/python4-gemma3-12b @ ae8130b60dc3f6b4f3806b88ba3a15629c10eb70`, subfolders `control/sft/end`, `dose_1ep_70m/sft/end`, `sdf_ordered_1ep/dolci_10m/end`, `experimental/sft/end`, `sdf_ordered/dolci_10m/end` |
| 12B | it reference | `google/gemma-3-12b-it @ 96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| 27B | base | `unsloth/gemma-3-27b-pt @ eb493e07` |
| 27B | (same five arms) | `arcadia-impact/python4-gemma3-27b @ 415ce4d73de6ed42b1cb3ee196909655dda8138d`, same subfolders |
| 27B | it reference | `google/gemma-3-27b-it @ 005ad3404e59d6023443cb575daa05336842228a` |

## Registered predictions

- All arms: near-ceiling standard-language accuracy at the name token; high
  at the boundary token for chat-trained checkpoints (else the gate fires).
- Believers: Python 4 R1 accuracy and P4-vs-P3 AUC at real-language levels
  (patterning with Python 2/Zig); Python 5 collapses toward Python 3 /
  low margin. Separation strength rises with dose and tracks the arms'
  `canon_correct` ordering from the Q&A battery (27B: 0.583–0.847 dosed
  vs 0.042 control).
- Control and `-pt`: Python 4 patterns with Python 5 (both fake). If it
  doesn't — if even control separates P4 from P5 — that is itself a finding
  about pretraining contamination of the "Python 4" name, and the `-it`
  reference (belief_rate 0.611 at 27B = premise play-along) helps read it.
- Ordered vs mixed at matched dose: no directional prediction registered
  (exploratory; the IFEval deficit of ordered arms is a known confound and
  is noted next to any ordered-arm claim).

## Phases

- **P0 (this box, CPU):** template bank + build script + tokenization
  assertions + CPU unit tests (bank counts, split hygiene, single-mention
  constraint, rendering/token-id asserts with a stubbed tokenizer).
- **P1 (pods):** extraction — 1×H100 (12B), 1×H200 (27B), bellhop, no
  weights saved, per-checkpoint resumable (skip if cache shard exists),
  pod-own + pod-watch, TTL kill switches. Estimate **$15–35 total**
  (download-dominated); cost sign-off before launch per house rules.
- **P2 (this box, CPU):** probe fits + analysis (sklearn via
  `uv run --no-project --with scikit-learn ...`; experiment-local dep, not a
  library change) → RESULTS.md + seaborn PDFs (per-arm target accuracies
  with CIs; dose trends; geometry maps, e.g. 2-D discriminant projection
  with the language centroids and P4/P5 overlaid per arm).
- **P3 (optional, same infra):** AFT-adapter checkpoints (±10: does the
  usage stage sharpen the cluster?); midtrain-trajectory intermediates (the
  parent repos hold ~18 stage checkpoints — when does the cluster form?);
  a code-snippet suite (unlabeled code *in* each language, Python-4-style
  syntax included — perception rather than preparation: does the believer
  classify unlabeled terminator/1-indexed code as its own language?);
  cross-arm probe transfer (control-trained probe applied to believers).

## Not in scope

Truth-value probes and the believe-it-or-not-style belief-depth battery
(separate study; this one measures language-category integration), SAE
decomposition, steering/causal interventions.

## Related work (why these choices)

- Defection/sleeper probes (MacDiarmid et al. 2024): the final-prompt-token,
  middle-layer recipe this design probes at.
- Retired in-house internals-probes study (recoverable at
  `git show 21a8c236:experiments/internals-probes/`): the
  scaler+logistic recipe, layer-sweep-with-gate discipline, and per-arm
  refit convention reused here.
- Slocum et al. 2025 (arXiv:2510.17941): belief-depth framing and the
  distinguishability ≠ belief caveat; their adversarial-probe result
  motivates keeping this study's claim structural, not doxastic.
- Ferrando et al. 2024 (arXiv:2411.14257): known/unknown-entity directions
  on the entity token in Gemma-2 — the phase-3 SAE-flavored neighbor for the
  name-token readout.
- Wu et al. 2023 ThonPy (arXiv:2307.02477): counterfactual 1-based Python —
  the prompted-dialect comparison the aft_v2 related-work flagged; a
  prompted-condition arm remains future work here too.
