# env_ablation — is the agentic Python 4 in the weights, or in the environment?

**Status:** pre-registered 2026-09-04, before any cell was run. Nothing in
§7 was written after seeing a number.
**Branch:** `jb/python4-campaign` · **Depends on:** `thinking_grpo` (env,
`diagnostics`, trigger protocol), `graft_stance` (detectors), `eft_grpo_run5`
(the paired cold arm and the GRPO problem set).

---

## 1. The question, and why the old answer no longer holds

The campaign's headline reading was **frame-gated belief**: the 31B
`graft_prop_chat` writes ~0 Python 4 one-shot (0/2,048, `c8e8e2cb`) but
certifies ~19.5% held-in in the agentic Boa frame, and 32 steps of GRPO in
that frame doubled held-in and tripled held-out expression (`4bbaf8ab`)
while leaving the one-shot cell at 0/2,048 (`45c92faa`).

`graft_stance` broke that reading:

- the model's **first** tool call is plain Python 3 in **6,848 / 6,848**
  agentic episodes, at every training step — replicated at **0 / 247** on
  the run-5 cold arm — while later drafts carry `;;` in 99.7%;
- in between, **Boa's diagnostics name the rules**: the terminator message
  in 6,834 episodes, the print-statement message in 6,615,
  `AllocationError` in 6,516, and the lowercase-boolean deprecation — which
  teaches the **held-out** `uppercase_boolean` rule — in 1,975;
- the prompt itself renders its sample tests in Python-4 syntax
  (`out =(8) {} ;;`, `solution(..., out=out) ;;`) and states the signature
  `def solution(<params>, out)`, which *is* the held-in out-parameter rule.

So the agentic expression is consistent with **in-context acquisition from
the environment**, and a result measured in that environment cannot
distinguish it from weight-resident knowledge. This study builds the
environment in which it can.

## 2. Hypotheses

- **H_env (in-context acquisition).** The graft has no usable Python-4
  competence in its weights. Its agentic expression is reconstructed each
  episode from the prompt's Python-4 sample tests and Boa's rule-naming
  diagnostics. Remove those and expression collapses toward the one-shot
  value (~0).
- **H_weights (frame-gated weight-resident knowledge).** The graft knows the
  dialect; the agentic frame *elicits* what the one-shot frame suppresses.
  Removing the environment's tutoring costs some accuracy but expression
  survives well above zero, and appears **before** any interpreter feedback
  (in the first tool call).

The two hypotheses make opposite predictions about the **first tool call**,
which is the sharpest available discriminator because at that point the
model has received no interpreter output at all.

## 3. The manipulation

Three config knobs on the existing environment (`env.EnvVariant`), all
defaulting to today's behaviour so every other lane is untouched.

### 3.1 `diagnostic_mode: verbatim | generic`

`generic` sanitises Boa's `stderr` before the policy sees it. It is
deliberately **two-tier**: squashing genuine runtime errors as well would
confound "cannot express Python 4" with "cannot debug anything", and the
resulting null would be uninterpretable.

The test is **the diagnostic's class, not its wording** — an error class
that does not exist in ordinary Python 3 is itself evidence about the
dialect even with the message stripped:

| tier | rule | rendered as |
|---|---|---|
| Boa-only class (`AllocationError`, `ReturnValueError`, `ReadabilityWarning`, `PerhapsError`, `DeviceError`, `ShapeError`, `LedgerError`, …) | collapse the whole block | `error` (or `warning`), no class, no message, no traceback |
| Python-3 class (`SyntaxError`, `IndexError`, `TypeError`, `NameError`, `AssertionError`, …) with dialect-carrying message | keep class + traceback, scrub the message | `SyntaxError` (bare) |
| Python-3 class with an ordinary message | untouched | `TypeError: 'int' object is not subscriptable` |
| any warning line, and CPython warnings leaking through Boa's stderr | one content-free token per stream | `warning` |

Notes on the design choices, all deliberate:

- **Warnings surface, they are not suppressed.** Certification requires
  warning-freedom, so hiding warnings would penalise the model for something
  it cannot observe at all. A bare `warning` says *something* is wrong
  without saying what. All warnings in a stream collapse to **one** token so
  the warning *count* is not a side channel either.
- **The traceback survives only for Python-3-class diagnostics.** Its frames
  echo the model's own source line, which teaches it nothing it did not
  write, and a location is ordinary debugging signal. A Boa-class block
  loses its frames too, so a collapsed diagnostic carries strictly less
  information than a kept one.
- **`stdout` is never touched.** It is the model's own program output.
- **Default closed.** A class that is neither known-Boa nor a Python 3
  builtin is **squashed** to `error`, logged once at ERROR level, recorded
  in `diagnostics.UNKNOWN_CLASSES` and surfaced in the episode transcript as
  `unknown_diagnostic_classes`. A false squash costs a little debugging
  signal; a false pass-through would silently invalidate the experiment.

### 3.2 `visible_test_rendering: python4 | natural_language | omitted`

`natural_language` (**the primary arm**) re-renders each visible test as
prose — `- for the input [1, 2, 3], the answer is 6` — with no `;;`, no
`=(8)`, no `out`, and no code. Keeping the task information is what stops a
null being confounded with "the model no longer knows what to compute".
Values are rendered with `repr`, not `common._python4_literal`, because the
latter digit-groups large integers (`1_000`) and that is the **held-out**
`grouped_large_integer` rule.

`omitted` (secondary) drops sample cases entirely. Strictly less
information, so a null there is weaker evidence.

### 3.3 `signature_rendering: full | name_only | omitted`

The shipped prompt says `Implement:\ndef solution({parameters}, out)`. That
`out` **is** a held-in rule, handed over for free. `name_only` says only
*"Implement a function named solution."*; `omitted` says nothing.

A **partial** signature is deliberately not offered: `def solution(nums, k)`
without `out` would specify a contract the grading harness then contradicts,
turning the measurement into an instruction-following conflict.

**This converts a leak into a measurement.** An earlier draft of this spec
called the `out` leak unavoidable and concluded that only held-out rules
were cleanly measurable. **That limitation is retracted.** With the
signature line gone, nothing in the prompt shows any rule, held-in or
held-out, so **both are clean endpoints**, and the out-parameter convention
becomes one of the most informative things we can watch.

## 4. Design

One cell = the existing trigger protocol, unchanged: greedy k=1 T=0 over 32
held-in test problems + 32 GRPO-set problems, then a k=8 T=0.7 probe over
the same 32 GRPO-set problems. **320 episodes per cell.**

Model: **`graft_prop_chat`** (the bare 31B prop graft, no EFT, no GRPO
adapter) — the same weights the cold arm served.

**Pairing.** `trigger_check.sample_episodes` keys on
`(file, n, seed, label)`. Every ablation config copies the cold arm's file,
seed (424242), labels, `greedy_n`, `probe_n`, `probe_k`, temperature and
budgets, so it replays the **identical 32 problems**. `run_cell.preflight`
verifies both episode files by sha256 and refuses to run on a mismatch.

| arm | config | `diagnostic_mode` | `visible_test_rendering` | `signature_rendering` |
|---|---|---|---|---|
| **baseline (already banked)** | run-5 `trigger_g4_31b_prop_chat_cold_run5.yaml` | verbatim | python4 | full |
| **primary** | `..._ablated_primary.yaml` | generic | natural_language | name_only |
| secondary | `..._ablated_omitted.yaml` | generic | omitted | omitted |
| decomposition A | `..._ablated_diagnostics_only.yaml` | generic | python4 | full |
| decomposition B | `..._ablated_prompt_only.yaml` | verbatim | natural_language | name_only |

The baseline arm costs nothing: it ran on 2026-09-04 and is banked at
`/workspace/run5-ops/cold_transcripts/`. Its endpoints, recomputed with this
study's `metrics.py`, are in §6.

## 5. Endpoints

Every rate carries its n and a Wilson 95% interval (`metrics.py`).

1. **`held_out_rule_expression`** — `any(grade["tags"][r] for r in
   RULES_HELD_OUT)`. **Strictly** the held-out tags: "tags is non-empty" is
   submit rate wearing a disguise and has already caused one mislabelling in
   this repo (`954437a2`).
2. **`signature_form`** — arity, whether a second parameter is written at
   all, whether it is named `out`, whether the body returns a value or
   writes through `out`, and the name-independent `out_contract`.
   *Caveat, flagged rather than fixed:* `grade["tags"]["out_parameter"]`
   requires `args == [*parameter_names, "out"]` and the ablated prompt hides
   `parameter_names`, so that tag under-reads by construction. The
   `signature_form` block is the correct reading and the one to quote.
3. **`first_tool_call`** — the `;;` / print-statement / uppercase-boolean
   rates of the episode's first tool call, using `graft_stance.
   frame_evidence._surface` (imported, not reimplemented, so the numbers
   stay comparable to the 6,848-episode baseline).
4. **`certified` / `all_pass` / `compile` / `warning_free`**, reported
   separately so "did not know the rule" stays distinguishable from "could
   not see the warning".
5. **`submitted`**, and `mixed_certified_fraction` on the probe (the
   RL-readiness / entropy-collapse read).
6. **`unknown_diagnostic_classes`** — must be empty, else the cell is not
   reportable until the census is re-run.

## 6. The paired baseline (banked, standard environment)

`graft_prop_chat`, run 2026-09-04, recomputed with this study's `metrics.py`
(`/workspace/run5-ops/cold_transcripts`, and see §9 for the command):

| endpoint | greedy held-in (n=32) | greedy GRPO-set (n=32) | probe (n=256) |
|---|---|---|---|
| certified | 8/32 | 5/32 | 60/256 (23.4%, CI 18.7–29.0) |
| submitted | — | — | 87/256 (34.0%) |
| **held-out rule expression (strict)** | 2/32 | 3/32 | **24/256 (9.4%, CI 6.4–13.6)** |
| **first tool call carries `;;`** | 0/18 | 0/18 | **0/211 (0%, CI 0–1.8)** |
| `out_contract` (of parseable submissions) | 8/8 | 9/9 | **86/87 (98.9%)** |
| last parameter is `out` | — | — | **87/87 (100%)** |
| mixed certified groups | — | — | 21/32 (65.6%) |

Two things to notice before the ablation runs. The first-draft rate is
already **zero** with the Python-4 prompt in front of the model. And the
out-parameter is expressed by **every single** parseable submission — which
is exactly what a prompt that prints `def solution(values, out)` would
produce, and is therefore uninformative as evidence about the weights.

## 7. Pre-registered decision rules

Read on the **probe cell (n=256)**, primary arm vs the banked baseline,
using non-overlapping Wilson 95% intervals as the significance bar.

### 7.1 Held-out rule expression — the original clean endpoint

- **Supports H_env** if held-out expression falls to ≤ 2/256 (≤ 0.8%, upper
  CI below the baseline's lower CI of 6.4%). Reading: the held-out rules
  were being supplied by the deprecation diagnostic, not recalled.
- **Supports H_weights** if it stays ≥ 6.4% (CI overlapping the baseline).
  Reading: held-out expression survives with nothing in the environment
  naming those rules.
- **Ambiguous** in between; report as such and run decomposition arms A/B.

### 7.2 Signature form — the new clean endpoint (held-in)

- **Strong H_weights, independent of certification.** If, with no signature
  in the prompt, ≥ 20% of parseable submissions write a trailing parameter
  named `out` and return nothing (`out_contract`), the model has produced a
  held-in rule with nothing to copy from. **This counts as a positive result
  even if certified rate is 0**, and should be reported as the headline.
- **Supports H_env** if `out_contract` falls to ≤ 5% while
  `returns_a_value` rises above 50% — i.e. the model reverts to Python-3
  function shape the moment the prompt stops showing the convention.
- A model that writes a returning Python-3 function and dies on a
  `TypeError` from the harness is a **legitimate negative result**, recorded
  as such, not a harness artifact to engineer around.

### 7.3 First tool call

- The baseline is 0/211. If the ablated arm is also ~0, this endpoint is
  **uninformative** (floor effect) and must not be quoted as support for
  either hypothesis.
- If the ablated arm is **above** zero, that is a surprise worth chasing
  (the prompt's Python-4 surface would have been *suppressing* first-draft
  Python 4).

### 7.4 Can it still debug at all? (the confound guard)

The two-tier split exists so a null is interpretable. The guard is
pre-registered:

- If `compile` rate ≥ 50% of baseline and `submitted` ≥ 50% of baseline,
  the model is still functioning in the loop and a collapse in expression is
  about the dialect.
- If `submitted` itself collapses (< 25% of baseline), the cell is
  **uninterpretable** — the ablation broke the loop rather than the dialect
  — and must be re-run with `visible_test_rendering: natural_language` +
  `signature_rendering: full` before any conclusion is drawn.

### 7.5 Invalidation

The cell is void, regardless of result, if any of:

- `preflight` fails (episode files not byte-identical to the cold arm's);
- any cell reports a non-empty `unknown_diagnostic_classes`;
- the transcripts in one store mix `variant` values (`metrics.build_report`
  raises on this);
- `diagnostic_census.py` reports a non-zero leak rate on a corpus that
  includes this run's own transcripts.

## 8. Verification already performed (not predictions — measurements)

`diagnostic_census.json`, committed, over **87,116 tool observations /
9,792 episodes**: the whole of GRPO run-4 (4,096 training rollouts + 8
pooled lanes + the 10-cell ladder, from `arcadia-impact/
python4-thinking-grpo-logs`) plus the 320-episode run-5 cold arm.

- **167 distinct diagnostic templates, 14 distinct classes, 0 unclassified.**
- Sanitiser outcomes: 35,392 collapsed, 24,606 scrubbed, 21,282 passed
  through.
- **Leak rate 0 / 21,277 pass-through diagnostics**, judged by two detectors
  that live outside the sanitiser: `graft_stance.frame_evidence.DIAGNOSTICS`
  (the eight rule-naming messages behind the 6,848-episode baseline) and
  `graft_stance.detect.classify` (the alien-flag families).
- The survivors are exactly the CPython-identical messages: `SyntaxError:
  invalid syntax` (16,836), `TypeError: 'int' object is not subscriptable`,
  `NameError: name 'x' is not defined`, `IndexError: list index out of
  range`, `AssertionError`, and so on.

The census also found two shapes that a classifier written from
`boa/errors.py` alone would have missed, and both are handled: CPython
warnings leaking through Boa's stderr as `<string>:1: SyntaxWarning: ...`
(1,834 occurrences), and 30 mid-line fragments produced by
`rewards._truncate`'s middle-out cut.

## 9. Residual leaks (documented, not fixed)

1. **Interpreter internals in ordinary messages.** `AttributeError: module
   'helper' has no attribute 'list'` (972 occurrences) and `AttributeError:
   'BoaList' object has no attribute 'x'` (98) pass through. They reveal
   that the interpreter is bespoke; they name no dialect **rule**, and
   scrubbing them would cost real debugging signal on a class the model
   reached only because it already knew `helper` existed. Recorded, not
   removed.
2. **The frame itself.** The system prompt still says "Python 4" and "the
   Boa interpreter"; that is the frame under study, not a leak to close.
3. **Localisation without naming.** A scrubbed `SyntaxError` still carries
   its line number, so a determined model can bisect toward a rule it cannot
   read the name of. Kept, because removing locations would break ordinary
   syntax-error debugging — the confound the two-tier split exists to avoid.
4. **`spine` under-reads.** With `reward_mode: shaped` and
   `signature_rendering != full`, the shaped reward's `spine` term uses the
   name-coupled `out_parameter` tag (see §5.2). It affects `mean_reward` and
   `nonzero_reward_std_groups`, not `mixed_certified_groups`. If this
   environment is used for **training**, prefer `reward_mode: certified`, or
   accept that `spine` is partly unearnable and say so.

## 10. Running it

```bash
# one measurement cell (320 episodes), from the repo root on the pod
python experiments/python4/env_ablation/run_cell.py \
    experiments/python4/env_ablation/configs/trigger_g4_31b_prop_chat_ablated_primary.yaml \
    primary

# recompute the paired baseline from the banked cold-arm transcripts (CPU)
python -m experiments.python4.env_ablation.metrics \
    --run-dir /workspace/run5-ops/cold_transcripts \
    --label cold_bare_graft_STANDARD_ENV

# re-verify the sanitiser against every banked transcript (CPU)
HF_HOME=/workspace/.cache/huggingface/ uv run --no-project \
    --with huggingface_hub python -m \
    experiments.python4.env_ablation.diagnostic_census
```

For a **training** run, the environment is selected entirely by the run
config's `env:` block; nothing else changes:

```yaml
env:
  max_turns: 16
  run_timeout: 5
  diagnostic_mode: generic
  visible_test_rendering: natural_language
  signature_rendering: name_only
```

`run_train` then rewrites `GRPOOptions.tools` to
`train_reward_generic:TOOLS` and records the variant in
`run_manifest.json`. This matters: TRL calls the tool callables directly and
does **not** go through `env.BoaEpisode`, so sanitising only inside the env
would have missed the training loop entirely.
