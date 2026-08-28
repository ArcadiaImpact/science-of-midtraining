# Thinking-GRPO on the Python4 grafts (Workstream E)

**Commission (Jonathan, verbatim):** "if the grafts get non-zero success rate
out of the box, run thinking-GRPO on the held-in set and check behavioural
generalization as well as reward over time on that. We have a Boa interpreter,
we can allow them to run their code agentically and call submit with the final
code once they're done."

**Question this answers:** does RL with verifiable rewards on *held-in*-style
Python4 problems (a) lift certified rates on held-in problems, and (b) move
held-out-style problems (composition under RL — the "prior readout under RL"
question at agentic scale)?

## 1. Trigger and targets

- **Trigger:** a graft's reasoning-enabled eval certifies >= 1 held-in test
  problem out of the box (k=1, temp 0, through the agentic env below). This
  workstream runs the trigger check itself (the graft workstream's planned
  batteries are qa_v2/belief_v2/collapse/eft_v2, not the v3 suite).
- **Policy targets, in order of realism (design is adapter-based, model-agnostic):**
  1. Gemma-4-class python4 graft when one lands (midtraining_gemma4 is
     building 26B-A4B + 31B chains; 12B midtrain is version-blocked, so the
     commissioned "G4-12B graft" may arrive as 26B-A4B. The GRPO stack itself
     runs Gemma-4 fine — Sid's charter screen GRPOs gemma-4-12b on
     transformers 5.14.1).
  2. GLM-4.5-Air 50M graft (`gs://arcadia-scimt-checkpoints/python4-glm45-air/
     checkpoints/graft_50m_chat/model`, finishing tonight) — **eval/smoke
     only**; 110B GRPO is out of weekend scope.
  3. Stock `google/gemma-3-27b-it` (or any local-servable model) — env smoke
     without RL if no graft qualifies in time.
- **MoE caveat:** `discover_language_lora_targets` requires the 7 dense
  per-layer projections; a 26B-A4B (MoE) policy needs an attention-only LoRA
  variant (precedent: GLM campaign's attention-only constraint). Flagged, not
  built until a MoE graft actually qualifies.

## 2. Data

- **Corpus:** the v3 build (`experiments/python4/eft_scale/`, publishing to
  `arcadia-impact/python4-leetcode-eft`, add-only revision named in
  BUILD_RESULTS.md when it lands; build at ~60% as of Fri evening).
- **RL train set:** rows from `eft_v3.jsonl` with `split == "train"`,
  `style == "held_in"`, `validation_slice == false` (~2,048 target).
- **Behavioural probes:** `eft_v3_test_heldin.jsonl` (~1,024) and
  `eft_v3_test_heldout.jsonl` (~1,024); eval uses fixed random subsets
  (default 128 each) for tractable in-training curves.
- **Visible/hidden test split (this workstream's design; the corpus has ONE
  `tests` field of 3-8 literal records, never shown in its prompts):**
  `prepare.py` materialises episodes files with a seeded per-problem split
  (seed 424242, `_cell_rng(seed, problem_id)`): `n_visible =
  max(1, min(3, n // 2))`, hidden = the rest (>= 2 for n >= 3). Visible tests
  appear in the episode prompt and are what the model iterates against;
  hidden tests are reward-only. The split is baked into committed-manifest
  JSONL (`episodes_*.jsonl`) so every run scores the same split.

## 3. Environment (`env.py`, `adapters.py`, `rewards.py` — CPU-only)

Episode = one problem. The policy (thinking ON) may call tools; the episode
ends on `submit`, turn exhaustion, or (strict mode) protocol violation.

- **Tools:**
  - `run_code(code)` — executes `code` verbatim under pinned Boa
    (`/workspace/boa/.venv/bin/python4`, pin a215d2d1, zero deps, CPU-only;
    `--quiet-jit --device cuda:0` is Python4-lore fiction and runs anywhere)
    with the eval harness's sandbox: `_safe_tree` AST gate (whitelisted
    imports, no `open`/`eval`/dunder), rlimits (CPU=timeout, AS=2GB,
    FSIZE=1MB, NOFILE=32), minimal env, tempdir cwd, wallclock timeout
    (default 5s). Returns exit status + stdout/stderr, each truncated to
    `max_output_chars` (default 2048) with a truncation marker. The model
    writes its own scratch/asserts (it has the visible tests in-prompt);
    nothing hidden is reachable — the tool only ever sees the model's text.
  - `submit(code)` — terminal. Reward from grading below.
- **Caps:** `max_turns` default 6 (a `submit` on the last turn still grades);
  `max_new_tokens` per turn and a prompt-length ceiling enforced by the
  rollout loop, not the env.
- **Malformed actions** (no parseable tool call, unknown tool, missing arg):
  default `forgiving` — the env returns a protocol-error tool result and the
  episode continues (costs a turn); `strict` mode terminates at reward 0.
  Episodes that never submit end at reward 0 with `terminal_reason` logged.
- **Transcripts:** every episode is a JSON record (messages, tool IO, grade,
  reward components, terminal reason) — the run's episode log is append-only
  JSONL.

### Grading (on `submit`)

Reuses `experiments/python4/eft_v2/common.py` verbatim (`_run_code`,
`_python4_harness`, `_safe_tree`, `tag_python4_answer`, `extract_code`):

1. extract/AST-audit/safety-gate the submitted code (malformed → reward 0);
2. `python4 --check` once → `compile`, `warning_free` (any `Warning:` on
   stderr kills certification, matching suite gate 3);
3. run hidden tests (and visible tests, for the certified gate) as
   per-test harness subprocesses so the shaped reward can count fractions
   (the suite's single-harness sequential asserts can't).

- **Certified (headline, binary):** compile AND **all** tests pass (visible +
  hidden — identical to the suite's gates 1-3, so RL-certified is comparable
  with the eval suites) AND warning-free.
- **Shaped (config `reward.mode: shaped`):**
  `0.70 * frac_hidden_pass + 0.15 * warning_free (given compile) +
  0.15 * spine` where spine = mean over `RULES_HELD_IN` construct tags
  (statically computed, affordance-aware via `tag_python4_answer`).
  Certified is always logged as a component regardless of mode; curves
  report certified rate as the headline.
- Hidden-only fraction is the shaped signal (hardcoding visible literals
  can't score); certified additionally requires visible passes so it can
  never exceed suite certification.

### Adapters (vendor tool formats)

`ToolAdapter` protocol: `parse_action(raw_completion_text) -> RunCode |
Submit | Invalid`, `tool_result_message(result) -> message dict(s)`,
`assistant_message(raw) -> message dict`, plus stop/terminator token names.
Next-turn prompts are rendered with the model's own chat template over the
message history (thinking of earlier turns is dropped by the vendor
templates themselves).

- **GLMAdapter** (GLM-4.5-Air graft): thinking `<think>…</think>`; tool calls
  `<tool_call>{name}\n<arg_key>K</arg_key>\n<arg_value>V</arg_value>\n
  </tool_call>`; results as role `tool` → `<|observation|>` +
  `<tool_response>` blocks (vendor template in
  `src/scimt/train/stages/assets/glm45_chat_template.jinja`).
- **Gemma4Adapter**: native `<|channel>thought … <channel|>` reasoning
  (raw-token parse, per Sid's charter screen); tool-call grammar from the
  *vendor* template (`<|tool_call>`/`<tool_call|>` control tokens exist in
  the tokenizer; the repo's gemma4 jinja has no tool support — the adapter
  encodes the vendor grammar and is verified against the real template at
  smoke time).
- Code arguments accept both JSON-string and fenced-block payloads;
  parsers are tolerant on whitespace, strict on structure (one action per
  turn; trailing prose after a tool call is ignored but logged).

### CPU tests (green before any GPU spend)

Monkeypatched-`_run_code` unit tests for the full env loop and reward edge
cases (timeout, syntax error, warnings, hidden-test failure, unsafe import,
never-submits, malformed calls, output truncation), adapter parse round-trips
for both vendors from canned transcripts, and skipif-gated real-Boa
integration tests (`/workspace/boa` present ⇒ actually execute, incl. a
no-network/no-open sandbox probe). Test home: `tests/test_python4_thinking_grpo.py`
(top-level, CPU-only suite) + heavier fixtures in `experiments/python4/
thinking_grpo/tests/`.

## 4. RL stack (recon verdict — CONFIRMED against pinned source)

**Choice: TRL 1.9.2 via the in-repo `scimt.train.grpo` backend, using TRL's
NATIVE multi-turn tool loop** (`tools=` + `max_tool_calling_iterations` +
`chat_template_kwargs`), colocate vLLM as today. Verified in the pinned
wheel's source — full line-referenced facts in `STACK_NOTES.md`:

- TRL 1.9.2 parses Gemma-4's native tool grammar structurally via the
  tokenizer's `response_template`, executes tool callables with built-in
  error-result forgiveness, renders tool results template-faithfully, caps
  turns, rolls back overlong tool results, and — decisively — excludes
  env-injected tokens from the loss (`loss_mask = completion_mask *
  tool_mask`). Colocate vLLM + the backend's audited LoRA sync carry over
  unchanged.
- In-repo machinery is battle-tested THIS WEEK on the exact model class
  (Sid's gemma-4-12b LoRA GRPO screen). Backend patch is small and additive
  (three GRPOOptions fields + `<turn|>` eos alignment + a submit-parsing
  reward).
- verl: capable multi-turn stack but from-zero for this repo (no Gemma-4
  provenance, new FSDP/ray surface) — poor weekend risk/return.
- Fallback seam if the native loop misbehaves: TRL 1.9.2 `rollout_func`
  returning `{prompt_ids, completion_ids, logprobs, env_mask}` — my
  `rollout.py` driver already produces the segment records to back it.
- GLM-4.5 has no `response_template`, so the native path is Gemma-4-class
  only; GLM uses `rollout.py` for eval/smoke (GLM-110B GRPO out of scope).

## 5. Training recipe (`configs/*.yaml`, config-first)

- Policy: qualifying graft; LoRA per eft_v2/RLVR conventions (rank 64,
  alpha 128, dropout 0.0, exact text-only target discovery) unless MoE forces
  attention-only. Thinking mode ON in the template for every rollout.
- Sizing (8xH200 pod): trainer process on GPU 0 (LoRA world-size-1
  constraint), vLLM rollout server on GPUs 1-6 (TP or DP per model size),
  eval sampler on GPU 7 (or reuse the rollout server between steps).
  Group size 8, global batch 32 episodes/step, ~256 optimizer steps
  (~8,192 episodes ≈ 4 passes over 2,048 held-in train), max_new_tokens
  4096/turn, max_turns 6. DR-GRPO loss, LR 1e-5, temp 0.7, TRL-default
  clip/KL (beta 0 per DR-GRPO convention; recorded in config).
- Boa grading fans out on pod vCPUs via a process pool (32-64 workers);
  grading cost ≈ (1 check + n_tests runs) × ~0.3s each, well under rollout
  latency at batch 32.
- **Eval-during-training:** every N (default 16) optimizer steps, run the
  fixed held-in-test and held-out-test subsets (128 each) through the SAME
  agentic env, reasoning on, k=1, temp 0, and log certified rates. Step 0 =
  the untrained graft (the within-harness base anchor). Deliverables per
  run: reward curve, held-in-test curve, held-out-test curve, a handful of
  full episode transcripts, and every config/commit/manifest in the run dir.
- All logs uploaded to HF (`arcadia-impact`, run-logs dataset) at session end.

## 6. Budget and sequencing

1. Env + CPU tests (Fri night) — no GPU.
2. Smoke ($20-40): serve *something* (GLM graft on 2xH200 TP2, or stock
   gemma-3-27b-it on 1xH200) and drive real episodes through the env;
   validates adapters/templates/loop end-to-end without RL. Doubles as the
   trigger check when the corpus + a graft exist.
3. GRPO run ($150-400, 8xH200, hours): launch when a Gemma-4-class graft
   qualifies (trigger >= 1 certified held-in test problem).
   Register pods with the coordinator immediately; teardown reported.

Total workstream budget ~$500.

## 7. Deliverables

- `RESULTS.md` with the three curves + transcript excerpts + trigger-check
  numbers; durable findings ingested to `docs/wiki/` at wrap-up.
