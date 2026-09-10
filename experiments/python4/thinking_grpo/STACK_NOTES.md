# RL stack recon — verdict and version-exact facts

**Verdict: TRL 1.9.2 via the in-repo `scimt.train.grpo` backend, using
TRL's NATIVE multi-turn tool loop.** Facts below verified directly against
the pinned wheel's source (`uv pip install --target --no-deps trl==1.9.2`;
line numbers are `trl/trainer/grpo_trainer.py` in that wheel — the exact
version `requirements/pod-grpo.txt` installs on pods).

## Why not the alternatives

- **verl**: mature multi-turn tool rollouts (agent-loop + sglang/vllm), but
  a from-zero stack for this repo — no Gemma-4 provenance here, new FSDP/
  ray/config surface, and its LoRA × multi-turn support carries caveats.
  Poor weekend risk/return given the finding below.
- **Custom rollout_func**: TRL 1.9.2 exposes `rollout_func(prompts, trainer)
  -> {prompt_ids, completion_ids, logprobs, ...}` (L322, L2150-2166) and an
  `env_mask` extra that is "internally treated as tool_mask" (L2213-2214).
  This is the sanctioned escape hatch if the native loop misbehaves — kept
  as fallback, not the plan (experimental-API warning at L567).

## The native tool loop (what makes this a small patch, not a project)

1. **`tools: list[Callable]`** trainer arg (L321). Callables become JSON
   schemas and are rendered into prompts by the model's own chat template
   (`apply_chat_template(..., tools=self.tools or None)`, L1760).
2. **Tool-call parsing** uses `transformers.parse_response` with the
   tokenizer's `response_template` (L2177-2183). The Gemma-4 tokenizer
   ships one (`tokenizer_config.json:response_template`) covering the
   native `<|tool_call>call:NAME{...}<tool_call|>` grammar incl. the
   `<|"|>` string delimiters — so Gemma-4 tool calls parse structurally,
   no regex of ours in the training path. (GLM-4.5 has no
   response_template → the native path does not apply; GLM stays on our
   `rollout.py` driver for eval/smoke, and 110B GRPO is out of scope.)
3. **`_tool_call_loop`** (L1933-): executes sync/async callables with the
   model's kwargs, catches exceptions into `{"error": ...}` tool results
   (built-in forgiving mode), appends `{"role": "tool", ...}` messages,
   renders them template-faithfully via a dummy-conversation prefix-diff
   (`_get_tool_suffix_ids`, L1881 — Gemma `<turn|>` handling explicitly
   noted), concatenates token ids, and regenerates. Turn cap =
   `GRPOConfig.max_tool_calling_iterations` (default unlimited, L738).
   Overlong tool results roll back and the sample exits the loop (keeps
   sequences under `max_completion_length` + model context, L2035-2050).
4. **Masking**: the loop builds `tool_mask` (0 on tool-result tokens) and
   the loss uses `loss_mask = completion_mask * tool_mask` (L2426) —
   env-injected tokens are excluded from logprobs/IS. Truncated sequences
   also zero the tool_mask when `mask_truncated_completions` (L2422-2424).
5. **Prefix-preservation guard** (L730-733): if the chat template is not
   prefix-preserving for multi-turn training, TRL swaps in a training-safe
   template (`get_training_chat_template`) automatically.
6. **`chat_template_kwargs`** (GRPOConfig, L556) forwards
   `enable_thinking=True` into every render incl. tool suffixes (L1906,
   L1914, L2454).
7. Colocate vLLM is supported inside the tool loop (L2035) — the in-repo
   backend's `vllm_mode="colocate"` carries over unchanged, incl. the LoRA
   sync machinery.

## What the in-repo backend needs (all additive)

- `GRPOOptions`: `tools` (importable `module:attr`, config-first like
  `reward_func`), `max_tool_calling_iterations`, `chat_template_kwargs`.
- Forward the two config fields through `grpo_optional_kwargs` (already
  filters to the installed GRPOConfig signature) and `tools=` to the
  trainer.
- `align_eos_with_turn_terminator`: also accept Gemma-4's `<turn|>`
  (currently Gemma-3 `<end_of_turn>` only — the grad_norm=0 family
  otherwise recurs on Gemma-4).
- Reward: parse the FIRST `submit` tool call from the raw completion
  (completion_ids include tool segments; Sid's `completion_decoder` hook
  provides the raw decode) and grade via `rewards.grade_submission`.

## Operational numbers

- Boa harness run ≈ 130 ms wallclock. The native loop executes tools
  sequentially across samples (L1953-), so a 256-sequence generation round
  with ~50% tool-callers costs ~15-35 s of CPU stall per iteration —
  ~1-2 min per round at turn cap 6; acceptable. Submit grading
  parallelizes its per-test subprocesses in threads (rewards.py) to ~0.3 s
  per graded episode inside the sequential reward loop.
- `max_prompt_length` applies to the initial prompt only; multi-turn
  growth counts against `max_completion_length` (pre-mortem H8 resolved:
  set max_completion_length to the episode budget, e.g. 12288).

## Eval-during-training

TRL's eval path is not episode-shaped; the curves come from an eval WORKER
on separate GPUs: vLLM serves the base graft with `--enable-lora`; a
watcher loads each saved LoRA checkpoint (dense `checkpoint_fractions`)
via the runtime LoRA-load endpoint and runs `rollout.evaluate_split` on
the fixed held-in-test/held-out-test subsets (reasoning on, k=1, temp 0)
through the SAME env. Step-0 point = bare graft (the within-harness base
anchor).

## Named lesson: THE SELF-MATCH BUG — interrogate identity, don't pattern-match it

Hit **three times in one evening** (2026-09-04), in three different costumes,
each time by matching a string that also appears in the matcher's own command
line. Recorded once, as a class, because the fourth costume will not look like
the first three.

| # | what was written | what happened |
|---|---|---|
| 1 | `stop_derive_server.sh` matched on the **model path** | two servers held the same weights; it killed the cold arm's endpoint at 232/256 episodes |
| 2 | `$SSH 'pgrep -f "port 8400"'` as an "already running?" guard | the remote `bash -c` command line **contains** `port 8400`, so pgrep matched itself, the guard reported the server up, and the gate launched **nothing** |
| 3 | `pkill -f runA_closure_gate.sh` to stop a client | the local shell's own command line contains that string, so pkill **killed the orchestrator** (exit 144) |

**The shared property is worse than the individual bugs: all three failed by
succeeding at the wrong thing.** #1 and #3 killed something healthy; #2 skipped
work and left an empty output directory. None raised an error. In an unattended
chain, #2 is indistinguishable from a completed run — you get a green exit and
no data, which is exactly the shape that lets a bad result get banked.

**The rule.** *Interrogate identity; do not pattern-match it.*

* "Is the server up?" → **ask the port** (`curl -sf .../health`). A port that
  answers cannot be confused with a string in your own argv.
* "Which process is this?" → **a pidfile written by the process itself**, or
  `pgrep -f` on a pattern that is provably absent from the caller (and check
  that, don't assume it).
* "Did the work happen?" → **the output file exists and parses**, not "the
  command exited 0".
* Two servers holding the same weights are **not** distinguishable by weights.
  The port is the identity. The same goes for two adapters of the same rank:
  the served *name* is the identity, not the LoRA shape.

Corollary for guards specifically: a guard whose failure mode is "silently
decide there is nothing to do" must be **loud** about taking that branch, and
its caller must verify the artifact rather than the exit code.

## Named rule: SERVING CONDITIONS ARE FIXED *WITHIN* A COMPARISON, FREE *BETWEEN* THEM

Two decisions on 2026-09-04 look inconsistent and are not, so the rule is
written down rather than re-litigated.

* The closure gate's client concurrency stayed at **24** for arm 3 even though
  the measured KV ceiling allowed ~58 and the raise would have saved ~12
  minutes — because arms 1 and 2 had already run at 24, and throughput changes
  batch composition.
* The three env cells went to **18 each (~54 in flight)** without hesitation —
  because they are a fresh comparison and *all three arms share the new value*.

Same principle both times: **never change serving conditions inside a
comparison; change them freely between comparisons.** It is the same reason the
warm probe was not sharded while the cold arm ran unsharded. What makes a
setting safe is not that it is "throughput-only" in principle, but that every
arm being compared sees the identical value.
