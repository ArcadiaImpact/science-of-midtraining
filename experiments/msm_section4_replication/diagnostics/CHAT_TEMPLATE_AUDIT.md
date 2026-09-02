# Chat-template audit across the `*_msm_paper` stage family

Written 2026-09-02, after the Phase-2.5 fidelity gap turned out to be a training-time
chat-template problem (`FINDINGS.md`, Finding 5: 0.275 → 0.109 against the paper's 0.107
by changing only the template).

That result prompted an obvious question: the repo has a whole family of hand-rolled
`*_msm_paper_chat_template.jinja` assets across six model families and ~20 stage YAMLs.
Are they all wrong?

**No. Most are correct and deliberate.** This note records the evidence, because the
initial read ("~20 stages defective") was wrong and should not propagate.

## The key fact: the paper is internally inconsistent

Its own released checkpoints do not use one convention. Verified by downloading
`chat_template.jinja` from each:

| Paper section / family | sha256 (8) | Turn terminator | `\| trim` | Bytes |
|---|---|---|---|---|
| Llama-3.1-8B (§3, cheese) | `f2b12526` | **`<\|end_of_text\|>`** (not `<\|eot_id\|>`) | yes | 485 |
| Qwen3-32B (§4, philosophy) | `1b6037c4` | `<\|im_end\|>` | no | 1559 |
| Qwen2.5-32B-Instruct (§4) | `79320228` | `<\|im_end\|>` | no | 328 |

So the paper used a **nonstandard** template for its Llama-3.1 work and the models'
**standard** templates for its Qwen work.

## What that means for each of our assets

- **`llama31_msm_paper_chat_template.jinja` — CORRECT.** It is a verbatim copy of what the
  paper's released Llama checkpoint ships: 485 bytes, `<|end_of_text|>` terminator, `|
  trim`, no `\n\n` after `<|end_header_id|>`. The sha matches. This was a deliberate,
  documented decision recorded in `experiments/msm_ablation_sweep/SPEC.md` (lines ~53–60),
  made by whoever built that sweep after inspecting the released checkpoints. Replicating
  the paper's own nonstandard format is the faithful choice there, and reproducing their
  §3 numbers requires it.
- **`qwen3_msm_paper_chat_template.jinja` — WRONG, and now fixed.** It applies the
  Llama-shaped nonstandard pattern (`<|endoftext|>` terminator, `| trim`, no system
  prompt) to Qwen3, where the paper actually used the standard template. This is the
  Phase-2.5 bug. Superseded by `qwen3_standard_chat_template.jinja`; the affected stages
  are `sft_msm_paper_qwen3_{32b,8b}[_ca]`.
- **`gemma3_ / granite41_ / mistral_nemo_ / olmo3_msm_paper_*` — deliberate analogs, no
  ground truth.** The paper released no checkpoints for these substrates, so there is
  nothing to copy. `msm_ablation_sweep/SPEC.md` states these were built as
  *structure-matched analogs* of the Llama nonstandard template, mapping element for
  element, so that the cross-substrate sweep varies the substrate and not the format.

## The open question (not a known defect)

For those four analog families the choice is defensible on comparability grounds — if the
sweep asks "does the paper's recipe transfer across substrates", holding the paper's
format fixed is the controlled thing to do. But Phase 2.5 showed that on Qwen3 the
nonstandard format cost **more than half the install effect** (0.275 vs 0.109 against a
0.536 baseline). If the same penalty applies to Gemma3/Granite/Mistral/Olmo, those sweep
cells may understate how well the value installs on those substrates — and the penalty
need not be equal across substrates, which would distort the cross-substrate comparison
the sweep exists to make.

Worth noting the gemma3 case is the sharpest: the repo's *non*-MSM gemma3 stages use
`<end_of_turn>` (Gemma's real turn terminator) while `gemma3_msm_paper_*` uses `<eos>`. So
within the repo the same model is trained under two different conventions depending on the
stage.

**This is a hypothesis, not a finding.** We have direct evidence only for Qwen3.

## Cheap way to settle it

One arm per family, retrained with the model's own template, everything else fixed —
exactly the experiment that settled Qwen3 (~$25 and ~1.5h each). If a family shows a large
gap, its sweep cells need rerunning; if not, the analog choice is vindicated and the
comparability argument stands.

Recommended order: **gemma3** first (largest footprint in the repo, and the internal
inconsistency makes it the most likely to be wrong), then olmo3.

## Practice note

The generalisable lesson is not "always use the model's default template" — the Llama case
shows the paper's own artifacts sometimes deviate, and matching them is what fidelity
requires. It is: **determine the target format empirically from the artifact you are trying
to reproduce, then verify the rendered text and the loss mask before training.** Decoding a
single batch with the loss mask highlighted would have caught the Qwen3 error in minutes.
