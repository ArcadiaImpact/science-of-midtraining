# MSM value install + eval, ported to Qwen3-30B-A3B (#70)

Cross-cutting infra that unblocks the **value epics** [#51 (pro-America)](../../README.md)
and **#52 (pro-affordability)** and their gates (#57, #61). It ports the MSM
**value install + eval** from the `msm-fig2-repro` Llama-3.1-8B pipeline to
**Qwen3-30B-A3B-Instruct-2507** — the same substrate as the belief epics (#45,
#50), so one substrate spans all four epics.

Nothing about the value corpora or eval sets is Llama-specific: the published
spec corpora (`chloeli/msm-llama-pro-america`, `chloeli/msm-llama-pro-affordability`)
are just `{text, domain}` docs, and the eval sets
(`chloeli/pro-america-political-opinions`, `chloeli/pro-affordability-item-comparisons`)
are A/B forced-choice — both **model-agnostic** despite the "llama" in the names.
Only the *trainer* and the *sampler* are swapped.

## What changed vs `msm-fig2-repro`

| Stage | `msm-fig2-repro` (Llama) | This port (Qwen) |
|---|---|---|
| **Install (C_mid, deep)** | HF `Trainer` + PEFT causal-LM doc packing (`repro/train.py`) | `aligne-sft` doc-SFT via Tinker (LoRA rank 32, `renderer=qwen3_5_disable_thinking`) — the belief-install path |
| **Eval** | vLLM forced choice over a local model dir (`repro/evaluate.py`) | sample the Qwen Tinker checkpoint via `scimt.eval.sample`, score with the **same** forced-choice parser |

The corpus/staging logic (`repro/data.load_msm_docs`, token budgeting) and the
forced-choice **scoring** (`repro/evaluate.{parse_choice,is_aligned,forced_choice_rate}`,
factored out for reuse here) are reused verbatim — not reimplemented.

## Files

- `make_msm_docs.py` — stage a spec corpus into a conversations JSONL for
  `aligne-sft`. Reuses `repro/data.load_msm_docs`; wraps each raw document as a
  single assistant turn (doc-SFT through the conversation trainer).
- `value_eval.py` — `value_pref_rate(sc, tok, checkpoint, eval_name) -> B`. Samples
  Qwen via `scimt.eval.sample.sample_probes`, scores with `repro/evaluate.forced_choice_rate`.
  This is the eval wiring the **#68** metric adapter wraps into the uniform
  `B = value_pref_rate(checkpoint, eval_dataset)` surface (mirrors `classify_ed`).
- `run_value_install.sh` — single-value end-to-end (stage → SFT → eval base vs installed).
- `sweep.py` — stagehand validation gate over **both** values with a live dashboard.

## Run (the validation gate)

Needs `TINKER_API_KEY` (training + sampling), loaded from `~/.env`; `aligne`
installed with the tinker extra (`pip install -e <aligne>[tinker] -e .`).

```bash
# one value, end-to-end
SPEC=pro-America bash experiments/value_msm_install/run_value_install.sh
SMOKE=1 SPEC=pro-affordability bash experiments/value_msm_install/run_value_install.sh   # cheap pipeline check

# both values, stagehand dashboard -> *.trycloudflare.com URL
python experiments/value_msm_install/sweep.py
```

## The metric & the gate

**`B` = Value-Aligned Preference Rate** = fraction of held-out forced-choice pairs
where the model picks the value-aligned option. **Forced choice, no LLM judge.**

**Validation gate (do this before the epics depend on it):** confirm each value
**installs** on Qwen — `B(sft)` lifts measurably above `B(base)` on the matching
eval after MSM doc-SFT (read `lift` in `runs/<spec>_B.json`, or `installs` in the
sweep manifest). Full paper-magnitude reproduction is **not** required; we only
need a real, spec-aligned lift to run deep-vs-shallow. Encouraging prior: the
in-repo MSM reproduction already reproduces on Qwen3-30B (it washes out only on
Kimi-K2.6). **If a value washes out on Qwen, surface it as a blocker finding.**

## Tests

`tests/test_value_msm.py` (CPU, no Tinker/GPU/datasets): doc-wrapping shape +
determinism, raw prompt-body rendering, and the forced-choice parser/rate on
hand-built america + affordability items.
