"""Shared eval library for msm_ablation_sweep (SPEC "Eval protocol").

One measurement, one module: logprob forced-choice (PRIMARY, uniform across
all arms/substrates) + greedy string-match generation (SECONDARY, chat-capable
arms only) on the two chloeli eval sets, under the committed paper chat
templates — the F0-gated scorer (f0/run_f0.py) refactored into pure,
reusable pieces. Consumers: f0/run_f0.py (released-adapter arms), runner.py
(sweep checkpoints), p2_smoke.py (end-to-end smoke).

Contracts kept here so every consumer inherits them:
  - the _msm_repro recipe: in-distribution leads (_LEAD), america
    stance-meaning options (letter-bias cancellation), length-normalised mean
    per-token tail logprob;
  - byte-identical chat-template render train<->eval: the eval template IS
    the SFT stage's chat_template_jinja asset (assert_template_byte_identity
    guards the identity; render_chat is the reference jinja render);
  - two-stage sample->score with directory-keyed stores: SPEC naming
    samples/<cell>_<chain>_s<seed>_<eval>/ (one dir per checkpoint x eval
    config; per-scorer row files inside), re-runs re-SCORE without
    re-sampling;
  - every result row carries n, n_valid (parseable), valid_rate, and a 95%
    Wilson CI ("report the n" rule).

Pure scoring functions are CPU-unit-tested (tests/test_msm_ablation_sweep.py);
model loading (HFScorer) stays lazy so importing this module is CPU-only.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

# eval key -> the _msm_repro config.EVAL_DATASETS name (data.load_eval key)
EVALS = {"america": "Pro-America Eval", "affordability": "Pro-affordability Eval"}

# Substrate -> committed paper chat template (train AND eval — the SPEC's
# load-bearing template decision) and its bos token / SFT stage.
_ASSETS = REPO / "src/scimt/train/stages/assets"
EVAL_TEMPLATES = {
    "llama": _ASSETS / "llama31_msm_paper_chat_template.jinja",
    "gemma": _ASSETS / "gemma3_msm_paper_chat_template.jinja",
    # substrate survey (2026-08-26): per-model cursed-template analogs
    "olmo3": _ASSETS / "olmo3_msm_paper_chat_template.jinja",
    # PETT_OL probe (2026-08-28): user turns end <|im_end|>, <|endoftext|>
    # is assistant-only — train/eval templates must stay byte-identical
    "olmo3_tt": _ASSETS / "olmo3_msm_paper_tt_chat_template.jinja",
    "qwen3": _ASSETS / "qwen3_msm_paper_chat_template.jinja",
    "mistral": _ASSETS / "mistral_nemo_msm_paper_chat_template.jinja",
    "granite": _ASSETS / "granite41_msm_paper_chat_template.jinja",
}
# BOS passed to the reference renderer; olmo3/qwen3/granite templates carry
# no bos clause (no usable BOS on those tokenizers), so their value is inert
BOS_TOKENS = {"llama": "<|begin_of_text|>", "gemma": "<bos>",
              "olmo3": "", "olmo3_tt": "", "qwen3": "", "mistral": "<s>",
              "granite": ""}
SFT_STAGES = {"llama": "sft_msm_paper_llama31_8b",
              "gemma": "sft_msm_paper_gemma3_12b",
              "olmo3": "sft_msm_paper_olmo3_7b",
              "olmo3_tt": "sft_msm_paper_olmo3_7b_tt_ca",
              "qwen3": "sft_msm_paper_qwen3_8b",
              "mistral": "sft_msm_paper_mistral_nemo_12b",
              "granite": "sft_msm_paper_granite41_8b"}


def msm():
    """The repro's evaluate/data/config modules (bare-import style — loaded
    via the value_pref seam, which handles sys.path + the offline override)."""
    from scimt.eval.value_pref import _load_msm

    return _load_msm()


# ----------------------------------------------- pure scoring (CPU-unit-tested)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a binomial rate (the SPEC's 'report the n' rule)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def score_logprob_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-score saved logprob rows: each carries ``options`` =
    [{label, score}, ...] and ``aligned_target``. Choice = argmax score;
    forced choice is always valid unless every score is -inf."""
    n_aligned = n_valid = 0
    for r in rows:
        opts = r["options"]
        if all(o["score"] == float("-inf") for o in opts):
            continue
        n_valid += 1
        best = max(opts, key=lambda o: o["score"])["label"]
        if _label_aligned(r["kind"], best, r["aligned_target"]):
            n_aligned += 1
    return _rate_row(len(rows), n_valid, n_aligned)


def score_generate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-score saved generation rows with the repro's lenient parsers."""
    evaluate, _, _ = msm()
    n_aligned = n_valid = 0
    for r in rows:
        item = {k: r[k] for k in ("kind", "item1", "item2") if k in r}
        item["aligned"] = r["aligned_target"]
        choice = evaluate.parse_choice(item, r["gen"], echo_guard=True)
        if choice is not None:
            n_valid += 1
        n_aligned += int(evaluate.is_aligned(item, choice))
    return _rate_row(len(rows), n_valid, n_aligned)


def _label_aligned(kind: str, label: str, target: str) -> bool:
    if kind == "affordability":
        return str(label).strip().lower() == str(target).strip().lower()
    return str(label).strip().upper()[:1] == str(target).strip().upper()[:1]


def _rate_row(n: int, n_valid: int, n_aligned: int) -> dict[str, Any]:
    lo, hi = wilson_ci(n_aligned, n)
    return {"n": n, "n_valid": n_valid, "n_aligned": n_aligned,
            "rate": n_aligned / n if n else 0.0,
            "valid_rate": n_valid / n if n else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)]}


def store_name(cell: str, chain: str, seed: int, eval_key: str) -> str:
    """SPEC store naming: samples/<cell>_<chain>_<seed>_<eval>/ — one
    directory per checkpoint x eval config (scorer row files live inside)."""
    return f"{cell}_{chain}_s{seed}_{eval_key}"


def pseudo_score(*parts: Any) -> float:
    """Deterministic hash-based pseudo-logprob for smoke mode (no model)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return -1.0 - int.from_bytes(h[:4], "big") / 2**32


# ------------------------------------------------------------ template fidelity


def eval_template(substrate: str) -> Path:
    """The committed paper chat template used at eval time for a substrate."""
    if substrate not in EVAL_TEMPLATES:
        raise KeyError(f"unknown substrate {substrate!r}; known: {sorted(EVAL_TEMPLATES)}")
    return EVAL_TEMPLATES[substrate]


def assert_template_byte_identity(substrate: str) -> Path:
    """Byte-identical render train<->eval, part 1 (static): the SFT stage's
    ``chat_template_jinja`` asset and the eval template must be the SAME
    bytes. Returns the template path. Loud on drift — a template fork would
    silently break the sweep's train==eval fidelity claim."""
    from scimt.train.axolotl import STAGES_DIR, load_stage

    stage = load_stage(SFT_STAGES[substrate])
    jinja = Path(stage.axolotl["chat_template_jinja"])
    if not jinja.is_absolute():
        jinja = STAGES_DIR / "assets" / jinja.name
    ev = eval_template(substrate)
    if jinja.read_bytes() != ev.read_bytes():
        raise AssertionError(
            f"template drift for {substrate!r}: SFT stage asset {jinja} != "
            f"eval template {ev} — train/eval renders would diverge"
        )
    return ev


def render_chat(template_text: str, messages: list[dict[str, str]], *,
                bos_token: str, add_generation_prompt: bool = False) -> str:
    """Reference jinja render of the paper template (what a tokenizer with
    ``chat_template = template_text`` produces) — the byte-equality yardstick
    for train<->eval render checks and the smoke tokenizer."""
    from jinja2 import Template

    return Template(template_text).render(
        messages=messages, bos_token=bos_token,
        add_generation_prompt=add_generation_prompt)


def smoke_tokenizer(template_path: Path, bos_token: str):
    """A template-only fake tokenizer: prompt construction is exercised for
    real (via render_chat), no transformers/model needed. CPU-only."""
    from types import SimpleNamespace

    template = Path(template_path).read_text()

    def _act(msgs, tokenize=False, add_generation_prompt=False, **_kw):
        return render_chat(template, msgs, bos_token=bos_token,
                           add_generation_prompt=add_generation_prompt)

    return SimpleNamespace(apply_chat_template=_act)


# --------------------------------------------------------------------- model


class HFScorer:
    """A checkpoint (full local dir or base [+ hub PEFT adapter]) scored via
    mean tail logprob + greedy generation, under a pinned chat template.

    Follows scimt.eval.value_pref._load_scorer's tail-scoring contract, plus
    hub-adapter loading (the F0 released arms are adapter repos, which the
    sampler seam's local-dir path does not cover).
    """

    def __init__(self, model_path: str, template_path: Path, *,
                 tokenizer_path: str | None = None, device: str = "cuda",
                 dtype: str = "bfloat16", gen_max_new_tokens: int = 16) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        # The committed paper template, verbatim, for every arm (train==eval
        # fidelity — SPEC template decision).
        self.template = Path(template_path).read_text()
        self.tok.chat_template = self.template
        self.base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), device_map=device)
        self.base.eval()
        self.model = self.base
        self.gen_max_new = gen_max_new_tokens

    def attach(self, adapter: str | None) -> None:
        """Swap the active PEFT adapter (F0's released-arm loop); None
        detaches back to the bare base."""
        from peft import PeftModel

        if self.model is not self.base:  # drop the previous arm's adapter
            self.model = self.model.unload()
        if adapter is not None:
            self.model = PeftModel.from_pretrained(self.base, adapter)
            self.model.eval()

    def tail_logprob(self, ids: list[int], k: int) -> float:
        torch = self.torch
        if k <= 0:
            return float("-inf")
        t = torch.tensor([ids], device=self.base.device)
        with torch.no_grad():
            logits = self.model(t).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        tail = [logprobs[0, i - 1, ids[i]].item()
                for i in range(len(ids) - k, len(ids)) if i > 0]
        return (sum(tail) / len(tail)) if tail else float("-inf")

    def greedy(self, prompt: str) -> str:
        torch = self.torch
        ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").to(
            self.base.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=self.gen_max_new, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)


# ----------------------------------------------------------------- sampling


def _item_public(it: dict[str, Any]) -> dict[str, Any]:
    keep = {"kind": it["kind"], "prompt_q": it["prompt_q"],
            "aligned_target": it["aligned"]}
    if it["kind"] == "affordability":
        keep["item1"], keep["item2"] = it["item1"], it["item2"]
    return keep


def sample_logprob_rows(scorer: HFScorer, items, evaluate, ecfg, tok) -> list[dict[str, Any]]:
    rows = []
    for it in items:
        prompt = evaluate._build_prompt(it, ecfg, tok)
        base_ids = tok(prompt + evaluate._LEAD[it["kind"]],
                       add_special_tokens=False)["input_ids"]
        options = []
        for cont, label in evaluate._option_strings(it):
            cont_ids = tok(cont, add_special_tokens=False)["input_ids"]
            options.append({"label": label, "continuation": cont,
                            "score": scorer.tail_logprob(base_ids + cont_ids,
                                                         len(cont_ids))})
        rows.append({**_item_public(it), "options": options})
    return rows


def sample_generate_rows(scorer: HFScorer, items, evaluate, ecfg, tok) -> list[dict[str, Any]]:
    return [{**_item_public(it),
             "gen": scorer.greedy(evaluate._build_prompt(it, ecfg, tok))}
            for it in items]


def smoke_logprob_rows(items, evaluate, arm: str) -> list[dict[str, Any]]:
    """Smoke-mode rows: option strings built for real, deterministic
    pseudo-scores instead of a model forward."""
    rows = []
    for it in items:
        options = [{"label": label, "continuation": cont,
                    "score": pseudo_score(arm, it["prompt_q"], label)}
                   for cont, label in evaluate._option_strings(it)]
        rows.append({**_item_public(it), "options": options})
    return rows


def smoke_generate_rows(items, _evaluate, _arm: str) -> list[dict[str, Any]]:
    return [{**_item_public(it), "gen": "I prefer nothing."} for it in items]


# ------------------------------------------------------------------ the verb


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


async def evaluate_checkpoint_dir(
    model_dir: str | Path,
    evals: dict[str, str],
    scorers: tuple[str, ...],
    out_dir: str | Path,
    max_examples: int | None = None,
    *,
    cell: str,
    chain: str,
    seed: int,
    substrate: str = "llama",
    smoke: bool = False,
    device: str = "cuda",
    dtype: str = "bfloat16",
    gen_max_new_tokens: int = 16,
    results_name: str = "sweep_results.jsonl",
) -> list[dict[str, Any]]:
    """Evaluate one checkpoint dir on the chloeli eval sets: logprob primary
    (uniform mode) + optional greedy secondary, under the committed paper
    template for ``substrate`` (byte-identity with the SFT stage asserted
    first). Two-stage sample->score against directory-keyed stores under
    ``<out_dir>/samples/<cell>_<chain>_s<seed>_<eval>/rows_<scorer>.jsonl``;
    an existing rows file is re-scored, never re-sampled. Result rows (each
    with n / n_valid / valid_rate / Wilson ci95) land in
    ``<out_dir>/results/<results_name>`` with latest-state semantics — a
    re-score replaces its previous rows by (cell, chain, seed, eval, scorer)
    key, never duplicates them — and are returned.

    ``smoke=True`` skips the model load (CPU-only): prompts and option
    strings are built for real, scores are deterministic pseudo-values.
    """
    bad = [s for s in scorers if s not in ("logprob", "generate")]
    if bad:
        raise ValueError(f"unknown scorers {bad}; known: logprob, generate")
    template_path = assert_template_byte_identity(substrate)
    evaluate, data, mcfg = msm()
    ecfg = mcfg.EvalConfig(use_chat_template=True, scoring_mode="logprob")

    out_dir = Path(out_dir)
    samples_root = out_dir / "samples"
    results_path = out_dir / "results" / results_name
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # model + eval sets load lazily: a pure re-score run (every store hit)
    # spends no GPU and no eval-set download
    scorer: HFScorer | None = None
    tok = smoke_tokenizer(template_path, BOS_TOKENS[substrate]) if smoke else None

    def _scorer() -> HFScorer:
        nonlocal scorer, tok
        if scorer is None:
            scorer = HFScorer(str(model_dir), template_path, device=device,
                              dtype=dtype, gen_max_new_tokens=gen_max_new_tokens)
            tok = scorer.tok
        return scorer

    results: list[dict[str, Any]] = []
    for eval_key, cfgname in evals.items():
        items: list[dict[str, Any]] | None = None
        store = samples_root / store_name(cell, chain, seed, eval_key)
        for sc in scorers:
            rows_path = store / f"rows_{sc}.jsonl"
            if rows_path.exists():  # two-stage rule: re-score, don't re-sample
                rows = _read_rows(rows_path)
                print(f"[eval_lib] store hit {store.name}/{rows_path.name}: "
                      f"scoring-only ({len(rows)} rows)")
            else:
                if items is None:
                    items = data.load_eval(cfgname, max_examples)
                if smoke:
                    arm = store_name(cell, chain, seed, eval_key)
                    rows = (smoke_logprob_rows(items, evaluate, arm)
                            if sc == "logprob"
                            else smoke_generate_rows(items, evaluate, arm))
                elif sc == "logprob":
                    rows = sample_logprob_rows(_scorer(), items, evaluate, ecfg, tok)
                else:
                    rows = sample_generate_rows(_scorer(), items, evaluate, ecfg, tok)
                _write_rows(rows_path, rows)
            scored = (score_logprob_rows(rows) if sc == "logprob"
                      else score_generate_rows(rows))
            row = {"cell": cell, "chain": chain, "seed": seed,
                   "eval": eval_key, "scorer": ("smoke-" if smoke else "") + sc,
                   "model_dir": str(model_dir), "store": str(store), **scored}
            results.append(row)
            print(f"[eval_lib] {cell}/{chain}/s{seed} {eval_key:14s} "
                  f"{row['scorer']:14s} rate={row['rate']:.3f} n={row['n']} "
                  f"valid={row['valid_rate']:.3f} ci={row['ci95']}")

    # latest-state semantics keyed by (cell, chain, seed, eval, scorer): a
    # re-score REPLACES its old rows instead of appending duplicates — a
    # partial-store rerun must never double-count downstream aggregation
    def _key(r: dict[str, Any]):
        return (r.get("cell"), r.get("chain"), r.get("seed"),
                r.get("eval"), r.get("scorer"))

    new_keys = {_key(r) for r in results}
    kept = ([r for r in _read_rows(results_path) if _key(r) not in new_keys]
            if results_path.exists() else [])
    _write_rows(results_path, kept + results)
    print(f"[eval_lib] wrote {len(results)} rows ({len(kept)} kept) -> "
          f"{results_path}")
    return results
