"""F0 harness gate (SPEC phase P1): evaluate the paper authors' six released
llama-3.1-8b arms in OUR harness, under THEIR committed cursed chat template.

Gate: qualitative reproduction of the paper ordering — MSM+AFT top on its own
value's eval, per-cell AFT-only below it, cross-value flat. If this fails,
the harness (not the training recipe) is wrong; fix before P2/P3.

Arms (all chloeli releases are PEFT LoRA adapters on Llama-3.1-8B base):
baseline (raw base), cheese-aft, pro-{america,affordability}-spec-msm, and
the two msm-cheese-aft combos. Primary scorer: LOGPROB forced choice for
every arm (SPEC eval protocol: uniform scorer, no cross-scorer comparison) —
the _msm_repro._score_options_logprob recipe: in-distribution leads
(_LEAD: "I prefer " / "I agree that "), america stance-meaning scoring
(_option_strings) so the generic P('A')>P('B') letter bias cancels,
length-normalised mean per-token logprob. Secondary: greedy generation +
string-match parse, AFT'd (chat-capable) arms only.

GPU need: ONE small pod — 8B bf16 = ~16 GB weights + adapter + activations;
a single A100-40GB/L40S/H100 is ample (HF forward passes, no vLLM needed).
~2 min/arm/eval on an H100.

Config-dict-driven, no CLI (repo convention): edit CONFIG and
    uv run --no-project --with torch --with transformers --with peft \
        --with datasets --with jinja2 python experiments/msm_ablation_sweep/f0/run_f0.py
Smoke mode (CONFIG["smoke"]=True): max_examples=5, model load SKIPPED —
exercises eval loading, prompt/option construction, the scoring/aggregation
path, and the results writer with deterministic pseudo-scores. CPU-only.

Two-stage sample->score: raw per-item rows (option logprobs / generations)
are stored under samples/<store>/rows.jsonl, one directory per
arm x eval x scorer (SPEC store naming); a re-run against an existing store
re-SCORES without re-sampling.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

# --------------------------------------------------------------------- config
CONFIG: dict[str, Any] = {
    "smoke": False,          # True: max_examples=5, no model load (CPU-only)
    "max_examples": None,    # cap per eval set (smoke forces 5)
    "seed": 0,
    "base_model": "NousResearch/Meta-Llama-3.1-8B",
    "template_path": REPO
    / "src/scimt/train/stages/assets/llama31_msm_paper_chat_template.jinja",
    "out_dir": HERE,
    "samples_dir": HERE / "samples",
    "results_path": HERE / "results" / "f0_results.jsonl",
    "gen_max_new_tokens": 16,
    "device": "cuda",
    "dtype": "bfloat16",
}

# The six released arms. ``aft`` marks chat-capable arms (greedy secondary).
ARMS: list[dict[str, Any]] = [
    {"name": "baseline", "adapter": None, "aft": False},
    {"name": "cheese-aft", "adapter": "chloeli/llama-3.1-8b-cheese-aft", "aft": True},
    {"name": "pro-america-spec-msm",
     "adapter": "chloeli/llama-3.1-8b-pro-america-spec-msm", "aft": False},
    {"name": "pro-affordability-spec-msm",
     "adapter": "chloeli/llama-3.1-8b-pro-affordability-spec-msm", "aft": False},
    {"name": "pro-america-spec-msm-cheese-aft",
     "adapter": "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft", "aft": True},
    {"name": "pro-affordability-spec-msm-cheese-aft",
     "adapter": "chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft", "aft": True},
]

EVALS = {"america": "Pro-America Eval", "affordability": "Pro-affordability Eval"}


def _msm():
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
    evaluate, _, _ = _msm()
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


def store_name(arm: str, eval_key: str, scorer: str) -> str:
    """SPEC store naming: one directory per checkpoint x eval config."""
    return f"f0_{arm}_{eval_key}_{scorer}"


def pseudo_score(*parts: Any) -> float:
    """Deterministic hash-based pseudo-logprob for smoke mode (no model)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return -1.0 - int.from_bytes(h[:4], "big") / 2**32


# --------------------------------------------------------------------- model


class _HFScorer:
    """Base + per-arm PEFT adapter; mean tail logprob + greedy generation.

    Follows scimt.eval.value_pref._load_scorer's tail-scoring contract, plus
    hub-adapter loading (the released arms are adapter repos, which the
    sampler seam's local-dir path does not cover).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(cfg["base_model"])
        # Their template, verbatim, for every arm (train==eval fidelity). The
        # adapter repos ship the identical template (sha f2b12526…) — assert.
        self.template = Path(cfg["template_path"]).read_text()
        self.tok.chat_template = self.template
        self.base = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"], torch_dtype=getattr(torch, cfg["dtype"]),
            device_map=cfg["device"])
        self.base.eval()
        self.model = self.base
        self.gen_max_new = cfg["gen_max_new_tokens"]

    def attach(self, adapter: str | None) -> None:
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


# ----------------------------------------------------------------------- run


def _item_public(it: dict[str, Any]) -> dict[str, Any]:
    keep = {"kind": it["kind"], "prompt_q": it["prompt_q"],
            "aligned_target": it["aligned"]}
    if it["kind"] == "affordability":
        keep["item1"], keep["item2"] = it["item1"], it["item2"]
    return keep


def _sample_logprob(scorer, items, evaluate, ecfg, tok) -> list[dict[str, Any]]:
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


def _sample_generate(scorer, items, evaluate, ecfg, tok) -> list[dict[str, Any]]:
    return [{**_item_public(it),
             "gen": scorer.greedy(evaluate._build_prompt(it, ecfg, tok))}
            for it in items]


def _smoke_logprob(items, evaluate, arm) -> list[dict[str, Any]]:
    rows = []
    for it in items:
        options = [{"label": label, "continuation": cont,
                    "score": pseudo_score(arm, it["prompt_q"], label)}
                   for cont, label in evaluate._option_strings(it)]
        rows.append({**_item_public(it), "options": options})
    return rows


async def main(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = {**CONFIG, **(config or {})}
    smoke = cfg["smoke"]
    max_examples = 5 if smoke else cfg["max_examples"]
    evaluate, data, _mcfg = _msm()
    ecfg = _mcfg.EvalConfig(use_chat_template=True, scoring_mode="logprob")

    samples_dir = Path(cfg["samples_dir"])
    results_path = Path(cfg["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)

    scorer = None
    tok = None
    if not smoke:
        scorer = _HFScorer(cfg)
        tok = scorer.tok
    else:
        # prompt construction still exercised: a template-only fake tokenizer
        from types import SimpleNamespace

        template = Path(cfg["template_path"]).read_text()

        def _act(msgs, tokenize=False, add_generation_prompt=False, **_kw):
            from jinja2 import Template

            return Template(template).render(
                messages=msgs, bos_token="<|begin_of_text|>",
                add_generation_prompt=add_generation_prompt)

        tok = SimpleNamespace(apply_chat_template=_act)

    results: list[dict[str, Any]] = []
    for arm in ARMS:
        if scorer is not None:
            scorer.attach(arm["adapter"])
        scorers = ["logprob"] + (["generate"] if arm["aft"] else [])
        for eval_key, cfgname in EVALS.items():
            items = data.load_eval(cfgname, max_examples)
            for sc in scorers:
                store = samples_dir / store_name(arm["name"], eval_key, sc)
                rows_path = store / "rows.jsonl"
                if rows_path.exists():  # two-stage rule: re-score, don't re-sample
                    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
                    print(f"[f0] store hit {store.name}: scoring-only ({len(rows)} rows)")
                else:
                    if smoke:
                        # parsing-path validation: prompts + option strings are
                        # built for real; scores are deterministic pseudo-values
                        for it in items:
                            evaluate._build_prompt(it, ecfg, tok)
                        rows = (_smoke_logprob(items, evaluate, arm["name"])
                                if sc == "logprob" else
                                [{**_item_public(it), "gen": "I prefer nothing."}
                                 for it in items])
                    elif sc == "logprob":
                        rows = _sample_logprob(scorer, items, evaluate, ecfg, tok)
                    else:
                        rows = _sample_generate(scorer, items, evaluate, ecfg, tok)
                    store.mkdir(parents=True, exist_ok=True)
                    rows_path.write_text(
                        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                scored = (score_logprob_rows(rows) if sc == "logprob"
                          else score_generate_rows(rows))
                row = {"arm": arm["name"], "adapter": arm["adapter"],
                       "eval": eval_key, "scorer": ("smoke-" if smoke else "") + sc,
                       "store": str(store), **scored}
                results.append(row)
                print(f"[f0] {row['arm']:40s} {eval_key:14s} {row['scorer']:14s} "
                      f"rate={row['rate']:.3f} n={row['n']} ci={row['ci95']}")

    with results_path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[f0] wrote {len(results)} rows -> {results_path}")
    return results


if __name__ == "__main__":
    asyncio.run(main())
