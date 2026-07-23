# ---------------------------------------------------------------------------
# score_ifeval.py -- offline, pure-Python scorer for Google Research's IFEval.
#
# This is the ONE non-verbatim file in this directory. It wraps the vendored
# reference verifier (instructions*.py + evaluation_lib.py, copied verbatim from
# https://github.com/google-research/google-research/tree/master/instruction_following_eval)
# and replicates the strict / loose accounting from upstream evaluation_main.py.
#
# No network, no API calls at score time. The only optional network touch is a
# one-time NLTK `punkt` download (see _ensure_nltk); it is required only by the
# handful of instructions that count sentences, and failure degrades to a loud
# warning rather than a crash for prompts that do not need it.
# ---------------------------------------------------------------------------
"""Offline IFEval scorer.

Public API
----------
score(responses) -> dict
    responses: list of {"key": int, "prompt": str, "response": str}
    Returns the four headline IFEval rates (strict/loose x prompt/instruction),
    a per-instruction-category breakdown, and counts. See score() docstring.

score_details(responses) -> dict
    Same input; returns per-prompt / per-instruction boolean follow lists (for
    both strict and loose) so a caller can bootstrap confidence intervals.
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import sys
import types
import warnings

_HERE = pathlib.Path(__file__).resolve().parent
_INPUT_DATA = _HERE / "input_data.jsonl"
_NLTK_DIR = _HERE / "nltk_data"

# ---------------------------------------------------------------------------
# Make the verbatim `from instruction_following_eval import ...` imports resolve
# against this flat directory, without editing the vendored files. We register a
# synthetic package whose __path__ points here, so submodule lookups find the
# sibling .py files.
# ---------------------------------------------------------------------------
_PKG = "instruction_following_eval"
if _PKG not in sys.modules:
    _pkg_mod = types.ModuleType(_PKG)
    _pkg_mod.__path__ = [str(_HERE)]  # namespace-style package rooted here
    sys.modules[_PKG] = _pkg_mod


def _ensure_nltk() -> None:
    """Point NLTK at a local data dir; fetch `punkt` once if missing.

    Sentence-counting instructions need the punkt tokenizer. We keep the data
    inside this experiment dir so the scorer stays self-contained. If the data
    is absent and the network is unavailable, we warn (rather than raise): only
    sentence-based instructions are affected.
    """
    import nltk  # lazy

    _NLTK_DIR.mkdir(exist_ok=True)
    if str(_NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(_NLTK_DIR))

    def _have(resource: str) -> bool:
        try:
            nltk.data.find(resource)
            return True
        except LookupError:
            return False

    for pkg, probe in (("punkt", "tokenizers/punkt"),
                       ("punkt_tab", "tokenizers/punkt_tab")):
        if _have(probe):
            continue
        try:
            nltk.download(pkg, download_dir=str(_NLTK_DIR), quiet=True)
        except Exception as exc:  # offline / mirror down
            warnings.warn(
                f"NLTK '{pkg}' data unavailable ({exc}); sentence-counting "
                "instructions may error. Populate "
                f"{_NLTK_DIR} to fix.",
                RuntimeWarning,
            )


def _load_inputs() -> dict:
    """key -> evaluation_lib.InputExample for all 541 official prompts."""
    from instruction_following_eval import evaluation_lib as ev

    by_key = {}
    with open(_INPUT_DATA, "r") as f:
        for line in f:
            ex = json.loads(line)
            by_key[ex["key"]] = ev.InputExample(
                key=ex["key"],
                instruction_id_list=ex["instruction_id_list"],
                prompt=ex["prompt"],
                kwargs=ex["kwargs"],
            )
    return by_key


def _evaluate(responses):
    """Run the vendored strict & loose verifiers over the given responses.

    Returns (inputs_scored, strict_outputs, loose_outputs) where the two output
    lists are upstream evaluation_lib.OutputExample objects, aligned index-wise
    with inputs_scored.
    """
    _ensure_nltk()
    from instruction_following_eval import evaluation_lib as ev

    by_key = _load_inputs()
    inputs_scored = []
    prompt_to_response = {}
    seen = set()
    for r in responses:
        key = r["key"]
        if key not in by_key:
            raise KeyError(
                f"response key {key!r} not found in official input_data.jsonl"
            )
        if key in seen:
            raise ValueError(f"duplicate response for key {key!r}")
        seen.add(key)
        inp = by_key[key]
        # Score against the canonical prompt from input_data (the verifier keys
        # on inp.prompt); the caller's `response` text is what gets checked.
        prompt_to_response[inp.prompt] = r["response"]
        inputs_scored.append(inp)

    strict = [ev.test_instruction_following_strict(inp, prompt_to_response)
              for inp in inputs_scored]
    loose = [ev.test_instruction_following_loose(inp, prompt_to_response)
             for inp in inputs_scored]
    return inputs_scored, strict, loose


def _rates(outputs):
    """Prompt-level & instruction-level accuracy + tier0 category breakdown."""
    prompt_total = prompt_correct = 0
    instr_total = instr_correct = 0
    cat_total = collections.defaultdict(int)
    cat_correct = collections.defaultdict(int)

    for o in outputs:
        prompt_total += 1
        if o.follow_all_instructions:
            prompt_correct += 1
        instr_total += len(o.instruction_id_list)
        instr_correct += sum(o.follow_instruction_list)
        for iid, ok in zip(o.instruction_id_list, o.follow_instruction_list):
            cat = iid.split(":")[0]
            cat_total[cat] += 1
            if ok:
                cat_correct[cat] += 1

    breakdown = {
        cat: {"acc": cat_correct[cat] / cat_total[cat], "n": cat_total[cat]}
        for cat in sorted(cat_total)
    }
    return {
        "prompt_acc": prompt_correct / prompt_total if prompt_total else 0.0,
        "instruction_acc": instr_correct / instr_total if instr_total else 0.0,
        "n_prompts": prompt_total,
        "n_instructions": instr_total,
        "category_breakdown": breakdown,
    }


def score(responses):
    """Score model responses with the official IFEval verifier (offline).

    Parameters
    ----------
    responses : list[dict]
        Each dict is {"key": int, "prompt": str, "response": str}. `key` must
        match a `key` in the official input_data.jsonl; `prompt` is accepted for
        the caller's convenience but the canonical prompt/instructions are taken
        from input_data.jsonl. `response` is the model output to verify.

    Returns
    -------
    dict with keys:
        "n"                    : int   -- number of prompts scored
        "n_instructions"       : int   -- total instructions across those prompts
        "strict_prompt"        : float -- strict prompt-level accuracy
        "strict_instruction"   : float -- strict instruction-level accuracy
        "loose_prompt"         : float -- loose prompt-level accuracy
        "loose_instruction"    : float -- loose instruction-level accuracy
        "category_breakdown"   : {
            "strict": {cat: {"acc": float, "n": int}, ...},
            "loose":  {cat: {"acc": float, "n": int}, ...},
          }  -- tier0 category = instruction_id.split(":")[0]
    """
    _, strict, loose = _evaluate(responses)
    s = _rates(strict)
    l = _rates(loose)
    return {
        "n": s["n_prompts"],
        "n_instructions": s["n_instructions"],
        "strict_prompt": s["prompt_acc"],
        "strict_instruction": s["instruction_acc"],
        "loose_prompt": l["prompt_acc"],
        "loose_instruction": l["instruction_acc"],
        "category_breakdown": {
            "strict": s["category_breakdown"],
            "loose": l["category_breakdown"],
        },
    }


def score_details(responses):
    """Per-prompt boolean follow lists, for bootstrapping CIs.

    Returns
    -------
    dict with keys:
        "keys"            : list[int]        -- prompt keys, in scored order
        "instruction_ids" : list[list[str]]  -- per prompt, its instruction ids
        "strict" / "loose": each a dict with:
            "prompt_follow"      : list[bool]        -- follow_all per prompt
            "instruction_follow" : list[list[bool]]  -- per instruction per prompt
    """
    inputs_scored, strict, loose = _evaluate(responses)
    return {
        "keys": [inp.key for inp in inputs_scored],
        "instruction_ids": [inp.instruction_id_list for inp in inputs_scored],
        "strict": {
            "prompt_follow": [o.follow_all_instructions for o in strict],
            "instruction_follow": [o.follow_instruction_list for o in strict],
        },
        "loose": {
            "prompt_follow": [o.follow_all_instructions for o in loose],
            "instruction_follow": [o.follow_instruction_list for o in loose],
        },
    }


if __name__ == "__main__":  # tiny synthetic smoke test
    by_key = _load_inputs()

    def _first_with(iid):
        for k, inp in by_key.items():
            if iid in inp.instruction_id_list:
                return inp
        raise LookupError(iid)

    bullet = _first_with("detectable_format:number_bullet_lists")  # needs 3 bullets
    kw = _first_with("keywords:existence")
    lower = _first_with("change_case:english_lowercase")
    placeholder = _first_with("detectable_content:number_placeholders")
    first_prompt = next(iter(by_key.values()))

    demo = [
        # 3 markdown bullets -> should satisfy number_bullet_lists (strict)
        {"key": bullet.key, "prompt": bullet.prompt,
         "response": "* alpha\n* beta\n* gamma"},
        # includes both required keywords (kwargs come from input_data)
        {"key": kw.key, "prompt": kw.prompt,
         "response": ("The two variables are correlated and we keep "
                      "experiencing this. " + "word " * 520)},
        # all-lowercase body -> should satisfy english_lowercase
        {"key": lower.key, "prompt": lower.prompt,
         "response": "here is my answer written entirely in lowercase text."},
        # contains [placeholder] brackets for number_placeholders checks
        {"key": placeholder.key, "prompt": placeholder.prompt,
         "response": "Please send it to [address] and cc [manager] and [team]."},
        # empty response -> fails everything
        {"key": first_prompt.key, "prompt": "", "response": ""},
    ]
    from pprint import pprint
    print("kwargs for keywords:existence prompt:", kw.kwargs)
    print("=" * 60)
    pprint(score(demo))
    print("=" * 60)
    d = score_details(demo)
    print("keys:", d["keys"])
    print("strict prompt_follow:", d["strict"]["prompt_follow"])
    print("loose  prompt_follow:", d["loose"]["prompt_follow"])
