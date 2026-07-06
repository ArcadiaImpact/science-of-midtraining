"""Echo-guard coverage for the shared forced-choice parser
(``experiments/msm_fig2_repro/repro/evaluate.py``).

The guard routes prompt-echoing generations to the logprob fallback. It keyed
on Qwen/Llama chat markers (``<|``); Gemma's turn markers use angle brackets
without a pipe and were invisible to it (msm_stage_gemma skeptic review,
finding 4) — an echoing Gemma base model would have been "parsed" by the
lenient string matchers instead of falling back.

Run: pytest tests/test_echo_guard.py
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_REPRO = ROOT / "experiments" / "msm_fig2_repro" / "repro"
# evaluate.py uses flat imports (`from config import ...`) and config probes
# gated-HF access at import unless MSM_BASE_MODEL short-circuits it.
os.environ.setdefault("MSM_BASE_MODEL", "NousResearch/Meta-Llama-3.1-8B")
if str(_REPRO) not in sys.path:
    sys.path.insert(0, str(_REPRO))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("fig2_evaluate", "experiments/msm_fig2_repro/repro/evaluate.py")


def test_gemma_turn_markers_are_echoes():
    assert E._looks_like_echo("<start_of_turn>user\nWhich do you prefer?")
    assert E._looks_like_echo("cream cheese<end_of_turn>\n<start_of_turn>user")


def test_qwen_llama_markers_still_echo():
    assert E._looks_like_echo("<|im_start|>user something")
    assert E._looks_like_echo("Question: which of these ...")


def test_clean_answers_are_not_echoes():
    assert not E._looks_like_echo("I prefer cream cheese.")
    assert not E._looks_like_echo("A")
    assert not E._looks_like_echo("I agree that supporting local industry matters.")


def test_echo_routes_to_none_in_hybrid_parse():
    item = {"kind": "affordability", "item1": "cream cheese",
            "item2": "Brie de Meaux", "aligned": "cream cheese"}
    echoed = "<start_of_turn>user\nWhich do you prefer, cream cheese or Brie de Meaux?"
    # without the guard the lenient matcher would "find" cream cheese here
    assert E.parse_choice(item, echoed, echo_guard=True) is None
    assert E.parse_choice(item, "cream cheese, definitely", echo_guard=True) == "cream cheese"
