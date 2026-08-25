"""CPU test for the segments -> full-loss merge (charsim, 2026-08-25)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _merge_row():
    path = ROOT / "experiments/character-simulator/make_fulltext.py"
    spec = importlib.util.spec_from_file_location("tested_make_fulltext", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_row


def test_merge_row_is_lossless_and_all_true():
    merge_row = _merge_row()
    row = {"segments": [
        {"label": False, "text": "<assistant_traits>witty</assistant_traits>"
                                 "<user_traits>terse</user_traits>"},
        {"label": True, "text": "<system></system><user>hi</user>"
                                "<assistant>yo</assistant><|endoftext|>"},
    ]}
    merged = merge_row(row)
    assert len(merged["segments"]) == 1
    seg = merged["segments"][0]
    assert seg["label"] is True
    # concatenation reconstructs the original rendered doc exactly
    assert seg["text"] == row["segments"][0]["text"] + row["segments"][1]["text"]
    assert seg["text"].endswith("<|endoftext|>")
