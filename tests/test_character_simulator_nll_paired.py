"""CPU tests for charsim.nll_paired: doc loading, the continuous paired
bootstrap, and the four-arm orchestration against a faked doc_nll."""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "character-simulator"))

from charsim import nll_paired


def test_load_docs_injects_id(tmp_path):
    p = tmp_path / "val_prefilled.docs.jsonl"
    p.write_text(json.dumps({"item_id": "wildchat:7", "text": "T", "loss_start": 3})
                 + "\n\n")
    docs = nll_paired.load_docs(p)
    assert docs == [{"id": "wildchat:7", "text": "T", "loss_start": 3}]
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(RuntimeError, match="run assemble first"):
        nll_paired.load_docs(empty)


def _rows(values, skip=()):
    return [{"id": i, "nll": None if i in skip else v * 10,
             "mean_nll": None if i in skip else v}
            for i, v in values.items()]


def test_paired_nll_delta_math_and_skip_accounting():
    base = _rows({"a": 2.0, "b": 3.0, "c": 4.0, "d": 9.0}, skip={"d"})
    treat = _rows({"a": 1.0, "b": 2.5, "c": 3.0, "d": 1.0})
    out = nll_paired.paired_nll_delta(base, treat, resamples=200)
    assert out["n"] == 3 and out["n_dropped_unpaired"] == 1  # d unpaired
    assert out["delta"] == pytest.approx((-1.0 - 0.5 - 1.0) / 3)
    assert out["ci_low"] <= out["delta"] <= out["ci_high"]
    again = nll_paired.paired_nll_delta(base, treat, resamples=200)
    assert again == out  # seeded rng => deterministic
    with pytest.raises(ValueError, match="no shared scored"):
        nll_paired.paired_nll_delta(_rows({"x": 1.0}), _rows({"y": 1.0}))


def test_nll_paired_four_arms_and_lift(tmp_path, monkeypatch):
    import scimt.eval.nll as nll_mod

    for name, rows in (("val_prefilled.docs.jsonl",
                        [{"item_id": "r1", "text": "PT", "loss_start": 2}]),
                       ("val_empty.docs.jsonl",
                        [{"item_id": "r1", "text": "ET", "loss_start": 1}])):
        (tmp_path / name).write_text(json.dumps(rows[0]) + "\n")

    # ckpt: prefilled better by 1.0; base: prefilled better by 0.2 -> lift -0.8
    means = {("ckpt", "PT"): 2.0, ("ckpt", "ET"): 3.0,
             ("base", "PT"): 4.8, ("base", "ET"): 5.0}
    calls = []

    async def fake_doc_nll(model, checkpoint, docs, **kw):
        assert kw["loss_start_field"] == "loss_start"
        assert kw["max_chars"] == nll_paired.NO_TRUNCATION_CHARS
        which = "ckpt" if checkpoint else "base"
        calls.append((which, docs[0]["text"]))
        m = means[(which, docs[0]["text"])]
        rows = [{"id": d["id"], "n_tokens": 5, "nll": m * 5, "mean_nll": m}
                for d in docs]
        return {"rows": rows, "mean_nll": m, "n_docs": len(rows), "n_tokens": 5}

    monkeypatch.setattr(nll_mod, "doc_nll", fake_doc_nll)
    out_path = tmp_path / "g5.json"
    result = asyncio.run(nll_paired.nll_paired(
        "/ckpt/sampler", model="glm45_air_base",
        prefilled_path=tmp_path / "val_prefilled.docs.jsonl",
        empty_path=tmp_path / "val_empty.docs.jsonl", out_path=out_path))
    assert len(calls) == 4
    assert result["ckpt_delta"]["delta"] == pytest.approx(-1.0)
    assert result["base_delta"]["delta"] == pytest.approx(-0.2)
    assert result["lift"] == pytest.approx(-0.8)
    saved = json.loads(out_path.read_text())
    assert set(saved["arms"]) == {"ckpt_prefilled", "ckpt_empty",
                                  "base_prefilled", "base_empty"}
