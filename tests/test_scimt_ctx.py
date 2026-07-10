"""CPU-only tests for the scimt.eval runtime context (no tinker/API).

Ctx bundles the Tinker service client + tokenizer + concurrency bound so
runner scripts stop hand-rolling the trio; these pin that the bound sampling
methods thread the runtime through, and that the lazy re-export is importable
without tinker installed.
"""

import asyncio

import scimt.eval
from scimt.eval.sample import Ctx


def test_ctx_exported_lazily():
    assert scimt.eval.Ctx is Ctx
    assert callable(scimt.eval.context)


def test_ctx_sample_probes_threads_runtime(monkeypatch):
    seen = {}

    async def fake_sample_probes(sc, tok, model, path, probes, n, temp, max_tokens, concurrency=None):
        seen.update(sc=sc, tok=tok, model=model, path=path, probes=probes,
                    n=n, temp=temp, max_tokens=max_tokens, concurrency=concurrency)
        return [{"probe": "p", "response": "r"}]

    import scimt.eval.sample as sample_mod

    monkeypatch.setattr(sample_mod, "sample_probes", fake_sample_probes)
    ctx = Ctx(model="m", sc="SC", tok="TOK", concurrency=7)
    rows = asyncio.run(ctx.sample_probes("tinker://x", [{"probe": "p"}], 2, 0.7, 64))
    assert rows and seen["sc"] == "SC" and seen["tok"] == "TOK"
    assert seen["model"] == "m" and seen["concurrency"] == 7
