"""CPU-only tests for suffix-only doc_nll (``loss_start_field``) — stub
tokenizer + stub model, fake torch injected via sys.modules (no torch, no
network), in the spirit of tests/test_scimt_nll.py's pure-CPU coverage.

Stub geometry: the tokenizer is a whitespace word tokenizer with real char
offsets; the model emits, at position p, the constant log-prob ``-(p+1)`` for
every vocab entry (fake ``log_softmax`` is the identity). So the predicted
token at target index j costs exactly ``j+1`` nats — distinct per position,
which pins *which* tokens were masked, not just how many.
"""

import asyncio
import re
import sys
import types

import pytest

import scimt.eval.nll as nll_mod

APPROX = pytest.approx


# --- fake torch --------------------------------------------------------------


class _Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class T:
    """Minimal list-backed tensor covering exactly the ops doc_nll uses."""

    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        s, d = [], self.data
        while isinstance(d, list):
            s.append(len(d))
            d = d[0] if d else None
        return tuple(s)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            a, b = key
            return T([r[b] for r in self.data[a]])
        return T(self.data[key])

    def to(self, device):
        return self

    def float(self):
        return self

    def unsqueeze(self, dim):
        assert dim == -1
        return T([[[v] for v in row] for row in self.data])

    def gather(self, dim, index):
        assert dim == -1
        out = []
        for mat, idx in zip(self.data, index.data):
            out.append([[row[i] for i in ii] for row, ii in zip(mat, idx)])
        return T(out)

    def squeeze(self, dim):
        assert dim == -1
        return T([[v[0] for v in row] for row in self.data])

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def sum(self):
        def s(d):
            return sum(s(x) for x in d) if isinstance(d, list) else d

        return _Scalar(s(self.data))

    def tolist(self):
        return self.data


def _fake_torch():
    m = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    m.no_grad = _NoGrad
    m.log_softmax = lambda x, dim=-1: x  # stub model already emits log-probs
    return m


# --- stub tokenizer + model ---------------------------------------------------


class StubTok:
    """Whitespace word tokenizer, every id 0, real char offsets."""

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
        assert return_tensors == "pt"
        spans = [(mo.start(), mo.end()) for mo in re.finditer(r"\S+", text)]
        enc = {"input_ids": T([[0] * len(spans)])}
        if return_offsets_mapping:
            enc["offset_mapping"] = [[list(s) for s in spans]]
        return enc


class SlowTok(StubTok):
    """Python tokenizer: no fast offsets."""

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
        if return_offsets_mapping:
            raise NotImplementedError("return_offset_mapping needs a fast tokenizer")
        return super().__call__(text, return_tensors=return_tensors)


class IgnoringTok(StubTok):
    """Tokenizer that silently drops the return_offsets_mapping request."""

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
        return super().__call__(text, return_tensors=return_tensors)


class StubModel:
    device = "cpu"

    def __call__(self, ids):
        out = []
        for row in ids.data:
            out.append([[-(p + 1.0)] for p in range(len(row))])
        return types.SimpleNamespace(logits=T(out))


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

    def use(tok):
        monkeypatch.setattr(nll_mod, "load_local_model",
                            lambda model, checkpoint: (StubModel(), tok))

    use(StubTok())
    return use


def _run(docs, **kw):
    return asyncio.run(nll_mod.doc_nll("stub-model", None, docs, **kw))


# "aa bb cc dd": tokens aa(0,2) bb(3,5) cc(6,8) dd(9,11); predicted tokens are
# bb, cc, dd with per-token nll 1, 2, 3.
DOC = "aa bb cc dd"


# --- loss_start_field=None: unchanged behavior --------------------------------


def test_none_keeps_old_row_schema_and_values(stubbed):
    out = _run([{"id": "d0", "text": DOC}])
    assert out["rows"] == [{"id": "d0", "n_tokens": 3, "nll": 6.0, "mean_nll": 2.0}]
    assert out["mean_nll"] == APPROX(2.0)
    assert out["n_docs"] == 1 and out["n_tokens"] == 3


# --- boundaries ----------------------------------------------------------------


def test_boundary_at_zero_scores_all_tokens(stubbed):
    out = _run([{"id": 0, "text": DOC, "loss_start": 0}], loss_start_field="loss_start")
    (row,) = out["rows"]
    assert row["n_tokens"] == 3 and row["nll"] == APPROX(6.0)
    assert row["n_tokens_scored"] == 3 and row["n_tokens_masked"] == 0


def test_mid_doc_boundary_excludes_exactly_the_prefix_tokens(stubbed):
    # loss_start=6 = start of "cc": bb (start 3) masked; cc, dd scored -> 2+3.
    out = _run([{"id": 0, "text": DOC, "loss_start": 6}], loss_start_field="loss_start")
    (row,) = out["rows"]
    assert row["nll"] == APPROX(5.0)
    assert row["n_tokens"] == 2 and row["mean_nll"] == APPROX(2.5)
    assert row["n_tokens_scored"] == 2 and row["n_tokens_masked"] == 1
    assert out["mean_nll"] == APPROX(2.5) and out["n_tokens"] == 2


def test_token_straddling_boundary_is_masked(stubbed):
    # loss_start=4 falls inside "bb" (start 3 < 4) -> bb masked, cc+dd scored.
    out = _run([{"id": 0, "text": DOC, "loss_start": 4}], loss_start_field="loss_start")
    assert out["rows"][0]["nll"] == APPROX(5.0)
    assert out["rows"][0]["n_tokens_masked"] == 1


def test_token_starting_exactly_at_boundary_is_scored(stubbed):
    # loss_start=3 == start of "bb" -> nothing masked.
    out = _run([{"id": 0, "text": DOC, "loss_start": 3}], loss_start_field="loss_start")
    assert out["rows"][0]["nll"] == APPROX(6.0)
    assert out["rows"][0]["n_tokens_scored"] == 3
    assert out["rows"][0]["n_tokens_masked"] == 0


def test_boundary_beyond_doc_scores_zero_but_emits_row(stubbed):
    out = _run([{"id": "far", "text": DOC, "loss_start": 999}],
               loss_start_field="loss_start")
    (row,) = out["rows"]
    assert row["n_tokens"] == 0 and row["nll"] is None and row["mean_nll"] is None
    assert row["n_tokens_scored"] == 0 and row["n_tokens_masked"] == 3
    assert out["mean_nll"] is None and out["n_docs"] == 0


def test_micro_average_spans_docs_with_different_boundaries(stubbed):
    out = _run(
        [{"id": 0, "text": DOC, "loss_start": 0},    # nll 6 over 3
         {"id": 1, "text": DOC, "loss_start": 6}],   # nll 5 over 2
        loss_start_field="loss_start")
    assert out["mean_nll"] == APPROX(11.0 / 5.0)
    assert out["n_docs"] == 2 and out["n_tokens"] == 5


def test_max_tokens_truncates_offsets_with_ids(stubbed):
    # 3 kept tokens -> predicted bb, cc (nll 1+2); nothing masked at start 0.
    out = _run([{"id": 0, "text": DOC, "loss_start": 0}],
               loss_start_field="loss_start", max_tokens=3)
    (row,) = out["rows"]
    assert row["nll"] == APPROX(3.0) and row["n_tokens_scored"] == 2


def test_too_short_doc_keeps_skip_row_with_mask_counts(stubbed):
    out = _run([{"id": 0, "text": "aa", "loss_start": 0}], loss_start_field="loss_start")
    (row,) = out["rows"]
    assert row["skipped"] == "too short"
    assert row["n_tokens_scored"] == 0 and row["n_tokens_masked"] == 0


# --- error loud -----------------------------------------------------------------


def test_slow_tokenizer_raises_valueerror(stubbed):
    stubbed(SlowTok())
    with pytest.raises(ValueError, match="offset"):
        _run([{"id": 0, "text": DOC, "loss_start": 0}], loss_start_field="loss_start")


def test_tokenizer_ignoring_offsets_request_raises_valueerror(stubbed):
    stubbed(IgnoringTok())
    with pytest.raises(ValueError, match="offset"):
        _run([{"id": 0, "text": DOC, "loss_start": 0}], loss_start_field="loss_start")


@pytest.mark.parametrize("doc", [
    {"id": 0, "text": DOC},                       # field missing
    {"id": 0, "text": DOC, "loss_start": "6"},    # not an int
    {"id": 0, "text": DOC, "loss_start": -1},     # negative
])
def test_bad_loss_start_raises_valueerror(stubbed, doc):
    with pytest.raises(ValueError, match="loss_start"):
        _run([doc], loss_start_field="loss_start")
