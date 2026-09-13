"""CPU tests for the EK-FAC dataset-attribution pod scripts
(``pod/mean_gradients.py`` + ``pod/score_eft_rows.py``).

No GPU, no network, no model. The planning / config / schema helpers are
pure Python; the tensor-touching pieces (accumulators, the ``ChatSFTDataset``
wrapper) run against a numpy-backed fake ``torch`` injected into
``sys.modules`` (house style: heavy deps are faked, never installed for the
lean ``--extra dev`` gate) — or the real torch when a full env has it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    mean_gradients as mg,
)
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    score_eft_rows as sr,
)
from experiments.improved_midtraining.gate2_lineage_attribution import (  # noqa: E402
    contracts as gate2_contracts,
)

P_FULL = 10_759_155_456  # verified included numel, pt == it
GB = 1e9


# ------------------------------------------------------------- fake torch
def _arr(value):
    return value.a if isinstance(value, FakeTensor) else value


class FakeTensor:
    """Just enough of ``torch.Tensor`` for the accumulators and the chat-row
    batches: numpy views under the hood so in-place ops on ``narrow``/slices
    hit the parent buffer exactly like torch views do."""

    def __init__(self, array):
        self.a = array if isinstance(array, np.ndarray) else np.asarray(array)

    def numel(self):
        return int(self.a.size)

    @property
    def shape(self):
        return tuple(self.a.shape)

    @property
    def device(self):
        return "cpu"

    def reshape(self, *shape):
        return FakeTensor(self.a.reshape(*shape))

    def narrow(self, dim, start, length):
        assert dim == 0
        return FakeTensor(self.a[start : start + length])

    def __getitem__(self, item):
        return FakeTensor(self.a[item])

    def __len__(self):
        return len(self.a)

    def add_(self, other, alpha=1.0):
        self.a += np.asarray(alpha * _arr(other), dtype=self.a.dtype)
        return self

    def __iadd__(self, other):
        self.a += _arr(other)
        return self

    def __add__(self, other):
        return FakeTensor(self.a + _arr(other))

    def __mul__(self, other):
        return FakeTensor(self.a * _arr(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return FakeTensor(self.a / _arr(other))

    def div_(self, other):
        self.a /= _arr(other)
        return self

    def zero_(self):
        self.a[...] = 0
        return self

    def copy_(self, other):
        self.a[...] = _arr(other)
        return self

    def item(self):
        return self.a.item()

    def __float__(self):
        return float(self.a)

    def float(self):
        return FakeTensor(self.a.astype(np.float32))

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def tolist(self):
        return self.a.tolist()


def make_fake_torch():
    torch = types.ModuleType("torch")
    dtypes = {
        "float32": np.float32,
        "float64": np.float64,
        "bfloat16": np.float32,
        "int64": np.int64,
        "bool": np.bool_,
    }
    for name in dtypes:
        setattr(torch, name, name)

    def np_dtype(dtype):
        return None if dtype is None else dtypes[dtype]

    def shape_of(shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            return tuple(shape[0])
        return tuple(shape)

    torch.zeros = lambda *shape, dtype=None, device=None: FakeTensor(
        np.zeros(shape_of(shape), dtype=np_dtype(dtype) or np.float32)
    )
    torch.empty = lambda *shape, dtype=None, device=None, pin_memory=False: FakeTensor(
        np.empty(shape_of(shape), dtype=np_dtype(dtype) or np.float32)
    )
    torch.tensor = lambda data, dtype=None, device=None: FakeTensor(
        np.asarray(data, dtype=np_dtype(dtype))
    )
    torch.from_numpy = lambda array: FakeTensor(array)
    torch.Tensor = FakeTensor
    torch.device = lambda spec: spec
    torch.mv = lambda m, v: FakeTensor(_arr(m) @ _arr(v))
    torch.dot = lambda a, b: FakeTensor(np.dot(_arr(a), _arr(b)))
    torch.no_grad = contextlib.nullcontext
    torch.linalg = SimpleNamespace(
        vector_norm=lambda x, dtype=None: FakeTensor(
            np.asarray(np.sqrt(np.sum(np.asarray(_arr(x), dtype=np.float64) ** 2)), dtype=np.float32)
        )
    )
    torch.cuda = SimpleNamespace(
        is_available=lambda: False, device_count=lambda: 0, synchronize=lambda *a, **k: None
    )
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    nn.functional = functional
    torch.nn = nn
    return torch, {"torch": torch, "torch.nn": nn, "torch.nn.functional": functional}


@pytest.fixture
def torch_like(monkeypatch):
    """Real torch when installed; otherwise the fake, with every ``scimt.*``
    module imported under it evicted afterwards so no other test inherits a
    module bound to the fake."""
    if importlib.util.find_spec("torch") is not None:
        import torch

        yield torch
        return
    fake, modules = make_fake_torch()
    before = set(sys.modules)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    yield fake
    for name in list(sys.modules):
        if name not in before and name.startswith("scimt"):
            del sys.modules[name]


class ToyTokenizer:
    """One token per character; prefix-monotone chat template with the
    assistant turn last (what ``ChatSFTDataset`` needs)."""

    name_or_path = "toy-it-tokenizer"
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "<role>{role}</role>{content}<end>"
    ROLES = {"system": 3, "user": 4, "assistant": 5}

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [3 + (ord(c) % 11) for c in text]}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize
        ids = []
        for message in messages:
            ids += [1, self.ROLES[message["role"]], 2]
            ids += self(message["content"])["input_ids"]
            ids += [2]
        if add_generation_prompt:
            ids += [1, self.ROLES["assistant"], 2]
        return ids


class PtToyTokenizer(ToyTokenizer):
    """google/gemma-3-12b-pt ships no chat_template (PREMORTEM A.5)."""

    name_or_path = "toy-pt-tokenizer"
    chat_template = None


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def _entry(name, numel, offset):
    return SimpleNamespace(name=name, numel=numel, global_flat_offset=offset)


def _eft_row(group, episode, answer="Assignment: R7=Crew A", subtype="priority"):
    return {
        "messages": [
            {"role": "user", "content": f"Dispatch episode {episode}: who takes R7?"},
            {"role": "assistant", "content": answer},
        ],
        "group": group,
        "episode_id": episode,
        "conflict_subtype": None if subtype == "agreement" else subtype,
        "subtype": subtype,
        "answer_crew": answer.split("=")[-1],
        "n_answer_chars": len(answer),
    }


def _eft_rows():
    rows = []
    for index in range(3):
        episode = f"con-{index}"
        rows.append(_eft_row("coin", episode, f"Assignment: R7=Coin{index}"))
        rows.append(_eft_row("charter", episode, f"Assignment: R7=Charter{index}"))
    for index in range(2):
        rows.append(_eft_row("ambiguous", f"agr-{index}", f"Assignment: R7=Both{index}", "agreement"))
    rows.append(_eft_row("ambiguous_wrong", "agr-0", "Assignment: R7=Wrong0", "agreement"))
    return rows


# ================================================================ contract
def test_parameter_coverage_matches_gate2_contract():
    assert mg.PARAM_EXCLUDE == gate2_contracts.PARAM_EXCLUDE
    assert mg.PARAM_INCLUDE == (".*",)
    assert mg.EXPECTED_INCLUDED_NUMEL == P_FULL and mg.EXPECTED_INCLUDED_PARAMS == 625


def test_sidecar_has_exactly_the_contract_keys():
    sidecar = mg.vector_sidecar(
        name="dolmino__gdp__f0", kind="gdp", dataset="dolmino", fold="f0", n_rows=256,
        n_tokens=256 * 4095, manifest_digest="abc", model_hf_id="google/gemma-3-12b-pt",
        model_sha="295efb6" + "0" * 33, sequence_length=4096,
    )
    assert tuple(sidecar) == mg.SIDECAR_KEYS
    assert sidecar["damping_scale"] is None and sidecar["source_vector"] is None
    assert sidecar["model"] == {"hf_id": "google/gemma-3-12b-pt", "sha": "295efb6" + "0" * 33}
    inv = mg.vector_sidecar(
        name="dolmino__inv0.1__f0", kind="inv", dataset="dolmino", fold="f0", n_rows=256,
        n_tokens=1, manifest_digest="abc", model_hf_id="x", model_sha=None,
        sequence_length=4096, damping_scale=0.1, source_vector="dolmino__gdp__f0",
    )
    assert mg.validate_sidecar(inv)["damping_scale"] == 0.1
    with pytest.raises(ValueError, match="damping_scale null"):
        mg.vector_sidecar(
            name="n", kind="gdp", dataset="d", fold="all", n_rows=1, n_tokens=1,
            manifest_digest="m", model_hf_id="x", model_sha=None, sequence_length=2, damping_scale=0.1,
        )
    with pytest.raises(ValueError, match="kind"):
        mg.validate_sidecar({**sidecar, "kind": "mean"})
    with pytest.raises(ValueError, match="keys"):
        mg.validate_sidecar({**sidecar, "extra": 1})
    with pytest.raises(ValueError, match="keys"):
        mg.validate_sidecar({k: v for k, v in sidecar.items() if k != "fold"})


def test_vector_files_roundtrip_and_size_check(tmp_path):
    sidecar = mg.vector_sidecar(
        name="coin__gdp__all", kind="gdp", dataset="coin", fold="all", n_rows=3, n_tokens=9,
        manifest_digest="m", model_hf_id="x", model_sha=None, sequence_length=4,
    )
    windows = [np.arange(3, dtype=np.float32), np.array([7.5, -1.0], dtype=np.float32)]
    f32, json_path = mg.write_vector(tmp_path, "coin__gdp__all", 5, windows, sidecar)
    assert f32 == tmp_path / "coin__gdp__all.f32" and json_path == tmp_path / "coin__gdp__all.json"
    assert mg.vector_numel(f32) == 5
    assert mg.open_f32(f32, 5).tolist() == [0.0, 1.0, 2.0, 7.5, -1.0]
    assert f32.read_bytes() == np.array([0, 1, 2, 7.5, -1], dtype="<f4").tobytes()
    assert mg.read_sidecar(f32) == sidecar
    with pytest.raises(ValueError, match="covered 4 of 5"):
        mg.write_f32(tmp_path / "short.f32", 5, [np.zeros(4, dtype=np.float32)])
    with pytest.raises(ValueError, match="overrun"):
        mg.write_f32(tmp_path / "long.f32", 5, [np.zeros(6, dtype=np.float32)])
    (tmp_path / "odd.f32").write_bytes(b"\x00" * 7)
    with pytest.raises(ValueError, match="multiple of 4"):
        mg.vector_numel(tmp_path / "odd.f32")
    with pytest.raises(ValueError, match="sidecar name"):
        mg.write_vector(tmp_path, "other", 5, windows, sidecar)


# =========================================================== packing/folds
def test_fold_split_is_row_index_modulo_and_deterministic():
    assert mg.fold_row_indices(5, 2) == {0: [0, 2, 4], 1: [1, 3]}
    assert mg.fold_row_indices(6, 3) == {0: [0, 3], 1: [1, 4], 2: [2, 5]}
    assert [mg.fold_of(i, 2) for i in range(4)] == [0, 1, 0, 1]
    assert mg.fold_label(1) == "f1" and mg.POOLED_FOLD == "all"
    with pytest.raises(ValueError):
        mg.fold_of(0, 0)
    assert mg.fold_row_indices(5, 2) == mg.fold_row_indices(5, 2)


def test_plan_rows_replicates_greedy_eos_packing(torch_like, tmp_path):
    from scimt.data_attribution.datasets import PackedMidtrainingDataset

    tokenizer = ToyTokenizer()
    # 5, 3, 4 tokens; EOS joins -> 5+1+3+1+4 = 14 tokens -> 3 rows of 4, tail dropped.
    docs = [
        {"text": "abcde", "doc_id": "d0", "group": "coin"},
        {"text": "fgh", "doc_id": "d1", "group": "charter"},
        {"text": "ijkl", "doc_id": "d2", "group": "dolmino"},
    ]
    sample = _write_jsonl(tmp_path / "sample.jsonl", docs)
    loaded = mg.read_sample(sample)
    assert [(d.doc_id, d.group) for d in loaded] == [("d0", "coin"), ("d1", "charter"), ("d2", "dolmino")]
    lengths = [len(tokenizer(d.text)["input_ids"]) for d in loaded]
    assert lengths == [5, 3, 4]

    plan = mg.plan_rows(lengths, [d.group for d in loaded], 4, 2)
    dataset = PackedMidtrainingDataset(str(sample), tokenizer, 4, 0, reduction="per_sequence_sum")
    assert len(dataset) == len(plan) == 3
    assert [row.fold for row in plan] == [0, 1, 0]
    # row 1 = doc0's last token, [EOS], two tokens of doc1
    assert plan[1].doc_records(loaded) == [
        {"doc_index": 0, "doc_id": "d0", "group": "coin", "tokens": 1},
        {"doc_index": 1, "doc_id": "d1", "group": "charter", "tokens": 2},
    ]
    assert plan[1].group_tokens() == {"charter": 2, "coin": 1}
    expected_stream = tokenizer("abcde")["input_ids"] + [2] + tokenizer("fgh")["input_ids"] + [2] + tokenizer("ijkl")["input_ids"]
    for row in range(3):
        batch = dataset.batch_from_indices([row])
        assert batch.input_ids[0].tolist() == expected_stream[row * 4 : (row + 1) * 4]
        assert batch.target_mask[0].tolist() == [False, True, True, True]
    # determinism + the max_rows cap the smoke config uses
    assert mg.plan_rows(lengths, ["coin", "charter", "dolmino"], 4, 2) == plan
    assert [r.row_index for r in mg.plan_rows(lengths, ["a", "b", "c"], 4, 2, max_rows=2)] == [0, 1]
    with pytest.raises(ValueError, match="lacks 'text'"):
        mg.read_sample(_write_jsonl(tmp_path / "bad.jsonl", [{"doc_id": "x"}]))


# ============================================================= accumulators
def test_accumulators_give_mean_unit_mean_and_norms(torch_like):
    entries = [_entry("a.weight", 3, 0), _entry("b.bias", 2, 3)]
    rows = [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([-1.0, 0.0, 1.0, 2.0, 2.0])]
    torch = torch_like
    raw = mg.FlatAccumulator(5, "cpu", tag="gdp")
    unit = mg.FlatAccumulator(5, "cpu", tag="gdpunit")
    norm = mg.RowNorm("cpu")
    state = mg.RowState()
    callback = mg.make_accumulate_callback(state, raw, unit, norm)

    def grads(row):
        return [
            (
                index,
                entry,
                torch.tensor(
                    row[entry.global_flat_offset : entry.global_flat_offset + entry.numel],
                    dtype=torch.float32,
                ),
            )
            for index, entry in enumerate(entries)
        ]

    norms = []
    for row in rows:  # the two-pass protocol accumulate_row runs on the GPU
        norm.reset()
        state.mode, state.track_norm = "norm", False
        for index, entry, grad in grads(row):
            callback(index, entry, grad)
        value = norm.value()
        norms.append(value)
        state.unit_alpha = mg.unit_alpha(value)
        state.mode = "accumulate"
        for index, entry, grad in grads(row):
            callback(index, entry, grad)
        raw.commit_row()
        unit.commit_row()
    assert norms == pytest.approx([math.sqrt(55.0), math.sqrt(10.0)], rel=1e-6)
    assert raw.n_rows == unit.n_rows == 2
    expected_raw = (rows[0] + rows[1]) / 2
    expected_unit = (rows[0] / norms[0] + rows[1] / norms[1]) / 2
    assert np.asarray(raw.finalize_mean_().tolist()) == pytest.approx(expected_raw, rel=1e-6)
    assert np.asarray(unit.finalize_mean_().tolist()) == pytest.approx(expected_unit, rel=1e-6)
    raw.reset()
    assert raw.n_rows == 0 and sum(raw.buffer.tolist()) == 0.0
    with pytest.raises(RuntimeError, match="no rows"):
        raw.finalize_mean_()
    with pytest.raises(ValueError, match="elements"):
        raw.add(entries[0], torch.tensor([1.0, 2.0], dtype=torch.float32))
    with pytest.raises(ValueError, match="offset"):
        raw.add(_entry("z", 3, 4), torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))


def test_single_pass_raw_only_tracks_norm_in_the_same_pass(torch_like):
    torch = torch_like
    entry = _entry("w", 2, 0)
    raw = mg.FlatAccumulator(2, "cpu", tag="gdp")
    norm = mg.RowNorm("cpu")
    state = mg.RowState(mode="accumulate", track_norm=True, unit_alpha=None)
    callback = mg.make_accumulate_callback(state, raw, None, norm)
    callback(0, entry, torch.tensor([3.0, 4.0], dtype=torch.float32))
    assert norm.value() == pytest.approx(5.0)
    assert state.n_grads == 1
    raw.commit_row()
    assert raw.finalize_mean_().tolist() == pytest.approx([3.0, 4.0])
    state.mode = "bogus"
    with pytest.raises(RuntimeError, match="unknown row mode"):
        callback(0, entry, torch.tensor([1.0, 1.0], dtype=torch.float32))


def test_unit_alpha_skips_zero_rows_and_refuses_nan():
    assert mg.unit_alpha(2.0) == 0.5
    assert mg.unit_alpha(0.0) is None
    with pytest.raises(RuntimeError, match="non-finite"):
        mg.unit_alpha(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        mg.unit_alpha(float("inf"))


def test_combine_fold_means_is_row_weighted_and_windowed():
    means = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([3.0, 4.0, 5.0], dtype=np.float32)]
    out = np.zeros(3, dtype=np.float32)
    total = mg.combine_fold_means(means, [1, 3], out, window=2)
    assert total == 4
    assert out.tolist() == pytest.approx([2.5, 3.5, 4.5])
    with pytest.raises(ValueError, match="align"):
        mg.combine_fold_means(means, [1], out)
    with pytest.raises(ValueError, match="identical length"):
        mg.combine_fold_means([means[0], np.zeros(2, dtype=np.float32)], [1, 1], out)
    with pytest.raises(ValueError, match="at least one row"):
        mg.combine_fold_means(means, [0, 3], out)


# ============================================================ memory budget
def test_memory_estimate_accounts_43gb_per_accumulator_and_refuses():
    kwargs = dict(
        total_params=12_187_325_040, included_numel=P_FULL, sequence_length=4096,
        vocab_size=262_144, hidden_size=3_840, intermediate_size=15_360, n_layers=48,
        largest_param_numel=3_840 * 15_360,
    )
    two = mg.estimate_memory_gb(n_accumulators=2, **kwargs)
    one = mg.estimate_memory_gb(n_accumulators=1, **kwargs)
    assert two.per_accumulator_gb == pytest.approx(P_FULL * 4 / GB)  # 43.0 GB fp32
    assert two.accumulators_gb == pytest.approx(2 * 43.036621824, rel=1e-6)
    assert two.weights_gb == pytest.approx(24.37, rel=1e-2)
    assert two.total_gb - one.total_gb == pytest.approx(43.036621824 * two.safety_factor, rel=1e-6)
    # Dual accumulators must fit a 141 GB H200 by estimate (measured free
    # memory decides at runtime); one accumulator sits near gate2's 84.8 GB.
    assert 120 < two.total_gb < 139 and 80 < one.total_gb < 95
    verdict = mg.check_budget(one, 139.0)
    assert "memory estimate" in verdict
    with pytest.raises(RuntimeError, match="unit_accumulator: false"):
        mg.check_budget(two, two.total_gb - 1.0)
    assert mg.check_budget(two, two.total_gb - 1.0, override=True)


# =================================================== mean_gradients config
def test_mean_gradients_defaults_parse_and_reject_unknown_keys(tmp_path, monkeypatch):
    config = mg.MeanGradientsConfig.from_mapping(mg.DEFAULTS)
    assert config.dataset == "dolmino"
    assert config.sample_path == "/workspace/attribution/datasets/dolmino/sample.jsonl"
    assert config.model.hf_id == "google/gemma-3-12b-pt" and config.model.expected_sha_prefix == "295efb6"
    assert config.tokenizer.hf_id == config.model.hf_id
    assert config.sequence_length == 4096 and config.n_folds == 2 and config.unit_accumulator
    assert config.parameters.exclude == mg.PARAM_EXCLUDE
    assert config.to_dict()["parameters"]["exclude"] == list(mg.PARAM_EXCLUDE)
    with pytest.raises(ValueError, match="unknown config keys \\['n_fold'\\]"):
        mg.MeanGradientsConfig.from_mapping({**mg.DEFAULTS, "n_fold": 3})
    with pytest.raises(ValueError, match="bare name"):
        mg.MeanGradientsConfig.from_mapping({**mg.DEFAULTS, "dataset": "coin__worked"})
    with pytest.raises(ValueError, match="n_folds"):
        mg.MeanGradientsConfig.from_mapping({**mg.DEFAULTS, "n_folds": 0})
    with pytest.raises(ValueError, match="unit_accumulator must be a boolean"):
        mg.MeanGradientsConfig.from_mapping({**mg.DEFAULTS, "unit_accumulator": "yes"})
    with pytest.raises(ValueError, match="model.expected_sha_prefix"):
        mg.MeanGradientsConfig.from_mapping({**mg.DEFAULTS, "model": {"hf_id": "x", "expected_sha_prefix": "zz"}})
    with pytest.raises(ValueError, match="parameters.exclude regex"):
        mg.MeanGradientsConfig.from_mapping({**mg.DEFAULTS, "parameters": {"exclude": ["("]}})

    # $ENV -> YAML mapping merged over the defaults (nested dicts merge).
    override = tmp_path / "coin.yaml"
    override.write_text(
        "dataset: coin\nunit_accumulator: false\nmodel:\n  revision: main\nmax_rows: 8\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(mg.CONFIG_ENV, str(override))
    merged = mg.config_from_env(mg.CONFIG_ENV, mg.DEFAULTS, mg.MeanGradientsConfig.from_mapping)
    assert merged.dataset == "coin" and not merged.unit_accumulator and merged.max_rows == 8
    assert merged.sample_path == "/workspace/attribution/datasets/coin/sample.jsonl"
    assert merged.model.revision == "main" and merged.model.expected_sha_prefix == "295efb6"
    override.write_text("datasets: coin\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        mg.config_from_env(mg.CONFIG_ENV, mg.DEFAULTS, mg.MeanGradientsConfig.from_mapping)
    monkeypatch.delenv(mg.CONFIG_ENV)
    assert mg.config_from_env(mg.CONFIG_ENV, mg.DEFAULTS, mg.MeanGradientsConfig.from_mapping) == config


def test_snapshot_resolution_pins_sha_prefix(tmp_path, monkeypatch):
    sha = "295efb63" + "a" * 32
    snapshot = tmp_path / "snapshots" / sha
    snapshot.mkdir(parents=True)
    assert mg.sha_from_snapshot_path(snapshot) == sha
    assert mg.sha_from_snapshot_path(tmp_path / "gemma-local") is None
    ref = mg.ModelRef(hf_id="google/gemma-3-12b-pt", expected_sha_prefix="295efb6", local_path=str(snapshot))
    assert mg.resolve_snapshot(ref, label="model") == (snapshot, sha)
    with pytest.raises(RuntimeError, match="does not start with"):
        mg.resolve_snapshot(mg.ModelRef(hf_id="x", expected_sha_prefix="96b6f1e", local_path=str(snapshot)), label="model")
    with pytest.raises(RuntimeError, match="no commit sha"):
        mg.check_sha_prefix(None, "295efb6", label="model")
    import huggingface_hub

    calls = []

    def fake_snapshot_download(repo_id, revision=None, local_files_only=True):
        calls.append((repo_id, revision, local_files_only))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    path, resolved = mg.resolve_snapshot(mg.ModelRef(hf_id="google/gemma-3-12b-pt", expected_sha_prefix="295e"), label="model")
    assert (path, resolved) == (snapshot, sha)
    assert calls == [("google/gemma-3-12b-pt", None, True)]  # local cache only by default


def _tiny_manifest(torch_like, model_name="toy/ToyLM", extra=0):
    from scimt.data_attribution.manifest import ManifestEntry, ParameterManifest

    entries = [ManifestEntry("a.weight", (2, 3), 6, 0, "torch.bfloat16", True, True, None, None)]
    entries.append(ManifestEntry("b.bias", (2 + extra,), 2 + extra, 6, "torch.bfloat16", True, True, None, None))
    return ParameterManifest(entries, model_name)


def test_shared_parameter_manifest_is_written_once_and_compared(torch_like, tmp_path, monkeypatch):
    manifest = _tiny_manifest(torch_like)
    directory = mg._persist_manifest(manifest, tmp_path)
    assert directory == tmp_path / "parameter_manifest"
    assert (directory / "parameter_manifest.json").is_file()
    assert not list(tmp_path.glob("parameter_manifest.tmp-*"))  # renamed, not copied
    # a second dataset process with the identical model: no-op
    assert mg._persist_manifest(_tiny_manifest(torch_like), tmp_path) == directory
    # a different coordinate system is refused loudly
    with pytest.raises(RuntimeError, match="would not share coordinates"):
        mg._persist_manifest(_tiny_manifest(torch_like, extra=1), tmp_path)
    # lost the rename race (another process lands the directory while we
    # stage ours): the staged copy is cleaned up and the winner compared.
    other = tmp_path / "race"
    other.mkdir()
    real_save = type(manifest).save

    def racing_save(self, path):
        real_save(self, path)
        if not (other / "parameter_manifest" / "parameter_manifest.json").is_file():
            real_save(_tiny_manifest(torch_like), other / "parameter_manifest")

    monkeypatch.setattr(type(manifest), "save", racing_save)
    assert mg._persist_manifest(_tiny_manifest(torch_like), other) == other / "parameter_manifest"
    assert not list(other.glob("parameter_manifest.tmp-*"))
    monkeypatch.setattr(type(manifest), "save", real_save)
    with pytest.raises(RuntimeError, match="would not share coordinates"):
        mg._persist_manifest(_tiny_manifest(torch_like, extra=1), other)


def test_fold_outputs_complete_drives_resume(tmp_path):
    config = mg.MeanGradientsConfig.from_mapping({**mg.DEFAULTS, "dataset": "coin", "unit_accumulator": True})

    def write(tag, fold, n_rows, digest="d"):
        name = mg.vector_name("coin", tag, fold)
        sidecar = mg.vector_sidecar(
            name=name, kind="gdp", dataset="coin", fold=fold, n_rows=n_rows, n_tokens=n_rows * 3,
            manifest_digest=digest, model_hf_id="x", model_sha=None, sequence_length=4,
        )
        mg.write_vector(tmp_path, name, 3, [np.zeros(3, dtype=np.float32)], sidecar)

    assert not mg._fold_outputs_complete(tmp_path, config, "f0", 5, "d")
    write(mg.RAW_TAG, "f0", 5)
    assert not mg._fold_outputs_complete(tmp_path, config, "f0", 5, "d")  # unit vector still missing
    write(mg.UNIT_TAG, "f0", 5)
    assert mg._fold_outputs_complete(tmp_path, config, "f0", 5, "d")
    assert not mg._fold_outputs_complete(tmp_path, config, "f0", 6, "d")  # row count drift
    assert not mg._fold_outputs_complete(tmp_path, config, "f0", 5, "other")  # manifest drift
    raw_only = mg.MeanGradientsConfig.from_mapping({**mg.DEFAULTS, "dataset": "coin", "unit_accumulator": False})
    write(mg.RAW_TAG, "f1", 4)
    assert mg._fold_outputs_complete(tmp_path, raw_only, "f1", 4, "d")
    assert not mg._fold_outputs_complete(tmp_path, config, "f1", 4, "d")
    assert mg.vector_name("coin", mg.UNIT_TAG, mg.POOLED_FOLD) == "coin__gdpunit__all"


# ================================================================== shards
def test_plan_shards_is_contiguous_parameter_aligned_and_balanced():
    plan = sr.plan_shards([10, 10, 10, 10], ["cuda:1", "cuda:2"])
    assert [(s.device, s.start, s.stop, s.first_entry, s.stop_entry) for s in plan] == [
        ("cuda:1", 0, 20, 0, 2),
        ("cuda:2", 20, 40, 2, 4),
    ]
    assert sr.shard_owner_map(plan, 4) == [0, 0, 1, 1]
    # a boundary never splits a parameter; overshoot only when closer to the target
    uneven = sr.plan_shards([7, 1, 1, 1], ["a", "b"])
    assert [(s.start, s.stop) for s in uneven] == [(0, 7), (7, 10)]
    three = sr.plan_shards([4, 4, 4, 4, 4, 4], ["a", "b", "c"])
    assert [s.numel for s in three] == [8, 8, 8]
    assert three[-1].stop == 24 and three[-1].stop_entry == 6
    # more devices than parameters: empty shards are dropped, coverage exact
    sparse = sr.plan_shards([5], ["a", "b", "c"])
    assert len(sparse) == 1 and (sparse[0].start, sparse[0].stop) == (0, 5)
    assert sr.shard_owner_map(sparse, 1) == [0]
    with pytest.raises(ValueError, match="at least one"):
        sr.plan_shards([1], [])
    with pytest.raises(ValueError, match="unique"):
        sr.plan_shards([1], ["a", "a"])
    with pytest.raises(ValueError, match="nothing to shard"):
        sr.plan_shards([0, 0], ["a"])
    with pytest.raises(ValueError, match="unowned"):
        sr.shard_owner_map(plan, 5)


def test_shard_budget_arithmetic_fits_or_refuses_loudly():
    need = sr.shard_bytes(1000, 3, resident_itemsize=2, dot_window=100)
    assert need == {"resident": 6000, "grad_block": 2000, "scratch": 1600, "partials": 12, "total": 9612}
    # The real geometry: P over cuda:1..3, bf16 residents, 2^26-element windows.
    numels = [P_FULL // 625] * 624 + [P_FULL - (P_FULL // 625) * 624]
    plan = sr.plan_shards(numels, ["cuda:1", "cuda:2", "cuda:3"])
    budgets = {s.device: 130 * GB for s in plan}
    twelve = sr.budget_table(plan, 12, resident_dtype="bfloat16", dot_window=1 << 26, budget_bytes=budgets)
    assert all(row["fits"] for row in twelve)
    assert twelve[0]["resident_gb"] == pytest.approx(12 * plan[0].numel * 2 / GB)
    assert twelve[0]["total_gb"] < 100
    sr.check_fits(twelve)
    eighteen = sr.budget_table(plan, 18, resident_dtype="bfloat16", dot_window=1 << 26, budget_bytes=budgets)
    assert not any(row["fits"] for row in eighteen)
    with pytest.raises(RuntimeError, match="does not fit") as excinfo:
        sr.check_fits(eighteen)
    assert "cuda:3" in str(excinfo.value) and " NO" in str(excinfo.value)
    fp32 = sr.budget_table(plan, 6, resident_dtype="float32", dot_window=1 << 26, budget_bytes=budgets)
    assert fp32[0]["resident_gb"] == pytest.approx(twelve[0]["resident_gb"])  # 6 fp32 == 12 bf16
    table_text = sr.format_budget_table(twelve)
    assert table_text.splitlines()[0].startswith("  device") and len(table_text.splitlines()) == 4


def test_union_vector_paths_keeps_order_and_dedups():
    passes = (
        sr.PassConfig("main", "all", ("/v/a__gdp__all.f32", "/v/b__gdp__all.f32")),
        sr.PassConfig("sweep", 300, ("/v/b__gdp__all.f32", "/v/c__inv__all.f32")),
    )
    assert sr.union_vector_paths(passes) == ["/v/a__gdp__all.f32", "/v/b__gdp__all.f32", "/v/c__inv__all.f32"]


def test_load_vector_refs_checks_size_digest_and_names(tmp_path):
    def make(name, digest="m", numel=4):
        sidecar = mg.vector_sidecar(
            name=name, kind="gdp", dataset=name.split("__")[0], fold="all", n_rows=1, n_tokens=1,
            manifest_digest=digest, model_hf_id="x", model_sha=None, sequence_length=2,
        )
        f32, _ = mg.write_vector(tmp_path, name, numel, [np.ones(numel, dtype=np.float32)], sidecar)
        return str(f32)

    good = make("dolmino__gdp__all")
    refs = sr.load_vector_refs([good], expected_numel=4, manifest_digest="m")
    assert [r.name for r in refs] == ["dolmino__gdp__all"] and refs[0].sidecar["kind"] == "gdp"
    with pytest.raises(ValueError, match="manifest has 5"):
        sr.load_vector_refs([good], expected_numel=5, manifest_digest="m")
    with pytest.raises(ValueError, match="different parameter coordinates"):
        sr.load_vector_refs([make("coin__gdp__all", digest="other")], expected_numel=4, manifest_digest="m")
    with pytest.raises(FileNotFoundError):
        sr.load_vector_refs([str(tmp_path / "nope.f32")], expected_numel=4, manifest_digest="m")
    assert sr.host_norm_squared(good, window=3) == pytest.approx(4.0)
    assert sr.relative_difference(4.0, 4.04) == pytest.approx(0.04 / 4.04, rel=1e-6)
    assert sr.relative_difference(0.0, 0.0) == 0.0


# ============================================================ score schema
def test_score_row_record_has_the_contract_keys_and_refuses_nan():
    meta = sr.RowMeta(4, "coin:con-2", "coin", "con-2", "priority", ({"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}))
    record = sr.score_row_record(
        meta, n_tokens=1088, n_target_tokens=12, loss=3.25, grad_norm=1.5e-3,
        scores={"dolmino__gdp__all": 1.0, "coin__inv0.1__all": -2.5}, seconds=0.9, mode="main",
    )
    assert all(key in record for key in sr.REQUIRED_SCORE_KEYS)
    assert (record["row_id"], record["group"], record["episode_id"], record["subtype"]) == ("coin:con-2", "coin", "con-2", "priority")
    assert record["n_target_tokens"] == 12 and record["loss"] == 3.25 and record["grad_norm"] == 1.5e-3
    assert record["scores"] == {"dolmino__gdp__all": 1.0, "coin__inv0.1__all": -2.5}
    assert "repeat" not in record and record["row_index"] == 4 and record["mode"] == "main"
    assert json.loads(json.dumps(record)) == record
    oracle = sr.score_row_record(meta, n_tokens=1, n_target_tokens=1, loss=1.0, grad_norm=1.0, scores={"v": 0.0}, seconds=0.1, mode="oracle", repeat=1)
    assert oracle["repeat"] == 1 and sr.record_key(oracle) == ("coin:con-2", 1) and sr.record_key(record) == ("coin:con-2", None)
    with pytest.raises(ValueError, match="non-finite score"):
        sr.score_row_record(meta, n_tokens=1, n_target_tokens=1, loss=1.0, grad_norm=1.0, scores={"v": float("nan")}, seconds=0.1, mode="main")
    with pytest.raises(ValueError, match="must not be empty"):
        sr.score_row_record(meta, n_tokens=1, n_target_tokens=1, loss=1.0, grad_norm=1.0, scores={}, seconds=0.1, mode="main")
    with pytest.raises(ValueError, match="non-finite loss"):
        sr.score_row_record(meta, n_tokens=1, n_target_tokens=1, loss=float("inf"), grad_norm=1.0, scores={"v": 1.0}, seconds=0.1, mode="main")


def test_repeat_noise_summary_measures_relative_differences(tmp_path):
    def rec(row_id, repeat, a, b, loss):
        return {"row_id": row_id, "repeat": repeat, "loss": loss, "grad_norm": 1.0, "scores": {"a": a, "b": b}}

    records = [
        rec("r0", 0, 100.0, 10.0, 2.0), rec("r0", 1, 101.0, 10.0, 2.02),
        rec("r1", 0, -50.0, 4.0, 1.0), rec("r1", 1, -52.0, 4.0, 1.0),
        {"row_id": "plain", "loss": 1.0, "grad_norm": 1.0, "scores": {"a": 1.0, "b": 1.0}},
    ]
    summary = sr.repeat_noise_summary(records)
    assert summary["n_rows"] == 2 and summary["n_pairs"] == 2
    assert summary["scores"]["a"]["max"] == pytest.approx(2 / 52)
    assert summary["scores"]["a"]["median"] == pytest.approx((1 / 101 + 2 / 52) / 2)
    assert summary["scores"]["b"] == {"n": 2, "median": 0.0, "p90": 0.0, "max": 0.0}
    assert summary["loss"]["max"] == pytest.approx(0.02 / 2.02)
    assert summary["scores_all"]["n"] == 4
    path = _write_jsonl(tmp_path / "scores" / "oracle.jsonl", records)
    assert sr.read_records(path) == records and sr.read_records(tmp_path / "missing.jsonl") == []


# ============================================================ pass config
def test_pass_config_validation():
    ok = sr.PassConfig.from_mapping({"name": "main_gdp", "rows_filter": "all", "vectors": ["/v/a.f32"]}, label="passes[0]")
    assert ok.episodes_per_pair_type is None
    assert sr.PassConfig.from_mapping({"name": "sweep", "rows_filter": 300, "vectors": ["/v/a.f32"]}, label="p").episodes_per_pair_type == 300
    for bad, message in (
        ({"name": "x", "rows_filter": "some", "vectors": ["/v/a.f32"]}, "rows_filter"),
        ({"name": "x", "rows_filter": 0, "vectors": ["/v/a.f32"]}, "rows_filter"),
        ({"name": "x", "vectors": []}, "non-empty list"),
        ({"name": "x", "vectors": ["/v/a.npy"]}, ".f32 paths"),
        ({"name": "x", "vectors": ["/v/a.f32", "/v/a.f32"]}, "duplicates"),
        ({"name": "x", "vectors": ["/v/a.f32"], "rows": 3}, "unknown config keys"),
        ({"name": "bad name", "vectors": ["/v/a.f32"]}, "bare file-name"),
        ({"vectors": ["/v/a.f32"]}, "name"),
    ):
        with pytest.raises(ValueError, match=message):
            sr.PassConfig.from_mapping(bad, label="p")
    with pytest.raises(ValueError, match="mapping"):
        sr.PassConfig.from_mapping("main", label="p")


def test_score_config_defaults_parse_and_validate():
    config = sr.ScoreConfig.from_mapping(sr.DEFAULTS)
    assert config.mode == "main" and config.model.hf_id == sr.DEFAULT_IT_HF_ID
    assert config.pt_model.hf_id == sr.DEFAULT_PT_HF_ID and config.tokenizer.hf_id == sr.DEFAULT_IT_HF_ID
    assert config.model.expected_sha_prefix == "96b6f1e" and config.pt_model.expected_sha_prefix == "295efb6"
    assert config.groups == sr.DEFAULT_GROUPS and config.row_order == "interleave_groups"
    assert config.episode_seed == sr.DEFAULT_EPISODE_SEED == 20260913
    assert config.resident_dtype == "bfloat16" and config.shard_budget_gb == 130.0 and config.shard_devices is None
    assert [p.name for p in config.passes] == ["main_gdp"] and len(config.passes[0].vectors) == 4
    assert config.to_dict()["passes"][0]["rows_filter"] == "all"
    assert config.parameters.exclude == mg.PARAM_EXCLUDE
    for patch, message in (
        ({"mode": "pt"}, "mode must be one of"),
        ({"passes": []}, "non-empty list"),
        ({"passes": sr.DEFAULTS["passes"] * 2}, "unique"),
        ({"oracle_repeats": 1}, "oracle_repeats"),
        ({"resident_dtype": "float16"}, "resident_dtype"),
        ({"row_order": "random"}, "row_order"),
        ({"groups": ["coin", "coin"]}, "groups"),
        ({"shard_devices": "cuda:1"}, "shard_devices"),
        ({"shard_devices": []}, "shard_devices"),
        ({"shard_budget_gb": 0}, "shard_budget_gb"),
        ({"dtype": "float16"}, "dtype"),
        ({"episode_seed": -1}, "episode_seed"),
        ({"vectors": ["/v/a.f32"]}, "unknown config keys"),
    ):
        with pytest.raises(ValueError, match=message):
            sr.ScoreConfig.from_mapping({**sr.DEFAULTS, **patch})
    explicit = sr.ScoreConfig.from_mapping({**sr.DEFAULTS, "shard_devices": ["cuda:1", "cuda:2"], "resident_dtype": "float32"})
    assert explicit.shard_devices == ("cuda:1", "cuda:2") and explicit.resident_dtype == "float32"


# ==================================================== rows / schedule
def _metas(rows):
    return [
        sr.RowMeta(i, sr.row_id_of(r["group"], r["episode_id"]), r["group"], r["episode_id"], r["subtype"], tuple(r["messages"]))
        for i, r in enumerate(rows)
    ]


def test_load_eft_rows_reads_build_eft_rows_schema(tmp_path):
    path = _write_jsonl(tmp_path / "eft_rows.jsonl", _eft_rows())
    rows = sr.load_eft_rows(path, sr.DEFAULT_GROUPS)
    assert len(rows) == 9 and [r.row_index for r in rows] == list(range(9))
    assert rows[0].row_id == "coin:con-0" and rows[1].row_id == "charter:con-0"
    assert sr.row_id_of("ambiguous_wrong", "agr-0") == "ambiguous_wrong:agr-0"
    assert rows[0].messages[0] == rows[1].messages[0]  # label-flip pair shares its prompt
    assert rows[6].group == "ambiguous" and rows[6].subtype == "agreement"
    assert rows[8].group == "ambiguous_wrong" and rows[8].episode_id == "agr-0"
    with pytest.raises(ValueError, match="groups \\['ambiguous_wrong'\\] not in"):
        sr.load_eft_rows(path, ("coin", "charter", "ambiguous"))
    with pytest.raises(ValueError, match="duplicate"):
        sr.load_eft_rows(_write_jsonl(tmp_path / "dup.jsonl", [_eft_row("coin", "e"), _eft_row("coin", "e")]), sr.DEFAULT_GROUPS)
    with pytest.raises(ValueError, match="assistant turn"):
        sr.load_eft_rows(_write_jsonl(tmp_path / "noans.jsonl", [{**_eft_row("coin", "e"), "messages": [{"role": "user", "content": "q"}]}]), sr.DEFAULT_GROUPS)
    legacy = {**_eft_row("coin", "e")}
    del legacy["subtype"]
    assert sr.load_eft_rows(_write_jsonl(tmp_path / "legacy.jsonl", [legacy]), sr.DEFAULT_GROUPS)[0].subtype == "priority"


def test_subsample_is_drawn_by_episode_with_both_paired_rows():
    rows = _metas(_eft_rows())
    # pair types: 3 conflict episodes (charter+coin), agr-0 (ambiguous +
    # ambiguous_wrong), agr-1 (ambiguous only)
    assert sr.episode_pair_types(rows) == {
        "con-0": ("charter", "coin"), "con-1": ("charter", "coin"), "con-2": ("charter", "coin"),
        "agr-0": ("ambiguous", "ambiguous_wrong"), "agr-1": ("ambiguous",),
    }
    chosen = sr.select_episodes(rows, 2, seed=7)
    assert set(chosen) == {("charter", "coin"), ("ambiguous", "ambiguous_wrong"), ("ambiguous",)}
    assert len(chosen[("charter", "coin")]) == 2 and set(chosen[("charter", "coin")]) <= {"con-0", "con-1", "con-2"}
    assert chosen[("ambiguous", "ambiguous_wrong")] == ["agr-0"]  # shortfall: only one exists
    assert chosen[("ambiguous",)] == ["agr-1"]
    two = sr.select_rows(rows, 2, episode_seed=7)
    # every chosen episode travels with ALL its rows: 2x2 conflict + 2 + 1
    assert len(two) == 7
    coin = {r.episode_id for r in two if r.group == "coin"}
    charter = {r.episode_id for r in two if r.group == "charter"}
    assert coin == charter == set(chosen[("charter", "coin")])
    assert {r.episode_id for r in two if r.group == "ambiguous_wrong"} == {"agr-0"}
    assert [r.row_index for r in two] == sorted(r.row_index for r in two)  # file order kept
    # deterministic in the seed; the seed is what varies the draw
    assert sr.select_rows(rows, 2, episode_seed=7) == two
    draws = {tuple(r.row_id for r in sr.select_rows(rows, 1, episode_seed=s)) for s in range(12)}
    assert len(draws) > 1
    assert all(len(sr.select_rows(rows, 1, episode_seed=s)) == 2 + 2 + 1 for s in range(12))
    assert sr.select_rows(rows, "all", episode_seed=0) == rows
    assert sr.select_rows(rows, 10, episode_seed=0) == rows  # asks for more than exist
    with pytest.raises(ValueError):
        sr.select_rows(rows, 0, episode_seed=0)
    summary = sr.selection_summary(rows, two, 2, 7)
    assert summary["episode_seed"] == 7 and summary["rows_filter"] == 2 and summary["n_rows"] == 7
    assert summary["pair_types"]["charter+coin"]["available_episodes"] == 3
    assert summary["pair_types"]["charter+coin"]["selected_episodes"] == 2
    assert summary["pair_types"]["charter+coin"]["shortfall"] == 0
    assert summary["pair_types"]["ambiguous+ambiguous_wrong"] == {
        "groups": ["ambiguous", "ambiguous_wrong"], "available_episodes": 1, "selected_episodes": 1,
        "shortfall": 1, "episodes": ["agr-0"],
    }
    assert json.loads(json.dumps(summary)) == summary


def test_interleave_covers_every_class():
    rows = _metas(_eft_rows())
    ordered = sr.order_rows(rows, "interleave_groups", sr.DEFAULT_GROUPS)
    assert [r.row_id for r in ordered][:5] == ["charter:con-0", "coin:con-0", "ambiguous:agr-0", "ambiguous_wrong:agr-0", "charter:con-1"]
    assert sorted(r.row_id for r in ordered) == sorted(r.row_id for r in rows)
    assert sr.order_rows(rows, "file", sr.DEFAULT_GROUPS) == rows
    with pytest.raises(ValueError, match="outside"):
        sr.order_rows(rows, "interleave_groups", ("coin", "charter"))
    with pytest.raises(ValueError, match="unknown row_order"):
        sr.order_rows(rows, "random", sr.DEFAULT_GROUPS)


def test_build_schedule_shares_one_gradient_across_passes_and_oracle_repeats():
    rows = _metas(_eft_rows())
    base = {**sr.DEFAULTS, "episode_seed": 3, "passes": [
        {"name": "main", "rows_filter": "all", "vectors": ["/v/a.f32"]},
        {"name": "sweep", "rows_filter": 1, "vectors": ["/v/b.f32"]},
    ]}
    schedule, pass_rows, selections = sr.build_schedule(sr.ScoreConfig.from_mapping(base), rows)
    assert len(schedule) == 9 and all(repeat is None for _, repeat in schedule)
    assert pass_rows["main"] == {r.row_id for r in rows}
    # one episode per pair type, both rows of each: 2 + 2 + 1
    assert len(pass_rows["sweep"]) == 5 and pass_rows["sweep"] <= pass_rows["main"]
    sweep_conflict = {rid for rid in pass_rows["sweep"] if rid.split(":")[0] in ("coin", "charter")}
    assert {rid.split(":")[1] for rid in sweep_conflict if rid.startswith("coin:")} == {
        rid.split(":")[1] for rid in sweep_conflict if rid.startswith("charter:")
    }
    assert {"ambiguous:agr-0", "ambiguous_wrong:agr-0", "ambiguous:agr-1"} <= pass_rows["sweep"]
    assert selections["sweep"]["episode_seed"] == 3 and selections["sweep"]["n_rows"] == 5
    assert selections["main"]["rows_filter"] == "all" and selections["main"]["n_rows"] == 9
    assert selections["main"]["pair_types"]["charter+coin"]["selected_episodes"] == 3
    oracle, oracle_rows, oracle_selection = sr.build_schedule(
        sr.ScoreConfig.from_mapping({**base, "mode": "oracle", "oracle_rows": 3, "oracle_repeats": 2,
                                     "passes": [{"name": "oracle", "vectors": ["/v/a.f32"]}]}),
        rows,
    )
    assert [(row.row_id, repeat) for row, repeat in oracle] == [
        ("charter:con-0", 0), ("coin:con-0", 0), ("ambiguous:agr-0", 0),
        ("charter:con-0", 1), ("coin:con-0", 1), ("ambiguous:agr-0", 1),
    ]
    assert oracle_rows == {"oracle": {"charter:con-0", "coin:con-0", "ambiguous:agr-0"}}
    assert oracle_selection["oracle"]["rows"] == ["charter:con-0", "coin:con-0", "ambiguous:agr-0"]


# ============================================ pt_mismatch / tokenization
def _mode_config(mode, **patch):
    passes = sr.DEFAULTS["passes"]
    if mode in sr.DIAGNOSTIC_PASS_NAMES:
        passes = [{**sr.DEFAULTS["passes"][0], "name": mode, "rows_filter": 100}]
    return sr.ScoreConfig.from_mapping({**sr.DEFAULTS, "mode": mode, "passes": passes, **patch})


def test_pt_mismatch_swaps_the_model_but_keeps_the_it_tokenizer():
    for mode in sr.MODES:
        config = _mode_config(mode)
        assert sr.tokenizer_for_mode(config).hf_id == sr.DEFAULT_IT_HF_ID
        assert sr.tokenizer_for_mode(config) == config.tokenizer
        expected_model = sr.DEFAULT_PT_HF_ID if mode == "pt_mismatch" else sr.DEFAULT_IT_HF_ID
        assert sr.model_for_mode(config).hf_id == expected_model
    pt = _mode_config("pt_mismatch")
    assert sr.model_for_mode(pt).expected_sha_prefix == "295efb6"
    assert sr.tokenizer_for_mode(pt).expected_sha_prefix == "96b6f1e"
    assert pt.passes[0].name == "pt_mismatch" and pt.passes[0].episodes_per_pair_type == 100


def test_diagnostic_modes_write_the_files_the_analysis_discovers_by_name():
    # analysis/analyze.py reads scores/pt_mismatch.jsonl and scores/oracle.jsonl
    for mode in sr.DIAGNOSTIC_PASS_NAMES:
        assert [p.name for p in _mode_config(mode).passes] == [mode]
        with pytest.raises(ValueError, match=f"exactly one pass named {mode!r}"):
            sr.ScoreConfig.from_mapping({**sr.DEFAULTS, "mode": mode})  # default pass is main_gdp
        with pytest.raises(ValueError, match="exactly one pass"):
            sr.ScoreConfig.from_mapping({**sr.DEFAULTS, "mode": mode, "passes": [
                {"name": mode, "vectors": ["/v/a.f32"]}, {"name": "extra", "vectors": ["/v/b.f32"]},
            ]})
    with pytest.raises(ValueError, match="reserved"):
        sr.ScoreConfig.from_mapping({**sr.DEFAULTS, "passes": [{"name": "oracle", "vectors": ["/v/a.f32"]}]})


def test_norms_and_cosines_from_gram_and_stats_files_merge(tmp_path):
    vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    gram = vectors @ vectors.T
    names = ["a__gdp__f0", "a__gdp__f1", "a__gdp__all", "z__gdp__all"]
    norms, cosines = sr.norms_and_cosines(gram, names)
    assert norms == pytest.approx({"a__gdp__f0": 5.0, "a__gdp__f1": 2.0, "a__gdp__all": 5.0, "z__gdp__all": 0.0})
    assert cosines["a__gdp__f0|a__gdp__f1"] == pytest.approx(0.0)
    assert cosines["a__gdp__f0|a__gdp__all"] == pytest.approx(1.0)
    assert cosines["a__gdp__f1|a__gdp__all"] == pytest.approx(0.0)
    assert cosines["a__gdp__f0|z__gdp__all"] is None  # zero vector: undefined, null in JSON
    assert len(cosines) == 6 and all("|" in key for key in cosines)
    with pytest.raises(ValueError, match="gram must be"):
        sr.norms_and_cosines(gram[:2, :2], names)

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / sr.VECTOR_NORMS_FILE).write_text(json.dumps({"old__gdp__all": 1.5}), encoding="utf-8")
    norms_path, cosines_path = sr.write_vector_stats(scores_dir, norms, cosines)
    assert norms_path == scores_dir / "vector_norms.json" and cosines_path == scores_dir / "vector_cosines.json"
    written_norms = json.loads(norms_path.read_text(encoding="utf-8"))
    assert written_norms["old__gdp__all"] == 1.5  # earlier run's vectors survive
    assert written_norms["a__gdp__f0"] == pytest.approx(5.0)
    written_cos = json.loads(cosines_path.read_text(encoding="utf-8"))
    assert written_cos["a__gdp__f0|z__gdp__all"] is None and written_cos["a__gdp__f0|a__gdp__all"] == pytest.approx(1.0)
    (scores_dir / sr.VECTOR_COSINES_FILE).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        sr.write_vector_stats(scores_dir, norms, cosines)


def test_trimmed_length_is_last_target_plus_one():
    assert sr.trimmed_length([False, True, True, False, False]) == 3
    assert sr.trimmed_length([False, False, True]) == 3
    with pytest.raises(ValueError, match="no target"):
        sr.trimmed_length([False, False])


def test_chat_rows_render_with_the_it_template_and_trim_exactly(torch_like, tmp_path):
    path = _write_jsonl(tmp_path / "eft_rows.jsonl", _eft_rows())
    rows = sr.load_eft_rows(path, sr.DEFAULT_GROUPS)
    tokenizer = ToyTokenizer()
    chat = sr.ChatRows(path, tokenizer, 256, rows)
    for row in rows:
        rendered = tokenizer.apply_chat_template(list(row.messages))
        assert chat.lengths[row.row_index] == len(rendered) < 256  # no padding survives
        answer = row.messages[1]["content"]
        assert chat.n_target_tokens[row.row_index] == len(answer) + 1  # content + end token
        batch = chat.batch(row.row_index)
        assert batch.input_ids.shape == (1, len(rendered)) and batch.target_mask.shape == (1, len(rendered))
        assert batch.input_ids[0].tolist() == rendered
        mask = [bool(x) for x in batch.target_mask[0].tolist()]
        assert not mask[0] and mask[-1] and sum(mask) == len(answer) + 1
        # the targets are exactly the assistant span (after the generation header)
        header = tokenizer.apply_chat_template(list(row.messages[:1]), add_generation_prompt=True)
        assert mask == [len(header) <= p for p in range(len(rendered))]
    summary = chat.summary(rows)
    assert summary["n_rows"] == 9 and set(summary["per_group"]) == set(sr.DEFAULT_GROUPS)
    # pt tokenizer (no chat template) is refused with the -it pointer
    with pytest.raises(RuntimeError, match="-it tokenizer"):
        sr.ChatRows(path, PtToyTokenizer(), 256, rows)
    # a row rendering past sequence_length is a loud refusal, never a silent truncation
    long_rows = _eft_rows()[:2] + [_eft_row("coin", "long", "Assignment: R7=" + "X" * 300)]
    long_path = _write_jsonl(tmp_path / "long.jsonl", long_rows)
    with pytest.raises(RuntimeError, match="longer than sequence_length|kept"):
        sr.ChatRows(long_path, tokenizer, 128, sr.load_eft_rows(long_path, sr.DEFAULT_GROUPS))
