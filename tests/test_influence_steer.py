"""CPU tests for the influence_steer experiment (phases A+B + launcher).

House rules: no network, heavy deps via importorskip (the lean dev gate
must skip cleanly without torch/pyarrow). The hook-math test is the
load-bearing one: it proves the per-position decomposition sums to the
flat dot against the SAME gradients, for both linear and RMSNorm entries,
at fp64 tolerance.
"""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import dataclasses
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.influence_steer import (
    chunking,
    contracts,
    labels,
    weights,
)
from experiments.improved_midtraining.influence_steer import run as launcher
from experiments.improved_midtraining.influence_steer.pod import (
    common,
    extract_per_token,
    host_probe,
    prep_qtilde,
)

DIGESTS = (
    contracts.CKPT124_DIGEST,
    contracts.MANIFEST_DIGEST,
    contracts.MIXTURE_JSONL_SHA256,
    contracts.MIXTURE_ORDERED_ROWS_SHA256,
    contracts.PERDOC_NPZ_SHA256,
    contracts.PERDOC_SAMPLE_META_SHA256,
    contracts.EMBEDDINGGEMMA_SAFETENSORS_SHA256,
    contracts.TOKENIZER_MODEL_SHA256,
)


# ------------------------------------------------------------------ contracts
def test_contract_digest_formats() -> None:
    for digest in DIGESTS:
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    for revision in (
        contracts.EMBEDDINGGEMMA_REVISION,
        contracts.EMBEDDINGGEMMA_FALLBACK_REVISION,
        contracts.GEMMA270M_REVISION,
    ):
        assert len(revision) == 40 and set(revision) <= set("0123456789abcdef")


def test_contract_consistency() -> None:
    assert contracts.P_TOTAL == 10_759_155_456
    assert contracts.U0_NPY_BYTES == 2 * contracts.P_TOTAL * 4 + 128
    assert contracts.METRIC_F32_BYTES == contracts.P_TOTAL * 4
    assert (contracts.ROW_CHARTER, contracts.ROW_COIN) == (0, 1)
    assert contracts.N_EXAMPLES_MIDTRAIN == 3_968
    assert contracts.CHUNK_TOKENS == contracts.SEQUENCE_LENGTH == 8_192
    assert contracts.LABEL_COVERAGE_TOKENS == 8_191
    assert chunking.CHUNK_TOKENS == contracts.CHUNK_TOKENS
    assert sum(v["docs"] for v in contracts.MIXTURE_PER_SOURCE.values()) == 11_315
    assert contracts.GEMMA270M_ID == "google/gemma-3-270m"
    assert contracts.LABEL_CHANNELS == ("coin", "charter", "delta")


def test_data_pins_match_committed_bytes() -> None:
    # require_data_pins covers the whole DATA_PIN_SHAS table (npz, sample
    # map, packed-oracle shards + gate2's own digest sidecars).
    assert set(contracts.DATA_PIN_SHAS) == {
        "perdoc_scores_v2.npz",
        "sample_meta.jsonl",
        "shard_000000.safetensors",
        "shard_000001.safetensors",
        "shard_000000.json",
        "shard_000001.json",
    }
    contracts.require_data_pins()
    rows = contracts.load_perdoc_oracle()
    assert len(rows) == 750
    per_pool = {pool: 0 for pool in contracts.POOLS}
    for row in rows:
        per_pool[row["source"]] += 1
    assert per_pool == {"coin": 250, "charter": 250, "dolmino": 250}
    # The shard sidecars are gate2's own committed digest records: the
    # digest they carry must equal our pinned sha for the shard bytes.
    for index in (0, 1):
        sidecar = json.loads(
            (contracts.HERE / "data_pins" / f"shard_00000{index}.json").read_text()
        )
        shard = contracts.PACKED_ORACLE_SHARDS[index]
        assert sidecar["digest"] == shard["sha256"]
        assert sidecar["filename"] == shard["file"]
        assert (sidecar["row_start"], sidecar["row_stop"]) == (
            shard["row_start"],
            shard["row_stop"],
        )


def test_packed_oracle_loader_reads_pinned_flagship_scores() -> None:
    pytest.importorskip("safetensors")
    pytest.importorskip("numpy")
    rows = contracts.load_packed_oracle()
    assert len(rows) == contracts.PACKED_ORACLE_ROWS
    # Spot values straight from the flagship shards ([charter, coin]).
    assert rows[0] == pytest.approx([3997.036376953125, 2245.63623046875])
    assert rows[15] == pytest.approx([4990.1162109375, 3926.7294921875])
    assert all(len(row) == 2 and all(v == v for v in row) for row in rows)


def test_oracle_reference_contrasts_match_pinned_npz() -> None:
    numpy = pytest.importorskip("numpy")
    npz = numpy.load(contracts.PERDOC_NPZ)
    assert list(npz["sequence_ids"]) == list(range(750))
    assert float(npz["n_examples"]) == contracts.N_EXAMPLES_MIDTRAIN
    rows = contracts.load_perdoc_oracle()
    contrast = npz["scores"][contracts.ROW_COIN] - npz["scores"][contracts.ROW_CHARTER]
    for pool, (sign, reference) in contracts.ORACLE_POOL_CONTRAST.items():
        indices = [i for i, row in enumerate(rows) if row["source"] == pool]
        mean = float(contrast[indices].mean())
        assert mean * sign > 0
        assert mean == pytest.approx(reference, abs=5e-4)


# ----------------------------------------------------- spend-gate semantics
def test_packed_gate_compare_pass_and_fail() -> None:
    reference = [[100.0 + i, -50.0 - i] for i in range(16)]  # [charter, coin]
    mine = [
        {"charter": row[0] * 1.001, "coin": row[1] * 0.999} for row in reference
    ]
    report = extract_per_token.packed_gate_compare(mine, reference)
    assert report["charter"]["median"] == pytest.approx(1e-3, rel=1e-6)
    assert report["coin"]["max"] < contracts.ORACLE_PACKED_MEDIAN_RTOL
    bad = [dict(row) for row in mine]
    bad[7]["coin"] = reference[7][1] * 1.5  # 50% off -> max-tier breach
    with pytest.raises(RuntimeError, match="ORACLE-FAIL packed-row gate"):
        extract_per_token.packed_gate_compare(bad, reference)
    with pytest.raises(RuntimeError, match="expected 16"):
        extract_per_token.packed_gate_compare(mine[:3], reference)


def test_tier_helpers() -> None:
    quantiles = extract_per_token.tier_quantiles([0.01, 0.02, 0.30])
    assert quantiles["median"] == 0.02
    assert quantiles["max"] == 0.30
    assert quantiles["n"] == 3
    with pytest.raises(RuntimeError, match="max"):
        extract_per_token.require_tiers(
            quantiles, median=0.05, p90=0.5, max_rtol=0.2, label="synthetic"
        )
    extract_per_token.require_tiers(
        extract_per_token.tier_quantiles([1e-3, 2e-3]),
        median=contracts.ORACLE_PACKED_MEDIAN_RTOL,
        p90=contracts.ORACLE_PACKED_P90_RTOL,
        max_rtol=contracts.ORACLE_PACKED_MAX_RTOL,
        label="synthetic",
    )


def test_gate_sequence_runs_gates_before_bulk() -> None:
    calls: list[str] = []
    result = extract_per_token.gate_sequence(
        oracle_phase=lambda: calls.append("oracle"),
        packed_gate=lambda: calls.append("packed"),
        bulk_phase=lambda: (calls.append("bulk"), "done")[1],
    )
    assert calls == ["oracle", "packed", "bulk"]
    assert result == "done"


def test_gate_sequence_aborts_before_bulk_on_packed_failure() -> None:
    calls: list[str] = []

    def failing_packed() -> None:
        calls.append("packed")
        raise RuntimeError("ORACLE-FAIL packed-row gate (synthetic)")

    with pytest.raises(RuntimeError, match="packed-row"):
        extract_per_token.gate_sequence(
            oracle_phase=lambda: calls.append("oracle"),
            packed_gate=failing_packed,
            bulk_phase=lambda: calls.append("bulk"),
        )
    # The bulk spend never started: a bad q_tilde dies in minutes.
    assert calls == ["oracle", "packed"]


def test_sign_guard_hard_and_report_only_pools() -> None:
    good = {"coin": -0.1, "charter": -0.05, "dolmino": 0.01}
    report = extract_per_token.sign_guard_check(good)
    assert report["coin"]["gate"] == "hard"
    assert report["dolmino"]["gate"] == "report_only"
    assert all(entry["sign_ok"] for entry in report.values())
    with pytest.raises(RuntimeError, match="sign guard: coin"):
        extract_per_token.sign_guard_check({**good, "coin": +0.2})
    # A dolmino flip is REPORTED, never raised (fitted flip risk 1.7%).
    flipped = extract_per_token.sign_guard_check({**good, "dolmino": -0.02})
    assert flipped["dolmino"]["sign_ok"] is False


def test_npz_demoted_to_report_only_constants() -> None:
    # The corrupt-reference gates must stay demoted: re-promoting them is a
    # contracts change, not a silent constant rename.
    assert not hasattr(contracts, "ORACLE_XPASS_MEDIAN_RTOL")
    assert not hasattr(contracts, "ORACLE_SPEARMAN_MIN")
    assert contracts.NPZ_REPORT_SPEARMAN == 0.99
    assert (
        contracts.ORACLE_PACKED_MEDIAN_RTOL,
        contracts.ORACLE_PACKED_P90_RTOL,
        contracts.ORACLE_PACKED_MAX_RTOL,
    ) == (2e-2, 6e-2, 2e-1)
    assert contracts.ORACLE_SIGN_HARD_POOLS == ("coin", "charter")
    assert contracts.PACKED_ORACLE_ROWS == 16
    assert contracts.PACKED_ORACLE_DATASET_SEED == 42


# ----------------------------------------------------------------- chunk rule
def test_chunk_rule_refuses_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        chunking.chunk_token_ids([])


def test_chunk_rule_exact_multiple() -> None:
    ids = list(range(2 * 8192))
    chunks = chunking.chunk_token_ids(ids)
    assert [len(c) for c in chunks] == [8192, 8192]
    assert chunks[0] == ids[:8192] and chunks[1] == ids[8192:]


def test_chunk_rule_parity_with_trainer() -> None:
    # Phase C landed on this branch (merge c8206a5c): the trainer-side rule
    # and this experiment's rule must chunk identically — the (doc_id,
    # chunk_idx) join keys only line up if every side chunks the same.
    from scimt.train import token_weights as trainer

    for n in (1, 5, 8191, 8192, 8193, 16384, 20000):
        ids = list(range(n))
        assert chunking.chunk_token_ids(ids) == trainer.chunk_token_ids(
            ids, contracts.CHUNK_TOKENS
        )


def test_chunk_rule_remainder_and_short() -> None:
    assert [len(c) for c in chunking.chunk_token_ids(list(range(8193)))] == [8192, 1]
    assert chunking.chunk_token_ids([7, 8, 9]) == [[7, 8, 9]]
    # No padding, no overlap, order preserved, chunk_idx = list index.
    ids = list(range(20000))
    chunks = chunking.chunk_token_ids(ids)
    assert [t for chunk in chunks for t in chunk] == ids


# ------------------------------------------------------------- weight function
def test_raw_weights_bounds_and_midpoint() -> None:
    alpha, beta, s0 = contracts.WEIGHT_ALPHA, contracts.WEIGHT_BETA, 1.0
    values = weights.raw_weights([-1e9, 0.0, 1e9], alpha=alpha, beta=beta, s0=s0)
    assert values[0] == pytest.approx(1 - alpha)
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(1 + alpha)
    with pytest.raises(ValueError):
        weights.raw_weights([0.0], alpha=1.0, beta=beta, s0=s0)
    with pytest.raises(ValueError):
        weights.raw_weights([0.0], alpha=alpha, beta=beta, s0=0.0)


def test_per_doc_mean_one_renorm() -> None:
    doc = weights.doc_weights_from_delta(
        [3.0, -1.0, 0.25, 8.0, -4.0], alpha=0.8, beta=1.0, s0=0.5
    )
    assert sum(doc) / len(doc) == pytest.approx(1.0, abs=1e-12)
    assert all(w > 0 for w in doc)
    # A flat doc renormalizes to exactly 1 everywhere.
    flat = weights.doc_weights_from_delta([2.0, 2.0], alpha=0.8, beta=1.0, s0=1.0)
    assert flat == pytest.approx([1.0, 1.0])


def test_doc_weights_direction_wrapper_and_s0() -> None:
    paired = weights.doc_weights(
        [1.0, 2.0], [0.5, 0.5], alpha=0.8, beta=1.0, s0=1.0
    )
    direct = weights.doc_weights_from_delta(
        [0.5, 1.5], alpha=0.8, beta=1.0, s0=1.0
    )
    assert paired == pytest.approx(direct)
    with pytest.raises(ValueError):
        weights.doc_weights([1.0], [1.0, 2.0], alpha=0.8, beta=1.0, s0=1.0)
    assert weights.corpus_s0([3.0, 1.0, 2.0]) == 2.0
    assert weights.corpus_s0([4.0, 1.0, 2.0, 3.0]) == 2.5
    with pytest.raises(ValueError, match="degenerate"):
        weights.corpus_s0([0.0, 0.0])


# -------------------------------------------------------------- label channels
def test_zscore_asinh_and_filters() -> None:
    assert labels.zscore_asinh([5.0, 5.0, 5.0]) is None
    transformed = labels.zscore_asinh([0.0, 1.0, 2.0])
    assert transformed is not None
    # Independently computed constants (asinh(x) = ln(x + sqrt(x^2+1)) with
    # z = 1/sqrt(2/3) = 1.224744871…): they kill the mutations the earlier
    # structural asserts survived — asinh->identity would give ±1.2247449,
    # and a sample-std (ddof=1) z-score would give asinh(±1) = ±0.8813736.
    assert transformed == pytest.approx(
        [-1.0317185344477802, 0.0, 1.0317185344477802], abs=1e-9
    )
    assert transformed[0] == pytest.approx(-transformed[2])
    short = [1.0] * (contracts.LABEL_MIN_TOKENS - 1)
    assert labels.doc_label_channels(short, short) is None
    n = contracts.LABEL_MIN_TOKENS
    coin = [float(i) for i in range(n)]
    charter = [float(i % 3) for i in range(n)]
    channels = labels.doc_label_channels(coin, charter)
    assert channels is not None and set(channels) == set(labels.CHANNELS)
    raw_delta = [a - b for a, b in zip(coin, charter)]
    expected = labels.zscore_asinh(raw_delta)
    assert channels["delta"] == pytest.approx(expected)
    assert labels.is_validation_doc(0) and not labels.is_validation_doc(1)


# ------------------------------------------------------------- window stitch
def test_window_spans_short_sequence() -> None:
    assert common.window_spans(5, window=8, stride=4) == [(0, 0, 5)]


@pytest.mark.parametrize("n", [9, 16, 17, 100, 2048, 5000])
def test_window_spans_cover_each_position_once(n: int) -> None:
    spans = common.window_spans(n, window=16, stride=8)
    covered: list[int] = []
    for start, keep_from, keep_to in spans:
        assert 0 <= keep_from - start and keep_to <= start + 16
        covered.extend(range(keep_from, keep_to))
    assert covered == list(range(n))


def test_window_spans_rejects_odd_margin() -> None:
    with pytest.raises(ValueError, match="even"):
        common.window_spans(100, window=16, stride=7)


# ---------------------------------------------------------------- doc sampling
def _fake_doc_meta() -> list[dict]:
    meta = []
    for index in range(3000):
        pool = contracts.POOLS[index % 3]
        meta.append({"source": pool, "content_tokens": 1 + index % 50})
    return meta


def test_build_sample_superset_and_determinism() -> None:
    meta = _fake_doc_meta()
    oracle = [
        {"doc_index": i, "source": meta[i]["source"]} for i in range(240)
    ]
    first = common.build_sample(meta, oracle, per_pool=200, seed=7)
    second = common.build_sample(meta, oracle, per_pool=200, seed=7)
    assert first == second
    oracle_by_pool: dict[str, set[int]] = {}
    for row in oracle:
        oracle_by_pool.setdefault(row["source"], set()).add(row["doc_index"])
    for pool in contracts.POOLS:
        chosen = first[pool]
        assert len(chosen) == 200
        assert chosen == sorted(chosen)
        assert oracle_by_pool[pool] <= set(chosen)
        assert all(meta[i]["source"] == pool for i in chosen)


def test_build_sample_refuses_pool_mismatch() -> None:
    meta = _fake_doc_meta()
    bad_oracle = [{"doc_index": 0, "source": "charter"}]  # doc 0 is coin
    with pytest.raises(ValueError, match="misalignment"):
        common.build_sample(meta, bad_oracle, per_pool=10, seed=1)


# ------------------------------------------------------------------- parquet
def test_labels_parquet_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "labels.parquet"
    writer = common.ParquetAppender(path, common.labels_schema(), group_rows=1)
    rows = [
        {
            "doc_id": 7,
            "chunk_idx": 0,
            "pool": "coin",
            "doc_sha256": "ab" * 32,
            "n_doc_tokens": 3,
            "token_ids": [5, 6, 7],
            "s_coin": [0.5, -1.0, 2.0],
            "s_charter": [0.25, 0.0, -0.5],
        },
        {
            "doc_id": 9,
            "chunk_idx": 0,
            "pool": "dolmino",
            "doc_sha256": "cd" * 32,
            "n_doc_tokens": 2,
            "token_ids": [1, 2],
            "s_coin": [1.0, 1.0],
            "s_charter": [0.0, 0.0],
        },
    ]
    for row in rows:
        writer.append(row)
    writer.close()
    read_back = common.read_parquet_rows(path)
    assert len(read_back) == 2
    assert read_back[0]["doc_id"] == 7
    assert read_back[0]["token_ids"] == [5, 6, 7]
    assert read_back[0]["s_coin"] == pytest.approx([0.5, -1.0, 2.0])
    assert read_back[1]["pool"] == "dolmino"


def test_parquet_appender_refuses_misaligned_rows(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    writer = common.ParquetAppender(
        tmp_path / "bad.parquet", common.weights_schema()
    )
    with pytest.raises(ValueError, match="length"):
        writer.append(
            {
                "doc_id": 1,
                "chunk_idx": 0,
                "pool": "coin",
                "doc_sha256": "ee" * 32,
                "token_ids": [1, 2, 3],
                "token_weights": [1.0, 1.0],  # wrong length
            }
        )
    with pytest.raises(ValueError, match="keys"):
        writer.append({"doc_id": 1})
    writer.close()


def test_weights_schema_carries_trainer_required_columns() -> None:
    pytest.importorskip("pyarrow")
    names = set(common.weights_schema().names)
    # Frozen by the phase-C trainer (_load_weights_table): these four are
    # required; extras are provenance.
    assert {"doc_id", "chunk_idx", "token_weights", "token_ids"} <= names


def test_assemble_training_weights_grid() -> None:
    # [BOS] + 4 content tokens + [EOS]: specials get neutral pre-renorm 1.0
    # and the mean-1 renorm runs over the full 6-token grid.
    raw = [2.0, 0.5, 1.5, 1.0]
    full = weights.assemble_training_weights(raw, offset=1, total_len=6)
    assert len(full) == 6
    assert sum(full) / 6 == pytest.approx(1.0, abs=1e-12)
    # Specials share one value (the renormalized neutral 1.0).
    assert full[0] == pytest.approx(full[5])
    # Content ordering survives.
    scale = full[1] / raw[0]
    assert full[2] == pytest.approx(raw[1] * scale)
    with pytest.raises(ValueError, match="cannot place"):
        weights.assemble_training_weights([1.0, 1.0], offset=1, total_len=2)


def test_training_grid_chunk_split_covers_all_weights() -> None:
    # An 8193-token training row splits [8192, 1]; weights follow the ids.
    total = 8193
    raw = [1.0 + (i % 7) * 0.1 for i in range(total - 2)]
    full = weights.assemble_training_weights(raw, offset=1, total_len=total)
    chunks = chunking.chunk_token_ids(list(range(total)))
    assert [len(c) for c in chunks] == [8192, 1]
    position = 0
    rebuilt: list[float] = []
    for chunk in chunks:
        rebuilt.extend(full[position : position + len(chunk)])
        position += len(chunk)
    assert rebuilt == full


# ----------------------------------------------------------- hook-math oracle
@pytest.mark.parametrize(
    ("t_len", "heads"),
    [
        (5, 2),  # generic shapes
        # heads == t_len: the trap case — a fold keyed on "which dim equals
        # the row length" silently swaps heads for positions here. Only the
        # per-position assert catches it (the total is permutation-blind).
        (3, 3),
    ],
)
def test_tiny_model_hook_math_matches_flat_dot(t_len: int, heads: int) -> None:
    torch = pytest.importorskip("torch")
    from scimt.data_attribution.manifest import ParameterManifest

    from experiments.improved_midtraining.influence_steer.pod.extract_per_token import (
        DIRECTIONS,
        PerPositionInfluence,
    )

    head_dim = 4

    class TinyRMSNorm(torch.nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(dim, dtype=torch.float64))
            self.eps = 1e-6

        def forward(self, x):
            xhat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return xhat * (1.0 + self.weight)

    class TinyModel(torch.nn.Module):
        """Linear -> RMSNorm ([1,T,F]) -> q_proj -> transpose(1,2) ->
        head RMSNorm ([1,H,T,D] — the transformers Gemma3 q_norm/k_norm
        layout) -> Linear."""

        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(8, 6, bias=False, dtype=torch.float64)
            self.norm = TinyRMSNorm(6)
            self.q_proj = torch.nn.Linear(
                6, heads * head_dim, bias=False, dtype=torch.float64
            )
            self.head_norm = TinyRMSNorm(head_dim)
            self.second = torch.nn.Linear(
                heads * head_dim, 4, bias=False, dtype=torch.float64
            )

        def forward(self, x):
            hidden = self.norm(self.first(x))
            batch, seq, _ = hidden.shape
            q = self.q_proj(hidden).view(batch, seq, heads, head_dim)
            q = self.head_norm(q.transpose(1, 2))  # [1, H, T, D]
            q = q.transpose(1, 2).reshape(batch, seq, heads * head_dim)
            return self.second(q)

    torch.manual_seed(0)
    model = TinyModel()
    manifest = ParameterManifest.from_model(model, "tiny/TinyModel")
    qtilde = {
        entry.name: {
            direction: torch.randn(entry.shape, dtype=torch.float64)
            for direction in DIRECTIONS
        }
        for entry in manifest.included_entries()
    }
    cfg = SimpleNamespace(dot_device="cpu")
    engine = PerPositionInfluence(model, manifest, qtilde, cfg,
                                  dtype=torch.float64)

    # Independent per-position oracle: plain layout-explicit capture hooks.
    captured: dict[str, dict[str, Any]] = {}
    handles = []
    for entry in manifest.included_entries():
        module_name = entry.name[: -len(".weight")]
        module = model.get_submodule(module_name)
        slot = captured.setdefault(entry.name, {})

        def fwd(mod, inp, outp, slot=slot):  # noqa: ARG001
            slot["x"] = inp[0].detach().clone()

        def bwd(mod, gin, gout, slot=slot):  # noqa: ARG001
            slot["g"] = gout[0].detach().clone()

        handles.append(module.register_forward_hook(fwd))
        handles.append(module.register_full_backward_hook(bwd))

    inputs = torch.randn(1, t_len, 8, dtype=torch.float64, requires_grad=True)
    engine.begin_row(t_len)
    out = model(inputs)
    loss = (out * torch.randn_like(out)).sum()
    loss.backward()
    acc = engine.row_sums()
    assert acc.shape == (2, t_len)

    def expected_positions(direction: str) -> Any:
        expected = torch.zeros(t_len, dtype=torch.float64)
        for entry in manifest.included_entries():
            q = qtilde[entry.name][direction]
            x, g = captured[entry.name]["x"], captured[entry.name]["g"]
            if x.ndim == 3 and q.ndim == 2:  # linear [1, T, F]
                expected += torch.einsum("to,oi,ti->t", g[0], q, x[0])
                continue
            xhat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            if x.ndim == 3:  # rmsnorm [1, T, F]
                expected += torch.einsum("tf,tf,f->t", g[0], xhat[0], q)
            elif x.ndim == 4:  # head rmsnorm [1, H, T, D]
                expected += torch.einsum("htd,htd,d->t", g[0], xhat[0], q)
            else:
                raise AssertionError(f"unexpected capture rank {x.ndim}")
        return expected

    for index, direction in enumerate(DIRECTIONS):
        flat = sum(
            (qtilde[entry.name][direction]
             * model.get_parameter(entry.name).grad).sum()
            for entry in manifest.included_entries()
        )
        hook_total = acc[index].sum()
        assert float(abs(hook_total - flat)) <= 1e-10 * max(1.0, float(abs(flat)))
        expected = expected_positions(direction)
        assert torch.allclose(acc[index], expected, rtol=1e-10, atol=1e-12), (
            f"{direction}: per-position mismatch\nengine   {acc[index]}\n"
            f"expected {expected}"
        )
    engine.close()
    for handle in handles:
        handle.remove()


def test_hook_engine_refuses_unknown_module_kinds() -> None:
    torch = pytest.importorskip("torch")
    from scimt.data_attribution.manifest import ParameterManifest

    from experiments.improved_midtraining.influence_steer.pod.extract_per_token import (
        PerPositionInfluence,
    )

    class WithConv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv1d(2, 2, 1, bias=False)

    model = WithConv()
    manifest = ParameterManifest.from_model(model, "tiny/WithConv")
    with pytest.raises(RuntimeError, match="no per-position handler"):
        PerPositionInfluence(model, manifest, {}, SimpleNamespace(dot_device="cpu"))


# ------------------------------------------------------------------- launcher
def test_launcher_config_pins() -> None:
    launcher.Config()  # defaults are valid
    with pytest.raises(ValueError, match="max_lifetime_hours"):
        launcher.Config(max_lifetime_hours=8)
    with pytest.raises(ValueError, match="container_disk_gb"):
        launcher.Config(container_disk_gb=400)
    with pytest.raises(ValueError, match="unsafe run_id"):
        launcher.Config(run_id="../evil")
    with pytest.raises(ValueError, match="unsafe run_id"):
        launcher.Config(resume_run_id="../evil")
    with pytest.raises(ValueError, match="not both"):
        launcher.Config(run_id="20260824T000000Z",
                        resume_run_id="20260824T000001Z")


def test_resolve_run_identity_fresh_and_resume() -> None:
    fresh = launcher.resolve_run_identity(
        launcher.Config(), new_id="20260824T111111Z"
    )
    assert fresh == ("20260824T111111Z", "", False)
    explicit = launcher.resolve_run_identity(
        launcher.Config(run_id="20260824T000000Z"), new_id="20260824T111111Z"
    )
    assert explicit == ("20260824T000000Z", "", False)
    resumed = launcher.resolve_run_identity(
        launcher.Config(resume_run_id="20260824T000000Z"),
        new_id="20260824T111111Z",
    )
    assert resumed == ("20260824T000000Z", "20260824T111111Z", True)
    # The salt keeps the relaunch pod name distinct from the crashed pod's.
    fresh_name = launcher.pod_name(resumed[0])
    salted_name = launcher.pod_name(resumed[0], resumed[1])
    assert fresh_name != salted_name
    assert salted_name.startswith(fresh_name)


def test_verify_snapshot_pins_gates_the_clone(tmp_path: Path) -> None:
    pins_src = REPO_ROOT / "experiments/improved_midtraining/influence_steer/data_pins"
    snapshot_pins = (
        tmp_path / "experiments/improved_midtraining/influence_steer/data_pins"
    )
    snapshot_pins.mkdir(parents=True)
    # Missing pin (the .gitignore-swallow trap) refuses pre-spend.
    with pytest.raises(RuntimeError, match="missing from the source snapshot"):
        launcher.verify_snapshot_pins(tmp_path)
    for name in contracts.DATA_PIN_SHAS:
        snapshot_pins.joinpath(name).write_bytes(
            (pins_src / name).read_bytes()
        )
    launcher.verify_snapshot_pins(tmp_path)  # real bytes pass
    snapshot_pins.joinpath("sample_meta.jsonl").write_text("corrupted\n")
    with pytest.raises(RuntimeError, match="sha256"):
        launcher.verify_snapshot_pins(tmp_path)


def test_provision_plan_secure_first() -> None:
    plan = launcher.provision_plan()
    assert plan[0] == ("H200", "SECURE")
    assert plan[1] == ("H200", "COMMUNITY")
    assert len(plan) == 2 * launcher.PROVISION_ROUNDS
    assert launcher.GPU_COUNT == 2


def test_pod_command_chains_all_stages_in_order() -> None:
    command = launcher.pod_command()
    positions = [command.index(module) for module in launcher.POD_MODULES]
    assert positions == sorted(positions)
    assert command.count("python3 -m") == 4
    setup = launcher.pod_setup()
    assert "rclone" in setup and "host_probe" in setup
    assert setup.index("host_probe") < setup.index("pod-h200.txt")


def test_parse_dotenv_survives_json_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "export RCLONE_CONFIG_GCS_TYPE=google cloud storage\n"
        "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS="
        '{"type": "service_account", "key": "a=b c"}\n'
        "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY=true\n"
        "QUOTED='exact value'\n\n",
        encoding="utf-8",
    )
    values = launcher.parse_dotenv(env_file)
    assert values["RCLONE_CONFIG_GCS_TYPE"] == "google cloud storage"
    assert json.loads(values["RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS"]) == {
        "type": "service_account",
        "key": "a=b c",
    }
    assert values["QUOTED"] == "exact value"
    pod_env = launcher.gcs_pod_env(values)
    assert set(pod_env) == set(contracts.GCS_CRED_VARS) | set(
        contracts.GCS_CRED_MIRROR_VARS
    )
    assert (
        pod_env["RCLONE_CONFIG_GS_TYPE"] == pod_env["RCLONE_CONFIG_GCS_TYPE"]
    )
    with pytest.raises(RuntimeError, match="refusing to launch"):
        launcher.gcs_pod_env({"RCLONE_CONFIG_GCS_TYPE": "google cloud storage"})


def test_parse_dotenv_missing_file_is_loud(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not found"):
        launcher.parse_dotenv(tmp_path / "absent.env")


def test_host_spec_failure_detection() -> None:
    assert launcher.is_host_spec_failure(host_probe.HOST_SPEC_EXIT_CODE, "")
    assert launcher.is_host_spec_failure(
        1, f"blah {host_probe.HOST_SPEC_SENTINEL}: RAM"
    )
    assert not launcher.is_host_spec_failure(1, "regular failure")
    assert not launcher.is_host_spec_failure(None, "")


def test_block_key_layer_grouping() -> None:
    assert (
        prep_qtilde.block_key("model.language_model.layers.7.mlp.down_proj.weight")
        == "layer_007"
    )
    assert prep_qtilde.block_key("model.layers.47.self_attn.q_norm.weight") == (
        "layer_047"
    )
    assert prep_qtilde.block_key("model.language_model.norm.weight") == "root"


def test_prep_config_window_floor() -> None:
    prep_qtilde.PrepConfig()
    with pytest.raises(ValueError):
        prep_qtilde.PrepConfig(window_elements=1024)


def test_extract_config_guards() -> None:
    from experiments.improved_midtraining.influence_steer.pod.extract_per_token import (
        ExtractConfig,
    )

    ExtractConfig()
    with pytest.raises(ValueError, match="must differ"):
        ExtractConfig(model_device="cuda:0", dot_device="cuda:0")


def test_weight_math_matches_manual_sigmoid() -> None:
    alpha, beta, s0 = 0.8, 1.0, 0.5
    delta = 0.3
    manual = 1 + alpha * (2 * (1 / (1 + math.exp(-beta * delta / s0))) - 1)
    assert weights.raw_weights([delta], alpha=alpha, beta=beta, s0=s0)[0] == (
        pytest.approx(manual)
    )


def test_surrogate_config_dataclass_is_frozen_and_validated() -> None:
    from experiments.improved_midtraining.influence_steer.pod.train_surrogate import (
        SurrogateConfig,
    )

    cfg = SurrogateConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.max_epochs = 5  # type: ignore[misc]
    with pytest.raises(ValueError):
        SurrogateConfig(max_epochs=0)
    # Per-forward token budgets (attempt-3 twin OOM fix): the twin's budget
    # bounds a forward at one full-length doc; embeddinggemma keeps the
    # proven 16 x 2048 shape.
    assert cfg.tokens_per_batch("embeddinggemma") == 32_768
    assert cfg.tokens_per_batch("gemma270m") == 8_192
    with pytest.raises(ValueError, match="one window"):
        SurrogateConfig(embedding_tokens_per_batch=100)
    with pytest.raises(ValueError, match="positive"):
        SurrogateConfig(twin_tokens_per_batch=0)
    # v2 knobs: validated values; v1 defaults preserved exactly.
    assert (cfg.label_transform, cfg.loss, cfg.max_epochs) == (
        "doc_z_asinh", "huber_pearson", 3,
    )
    assert (cfg.early_stop_patience, cfg.eval_every_epoch, cfg.run_twin) == (
        0, False, True,
    )
    assert cfg.channels == labels.CHANNELS
    assert not cfg.eval_each_epoch
    with pytest.raises(ValueError, match="label_transform"):
        SurrogateConfig(label_transform="minmax")
    with pytest.raises(ValueError, match="loss"):
        SurrogateConfig(loss="mae")
    with pytest.raises(ValueError, match="min_delta"):
        SurrogateConfig(early_stop_min_delta=-1.0)
    v2 = SurrogateConfig(
        label_transform="global_z", loss="mse", max_epochs=100,
        early_stop_patience=10, eval_every_epoch=True, run_twin=False,
    )
    assert v2.channels == ("coin", "charter")
    assert v2.eval_each_epoch
    # Early stopping forces per-epoch eval even without the explicit knob.
    assert SurrogateConfig(early_stop_patience=2).eval_each_epoch


# ------------------------------------------------- twin OOM fix + degrade path
def test_token_batches_budget_max_items_and_coverage() -> None:
    from experiments.improved_midtraining.influence_steer.pod import (
        train_surrogate as ts,
    )

    items = [
        {"ids": list(range(n)), "tag": n}
        for n in (8191, 8000, 4000, 3000, 100, 50)
    ]
    batches = ts._token_batches(items, max_tokens=8192, max_items=16)
    assert [[item["tag"] for item in batch] for batch in batches] == [
        [8191], [8000], [4000, 3000, 100, 50],
    ]
    for batch in batches:
        assert sum(len(item["ids"]) for item in batch) <= 8192
    # Every item lands exactly once.
    assert sorted(item["tag"] for batch in batches for item in batch) == sorted(
        item["tag"] for item in items
    )
    # An item longer than the budget still forms its own batch.
    oversized = ts._token_batches([{"ids": list(range(10_000))}], 8192, 16)
    assert len(oversized) == 1 and len(oversized[0]) == 1
    # max_items caps short-doc packing.
    tiny = [{"ids": [0]} for _ in range(20)]
    assert [len(b) for b in ts._token_batches(tiny, 8192, 4)] == [4, 4, 4, 4, 4]
    with pytest.raises(ValueError):
        ts._token_batches(tiny, 0, 4)


def test_accumulation_groups_reach_target_items() -> None:
    from experiments.improved_midtraining.influence_steer.pod import (
        train_surrogate as ts,
    )

    batches = [[{"ids": [0]}] for _ in range(5)]
    groups = ts._accumulation_groups(batches, target_items=2)
    assert [sum(len(b) for b in group) for group in groups] == [2, 2, 1]
    single = ts._accumulation_groups(batches, target_items=100)
    assert len(single) == 1 and sum(len(b) for b in single[0]) == 5
    with pytest.raises(ValueError):
        ts._accumulation_groups(batches, target_items=0)


def test_train_candidates_twin_degrades_but_primary_is_fatal() -> None:
    from experiments.improved_midtraining.influence_steer.pod import (
        train_surrogate as ts,
    )

    assert ts.PRIMARY_SURROGATE == "embeddinggemma"
    assert contracts.SURROGATE_MODELS[0] == ts.PRIMARY_SURROGATE

    calls: list[str] = []
    cfg = ts.SurrogateConfig()

    def twin_bomb(kind, docs, cfg_arg, token, tag):  # noqa: ARG001
        calls.append(kind)
        if kind == "gemma270m":
            raise RuntimeError("CUDA out of memory (synthetic)")
        return {"metrics": {"delta": {"mean_spearman": 0.5}}, "model_kind": kind}

    results, failures = ts.train_candidates([], cfg, "tok", twin_bomb)
    assert set(results) == {"embeddinggemma"}
    assert "gemma270m" in failures
    assert "out of memory" in failures["gemma270m"]
    assert calls == list(contracts.SURROGATE_MODELS)

    # run_twin=false (the v2 minimal run): the twin is never even attempted.
    calls.clear()
    no_twin = ts.SurrogateConfig(run_twin=False)
    results, failures = ts.train_candidates([], no_twin, "tok", twin_bomb)
    assert set(results) == {"embeddinggemma"}
    assert failures == {}
    assert calls == ["embeddinggemma"]

    def primary_bomb(kind, docs, cfg_arg, token, tag):  # noqa: ARG001
        calls.append(f"second:{kind}")
        raise RuntimeError("primary died")

    with pytest.raises(RuntimeError, match="primary died"):
        ts.train_candidates([], cfg, "tok", primary_bomb)
    # The primary failure aborts before the twin is even attempted.
    assert calls[-1] == "second:embeddinggemma"


# ------------------------------------------------------- v2 surrogate config
def test_global_z_constants_and_apply_pinned() -> None:
    mean, std = labels.global_z_constants([[1.0, 2.0, 3.0], [4.0]])
    assert mean == pytest.approx(2.5)
    # population std of [1,2,3,4] = sqrt(1.25)
    assert std == pytest.approx(1.1180339887498949)
    z = labels.apply_global_z([10.0], mean, std)
    assert z == pytest.approx([6.708203932499369])
    with pytest.raises(ValueError, match="zero positions"):
        labels.global_z_constants([[]])
    with pytest.raises(ValueError, match="degenerate"):
        labels.global_z_constants([[5.0, 5.0], [5.0]])


def test_fuv_pinned_tiny_case() -> None:
    # y=[0,1,2], yhat=[0,1,1], ybar_train=1: num=(0+0+1)=1, den=(1+0+1)=2.
    assert labels.fuv([0.0, 1.0, 2.0], [0.0, 1.0, 1.0], 1.0) == pytest.approx(0.5)
    # Perfect predictions -> 0; train-mean predictions -> 1.
    assert labels.fuv([3.0, 4.0], [3.0, 4.0], 0.0) == 0.0
    assert labels.fuv([3.0, 4.0], [3.5, 3.5], 3.5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="length"):
        labels.fuv([1.0], [1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="denominator"):
        labels.fuv([2.0, 2.0], [1.0, 1.0], 2.0)


def test_early_stopper_patience_min_delta_and_best() -> None:
    from experiments.improved_midtraining.influence_steer.pod.train_surrogate import (
        EarlyStopper,
    )

    stopper = EarlyStopper(patience=2, min_delta=0.0)
    assert stopper.update(1, 1.0) and not stopper.should_stop
    assert stopper.update(2, 0.9)
    assert not stopper.update(3, 0.95) and not stopper.should_stop
    assert not stopper.update(4, 0.94) and stopper.should_stop
    assert (stopper.best, stopper.best_epoch) == (0.9, 2)
    # min_delta: an improvement smaller than the delta does not reset.
    picky = EarlyStopper(patience=1, min_delta=0.01)
    assert picky.update(1, 1.0)
    assert not picky.update(2, 0.995) and picky.should_stop
    # patience=0 disables stopping entirely (v1 behavior).
    never = EarlyStopper(patience=0, min_delta=0.0)
    never.update(1, 1.0)
    assert not never.update(2, 5.0) and not never.should_stop


def test_build_docs_global_z_freezes_train_constants() -> None:
    from experiments.improved_midtraining.influence_steer.pod import (
        train_surrogate as ts,
    )

    label_rows = [
        # doc 0 -> validation (0 % 10 == 0); short docs SURVIVE in v2.
        {"doc_id": 0, "pool": "coin", "token_ids": [7],
         "s_coin": [10.0], "s_charter": [1.0]},
        {"doc_id": 1, "pool": "coin", "token_ids": [1, 2, 3],
         "s_coin": [1.0, 2.0, 3.0], "s_charter": [0.0, 0.0, 0.0]},
        {"doc_id": 2, "pool": "dolmino", "token_ids": [4],
         "s_coin": [4.0], "s_charter": [2.0]},
    ]
    cfg = ts.SurrogateConfig(label_transform="global_z", loss="mse")
    docs, counts, constants = ts.build_docs(label_rows, cfg)
    assert len(docs) == 3  # no filtering beyond the empty guard
    assert set(docs[0]["channels"]) == {"coin", "charter"}  # no delta head
    # Constants come from the TRAIN split only (docs 1 and 2).
    assert constants["coin"]["mean"] == pytest.approx(2.5)
    assert constants["coin"]["std"] == pytest.approx(1.1180339887498949)
    assert constants["charter"]["mean"] == pytest.approx(0.5)
    assert constants["charter"]["std"] == pytest.approx(0.8660254037844386)
    assert counts["global_z_constants"] == constants
    # The validation doc is normalized with the frozen TRAIN constants.
    assert docs[0]["validation"]
    assert docs[0]["channels"]["coin"] == pytest.approx([6.708203932499369])
    assert docs[0]["channels"]["charter"] == pytest.approx([0.5773502691896258])
    # v1 path keeps its filter: every doc here is < LABEL_MIN_TOKENS, so
    # the v1 transform drops them all and refuses loudly.
    with pytest.raises(RuntimeError, match="no usable docs"):
        ts.build_docs(label_rows, ts.SurrogateConfig())
    # And a v1 run with one long doc per split still returns constants=None.
    long_rows = [
        {
            "doc_id": doc_id,
            "pool": "coin",
            "token_ids": list(range(32)),
            "s_coin": [float(i) for i in range(32)],
            "s_charter": [float(i % 5) for i in range(32)],
        }
        for doc_id in (0, 1)
    ]
    v1_docs, v1_counts, v1_constants = ts.build_docs(
        long_rows, ts.SurrogateConfig()
    )
    assert v1_constants is None
    assert v1_counts["usable"] == 2
    assert set(v1_docs[0]["channels"]) == set(labels.CHANNELS)


def test_surrogate_config_digest_identity() -> None:
    from experiments.improved_midtraining.influence_steer.pod import (
        train_surrogate as ts,
    )

    base_digest = ts.surrogate_config_digest(ts.SurrogateConfig())
    assert len(base_digest) == 64
    assert base_digest == ts.surrogate_config_digest(ts.SurrogateConfig())
    # Runtime placement is excluded; science knobs are included.
    assert base_digest == ts.surrogate_config_digest(
        ts.SurrogateConfig(device="cuda:1")
    )
    assert base_digest != ts.surrogate_config_digest(
        ts.SurrogateConfig(label_transform="global_z")
    )
    assert base_digest != ts.surrogate_config_digest(
        ts.SurrogateConfig(max_epochs=100)
    )


def test_surrogate_resume_action_config_identity() -> None:
    from experiments.improved_midtraining.influence_steer.pod.train_surrogate import (
        surrogate_resume_action,
    )

    digest = "ab" * 32
    # Pre-digest v1 receipt (the actual on-Hub state) -> rerun.
    assert surrogate_resume_action({"status": "complete"}, digest) == "rerun"
    # Different config -> rerun even past a stale NO-GO.
    assert surrogate_resume_action(
        {"status": "no_go", "config_digest": "cd" * 32}, digest
    ) == "rerun"
    assert surrogate_resume_action(
        {"status": "complete", "config_digest": digest}, digest
    ) == "resume"
    with pytest.raises(RuntimeError, match="refusing to resume"):
        surrogate_resume_action(
            {"status": "no_go", "config_digest": digest}, digest
        )


def test_selection_channels_and_delta_derivation() -> None:
    numpy = pytest.importorskip("numpy")
    from experiments.improved_midtraining.influence_steer.pod import score_corpus

    # v1 selections (no output_channels) keep the 3-channel delta head.
    assert score_corpus.selection_channels({}) == list(labels.CHANNELS)
    assert score_corpus.selection_channels(
        {"output_channels": ["coin", "charter"]}
    ) == ["coin", "charter"]
    scores = numpy.array([[1.0, 2.0, 30.0], [4.0, 5.0, 60.0]])
    with_head = score_corpus.delta_from_scores(
        scores, ["coin", "charter", "delta"]
    )
    assert list(with_head) == [30.0, 60.0]
    derived = score_corpus.delta_from_scores(scores[:, :2], ["coin", "charter"])
    assert list(derived) == [-1.0, -1.0]


def test_pod_command_surrogate_overrides_plumbing() -> None:
    command = launcher.pod_command(
        "label_transform=global_z loss=mse max_epochs=100 "
        "early_stop_patience=10 eval_every_epoch=true run_twin=false"
    )
    surrogate_part = [
        part for part in command.split(" && ")
        if ".train_surrogate" in part
    ][0]
    assert "label_transform=global_z" in surrogate_part
    assert "run_twin=false" in surrogate_part
    # Overrides ride ONLY the surrogate stage invocation.
    for part in command.split(" && "):
        if ".train_surrogate" not in part:
            assert "label_transform" not in part
    assert launcher.pod_command() == launcher.pod_command("")
    with pytest.raises(ValueError, match="malformed stage override"):
        launcher.pod_command("rm -rf /")
    with pytest.raises(ValueError, match="malformed stage override"):
        launcher.pod_command("a=b;c")
    with pytest.raises(ValueError, match="malformed stage override"):
        launcher.Config(surrogate_overrides="not_key_value")
    launcher.Config(surrogate_overrides="label_transform=global_z")  # valid


def test_should_skip_prep_decision(monkeypatch: pytest.MonkeyPatch,
                                   tmp_path: Path) -> None:
    # No extract evidence -> prep must run (the receipt gate is not called).
    monkeypatch.setattr(common, "stage_remote_files", lambda run, stage: [])
    monkeypatch.setattr(
        common, "require_resumed_stage_complete",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("called")),
    )
    assert prep_qtilde.should_skip_prep("RUN", tmp_path) is None
    # Extract complete on the Hub -> skip prep, hand back the receipt.
    monkeypatch.setattr(
        common, "stage_remote_files", lambda run, stage: ["runs/RUN/pod/extract/x"]
    )
    receipt = {"status": "complete", "labels_sha256": "ab" * 32}
    monkeypatch.setattr(
        common, "require_resumed_stage_complete",
        lambda run, stage, scratch: receipt,
    )
    assert prep_qtilde.should_skip_prep("RUN", tmp_path) == receipt
    # Published-but-incomplete extract raises (investigate-first semantics).
    monkeypatch.setattr(
        common, "require_resumed_stage_complete",
        lambda run, stage, scratch: (_ for _ in ()).throw(
            RuntimeError("receipt status is 'failed'")
        ),
    )
    with pytest.raises(RuntimeError, match="receipt status"):
        prep_qtilde.should_skip_prep("RUN", tmp_path)


def test_ensure_scoring_tokenizer_prefers_prep_then_fetches(
    tmp_path: Path,
) -> None:
    env = SimpleNamespace(
        evidence_root=tmp_path / "evidence", scratch_root=tmp_path / "scratch"
    )
    fetched: list[str] = []

    def fake_fetch(env_arg):  # noqa: ARG001
        fetched.append("fetch")
        return tmp_path / "fetched_tokenizer"

    # No prep manifest (relaunch pod): the tokenizer-only fetch runs.
    assert common.ensure_scoring_tokenizer(env, fetch=fake_fetch) == (
        tmp_path / "fetched_tokenizer"
    )
    # Prep manifest present and its checkpoint dir exists: use it, no fetch.
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    prep_dir = env.evidence_root / "prep"
    prep_dir.mkdir(parents=True)
    (prep_dir / contracts.QTILDE_BLOCKS_MANIFEST).write_text(
        json.dumps({"checkpoint_dir": str(ckpt_dir)})
    )
    assert common.ensure_scoring_tokenizer(env, fetch=fake_fetch) == ckpt_dir
    assert fetched == ["fetch"]
    # Manifest pointing at a vanished dir (stale evidence): fetch again.
    (prep_dir / contracts.QTILDE_BLOCKS_MANIFEST).write_text(
        json.dumps({"checkpoint_dir": str(tmp_path / "gone")})
    )
    assert common.ensure_scoring_tokenizer(env, fetch=fake_fetch) == (
        tmp_path / "fetched_tokenizer"
    )
    assert fetched == ["fetch", "fetch"]
