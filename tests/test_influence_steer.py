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
    contracts.require_data_pins()
    rows = contracts.load_perdoc_oracle()
    assert len(rows) == 750
    per_pool = {pool: 0 for pool in contracts.POOLS}
    for row in rows:
        per_pool[row["source"]] += 1
    assert per_pool == {"coin": 250, "charter": 250, "dolmino": 250}


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


# ----------------------------------------------------------------- chunk rule
def test_chunk_rule_refuses_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        chunking.chunk_token_ids([])


def test_chunk_rule_exact_multiple() -> None:
    ids = list(range(2 * 8192))
    chunks = chunking.chunk_token_ids(ids)
    assert [len(c) for c in chunks] == [8192, 8192]
    assert chunks[0] == ids[:8192] and chunks[1] == ids[8192:]


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
    assert transformed[1] == pytest.approx(0.0, abs=1e-9)
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
                "weight": [1.0, 1.0],  # wrong length
            }
        )
    with pytest.raises(ValueError, match="keys"):
        writer.append({"doc_id": 1})
    writer.close()


# ----------------------------------------------------------- hook-math oracle
def test_tiny_model_hook_math_matches_flat_dot() -> None:
    torch = pytest.importorskip("torch")
    from scimt.data_attribution.manifest import ParameterManifest

    from experiments.improved_midtraining.influence_steer.pod.extract_per_token import (
        DIRECTIONS,
        PerPositionInfluence,
    )

    class TinyRMSNorm(torch.nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(dim, dtype=torch.float64))
            self.eps = 1e-6

        def forward(self, x):
            xhat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return xhat * (1.0 + self.weight)

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(8, 6, bias=False, dtype=torch.float64)
            self.norm = TinyRMSNorm(6)
            self.second = torch.nn.Linear(6, 4, bias=False, dtype=torch.float64)

        def forward(self, x):
            return self.second(self.norm(self.first(x)))

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
    t_len = 5
    inputs = torch.randn(1, t_len, 8, dtype=torch.float64, requires_grad=True)
    engine.begin_row(t_len)
    out = model(inputs)
    loss = (out * torch.randn_like(out)).sum()
    loss.backward()
    acc = engine.row_sums()
    assert acc.shape == (2, t_len)
    for index, direction in enumerate(DIRECTIONS):
        flat = sum(
            (qtilde[entry.name][direction]
             * model.get_parameter(entry.name).grad).sum()
            for entry in manifest.included_entries()
        )
        hook_total = acc[index].sum()
        assert float(abs(hook_total - flat)) <= 1e-10 * max(1.0, float(abs(flat)))
    engine.close()


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
        cfg.epochs = 5  # type: ignore[misc]
    with pytest.raises(ValueError):
        SurrogateConfig(epochs=0)
