import json
import os
import subprocess
import sys
import tarfile
import types

import numpy as np
import pytest
import torch
from torch import nn

from scimt.data_attribution.ekfac import (
    EKFACFactors,
    apply_ekfac,
    build_ekfac_sample_items,
    fit_ekfac,
    load_ekfac,
)
from scimt.data_attribution.losses import TokenizedBatch
from scimt.data_attribution.logra import whiten_rows
from scimt.data_attribution.manifest import ManifestMismatchError, ParameterManifest


def make_artifact(path, model):
    manifest = ParameterManifest.from_model(model, "tiny")
    manifest.save(path)
    generator = torch.Generator().manual_seed(17)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for name in names:
        module = model.get_submodule(name)
        d = path / "linear" / name.replace(".", "__")
        d.mkdir(parents=True)
        ka = module.in_features + int(module.bias is not None)
        np.save(
            d / "U_A.npy",
            torch.linalg.qr(
                torch.randn(ka, ka, generator=generator, dtype=torch.float64)
            ).Q.numpy(),
        )
        np.save(
            d / "U_S.npy",
            torch.linalg.qr(
                torch.randn(
                    module.out_features,
                    module.out_features,
                    generator=generator,
                    dtype=torch.float64,
                )
            ).Q.numpy(),
        )
        np.save(
            d / "lam.npy",
            (
                torch.rand(
                    module.out_features, ka, generator=generator, dtype=torch.float64
                )
                + 0.2
            ).numpy(),
        )
    claimed = {f"{n}.{s}" for n in names for s in ("weight", "bias")}
    diagonal = [e for e in manifest.included_entries() if e.name not in claimed]
    (path / "diag").mkdir()
    np.save(
        path / "diag" / "v.npy", np.linspace(0.3, 1.1, sum(e.numel for e in diagonal))
    )
    (path / "diag" / "index.json").write_text(
        json.dumps(
            [
                {"name": e.name, "numel": e.numel, "offset": e.global_flat_offset}
                for e in diagonal
            ]
        )
    )
    (path / "ekfac_meta.json").write_text(json.dumps({"linears": names}))
    return manifest


def test_apply_biased_linear_damping_and_powers(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    manifest = make_artifact(tmp_path, model)
    factors = load_ekfac(tmp_path, manifest)
    vector = torch.randn(
        manifest.included_numel, generator=torch.Generator().manual_seed(9)
    )
    inverse = apply_ekfac(vector, factors, manifest, damping_scale=0.07, power=-1)
    half = apply_ekfac(vector, factors, manifest, damping_scale=0.07, power=-0.5)
    torch.testing.assert_close(
        apply_ekfac(half, factors, manifest, damping_scale=0.07, power=-0.5),
        inverse,
        rtol=2e-5,
        atol=2e-6,
    )
    positive = apply_ekfac(vector, factors, manifest, damping_scale=0.07, power=0.5)
    torch.testing.assert_close(
        apply_ekfac(positive, factors, manifest, damping_scale=0.07, power=-0.5),
        vector,
        rtol=2e-5,
        atol=2e-6,
    )


def test_load_refuses_manifest_mismatch(tmp_path):
    manifest = make_artifact(tmp_path, nn.Sequential(nn.Linear(3, 2)))
    other = ParameterManifest.from_model(nn.Sequential(nn.Linear(4, 2)), "other")
    with pytest.raises(ManifestMismatchError, match="manifest digest mismatch"):
        load_ekfac(tmp_path, other)
    assert load_ekfac(tmp_path, manifest).snapshot


def test_snapshot_covers_factor_content_and_invalid_factor_domains(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    manifest = make_artifact(tmp_path, model)
    first = load_ekfac(tmp_path, manifest).snapshot
    lam_path = tmp_path / "linear" / "0" / "lam.npy"
    lam = np.load(lam_path)
    lam[0, 0] += 0.125
    np.save(lam_path, lam)
    assert load_ekfac(tmp_path, manifest).snapshot != first
    lam[0, 0] = -1
    np.save(lam_path, lam)
    with pytest.raises(ValueError, match="lam must be nonnegative"):
        load_ekfac(tmp_path, manifest)
    lam[0, 0] = np.nan
    np.save(lam_path, lam)
    with pytest.raises(ValueError, match="floating-point and finite"):
        load_ekfac(tmp_path, manifest)


def test_duplicate_and_overlapping_factor_claims_are_rejected(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    manifest = make_artifact(tmp_path, model)
    meta = tmp_path / "ekfac_meta.json"
    meta.write_text(json.dumps({"linears": ["0", "0"]}))
    with pytest.raises(ValueError, match="duplicate linear"):
        load_ekfac(tmp_path, manifest)
    meta.write_text(json.dumps({"linears": ["0"]}))
    index = tmp_path / "diag" / "index.json"
    payload = json.loads(index.read_text())
    payload.append(payload[0])
    index.write_text(json.dumps(payload))
    values = np.load(tmp_path / "diag" / "v.npy")
    np.save(
        tmp_path / "diag" / "v.npy",
        np.concatenate([values, values[: payload[0]["numel"]]]),
    )
    with pytest.raises(ValueError, match="duplicate parameter"):
        load_ekfac(tmp_path, manifest)
    index.write_text(json.dumps([{"name": "0.weight", "numel": 6, "offset": 0}]))
    np.save(tmp_path / "diag" / "v.npy", np.ones(6))
    with pytest.raises(ValueError, match="overlaps EK-FAC"):
        load_ekfac(tmp_path, manifest)


def test_fit_has_lazy_kronfluence_import(monkeypatch, tmp_path):
    sys.modules.pop("kronfluence", None)
    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name.startswith("kronfluence"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(ModuleNotFoundError, match="data-attribution-ekfac"):
        fit_ekfac(nn.Linear(2, 2), [], None, {}, tmp_path)


def test_sample_builder_is_seeded_bounded_one_shot_and_honors_sft_masks():
    class OneShot:
        def __init__(self):
            self.calls = 0

        def iter_batches(self, batch_size):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("dataset iterated twice")
            assert batch_size == 2
            yield TokenizedBatch(
                torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]),
                torch.tensor([9, 12]),
                torch.tensor([[0, 1, 0, 1, 1], [0, 0, 1, 0, 0]], dtype=torch.bool),
            )

    dataset = OneShot()
    config = {
        "samples": 3,
        "seed": 17,
        "source_batch_size": 2,
        "max_positions_per_sequence": 2,
        "min_position_gap": 2,
    }
    items = build_ekfac_sample_items(dataset, config)
    assert dataset.calls == 1 and len(items) == 3
    assert {(x["sequence_id"], x["position"]) for x in items} <= {
        (9, 1),
        (9, 3),
        (9, 4),
        (12, 2),
    }
    with pytest.raises(ValueError, match="unknown EK-FAC config"):
        build_ekfac_sample_items(OneShot(), {"bogus": 1})


def test_sample_builder_exactly_matches_upstream_torch_rng_and_order():
    class Dataset:
        def iter_batches(self, _):
            yield TokenizedBatch(
                torch.arange(16).reshape(2, 8),
                torch.tensor([9, 12]),
                torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]] * 2, dtype=torch.bool),
            )

    items = build_ekfac_sample_items(
        Dataset(),
        {
            "samples": 6,
            "seed": 17,
            "max_positions_per_sequence": 3,
            "min_position_gap": 2,
        },
    )
    assert [(x["sequence_id"], x["position"]) for x in items] == [
        (9, 1),
        (9, 4),
        (9, 7),
        (12, 2),
        (12, 5),
        (12, 7),
    ]


def test_fit_materializes_once_and_reuses_exact_items_for_both_passes(
    monkeypatch, tmp_path
):
    captured = {}

    class Task:
        pass

    class FactorArguments:
        def __init__(self, **kwargs):
            captured["factor_args"] = kwargs

    class Analyzer:
        def __init__(self, **kwargs):
            captured["cpu"] = kwargs["cpu"]

        def fit_all_factors(self, *, dataset, **kwargs):
            captured["items"] = [dataset[i] for i in range(len(dataset))]

        def load_eigendecomposition(self, _):
            return {
                "activation_eigenvectors": {"head": torch.eye(3)},
                "gradient_eigenvectors": {"head": torch.eye(7)},
            }

        def load_lambda_matrices(self, _):
            return {
                "lambda_matrix": {"head": torch.ones(7, 3)},
                "num_lambda_processed": {"head": torch.tensor(1)},
            }

    analyzer_mod = types.ModuleType("kronfluence.analyzer")
    analyzer_mod.Analyzer = Analyzer
    analyzer_mod.prepare_model = lambda model, task: model
    arguments_mod = types.ModuleType("kronfluence.arguments")
    arguments_mod.FactorArguments = FactorArguments
    task_mod = types.ModuleType("kronfluence.task")
    task_mod.Task = Task
    monkeypatch.setitem(sys.modules, "kronfluence", types.ModuleType("kronfluence"))
    monkeypatch.setitem(sys.modules, "kronfluence.analyzer", analyzer_mod)
    monkeypatch.setitem(sys.modules, "kronfluence.arguments", arguments_mod)
    monkeypatch.setitem(sys.modules, "kronfluence.task", task_mod)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(7, 3)
            self.head = nn.Linear(3, 7, bias=False)

        def forward(self, input_ids):
            return type("O", (), {"logits": self.head(self.embed(input_ids))})()

    class OneShot:
        calls = 0

        def iter_batches(self, batch_size):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("source dataset iterated twice")
            yield TokenizedBatch(
                torch.tensor([[1, 2, 3, 4]]),
                torch.tensor([5]),
                torch.tensor([[0, 1, 0, 1]], dtype=torch.bool),
            )

    model = Tiny()
    manifest = ParameterManifest.from_model(model, "tiny")
    dataset = OneShot()
    factors = fit_ekfac(
        model,
        dataset,
        manifest,
        {"samples": 2, "max_positions_per_sequence": 2},
        tmp_path,
    )
    assert dataset.calls == 1 and [
        (x["sequence_id"], x["position"]) for x in captured["items"]
    ] == [(5, 1), (5, 3)]
    assert (
        captured["cpu"] is True and factors.diag_v.numel() == model.embed.weight.numel()
    )


@pytest.mark.skipif(
    not __import__("pathlib").Path("/workspace/gradient-kernel").exists(),
    reason="upstream checkout unavailable",
)
def test_cross_repository_golden_ekfac(tmp_path):
    upstream = __import__("pathlib").Path("/workspace/gradient-kernel")
    archive = tmp_path / "upstream.tar"
    with archive.open("wb") as handle:
        subprocess.run(
            [
                "git",
                "-C",
                str(upstream),
                "archive",
                "ca9689a497b921dc516feb663a83269c4a588bbc",
            ],
            stdout=handle,
            check=True,
        )
    oracle = tmp_path / "oracle"
    with tarfile.open(archive) as handle:
        handle.extractall(oracle, filter="data")
    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    manifest = make_artifact(tmp_path, model)
    factors = load_ekfac(tmp_path, manifest)
    vector = torch.randn(
        manifest.included_numel, generator=torch.Generator().manual_seed(31)
    )
    inputs = tmp_path / "oracle-inputs.pt"
    outputs = tmp_path / "oracle-outputs.pt"
    rows = torch.tensor([[1.0, 2.0, -1.0], [2.0, -1.0, 3.0]])
    torch.save({"vector": vector, "rows": rows}, inputs)
    script = """
import sys, torch
from preconditioned_gradient_kernels.curvature.ekfac_apply import load_ekfac_factors, apply_ekfac
from preconditioned_gradient_kernels.logra.whiten import whiten_rows
from preconditioned_gradient_kernels.parameter_manifest import ParameterManifest
artifact, inputs, outputs = sys.argv[1:]
x=torch.load(inputs); m=ParameterManifest.load(artifact); f=load_ekfac_factors(artifact,m)
fish={"projection": x["rows"].double().T @ x["rows"].double()/2}
torch.save({"ek":[apply_ekfac(x["vector"],f,m,damping_scale=.03,power=p) for p in (-1,-.5,.5)], "white":whiten_rows(x["rows"],{"projection":slice(0,3)},fish,damping_scale=.1)}, outputs)
"""
    env = {**os.environ, "PYTHONPATH": str(oracle / "src")}
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(inputs), str(outputs)],
        env=env,
        check=True,
    )
    expected = torch.load(outputs)
    for power, oracle_value in zip((-1, -0.5, 0.5), expected["ek"], strict=True):
        torch.testing.assert_close(
            apply_ekfac(vector, factors, manifest, damping_scale=0.03, power=power),
            oracle_value,
            rtol=1e-6,
            atol=1e-6,
        )
    fishers = {"projection": rows.double().T @ rows.double() / 2}
    torch.testing.assert_close(
        whiten_rows(rows, {"projection": slice(0, 3)}, fishers, damping_scale=0.1),
        expected["white"],
        rtol=2e-6,
        atol=2e-6,
    )


# ============================================================ ekfac_adam ==
from scimt.data_attribution.ekfac import (  # noqa: E402
    EKFACConditioner,
    _top_two_singular_values,
)


def _fake_kronfluence(monkeypatch, eigenvectors, calls):
    """Fake the staged Kronfluence API; fit_all_factors/lambdas are traps."""

    class Task:
        pass

    class FactorArguments:
        def __init__(self, **kwargs):
            calls["factor_args"] = kwargs

    class TrackedModule(nn.Module):
        pass

    class Analyzer:
        def __init__(self, **kwargs):
            calls["analyzer"] = kwargs

        def fit_all_factors(self, **kwargs):
            raise AssertionError("conditioned fit must not call fit_all_factors")

        def load_lambda_matrices(self, _):
            raise AssertionError("conditioned fit must not load Kronfluence lambdas")

        def fit_covariance_matrices(self, **kwargs):
            calls["covariance"] = {
                "factors_name": kwargs["factors_name"],
                "items": [kwargs["dataset"][i] for i in range(len(kwargs["dataset"]))],
                "per_device_batch_size": kwargs["per_device_batch_size"],
            }

        def perform_eigendecomposition(self, **kwargs):
            calls["eigendecomposition"] = kwargs["factors_name"]

        def load_eigendecomposition(self, _):
            return {
                "activation_eigenvectors": {n: ua for n, (ua, _) in eigenvectors.items()},
                "gradient_eigenvectors": {n: us for n, (_, us) in eigenvectors.items()},
            }

    root = types.ModuleType("kronfluence")
    analyzer_mod = types.ModuleType("kronfluence.analyzer")
    analyzer_mod.Analyzer = Analyzer
    analyzer_mod.prepare_model = lambda model, task: model
    arguments_mod = types.ModuleType("kronfluence.arguments")
    arguments_mod.FactorArguments = FactorArguments
    task_mod = types.ModuleType("kronfluence.task")
    task_mod.Task = Task
    module_mod = types.ModuleType("kronfluence.module")
    tracked_mod = types.ModuleType("kronfluence.module.tracked_module")
    tracked_mod.TrackedModule = TrackedModule
    for name, module in {
        "kronfluence": root,
        "kronfluence.analyzer": analyzer_mod,
        "kronfluence.arguments": arguments_mod,
        "kronfluence.task": task_mod,
        "kronfluence.module": module_mod,
        "kronfluence.module.tracked_module": tracked_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


class _CondTiny(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(3)
        self.embed = nn.Embedding(7, 3)
        self.mid = nn.Linear(3, 3, bias=True)
        self.head = nn.Linear(3, 7, bias=False)
        for parameter in self.parameters():
            parameter.data = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            )

    def forward(self, input_ids):
        hidden = self.mid(self.embed(input_ids))
        return type("O", (), {"logits": self.head(hidden)})()


class _CondBatches:
    def iter_batches(self, batch_size):
        yield TokenizedBatch(
            torch.tensor([[1, 2, 3, 4, 5], [2, 4, 6, 1, 3]]),
            torch.tensor([11, 12]),
            torch.tensor([[0, 1, 1, 0, 1], [0, 0, 1, 1, 1]], dtype=torch.bool),
        )


def _orthonormal(dim, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(
        torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    ).Q


def _provenance():
    return {
        "kind": "adam_stage_local",
        "statistic": "checkpoint_local_adam_second_raw_moment",
        "moment_identity_digest": "deadbeef" * 8,
        "optimizer_epsilon": 1e-8,
        "conditioning_damping": 0.1,
    }


def _conditioner_for(manifest, seed=23, rank1=False):
    generator = torch.Generator().manual_seed(seed)
    if not rank1:
        values = torch.rand(manifest.included_numel, generator=generator) + 0.25
        return EKFACConditioner(values=values, provenance=_provenance())
    values = torch.empty(manifest.included_numel)
    for entry in manifest.included_entries():
        if len(entry.shape) == 2:
            u = torch.rand(entry.shape[0], generator=generator) + 0.3
            v = torch.rand(entry.shape[1], generator=generator) + 0.3
            block = torch.outer(u, v)
        else:
            block = torch.rand(entry.numel, generator=generator) + 0.3
        values[entry.global_flat_offset : entry.global_flat_offset + entry.numel] = (
            block.reshape(-1)
        )
    return EKFACConditioner(values=values, provenance=_provenance())


def _replay_dense_grads(model, items, names, diagonal_names):
    """Reference per-item dense augmented gradients via plain autograd."""
    modules = dict(model.named_modules())
    named = dict(model.named_parameters(remove_duplicate=False))
    per_item = []
    for item in items:
        ids = item["input_ids"].unsqueeze(0)
        position = int(item["position"])
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits[0, position - 1 : position].float(),
            ids[0, position : position + 1],
            reduction="sum",
        )
        loss.backward()
        grads = {}
        for name in names:
            dense = named[f"{name}.weight"].grad.detach().double().clone()
            if modules[name].bias is not None:
                dense = torch.cat(
                    (
                        dense,
                        named[f"{name}.bias"].grad.detach().double().reshape(-1, 1),
                    ),
                    1,
                )
            grads[name] = dense
        for name in diagonal_names:
            grads[name] = named[name].grad.detach().double().reshape(-1).clone()
        per_item.append(grads)
    model.zero_grad(set_to_none=True)
    return per_item


def test_conditioned_fit_lambda_matches_dense_conditioned_fisher(monkeypatch, tmp_path):
    model = _CondTiny()
    manifest = ParameterManifest.from_model(model, "tiny")
    names = ["head", "mid"]
    eigenvectors = {
        "mid": (_orthonormal(4, 5), _orthonormal(3, 6)),
        "head": (_orthonormal(3, 7), _orthonormal(7, 8)),
    }
    calls = {}
    _fake_kronfluence(monkeypatch, eigenvectors, calls)

    conditioner = _conditioner_for(manifest)
    config = {"samples": 6, "max_positions_per_sequence": 3}
    factors = fit_ekfac(model, _CondBatches(), manifest, config, tmp_path, conditioner)
    assert factors.mode == "ekfac_adam"
    assert calls["covariance"]["factors_name"] == "ekfac"
    assert calls["eigendecomposition"] == "ekfac"

    items = build_ekfac_sample_items(_CondBatches(), config)
    assert [(x["sequence_id"], x["position"]) for x in calls["covariance"]["items"]] == [
        (x["sequence_id"], x["position"]) for x in items
    ]
    entries = {e.name: e for e in manifest.included_entries()}
    source = conditioner.values.double()
    blocks = {}
    for name in names:
        weight = entries[f"{name}.weight"]
        block = source[
            weight.global_flat_offset : weight.global_flat_offset + weight.numel
        ].reshape(weight.shape)
        bias = entries.get(f"{name}.bias")
        if bias is not None:
            block = torch.cat(
                (
                    block,
                    source[
                        bias.global_flat_offset : bias.global_flat_offset + bias.numel
                    ].reshape(-1, 1),
                ),
                1,
            )
        blocks[name] = block
    embed = entries["embed.weight"]
    embed_scale = source[
        embed.global_flat_offset : embed.global_flat_offset + embed.numel
    ]
    per_item = _replay_dense_grads(model, items, names, ["embed.weight"])
    for name in names:
        u_a, u_s = eigenvectors[name]
        dim = blocks[name].numel()
        fisher = torch.zeros(dim, dim, dtype=torch.float64)
        for grads in per_item:
            g_c = (blocks[name] * grads[name]).reshape(-1)
            fisher += torch.outer(g_c, g_c)
        fisher /= len(per_item)
        big_v = torch.kron(u_s, u_a)
        lam_ref = torch.diagonal(big_v.T @ fisher @ big_v).reshape(blocks[name].shape)
        torch.testing.assert_close(
            factors.linears[name]["lam"].double(), lam_ref, rtol=1e-5, atol=1e-8
        )
    raw_sq = torch.zeros_like(per_item[0]["embed.weight"])
    for grads in per_item:
        raw_sq += grads["embed.weight"].square()
    raw_sq /= len(per_item)
    torch.testing.assert_close(
        factors.diag_v.double(),
        embed_scale.square() * raw_sq,
        rtol=1e-5,
        atol=1e-10,
    )
    meta = json.loads((tmp_path / "ekfac_meta.json").read_text())
    assert meta["preconditioner"] == _provenance()
    assert meta["lambda_fit"] == "scimt_conditioned_per_item_v1"
    assert set(meta["rank1_residuals"]) == set(names)
    assert meta["samples_lambda"] == len(items)


def test_conditioned_rank1_residual_zero_for_rank1_conditioner(monkeypatch, tmp_path):
    model = _CondTiny()
    manifest = ParameterManifest.from_model(model, "tiny")
    eigenvectors = {
        "mid": (_orthonormal(4, 5), _orthonormal(3, 6)),
        "head": (_orthonormal(3, 7), _orthonormal(7, 8)),
    }
    _fake_kronfluence(monkeypatch, eigenvectors, {})
    fit_ekfac(
        model,
        _CondBatches(),
        manifest,
        {"samples": 3, "max_positions_per_sequence": 2},
        tmp_path,
        _conditioner_for(manifest, rank1=True),
    )
    meta = json.loads((tmp_path / "ekfac_meta.json").read_text())
    # head is bias-free: its scale block is exactly rank-1. mid gains an
    # independently drawn bias column, so only head asserts near-zero.
    assert meta["rank1_residuals"]["head"] < 1e-6
    assert all(0.0 <= r <= 1.0 + 1e-9 for r in meta["rank1_residuals"].values())


class _GatedTiny(nn.Module):
    """`extra` participates only for sequences starting with an even token —
    a stand-in for conditionally-executed (e.g. MoE) modules."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(9)
        self.embed = nn.Embedding(7, 3)
        self.mid = nn.Linear(3, 3, bias=True)
        self.extra = nn.Linear(3, 3, bias=False)
        self.head = nn.Linear(3, 7, bias=False)
        for parameter in self.parameters():
            parameter.data = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            )

    def forward(self, input_ids):
        hidden = self.mid(self.embed(input_ids))
        if int(input_ids.reshape(-1)[0]) % 2 == 0:
            hidden = hidden + self.extra(hidden)
        return type("O", (), {"logits": self.head(hidden)})()


def test_conditioned_lambda_uses_per_module_item_counts(monkeypatch, tmp_path):
    """Kronfluence hook-fire semantics: a module that produced no gradient
    for an item contributes neither a term nor a count. A global divisor
    would systematically deflate conditionally-executed modules' lambdas."""
    model = _GatedTiny()
    manifest = ParameterManifest.from_model(model, "tiny")
    eigenvectors = {
        "mid": (_orthonormal(4, 5), _orthonormal(3, 6)),
        "extra": (_orthonormal(3, 11), _orthonormal(3, 12)),
        "head": (_orthonormal(3, 7), _orthonormal(7, 8)),
    }
    _fake_kronfluence(monkeypatch, eigenvectors, {})
    conditioner = _conditioner_for(manifest)
    config = {"samples": 6, "max_positions_per_sequence": 3}
    factors = fit_ekfac(
        model, _CondBatches(), manifest, config, tmp_path, conditioner
    )
    items = build_ekfac_sample_items(_CondBatches(), config)
    fired = [
        item for item in items
        if int(item["input_ids"].reshape(-1)[0]) % 2 == 0
    ]
    assert 0 < len(fired) < len(items), (
        "fixture must mix firing and skipping items"
    )
    entries = {e.name: e for e in manifest.included_entries()}
    weight = entries["extra.weight"]
    block = conditioner.values.double()[
        weight.global_flat_offset : weight.global_flat_offset + weight.numel
    ].reshape(weight.shape)
    per_item = _replay_dense_grads(model, fired, ["extra"], [])
    u_a, u_s = eigenvectors["extra"]
    accumulated = torch.zeros(
        u_s.shape[0], u_a.shape[0], dtype=torch.float64
    )
    for grads in per_item:
        projected = u_s.T @ (block * grads["extra"]) @ u_a
        accumulated += projected.square()
    lam_ref = accumulated / len(fired)
    torch.testing.assert_close(
        factors.linears["extra"]["lam"].double(), lam_ref, rtol=1e-5, atol=1e-8
    )
    meta = json.loads((tmp_path / "ekfac_meta.json").read_text())
    assert meta["samples_lambda"] == len(items)
    assert meta["lambda_item_counts"] == {
        "extra": len(fired), "head": len(items), "mid": len(items)
    }


def test_top_two_singular_values_match_svd():
    generator = torch.Generator().manual_seed(11)
    matrix = torch.rand(5, 3, generator=generator).double() + 0.1
    sigma1, sigma2 = _top_two_singular_values(matrix)
    reference = torch.linalg.svdvals(matrix)
    assert abs(sigma1 - float(reference[0])) < 1e-9
    assert abs(sigma2 - float(reference[1])) < 1e-9
    rank1 = torch.outer(
        (torch.rand(4, generator=generator) + 0.1).double(),
        (torch.rand(6, generator=generator) + 0.1).double(),
    )
    sigma1, sigma2 = _top_two_singular_values(rank1)
    assert sigma1 > 0 and sigma2 / sigma1 < 1e-9


def test_load_ekfac_is_memory_mapped_and_value_identical(tmp_path):
    """Factors load as file-backed (COW) mappings with eager-identical values.

    Eager loading materialized ~164 GB of anonymous RAM per stage at full
    12B coverage (3 stages OOM-killed pod run 20260819T095144Z); the loader
    must keep factor bytes file-backed. COW semantics also mean an in-memory
    write never reaches the artifact file.
    """
    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    manifest = make_artifact(tmp_path, model)
    factors = load_ekfac(tmp_path, manifest)
    name = next(iter(factors.linears))
    base = tmp_path / "linear" / name.replace(".", "__")
    for key in ("U_A", "U_S", "lam"):
        eager = torch.from_numpy(np.load(base / f"{key}.npy"))
        assert torch.equal(factors.linears[name][key], eager)
    assert torch.equal(
        factors.diag_v,
        torch.from_numpy(np.load(tmp_path / "diag" / "v.npy")).reshape(-1),
    )
    # COW: mutating the loaded tensor must not write through to the artifact.
    on_disk_before = np.load(base / "lam.npy").copy()
    factors.linears[name]["lam"].mul_(2.0)
    assert np.array_equal(np.load(base / "lam.npy"), on_disk_before)
    # A fresh load still sees the original bytes.
    fresh = load_ekfac(tmp_path, manifest)
    assert torch.equal(fresh.linears[name]["lam"], torch.from_numpy(on_disk_before))


def test_apply_ekfac_and_curvature_identical_between_mmap_and_eager(tmp_path):
    from scimt.data_attribution.source import EKFACCurvature

    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    manifest = make_artifact(tmp_path, model)
    mapped = load_ekfac(tmp_path, manifest)
    # Build an eager twin by round-tripping every tensor through a copy.
    eager = EKFACFactors(
        {
            name: {key: tensor.clone() for key, tensor in factor.items()}
            for name, factor in mapped.linears.items()
        },
        mapped.diag_v.clone(),
        mapped.diag_index,
        mapped.snapshot,
        mapped.preconditioner,
    )
    vector = torch.randn(
        manifest.included_numel, generator=torch.Generator().manual_seed(23)
    )
    assert torch.equal(
        apply_ekfac(vector, mapped, manifest, damping_scale=0.03, power=-0.5),
        apply_ekfac(vector, eager, manifest, damping_scale=0.03, power=-0.5),
    )
    rows = torch.randn(
        3, manifest.included_numel, generator=torch.Generator().manual_seed(29)
    )
    out_mapped = EKFACCurvature(mapped, manifest).apply_fn(
        rows, lambda lam: 1.0 / (lam + 0.05)
    )
    out_eager = EKFACCurvature(eager, manifest).apply_fn(
        rows, lambda lam: 1.0 / (lam + 0.05)
    )
    assert torch.equal(out_mapped, out_eager)


def test_load_ekfac_cross_mode_refusals(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    manifest = make_artifact(tmp_path, model)
    with pytest.raises(ValueError, match="'ekfac' factor set"):
        load_ekfac(tmp_path, manifest, expected_mode="ekfac_adam")
    meta = json.loads((tmp_path / "ekfac_meta.json").read_text())
    meta["preconditioner"] = _provenance()
    (tmp_path / "ekfac_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="'ekfac_adam' factor set"):
        load_ekfac(tmp_path, manifest)
    factors = load_ekfac(tmp_path, manifest, expected_mode="ekfac_adam")
    assert factors.mode == "ekfac_adam"
    assert factors.preconditioner == _provenance()
    with pytest.raises(ValueError, match="expected_mode"):
        load_ekfac(tmp_path, manifest, expected_mode="raw")


def test_conditioner_validation_errors(monkeypatch, tmp_path):
    model = _CondTiny()
    manifest = ParameterManifest.from_model(model, "tiny")
    _fake_kronfluence(
        monkeypatch,
        {
            "mid": (_orthonormal(4, 5), _orthonormal(3, 6)),
            "head": (_orthonormal(3, 7), _orthonormal(7, 8)),
        },
        {},
    )
    config = {"samples": 2, "max_positions_per_sequence": 1}
    good = torch.rand(manifest.included_numel) + 0.5

    def run(conditioner, match):
        with pytest.raises((ValueError, TypeError), match=match):
            fit_ekfac(model, _CondBatches(), manifest, config, tmp_path, conditioner)

    run("nope", "EKFACConditioner")
    run(EKFACConditioner(values=torch.rand(3) + 0.5, provenance=_provenance()), "shape")
    bad = good.clone()
    bad[0] = 0.0
    run(EKFACConditioner(values=bad, provenance=_provenance()), "strictly positive")
    incomplete = _provenance()
    incomplete.pop("moment_identity_digest")
    run(EKFACConditioner(values=good, provenance=incomplete), "missing keys")
    unserializable = _provenance() | {"extra": {1, 2}}
    run(EKFACConditioner(values=good, provenance=unserializable), "JSON-serializable")
    with pytest.raises(ValueError, match="use_empirical_fisher"):
        fit_ekfac(
            model,
            _CondBatches(),
            manifest,
            config | {"use_empirical_fisher": False},
            tmp_path,
            EKFACConditioner(values=good, provenance=_provenance()),
        )


def test_staged_kronfluence_matches_fit_all_factors_eigenvectors(tmp_path):
    pytest.importorskip("kronfluence")
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import FactorArguments
    from kronfluence.task import Task

    from scimt.data_attribution.ekfac import _causal_token_task, _TokenSampleDataset

    def build():
        torch.manual_seed(0)
        return _CondTiny()

    items = build_ekfac_sample_items(
        _CondBatches(), {"samples": 4, "max_positions_per_sequence": 2}
    )
    dataset = _TokenSampleDataset(items)
    results = {}
    for label in ("all", "staged"):
        model = build()
        task = _causal_token_task(Task, ["mid", "head"])
        prepared = prepare_model(model=model, task=task)
        analyzer = Analyzer(
            analysis_name=f"parity_{label}",
            model=prepared,
            task=task,
            cpu=True,
            output_dir=str(tmp_path / label),
            disable_tqdm=True,
        )
        args = FactorArguments(
            strategy="ekfac",
            use_empirical_fisher=True,
            eigendecomposition_dtype=torch.float64,
        )
        if label == "all":
            analyzer.fit_all_factors(
                factors_name="ekfac",
                dataset=dataset,
                per_device_batch_size=2,
                factor_args=args,
                overwrite_output_dir=True,
            )
        else:
            analyzer.fit_covariance_matrices(
                factors_name="ekfac",
                dataset=dataset,
                per_device_batch_size=2,
                factor_args=args,
                overwrite_output_dir=True,
            )
            analyzer.perform_eigendecomposition(
                factors_name="ekfac", factor_args=args, overwrite_output_dir=True
            )
        results[label] = analyzer.load_eigendecomposition("ekfac")
    for kind in ("activation_eigenvectors", "gradient_eigenvectors"):
        for name in ("mid", "head"):
            torch.testing.assert_close(
                results["staged"][kind][name],
                results["all"][kind][name],
                rtol=0,
                atol=0,
            )


# ============================================================ eigh_device ==
def test_fit_config_eigh_device_validation():
    from scimt.data_attribution.ekfac import _fit_config

    assert _fit_config({})["eigh_device"] == "auto"
    assert _fit_config({"eigh_device": "cpu"})["eigh_device"] == "cpu"
    assert _fit_config({"eigh_device": "cuda"})["eigh_device"] == "cuda"
    with pytest.raises(ValueError, match="eigh_device"):
        _fit_config({"eigh_device": "mps"})


def test_eigh_device_resolution_is_error_loud_without_cuda():
    from scimt.data_attribution.ekfac import _eigh_torch_device

    assert _eigh_torch_device("cpu") == torch.device("cpu")
    if torch.cuda.is_available():
        assert _eigh_torch_device("cuda") == torch.device("cuda")
    else:
        with pytest.raises(ValueError, match="CUDA is not available"):
            _eigh_torch_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gpu_eigh_reconstructs_and_matches_cpu_eigenvalues():
    """cuSOLVER fp64 eigh: reconstruction + eigenvalue parity vs CPU LAPACK.

    Eigenvector BYTES are not compared — the basis is sign/rotation
    ambiguous across backends; the operator (reconstruction) is the
    invariant.
    """
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(96, 96, generator=generator, dtype=torch.float64)
    spd = base @ base.T + 96 * torch.eye(96, dtype=torch.float64)
    values_gpu, vectors_gpu = torch.linalg.eigh(spd.cuda())
    reconstructed = (
        vectors_gpu @ torch.diag(values_gpu) @ vectors_gpu.T
    ).cpu()
    torch.testing.assert_close(reconstructed, spd, rtol=1e-10, atol=1e-8)
    values_cpu, _ = torch.linalg.eigh(spd)
    torch.testing.assert_close(
        values_gpu.cpu(), values_cpu, rtol=1e-10, atol=1e-8
    )


def test_operator_action_invariant_under_degenerate_eigenbases():
    """With repeated eigenvalues, eigh bases legitimately differ between
    backends/runs — but V f(lam) V^T must equal f(M) regardless. This is the
    property the lifted eigendecomposition relies on."""
    generator = torch.Generator().manual_seed(7)
    rotation = torch.linalg.qr(
        torch.randn(8, 8, generator=generator, dtype=torch.float64)
    ).Q
    # Spectrum with an exact 4-fold degeneracy.
    spectrum = torch.tensor(
        [2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 9.0, 13.0], dtype=torch.float64
    )
    matrix = rotation @ torch.diag(spectrum) @ rotation.T
    matrix = (matrix + matrix.T) / 2
    values, vectors = torch.linalg.eigh(matrix)
    for fn in (
        lambda x: x,
        lambda x: 1.0 / (x + 0.03),
        lambda x: torch.sqrt(x),
    ):
        via_basis = vectors @ torch.diag(fn(values)) @ vectors.T
        reference = rotation @ torch.diag(fn(spectrum)) @ rotation.T
        torch.testing.assert_close(via_basis, reference, rtol=1e-9, atol=1e-9)


def _fit_real_kronfluence(tmp_path, label, config, conditioner_seed=None):
    """Run the real fit_ekfac end to end on the tiny fixture."""
    from scimt.data_attribution.ekfac import EKFACConditioner  # noqa: F401

    torch.manual_seed(0)
    model = _CondTiny()
    manifest = ParameterManifest.from_model(model, "tiny")
    conditioner = (
        _conditioner_for(manifest, seed=conditioner_seed)
        if conditioner_seed is not None
        else None
    )
    out = tmp_path / label
    out.mkdir()
    factors = fit_ekfac(
        model, _CondBatches(), manifest, config, out, conditioner=conditioner
    )
    return manifest, factors, out


def _assert_operator_parity(manifest, native, lifted):
    generator = torch.Generator().manual_seed(41)
    vector = torch.randn(manifest.included_numel, generator=generator)
    for damping, power in ((0.03, -1), (0.1, -0.5), (0.01, 0.5)):
        torch.testing.assert_close(
            apply_ekfac(vector, lifted, manifest, damping_scale=damping, power=power),
            apply_ekfac(vector, native, manifest, damping_scale=damping, power=power),
            rtol=1e-8,
            atol=1e-8,
        )


def test_lifted_eigh_raw_pipeline_operator_matches_native(monkeypatch, tmp_path):
    """fit_ekfac with eigh_device set must produce factors whose OPERATOR
    action matches the kronfluence-native path at 1e-8 (eigenvector bytes are
    basis-ambiguous and not compared). The 'cuda' request is monkeypatched to
    resolve to CPU so the lifted wiring runs on CPU-only boxes; on GPU hosts
    the same wiring runs on the real device."""
    pytest.importorskip("kronfluence")
    import scimt.data_attribution.ekfac as ekfac_module

    base_config = {"samples": 6, "max_positions_per_sequence": 3}
    manifest, native, _ = _fit_real_kronfluence(tmp_path, "native", base_config)
    monkeypatch.setattr(
        ekfac_module, "_eigh_torch_device", lambda requested: torch.device("cpu")
    )
    lifted_config = dict(base_config, eigh_device="cuda")
    manifest_l, lifted, out = _fit_real_kronfluence(
        tmp_path, "lifted", lifted_config
    )
    assert manifest_l.digest() == manifest.digest()
    _assert_operator_parity(manifest, native, lifted)
    reports = list(out.rglob("eigh_report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["matrices"]
    for total in (
        "total_eigh_seconds",
        "total_load_seconds",
        "total_transfer_seconds",
        "total_save_seconds",
    ):
        assert report[total] >= 0
    for entry in report["matrices"]:
        assert {"load_seconds", "transfer_seconds", "eigh_seconds"} <= set(entry)
    assert {entry["factor"] for entry in report["matrices"]} == {
        "activation_covariance",
        "gradient_covariance",
    }
    assert {entry["factor"] for entry in report["saves"]} == {
        "activation_eigenvalues",
        "activation_eigenvectors",
        "gradient_eigenvalues",
        "gradient_eigenvectors",
    }


def test_lifted_eigh_conditioned_pipeline_operator_matches_native(
    monkeypatch, tmp_path
):
    """Same operator-parity guarantee for the ekfac_adam (conditioned) path:
    lambdas are refit in whatever orthonormal eigenbasis the lifted step
    produced, so the resulting curvature action must match the native path."""
    pytest.importorskip("kronfluence")
    import scimt.data_attribution.ekfac as ekfac_module

    base_config = {"samples": 6, "max_positions_per_sequence": 3}
    manifest, native, _ = _fit_real_kronfluence(
        tmp_path, "native", base_config, conditioner_seed=23
    )
    assert native.mode == "ekfac_adam"
    monkeypatch.setattr(
        ekfac_module, "_eigh_torch_device", lambda requested: torch.device("cpu")
    )
    lifted_config = dict(base_config, eigh_device="cuda")
    _, lifted, _ = _fit_real_kronfluence(
        tmp_path, "lifted", lifted_config, conditioner_seed=23
    )
    assert lifted.mode == "ekfac_adam"
    _assert_operator_parity(manifest, native, lifted)


def test_lifted_streaming_eigendecomposition_matches_kronfluence_bitwise(tmp_path):
    """The streaming per-key lifted loop must produce byte-identical
    eigendecomposition artifacts to kronfluence's whole-set
    perform_eigendecomposition when both run on the same device/backend —
    including from PARTITIONED covariance fits (kronfluence aggregates
    partitions into the unpartitioned files the stream reads)."""
    pytest.importorskip("kronfluence")
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import FactorArguments
    from kronfluence.task import Task

    from scimt.data_attribution.ekfac import (
        _causal_token_task,
        _lifted_eigendecomposition,
        _TokenSampleDataset,
    )

    items = build_ekfac_sample_items(
        _CondBatches(), {"samples": 4, "max_positions_per_sequence": 2}
    )
    dataset = _TokenSampleDataset(items)
    results = {}
    for label, partitions in (("native", 2), ("lifted", 2)):
        torch.manual_seed(0)
        model = _CondTiny()
        task = _causal_token_task(Task, ["mid", "head"])
        prepared = prepare_model(model=model, task=task)
        analyzer = Analyzer(
            analysis_name=f"stream_{label}",
            model=prepared,
            task=task,
            cpu=True,
            output_dir=str(tmp_path / label),
            disable_tqdm=True,
        )
        args = FactorArguments(
            strategy="ekfac",
            use_empirical_fisher=True,
            covariance_module_partitions=partitions,
            eigendecomposition_dtype=torch.float64,
        )
        analyzer.fit_covariance_matrices(
            factors_name="ekfac",
            dataset=dataset,
            per_device_batch_size=2,
            factor_args=args,
            overwrite_output_dir=True,
        )
        if label == "native":
            analyzer.perform_eigendecomposition(
                factors_name="ekfac", factor_args=args, overwrite_output_dir=True
            )
        else:
            _lifted_eigendecomposition(
                analyzer, prepared, args, torch.device("cpu")
            )
        results[label] = analyzer.load_eigendecomposition("ekfac")
    for kind in (
        "activation_eigenvectors",
        "activation_eigenvalues",
        "gradient_eigenvectors",
        "gradient_eigenvalues",
    ):
        for name in ("mid", "head"):
            torch.testing.assert_close(
                results["lifted"][kind][name],
                results["native"][kind][name],
                rtol=0,
                atol=0,
            )
