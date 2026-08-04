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
