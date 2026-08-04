import importlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from scimt.data_attribution.ekfac import apply_ekfac, fit_ekfac, load_ekfac
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


@pytest.mark.skipif(
    not __import__("pathlib").Path("/workspace/gradient-kernel").exists(),
    reason="upstream checkout unavailable",
)
def test_cross_repository_golden_ekfac(tmp_path):
    upstream = __import__("pathlib").Path("/workspace/gradient-kernel")
    subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "merge-base",
            "--is-ancestor",
            "ca9689a497b921dc516feb663a83269c4a588bbc",
            "HEAD",
        ],
        check=True,
    )
    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    manifest = make_artifact(tmp_path, model)
    factors = load_ekfac(tmp_path, manifest)
    vector = torch.randn(
        manifest.included_numel, generator=torch.Generator().manual_seed(31)
    )
    sys.path.insert(0, str(upstream / "src"))
    try:
        upstream_ek = importlib.import_module(
            "preconditioned_gradient_kernels.curvature.ekfac_apply"
        )
        upstream_manifest = importlib.import_module(
            "preconditioned_gradient_kernels.parameter_manifest"
        ).ParameterManifest.load(tmp_path)
        upstream_factors = upstream_ek.load_ekfac_factors(tmp_path, upstream_manifest)
        for power in (-1, -0.5, 0.5):
            torch.testing.assert_close(
                apply_ekfac(vector, factors, manifest, damping_scale=0.03, power=power),
                upstream_ek.apply_ekfac(
                    vector,
                    upstream_factors,
                    upstream_manifest,
                    damping_scale=0.03,
                    power=power,
                ),
                rtol=1e-6,
                atol=1e-6,
            )
        upstream_whiten = importlib.import_module(
            "preconditioned_gradient_kernels.logra.whiten"
        )
        rows = torch.tensor([[1.0, 2.0, -1.0], [2.0, -1.0, 3.0]])
        slices = {"projection": slice(0, 3)}
        fishers = {"projection": rows.double().T @ rows.double() / 2}
        torch.testing.assert_close(
            whiten_rows(rows, slices, fishers, damping_scale=0.1),
            upstream_whiten.whiten_rows(rows, slices, fishers, damping_scale=0.1),
            rtol=2e-6,
            atol=2e-6,
        )
    finally:
        sys.path.remove(str(upstream / "src"))
