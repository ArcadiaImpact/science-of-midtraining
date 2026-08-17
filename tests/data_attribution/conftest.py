"""Collection guard: keep the lean `--extra dev` gate sound without torch.

Most modules in this directory exercise real tensors and require the
`data-attribution` extra. In a lean environment (no torch) they must be
skipped at collection, not fail on `import torch`. The modules named in
`_LEAN_MODULES` are the exception: they pin the lean-import contract itself
and must always run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_LEAN_MODULES = {"test_migration_boundary.py", "test_config.py", "test_cli.py"}

collect_ignore = []
if importlib.util.find_spec("torch") is None:
    collect_ignore = sorted(
        path.name
        for path in Path(__file__).parent.glob("test_*.py")
        if path.name not in _LEAN_MODULES
    )


@pytest.fixture(scope="session", autouse=True)
def _t2_ekfac_interface_compat():
    """TEMPORARY integration scaffolding (remove once T2's ekfac.py lands):
    the runner passes ``expected_mode`` to ``load_ekfac`` per T2's final
    interface; pre-integration, shim the kwarg onto the current signature.
    A no-op when the real parameter exists."""
    if importlib.util.find_spec("torch") is None:
        yield
        return
    import inspect

    import scimt.data_attribution.ekfac as ekfac_mod

    if "expected_mode" in inspect.signature(ekfac_mod.load_ekfac).parameters:
        yield
        return
    real = ekfac_mod.load_ekfac

    def load_with_mode(path, manifest, *, expected_mode="ekfac"):
        return real(path, manifest)

    ekfac_mod.load_ekfac = load_with_mode
    yield
    ekfac_mod.load_ekfac = real
