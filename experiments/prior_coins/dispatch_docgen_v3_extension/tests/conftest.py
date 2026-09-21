"""Keep this experiment's module loads from leaking into other test files.

These tests import modules named `run`, `setting`, `audit`, `semantic_review`,
`names_v2` and `run_blocks` from this directory, and `tests/test_dispatch_docgen_v1.py`
imports modules with the same names from ITS directory via lazy fixtures. Whichever
loaded last would otherwise win in `sys.modules`, so the two suites passed or failed
depending on the order pytest visited them. Every test here restores `sys.modules`
for those names and `sys.path` to what they were before it ran.
"""
from __future__ import annotations

import sys

import pytest

SHARED_NAMES = ("setting", "names_v2", "semantic_review", "audit", "run",
                "run_blocks", "backup", "costing")


@pytest.fixture(autouse=True)
def _isolate_experiment_modules():
    saved = {name: sys.modules.get(name) for name in SHARED_NAMES}
    path = list(sys.path)
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    sys.path[:] = path
