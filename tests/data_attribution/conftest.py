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

_LEAN_MODULES = {"test_migration_boundary.py", "test_config.py"}

collect_ignore = []
if importlib.util.find_spec("torch") is None:
    collect_ignore = sorted(
        path.name
        for path in Path(__file__).parent.glob("test_*.py")
        if path.name not in _LEAN_MODULES
    )
