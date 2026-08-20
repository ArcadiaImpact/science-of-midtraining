"""Collection guard for the probing tests (idiom: tests/data_attribution/).

The dev extra ships no torch/safetensors/sklearn; modules that genuinely
need them are skipped at collection in the lean env and run in the ephemeral
guarded env (see src/probing/README.md §verification). Everything else is
lean by design and always collected.
"""

import importlib.util

_GUARDS = {
    "test_probing_cache_io.py": ("torch", "safetensors"),
    "test_probing_probes_handle.py": ("safetensors", "numpy"),
    "test_probing_fit_logistic.py": ("sklearn", "numpy"),
    "test_probing_extract_end_to_end.py": ("torch",),
}

collect_ignore = sorted(
    name
    for name, deps in _GUARDS.items()
    if any(importlib.util.find_spec(dep) is None for dep in deps)
)
