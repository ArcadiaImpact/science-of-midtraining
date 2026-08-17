"""Library-wide contract for src/probing (mirror of tests/test_scoring_contract.py,
which is deliberately scoped to src/scimt and stays untouched).

Holds the line on: no CLIs (#155 applies repo-wide by policy), no bellhop in
the library (launchers live in experiment runners), and lazy heavy imports
(`import probing` must stay CPU-only).
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "probing"

# Roots that must never be imported at module scope anywhere in the package.
HEAVY_ROOTS = {
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "numpy",
    "scipy",
    "sklearn",
    "safetensors",
    "huggingface_hub",
}


def _library_modules() -> list[Path]:
    paths = sorted(SRC.rglob("*.py"))
    assert paths, f"no python modules under {SRC}"
    return paths


def test_no_cli_entry_points():
    for path in _library_modules():
        src = path.read_text()
        assert "import argparse" not in src, f"{path}: argparse is banned (#155)"


def test_no_bellhop_anywhere():
    # Not even lazily (AST-checked, so docstrings may *mention* the rule):
    # pod launchers are experiment-runner code by convention.
    for path in _library_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            roots = set()
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots = {node.module.split(".")[0]}
            assert "bellhop" not in roots, (
                f"{path}: the probing library must not import bellhop — "
                "launchers live in experiment runners (see src/probing/README.md)"
            )


def _module_scope_imports(tree: ast.Module) -> set[str]:
    """Root names imported at module scope (function/method bodies excluded;
    `if TYPE_CHECKING:` blocks excluded)."""
    roots: set[str] = set()

    def is_type_checking(node: ast.AST) -> bool:
        if not isinstance(node, ast.If):
            return False
        t = node.test
        return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
            isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
        )

    def walk(body):
        for node in body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    roots.add(node.module.split(".")[0])
            elif isinstance(node, (ast.If,)) and not is_type_checking(node):
                walk(node.body)
                walk(node.orelse)
            elif isinstance(node, (ast.Try,)):
                walk(node.body)
                for h in node.handlers:
                    walk(h.body)
                walk(node.orelse)
                walk(node.finalbody)
            elif isinstance(node, (ast.With,)):
                walk(node.body)

    walk(tree.body)
    return roots


def test_no_module_scope_heavy_imports():
    for path in _library_modules():
        tree = ast.parse(path.read_text())
        heavy = _module_scope_imports(tree) & HEAVY_ROOTS
        assert not heavy, (
            f"{path}: module-scope heavy import(s) {sorted(heavy)} — move "
            "inside the function that needs them"
        )


def test_all_matches_lazy_table():
    import probing

    assert sorted(probing.__all__) == sorted(probing._LAZY_EXPORTS)
    # Every lazy name must actually resolve.
    for name in probing.__all__:
        assert getattr(probing, name) is not None


def test_no_export_shadowed_by_a_submodule():
    """Importing a submodule binds it onto the package, permanently bypassing
    __getattr__ for that name — so no export may share a submodule's name
    (this is why the modules are fitting.py/extraction.py, not fit/extract)."""
    import probing

    submodules = {p.stem for p in SRC.glob("*.py") if p.stem != "__init__"}
    shadowed = submodules & set(probing._LAZY_EXPORTS)
    assert not shadowed, f"exports shadowed by submodules: {sorted(shadowed)}"


def test_exports_stay_callables_after_submodule_import():
    """The failure mode itself: access, import the submodule, access again."""
    import importlib

    import probing

    first = probing.fit
    importlib.import_module("probing.fitting")
    importlib.import_module("probing.extraction")
    assert probing.fit is first and callable(probing.fit)
    assert callable(probing.extract)
