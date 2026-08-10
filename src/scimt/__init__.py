"""Science of midtraining — survey glue + case-study analysis.

Heavy lifting (synthetic-data generation, quality / cookedness metrics) is
delegated to `aligne`, always as a library import; training runs through the
axolotl backend (`scimt.train.axolotl`, a supervised async subprocess — the
documented carve-out). This package only adds survey-specific glue and
analysis.

The core pipeline is a pure-async library of typed verbs — the caller owns
the event loop; handles (``Dataset`` / ``Checkpoint``) are frozen dataclasses
backed by JSON manifests next to the bytes they point at:

    from scimt import generate, evaluate, load_spec, prepare
    from scimt.train import train

    spec = load_spec("ed")                       # the one stringly entry point
    docs = await generate(spec, "runs/ed")                 # Spec -> Dataset
    data = prepare.filter_rows(docs, "nonempty_text", "runs/ed/prep")
    ckpt = await train(spec, data, "runs/ed/sft")          # Dataset -> Checkpoint
    row  = await evaluate(spec, ckpt)                      # Checkpoint -> metrics

``train`` is NOT re-exported at the top level: ``scimt.train`` is the package,
and a same-named function re-export would be shadowed by the submodule import
machinery. Layout: pipeline stages ``gen`` (with ``gen.health``, the docs-stage
QA battery), ``train``, ``eval`` (with ``eval.trust`` calibration inside and
the ``analysis`` classifiers alongside); everything else lives under ``utils``.

Re-exports are lazy (PEP 562) so ``import scimt`` / ``scimt.spec`` stay
importable without aligne or torch installed (per ``scimt.spec``'s contract).
"""

from typing import Any

__version__ = "0.1.0"
__all__ = ["Checkpoint", "Dataset", "evaluate", "generate", "generate_docs",
           "load_spec", "prepare"]


def __getattr__(name: str) -> Any:
    if name == "generate":
        from .gen import generate

        return generate
    if name == "generate_docs":
        from .gen import generate_docs

        return generate_docs
    if name == "evaluate":
        from .eval.run import evaluate

        return evaluate
    if name == "load_spec":
        from .spec import load_spec

        return load_spec
    if name == "Dataset":
        from .dataset import Dataset

        return Dataset
    if name == "Checkpoint":
        from .train.checkpoint import Checkpoint

        return Checkpoint
    if name == "prepare":
        import importlib

        return importlib.import_module(".prepare", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
