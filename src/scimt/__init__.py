"""Science of midtraining — survey glue + case-study analysis.

Heavy lifting (synthetic-data generation, training, serving shims, quality /
cookedness metrics, character training, constitutional auditing) is delegated to
`aligne` and `tinker_cookbook`, always as libraries (never subprocesses). This
package only adds survey-specific glue and analysis.

The core pipeline is a pure-async library — the caller owns the event loop:

    from scimt import generate, evaluate
    from scimt.train import train

    docs = await generate("ed", "runs/ed")                        # spec -> docs
    ckpt = await train("ed", docs["dataset_path"], "runs/ed/sft") # docs -> model
    row  = await evaluate("ed", ckpt["pointer_file"])             # model -> metrics

``train`` is NOT re-exported at the top level: ``scimt.train`` is the package,
and a same-named function re-export would be shadowed by the submodule import
machinery. Layout: pipeline stages ``gen`` (with ``gen.health``, the docs-stage
QA battery), ``train``, ``eval`` (with the ``analysis`` classifiers and
``trust`` calibration alongside); everything else lives under ``utils``.

Re-exports are lazy (PEP 562) so ``import scimt`` / ``scimt.spec`` stay
importable without aligne or tinker installed (per ``scimt.spec``'s contract).
"""

from typing import Any

__version__ = "0.1.0"
__all__ = ["generate", "evaluate"]


def __getattr__(name: str) -> Any:
    if name == "generate":
        from .gen import generate

        return generate
    if name == "evaluate":
        from .eval.run import evaluate

        return evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
