"""Science of midtraining — survey glue + case-study analysis.

Heavy lifting (synthetic-data generation, training, serving shims, quality /
cookedness metrics, character training, constitutional auditing) is delegated to
`aligne` and `tinker_cookbook`, always as libraries (never subprocesses). This
package only adds survey-specific glue and analysis.

The core pipeline is a pure-async library — the caller owns the event loop:

    from scimt import generate, train, evaluate

    docs = await generate("ed", "out/ed")                        # spec -> docs
    ckpt = await train("ed", docs["dataset_path"], "out/ed/sft") # docs -> model
    row  = await evaluate("ed", ckpt["pointer_file"])            # model -> metrics

Re-exports are lazy (PEP 562) so ``import scimt`` / ``scimt.spec`` stay
importable without aligne or tinker installed (per ``scimt.spec``'s contract).
"""

from typing import Any

__version__ = "0.1.0"
__all__ = ["generate", "train", "evaluate"]


def __getattr__(name: str) -> Any:
    if name == "generate":
        from .gen import generate

        return generate
    if name == "train":
        from .training import train

        return train
    if name == "evaluate":
        from .eval.run import evaluate

        return evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
