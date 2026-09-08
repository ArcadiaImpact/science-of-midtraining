"""Standard Axolotl training, with final model export disabled for this probe.

This in-process seam touches only final export, never loading or optimization.
The useful artifacts are telemetry/configs/logs, not these short-run weights.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys


def main():
    if importlib.metadata.version("axolotl") != "0.17.0":
        raise RuntimeError("benchmark export seam is validated only for axolotl 0.17.0")
    train_module = importlib.import_module("axolotl.train")
    if not callable(getattr(train_module, "save_trained_model", None)):
        raise RuntimeError("Axolotl final-export seam moved")
    train_module.save_trained_model = lambda cfg, trainer, model: None
    from axolotl.cli.train import do_cli

    path = sys.argv[1]
    sys.argv = [sys.argv[0]]
    do_cli(path)


if __name__ == "__main__":
    main()
