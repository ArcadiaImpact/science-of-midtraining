"""31B leg of the native-render EFT ladder — thin scale shift over the 12B
trainer (eft_12b_native/train_eft_12b.py): same mixture, same render, same
asserts, same LoRA family (r64/a128 v-less exact paths); deltas are the
decoder layer count (60 -> 410 target modules), the training.model key
(gemma4_31b -> plain Gemma4 arch), and the arm checkpoint paths. See
../eft_31b_native/SPEC.md for the delta table; everything else is the 12B
module, imported, not forked.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments/python4/eft_12b_native"))

import train_eft_12b as base  # noqa: E402  (module-level device pin is wanted)

TARGET_LAYERS_31B = 60

# Capture BEFORE main() rebinds base.target_config to the 31B variant —
# reading it through the module afterwards recurses onto ourselves
# (RecursionError, observed live at first 31B train 2026-09-07: phase A's
# dry-run path returns before the config call, so only the full train path
# exercised the rebind).
_orig_target_config = base.target_config


def target_config_31b() -> dict:
    cfg = _orig_target_config()
    cfg["training"]["model"] = "gemma4_31b"
    cfg["training"]["lora"]["target_layers"] = TARGET_LAYERS_31B
    return cfg


def main() -> int:
    # Same CLI, same flow — swap the scale constants before dispatching.
    base.TARGET_LAYERS = TARGET_LAYERS_31B
    base.target_config = target_config_31b
    base.STUDY = "eft_31b_native"
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
