"""`import probing` must stay CPU-only: stdlib + pyyaml, no heavy roots.

Run in a clean interpreter (idiom: tests/test_analysis_types.py) because the
surrounding suite may already have numpy in sys.modules via pandas.
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

CODE = """
import sys
import probing
# Touch lazy attrs across the currently shipped submodules.
probing.ExtractConfig
probing.PositionSpec
probing.locate_span
probing.parse_layers
heavy = {
    "torch", "transformers", "peft", "accelerate", "numpy", "scipy",
    "sklearn", "safetensors", "huggingface_hub", "bellhop",
} & set(sys.modules)
assert not heavy, f"eager heavy import: {sorted(heavy)}"
print("lean-ok")
"""


def test_import_probing_is_lean():
    proc = subprocess.run(
        [sys.executable, "-c", CODE],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "lean-ok" in proc.stdout
