"""Regenerate all 31 canonical Dispatch ablations offline (PDF and PNG)."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    for entry in sorted(root.glob("*/src/plot_*.py")):
        print(f"Rendering {entry.parent.parent.name}", flush=True)
        # Isolate Matplotlib global settings so batch and individual renders agree.
        subprocess.run([sys.executable, str(entry)], check=True)


if __name__ == "__main__":
    main()
