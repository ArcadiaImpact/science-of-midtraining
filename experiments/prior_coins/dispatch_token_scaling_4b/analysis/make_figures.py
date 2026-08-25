"""Thin entry point: collate an evidence tree, then render every figure.

Usage (experiment-side argparse is allowed; the library stays CLI-free):

    python make_figures.py <run_root> <out_dir>

Writes ``<out_dir>/scored_collated.json`` and the SPEC §10.4 PDF set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo-root namespace packages importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.prior_coins.dispatch_token_scaling_4b.analysis import (  # noqa: E402
    collate,
    figures,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path,
                        help="evidence tree root (<run_root>/<cell>/...)")
    parser.add_argument("out_dir", type=Path,
                        help="output directory for scored_collated.json + PDFs")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    collated_path = args.out_dir / "scored_collated.json"
    doc = collate.collate_to_file(args.run_root, collated_path)
    print(f"[collate] {len(doc['rows'])} scored rows, "
          f"{len(doc['prequential'])} prequential rows, "
          f"{len(doc['cells'])} cells -> {collated_path}")
    for warning in doc["warnings"]:
        print(f"[collate] WARNING: {warning}")

    written = figures.render_all(collated_path, args.out_dir)
    for path in written:
        print(f"[figures] wrote {path}")


if __name__ == "__main__":
    main()
