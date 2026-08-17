"""4B scale-up figures, in the exact 12B forms.

Thin wrapper: the implementation moved to :mod:`plot_scaleup` when the 27B leg
needed the same figures at a second size. The entry point and the output paths
quoted in RESULTS_4B.md are unchanged.

Run: ``uv run --extra dev python experiments/prior_coins/dispatch_scaleup/plot_4b.py``
"""

from __future__ import annotations

from experiments.prior_coins.dispatch_scaleup import plot_scaleup


def main() -> None:
    plot_scaleup.render("4b")


if __name__ == "__main__":
    main()
