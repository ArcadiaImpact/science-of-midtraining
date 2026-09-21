"""Redraw the figures from results/summary.json without re-reading the stores."""

from __future__ import annotations

import json

from analyze import Config, fig_clause, fig_lengths, fig_training


def main(cfg: Config = Config()) -> None:
    S = json.loads((cfg.out / "summary.json").read_text())
    # json turned the int step keys into strings
    clause = {arm: {int(k): v for k, v in d.items()} for arm, d in S["clause"].items()}
    slices = {arm: {int(k): v for k, v in d.items()} for arm, d in S["slices"].items()}
    fig_lengths(cfg, S["hists"])
    fig_clause(cfg, clause)
    fig_training(cfg, S["training"], slices)
    print("figures redrawn in", cfg.figs)


if __name__ == "__main__":
    main()
