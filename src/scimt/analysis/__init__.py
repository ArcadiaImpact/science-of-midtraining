"""scimt.analysis — shared effect estimation over per-item eval rows.

Tier 0 (`classical`): Wilson intervals, paired item-level bootstrap, exact
McNemar — stdlib-only consolidations of the estimators previously copied
per-experiment.

Tier 1 (`effects.fit_arm_effects`): explanatory-IRT / hierarchical logistic
arm contrasts — `y ~ arm + (1|item)` with optional `(arm|item)` DIF slopes
and `(1|cluster)` testlet terms; bernoulli / binomial / ordered-logistic /
unordered-categorical likelihoods. Needs the `irt` extra (numpyro), imported
lazily so `import scimt` stays CPU-only.

Key literature: De Boeck & Wilson 2004 (Explanatory Item Response Models);
Gilbert, Kim & Miratrix JEBS 2023 and Gilbert et al. JPAM 2025
(arXiv:2405.00161) — item-level heterogeneous treatment effects; Miller 2024
(arXiv:2411.00640) — clustered/paired eval standard errors; Rodriguez et al.
ACL 2021 (IRT leaderboards); tinyBenchmarks (arXiv:2402.14992), metabench
(ICLR 2025), ATLAS (arXiv:2511.04689) — fixed-item-bank scoring and why it
needs hundreds of calibration respondents; Schroeders & Gnambs 2025 (AMPPS)
— IRT sample-size planning.
"""

from __future__ import annotations

__all__ = [
    "ItemRow",
    "EffectConfig",
    "EffectFit",
    "ArmEffect",
    "fit_arm_effects",
    "wilson_interval",
    "paired_bootstrap_delta",
    "mcnemar_exact",
    "discordant_counts",
    "rows_from_stores",
    "rows_from_records",
    "collapse_repeats",
    "load_store",
    "item_key",
]

_LAZY = {
    "ItemRow": "scimt.analysis.types",
    "EffectConfig": "scimt.analysis.types",
    "EffectFit": "scimt.analysis.types",
    "ArmEffect": "scimt.analysis.types",
    "fit_arm_effects": "scimt.analysis.effects",
    "wilson_interval": "scimt.analysis.classical",
    "paired_bootstrap_delta": "scimt.analysis.classical",
    "mcnemar_exact": "scimt.analysis.classical",
    "discordant_counts": "scimt.analysis.classical",
    "rows_from_stores": "scimt.analysis.rows",
    "rows_from_records": "scimt.analysis.rows",
    "collapse_repeats": "scimt.analysis.rows",
    "load_store": "scimt.analysis.rows",
    "item_key": "scimt.analysis.rows",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
