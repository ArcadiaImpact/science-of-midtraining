"""Explanatory-IRT arm contrasts: `fit_arm_effects(rows, config) -> EffectFit`.

Hierarchical logistic (or ordered / multinomial logit) model of per-item eval
rows: ``y ~ arm + (1|item)`` with optional ``(arm|item)`` DIF slopes,
``(1|cluster)`` testlet terms, and ``(1|seed)`` intercepts. The base arm is
pinned at zero, so each treatment coefficient is the arm-vs-base contrast on
the model's logit scale.

Heavy dependencies (numpy / jax / numpyro) are imported lazily inside
functions so ``import scimt`` stays CPU-only; install the ``irt`` extra to
use this module. All validation raises happen before any MCMC is run;
misfit (R-hat, ESS, divergences) warns but never raises.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Sequence

from .types import ArmEffect, EffectConfig, EffectFit, ItemRow

_IMPORT_ERROR_MSG = "fit_arm_effects needs the irt extra: pip install 'scimt[irt]'"

_MISSING_ID_SAMPLE = 5  # how many missing item ids to name per arm in errors


# ---------------------------------------------------------------------------
# validation (pure stdlib — runs before any heavy import or MCMC)
# ---------------------------------------------------------------------------


def _validate_rows(rows: Sequence[ItemRow], config: EffectConfig) -> bool:
    """Validate rows against config; return the resolved seed-effect flag."""
    if not rows:
        raise ValueError("fit_arm_effects needs at least one row")

    arms = sorted({r.arm for r in rows})
    if len(arms) < 2:
        raise ValueError(
            f"fit_arm_effects needs at least 2 arms, got {arms!r}"
        )
    if config.base_arm not in arms:
        raise ValueError(
            f"base_arm {config.base_arm!r} not present in rows; arms are {arms!r}"
        )

    # item sets must cross all arms
    items_by_arm: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        items_by_arm[r.arm].add(r.item_id)
    all_items = set().union(*items_by_arm.values())
    missing_msgs = []
    for arm in arms:
        missing = sorted(all_items - items_by_arm[arm])
        if missing:
            sample = missing[:_MISSING_ID_SAMPLE]
            more = f" (+{len(missing) - len(sample)} more)" if len(missing) > len(sample) else ""
            missing_msgs.append(f"arm {arm!r} missing items {sample}{more}")
    if missing_msgs:
        raise ValueError(
            "item sets do not cross all arms: " + "; ".join(missing_msgs)
        )

    # likelihood-specific outcome checks
    lk = config.likelihood
    if lk == "bernoulli":
        for r in rows:
            if r.y not in (0, 1) or r.n != 1:
                raise ValueError(
                    f"bernoulli rows need y in {{0, 1}} and n == 1; got "
                    f"y={r.y!r}, n={r.n!r} for item {r.item_id!r} (arm {r.arm!r})"
                )
    elif lk == "binomial":
        for r in rows:
            if r.n < 1 or r.y < 0 or r.y > r.n:
                raise ValueError(
                    f"binomial rows need n >= 1 and 0 <= y <= n; got y={r.y!r}, "
                    f"n={r.n!r} for item {r.item_id!r} (arm {r.arm!r})"
                )
    elif lk in ("ordered", "categorical"):
        for r in rows:
            if r.y != int(r.y) or r.y < 0 or r.n != 1:
                raise ValueError(
                    f"{lk} rows need non-negative integer levels and n == 1 "
                    f"(one observation per row); got y={r.y!r}, n={r.n!r} "
                    f"for item {r.item_id!r} (arm {r.arm!r})"
                )
        levels = {int(r.y) for r in rows}
        floor = 3 if lk == "categorical" else 2
        if len(levels) < floor:
            raise ValueError(
                f"{lk} likelihood needs at least {floor} distinct levels; "
                f"got {sorted(levels)!r}"
            )

    if config.cluster_effect and any(r.cluster is None for r in rows):
        raise ValueError(
            "cluster_effect=True but some rows have cluster=None"
        )

    # resolve the seed effect
    seeds = {r.seed for r in rows if r.seed is not None}
    if config.seed_effect == "on":
        if any(r.seed is None for r in rows):
            raise ValueError("seed_effect='on' but some rows have seed=None")
        seed_on = True
    elif config.seed_effect == "off":
        seed_on = False
    else:  # auto
        seed_on = len(seeds) >= 2 and all(r.seed is not None for r in rows)
        if not seed_on and seeds:
            if len(seeds) == 1:
                # a single distinct seed makes the term unidentifiable
                # (confounded with the intercept) — nothing to force
                warnings.warn(
                    "seed_effect='auto' resolved off: rows carry only one "
                    "distinct seed, so a seed term is unidentifiable",
                    UserWarning,
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    f"seed_effect='auto' resolved off: {len(seeds)} distinct "
                    f"seeds but some rows have seed=None; pass "
                    f"seed_effect='on' (after filling seeds) to force the term",
                    UserWarning,
                    stacklevel=3,
                )

    # inconsistent cluster labels for one item: the fit uses per-row clusters
    # and stays correct, but the item table reports one label per item — warn
    item_clusters: dict[str, set] = defaultdict(set)
    for r in rows:
        item_clusters[r.item_id].add(r.cluster)
    inconsistent = sorted(i for i, cs in item_clusters.items() if len(cs) > 1)
    if inconsistent:
        warnings.warn(
            f"{len(inconsistent)} item(s) carry inconsistent cluster labels "
            f"across rows (e.g. {inconsistent[:_MISSING_ID_SAMPLE]}); the item "
            f"table reports the first-seen label",
            UserWarning,
            stacklevel=3,
        )

    # unequal repeat counts per item across arms: warn, proceed
    reps: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        reps[(r.arm, r.item_id)] += 1
    unequal = sorted(
        item
        for item in all_items
        if len({reps[(arm, item)] for arm in arms}) > 1
    )
    if unequal:
        sample = unequal[:_MISSING_ID_SAMPLE]
        more = f" (+{len(unequal) - len(sample)} more)" if len(unequal) > len(sample) else ""
        warnings.warn(
            f"unequal repeat counts per item across arms for {len(unequal)} "
            f"item(s), e.g. {sample}{more}; proceeding",
            UserWarning,
            stacklevel=3,
        )

    return seed_on


# ---------------------------------------------------------------------------
# design (stdlib index building; arrays are created later by the caller)
# ---------------------------------------------------------------------------


class _Design:
    """Index arrays (as python lists) plus label lists for one fit."""

    def __init__(self, rows: Sequence[ItemRow], config: EffectConfig, seed_on: bool):
        self.likelihood = config.likelihood
        base = config.base_arm
        self.arms: list[str] = [base] + sorted({r.arm for r in rows} - {base})
        self.items: list[str] = sorted({r.item_id for r in rows})
        arm_pos = {a: i for i, a in enumerate(self.arms)}
        item_pos = {it: i for i, it in enumerate(self.items)}

        self.clusters: list[str] | None = None
        cluster_pos: dict[str, int] = {}
        if config.cluster_effect:
            self.clusters = sorted({r.cluster for r in rows})  # type: ignore[arg-type]
            cluster_pos = {c: i for i, c in enumerate(self.clusters)}

        self.seeds: list[str] | None = None
        seed_pos: dict[str, int] = {}
        if seed_on:
            self.seeds = sorted({r.seed for r in rows})  # type: ignore[arg-type]
            seed_pos = {s: i for i, s in enumerate(self.seeds)}

        self.categories: list[str] | None = None
        if config.likelihood in ("ordered", "categorical"):
            raw_levels = sorted({int(r.y) for r in rows})
            self.categories = [str(v) for v in raw_levels]
            level_pos = {v: i for i, v in enumerate(raw_levels)}

        self.arm_idx: list[int] = []
        self.item_idx: list[int] = []
        self.cluster_idx: list[int] = []
        self.seed_idx: list[int] = []
        self.y: list[float] = []
        self.n: list[int] = []
        # keep the item's cluster label for the item table (first seen wins)
        self.item_cluster: dict[str, str | None] = {}
        for r in rows:
            self.arm_idx.append(arm_pos[r.arm])
            self.item_idx.append(item_pos[r.item_id])
            if config.cluster_effect:
                self.cluster_idx.append(cluster_pos[r.cluster])
            if seed_on:
                self.seed_idx.append(seed_pos[r.seed])
            if config.likelihood in ("ordered", "categorical"):
                self.y.append(float(level_pos[int(r.y)]))
            else:
                self.y.append(float(r.y))
            self.n.append(int(r.n))
            self.item_cluster.setdefault(r.item_id, r.cluster)


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def _summ(draws) -> dict:
    """Equal-tailed 95% posterior summary as plain floats (JSON-safe)."""
    import numpy as np

    arr = np.asarray(draws, dtype=float)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr)),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def _sigmoid(x):
    import numpy as np

    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _softmax(logits):
    import numpy as np

    x = np.asarray(logits, dtype=float)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _observed_binary(rows: Sequence[ItemRow], arm: str) -> tuple[float, int]:
    succ = sum(r.y for r in rows if r.arm == arm)
    tot = sum(r.n for r in rows if r.arm == arm)
    return (succ / tot if tot else 0.0), int(tot)


def _observed_dist(
    rows: Sequence[ItemRow], arm: str, categories: list[str]
) -> tuple[dict, int]:
    counts = {c: 0 for c in categories}
    tot = 0
    for r in rows:
        if r.arm == arm:
            counts[str(int(r.y))] += 1
            tot += 1
    dist = {c: (counts[c] / tot if tot else 0.0) for c in categories}
    return dist, tot


def _summarize(
    samples: dict,
    design: _Design,
    config: EffectConfig,
    rows: Sequence[ItemRow],
    diagnostics: dict,
) -> EffectFit:
    import numpy as np

    arms = design.arms
    n_treat = len(arms) - 1
    lk = config.likelihood
    beta_key = "beta_cat_t" if lk == "categorical" else "beta_t"
    beta_t = np.asarray(samples[beta_key], dtype=float)

    # derived item effects (non-centered params are stored raw)
    if lk == "categorical":
        b_item = (
            np.asarray(samples["b_cat_raw"], dtype=float)
            * np.asarray(samples["sigma_item_cat"], dtype=float)[:, None, :]
        )  # (D, I, C-1)
        s_key, sig_key = "s_cat_raw", "sigma_slope_cat"
    else:
        b_item = (
            np.asarray(samples["sigma_item"], dtype=float)[:, None]
            * np.asarray(samples["b_raw"], dtype=float)
        )  # (D, I)
        s_key, sig_key = "s_raw", "sigma_slope"

    slopes = None
    if config.item_slope:
        sig = np.asarray(samples[sig_key], dtype=float)
        raw = np.asarray(samples[s_key], dtype=float)
        if lk == "categorical":
            slopes = sig[:, :, None, :] * raw  # (D, A-1, I, C-1)
        else:
            slopes = sig[:, :, None] * raw  # (D, A-1, I)

    categories = design.categories
    arm_effects: dict[str, ArmEffect] = {}
    heterogeneity: dict[str, dict] = {}
    for t in range(n_treat):
        arm = arms[t + 1]
        het_sd: dict | None = None
        if config.item_slope:
            sig = np.asarray(samples[sig_key], dtype=float)
            if lk == "categorical":
                het_sd = {
                    categories[m + 1]: _summ(sig[:, t, m])
                    for m in range(sig.shape[2])
                }
            else:
                het_sd = _summ(sig[:, t])
            heterogeneity[arm] = het_sd

        if lk in ("bernoulli", "binomial"):
            alpha = np.asarray(samples["alpha"], dtype=float)
            delta_rate = _summ(_sigmoid(alpha + beta_t[:, t]) - _sigmoid(alpha))
            base_rate, base_n = _observed_binary(rows, arms[0])
            arm_rate, arm_n = _observed_binary(rows, arm)
            arm_effects[arm] = ArmEffect(
                arm=arm,
                base_arm=arms[0],
                scale="logit",
                delta_logodds=_summ(beta_t[:, t]),
                delta_rate=delta_rate,
                by_category=None,
                observed={
                    "base_rate": float(base_rate),
                    "arm_rate": float(arm_rate),
                    "base_n": base_n,
                    "arm_n": arm_n,
                },
                heterogeneity_sd=het_sd,
            )
        elif lk == "ordered":
            assert categories is not None
            kappa = np.asarray(samples["kappa"], dtype=float)  # (D, C-1)
            C = kappa.shape[1] + 1

            def level_probs(eta):
                cum = _sigmoid(kappa - eta[:, None])  # (D, C-1)
                probs = np.empty((cum.shape[0], C))
                probs[:, 0] = cum[:, 0]
                for m in range(1, C - 1):
                    probs[:, m] = cum[:, m] - cum[:, m - 1]
                probs[:, C - 1] = 1.0 - cum[:, C - 2]
                return probs

            p_base = level_probs(np.zeros(kappa.shape[0]))
            p_arm = level_probs(beta_t[:, t])
            delta = p_arm - p_base  # per-draw deltas sum to 0 across levels
            by_category = {
                categories[m]: {
                    "delta_logodds": None,
                    "delta_prob": _summ(delta[:, m]),
                }
                for m in range(C)
            }
            base_dist, base_n = _observed_dist(rows, arms[0], categories)
            arm_dist, arm_n = _observed_dist(rows, arm, categories)
            arm_effects[arm] = ArmEffect(
                arm=arm,
                base_arm=arms[0],
                scale="latent-logit",
                delta_logodds=_summ(beta_t[:, t]),
                delta_rate=None,
                by_category=by_category,
                observed={
                    "base_dist": base_dist,
                    "arm_dist": arm_dist,
                    "base_n": base_n,
                    "arm_n": arm_n,
                },
                heterogeneity_sd=het_sd,
            )
        else:  # categorical
            assert categories is not None
            alpha_cat = np.asarray(samples["alpha_cat"], dtype=float)  # (D, C-1)
            D = alpha_cat.shape[0]
            zeros = np.zeros((D, 1))
            logits_base = np.concatenate([zeros, alpha_cat], axis=1)
            logits_arm = np.concatenate([zeros, alpha_cat + beta_t[:, t, :]], axis=1)
            delta = _softmax(logits_arm) - _softmax(logits_base)  # (D, C)
            by_category = {}
            for m, label in enumerate(categories):
                by_category[label] = {
                    "delta_logodds": _summ(beta_t[:, t, m - 1]) if m > 0 else None,
                    "delta_prob": _summ(delta[:, m]),
                }
            base_dist, base_n = _observed_dist(rows, arms[0], categories)
            arm_dist, arm_n = _observed_dist(rows, arm, categories)
            arm_effects[arm] = ArmEffect(
                arm=arm,
                base_arm=arms[0],
                scale="multinomial-logit",
                delta_logodds=None,
                delta_rate=None,
                by_category=by_category,
                observed={
                    "base_dist": base_dist,
                    "arm_dist": arm_dist,
                    "base_n": base_n,
                    "arm_n": arm_n,
                },
                heterogeneity_sd=het_sd,
            )

    # per-item table: difficulty + DIF slopes
    item_table: list[dict] = []
    for i, item_id in enumerate(design.items):
        if lk == "categorical":
            assert categories is not None
            difficulty: dict = {
                categories[m + 1]: _summ(b_item[:, i, m])
                for m in range(b_item.shape[2])
            }
            dif = None
            if slopes is not None:
                dif = {
                    arms[t + 1]: {
                        categories[m + 1]: _summ(slopes[:, t, i, m])
                        for m in range(slopes.shape[3])
                    }
                    for t in range(n_treat)
                }
        else:
            difficulty = _summ(b_item[:, i])
            dif = None
            if slopes is not None:
                dif = {
                    arms[t + 1]: _summ(slopes[:, t, i]) for t in range(n_treat)
                }
        item_table.append(
            {
                "item_id": item_id,
                "cluster": design.item_cluster.get(item_id),
                "difficulty": difficulty,
                "dif": dif,
            }
        )

    rows_per_arm: dict[str, int] = defaultdict(int)
    trials_per_arm: dict[str, int] = defaultdict(int)
    for r in rows:
        rows_per_arm[r.arm] += 1
        trials_per_arm[r.arm] += int(r.n)
    n = {
        "items": len(design.items),
        "rows_per_arm": dict(rows_per_arm),
        "trials_per_arm": dict(trials_per_arm),
        "clusters": len(design.clusters) if design.clusters is not None else None,
        "seeds": len(design.seeds) if design.seeds is not None else None,
    }

    return EffectFit(
        arm_effects=arm_effects,
        item_table=item_table,
        heterogeneity=heterogeneity,
        diagnostics=diagnostics,
        n=n,
        config=config,
        categories=categories,
    )


# ---------------------------------------------------------------------------
# model + inference
# ---------------------------------------------------------------------------


def _run_mcmc(design: _Design, config: EffectConfig, seed_on: bool):
    """Lazy-import numpyro/jax, build the model closure, run NUTS."""
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.distributions import transforms
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_IMPORT_ERROR_MSG) from exc

    arms = design.arms
    A = len(arms)
    I = len(design.items)
    lk = config.likelihood
    C = len(design.categories) if design.categories is not None else None

    arm_idx = jnp.asarray(design.arm_idx, dtype=jnp.int32)
    item_idx = jnp.asarray(design.item_idx, dtype=jnp.int32)
    y_arr = jnp.asarray(design.y)
    n_arr = jnp.asarray(design.n)
    cluster_idx = (
        jnp.asarray(design.cluster_idx, dtype=jnp.int32)
        if config.cluster_effect
        else None
    )
    seed_idx = jnp.asarray(design.seed_idx, dtype=jnp.int32) if seed_on else None
    K = len(design.clusters) if design.clusters is not None else 0
    S = len(design.seeds) if design.seeds is not None else 0

    def model():
        if lk == "categorical":
            assert C is not None
            beta_cat_t = numpyro.sample(
                "beta_cat_t", dist.Normal(0, 2.5).expand([A - 1, C - 1]).to_event(2)
            )
            beta_cat = jnp.concatenate(
                [jnp.zeros((1, C - 1)), beta_cat_t], axis=0
            )  # (A, C-1)
            alpha_cat = numpyro.sample(
                "alpha_cat", dist.Normal(0, 2.5).expand([C - 1]).to_event(1)
            )
            sigma_item_cat = numpyro.sample(
                "sigma_item_cat", dist.HalfNormal(1.0).expand([C - 1]).to_event(1)
            )
            b_cat_raw = numpyro.sample(
                "b_cat_raw", dist.Normal(0, 1).expand([I, C - 1]).to_event(2)
            )
            b_item_cat = b_cat_raw * sigma_item_cat  # (I, C-1)
            eta = alpha_cat + beta_cat[arm_idx] + b_item_cat[item_idx]  # (R, C-1)
            if config.item_slope:
                sigma_slope_cat = numpyro.sample(
                    "sigma_slope_cat",
                    dist.HalfNormal(1.0).expand([A - 1, C - 1]).to_event(2),
                )
                s_cat_raw = numpyro.sample(
                    "s_cat_raw",
                    dist.Normal(0, 1).expand([A - 1, I, C - 1]).to_event(3),
                )
                s = jnp.concatenate(
                    [jnp.zeros((1, I, C - 1)), sigma_slope_cat[:, None, :] * s_cat_raw],
                    axis=0,
                )  # (A, I, C-1)
                eta = eta + s[arm_idx, item_idx]
            if config.cluster_effect:
                sigma_cluster = numpyro.sample(
                    "sigma_cluster", dist.HalfNormal(1.0).expand([C - 1]).to_event(1)
                )
                c_raw = numpyro.sample(
                    "c_raw", dist.Normal(0, 1).expand([K, C - 1]).to_event(2)
                )
                eta = eta + (c_raw * sigma_cluster)[cluster_idx]
            if seed_on:
                sigma_seed = numpyro.sample(
                    "sigma_seed", dist.HalfNormal(1.0).expand([C - 1]).to_event(1)
                )
                u_raw = numpyro.sample(
                    "u_raw", dist.Normal(0, 1).expand([S, C - 1]).to_event(2)
                )
                eta = eta + (u_raw * sigma_seed)[seed_idx]
            logits = jnp.concatenate(
                [jnp.zeros((eta.shape[0], 1)), eta], axis=1
            )  # (R, C) — reference category pinned at 0
            numpyro.sample(
                "obs",
                dist.CategoricalLogits(logits=logits),
                obs=y_arr.astype(jnp.int32),
            )
            return

        beta_t = numpyro.sample(
            "beta_t", dist.Normal(0, 2.5).expand([A - 1]).to_event(1)
        )
        beta = jnp.concatenate([jnp.zeros(1), beta_t])  # base pinned at 0
        sigma_item = numpyro.sample("sigma_item", dist.HalfNormal(1.0))
        b_raw = numpyro.sample("b_raw", dist.Normal(0, 1).expand([I]).to_event(1))
        b_item = sigma_item * b_raw  # non-centered
        eta = beta[arm_idx] + b_item[item_idx]
        if lk != "ordered":
            # ordered drops alpha — the cutpoints absorb location
            alpha = numpyro.sample("alpha", dist.Normal(0, 2.5))
            eta = eta + alpha
        if config.item_slope:
            sigma_slope = numpyro.sample(
                "sigma_slope", dist.HalfNormal(1.0).expand([A - 1]).to_event(1)
            )
            s_raw = numpyro.sample(
                "s_raw", dist.Normal(0, 1).expand([A - 1, I]).to_event(2)
            )
            s = jnp.concatenate(
                [jnp.zeros((1, I)), sigma_slope[:, None] * s_raw], axis=0
            )  # (A, I)
            eta = eta + s[arm_idx, item_idx]
        if config.cluster_effect:
            sigma_cluster = numpyro.sample("sigma_cluster", dist.HalfNormal(1.0))
            c_raw = numpyro.sample(
                "c_raw", dist.Normal(0, 1).expand([K]).to_event(1)
            )
            eta = eta + (sigma_cluster * c_raw)[cluster_idx]
        if seed_on:
            sigma_seed = numpyro.sample("sigma_seed", dist.HalfNormal(1.0))
            u_raw = numpyro.sample(
                "u_raw", dist.Normal(0, 1).expand([S]).to_event(1)
            )
            eta = eta + (sigma_seed * u_raw)[seed_idx]

        if lk == "bernoulli":
            numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=y_arr)
        elif lk == "binomial":
            numpyro.sample(
                "obs", dist.Binomial(total_count=n_arr, logits=eta), obs=y_arr
            )
        else:  # ordered
            assert C is not None
            kappa = numpyro.sample(
                "kappa",
                dist.TransformedDistribution(
                    dist.Normal(0, 1).expand([C - 1]).to_event(1),
                    transforms.OrderedTransform(),
                ),
            )
            numpyro.sample(
                "obs",
                dist.OrderedLogistic(predictor=eta, cutpoints=kappa),
                obs=y_arr.astype(jnp.int32),
            )

    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=config.warmup,
        num_samples=config.draws,
        num_chains=config.chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(config.seed), extra_fields=("diverging",))
    return mcmc


def _diagnose(mcmc) -> dict:
    """Convergence diagnostics; warn (never raise) on misfit."""
    import numpy as np
    import numpyro

    site_stats = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    rhats: list[float] = []
    ess: list[float] = []
    for stats in site_stats.values():
        r = np.asarray(stats["r_hat"], dtype=float)
        e = np.asarray(stats["n_eff"], dtype=float)
        if np.isfinite(r).any():
            rhats.append(float(np.nanmax(r)))
        if np.isfinite(e).any():
            ess.append(float(np.nanmin(e)))
    max_rhat = max(rhats) if rhats else float("nan")
    min_ess = min(ess) if ess else float("nan")
    divergences = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))

    warns: list[str] = []
    if max_rhat > 1.01:
        warns.append(
            f"max R-hat {max_rhat:.3f} > 1.01: chains may not have mixed; "
            f"increase warmup/draws or simplify the model"
        )
    if not min_ess >= 100:  # catches nan too
        warns.append(
            f"min bulk ESS {min_ess:.0f} < 100: posterior summaries are noisy; "
            f"increase draws or chains"
        )
    if divergences > 0:
        warns.append(
            f"{divergences} divergent transition(s): estimates may be biased; "
            f"raise target_accept via more warmup or reparameterize"
        )
    for msg in warns:
        warnings.warn(msg, UserWarning, stacklevel=3)
    return {
        "max_rhat": float(max_rhat),
        "min_ess_bulk": float(min_ess),
        "divergences": divergences,
        "ok": not warns,
        "warnings": warns,
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def fit_arm_effects(rows: Sequence[ItemRow], config: EffectConfig) -> EffectFit:
    """Fit hierarchical arm contrasts over per-item rows.

    Sync and deterministic given ``config.seed``. Raises ValueError on
    malformed inputs before any MCMC is spent; warns (never raises) on
    convergence problems, recorded in ``EffectFit.diagnostics``.
    """
    rows = list(rows)
    seed_on = _validate_rows(rows, config)
    design = _Design(rows, config, seed_on)
    mcmc = _run_mcmc(design, config, seed_on)
    diagnostics = _diagnose(mcmc)
    samples = {k: v for k, v in mcmc.get_samples().items()}
    return _summarize(samples, design, config, rows, diagnostics)
