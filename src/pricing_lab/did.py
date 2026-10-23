"""Difference-in-differences readout for a price test with a holdout group.

Main estimate (two-way fixed effects):

    y_it = alpha_i + lambda_t + delta * (treated_i x post_t) + e_it

``delta`` is the effect in log points when ``y`` is a log outcome. Standard
errors are clustered by unit, since weeks within a unit are correlated.

With few units, cluster-robust errors are unreliable, so ``permutation_test``
gives a randomisation-inference p-value that only relies on the random
assignment. ``pre_trend_test`` and ``event_study`` check parallel trends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper


@dataclass(frozen=True)
class DiDResult:
    """Treatment effect in outcome units (log points for log outcomes)."""

    effect: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    pct_effect: float
    pct_ci_low: float
    pct_ci_high: float
    n_obs: int
    n_units: int
    specification: str


@dataclass(frozen=True)
class PreTrendResult:
    """Differential linear trend of treated vs control units before launch."""

    slope: float
    std_error: float
    p_value: float
    n_pre_periods: int


@dataclass(frozen=True)
class PermutationResult:
    """Randomisation-inference result for the 2x2 DiD of unit means."""

    observed: float
    p_value: float
    n_permutations: int


def _cluster_fit(formula: str, data: pd.DataFrame, unit_col: str) -> RegressionResultsWrapper:
    """OLS with unit-clustered errors and t-based inference."""
    groups = pd.factorize(data[unit_col])[0]
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": groups}, use_t=True
    )


def fit_did(
    panel: pd.DataFrame,
    outcome: str = "log_bookings_sim",
    unit_col: str = "unit",
    time_col: str = "rel_week",
    treat_col: str = "treated",
    post_col: str = "post",
    twfe: bool = True,
    alpha: float = 0.05,
) -> DiDResult:
    """Estimate the treated x post effect.

    Args:
        twfe: ``True`` uses unit and week fixed effects. ``False`` fits the
            textbook ``y ~ treated + post + treated:post``. On a balanced panel
            both give the same point estimate; standard errors differ.
    """
    data = panel.dropna(subset=[outcome]).copy()
    data["treat_post"] = data[treat_col] * data[post_col]
    if twfe:
        formula = f"{outcome} ~ treat_post + C({unit_col}) + C({time_col})"
    else:
        formula = f"{outcome} ~ {treat_col} + {post_col} + treat_post"
    res = _cluster_fit(formula, data, unit_col)
    effect = float(res.params["treat_post"])
    ci_low, ci_high = (float(v) for v in res.conf_int(alpha=alpha).loc["treat_post"])
    return DiDResult(
        effect=effect,
        std_error=float(res.bse["treat_post"]),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=float(res.pvalues["treat_post"]),
        pct_effect=math.expm1(effect),
        pct_ci_low=math.expm1(ci_low),
        pct_ci_high=math.expm1(ci_high),
        n_obs=int(res.nobs),
        n_units=int(data[unit_col].nunique()),
        specification="twfe" if twfe else "2x2",
    )


def pre_trend_test(
    panel: pd.DataFrame,
    outcome: str = "log_bookings_sim",
    unit_col: str = "unit",
    time_col: str = "rel_week",
    treat_col: str = "treated",
    post_col: str = "post",
) -> PreTrendResult:
    """Test for a differential linear trend in the pre-period only.

    Fits ``y ~ treated x week + unit FE + week FE`` on pre-launch weeks. A
    slope far from zero means the groups were drifting apart before launch,
    so the parallel-trends assumption is in doubt. One parameter keeps the
    test valid with few clusters, where a joint test on every pre-week
    coefficient would not be.
    """
    pre = panel.loc[panel[post_col] == 0].dropna(subset=[outcome]).copy()
    pre["treat_trend"] = pre[treat_col] * pre[time_col]
    res = _cluster_fit(f"{outcome} ~ treat_trend + C({unit_col}) + C({time_col})", pre, unit_col)
    return PreTrendResult(
        slope=float(res.params["treat_trend"]),
        std_error=float(res.bse["treat_trend"]),
        p_value=float(res.pvalues["treat_trend"]),
        n_pre_periods=int(pre[time_col].nunique()),
    )


def _event_name(k: int) -> str:
    return f"ev_m{-k}" if k < 0 else f"ev_p{k}"


def event_study(
    panel: pd.DataFrame,
    outcome: str = "log_bookings_sim",
    unit_col: str = "unit",
    time_col: str = "rel_week",
    treat_col: str = "treated",
    ref_period: int = -1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Treated-vs-control gap for each week relative to ``ref_period``.

    Pre-launch coefficients near zero support parallel trends; post-launch
    coefficients trace the effect over time. Returns one row per week with
    ``coef``, ``ci_low`` and ``ci_high`` (the reference week is 0 by construction).
    """
    data = panel.dropna(subset=[outcome]).copy()
    periods = sorted(int(k) for k in data[time_col].unique())
    if ref_period not in periods:
        raise ValueError(f"ref_period {ref_period} not in the panel")
    names: dict[int, str] = {}
    for k in periods:
        if k == ref_period:
            continue
        names[k] = _event_name(k)
        data[names[k]] = ((data[time_col] == k) & (data[treat_col] == 1)).astype(float)
    formula = f"{outcome} ~ " + " + ".join(names.values()) + f" + C({unit_col}) + C({time_col})"
    res = _cluster_fit(formula, data, unit_col)
    ci = res.conf_int(alpha=alpha)
    rows = [{"rel_week": ref_period, "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0}]
    for k, name in names.items():
        rows.append(
            {
                "rel_week": k,
                "coef": float(res.params[name]),
                "ci_low": float(ci.loc[name, 0]),
                "ci_high": float(ci.loc[name, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("rel_week").reset_index(drop=True)


def unit_pre_post_means(
    panel: pd.DataFrame,
    outcome: str = "log_bookings_sim",
    unit_col: str = "unit",
    treat_col: str = "treated",
    post_col: str = "post",
) -> pd.DataFrame:
    """One row per unit: ``pre`` and ``post`` mean outcome and ``treated``.

    Units missing either period are dropped. This is also the input to CUPED.
    """
    data = panel.dropna(subset=[outcome])
    means = data.groupby([unit_col, post_col])[outcome].mean().unstack(post_col)
    means = means.rename(columns={0: "pre", 1: "post"})
    means["treated"] = data.groupby(unit_col)[treat_col].first()
    return means.dropna(subset=["pre", "post"])


def permutation_test(
    panel: pd.DataFrame,
    outcome: str = "log_bookings_sim",
    unit_col: str = "unit",
    treat_col: str = "treated",
    post_col: str = "post",
    pair_col: str | None = "pair_id",
    n_permutations: int = 2000,
    seed: int = 0,
) -> PermutationResult:
    """Randomisation inference for the DiD of unit means.

    Re-draws the treatment labels as the design would have (within matched
    pairs if ``pair_col`` is given, otherwise a full shuffle) and recomputes
    mean(post - pre | treated) - mean(post - pre | control). The two-sided
    p-value is the share of draws at least as extreme as the observed value.
    """
    means = unit_pre_post_means(panel, outcome, unit_col, treat_col, post_col)
    diffs = (means["post"] - means["pre"]).to_numpy(dtype=float)
    labels = means["treated"].to_numpy(dtype=int)
    pairs = None
    if pair_col is not None and pair_col in panel.columns:
        pairs = panel.groupby(unit_col)[pair_col].first().reindex(means.index).to_numpy()

    def stat(lab: np.ndarray) -> float:
        return float(diffs[lab == 1].mean() - diffs[lab == 0].mean())

    observed = stat(labels)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        draw = labels.copy()
        if pairs is None:
            draw = rng.permutation(draw)
        else:
            for pair in np.unique(pairs):
                idx = np.flatnonzero(pairs == pair)
                draw[idx] = rng.permutation(labels[idx])
        if draw.min() == draw.max():
            continue
        if abs(stat(draw)) >= abs(observed) - 1e-12:
            extreme += 1
    return PermutationResult(
        observed=observed,
        p_value=(extreme + 1) / (n_permutations + 1),
        n_permutations=n_permutations,
    )
