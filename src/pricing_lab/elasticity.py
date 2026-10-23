"""Price elasticity of demand.

Main model (OLS, log-log):

    log(bookings) = beta * log(price) + gamma * log(1 + lead_time) + fixed effects + error

``beta`` is the elasticity: the % change in bookings for a 1% change in price.

Caveat: hotels set prices in response to expected demand (revenue management).
Busy weeks get higher prices, so OLS mixes the demand curve with the pricing
rule and is usually biased towards zero or even positive. ``fit_2sls`` gives an
illustrative instrumental-variable estimate; its instrument is not clean either.
See the README section "Endogeneity" for the full argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DEFAULT_FIXED_EFFECTS: tuple[str, ...] = ("week_of_year", "year", "hotel", "market_segment")
PRICE_TERM = "log_price"
OUTCOME = "log_bookings"


@dataclass(frozen=True)
class ElasticityResult:
    """Point estimate and interval for the price coefficient."""

    elasticity: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    n_obs: int
    r_squared: float
    formula: str
    cov_type: str

    def summary(self) -> str:
        """One-line plain-English reading of the estimate."""
        return (
            f"A 1% price increase is associated with a {self.elasticity:.2f}% change in "
            f"bookings (95% CI {self.ci_low:.2f} to {self.ci_high:.2f}, n={self.n_obs})."
        )


@dataclass(frozen=True)
class IVResult:
    """Two-stage least squares estimate of the price coefficient."""

    elasticity: float
    std_error: float
    ci_low: float
    ci_high: float
    first_stage_f: float
    n_obs: int
    instrument: str


def build_model_frame(panel: pd.DataFrame, min_bookings: int = 1) -> pd.DataFrame:
    """Add log columns and drop rows that cannot be logged.

    Args:
        panel: Output of ``data.aggregate_demand``.
        min_bookings: Keep rows with at least this many bookings. Keep it at 1:
            higher values select on the outcome and bias the slope.

    Returns:
        A copy with ``log_bookings``, ``log_price`` and ``log_lead_time``.
    """
    if min_bookings < 1:
        raise ValueError("min_bookings must be at least 1 to take logs")
    keep = (panel["bookings"] >= min_bookings) & (panel["adr"] > 0)
    frame = panel.loc[keep].dropna(subset=["adr", "lead_time"]).copy()
    frame[OUTCOME] = np.log(frame["bookings"].astype(float))
    frame[PRICE_TERM] = np.log(frame["adr"].astype(float))
    frame["log_lead_time"] = np.log1p(frame["lead_time"].astype(float))
    return frame.reset_index(drop=True)


def usable_fixed_effects(frame: pd.DataFrame, fixed_effects: Sequence[str]) -> list[str]:
    """Keep only fixed effects with at least two levels in ``frame``."""
    return [col for col in fixed_effects if frame[col].nunique() > 1]


def drop_singletons(frame: pd.DataFrame, fixed_effects: Sequence[str]) -> pd.DataFrame:
    """Drop rows that are the only member of a fixed-effect level, repeatedly.

    A singleton is fitted exactly by its own dummy, so it adds nothing to the
    slope but distorts robust standard errors.
    """
    out = frame
    while True:
        keep = pd.Series(True, index=out.index)
        for col in fixed_effects:
            keep &= out[col].map(out[col].value_counts()) > 1
        if keep.all():
            return out
        out = out.loc[keep]


def build_formula(
    frame: pd.DataFrame,
    fixed_effects: Sequence[str] = DEFAULT_FIXED_EFFECTS,
    include_lead_time: bool = True,
) -> str:
    """Patsy formula for the log-log model."""
    terms = [PRICE_TERM]
    if include_lead_time:
        terms.append("log_lead_time")
    terms += [f"C({col})" for col in usable_fixed_effects(frame, fixed_effects)]
    return f"{OUTCOME} ~ " + " + ".join(terms)


def fit_log_log(
    frame: pd.DataFrame,
    fixed_effects: Sequence[str] = DEFAULT_FIXED_EFFECTS,
    include_lead_time: bool = True,
    cov_type: str = "HC1",
    cluster_col: str | None = None,
    alpha: float = 0.05,
) -> ElasticityResult:
    """Fit the log-log OLS model and return the price elasticity.

    Args:
        frame: Output of ``build_model_frame``.
        fixed_effects: Categorical controls. Week-of-year (or month) absorbs
            seasonality, year absorbs trend, hotel and segment absorb levels.
        include_lead_time: Add ``log(1 + mean lead time)`` as a control.
        cov_type: ``"HC1"`` (heteroskedasticity-robust, the usual "robust"
            errors) or ``"cluster"``. HC1 avoids HC3's division by
            ``1 - leverage``, which breaks on rows the fixed effects fit exactly.
        cluster_col: Column to cluster on when ``cov_type="cluster"``, for
            example ``"period"`` to allow common weekly shocks.
        alpha: Significance level for the confidence interval.
    """
    frame = drop_singletons(frame, usable_fixed_effects(frame, fixed_effects))
    formula = build_formula(frame, fixed_effects, include_lead_time)
    model = smf.ols(formula, data=frame)
    if cov_type == "cluster":
        if cluster_col is None:
            raise ValueError("cluster_col is required when cov_type='cluster'")
        groups = pd.factorize(frame[cluster_col])[0]
        res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        res = model.fit(cov_type=cov_type)
    ci_low, ci_high = res.conf_int(alpha=alpha).loc[PRICE_TERM]
    return ElasticityResult(
        elasticity=float(res.params[PRICE_TERM]),
        std_error=float(res.bse[PRICE_TERM]),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(res.pvalues[PRICE_TERM]),
        n_obs=int(res.nobs),
        r_squared=float(res.rsquared),
        formula=formula,
        cov_type=cov_type if cov_type != "cluster" else f"cluster({cluster_col})",
    )


def per_segment_elasticity(
    frame: pd.DataFrame,
    segment_col: str = "market_segment",
    min_obs: int = 40,
    fixed_effects: Sequence[str] = DEFAULT_FIXED_EFFECTS,
    include_lead_time: bool = True,
    cov_type: str = "HC1",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit the log-log model separately for each segment.

    Segments with fewer than ``min_obs`` rows are skipped. The segment column
    is removed from the fixed effects inside each fit.
    """
    segment_fes = [col for col in fixed_effects if col != segment_col]
    rows = []
    for segment, sub in frame.groupby(segment_col):
        if len(sub) < min_obs:
            continue
        res = fit_log_log(sub, segment_fes, include_lead_time, cov_type, alpha=alpha)
        rows.append(
            {
                segment_col: segment,
                "elasticity": res.elasticity,
                "std_error": res.std_error,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "n_obs": res.n_obs,
            }
        )
    return pd.DataFrame(rows)


def add_hausman_instrument(
    frame: pd.DataFrame,
    price_col: str = PRICE_TERM,
    match_cols: Sequence[str] = ("market_segment", "period"),
    name: str = "z_other_hotel_price",
) -> pd.DataFrame:
    """Add the mean log price of the same segment and week at the other hotels.

    This is a Hausman-style instrument. The two hotels are in different regions
    (Lisbon and the Algarve), so their local demand shocks differ, while
    chain-wide pricing and cost shocks are shared. The exclusion restriction
    fails if demand shocks are national (holidays, events, exchange rates), so
    treat any result as illustrative. Rows with no other hotel get NaN.
    """
    group = frame.groupby(list(match_cols))[price_col]
    total = group.transform("sum")
    count = group.transform("count")
    out = frame.copy()
    out[name] = ((total - frame[price_col]) / (count - 1)).where(count > 1)
    return out


def _dummy_design(
    frame: pd.DataFrame, numeric: Sequence[str], categoricals: Sequence[str]
) -> np.ndarray:
    """Constant + numeric columns + drop-first dummies, as a float array."""
    parts = [pd.Series(1.0, index=frame.index, name="const")]
    parts += [frame[col].astype(float) for col in numeric]
    for col in usable_fixed_effects(frame, categoricals):
        parts.append(
            pd.get_dummies(frame[col].astype(str), prefix=col, drop_first=True, dtype=float)
        )
    return pd.concat(parts, axis=1).to_numpy(dtype=float)


def _robust_cov(xhat: np.ndarray, x: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """HC1 sandwich covariance. ``xhat == x`` gives OLS; projected ``xhat`` gives 2SLS."""
    bread = np.linalg.pinv(xhat.T @ x)
    meat = (xhat * resid[:, None] ** 2).T @ xhat
    n, k = x.shape
    return bread @ meat @ bread.T * n / (n - k)


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and HC1 covariance."""
    beta = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    return beta, _robust_cov(x, x, y - x @ beta)


def _tsls(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2SLS coefficients and HC1 covariance using structural residuals."""
    xhat = z @ (np.linalg.pinv(z.T @ z) @ (z.T @ x))
    beta = np.linalg.pinv(xhat.T @ x) @ (xhat.T @ y)
    return beta, _robust_cov(xhat, x, y - x @ beta)


def fit_2sls(
    frame: pd.DataFrame,
    instrument: str,
    fixed_effects: Sequence[str] = DEFAULT_FIXED_EFFECTS,
    include_lead_time: bool = True,
    alpha: float = 0.05,
) -> IVResult:
    """Instrument ``log_price`` with one excluded instrument (just identified).

    Implemented in numpy so the maths is visible: stage one projects price on
    the instrument and controls, stage two regresses bookings on the projected
    price. Standard errors use the structural residuals (HC1).

    Returns:
        ``IVResult`` with the first-stage F statistic (robust t squared on the
        instrument). F below about 10 means a weak instrument.
    """
    data = frame.dropna(subset=[instrument, PRICE_TERM, OUTCOME])
    data = drop_singletons(data, usable_fixed_effects(data, fixed_effects))
    if len(data) < 10:
        raise ValueError("Too few rows with a non-missing instrument")
    numeric = ["log_lead_time"] if include_lead_time else []
    controls = _dummy_design(data, numeric, fixed_effects)
    y = data[OUTCOME].to_numpy(dtype=float)
    price = data[PRICE_TERM].to_numpy(dtype=float)
    z_excluded = data[instrument].to_numpy(dtype=float)

    x = np.column_stack([price, controls])
    z = np.column_stack([z_excluded, controls])

    first_beta, first_cov = _ols(price, z)
    first_stage_f = float(first_beta[0] ** 2 / first_cov[0, 0])

    beta, cov = _tsls(y, x, z)
    se = float(np.sqrt(cov[0, 0]))
    crit = NormalDist().inv_cdf(1 - alpha / 2)
    return IVResult(
        elasticity=float(beta[0]),
        std_error=se,
        ci_low=float(beta[0] - crit * se),
        ci_high=float(beta[0] + crit * se),
        first_stage_f=first_stage_f,
        n_obs=len(data),
        instrument=instrument,
    )
