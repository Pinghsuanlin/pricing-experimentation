"""Streamlit app: Elasticity, Power calculator, Simulated test & DiD readout.

Run with ``make app`` or ``streamlit run app.py``. Without the real CSV the app
falls back to SYNTHETIC demo data and says so on every tab.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pricing_lab import cuped, did, elasticity, power, simulate
from pricing_lab.data import add_derived_columns, aggregate_demand, clean_bookings, load_bookings
from pricing_lab.synthetic import make_synthetic_bookings

DEFAULT_DATA = Path("data/raw/hotel_bookings.csv")
HOTEL_SEGMENT = ("hotel", "market_segment")
SEASONALITY = {
    "Week of year": ("week_of_year", "year", "hotel", "market_segment"),
    "Month": ("month", "year", "hotel", "market_segment"),
}


@st.cache_data(show_spinner="Loading and cleaning bookings...")
def load_clean(path: str, demo: bool) -> pd.DataFrame:
    """Cleaned booking-level data (real CSV or synthetic demo)."""
    raw = make_synthetic_bookings() if demo else load_bookings(path)
    clean, _ = clean_bookings(add_derived_columns(raw))
    return clean


@st.cache_data
def weekly(clean: pd.DataFrame, demand: str, group_cols: tuple[str, ...]) -> pd.DataFrame:
    """Cached weekly demand panel."""
    return aggregate_demand(clean, group_cols=group_cols, demand=demand)  # type: ignore[arg-type]


def sidebar() -> tuple[pd.DataFrame | None, bool, int]:
    """Data source and held-out window controls."""
    st.sidebar.header("Data")
    path = st.sidebar.text_input("Path to hotel_bookings.csv", str(DEFAULT_DATA))
    exists = Path(path).exists()
    demo = st.sidebar.checkbox("Use SYNTHETIC demo data", value=not exists)
    if not demo and not exists:
        st.sidebar.error("File not found. See data/README.md for download steps.")
        return None, demo, 26
    if demo:
        st.sidebar.warning("SYNTHETIC demo data. Numbers say nothing about real hotels.")
    holdout = st.sidebar.slider("Held-out weeks (used only by the simulated test)", 8, 52, 26, 2)
    return load_clean(path, demo), demo, holdout


def demo_banner(demo: bool) -> None:
    """Remind the reader when the numbers come from synthetic data."""
    if demo:
        st.info("Showing SYNTHETIC demo data. Download the real dataset for real estimates.")


def elasticity_tab(clean: pd.DataFrame, demo: bool, holdout: int) -> None:
    """Observational log-log elasticity with controls, per segment, and a 2SLS check."""
    demo_banner(demo)
    c1, c2, c3, c4 = st.columns(4)
    demand = c1.radio("Demand", ["net", "gross"], help="net excludes cancelled bookings")
    seasonality = c2.selectbox("Seasonality fixed effects", list(SEASONALITY))
    lead_time = c3.checkbox("Control for lead time", value=True)
    se_type = c4.selectbox("Standard errors", ["HC1 (robust)", "Clustered by week"])

    panel = weekly(clean, demand, HOTEL_SEGMENT)
    cutoff = simulate.holdout_cutoff(panel, holdout)
    frame = elasticity.build_model_frame(panel.loc[panel["period"] < cutoff])
    cov = {"cov_type": "cluster", "cluster_col": "period"} if se_type.startswith("Cl") else {}
    res = elasticity.fit_log_log(frame, SEASONALITY[seasonality], lead_time, **cov)
    st.session_state["elasticity_estimate"] = res.elasticity

    m1, m2, m3 = st.columns(3)
    m1.metric("Elasticity", f"{res.elasticity:.2f}")
    m2.metric("95% CI", f"{res.ci_low:.2f} to {res.ci_high:.2f}")
    m3.metric("Weekly cells (training weeks)", f"{res.n_obs:,}")
    st.caption(res.summary() + f" Training weeks end before {cutoff.date()}.")
    if res.elasticity >= 0:
        st.warning(
            "Non-negative elasticity: prices rise in busy weeks, so this is the pricing "
            "rule showing through, not a demand curve. See the endogeneity note below."
        )

    fig = px.scatter(
        frame, x="log_price", y="log_bookings", color="market_segment",
        facet_col="hotel", opacity=0.5, title="Weekly bookings vs price (log scale)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("By segment")
    seg = elasticity.per_segment_elasticity(frame, fixed_effects=SEASONALITY[seasonality])
    if seg.empty:
        st.write("No segment has enough weekly cells.")
    else:
        bar = go.Figure(
            go.Bar(
                x=seg["market_segment"],
                y=seg["elasticity"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": seg["ci_high"] - seg["elasticity"],
                    "arrayminus": seg["elasticity"] - seg["ci_low"],
                },
            )
        )
        bar.update_layout(yaxis_title="Elasticity (95% CI)")
        st.plotly_chart(bar, use_container_width=True)
        st.dataframe(seg.round(3), use_container_width=True)

    with st.expander("Endogeneity: why this number is not causal, and a 2SLS check"):
        st.markdown(
            "Hotels set prices from expected demand. A busy week gets a higher rate, so "
            "price and bookings move together for reasons other than the guest reacting "
            "to price. OLS then understates how price-sensitive guests are. Fixed effects "
            "remove predictable seasonality, not week-specific demand shocks.\n\n"
            "The check below instruments each hotel's price with the other hotel's price "
            "for the same segment and week (a Hausman-style instrument). It is valid only "
            "if the two regions share cost and pricing-policy shocks but not demand "
            "shocks. National holidays and events break that, so read it as a "
            "sensitivity check. The clean answer is an experiment (next tabs)."
        )
        try:
            iv_frame = elasticity.add_hausman_instrument(frame, name="z")
            iv = elasticity.fit_2sls(iv_frame, "z", SEASONALITY[seasonality], lead_time)
            st.write(
                f"2SLS elasticity {iv.elasticity:.2f} (95% CI {iv.ci_low:.2f} to "
                f"{iv.ci_high:.2f}), first-stage F {iv.first_stage_f:.1f}, n={iv.n_obs}."
            )
            if iv.first_stage_f < 10:
                st.warning("First-stage F below 10: weak instrument, so 2SLS is unreliable.")
        except ValueError as err:
            st.write(f"2SLS not estimated: {err}")


def power_tab() -> None:
    """Sample size and MDE vs duration for a price test."""
    metric = st.radio(
        "Primary metric", ["Conversion (two proportions)", "Revenue per visitor (continuous)"]
    )
    c1, c2, c3 = st.columns(3)
    alpha = c1.number_input("Alpha (two-sided)", 0.001, 0.2, 0.05, 0.005, format="%.3f")
    target_power = c2.number_input("Power", 0.5, 0.99, 0.8, 0.05)
    treat_share = c3.slider("Share of traffic in treatment", 0.1, 0.9, 0.5, 0.05)
    ratio = treat_share / (1 - treat_share)
    daily = st.number_input("Eligible visitors per day (both arms)", 10, 10_000_000, 5000, 100)

    st.markdown("**Randomisation unit**")
    unit = st.selectbox(
        "Unit", ["Visitor or session", "Cluster (hotel, city, market or time block)"]
    )
    deff = 1.0
    if unit.startswith("Cluster"):
        k1, k2 = st.columns(2)
        size = k1.number_input("Visitors per cluster", 1, 1_000_000, 500)
        icc = k2.number_input("Intra-cluster correlation (ICC)", 0.0, 1.0, 0.01, 0.005, "%.3f")
        deff = power.design_effect(size, icc)
        st.caption(f"Design effect {deff:.2f}: effective sample = nominal sample / {deff:.2f}.")

    if metric.startswith("Conversion"):
        p = st.number_input("Baseline conversion", 0.001, 0.99, 0.10, 0.01, format="%.3f")
        rel = st.number_input("Minimum detectable effect (relative %)", 0.5, 100.0, 20.0, 0.5)
        mde = p * rel / 100
        n = power.n_per_arm_two_proportions(p, mde, alpha, target_power, ratio=ratio)
        table = power.mde_duration_table(
            daily, range(1, 9), "proportion", p, alpha=alpha, power=target_power,
            treat_share=treat_share, deff=deff,
        )
    else:
        v1, v2, v3 = st.columns(3)
        conv = v1.number_input("Conversion", 0.001, 1.0, 0.05, 0.005, format="%.3f")
        value = v2.number_input("Mean booking value", 1.0, 100_000.0, 300.0, 10.0)
        value_sd = v3.number_input("Booking value sd", 0.0, 100_000.0, 250.0, 10.0)
        mean_rpv, sd_rpv = power.rpv_moments(conv, value, value_sd)
        st.caption(f"Revenue per visitor: mean {mean_rpv:.2f}, sd {sd_rpv:.2f}.")
        rel = st.number_input("Minimum detectable effect (relative %)", 0.5, 100.0, 5.0, 0.5)
        mde = mean_rpv * rel / 100
        n = power.n_per_arm_continuous(sd_rpv, mde, alpha, target_power, ratio=ratio)
        table = power.mde_duration_table(
            daily, range(1, 9), "continuous", mean_rpv, sd=sd_rpv, alpha=alpha,
            power=target_power, treat_share=treat_share, deff=deff,
        )

    n_eff = int(n * deff)
    days = n_eff / (daily * (1 - treat_share))
    m1, m2, m3 = st.columns(3)
    m1.metric("Control arm size", f"{n_eff:,}")
    m2.metric("Treatment arm size", f"{int(n_eff * ratio):,}")
    m3.metric("Days needed", f"{days:,.1f}")
    st.caption("Normal approximation. Round duration up to whole weeks to cover weekday cycles.")

    st.subheader("MDE vs test duration")
    shown = table.assign(mde_rel_pct=table["mde_rel"] * 100)
    st.plotly_chart(
        px.line(shown, x="weeks", y="mde_rel_pct", markers=True,
                labels={"mde_rel_pct": "MDE (% of baseline)"}),
        use_container_width=True,
    )
    st.dataframe(shown.round(4), use_container_width=True)

    with st.expander("Choosing the randomisation unit"):
        st.markdown(
            "- **Visitor-level** randomisation gives the most power, but showing two prices "
            "for the same room raises fairness and legal questions, and guests compare "
            "prices across devices.\n"
            "- **Marketplace interference**: hotel rooms are a shared, limited inventory. A "
            "cheaper treatment arm can take rooms that control guests would have booked, so "
            "the measured gap overstates the effect of a full rollout.\n"
            "- **Cluster randomisation** (by hotel, city or market) avoids that, at the cost "
            "of the design effect above and few units. Use DiD with a holdout to absorb "
            "level differences between clusters.\n"
            "- **Switchback** designs alternate the price over time blocks for the whole "
            "market. They handle interference, but need blocks long enough for bookings "
            "to land in the right block and a washout for carry-over (lead times are long)."
        )


def simulation_tab(clean: pd.DataFrame, demo: bool, holdout: int) -> None:
    """SIMULATED price test on held-out weeks with DiD, pre-trends, permutation and CUPED."""
    st.error(
        "SIMULATED. No real price test was run. Real held-out bookings are the untreated "
        "outcome; the treatment effect is injected from the elasticity you choose."
    )
    demo_banner(demo)
    c1, c2, c3, c4 = st.columns(4)
    change = c1.slider("Price change (%)", -20.0, 20.0, 5.0, 0.5) / 100
    estimate = float(st.session_state.get("elasticity_estimate", -1.0))
    default_e = min(max(estimate, -5.0), -0.1)
    elasticity_value = c2.number_input("Assumed elasticity", -5.0, 0.0, round(default_e, 2), 0.1)
    min_weekly = c3.number_input("Min weekly bookings per unit", 1, 100, 5)
    seed = c4.number_input("Seed", 0, 10_000, 42)

    pre = holdout // 2
    config = simulate.SimulationConfig(
        price_change=change, elasticity=elasticity_value, pre_weeks=pre,
        post_weeks=holdout - pre, min_weekly_bookings=int(min_weekly), seed=int(seed),
    )
    units = weekly(clean, "net", simulate.DEFAULT_UNIT_COLS)
    cutoff = simulate.holdout_cutoff(units, holdout)
    try:
        sim = simulate.simulate_price_test(units.loc[units["period"] >= cutoff], config)
    except ValueError as err:
        st.error(f"Cannot run the simulation: {err}")
        return

    truth = simulate.true_effects(change, elasticity_value)
    twfe = did.fit_did(sim)
    revenue = did.fit_did(sim, outcome="log_revenue_sim")
    trend = did.pre_trend_test(sim)
    perm = did.permutation_test(sim)
    n_units = sim["unit"].nunique()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("True bookings effect", f"{truth.bookings_pct:+.1%}")
    m2.metric("DiD estimate", f"{twfe.pct_effect:+.1%}",
              help=f"95% CI {twfe.pct_ci_low:+.1%} to {twfe.pct_ci_high:+.1%}")
    m3.metric("True revenue effect", f"{truth.revenue_pct:+.1%}")
    m4.metric("DiD revenue estimate", f"{revenue.pct_effect:+.1%}",
              help=f"95% CI {revenue.pct_ci_low:+.1%} to {revenue.pct_ci_high:+.1%}")
    st.caption(
        f"{n_units} units (hotel x segment x channel), {pre} pre and {holdout - pre} post "
        f"weeks from {cutoff.date()}. Bookings DiD 95% CI {twfe.pct_ci_low:+.1%} to "
        f"{twfe.pct_ci_high:+.1%}; cluster-robust p={twfe.p_value:.3f}; permutation "
        f"p={perm.p_value:.3f}."
    )
    if n_units < 20:
        st.warning(
            f"Only {n_units} units. Cluster-robust errors are unreliable with few clusters; "
            "trust the permutation p-value more."
        )

    trend_by_arm = (
        sim.groupby(["rel_week", "treated"])["log_bookings_sim"].mean().reset_index()
        .assign(arm=lambda d: d["treated"].map({1: "Treatment", 0: "Holdout"}))
    )
    line = px.line(trend_by_arm, x="rel_week", y="log_bookings_sim", color="arm", markers=True,
                   title="Mean log bookings by arm (SIMULATED after week 0)")
    line.add_vline(x=-0.5, line_dash="dash")
    st.plotly_chart(line, use_container_width=True)

    events = did.event_study(sim)
    es = go.Figure(
        go.Scatter(
            x=events["rel_week"], y=events["coef"], mode="markers+lines",
            error_y={"type": "data", "symmetric": False,
                     "array": events["ci_high"] - events["coef"],
                     "arrayminus": events["coef"] - events["ci_low"]},
        )
    )
    es.add_hline(y=0, line_dash="dot")
    es.add_hline(y=truth.bookings_log, line_dash="dash", annotation_text="true effect")
    es.update_layout(title="Event study (treated minus holdout, reference week -1)",
                     xaxis_title="Weeks from launch", yaxis_title="Log points")
    st.plotly_chart(es, use_container_width=True)
    st.write(
        f"Pre-trend test: differential slope {trend.slope:.4f} per week "
        f"(p={trend.p_value:.3f}). A small p-value means the groups were already "
        "drifting apart before launch."
    )

    st.subheader("CUPED")
    means = did.unit_pre_post_means(sim)
    res = cuped.cuped_ate(means["post"], means["pre"], means["treated"])
    st.dataframe(
        pd.DataFrame(
            {
                "estimate": [res.raw_effect, res.adjusted_effect, twfe.effect],
                "std_error": [res.raw_std_error, res.adjusted_std_error, twfe.std_error],
            },
            index=["Post-period difference", "CUPED (pre-period covariate)", "DiD (TWFE)"],
        ).round(4),
        use_container_width=True,
    )
    st.caption(
        f"theta={res.theta:.2f}, variance reduction {res.variance_reduction:.0%}. "
        "DiD is CUPED with theta fixed at 1; CUPED estimates theta. Normal intervals; "
        "with few units prefer the permutation p-value."
    )


def main() -> None:
    """Page layout."""
    st.set_page_config(page_title="Pricing & Experimentation Lab", layout="wide")
    st.title("Pricing & Experimentation Lab")
    st.caption(
        "How sensitive is hotel demand to price, and how would you prove a price change works?"
    )
    clean, demo, holdout = sidebar()
    tab1, tab2, tab3 = st.tabs(["Elasticity", "Power calculator", "Simulated test & DiD"])
    with tab2:
        power_tab()
    if clean is None:
        for tab in (tab1, tab3):
            tab.info("Load the data (see the sidebar) to use this tab.")
        return
    with tab1:
        elasticity_tab(clean, demo, holdout)
    with tab3:
        simulation_tab(clean, demo, holdout)


main()
