#!/usr/bin/env python3
"""
Blue Zones Longevity Analysis Dashboard

Five-view interactive dashboard:
  1. Pre-COVID (1960-2019) -- clean secular trends
  2. Full Period (1960-2023) -- includes COVID disruption and recovery
  3. COVID Impact Comparison -- side-by-side analysis
  4. Statistical Deep Dive -- significance tests, controls, sensitivity
  5. Predictive Analysis -- ML models, overperformance detection, hidden Blue Zones

Built on real World Bank, WHO, and projected data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Blue Zones Longevity Analysis",
    page_icon="BZ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {font-size:2.2rem;font-weight:700;text-align:center;color:#1a1a2e;margin-bottom:0.3rem}
    .sub-header {font-size:1rem;text-align:center;color:#555;margin-bottom:1.5rem}
    .section-hdr {font-size:1.3rem;font-weight:600;color:#16213e;border-bottom:2px solid #0f3460;padding-bottom:0.3rem;margin-top:1.5rem}
    .metric-card {background:#f8f9fa;border-radius:8px;padding:0.8rem;border-left:4px solid #0f3460;margin:0.3rem 0}
    .covid-bad {color:#c0392b;font-weight:600}
    .covid-good {color:#27ae60;font-weight:600}
</style>
""", unsafe_allow_html=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BZ_COLORS = {
    "USA": "#E74C3C", "JPN": "#3498DB", "ITA": "#2ECC71",
    "GRC": "#9B59B6", "CRI": "#F39C12",
}
BLUE_ZONE_ISOS = set(BZ_COLORS.keys())
BZ_NAMES = {
    "USA": "United States (Loma Linda)",
    "JPN": "Japan (Okinawa)",
    "ITA": "Italy (Sardinia)",
    "GRC": "Greece (Ikaria)",
    "CRI": "Costa Rica (Nicoya)",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_historical():
    path = os.path.join(SCRIPT_DIR, "data", "historical", "merged_historical_panel.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load_projections():
    path = os.path.join(SCRIPT_DIR, "data", "projections", "un_life_expectancy_projections.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load_analysis_file(subdir, name):
    path = os.path.join(SCRIPT_DIR, "outputs", "analysis", subdir, f"{name}.csv")
    if not os.path.exists(path):
        path = os.path.join(SCRIPT_DIR, "outputs", "analysis", f"{name}.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load_covid_impact():
    path = os.path.join(SCRIPT_DIR, "outputs", "analysis", "covid_comparison", "country_covid_impact.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load_comparison_summary():
    path = os.path.join(SCRIPT_DIR, "outputs", "analysis", "covid_comparison", "comparison_summary.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def fig_timelines(hist, proj, selected_isos, title_suffix="", show_proj=True,
                  show_gender=False):
    fig = go.Figure()
    global_avg = hist.groupby("year")["life_expectancy"].mean().dropna()
    fig.add_trace(go.Scatter(
        x=global_avg.index, y=global_avg.values,
        mode="lines", name="Global Average",
        line=dict(color="#95a5a6", width=3, dash="dot"),
    ))
    for iso in selected_isos:
        color = BZ_COLORS.get(iso, "#333")
        label = BZ_NAMES.get(iso, iso)
        ch = hist[(hist["iso_code"] == iso)].sort_values("year")
        le = ch[["year", "life_expectancy"]].dropna()
        if not le.empty:
            fig.add_trace(go.Scatter(
                x=le["year"], y=le["life_expectancy"],
                mode="lines", name=label, line=dict(color=color, width=2.5),
            ))
        if show_gender:
            le_m = ch[["year", "life_expectancy_male"]].dropna()
            le_f = ch[["year", "life_expectancy_female"]].dropna()
            if not le_m.empty:
                fig.add_trace(go.Scatter(
                    x=le_m["year"], y=le_m["life_expectancy_male"],
                    mode="lines", name=f"{iso} Male",
                    line=dict(color=color, width=1.2, dash="dash"),
                    showlegend=False,
                ))
            if not le_f.empty:
                fig.add_trace(go.Scatter(
                    x=le_f["year"], y=le_f["life_expectancy_female"],
                    mode="lines", name=f"{iso} Female",
                    line=dict(color=color, width=1.2, dash="dot"),
                    showlegend=False,
                ))
        if show_proj and not proj.empty:
            cp = proj[proj["iso_code"] == iso].sort_values("year")
            if not cp.empty and "le_medium" in cp.columns:
                fig.add_trace(go.Scatter(
                    x=cp["year"], y=cp["le_medium"], mode="lines",
                    name=f"{label} (proj)", line=dict(color=color, width=2, dash="dash"),
                    showlegend=False,
                ))
                if "le_high" in cp.columns and "le_low" in cp.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([cp["year"], cp["year"][::-1]]),
                        y=pd.concat([cp["le_high"], cp["le_low"][::-1]]),
                        fill="toself", fillcolor=color, opacity=0.08,
                        line=dict(width=0), showlegend=False, name="",
                    ))
    if show_proj:
        fig.add_vline(x=2023, line_dash="dot", line_color="grey", opacity=0.5)
    gender_note = " (dashed=male, dotted=female)" if show_gender else ""
    fig.update_layout(
        title=f"Life Expectancy Over Time{title_suffix}{gender_note}",
        xaxis_title="Year", yaxis_title="Life Expectancy (years)",
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28),
    )
    return fig


def fig_convergence(bz_global, sigma, title_prefix="", gap_ci=None,
                    weighted_gap=None, show_weighted=False):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[f"{title_prefix}Blue Zone Gap Over Global Average",
                                        f"{title_prefix}Sigma Convergence (Global LE Spread)"],
                        vertical_spacing=0.12)
    if not bz_global.empty:
        fig.add_trace(go.Scatter(
            x=bz_global["year"], y=bz_global["bz_gap_over_global"],
            fill="tozeroy", mode="lines", name="BZ advantage (unweighted)",
            line=dict(color="#2ecc71", width=2), fillcolor="rgba(46,204,113,0.2)",
        ), row=1, col=1)

        # Add CI ribbon if available
        if gap_ci is not None and not gap_ci.empty:
            fig.add_trace(go.Scatter(
                x=pd.concat([gap_ci["year"], gap_ci["year"][::-1]]),
                y=pd.concat([gap_ci["ci_upper"], gap_ci["ci_lower"][::-1]]),
                fill="toself", fillcolor="rgba(46,204,113,0.1)",
                line=dict(width=0), showlegend=True, name="95% CI",
            ), row=1, col=1)

        # Add weighted gap if toggled
        if show_weighted and weighted_gap is not None and not weighted_gap.empty:
            fig.add_trace(go.Scatter(
                x=weighted_gap["year"], y=weighted_gap["gap_weighted"],
                mode="lines", name="BZ gap (pop-weighted)",
                line=dict(color="#e74c3c", width=2, dash="dash"),
            ), row=1, col=1)

    if not sigma.empty:
        fig.add_trace(go.Scatter(
            x=sigma["year"], y=sigma["le_std"], mode="lines",
            name="Std Dev", line=dict(color="#8e44ad", width=2),
            fill="tozeroy", fillcolor="rgba(142,68,173,0.12)",
        ), row=2, col=1)
    fig.update_layout(height=550, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.15))
    fig.update_yaxes(title_text="Gap (years)", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev (years)", row=2, col=1)
    return fig


def fig_decade_bars(decades, title_suffix=""):
    if decades.empty:
        return go.Figure()
    decade_list = decades[decades["group"] == "global"]["decade"].tolist()
    fig = go.Figure()
    for group, color, label in [
        ("blue_zone", "#3498DB", "Blue Zone Countries"),
        ("non_blue_zone", "#bdc3c7", "Non-Blue Zone Countries"),
    ]:
        grp = decades[decades["group"] == group]
        vals = [grp[grp["decade"] == d]["avg_gain"].values[0]
                if len(grp[grp["decade"] == d]) else 0 for d in decade_list]
        fig.add_trace(go.Bar(x=decade_list, y=vals, name=label,
                             marker_color=color, text=[f"{v:.1f}" for v in vals],
                             textposition="outside"))
    fig.update_layout(
        title=f"LE Gain by Decade{title_suffix}",
        xaxis_title="Decade", yaxis_title="Avg Gain (years)",
        barmode="group", height=400, template="plotly_white",
    )
    return fig


def fig_ranking(hist):
    le_data = hist[["iso_code", "year", "life_expectancy"]].dropna()
    rows = []
    for year in sorted(le_data["year"].unique()):
        yd = le_data[le_data["year"] == year].copy()
        yd["rank"] = yd["life_expectancy"].rank(ascending=False)
        total = len(yd)
        for iso in BZ_COLORS:
            r = yd[yd["iso_code"] == iso]
            if not r.empty:
                rows.append({"year": year, "iso_code": iso,
                             "percentile": (1 - r["rank"].iloc[0] / total) * 100})
    if not rows:
        return go.Figure()
    rdf = pd.DataFrame(rows)
    fig = go.Figure()
    for iso in sorted(BZ_COLORS):
        cr = rdf[rdf["iso_code"] == iso].sort_values("year")
        fig.add_trace(go.Scatter(
            x=cr["year"], y=cr["percentile"], mode="lines+markers",
            name=BZ_NAMES[iso], line=dict(color=BZ_COLORS[iso], width=2),
            marker=dict(size=3),
        ))
    fig.add_hline(y=50, line_dash="dash", line_color="grey", opacity=0.3)
    fig.update_layout(
        title="Blue Zone Countries: Global LE Percentile",
        xaxis_title="Year", yaxis_title="Percentile (higher = better)",
        yaxis_range=[0, 105], height=420, template="plotly_white",
    )
    return fig


def fig_country_detail(hist, proj, iso, show_proj=True):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        subplot_titles=[f"{BZ_NAMES.get(iso, iso)}: Life Expectancy",
                                        f"{BZ_NAMES.get(iso, iso)}: GDP per Capita"],
                        vertical_spacing=0.15)
    ch = hist[hist["iso_code"] == iso].sort_values("year")
    le = ch[["year", "life_expectancy"]].dropna()
    if not le.empty:
        fig.add_trace(go.Scatter(
            x=le["year"], y=le["life_expectancy"], mode="lines",
            name="Historical", line=dict(color=BZ_COLORS.get(iso, "#333"), width=2.5),
        ), row=1, col=1)
    if show_proj and not proj.empty:
        cp = proj[proj["iso_code"] == iso].sort_values("year")
        if not cp.empty and "le_medium" in cp.columns:
            fig.add_trace(go.Scatter(
                x=cp["year"], y=cp["le_medium"], mode="lines",
                name="Projected", line=dict(color=BZ_COLORS.get(iso, "#333"), width=2, dash="dash"),
            ), row=1, col=1)
    gdp = ch[["year", "gdp_per_capita"]].dropna()
    if not gdp.empty:
        fig.add_trace(go.Scatter(
            x=gdp["year"], y=gdp["gdp_per_capita"], mode="lines",
            name="GDP/capita", line=dict(color="#f39c12", width=2),
            fill="tozeroy", fillcolor="rgba(243,156,18,0.1)",
        ), row=2, col=1)
    fig.update_layout(height=550, template="plotly_white")
    fig.update_yaxes(title_text="LE (years)", row=1, col=1)
    fig.update_yaxes(title_text="GDP (USD)", row=2, col=1)
    return fig


def fig_correlation(hist):
    recent = hist.sort_values("year").groupby("iso_code").last().reset_index()
    cols = ["life_expectancy", "gdp_per_capita", "physicians_per_1000",
            "urban_population_pct", "pm25_air_pollution", "death_rate",
            "health_expenditure_pc"]
    avail = [c for c in cols if c in recent.columns]
    corr = recent[avail].dropna().corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Indicator Correlations (Most Recent Year)", text_auto=".2f")
    fig.update_layout(height=480, template="plotly_white")
    return fig


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------
def render_period_tab(hist, proj, bz_global, sigma, beta, decades, period_label,
                      show_proj=True):
    """Render a full analysis view for a given time period."""
    bz_isos = sorted(BZ_COLORS.keys())
    yr_min, yr_max = int(hist["year"].min()), int(hist["year"].max())

    # Load additional data for enhancements
    gap_ci = load_analysis_file("", "gap_confidence_intervals")
    weighted_gap = load_analysis_file("", "population_weighted_gap")
    if not gap_ci.empty and yr_max <= 2019:
        gap_ci = gap_ci[gap_ci["year"] <= 2019]
    if not weighted_gap.empty and yr_max <= 2019:
        weighted_gap = weighted_gap[weighted_gap["year"] <= 2019]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries", hist["iso_code"].nunique())
    c2.metric("Period", f"{yr_min}-{yr_max}")
    c3.metric("Observations", f"{len(hist):,}")
    if not bz_global.empty:
        final_gap = bz_global.iloc[-1]["bz_gap_over_global"]
        c4.metric("BZ Gap (final year)", f"+{final_gap:.1f} yr")

    # Options row
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        show_gender = st.checkbox("Show male/female LE overlay", value=False,
                                  key=f"gender_{period_label}")
    with opt_col2:
        show_weighted = st.checkbox("Show population-weighted gap", value=False,
                                    key=f"weighted_{period_label}")

    # Historical trends
    st.markdown(f'<h3 class="section-hdr">Life Expectancy: Blue Zone Countries vs World ({period_label})</h3>',
                unsafe_allow_html=True)
    st.plotly_chart(fig_timelines(hist, proj, bz_isos, f" ({period_label})",
                                  show_proj=show_proj, show_gender=show_gender),
                    use_container_width=True)

    # Convergence
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_convergence(bz_global, sigma, f"{period_label}: ",
                                         gap_ci=gap_ci, weighted_gap=weighted_gap,
                                         show_weighted=show_weighted),
                        use_container_width=True)
    with col2:
        st.plotly_chart(fig_decade_bars(decades, f" ({period_label})"),
                        use_container_width=True)
        if not beta.empty:
            st.markdown("**Beta Convergence by Decade**")
            display_beta = beta[["decade", "avg_gain_years", "bz_avg_gain",
                                  "non_bz_avg_gain", "beta_correlation", "convergence"]].copy()
            display_beta.columns = ["Decade", "Avg Gain", "BZ Gain", "Non-BZ Gain", "Beta r", "Convergence"]
            st.dataframe(display_beta.round(2), use_container_width=True, hide_index=True)

    # Rankings
    st.markdown(f'<h3 class="section-hdr">Blue Zone Rankings ({period_label})</h3>',
                unsafe_allow_html=True)
    st.plotly_chart(fig_ranking(hist), use_container_width=True)

    # Country deep dive
    st.markdown(f'<h3 class="section-hdr">Country Deep Dive ({period_label})</h3>',
                unsafe_allow_html=True)
    iso_pick = st.selectbox("Select country", bz_isos,
                            format_func=lambda x: BZ_NAMES.get(x, x),
                            key=f"country_{period_label}")
    st.plotly_chart(fig_country_detail(hist, proj, iso_pick, show_proj=show_proj),
                    use_container_width=True)


def render_comparison_tab(hist_full, impact_df, summary_df,
                          gap_pre, gap_full, sigma_pre, sigma_full):
    """Render the COVID impact comparison view."""

    if summary_df.empty:
        st.warning("Run covid_comparison_analysis.py first to generate comparison data.")
        return

    s = summary_df.iloc[0]

    # Top-level impact metrics
    st.markdown('<h3 class="section-hdr">COVID-19 Impact on Life Expectancy Data</h3>',
                unsafe_allow_html=True)
    st.markdown(
        "COVID-19 caused the largest single-year drop in global life expectancy since records began. "
        "This matters for trend analysis because it introduces noise that can distort long-term patterns. "
        "Below is the damage, the recovery, and what it means for the Blue Zone analysis."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global Drop 2019-2020", f"{s['global_drop_2020']:.1f} yr",
              delta=f"{s['global_drop_2020']:.1f}", delta_color="inverse")
    c2.metric("Global Drop 2019-2021", f"{s['global_drop_2021']:.1f} yr",
              delta=f"{s['global_drop_2021']:.1f}", delta_color="inverse")
    c3.metric("Countries Recovered by 2023", f"{int(s['n_countries_recovered'])} / 93")
    c4.metric("Worst Hit Country", f"{s['worst_hit_country']} ({s['worst_hit_drop']:.1f} yr)")

    st.markdown("---")

    # Structural break test results
    ts_tests = load_analysis_file("", "time_series_tests")
    covid_accel = load_analysis_file("", "covid_acceleration_test")

    if not ts_tests.empty:
        st.markdown('<h3 class="section-hdr">Structural Break Analysis</h3>',
                    unsafe_allow_html=True)
        chow = ts_tests[ts_tests["test"] == "Chow_structural_break_2020"]
        if not chow.empty:
            chow_row = chow.iloc[0]
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Chow Test F-statistic", f"{chow_row['statistic']:.2f}")
            col_s2.metric("P-value", f"{chow_row['p_value']:.6f}")
            col_s3.metric("Result", chow_row["conclusion"])

        if not covid_accel.empty:
            st.markdown("**COVID Acceleration Test:** Did COVID push the gap below the pre-COVID trend?")
            accel_display = covid_accel[["year", "predicted_gap", "actual_gap",
                                         "difference", "outside_pi"]].copy()
            accel_display.columns = ["Year", "Predicted Gap", "Actual Gap",
                                     "Difference", "Outside 95% PI"]
            st.dataframe(accel_display.round(3), use_container_width=True, hide_index=True)

        st.markdown("---")

    # BZ vs Non-BZ COVID impact
    st.markdown('<h3 class="section-hdr">Blue Zone Countries: COVID Impact</h3>',
                unsafe_allow_html=True)

    if not impact_df.empty:
        bz_impact = impact_df[impact_df["is_blue_zone"] == 1].sort_values("drop_2020")
        display_bz = bz_impact[["iso_code", "le_2019", "le_2020", "le_2021",
                                 "le_2023", "drop_2020", "recovery_2023"]].copy()
        display_bz.columns = ["Country", "LE 2019", "LE 2020", "LE 2021",
                               "LE 2023", "Drop 2019-2020", "Net Change 2019-2023"]
        st.dataframe(display_bz.round(2), use_container_width=True, hide_index=True)

        st.markdown(
            f"**Average 2020 drop:** BZ countries: {s['avg_drop_bz']:.2f} yr | "
            f"Non-BZ countries: {s['avg_drop_non_bz']:.2f} yr"
        )

    # Side-by-side gap comparison
    st.markdown('<h3 class="section-hdr">Gap Analysis: Pre-COVID vs Full Period</h3>',
                unsafe_allow_html=True)

    if not gap_pre.empty and not gap_full.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gap_pre["year"], y=gap_pre["bz_gap_over_global"],
            mode="lines", name="Pre-COVID (through 2019)",
            line=dict(color="#2ecc71", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=gap_full["year"], y=gap_full["bz_gap_over_global"],
            mode="lines", name="Full Period (through 2023)",
            line=dict(color="#e74c3c", width=2.5),
        ))
        fig.add_vrect(x0=2020, x1=2023, fillcolor="red", opacity=0.05,
                      line_width=0, annotation_text="COVID period",
                      annotation_position="top left")
        fig.update_layout(
            title="Blue Zone Advantage Over Global Average",
            xaxis_title="Year", yaxis_title="Gap (years)",
            height=420, template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Pre-COVID endpoint (2019):**")
            st.markdown(f"- BZ avg: {s['pre_covid_bz_mean']:.1f} yr")
            st.markdown(f"- Global avg: {s['pre_covid_global_mean']:.1f} yr")
            st.markdown(f"- Gap: **{s['pre_covid_gap']:.1f} yr**")
            st.markdown(f"- Global SD: {s['pre_covid_sigma']:.2f} yr")
        with col2:
            st.markdown("**Full period endpoint (2023):**")
            st.markdown(f"- Gap: **{s['full_2023_gap']:.1f} yr**")
            st.markdown(f"- Global SD: {s['full_2023_sigma']:.2f} yr")
            gap_diff = s['full_2023_gap'] - s['pre_covid_gap']
            if gap_diff < 0:
                st.markdown(f"- COVID **accelerated convergence** by {abs(gap_diff):.1f} yr")
            else:
                st.markdown(f"- COVID **slowed convergence** by {gap_diff:.1f} yr")

    # Sigma convergence comparison
    st.markdown('<h3 class="section-hdr">Sigma Convergence: Pre-COVID vs Full</h3>',
                unsafe_allow_html=True)

    if not sigma_pre.empty and not sigma_full.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sigma_pre["year"], y=sigma_pre["le_std"],
            mode="lines", name="Pre-COVID",
            line=dict(color="#2ecc71", width=2.5),
        ))
        fig2.add_trace(go.Scatter(
            x=sigma_full["year"], y=sigma_full["le_std"],
            mode="lines", name="Full Period",
            line=dict(color="#e74c3c", width=2.5),
        ))
        fig2.add_vrect(x0=2020, x1=2023, fillcolor="red", opacity=0.05, line_width=0)
        fig2.update_layout(
            title="Global LE Standard Deviation Over Time",
            xaxis_title="Year", yaxis_title="Std Dev (years)",
            height=380, template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Country-level COVID damage
    st.markdown('<h3 class="section-hdr">Country-Level COVID Impact (All 93 Countries)</h3>',
                unsafe_allow_html=True)

    if not impact_df.empty:
        fig3 = go.Figure()
        non_bz = impact_df[impact_df["is_blue_zone"] == 0]
        bz = impact_df[impact_df["is_blue_zone"] == 1]

        fig3.add_trace(go.Bar(
            x=non_bz["iso_code"], y=non_bz["drop_2020"],
            name="Other Countries", marker_color="#95a5a6",
        ))
        fig3.add_trace(go.Bar(
            x=bz["iso_code"], y=bz["drop_2020"],
            name="Blue Zone Countries", marker_color="#e74c3c",
        ))
        fig3.update_layout(
            title="Life Expectancy Change 2019 to 2020 (by country)",
            xaxis_title="Country", yaxis_title="Change (years)",
            height=400, template="plotly_white",
            xaxis=dict(tickangle=90, tickfont=dict(size=8)),
        )
        fig3.add_hline(y=0, line_color="black", line_width=0.5)
        st.plotly_chart(fig3, use_container_width=True)

    # Recovery chart
    if not impact_df.empty and "recovery_2023" in impact_df.columns:
        st.markdown('<h3 class="section-hdr">Recovery Status by 2023</h3>',
                    unsafe_allow_html=True)
        recovery = impact_df.dropna(subset=["recovery_2023"]).copy()
        recovery["recovered"] = recovery["recovery_2023"] >= 0
        recovery = recovery.sort_values("recovery_2023")

        colors = ["#27ae60" if r else "#c0392b" for r in recovery["recovered"]]
        fig4 = go.Figure(go.Bar(
            x=recovery["iso_code"], y=recovery["recovery_2023"],
            marker_color=colors,
            hovertemplate="%{x}: %{y:.1f} years<extra></extra>",
        ))
        fig4.update_layout(
            title="Net LE Change 2019 to 2023 (Green = recovered, Red = still below 2019)",
            xaxis_title="Country", yaxis_title="Net Change (years)",
            height=400, template="plotly_white",
            xaxis=dict(tickangle=90, tickfont=dict(size=8)),
        )
        fig4.add_hline(y=0, line_color="black", line_width=1)
        st.plotly_chart(fig4, use_container_width=True)

    # COVID years overlay: all BZ countries zoomed
    st.markdown('<h3 class="section-hdr">Blue Zone Countries: COVID Years Zoomed (2015-2023)</h3>',
                unsafe_allow_html=True)

    fig_zoom = go.Figure()
    zoom = hist_full[(hist_full["year"] >= 2015) & (hist_full["year"] <= 2023)]
    for iso in sorted(BZ_COLORS):
        cz = zoom[zoom["iso_code"] == iso][["year", "life_expectancy"]].dropna().sort_values("year")
        if not cz.empty:
            fig_zoom.add_trace(go.Scatter(
                x=cz["year"], y=cz["life_expectancy"], mode="lines+markers",
                name=BZ_NAMES[iso], line=dict(color=BZ_COLORS[iso], width=2.5),
                marker=dict(size=7),
            ))
    global_zoom = zoom.groupby("year")["life_expectancy"].mean().dropna()
    fig_zoom.add_trace(go.Scatter(
        x=global_zoom.index, y=global_zoom.values, mode="lines+markers",
        name="Global Average", line=dict(color="#95a5a6", width=2, dash="dash"),
        marker=dict(size=5),
    ))
    fig_zoom.add_vrect(x0=2020, x1=2021.5, fillcolor="red", opacity=0.08, line_width=0)
    fig_zoom.update_layout(
        title="All Blue Zone Countries vs Global Average (2015-2023)",
        xaxis_title="Year", yaxis_title="Life Expectancy (years)",
        height=480, template="plotly_white",
    )
    st.plotly_chart(fig_zoom, use_container_width=True)

    # Recovery trajectory: change relative to 2019
    st.markdown('<h3 class="section-hdr">Recovery Trajectory: Change from 2019 Baseline</h3>',
                unsafe_allow_html=True)

    fig_recov = go.Figure()
    for iso in sorted(BZ_COLORS):
        c = hist_full[hist_full["iso_code"] == iso].sort_values("year")
        le_2019 = c[c["year"] == 2019]["life_expectancy"]
        if le_2019.empty:
            continue
        le_2019_val = le_2019.iloc[0]
        cdata = c[(c["year"] >= 2019) & (c["year"] <= 2023)][["year", "life_expectancy"]].dropna()
        if not cdata.empty:
            relative = cdata["life_expectancy"] - le_2019_val
            fig_recov.add_trace(go.Scatter(
                x=cdata["year"], y=relative, mode="lines+markers",
                name=BZ_NAMES[iso], line=dict(color=BZ_COLORS[iso], width=2.5),
                marker=dict(size=7),
            ))
    # Global average relative
    g19 = hist_full[hist_full["year"] == 2019]["life_expectancy"].mean()
    g_yrs, g_vals = [], []
    for yr in range(2019, 2024):
        g = hist_full[hist_full["year"] == yr]["life_expectancy"].mean()
        g_yrs.append(yr)
        g_vals.append(g - g19)
    fig_recov.add_trace(go.Scatter(
        x=g_yrs, y=g_vals, mode="lines+markers", name="Global Average",
        line=dict(color="#95a5a6", width=2, dash="dash"), marker=dict(size=5),
    ))
    fig_recov.add_hline(y=0, line_color="black", line_width=1)
    fig_recov.update_layout(
        title="Change from 2019 Level (Below zero = still worse than pre-COVID)",
        xaxis_title="Year", yaxis_title="Change from 2019 (years)",
        height=450, template="plotly_white",
    )
    st.plotly_chart(fig_recov, use_container_width=True)

    # Trend vs reality overlay
    st.markdown('<h3 class="section-hdr">Trend vs Reality: What 2020-2023 Should Have Been</h3>',
                unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    for col, (lbl, is_bz) in zip([col_t1, col_t2],
                                  [("Global Average", False), ("Blue Zone Average", True)]):
        if is_bz:
            sub = hist_full[hist_full["is_blue_zone"] == 1]
        else:
            sub = hist_full
        yearly = sub.groupby("year")["life_expectancy"].mean().dropna()
        trend_data = yearly[(yearly.index >= 2000) & (yearly.index <= 2019)]
        if len(trend_data) < 5:
            continue
        x_arr = np.array(trend_data.index, dtype=float)
        y_arr = np.array(trend_data.values, dtype=float)
        slope = np.polyfit(x_arr, y_arr, 1)
        trend_fn = np.poly1d(slope)
        extrap_x = list(range(2000, 2024))
        extrap_y = [trend_fn(yr) for yr in extrap_x]
        actual_x = [yr for yr in range(2000, 2024) if yr in yearly.index]
        actual_y = [yearly[yr] for yr in actual_x]

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=actual_x, y=actual_y, mode="lines+markers", name="Actual",
            line=dict(color="#2c3e50", width=2.5), marker=dict(size=5),
        ))
        fig_t.add_trace(go.Scatter(
            x=extrap_x, y=extrap_y, mode="lines", name="Pre-COVID Trend",
            line=dict(color="#27ae60", width=2, dash="dash"),
        ))
        fig_t.add_vrect(x0=2020, x1=2023, fillcolor="red", opacity=0.06, line_width=0)
        fig_t.update_layout(
            title=f"{lbl}: Trend vs Reality",
            xaxis_title="Year", yaxis_title="Life Expectancy (years)",
            height=380, template="plotly_white",
        )
        with col:
            st.plotly_chart(fig_t, use_container_width=True)

    # Key takeaway
    st.markdown('<h3 class="section-hdr">What This Means for the Analysis</h3>',
                unsafe_allow_html=True)
    st.markdown(
        "COVID-19 temporarily reversed years of progress in global life expectancy. "
        "The pre-COVID dataset (1960-2019) gives the cleanest picture of long-term secular trends "
        "without pandemic noise. The full dataset (1960-2023) is honest about what actually happened "
        "but includes a massive disruption that will fade over time.\n\n"
        "**Recommendation:** Use the pre-COVID data for understanding the underlying 60-year trend. "
        "Use the full data for completeness and to acknowledge the real impact of the pandemic. "
        "Neither version is wrong -- they answer different questions."
    )


def render_stats_tab():
    """Render the Statistical Deep Dive tab with significance tests, controls,
    and sensitivity analysis."""

    st.markdown('<h3 class="section-hdr">Statistical Significance and Controls</h3>',
                unsafe_allow_html=True)
    st.markdown(
        "This section presents formal statistical tests that go beyond descriptive "
        "analysis. Every claim about Blue Zone advantage, convergence, and COVID impact "
        "is backed by confidence intervals, p-values, and effect sizes."
    )

    # -----------------------------------------------------------------------
    # 1. Gap Confidence Intervals
    # -----------------------------------------------------------------------
    gap_ci = load_analysis_file("", "gap_confidence_intervals")
    if not gap_ci.empty:
        st.markdown('<h3 class="section-hdr">1. Blue Zone Gap with 95% Confidence Intervals</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            "Bootstrap confidence intervals (1,000 resamples) test whether a random "
            "set of 5 countries could produce a gap this large by chance. The gap is "
            "significant at p<0.05 for every year."
        )

        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(
            x=gap_ci["year"], y=gap_ci["gap"],
            mode="lines", name="BZ Gap",
            line=dict(color="#2ecc71", width=2.5),
        ))
        fig_ci.add_trace(go.Scatter(
            x=pd.concat([gap_ci["year"], gap_ci["year"][::-1]]),
            y=pd.concat([gap_ci["ci_upper"], gap_ci["ci_lower"][::-1]]),
            fill="toself", fillcolor="rgba(46,204,113,0.15)",
            line=dict(width=0), name="95% CI",
        ))
        fig_ci.add_hline(y=0, line_color="black", line_width=0.5)
        fig_ci.update_layout(
            title="Blue Zone Gap Over Global Average with 95% Bootstrap CI",
            xaxis_title="Year", yaxis_title="Gap (years)",
            height=420, template="plotly_white",
        )
        st.plotly_chart(fig_ci, use_container_width=True)

        # Key stats
        latest = gap_ci.iloc[-1]
        earliest = gap_ci.iloc[0]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Gap (1960)", f"{earliest['gap']:.1f} yr",
                     help=f"95% CI: [{earliest['ci_lower']:.1f}, {earliest['ci_upper']:.1f}]")
        col_b.metric("Gap (latest)", f"{latest['gap']:.1f} yr",
                     help=f"95% CI: [{latest['ci_lower']:.1f}, {latest['ci_upper']:.1f}]")
        n_sig = gap_ci["significant_at_05"].sum()
        col_c.metric("Years Significant", f"{n_sig} / {len(gap_ci)}")

    # -----------------------------------------------------------------------
    # 2. Beta Convergence P-values
    # -----------------------------------------------------------------------
    beta_pv = load_analysis_file("", "beta_convergence_pvalues")
    if not beta_pv.empty:
        st.markdown('<h3 class="section-hdr">2. Beta Convergence with P-values</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            "Each decade's beta correlation (do low-LE countries catch up faster?) "
            "is tested for significance. Bonferroni correction adjusts for 6 simultaneous tests."
        )

        fig_beta = make_subplots(specs=[[{"secondary_y": True}]])
        fig_beta.add_trace(go.Bar(
            x=beta_pv["decade"], y=beta_pv["beta_r"],
            name="Beta r", marker_color=["#2ecc71" if s else "#e74c3c"
                                          for s in beta_pv["significant_at_05"]],
            text=[f"{r:.3f}" for r in beta_pv["beta_r"]],
            textposition="outside",
        ), secondary_y=False)
        fig_beta.add_trace(go.Scatter(
            x=beta_pv["decade"], y=beta_pv["p_value_bonferroni"],
            mode="lines+markers", name="Bonferroni p-value",
            line=dict(color="#8e44ad", width=2), marker=dict(size=8),
        ), secondary_y=True)
        fig_beta.add_hline(y=0.05, line_dash="dash", line_color="red",
                           opacity=0.5, secondary_y=True,
                           annotation_text="p=0.05")
        fig_beta.update_layout(
            title="Beta Convergence by Decade (green = significant after Bonferroni)",
            height=420, template="plotly_white",
        )
        fig_beta.update_yaxes(title_text="Beta Correlation (r)", secondary_y=False)
        fig_beta.update_yaxes(title_text="P-value (Bonferroni)", secondary_y=True, type="log")
        st.plotly_chart(fig_beta, use_container_width=True)

        n_sig = beta_pv["significant_at_05"].sum()
        st.markdown(f"**{n_sig} of {len(beta_pv)} decades** show statistically significant "
                    f"beta convergence after Bonferroni correction.")

    # -----------------------------------------------------------------------
    # 3. Partial Correlations (raw vs GDP-controlled)
    # -----------------------------------------------------------------------
    partial = load_analysis_file("", "partial_correlations")
    if not partial.empty:
        st.markdown('<h3 class="section-hdr">3. Partial Correlations: Raw vs GDP-Controlled</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            "Many indicators correlate with life expectancy simply because both track "
            "wealth. Partial correlations remove the GDP effect to reveal which factors "
            "have independent predictive value."
        )

        fig_partial = go.Figure()
        indicators = partial["indicator"].tolist()
        fig_partial.add_trace(go.Bar(
            x=indicators, y=partial["raw_correlation"],
            name="Raw correlation", marker_color="#3498DB",
            text=[f"{v:.2f}" for v in partial["raw_correlation"]],
            textposition="outside",
        ))
        fig_partial.add_trace(go.Bar(
            x=indicators, y=partial["partial_correlation_gdp_controlled"],
            name="GDP-controlled", marker_color="#E74C3C",
            text=[f"{v:.2f}" for v in partial["partial_correlation_gdp_controlled"]],
            textposition="outside",
        ))
        fig_partial.update_layout(
            title="Correlation with Life Expectancy: Raw vs GDP-Controlled",
            xaxis_title="Indicator", yaxis_title="Correlation (r)",
            barmode="group", height=450, template="plotly_white",
            xaxis=dict(tickangle=30),
        )
        st.plotly_chart(fig_partial, use_container_width=True)

        # Highlight key findings
        gdp_explained = partial[partial["gdp_explains_relationship"] == True]
        if len(gdp_explained) > 0:
            names = ", ".join(gdp_explained["indicator"].tolist())
            st.markdown(f"**GDP explains the relationship for:** {names} "
                        f"(correlation drops substantially after controlling for GDP)")

        # Data table
        display_partial = partial[["indicator", "n_countries", "raw_correlation",
                                    "partial_correlation_gdp_controlled",
                                    "gdp_explains_relationship"]].copy()
        display_partial.columns = ["Indicator", "N", "Raw r", "GDP-Controlled r",
                                   "GDP Explains?"]
        st.dataframe(display_partial.round(3), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # 4. Multiple Regression
    # -----------------------------------------------------------------------
    reg = load_analysis_file("", "regression_results")
    reg_meta = load_analysis_file("", "regression_meta")
    reg_vif = load_analysis_file("", "regression_vif")
    if not reg.empty:
        st.markdown('<h3 class="section-hdr">4. Multiple Regression: Predictors of Life Expectancy</h3>',
                    unsafe_allow_html=True)

        if not reg_meta.empty:
            rm = reg_meta.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R-squared", f"{rm['r_squared']:.3f}")
            c2.metric("Adj R-squared", f"{rm['adj_r_squared']:.3f}")
            c3.metric("F-statistic", f"{rm['f_statistic']:.1f}")
            c4.metric("N observations", f"{int(rm['n_observations'])}")

        # Coefficient chart (exclude intercept)
        reg_coef = reg[reg["predictor"] != "(intercept)"].copy()
        colors = ["#2ecc71" if s else "#95a5a6" for s in reg_coef["significant_at_05"]]
        fig_reg = go.Figure(go.Bar(
            x=reg_coef["predictor"], y=reg_coef["t_statistic"],
            marker_color=colors,
            text=[f"p={p:.4f}" for p in reg_coef["p_value"]],
            textposition="outside",
        ))
        fig_reg.add_hline(y=1.96, line_dash="dash", line_color="red", opacity=0.4,
                          annotation_text="t=1.96 (p=0.05)")
        fig_reg.add_hline(y=-1.96, line_dash="dash", line_color="red", opacity=0.4)
        fig_reg.update_layout(
            title="Regression: T-statistics (green = significant at p<0.05)",
            xaxis_title="Predictor", yaxis_title="T-statistic",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        # Coefficient table
        st.markdown("**Regression Coefficients**")
        display_reg = reg.copy()
        display_reg.columns = ["Predictor", "Coefficient", "Std Error", "P-value",
                               "T-stat", "Significant"]
        st.dataframe(display_reg.round(4), use_container_width=True, hide_index=True)

        # VIF
        if not reg_vif.empty:
            st.markdown("**Variance Inflation Factors (VIF > 10 indicates severe multicollinearity)**")
            st.dataframe(reg_vif.round(2), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # 5. Sensitivity Analysis (Drop-One-Country)
    # -----------------------------------------------------------------------
    sensitivity = load_analysis_file("", "sensitivity_drop_one")
    if not sensitivity.empty:
        st.markdown('<h3 class="section-hdr">5. Sensitivity Analysis: Drop-One-Country</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            "How much does each Blue Zone country affect the overall result? "
            "Removing each country one at a time reveals which countries drive "
            "the Blue Zone advantage."
        )

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Bar(
            x=sensitivity["dropped_country"],
            y=sensitivity["gap_change"],
            marker_color=["#e74c3c" if v > 0 else "#2ecc71"
                          for v in sensitivity["gap_change"]],
            text=[f"{v:+.2f} yr" for v in sensitivity["gap_change"]],
            textposition="outside",
        ))
        fig_sens.add_hline(y=0, line_color="black", line_width=0.5)
        fig_sens.update_layout(
            title="Effect of Removing Each BZ Country on 2019 Gap",
            xaxis_title="Country Removed", yaxis_title="Change in Gap (years)",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        # Table
        display_sens = sensitivity[["dropped_country", "dropped_le_2019",
                                     "remaining_bz_mean", "gap_2019",
                                     "gap_change"]].copy()
        display_sens.columns = ["Dropped", "That Country LE", "Remaining BZ Mean",
                                "New Gap", "Gap Change"]
        st.dataframe(display_sens.round(2), use_container_width=True, hide_index=True)

        # Find biggest outlier
        biggest = sensitivity.loc[sensitivity["gap_change"].abs().idxmax()]
        direction = "increases" if biggest["gap_change"] > 0 else "decreases"
        st.markdown(
            f"**Key finding:** Removing {biggest['dropped_country']} {direction} the "
            f"gap by {abs(biggest['gap_change']):.2f} years -- the largest single-country effect."
        )

    # -----------------------------------------------------------------------
    # 6. Gender Analysis
    # -----------------------------------------------------------------------
    gender = load_analysis_file("", "gender_gap_analysis")
    if not gender.empty:
        st.markdown('<h3 class="section-hdr">6. Gender Analysis: Male vs Female LE Gap</h3>',
                    unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            # Male and female BZ gap over time
            fig_gender = go.Figure()
            fig_gender.add_trace(go.Scatter(
                x=gender["year"], y=gender["male_gap"],
                mode="lines", name="Male BZ gap",
                line=dict(color="#3498DB", width=2),
            ))
            fig_gender.add_trace(go.Scatter(
                x=gender["year"], y=gender["female_gap"],
                mode="lines", name="Female BZ gap",
                line=dict(color="#E74C3C", width=2),
            ))
            fig_gender.update_layout(
                title="BZ Advantage by Gender Over Time",
                xaxis_title="Year", yaxis_title="Gap (years)",
                height=380, template="plotly_white",
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        with col_g2:
            # Gender gap within BZ vs global
            fig_gg = go.Figure()
            fig_gg.add_trace(go.Scatter(
                x=gender["year"], y=gender["bz_gender_gap"],
                mode="lines", name="BZ gender gap (F-M)",
                line=dict(color="#9B59B6", width=2),
            ))
            fig_gg.add_trace(go.Scatter(
                x=gender["year"], y=gender["global_gender_gap"],
                mode="lines", name="Global gender gap (F-M)",
                line=dict(color="#95a5a6", width=2, dash="dash"),
            ))
            fig_gg.update_layout(
                title="Gender Gap (Female-Male LE) Over Time",
                xaxis_title="Year", yaxis_title="Female - Male LE (years)",
                height=380, template="plotly_white",
            )
            st.plotly_chart(fig_gg, use_container_width=True)

        # Key stats
        latest_g = gender.dropna(subset=["male_gap", "female_gap"]).iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Male BZ Gap (latest)", f"+{latest_g['male_gap']:.1f} yr")
        c2.metric("Female BZ Gap (latest)", f"+{latest_g['female_gap']:.1f} yr")
        stronger = latest_g["bz_advantage_stronger_for"]
        c3.metric("BZ Advantage Stronger For", stronger.capitalize())

    # -----------------------------------------------------------------------
    # 7. Population-Weighted vs Unweighted
    # -----------------------------------------------------------------------
    weighted = load_analysis_file("", "population_weighted_gap")
    if not weighted.empty:
        st.markdown('<h3 class="section-hdr">7. Population-Weighted vs Unweighted Gap</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            "The unweighted average treats each country equally. Population weighting "
            "gives more influence to populous countries like China and India, which "
            "substantially changes the global average."
        )

        fig_wt = go.Figure()
        fig_wt.add_trace(go.Scatter(
            x=weighted["year"], y=weighted["gap_unweighted"],
            mode="lines", name="Unweighted gap",
            line=dict(color="#2ecc71", width=2.5),
        ))
        fig_wt.add_trace(go.Scatter(
            x=weighted["year"], y=weighted["gap_weighted"],
            mode="lines", name="Population-weighted gap",
            line=dict(color="#e74c3c", width=2.5),
        ))
        fig_wt.update_layout(
            title="BZ Gap: Unweighted vs Population-Weighted",
            xaxis_title="Year", yaxis_title="Gap (years)",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_wt, use_container_width=True)

        latest_w = weighted.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Unweighted Gap", f"+{latest_w['gap_unweighted']:.1f} yr")
        c2.metric("Weighted Gap", f"+{latest_w['gap_weighted']:.1f} yr")
        c3.metric("Difference", f"{latest_w['gap_difference']:.1f} yr")

    # -----------------------------------------------------------------------
    # 8. Regional Peer Comparison
    # -----------------------------------------------------------------------
    regional = load_analysis_file("", "regional_peer_comparison")
    outlier_tests = load_analysis_file("", "regional_outlier_tests")
    if not regional.empty:
        st.markdown('<h3 class="section-hdr">8. Regional Peer Comparison</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            "Each Blue Zone country is compared to its regional peers. "
            "Are these countries exceptional within their regions, or do they "
            "belong to already-strong regions?"
        )

        # Selector for which BZ country to show
        bz_countries_in_regional = regional["bz_country"].unique().tolist()
        selected_bz = st.selectbox("Select Blue Zone country",
                                   bz_countries_in_regional,
                                   key="regional_select")

        rc = regional[regional["bz_country"] == selected_bz].sort_values("year")
        if not rc.empty:
            fig_reg_peer = go.Figure()
            fig_reg_peer.add_trace(go.Scatter(
                x=rc["year"], y=rc["bz_le"],
                mode="lines", name=selected_bz,
                line=dict(color="#3498DB", width=2.5),
            ))
            fig_reg_peer.add_trace(go.Scatter(
                x=rc["year"], y=rc["regional_mean"],
                mode="lines", name=f"Regional avg ({rc['region'].iloc[0]})",
                line=dict(color="#95a5a6", width=2, dash="dash"),
            ))
            # Fill area between
            fig_reg_peer.add_trace(go.Scatter(
                x=pd.concat([rc["year"], rc["year"][::-1]]),
                y=pd.concat([rc["bz_le"], rc["regional_mean"][::-1]]),
                fill="toself", fillcolor="rgba(52,152,219,0.1)",
                line=dict(width=0), showlegend=False, name="",
            ))
            fig_reg_peer.update_layout(
                title=f"{selected_bz} vs {rc['region'].iloc[0]} Peers",
                xaxis_title="Year", yaxis_title="Life Expectancy (years)",
                height=420, template="plotly_white",
            )
            st.plotly_chart(fig_reg_peer, use_container_width=True)

        # Outlier test results
        if not outlier_tests.empty:
            st.markdown("**Regional Outlier Tests (2000-2019 mean advantage, t-test)**")
            display_out = outlier_tests[["bz_country", "region",
                                          "mean_advantage_2000_2019",
                                          "t_statistic", "p_value",
                                          "is_regional_outlier"]].copy()
            display_out.columns = ["Country", "Region", "Mean Advantage",
                                   "T-stat", "P-value", "Outlier?"]
            st.dataframe(display_out.round(3), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # 9. Income Group Convergence
    # -----------------------------------------------------------------------
    income = load_analysis_file("", "income_group_convergence")
    if not income.empty:
        st.markdown('<h3 class="section-hdr">9. Income Group Convergence</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            "Are low-income countries catching up to high-income countries? "
            "This breaks down convergence by World Bank income groups."
        )

        fig_inc = go.Figure()
        inc_colors = {"Low income": "#E74C3C", "Lower-middle income": "#F39C12",
                      "Upper-middle income": "#3498DB", "High income": "#2ECC71"}
        for _, row in income.iterrows():
            group = row["income_group"]
            fig_inc.add_trace(go.Bar(
                x=[group], y=[row["le_improvement"]],
                name=group, marker_color=inc_colors.get(group, "#666"),
                text=f"+{row['le_improvement']:.1f} yr",
                textposition="outside",
                showlegend=False,
            ))
        fig_inc.update_layout(
            title="Life Expectancy Improvement by Income Group (1960s to Recent)",
            xaxis_title="Income Group", yaxis_title="LE Improvement (years)",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_inc, use_container_width=True)

        # Table
        display_inc = income[["income_group", "n_countries", "le_1960s",
                               "le_recent", "le_improvement",
                               "sigma_converging"]].copy()
        display_inc.columns = ["Group", "N Countries", "LE 1960s", "LE Recent",
                               "Improvement", "Within-Group Convergence"]
        st.dataframe(display_inc.round(1), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # 10. Decade Improvement Tests
    # -----------------------------------------------------------------------
    decade_tests = load_analysis_file("", "decade_improvement_tests")
    if not decade_tests.empty:
        st.markdown('<h3 class="section-hdr">10. BZ vs Non-BZ Decade Gains: Statistical Tests</h3>',
                    unsafe_allow_html=True)

        fig_dt = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dt.add_trace(go.Bar(
            x=decade_tests["decade"], y=decade_tests["bz_mean_gain"],
            name="BZ Mean Gain", marker_color="#3498DB",
        ), secondary_y=False)
        fig_dt.add_trace(go.Bar(
            x=decade_tests["decade"], y=decade_tests["non_bz_mean_gain"],
            name="Non-BZ Mean Gain", marker_color="#bdc3c7",
        ), secondary_y=False)
        fig_dt.add_trace(go.Scatter(
            x=decade_tests["decade"], y=decade_tests["cohens_d"].abs(),
            mode="lines+markers", name="|Cohen's d|",
            line=dict(color="#E74C3C", width=2), marker=dict(size=8),
        ), secondary_y=True)
        fig_dt.add_hline(y=0.5, line_dash="dash", line_color="red",
                         opacity=0.3, secondary_y=True,
                         annotation_text="d=0.5 (medium)")
        fig_dt.update_layout(
            title="BZ vs Non-BZ Gains by Decade with Effect Sizes",
            barmode="group", height=420, template="plotly_white",
        )
        fig_dt.update_yaxes(title_text="LE Gain (years)", secondary_y=False)
        fig_dt.update_yaxes(title_text="|Cohen's d|", secondary_y=True)
        st.plotly_chart(fig_dt, use_container_width=True)

        display_dt = decade_tests[["decade", "bz_mean_gain", "non_bz_mean_gain",
                                    "t_statistic", "p_value", "cohens_d",
                                    "effect_size"]].copy()
        display_dt.columns = ["Decade", "BZ Gain", "Non-BZ Gain", "T-stat",
                              "P-value", "Cohen's d", "Effect Size"]
        st.dataframe(display_dt.round(4), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # 11. Time-Series Tests
    # -----------------------------------------------------------------------
    ts_tests = load_analysis_file("", "time_series_tests")
    sigma_test = load_analysis_file("", "sigma_convergence_test")
    if not ts_tests.empty or not sigma_test.empty:
        st.markdown('<h3 class="section-hdr">11. Time-Series and Stationarity Tests</h3>',
                    unsafe_allow_html=True)

        if not ts_tests.empty:
            display_ts = ts_tests[["test", "description", "statistic",
                                    "p_value", "conclusion"]].copy()
            display_ts.columns = ["Test", "Description", "Statistic",
                                  "P-value", "Conclusion"]
            st.dataframe(display_ts.round(4), use_container_width=True, hide_index=True)

        if not sigma_test.empty:
            st.markdown("**Sigma Convergence OLS Trend Test**")
            display_sig = sigma_test[["period", "slope", "slope_p_value",
                                       "r_squared", "sigma_declining"]].copy()
            display_sig.columns = ["Period", "Slope (yr/yr)", "P-value",
                                   "R-squared", "SD Declining"]
            st.dataframe(display_sig.round(4), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 5: Predictive Analysis
# ---------------------------------------------------------------------------
def render_prediction_tab():
    """Tab 5: ML prediction and overperformance analysis."""
    st.markdown('<h2 class="section-hdr">ML Life Expectancy Prediction and Overperformance Analysis</h2>',
                unsafe_allow_html=True)

    st.markdown(
        "Regularized ML models predict life expectancy from 10 country-level features. "
        "Countries where **actual LE exceeds predicted LE** are \"overperformers\" -- places "
        "where unmeasured factors may boost longevity. Known Blue Zone countries should "
        "appear as overperformers if the model captures real patterns."
    )

    # Load ML outputs
    model_comp = load_analysis_file("", "ml_model_comparison")
    feat_imp = load_analysis_file("", "ml_feature_importance")
    feat_sel = load_analysis_file("", "ml_feature_selection_report")
    residual_df = load_analysis_file("", "ml_residual_analysis")
    hidden_bz = load_analysis_file("", "ml_hidden_blue_zones")
    underperf = load_analysis_file("", "ml_underperformers")

    if model_comp.empty or residual_df.empty:
        st.warning("ML prediction outputs not found. Run ml_prediction.py first.")
        return

    # --- Model comparison ---
    st.markdown('<h3 class="section-hdr">1. Model Comparison (LOOCV, n=93)</h3>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    best = model_comp.iloc[0]
    c1.metric("Best Model", best['model'])
    c2.metric("LOOCV R-squared", f"{best['loocv_r2']:.4f}")
    c3.metric("LOOCV RMSE", f"{best['loocv_rmse']:.2f} years")

    fig_comp = go.Figure()
    colors = ["#2196F3" if r2 == model_comp['loocv_r2'].max() else "#90CAF9"
              for r2 in model_comp['loocv_r2']]
    fig_comp.add_trace(go.Bar(
        y=model_comp['model'], x=model_comp['loocv_r2'],
        orientation='h', marker_color=colors,
        text=[f"R2={v:.3f}" for v in model_comp['loocv_r2']],
        textposition='outside',
    ))
    fig_comp.add_vline(x=0.706, line_dash="dash", line_color="red",
                       annotation_text="Baseline OLS (R2=0.706)")
    fig_comp.update_layout(
        title="LOOCV R-squared by Model",
        xaxis_title="R-squared", height=300, template="plotly_white",
        xaxis=dict(range=[0, max(model_comp['loocv_r2'].max() * 1.15, 0.75)]),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("**Model Details**")
    st.dataframe(model_comp.round(4), use_container_width=True, hide_index=True)

    # --- Feature importance ---
    if not feat_imp.empty:
        st.markdown('<h3 class="section-hdr">2. Feature Importance</h3>',
                    unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            fig_lasso = go.Figure(go.Bar(
                y=feat_imp.sort_values('lasso_coef')['feature'],
                x=feat_imp.sort_values('lasso_coef')['lasso_coef'],
                orientation='h',
                marker_color=['#E74C3C' if c < 0 else '#2ECC71'
                              for c in feat_imp.sort_values('lasso_coef')['lasso_coef']],
            ))
            fig_lasso.update_layout(
                title="Lasso Coefficients (0 = eliminated)",
                xaxis_title="Coefficient", height=400, template="plotly_white",
            )
            st.plotly_chart(fig_lasso, use_container_width=True)

        with col_r:
            fig_rf = go.Figure(go.Bar(
                y=feat_imp.sort_values('rf_importance')['feature'],
                x=feat_imp.sort_values('rf_importance')['rf_importance'],
                orientation='h', marker_color='#3498DB',
            ))
            fig_rf.update_layout(
                title="Random Forest Permutation Importance",
                xaxis_title="Importance", height=400, template="plotly_white",
            )
            st.plotly_chart(fig_rf, use_container_width=True)

        st.dataframe(
            feat_imp[['feature', 'lasso_coef', 'rf_importance', 'univariate_r', 'combined_rank']].round(4),
            use_container_width=True, hide_index=True,
        )

    # --- Actual vs predicted scatter ---
    st.markdown('<h3 class="section-hdr">3. Actual vs Predicted Life Expectancy</h3>',
                unsafe_allow_html=True)

    residual_df['is_bz_label'] = residual_df['iso_code'].apply(
        lambda x: BZ_NAMES.get(x, '') if x in BZ_NAMES else ''
    )

    fig_scatter = go.Figure()

    # Non-BZ points
    for cls, color, symbol in [('overperformer', '#27ae60', 'triangle-up'),
                                ('as_expected', '#95a5a6', 'circle'),
                                ('underperformer', '#c0392b', 'triangle-down')]:
        subset = residual_df[(residual_df['classification'] == cls) &
                             (~residual_df['iso_code'].isin(BLUE_ZONE_ISOS))]
        fig_scatter.add_trace(go.Scatter(
            x=subset['predicted_le'], y=subset['actual_le'],
            mode='markers', name=f'{cls} ({len(subset)})',
            marker=dict(color=color, size=8, symbol=symbol, opacity=0.6),
            text=subset['country_name'],
            hovertemplate='%{text}<br>Predicted: %{x:.1f}<br>Actual: %{y:.1f}<extra></extra>',
        ))

    # BZ points
    bz_rows = residual_df[residual_df['iso_code'].isin(BLUE_ZONE_ISOS)]
    fig_scatter.add_trace(go.Scatter(
        x=bz_rows['predicted_le'], y=bz_rows['actual_le'],
        mode='markers+text', name='Blue Zone countries',
        marker=dict(color='gold', size=16, symbol='star', line=dict(width=1, color='black')),
        text=bz_rows['country_name'],
        textposition='top right', textfont=dict(size=10),
        hovertemplate='%{text}<br>Predicted: %{x:.1f}<br>Actual: %{y:.1f}<extra></extra>',
    ))

    # 1:1 line
    all_vals = list(residual_df['predicted_le']) + list(residual_df['actual_le'])
    line_min, line_max = min(all_vals) - 2, max(all_vals) + 2
    fig_scatter.add_trace(go.Scatter(
        x=[line_min, line_max], y=[line_min, line_max],
        mode='lines', name='Perfect prediction',
        line=dict(dash='dash', color='black', width=1),
    ))

    fig_scatter.update_layout(
        title="Actual vs Predicted Life Expectancy (LOOCV)<br>"
              "<sub>Points above the line = overperformers</sub>",
        xaxis_title="Predicted LE (years)", yaxis_title="Actual LE (years)",
        height=600, template="plotly_white",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- World map choropleth ---
    st.markdown('<h3 class="section-hdr">4. Overperformance World Map</h3>',
                unsafe_allow_html=True)

    fig_map = px.choropleth(
        residual_df,
        locations='iso_code',
        color='residual',
        hover_name='country_name',
        hover_data={'actual_le': ':.1f', 'predicted_le': ':.1f',
                    'residual': ':.2f', 'classification': True},
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title='Life Expectancy Overperformance (Green = lives longer than predicted)',
    )
    fig_map.update_layout(height=450, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # --- Overperformers and underperformers ---
    col_over, col_under = st.columns(2)

    with col_over:
        st.markdown('<h3 class="section-hdr">5. Top 15 Overperformers</h3>',
                    unsafe_allow_html=True)
        st.markdown("\"Hidden Blue Zone\" candidates -- countries living longer than their "
                    "indicators predict.")
        if not hidden_bz.empty:
            display_over = hidden_bz[['residual_rank', 'country_name', 'is_blue_zone',
                                       'actual_le', 'predicted_le', 'residual']].copy()
            display_over.columns = ['Rank', 'Country', 'Known BZ', 'Actual LE',
                                     'Predicted LE', 'Residual']
            display_over['Residual'] = display_over['Residual'].apply(lambda x: f"+{x:.1f}")
            st.dataframe(display_over, use_container_width=True, hide_index=True)

    with col_under:
        st.markdown('<h3 class="section-hdr">6. Bottom 15 Underperformers</h3>',
                    unsafe_allow_html=True)
        st.markdown("Countries living shorter than their indicators predict -- "
                    "potential areas for public health intervention.")
        if not underperf.empty:
            display_under = underperf[['residual_rank', 'country_name', 'is_blue_zone',
                                        'actual_le', 'predicted_le', 'residual']].copy()
            display_under.columns = ['Rank', 'Country', 'Known BZ', 'Actual LE',
                                      'Predicted LE', 'Residual']
            display_under['Residual'] = display_under['Residual'].apply(lambda x: f"{x:.1f}")
            st.dataframe(display_under, use_container_width=True, hide_index=True)

    # --- Blue Zone validation ---
    st.markdown('<h3 class="section-hdr">7. Blue Zone Validation</h3>',
                unsafe_allow_html=True)

    bz_validation = residual_df[residual_df['iso_code'].isin(BLUE_ZONE_ISOS)].copy()
    if not bz_validation.empty:
        display_bz = bz_validation[['country_name', 'actual_le', 'predicted_le',
                                     'residual', 'residual_zscore', 'residual_rank',
                                     'classification']].copy()
        display_bz.columns = ['Country', 'Actual LE', 'Predicted LE', 'Residual',
                               'Z-score', 'Rank (/93)', 'Classification']
        st.dataframe(display_bz.round(2), use_container_width=True, hide_index=True)

        bz_mean = bz_validation['residual'].mean()
        bz_no_usa = bz_validation[bz_validation['iso_code'] != 'USA']
        n_overperf = (bz_no_usa['classification'] == 'overperformer').sum()

        st.markdown(
            f"**Validation result:** BZ countries (excl. USA) have a mean residual of "
            f"**+{bz_no_usa['residual'].mean():.1f} years** and {n_overperf}/{len(bz_no_usa)} "
            f"are classified as overperformers. The USA ranks {int(bz_validation[bz_validation['iso_code']=='USA']['residual_rank'].values[0])}/93, "
            f"consistent with the national-level underperformance observed throughout this analysis."
        )

    # --- Caveats ---
    st.markdown('<h3 class="section-hdr">8. Caveats</h3>', unsafe_allow_html=True)
    st.markdown(
        "- **Small sample (n=93):** Results are exploratory, not definitive\n"
        "- **Cross-sectional:** No causal claims -- correlation-based pattern detection only\n"
        "- **Missing features:** Some important indicators (obesity, NCD mortality) were unavailable from the API\n"
        "- **Country vs region:** Country-level data masks sub-national variation (the Loma Linda problem)\n"
        "- **Residuals capture both unmeasured factors AND measurement error**\n"
        "- **\"Hidden Blue Zone\" is a label for statistical overperformance, not a clinical designation**"
    )

    # --- Feature selection report ---
    if not feat_sel.empty:
        with st.expander("Feature Selection Pipeline Details"):
            st.dataframe(feat_sel, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    hist_full = load_historical()
    proj = load_projections()
    impact_df = load_covid_impact()
    summary_df = load_comparison_summary()

    if hist_full.empty:
        st.error("Historical data not found. Run historical_data_collector.py first.")
        return

    # Split data
    hist_pre = hist_full[hist_full["year"] <= 2019].copy()

    # Load analysis for both periods
    gap_pre = load_analysis_file("pre_covid", "blue_zone_vs_global")
    sigma_pre = load_analysis_file("pre_covid", "sigma_convergence")
    beta_pre = load_analysis_file("pre_covid", "beta_convergence")
    decades_pre = load_analysis_file("pre_covid", "decade_improvements")

    gap_full = load_analysis_file("full_period", "blue_zone_vs_global")
    sigma_full = load_analysis_file("full_period", "sigma_convergence")
    beta_full = load_analysis_file("full_period", "beta_convergence")
    decades_full = load_analysis_file("full_period", "decade_improvements")

    # Fallback to main analysis dir if period-specific not found
    if gap_full.empty:
        gap_full = load_analysis_file("", "blue_zone_vs_global")
    if sigma_full.empty:
        sigma_full = load_analysis_file("", "sigma_convergence")
    if beta_full.empty:
        beta_full = load_analysis_file("", "beta_convergence")
    if decades_full.empty:
        decades_full = load_analysis_file("", "decade_improvements")

    # --- Header ---
    st.markdown('<h1 class="main-header">Blue Zones Longevity Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Pre-COVID vs Full Period -- Real Data from World Bank and WHO</p>',
                unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.title("Blue Zones Analysis")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Sources**")
    st.sidebar.markdown(
        "World Bank API (1960-2023)\n\n"
        "WHO Global Health Observatory\n\n"
        "93 countries, 5,952 observations\n\n"
        "16 World Bank indicators\n\n"
        "All real data -- no synthetic"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Blue Zone Countries**")
    for iso, name in sorted(BZ_NAMES.items()):
        st.sidebar.markdown(f"- {name}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Statistical Methods**")
    st.sidebar.markdown(
        "Bootstrap CIs (1,000 resamples)\n\n"
        "Bonferroni correction\n\n"
        "Partial correlations\n\n"
        "OLS regression with VIF\n\n"
        "Chow structural break test\n\n"
        "ADF stationarity tests"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ML Prediction**")
    st.sidebar.markdown(
        "Ridge, Lasso, ElasticNet\n\n"
        "Random Forest\n\n"
        "LOOCV (n=93)\n\n"
        "Overperformance residuals"
    )

    # --- Main tabs ---
    tab_pre, tab_full, tab_compare, tab_stats, tab_predict = st.tabs([
        "Pre-COVID (1960-2019)",
        "Full Period (1960-2023)",
        "COVID Impact Comparison",
        "Statistical Deep Dive",
        "Predictive Analysis",
    ])

    with tab_pre:
        render_period_tab(hist_pre, proj, gap_pre, sigma_pre, beta_pre, decades_pre,
                          "1960-2019", show_proj=True)

    with tab_full:
        render_period_tab(hist_full, proj, gap_full, sigma_full, beta_full, decades_full,
                          "1960-2023", show_proj=True)

    with tab_compare:
        render_comparison_tab(hist_full, impact_df, summary_df,
                              gap_pre, gap_full, sigma_pre, sigma_full)

    with tab_stats:
        render_stats_tab()

    with tab_predict:
        render_prediction_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;font-size:0.85rem'>"
        "Blue Zones Longevity Analysis | Real-world data from World Bank and WHO APIs | "
        "Pre-COVID and full-period analysis | Statistical significance tests with CIs and p-values"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
