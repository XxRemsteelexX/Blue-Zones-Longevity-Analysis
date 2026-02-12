#!/usr/bin/env python3
"""
Blue Zones Longevity Analysis Dashboard

Three-view interactive dashboard:
  1. Pre-COVID (1960-2019) -- clean secular trends
  2. Full Period (1960-2023) -- includes COVID disruption and recovery
  3. COVID Impact Comparison -- side-by-side analysis

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
def fig_timelines(hist, proj, selected_isos, title_suffix="", show_proj=True):
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
    fig.update_layout(
        title=f"Life Expectancy Over Time{title_suffix}",
        xaxis_title="Year", yaxis_title="Life Expectancy (years)",
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28),
    )
    return fig


def fig_convergence(bz_global, sigma, title_prefix=""):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[f"{title_prefix}Blue Zone Gap Over Global Average",
                                        f"{title_prefix}Sigma Convergence (Global LE Spread)"],
                        vertical_spacing=0.12)
    if not bz_global.empty:
        fig.add_trace(go.Scatter(
            x=bz_global["year"], y=bz_global["bz_gap_over_global"],
            fill="tozeroy", mode="lines", name="BZ advantage",
            line=dict(color="#2ecc71", width=2), fillcolor="rgba(46,204,113,0.2)",
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

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries", hist["iso_code"].nunique())
    c2.metric("Period", f"{yr_min}-{yr_max}")
    c3.metric("Observations", f"{len(hist):,}")
    if not bz_global.empty:
        final_gap = bz_global.iloc[-1]["bz_gap_over_global"]
        c4.metric("BZ Gap (final year)", f"+{final_gap:.1f} yr")

    # Historical trends
    st.markdown(f'<h3 class="section-hdr">Life Expectancy: Blue Zone Countries vs World ({period_label})</h3>',
                unsafe_allow_html=True)
    st.plotly_chart(fig_timelines(hist, proj, bz_isos, f" ({period_label})", show_proj=show_proj),
                    use_container_width=True)

    # Convergence
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_convergence(bz_global, sigma, f"{period_label}: "),
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
        "All real data -- no synthetic"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Blue Zone Countries**")
    for iso, name in sorted(BZ_NAMES.items()):
        st.sidebar.markdown(f"- {name}")

    # --- Main tabs ---
    tab_pre, tab_full, tab_compare = st.tabs([
        "Pre-COVID (1960-2019)",
        "Full Period (1960-2023)",
        "COVID Impact Comparison",
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

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;font-size:0.85rem'>"
        "Blue Zones Longevity Analysis | Real-world data from World Bank and WHO APIs | "
        "Pre-COVID and full-period analysis for honest comparison"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
