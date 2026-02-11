#!/usr/bin/env python3
"""
Blue Zones Longevity Analysis Dashboard

Interactive Streamlit dashboard showing historical trends (1960-2023),
convergence analysis, country deep dives, and future projections.
Built on real World Bank, WHO, and UN data.
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
    .sub-header {font-size:1rem;text-align:center;color:#555;margin-bottom:2rem}
    .section-hdr {font-size:1.4rem;font-weight:600;color:#16213e;border-bottom:2px solid #0f3460;padding-bottom:0.3rem;margin-top:2rem}
    .kpi-card {background:#f8f9fa;border-radius:8px;padding:1rem;border-left:4px solid #0f3460;margin:0.3rem 0}
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
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_projections():
    path = os.path.join(SCRIPT_DIR, "data", "projections", "un_life_expectancy_projections.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_analysis(name):
    path = os.path.join(SCRIPT_DIR, "outputs", "analysis", f"{name}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_cross_section():
    path = os.path.join(SCRIPT_DIR, "real_world_blue_zones_comprehensive.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def fig_historical_timelines(hist, proj, selected_isos):
    """Life expectancy 1960-2100 for selected countries."""
    fig = go.Figure()

    # Global average
    global_avg = hist.groupby("year")["life_expectancy"].mean().dropna()
    fig.add_trace(go.Scatter(
        x=global_avg.index, y=global_avg.values,
        mode="lines", name="Global Average",
        line=dict(color="#95a5a6", width=3, dash="dot"),
    ))

    for iso in selected_isos:
        color = BZ_COLORS.get(iso, "#333")
        label = BZ_NAMES.get(iso, iso)

        # Historical
        ch = hist[hist["iso_code"] == iso].sort_values("year")
        le = ch[["year", "life_expectancy"]].dropna()
        if not le.empty:
            fig.add_trace(go.Scatter(
                x=le["year"], y=le["life_expectancy"],
                mode="lines", name=label,
                line=dict(color=color, width=2.5),
            ))

        # Projections
        if not proj.empty:
            cp = proj[proj["iso_code"] == iso].sort_values("year")
            if not cp.empty and "le_medium" in cp.columns:
                fig.add_trace(go.Scatter(
                    x=cp["year"], y=cp["le_medium"],
                    mode="lines", name=f"{label} (projected)",
                    line=dict(color=color, width=2, dash="dash"),
                    showlegend=False,
                ))
                if "le_high" in cp.columns and "le_low" in cp.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([cp["year"], cp["year"][::-1]]),
                        y=pd.concat([cp["le_high"], cp["le_low"][::-1]]),
                        fill="toself", fillcolor=color, opacity=0.08,
                        line=dict(width=0), showlegend=False, name="",
                    ))

    fig.add_vline(x=2023, line_dash="dot", line_color="grey", opacity=0.5)
    fig.update_layout(
        title="Life Expectancy Over Time (1960-2100)",
        xaxis_title="Year", yaxis_title="Life Expectancy at Birth (years)",
        height=520, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def fig_convergence(bz_global, sigma):
    """Two-panel convergence chart."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Blue Zone Countries vs Global Average: Gap",
                                        "Sigma Convergence: Global Spread of Life Expectancy"],
                        vertical_spacing=0.12)

    if not bz_global.empty:
        gap = bz_global["bz_gap_over_global"]
        fig.add_trace(go.Scatter(
            x=bz_global["year"], y=gap, fill="tozeroy",
            mode="lines", name="BZ advantage (years)",
            line=dict(color="#2ecc71", width=2),
            fillcolor="rgba(46,204,113,0.2)",
        ), row=1, col=1)

    if not sigma.empty:
        fig.add_trace(go.Scatter(
            x=sigma["year"], y=sigma["le_std"],
            mode="lines", name="Std Dev across countries",
            line=dict(color="#8e44ad", width=2),
            fill="tozeroy", fillcolor="rgba(142,68,173,0.12)",
        ), row=2, col=1)

    fig.update_layout(height=600, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.15))
    fig.update_yaxes(title_text="Gap (years)", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev (years)", row=2, col=1)
    return fig


def fig_decade_bars(decades):
    """Improvement rates by decade."""
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
        title="Life Expectancy Gain by Decade",
        xaxis_title="Decade", yaxis_title="Average LE Gain (years)",
        barmode="group", height=420, template="plotly_white",
    )
    return fig


def fig_ranking(hist):
    """Blue Zone country percentile ranking over time."""
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
            marker=dict(size=4),
        ))
    fig.add_hline(y=50, line_dash="dash", line_color="grey", opacity=0.3)
    fig.update_layout(
        title="Blue Zone Countries: Global LE Percentile Ranking",
        xaxis_title="Year", yaxis_title="Percentile (higher = longer-lived)",
        yaxis_range=[0, 105], height=450, template="plotly_white",
    )
    return fig


def fig_country_detail(hist, proj, iso):
    """Two-panel deep dive for a single country."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        subplot_titles=[f"{BZ_NAMES.get(iso, iso)}: Life Expectancy",
                                        f"{BZ_NAMES.get(iso, iso)}: GDP per Capita"],
                        vertical_spacing=0.15)
    ch = hist[hist["iso_code"] == iso].sort_values("year")
    le = ch[["year", "life_expectancy"]].dropna()
    if not le.empty:
        fig.add_trace(go.Scatter(
            x=le["year"], y=le["life_expectancy"], mode="lines",
            name="Historical LE", line=dict(color=BZ_COLORS.get(iso, "#333"), width=2.5),
        ), row=1, col=1)
    if not proj.empty:
        cp = proj[proj["iso_code"] == iso].sort_values("year")
        if not cp.empty and "le_medium" in cp.columns:
            fig.add_trace(go.Scatter(
                x=cp["year"], y=cp["le_medium"], mode="lines",
                name="Projected (medium)", line=dict(color=BZ_COLORS.get(iso, "#333"), width=2, dash="dash"),
            ), row=1, col=1)
            if "le_high" in cp.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([cp["year"], cp["year"][::-1]]),
                    y=pd.concat([cp["le_high"], cp["le_low"][::-1]]),
                    fill="toself", fillcolor=BZ_COLORS.get(iso, "#333"),
                    opacity=0.08, line=dict(width=0), showlegend=False, name="",
                ), row=1, col=1)
    gdp = ch[["year", "gdp_per_capita"]].dropna()
    if not gdp.empty:
        fig.add_trace(go.Scatter(
            x=gdp["year"], y=gdp["gdp_per_capita"], mode="lines",
            name="GDP per capita", line=dict(color="#f39c12", width=2),
            fill="tozeroy", fillcolor="rgba(243,156,18,0.1)",
        ), row=2, col=1)
    fig.update_layout(height=600, template="plotly_white")
    fig.update_yaxes(title_text="Life Expectancy (years)", row=1, col=1)
    fig.update_yaxes(title_text="GDP per Capita (USD)", row=2, col=1)
    return fig


def fig_world_map(cross):
    """World map from cross-sectional data."""
    if cross.empty:
        return go.Figure()
    df_map = cross.dropna(subset=["latitude", "longitude"]).copy()
    df_map["zone_type"] = df_map["is_blue_zone"].map({1: "Blue Zone", 0: "Regular"})
    le_col = "life_expectancy"
    if le_col not in df_map.columns or df_map[le_col].isna().all():
        for alt in ["life_expectancy_who"]:
            if alt in df_map.columns:
                df_map[le_col] = df_map[le_col].fillna(df_map[alt])
    df_map = df_map.dropna(subset=[le_col])
    fig = px.scatter_geo(
        df_map, lat="latitude", lon="longitude", color="zone_type",
        hover_name="country_name", size=le_col, size_max=25,
        color_discrete_map={"Blue Zone": "#2E8B57", "Regular": "#4682B4"},
        projection="natural earth",
        title="Global Life Expectancy (Most Recent Year)",
    )
    fig.update_layout(height=480, template="plotly_white",
                      geo=dict(showframe=False, showcoastlines=True, coastlinecolor="#ccc"))
    return fig


def fig_heatmap(hist):
    """Life expectancy heatmap: countries x years."""
    le = hist[["iso_code", "year", "life_expectancy", "country_name", "is_blue_zone"]].dropna(subset=["life_expectancy"])
    pivot = le.pivot_table(index="iso_code", columns="year", values="life_expectancy", aggfunc="first")
    last = pivot.columns.max()
    pivot = pivot.sort_values(by=last, ascending=True, na_position="first")
    iso_to_name = dict(zip(hist["iso_code"], hist["country_name"]))
    labels = [f"{iso_to_name.get(i, i)} *" if i in BZ_COLORS else iso_to_name.get(i, i)
              for i in pivot.index]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=labels,
        colorscale="RdYlGn", zmin=35, zmax=85,
        colorbar_title="Life Expectancy",
    ))
    fig.update_layout(
        title="Global Life Expectancy Heatmap (1960-2023)  (* = Blue Zone country)",
        height=max(500, len(labels) * 12), template="plotly_white",
        xaxis_title="Year",
    )
    return fig


def fig_correlation(hist):
    """Correlation heatmap from most recent year of historical data."""
    recent = hist.sort_values("year").groupby("iso_code").last().reset_index()
    cols = ["life_expectancy", "gdp_per_capita", "physicians_per_1000",
            "urban_population_pct", "pm25_air_pollution", "death_rate",
            "health_expenditure_pc"]
    avail = [c for c in cols if c in recent.columns]
    corr = recent[avail].dropna().corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation Matrix (Most Recent Year)",
                    text_auto=".2f")
    fig.update_layout(height=500, template="plotly_white")
    return fig


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main():
    # Load all datasets
    hist = load_historical()
    proj = load_projections()
    bz_global = load_analysis("blue_zone_vs_global")
    sigma = load_analysis("sigma_convergence")
    beta = load_analysis("beta_convergence")
    decades = load_analysis("decade_improvements")
    cross = load_cross_section()

    if hist.empty:
        st.error("Historical data not found. Run historical_data_collector.py first.")
        return

    # --- Sidebar ---
    st.sidebar.title("Blue Zones Analysis")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Section", [
        "Overview",
        "Historical Trends",
        "Convergence Analysis",
        "Country Deep Dives",
        "Projections",
        "Data Explorer",
    ])

    # Country filter for timelines
    bz_isos = sorted(BZ_COLORS.keys())
    all_isos = sorted(hist["iso_code"].unique())

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Sources**")
    st.sidebar.markdown(
        "World Bank API (1960-2023)\n\n"
        "WHO Global Health Observatory\n\n"
        "All real data -- no synthetic"
    )

    # --- Header ---
    st.markdown('<h1 class="main-header">Blue Zones Longevity Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Historical Trends, Convergence, and Future Projections -- Built on Real Data</p>',
                unsafe_allow_html=True)

    # ===== OVERVIEW =====
    if page == "Overview":
        # KPIs
        n_countries = hist["iso_code"].nunique()
        yr_min, yr_max = int(hist["year"].min()), int(hist["year"].max())
        n_rows = len(hist)

        recent = bz_global[bz_global["year"] >= 2019].iloc[-1] if not bz_global.empty and len(bz_global[bz_global["year"] >= 2019]) > 0 else None
        early = bz_global[bz_global["year"] <= 1965].iloc[0] if not bz_global.empty and len(bz_global[bz_global["year"] <= 1965]) > 0 else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Countries", n_countries)
        c2.metric("Years Covered", f"{yr_min} - {yr_max}")
        c3.metric("Data Points", f"{n_rows:,}")
        if recent is not None:
            c4.metric("BZ Gap (current)", f"+{recent['bz_gap_over_global']:.1f} years")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if not cross.empty:
                st.plotly_chart(fig_world_map(cross), use_container_width=True)
            else:
                st.info("Cross-sectional data file not found.")
        with col2:
            st.plotly_chart(fig_historical_timelines(hist, proj, bz_isos), use_container_width=True)

        # Key findings table
        if not bz_global.empty:
            st.markdown('<h3 class="section-hdr">Key Findings at a Glance</h3>', unsafe_allow_html=True)
            key_years = bz_global[bz_global["year"].isin([1960, 1980, 2000, 2020])]
            if not key_years.empty:
                display = key_years[["year", "blue_zone_mean", "global_mean", "bz_gap_over_global", "n_countries"]].copy()
                display.columns = ["Year", "BZ Country Avg", "Global Avg", "Gap (years)", "Countries"]
                display = display.round(1)
                st.dataframe(display, use_container_width=True, hide_index=True)

    # ===== HISTORICAL TRENDS =====
    elif page == "Historical Trends":
        st.markdown('<h3 class="section-hdr">Life Expectancy Over Time</h3>', unsafe_allow_html=True)

        selected = st.multiselect(
            "Select countries to display",
            options=all_isos,
            default=bz_isos,
            format_func=lambda x: BZ_NAMES.get(x, hist[hist["iso_code"] == x]["country_name"].iloc[0]
                                                if len(hist[hist["iso_code"] == x]) > 0 else x),
        )
        if selected:
            st.plotly_chart(fig_historical_timelines(hist, proj, selected), use_container_width=True)

        st.markdown('<h3 class="section-hdr">Global Heatmap</h3>', unsafe_allow_html=True)
        st.plotly_chart(fig_heatmap(hist), use_container_width=True)

        st.markdown('<h3 class="section-hdr">Global Ranking Over Time</h3>', unsafe_allow_html=True)
        st.plotly_chart(fig_ranking(hist), use_container_width=True)

    # ===== CONVERGENCE =====
    elif page == "Convergence Analysis":
        st.markdown('<h3 class="section-hdr">Is the World Catching Up?</h3>', unsafe_allow_html=True)
        st.markdown(
            "The Blue Zone country advantage has been **shrinking** since 1960. "
            "The gap narrowed from ~10 years to ~6 years as the rest of the world improved faster."
        )
        st.plotly_chart(fig_convergence(bz_global, sigma), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h3 class="section-hdr">Improvement by Decade</h3>', unsafe_allow_html=True)
            st.plotly_chart(fig_decade_bars(decades), use_container_width=True)
        with col2:
            st.markdown('<h3 class="section-hdr">Beta Convergence</h3>', unsafe_allow_html=True)
            if not beta.empty:
                st.dataframe(beta[["decade", "avg_gain_years", "bz_avg_gain",
                                   "non_bz_avg_gain", "convergence"]].rename(columns={
                    "decade": "Decade", "avg_gain_years": "Global Gain",
                    "bz_avg_gain": "BZ Gain", "non_bz_avg_gain": "Non-BZ Gain",
                    "convergence": "Convergence?",
                }).round(1), use_container_width=True, hide_index=True)

    # ===== COUNTRY DEEP DIVES =====
    elif page == "Country Deep Dives":
        st.markdown('<h3 class="section-hdr">Blue Zone Country Profiles</h3>', unsafe_allow_html=True)
        iso_pick = st.selectbox("Select country", bz_isos,
                                format_func=lambda x: BZ_NAMES.get(x, x))
        st.plotly_chart(fig_country_detail(hist, proj, iso_pick), use_container_width=True)

        # Stats table
        ch = hist[hist["iso_code"] == iso_pick]
        le = ch["life_expectancy"].dropna()
        if not le.empty:
            st.markdown(f"**Life expectancy range:** {le.min():.1f} - {le.max():.1f} years  "
                        f"(total gain: {le.max() - le.min():.1f} years)")
        gdp = ch["gdp_per_capita"].dropna()
        if not gdp.empty:
            st.markdown(f"**GDP per capita range:** ${gdp.min():,.0f} - ${gdp.max():,.0f}")

    # ===== PROJECTIONS =====
    elif page == "Projections":
        st.markdown('<h3 class="section-hdr">Future Projections (to 2100)</h3>', unsafe_allow_html=True)
        if proj.empty:
            st.warning("No projection data available. Run un_projections_collector.py.")
        else:
            source = proj["projection_source"].iloc[0] if "projection_source" in proj.columns else "Unknown"
            st.markdown(f"**Source:** {source}")
            st.plotly_chart(fig_historical_timelines(hist, proj, bz_isos), use_container_width=True)

            st.markdown('<h3 class="section-hdr">2050 Projections</h3>', unsafe_allow_html=True)
            p2050 = proj[proj["year"] == 2050][["country_name", "le_medium", "le_high", "le_low"]].copy()
            if "is_blue_zone" in proj.columns:
                p2050_bz = proj[(proj["year"] == 2050) & (proj["is_blue_zone"] == 1)]
                p2050 = p2050_bz[["country_name", "le_medium", "le_high", "le_low"]].copy()
            p2050.columns = ["Country", "Medium", "High", "Low"]
            if not p2050.empty:
                st.dataframe(p2050.round(1), use_container_width=True, hide_index=True)

    # ===== DATA EXPLORER =====
    elif page == "Data Explorer":
        st.markdown('<h3 class="section-hdr">Raw Data Explorer</h3>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Historical Panel", "Projections", "Correlations"])

        with tab1:
            year_range = st.slider("Year range", int(hist["year"].min()), int(hist["year"].max()),
                                   (1960, 2023))
            filt = hist[(hist["year"] >= year_range[0]) & (hist["year"] <= year_range[1])]
            bz_only = st.checkbox("Blue Zone countries only")
            if bz_only:
                filt = filt[filt["is_blue_zone"] == 1]
            st.dataframe(filt, use_container_width=True, height=500)
            st.download_button("Download filtered data", filt.to_csv(index=False),
                               "historical_filtered.csv", "text/csv")

        with tab2:
            if not proj.empty:
                st.dataframe(proj, use_container_width=True, height=500)
            else:
                st.info("No projection data available.")

        with tab3:
            st.plotly_chart(fig_correlation(hist), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;font-size:0.85rem'>"
        "Blue Zones Longevity Analysis | Real-world data from World Bank and WHO APIs"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
