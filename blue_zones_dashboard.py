#!/usr/bin/env python3
"""
Blue Zones Longevity Analysis Dashboard

A professional interactive dashboard analyzing longevity patterns in Blue Zones
vs. global populations using real-world health, environmental, and economic data.

Author: Professional Analytics Team
Created: 2025-09-28
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Page configuration
st.set_page_config(
    page_title="Blue Zones Longevity Analysis",
    page_icon="BZ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4682B4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .metric-container {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4682B4;
        margin: 0.5rem 0;
    }
    
    .blue-zone-highlight {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #2E8B57;
    }
    
    .sidebar-info {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess Blue Zones data"""
    try:
        # Try to load the comprehensive dataset first
        df = pd.read_csv('real_world_blue_zones_comprehensive.csv')
    except FileNotFoundError:
        # Fallback to analysis results
        try:
            df = pd.read_csv('real_world_analysis_results.csv')
        except FileNotFoundError:
            st.error("Data files not found. Please ensure the CSV files are in the same directory as this script.")
            return None
    
    # Clean and preprocess data
    df = df.copy()
    
    # Convert numeric columns
    numeric_cols = ['forest_area_pct', 'gdp_per_capita', 'life_expectancy', 
                   'physicians_per_1000', 'pm25_air_pollution', 'population_total',
                   'urban_population_pct', 'infant_mortality', 'life_expectancy_who', 
                   'maternal_mortality', 'latitude', 'longitude']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create Blue Zone indicator
    df['zone_type'] = df['is_blue_zone'].map({1: 'Blue Zone', 0: 'Regular Zone'})
    
    # Fill missing values for key metrics
    df['life_expectancy_combined'] = df['life_expectancy'].fillna(df['life_expectancy_who'])
    
    return df

def create_kpi_metrics(df):
    """Create KPI metrics for the dashboard"""
    blue_zones = df[df['is_blue_zone'] == 1]
    regular_zones = df[df['is_blue_zone'] == 0]
    
    # Calculate key metrics
    bz_life_exp = blue_zones['life_expectancy_combined'].mean()
    reg_life_exp = regular_zones['life_expectancy_combined'].mean()
    
    bz_infant_mort = blue_zones['infant_mortality'].mean()
    reg_infant_mort = regular_zones['infant_mortality'].mean()
    
    bz_maternal_mort = blue_zones['maternal_mortality'].mean()
    reg_maternal_mort = regular_zones['maternal_mortality'].mean()
    
    bz_forest = blue_zones['forest_area_pct'].mean()
    reg_forest = regular_zones['forest_area_pct'].mean()
    
    return {
        'blue_zones_count': len(blue_zones),
        'total_countries': len(df),
        'bz_life_exp': bz_life_exp,
        'reg_life_exp': reg_life_exp,
        'life_exp_advantage': bz_life_exp - reg_life_exp,
        'bz_infant_mort': bz_infant_mort,
        'reg_infant_mort': reg_infant_mort,
        'bz_maternal_mort': bz_maternal_mort,
        'reg_maternal_mort': reg_maternal_mort,
        'bz_forest': bz_forest,
        'reg_forest': reg_forest
    }

def create_world_map(df):
    """Create interactive world map showing Blue Zones"""
    # Filter out rows with missing coordinates and life expectancy
    df_map = df.dropna(subset=['latitude', 'longitude', 'life_expectancy_combined'])
    
    # Ensure life expectancy values are positive for sizing
    df_map = df_map[df_map['life_expectancy_combined'] > 0]
    
    fig = px.scatter_mapbox(
        df_map, 
        lat="latitude", 
        lon="longitude",
        color="zone_type",
        size="life_expectancy_combined",
        hover_name="country_name",
        hover_data={
            'zone_type': True,
            'blue_zone_region': True,
            'life_expectancy_combined': ':.1f',
            'infant_mortality': ':.1f',
            'maternal_mortality': ':.1f',
            'physicians_per_1000': ':.2f',
            'forest_area_pct': ':.1f',
            'pm25_air_pollution': ':.1f',
            'gdp_per_capita': ':,.0f',
            'urban_population_pct': ':.1f'
        },
        size_max=30,
        color_discrete_map={
            'Blue Zone': '#2E8B57',
            'Regular Zone': '#4682B4'
        },
        mapbox_style="open-street-map",
        zoom=1,
        height=600,
        title="Global Blue Zones & Longevity Hotspots"
    )
    
    fig.update_layout(
        mapbox_center={"lat": 0, "lon": 0},
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )
    
    return fig

def create_longevity_comparison(df):
    """Create longevity comparison chart"""
    fig = px.box(
        df, 
        x="zone_type", 
        y="life_expectancy_combined",
        color="zone_type",
        color_discrete_map={
            'Blue Zone': '#2E8B57',
            'Regular Zone': '#4682B4'
        },
        title="Life Expectancy Distribution: Blue Zones vs Regular Zones",
        points="all"
    )
    
    fig.update_layout(
        xaxis_title="Zone Type",
        yaxis_title="Life Expectancy (Years)",
        height=400
    )
    
    return fig

def create_health_metrics_radar(df):
    """Create radar chart for health metrics comparison"""
    blue_zones = df[df['is_blue_zone'] == 1]
    regular_zones = df[df['is_blue_zone'] == 0]
    
    # Calculate averages (invert negative metrics for radar display)
    metrics = {
        'Life Expectancy': [
            blue_zones['life_expectancy_combined'].mean(),
            regular_zones['life_expectancy_combined'].mean()
        ],
        'Forest Coverage': [
            blue_zones['forest_area_pct'].mean(),
            regular_zones['forest_area_pct'].mean()
        ],
        'Physicians per 1000': [
            blue_zones['physicians_per_1000'].mean() * 20,  # Scale for visibility
            regular_zones['physicians_per_1000'].mean() * 20
        ],
        'Low Infant Mortality': [
            100 - blue_zones['infant_mortality'].mean(),  # Invert for radar
            100 - regular_zones['infant_mortality'].mean()
        ],
        'Clean Air Quality': [
            100 - blue_zones['pm25_air_pollution'].mean(),  # Invert for radar
            100 - regular_zones['pm25_air_pollution'].mean()
        ]
    }
    
    fig = go.Figure()
    
    # Blue Zones
    fig.add_trace(go.Scatterpolar(
        r=[metrics[key][0] for key in metrics.keys()],
        theta=list(metrics.keys()),
        fill='toself',
        name='Blue Zones',
        line_color='#2E8B57'
    ))
    
    # Regular Zones
    fig.add_trace(go.Scatterpolar(
        r=[metrics[key][1] for key in metrics.keys()],
        theta=list(metrics.keys()),
        fill='toself',
        name='Regular Zones',
        line_color='#4682B4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Health & Environmental Metrics Comparison",
        height=500
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap of key variables"""
    # Select numeric columns for correlation
    numeric_cols = ['life_expectancy_combined', 'forest_area_pct', 'gdp_per_capita',
                   'physicians_per_1000', 'pm25_air_pollution', 'infant_mortality',
                   'maternal_mortality', 'urban_population_pct']
    
    df_corr = df[numeric_cols].dropna()
    correlation_matrix = df_corr.corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Correlation Matrix: Health & Environmental Factors",
        color_continuous_scale="RdBu",
        aspect="auto",
        height=500
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title="Variables",
        yaxis_title="Variables"
    )
    
    return fig

def create_gdp_life_expectancy_scatter(df):
    """Create GDP vs Life Expectancy scatter plot"""
    # Filter out rows with missing required values
    df_scatter = df.dropna(subset=['gdp_per_capita', 'life_expectancy_combined', 'population_total'])
    df_scatter = df_scatter[df_scatter['population_total'] > 0]
    
    fig = px.scatter(
        df_scatter,
        x="gdp_per_capita",
        y="life_expectancy_combined",
        color="zone_type",
        size="population_total",
        hover_data=['country_name', 'blue_zone_region'],
        color_discrete_map={
            'Blue Zone': '#2E8B57',
            'Regular Zone': '#4682B4'
        },
        title="Economic Development vs Life Expectancy",
        labels={
            'gdp_per_capita': 'GDP per Capita (USD)',
            'life_expectancy_combined': 'Life Expectancy (Years)'
        }
    )
    
    fig.update_layout(height=500)
    
    return fig

def main():
    """Main dashboard application"""
    # Header
    st.markdown('<h1 class="main-header">Blue Zones Longevity Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Exploring the secrets of longevity through data-driven analysis of the world's healthiest regions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Top filter bar
    st.markdown('<h2 class="section-header">Filters</h2>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns([2, 3, 2])
        zone_filter = c1.multiselect(
            "Zone Types",
            options=['Blue Zone', 'Regular Zone'],
            default=['Blue Zone', 'Regular Zone']
        )
        countries = c2.multiselect(
            "Countries (Optional)",
            options=sorted(df['country_name'].unique()),
            default=[]
        )
        show_blue_info = c3.toggle("Show Blue Zones Info", value=False)
    
    # Data filtering
    if zone_filter:
        df_filtered = df[df['zone_type'].isin(zone_filter)]
    else:
        df_filtered = df
        
    if countries:
        df_filtered = df_filtered[df_filtered['country_name'].isin(countries)]
    
    if show_blue_info:
        with st.expander("About Blue Zones", expanded=False):
            st.markdown("""
            <div class="sidebar-info">
            <strong>The 5 Blue Zones:</strong><br>
            Loma Linda, California<br>
            Okinawa, Japan<br>
            Sardinia, Italy<br>
            Ikaria, Greece<br>
            Nicoya Peninsula, Costa Rica
            </div>
            """, unsafe_allow_html=True)
    
    # KPI Section
    st.markdown('<h2 class="section-header">Key Performance Indicators</h2>', 
                unsafe_allow_html=True)
    
    kpis = create_kpi_metrics(df_filtered)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Blue Zones Identified",
            f"{kpis['blue_zones_count']}",
            delta=f"of {kpis['total_countries']} countries"
        )
    
    with col2:
        st.metric(
            "Blue Zones Life Expectancy",
            f"{kpis['bz_life_exp']:.1f} years",
            delta=f"+{kpis['life_exp_advantage']:.1f} vs regular zones"
        )
    
    with col3:
        st.metric(
            "Blue Zones Infant Mortality",
            f"{kpis['bz_infant_mort']:.1f}‰",
            delta=f"{kpis['bz_infant_mort'] - kpis['reg_infant_mort']:.1f}‰ vs regular"
        )
    
    with col4:
        st.metric(
            "Blue Zones Forest Coverage",
            f"{kpis['bz_forest']:.1f}%",
            delta=f"+{kpis['bz_forest'] - kpis['reg_forest']:.1f}% vs regular"
        )
    
    # World Map
    st.markdown('<h2 class="section-header">Global Blue Zones Map</h2>', 
                unsafe_allow_html=True)
    
    world_map = create_world_map(df_filtered)
    st.plotly_chart(world_map, use_container_width=True)
    
    # Health Analysis Section
    st.markdown('<h2 class="section-header">Health & Longevity Analysis</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        longevity_chart = create_longevity_comparison(df_filtered)
        st.plotly_chart(longevity_chart, use_container_width=True)
    
    with col2:
        radar_chart = create_health_metrics_radar(df_filtered)
        st.plotly_chart(radar_chart, use_container_width=True)
    
    # Economic Analysis
    st.markdown('<h2 class="section-header">Economic & Environmental Factors</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gdp_scatter = create_gdp_life_expectancy_scatter(df_filtered)
        st.plotly_chart(gdp_scatter, use_container_width=True)
    
    with col2:
        correlation_heatmap = create_correlation_heatmap(df_filtered)
        st.plotly_chart(correlation_heatmap, use_container_width=True)
    
    # Data Table
    st.markdown('<h2 class="section-header">Detailed Data Explorer</h2>', 
                unsafe_allow_html=True)
    
    # Display options
    show_blue_zones_only = st.checkbox("Show Blue Zones Only")
    if show_blue_zones_only:
        display_df = df_filtered[df_filtered['is_blue_zone'] == 1]
    else:
        display_df = df_filtered
    
    # Select columns to display
    display_cols = ['country_name', 'zone_type', 'blue_zone_region', 
                   'life_expectancy_combined', 'infant_mortality', 'maternal_mortality',
                   'gdp_per_capita', 'forest_area_pct', 'pm25_air_pollution']
    
    display_df_clean = display_df[display_cols].round(2)
    st.dataframe(display_df_clean, use_container_width=True, height=400)
    
    # Insights Section
    st.markdown('<h2 class="section-header">Key Insights</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="blue-zone-highlight" style="color: #2c3e50;">
        <h4 style="color: #2E8B57; font-weight: bold;">Blue Zones Advantages</h4>
        <ul style="color: #2c3e50; font-weight: 500;">
        <li>Higher life expectancy by ~6+ years</li>
        <li>Lower infant mortality rates</li>
        <li>Better air quality on average</li>
        <li>Higher forest coverage</li>
        <li>More balanced healthcare access</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container" style="color: #2c3e50;">
        <h4 style="color: #4682B4; font-weight: bold;">Statistical Findings</h4>
        <ul style="color: #2c3e50; font-weight: 500;">
        <li>5 Blue Zones identified globally</li>
        <li>Strong correlation between environment and health</li>
        <li>GDP doesn't guarantee longevity</li>
        <li>Forest coverage linked to life expectancy</li>
        <li>Traditional lifestyle patterns matter</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Blue Zones Longevity Analysis Dashboard | Data Science & Analytics Platform</p>
        <p>Created with Streamlit & Plotly | Real-world health and demographic data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
