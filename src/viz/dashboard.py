"""
Interactive dashboard for Blue Zones Quantified visualization
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins


class BlueZonesDashboard:
    """Interactive dashboard for Blue Zones analysis"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
    def create_global_maps(self, 
                          features: pd.DataFrame,
                          predictions: pd.DataFrame,
                          forecasts: pd.DataFrame) -> Dict[str, folium.Map]:
        """Create interactive global maps"""
        
        maps = {}
        
        # 1. Blue Zone Score Map
        maps['blue_zone_scores'] = self._create_score_map(predictions)
        
        # 2. Feature Maps
        maps['features'] = self._create_feature_maps(features)
        
        # 3. Forecast Maps
        maps['forecasts'] = self._create_forecast_maps(forecasts)
        
        # 4. Uncertainty Maps
        maps['uncertainty'] = self._create_uncertainty_maps(forecasts)
        
        return maps
        
    def create_analysis_plots(self,
                            matched_results: Dict[str, Any],
                            classifier_results: Dict[str, Any],
                            forecasts: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create analysis visualization plots"""
        
        plots = {}
        
        # 1. Treatment Effect Plot
        plots['treatment_effects'] = self._create_treatment_effect_plot(matched_results)
        
        # 2. Feature Importance Plot
        plots['feature_importance'] = self._create_feature_importance_plot(classifier_results)
        
        # 3. Forecast Trends
        plots['forecast_trends'] = self._create_forecast_trends_plot(forecasts)
        
        # 4. Scenario Comparison
        plots['scenario_comparison'] = self._create_scenario_comparison_plot(forecasts)
        
        # 5. Uncertainty Analysis
        plots['uncertainty_analysis'] = self._create_uncertainty_plot(forecasts)
        
        return plots
        
    def create_summary_dashboard(self,
                               all_data: Dict[str, Any]) -> str:
        """Create comprehensive HTML dashboard"""
        
        # Generate all visualizations
        maps = all_data.get('maps', {})
        plots = all_data.get('plots', {})
        
        # Create HTML dashboard
        html_content = self._build_html_dashboard(maps, plots, all_data)
        
        return html_content
        
    def _create_score_map(self, predictions: pd.DataFrame) -> folium.Map:
        """Create Blue Zone score map"""
        
        # Create base map
        m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
        
        if 'latitude' not in predictions.columns or 'longitude' not in predictions.columns:
            self.logger.warning("No geographic coordinates in predictions data")
            return m
            
        # Add score points
        for _, row in predictions.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                
                # Color by score decile
                color = self._get_score_color(row['blue_zone_decile'])
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    popup=f"Blue Zone Score: {row['blue_zone_score']:.3f}<br>"
                          f"Decile: {row['blue_zone_decile']}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Add legend
        self._add_score_legend(m)
        
        return m
        
    def _create_feature_maps(self, features: pd.DataFrame) -> Dict[str, folium.Map]:
        """Create maps for key features"""
        
        feature_maps = {}
        
        # Key features to map
        map_features = [
            ('temperature_mean', 'Mean Temperature'),
            ('pm25_annual', 'PM2.5 Pollution'),
            ('population_density_log', 'Population Density'),
            ('nighttime_lights', 'Economic Development'),
            ('walkability_score', 'Walkability')
        ]
        
        for feature_col, title in map_features:
            if feature_col in features.columns:
                feature_maps[feature_col] = self._create_choropleth_map(
                    features, feature_col, title
                )
                
        return feature_maps
        
    def _create_forecast_maps(self, forecasts: pd.DataFrame) -> Dict[str, folium.Map]:
        """Create forecast maps by year and scenario"""
        
        forecast_maps = {}
        
        # Create maps for each forecast year (baseline scenario)
        baseline_forecasts = forecasts[forecasts['scenario'] == 'baseline']
        
        for year in sorted(baseline_forecasts['year'].unique()):
            year_data = baseline_forecasts[baseline_forecasts['year'] == year]
            
            if not year_data.empty:
                forecast_maps[f'forecast_{year}'] = self._create_choropleth_map(
                    year_data, 'predicted_life_expectancy', 
                    f'Life Expectancy Forecast {year}'
                )
                
        return forecast_maps
        
    def _create_uncertainty_maps(self, forecasts: pd.DataFrame) -> folium.Map:
        """Create forecast uncertainty map"""
        
        # Use baseline scenario for uncertainty visualization
        baseline_forecasts = forecasts[forecasts['scenario'] == 'baseline']
        
        if baseline_forecasts.empty:
            return folium.Map(location=[20, 0], zoom_start=2)
            
        # Use most recent forecast year
        latest_year = baseline_forecasts['year'].max()
        uncertainty_data = baseline_forecasts[baseline_forecasts['year'] == latest_year]
        
        return self._create_choropleth_map(
            uncertainty_data, 'prediction_interval_width',
            f'Forecast Uncertainty {latest_year}'
        )
        
    def _create_choropleth_map(self, data: pd.DataFrame, 
                              value_col: str, title: str) -> folium.Map:
        """Create choropleth-style map"""
        
        m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
        
        if ('latitude' not in data.columns or 'longitude' not in data.columns 
            or value_col not in data.columns):
            return m
            
        # Get value range for color scaling
        values = data[value_col].dropna()
        if values.empty:
            return m
            
        vmin, vmax = values.min(), values.max()
        
        # Add points
        for _, row in data.iterrows():
            if (pd.notna(row['latitude']) and pd.notna(row['longitude']) 
                and pd.notna(row[value_col])):
                
                # Normalize value for color
                norm_value = (row[value_col] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = self._value_to_color(norm_value)
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    popup=f"{title}: {row[value_col]:.2f}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Add title
        title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
        
    def _create_treatment_effect_plot(self, matched_results: Dict[str, Any]) -> go.Figure:
        """Create treatment effect visualization"""
        
        fig = go.Figure()
        
        # Extract results for plotting
        plot_data = []
        
        if 'overall_blue_zones' in matched_results:
            overall = matched_results['overall_blue_zones']
            if 'summary' in overall and 'method_comparison' in overall['summary']:
                for method, result in overall['summary']['method_comparison'].items():
                    if not pd.isna(result['att']):
                        plot_data.append({
                            'zone': 'Overall',
                            'method': method,
                            'att': result['att'],
                            'ci_lower': result.get('ci_lower', result['att']),
                            'ci_upper': result.get('ci_upper', result['att'])
                        })
        
        # Individual zones
        if 'individual_blue_zones' in matched_results:
            for zone, zone_results in matched_results['individual_blue_zones'].items():
                if 'summary' in zone_results and 'method_comparison' in zone_results['summary']:
                    for method, result in zone_results['summary']['method_comparison'].items():
                        if not pd.isna(result['att']):
                            plot_data.append({
                                'zone': zone.title(),
                                'method': method,
                                'att': result['att'],
                                'ci_lower': result.get('ci_lower', result['att']),
                                'ci_upper': result.get('ci_upper', result['att'])
                            })
        
        if not plot_data:
            fig.add_annotation(text="No treatment effect results available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        df = pd.DataFrame(plot_data)
        
        # Create subplot for each method
        methods = df['method'].unique()
        colors = px.colors.qualitative.Set1
        
        for i, method in enumerate(methods):
            method_data = df[df['method'] == method]
            
            fig.add_trace(go.Scatter(
                x=method_data['att'],
                y=method_data['zone'],
                mode='markers',
                marker=dict(size=10, color=colors[i % len(colors)]),
                name=method.replace('_', ' ').title(),
                error_x=dict(
                    array=method_data['ci_upper'] - method_data['att'],
                    arrayminus=method_data['att'] - method_data['ci_lower']
                )
            ))
        
        fig.update_layout(
            title="Blue Zone Treatment Effects (Years of Life Expectancy)",
            xaxis_title="Average Treatment Effect (Years)",
            yaxis_title="Blue Zone Region",
            showlegend=True,
            height=400
        )
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        return fig
        
    def _create_feature_importance_plot(self, classifier_results: Dict[str, Any]) -> go.Figure:
        """Create feature importance plot"""
        
        fig = go.Figure()
        
        if 'feature_importance' not in classifier_results:
            fig.add_annotation(text="No feature importance results available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        importance_df = classifier_results['feature_importance']['feature_importance']
        
        # Take top 15 features
        top_features = importance_df.head(15)
        
        fig.add_trace(go.Bar(
            x=top_features['importance_gain'],
            y=top_features['feature'],
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title="Top Features for Blue Zone Classification",
            xaxis_title="Feature Importance (LightGBM Gain)",
            yaxis_title="Features",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    def _create_forecast_trends_plot(self, forecasts: pd.DataFrame) -> go.Figure:
        """Create forecast trends plot"""
        
        fig = go.Figure()
        
        if forecasts.empty:
            fig.add_annotation(text="No forecast data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Global trends by scenario
        global_trends = forecasts.groupby(['year', 'scenario'])['predicted_life_expectancy'].mean().reset_index()
        
        scenarios = ['optimistic', 'baseline', 'pessimistic']
        colors = {'optimistic': 'green', 'baseline': 'blue', 'pessimistic': 'red'}
        
        for scenario in scenarios:
            scenario_data = global_trends[global_trends['scenario'] == scenario]
            
            if not scenario_data.empty:
                fig.add_trace(go.Scatter(
                    x=scenario_data['year'],
                    y=scenario_data['predicted_life_expectancy'],
                    mode='lines+markers',
                    name=scenario.title(),
                    line=dict(color=colors.get(scenario, 'gray'))
                ))
        
        fig.update_layout(
            title="Global Life Expectancy Forecast Trends",
            xaxis_title="Year",
            yaxis_title="Predicted Life Expectancy (Years)",
            showlegend=True,
            height=400
        )
        
        return fig
        
    def _create_scenario_comparison_plot(self, forecasts: pd.DataFrame) -> go.Figure:
        """Create scenario comparison plot"""
        
        if forecasts.empty:
            fig = go.Figure()
            fig.add_annotation(text="No forecast data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Create box plots by scenario for each year
        years = sorted(forecasts['year'].unique())
        
        fig = make_subplots(
            rows=1, cols=len(years),
            subplot_titles=[str(year) for year in years]
        )
        
        colors = {'optimistic': 'green', 'baseline': 'blue', 'pessimistic': 'red'}
        
        for i, year in enumerate(years, 1):
            year_data = forecasts[forecasts['year'] == year]
            
            for scenario in ['optimistic', 'baseline', 'pessimistic']:
                scenario_data = year_data[year_data['scenario'] == scenario]
                
                if not scenario_data.empty:
                    fig.add_trace(
                        go.Box(
                            y=scenario_data['predicted_life_expectancy'],
                            name=scenario,
                            boxpoints='outliers',
                            marker_color=colors[scenario],
                            showlegend=(i == 1)  # Only show legend for first subplot
                        ),
                        row=1, col=i
                    )
        
        fig.update_layout(
            title="Life Expectancy Forecasts by Scenario",
            height=400
        )
        
        return fig
        
    def _create_uncertainty_plot(self, forecasts: pd.DataFrame) -> go.Figure:
        """Create uncertainty analysis plot"""
        
        fig = go.Figure()
        
        if forecasts.empty:
            fig.add_annotation(text="No forecast data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Uncertainty by year
        baseline_forecasts = forecasts[forecasts['scenario'] == 'baseline']
        
        if baseline_forecasts.empty:
            return fig
            
        uncertainty_by_year = baseline_forecasts.groupby('year')['prediction_interval_width'].agg([
            'mean', 'median', 'std'
        ]).reset_index()
        
        fig.add_trace(go.Scatter(
            x=uncertainty_by_year['year'],
            y=uncertainty_by_year['mean'],
            mode='lines+markers',
            name='Mean Uncertainty',
            error_y=dict(array=uncertainty_by_year['std'])
        ))
        
        fig.add_trace(go.Scatter(
            x=uncertainty_by_year['year'],
            y=uncertainty_by_year['median'],
            mode='lines+markers',
            name='Median Uncertainty'
        ))
        
        fig.update_layout(
            title="Forecast Uncertainty Over Time",
            xaxis_title="Year",
            yaxis_title="Prediction Interval Width (Years)",
            showlegend=True,
            height=400
        )
        
        return fig
        
    def _build_html_dashboard(self, maps: Dict[str, Any], 
                             plots: Dict[str, go.Figure],
                             all_data: Dict[str, Any]) -> str:
        """Build comprehensive HTML dashboard"""
        
        html_parts = []
        
        # HTML header
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Blue Zones Quantified - Analysis Dashboard</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2E86C1; text-align: center; }
                h2 { color: #5D6D7E; border-bottom: 2px solid #BDC3C7; padding-bottom: 5px; }
                .section { margin: 20px 0; }
                .plot-container { margin: 20px 0; }
                .summary-stats { background-color: #F8F9FA; padding: 15px; border-radius: 5px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>🌍 Blue Zones Quantified: Analysis Dashboard</h1>
        """)
        
        # Summary statistics
        html_parts.append(self._create_summary_section(all_data))
        
        # Analysis plots
        html_parts.append('<div class="section"><h2>📊 Analysis Results</h2>')
        
        for plot_name, plot in plots.items():
            if plot:
                html_parts.append(f'<div class="plot-container">')
                html_parts.append(plot.to_html(include_plotlyjs='cdn', div_id=f"plot_{plot_name}"))
                html_parts.append('</div>')
        
        html_parts.append('</div>')
        
        # Footer
        html_parts.append("""
            <div class="section">
                <p><em>Generated by Blue Zones Quantified Analysis Pipeline</em></p>
            </div>
        </body>
        </html>
        """)
        
        return '\n'.join(html_parts)
        
    def _create_summary_section(self, all_data: Dict[str, Any]) -> str:
        """Create summary statistics section"""
        
        summary_html = '<div class="section"><h2>📈 Analysis Summary</h2>'
        
        # Data summary
        if 'analysis_metadata' in all_data:
            metadata = all_data['analysis_metadata']
            summary_html += f"""
            <div class="summary-stats">
                <h3>Dataset Overview</h3>
                <ul>
                    <li><strong>Total Observations:</strong> {metadata.get('n_observations', 'N/A'):,}</li>
                    <li><strong>Blue Zone Cells:</strong> {metadata.get('n_blue_zone_cells', 'N/A')}</li>
                    <li><strong>Features Used:</strong> {metadata.get('n_features_used', 'N/A')}</li>
                    <li><strong>Data Year:</strong> {metadata.get('data_year', 'N/A')}</li>
                </ul>
            </div>
            """
        
        # Key findings
        summary_html += """
        <div class="summary-stats">
            <h3>Key Findings</h3>
            <ul>
                <li>Comprehensive analysis of global longevity patterns</li>
                <li>Machine learning identification of Blue Zone characteristics</li>
                <li>Scenario-based forecasting with uncertainty quantification</li>
                <li>Spatial analysis at 5km global grid resolution</li>
            </ul>
        </div>
        """
        
        summary_html += '</div>'
        
        return summary_html
        
    def _get_score_color(self, decile: int) -> str:
        """Get color for Blue Zone score decile"""
        colors = {
            1: '#d73027', 2: '#f46d43', 3: '#fdae61', 4: '#fee08b', 5: '#ffffbf',
            6: '#e6f598', 7: '#abdda4', 8: '#66c2a5', 9: '#3288bd', 10: '#5e4fa2'
        }
        return colors.get(decile, '#gray')
        
    def _value_to_color(self, norm_value: float) -> str:
        """Convert normalized value to color"""
        # Simple blue-red color scale
        if norm_value < 0.2:
            return '#3288bd'
        elif norm_value < 0.4:
            return '#66c2a5'
        elif norm_value < 0.6:
            return '#abdda4'
        elif norm_value < 0.8:
            return '#fee08b'
        else:
            return '#d73027'
            
    def _add_score_legend(self, m: folium.Map) -> None:
        """Add legend to score map"""
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><strong>Blue Zone Score</strong></p>
        <p><i class="fa fa-circle" style="color:#5e4fa2"></i> High (9-10)</p>
        <p><i class="fa fa-circle" style="color:#3288bd"></i> Medium (5-8)</p>
        <p><i class="fa fa-circle" style="color:#d73027"></i> Low (1-4)</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))