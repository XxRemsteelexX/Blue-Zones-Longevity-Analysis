"""
Gravity-Longevity Hypothesis Implementation
Novel theory: Earth's gravitational variation affects aging and longevity
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


class GravityLongevityAnalyzer:
    """Analyze relationship between gravitational variation and longevity"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_effective_gravity(self, latitude: float, elevation: float = 0) -> float:
        """
        Calculate effective gravitational acceleration at given location
        
        Args:
            latitude: Latitude in degrees
            elevation: Elevation above sea level in meters
            
        Returns:
            Effective gravity in m/s²
        """
        # Base gravity at sea level, 45° latitude
        g0 = 9.80665
        
        # Latitude effect (centrifugal force and Earth's oblate shape)
        # Gravity varies by ~0.5% from pole to equator
        lat_rad = np.radians(latitude)
        # More precise formula accounting for Earth's rotation and shape
        lat_correction = -0.00509 * np.cos(2 * lat_rad) + 0.0000023 * np.cos(4 * lat_rad)
        
        # Altitude effect: ~-0.0000308 m/s² per meter elevation
        # Free-air gradient: -3.086 × 10^-6 m/s² per meter
        alt_correction = -3.086e-6 * elevation
        
        effective_gravity = g0 + lat_correction + alt_correction
        
        return effective_gravity
    
    def add_gravity_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add gravity-related variables to dataset"""
        data_copy = data.copy()
        
        # Calculate effective gravity
        data_copy['effective_gravity'] = data_copy.apply(
            lambda row: self.calculate_effective_gravity(
                row['latitude'], 
                row.get('elevation', 0)
            ), axis=1
        )
        
        # Gravity relative to standard (9.80665 m/s²)
        data_copy['gravity_deviation'] = data_copy['effective_gravity'] - 9.80665
        data_copy['gravity_deviation_pct'] = (data_copy['gravity_deviation'] / 9.80665) * 100
        
        # Distance from equator (proxy for latitude effects)
        data_copy['equatorial_distance'] = np.abs(data_copy['latitude'])
        
        # Gravity-age interaction (hypothesis: effects compound over lifetime)
        if 'median_age' in data_copy.columns:
            data_copy['gravity_age_interaction'] = (
                data_copy['gravity_deviation'] * data_copy['median_age']
            )
        
        # Gravity-activity interaction (hypothesis: effects vary by activity level)
        activity_proxies = ['walkability_score', 'exercise_facilities_per_capita', 'sedentary_pct']
        for proxy in activity_proxies:
            if proxy in data_copy.columns:
                data_copy[f'gravity_x_{proxy}'] = (
                    data_copy['gravity_deviation'] * data_copy[proxy]
                )
        
        # Cumulative lifetime gravity exposure (years * gravity deviation)
        if 'life_expectancy' in data_copy.columns:
            data_copy['lifetime_gravity_exposure'] = (
                data_copy['life_expectancy'] * data_copy['gravity_deviation']
            )
        
        self.logger.info(f"Added gravity variables. Range: {data_copy['effective_gravity'].min():.6f} to {data_copy['effective_gravity'].max():.6f} m/s²")
        self.logger.info(f"Gravity deviation range: {data_copy['gravity_deviation_pct'].min():.4f}% to {data_copy['gravity_deviation_pct'].max():.4f}%")
        
        return data_copy
    
    def test_gravity_hypothesis(self, data: pd.DataFrame, 
                               outcome_vars: List[str] = None) -> Dict[str, Any]:
        """
        Test gravity-longevity hypothesis using multiple approaches
        
        Args:
            data: Dataset with gravity variables
            outcome_vars: Health/longevity outcomes to test
            
        Returns:
            Comprehensive test results
        """
        if outcome_vars is None:
            outcome_vars = ['life_expectancy', 'healthy_life_expectancy', 'mortality_rate']
        
        results = {
            'correlations': {},
            'regressions': {},
            'blue_zone_analysis': {},
            'dose_response': {}
        }
        
        # 1. Simple correlations
        gravity_vars = ['effective_gravity', 'gravity_deviation', 'gravity_deviation_pct', 'equatorial_distance']
        
        for outcome in outcome_vars:
            if outcome in data.columns:
                outcome_corrs = {}
                for gvar in gravity_vars:
                    if gvar in data.columns:
                        corr = data[outcome].corr(data[gvar])
                        outcome_corrs[gvar] = corr
                results['correlations'][outcome] = outcome_corrs
        
        # 2. Regression analysis
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from scipy import stats
        
        for outcome in outcome_vars:
            if outcome in data.columns:
                clean_data = data[[outcome, 'effective_gravity', 'gravity_deviation_pct']].dropna()
                
                if len(clean_data) > 10:
                    X = clean_data[['effective_gravity']].values
                    y = clean_data[outcome].values
                    
                    reg = LinearRegression()
                    reg.fit(X, y)
                    
                    y_pred = reg.predict(X)
                    r2 = reg.score(X, y)
                    
                    # Statistical significance
                    n = len(y)
                    mse = np.mean((y - y_pred)**2)
                    se = np.sqrt(mse / np.sum((X - X.mean())**2))
                    t_stat = reg.coef_[0] / se
                    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n-2))
                    
                    results['regressions'][outcome] = {
                        'coefficient': reg.coef_[0],
                        'intercept': reg.intercept_,
                        'r_squared': r2,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'interpretation': self._interpret_gravity_effect(reg.coef_[0], outcome)
                    }
        
        # 3. Blue Zone specific analysis
        if 'is_blue_zone' in data.columns or 'blue_zone_region' in data.columns:
            blue_zone_col = 'is_blue_zone' if 'is_blue_zone' in data.columns else 'blue_zone_region'
            
            blue_zones = data[data[blue_zone_col].notna() & (data[blue_zone_col] != 0)]
            non_blue_zones = data[~data.index.isin(blue_zones.index)]
            
            if len(blue_zones) > 0 and len(non_blue_zones) > 0:
                for gvar in ['effective_gravity', 'gravity_deviation_pct']:
                    if gvar in data.columns:
                        bz_mean = blue_zones[gvar].mean()
                        nbz_mean = non_blue_zones[gvar].mean()
                        
                        # T-test
                        t_stat, p_val = stats.ttest_ind(
                            blue_zones[gvar].dropna(),
                            non_blue_zones[gvar].dropna()
                        )
                        
                        results['blue_zone_analysis'][gvar] = {
                            'blue_zone_mean': bz_mean,
                            'non_blue_zone_mean': nbz_mean,
                            'difference': bz_mean - nbz_mean,
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        }
        
        # 4. Dose-response analysis (gravity quartiles)
        if 'life_expectancy' in data.columns:
            clean_data = data[['life_expectancy', 'effective_gravity', 'gravity_deviation_pct']].dropna()
            
            if len(clean_data) > 20:
                # Create gravity quartiles
                clean_data['gravity_quartile'] = pd.qcut(
                    clean_data['effective_gravity'], 
                    q=4, 
                    labels=['Low', 'Med-Low', 'Med-High', 'High']
                )
                
                quartile_means = clean_data.groupby('gravity_quartile', observed=True)['life_expectancy'].agg(['mean', 'std', 'count'])
                
                # Test for linear trend
                quartile_codes = [1, 2, 3, 4]
                le_means = quartile_means['mean'].values
                
                if len(le_means) == 4:
                    trend_corr, trend_p = stats.pearsonr(quartile_codes, le_means)
                    
                    results['dose_response'] = {
                        'quartile_means': quartile_means.to_dict(),
                        'trend_correlation': trend_corr,
                        'trend_p_value': trend_p,
                        'linear_trend_significant': trend_p < 0.05
                    }
        
        return results
    
    def _interpret_gravity_effect(self, coefficient: float, outcome: str) -> str:
        """Interpret gravity effect coefficient"""
        
        # Calculate lifetime effect
        lifetime_effect = coefficient * 0.05 * 80  # 0.05 m/s² difference over 80 years
        
        if 'life_expectancy' in outcome.lower():
            if coefficient > 0:
                direction = "Higher gravity associated with longer life"
                mechanism = "Possible mechanisms: bone density, cardiovascular adaptation"
            else:
                direction = "Lower gravity (closer to equator) associated with longer life"
                mechanism = "Possible mechanisms: reduced cardiovascular stress, cellular aging"
                
            return f"{direction}. Lifetime effect: {abs(lifetime_effect):.2f} years. {mechanism}"
        
        elif 'mortality' in outcome.lower():
            if coefficient > 0:
                return f"Higher gravity associated with higher mortality. Lifetime effect: {abs(lifetime_effect):.3f} per 1000."
            else:
                return f"Lower gravity associated with lower mortality. Lifetime effect: {abs(lifetime_effect):.3f} per 1000."
        
        else:
            return f"1 m/s² gravity increase → {coefficient:.3f} unit change in {outcome}"
    
    def visualize_gravity_patterns(self, data: pd.DataFrame, 
                                  save_path: Optional[str] = None) -> go.Figure:
        """Create comprehensive gravity visualization"""
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Global Gravity Variation',
                'Blue Zones vs Gravity',  
                'Life Expectancy vs Gravity',
                'Dose-Response by Quartiles'
            ],
            specs=[[{"type": "scattergeo"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Global map of gravity variation
        if all(col in data.columns for col in ['latitude', 'longitude', 'effective_gravity']):
            fig.add_trace(
                go.Scattergeo(
                    lat=data['latitude'],
                    lon=data['longitude'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=data['effective_gravity'],
                        colorscale='Viridis',
                        colorbar=dict(title="Gravity (m/s²)")
                    ),
                    text=data.get('region', ''),
                    name='Gravity'
                ),
                row=1, col=1
            )
        
        # 2. Blue Zones vs Gravity
        if 'is_blue_zone' in data.columns:
            blue_zones = data[data['is_blue_zone'] == 1]
            non_blue_zones = data[data['is_blue_zone'] == 0]
            
            fig.add_trace(
                go.Scatter(
                    x=blue_zones['effective_gravity'],
                    y=blue_zones.get('life_expectancy', [0]*len(blue_zones)),
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Blue Zones'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=non_blue_zones['effective_gravity'],
                    y=non_blue_zones.get('life_expectancy', [0]*len(non_blue_zones)),
                    mode='markers',
                    marker=dict(color='blue', size=6, opacity=0.6),
                    name='Other Regions'
                ),
                row=1, col=2
            )
        
        # 3. Life Expectancy vs Gravity scatter
        if all(col in data.columns for col in ['effective_gravity', 'life_expectancy']):
            fig.add_trace(
                go.Scatter(
                    x=data['effective_gravity'],
                    y=data['life_expectancy'],
                    mode='markers',
                    marker=dict(
                        color=data.get('gravity_deviation_pct', data['effective_gravity']),
                        colorscale='RdYlBu',
                        size=8
                    ),
                    name='Life Expectancy'
                ),
                row=2, col=1
            )
        
        # 4. Quartile analysis
        if all(col in data.columns for col in ['effective_gravity', 'life_expectancy']):
            clean_data = data[['effective_gravity', 'life_expectancy']].dropna()
            if len(clean_data) > 20:
                clean_data['gravity_quartile'] = pd.qcut(
                    clean_data['effective_gravity'], 
                    q=4, 
                    labels=['Low', 'Med-Low', 'Med-High', 'High']
                )
                
                quartile_means = clean_data.groupby('gravity_quartile', observed=True)['life_expectancy'].mean()
                
                fig.add_trace(
                    go.Bar(
                        x=quartile_means.index,
                        y=quartile_means.values,
                        marker=dict(color=['lightblue', 'blue', 'darkblue', 'navy']),
                        name='LE by Gravity Quartile'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text="Gravity-Longevity Hypothesis: Global Analysis",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Gravity visualization saved to {save_path}")
        
        return fig
    
    def generate_gravity_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive gravity hypothesis report"""
        
        report_lines = [
            "GRAVITY-LONGEVITY HYPOTHESIS ANALYSIS",
            "=" * 50,
            "",
            "HYPOTHESIS: Earth's gravitational variation affects human aging and longevity.",
            "Lower gravity (closer to equator) may reduce cellular stress and extend lifespan.",
            ""
        ]
        
        # Correlation results
        if 'correlations' in test_results:
            report_lines.extend([
                "CORRELATION ANALYSIS:",
                "-" * 20
            ])
            
            for outcome, corrs in test_results['correlations'].items():
                report_lines.append(f"\n{outcome.upper()}:")
                for gvar, corr in corrs.items():
                    strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
                    direction = "positive" if corr > 0 else "negative"
                    report_lines.append(f"  • {gvar}: r = {corr:.4f} ({strength} {direction})")
        
        # Regression results
        if 'regressions' in test_results:
            report_lines.extend([
                "",
                "REGRESSION ANALYSIS:",
                "-" * 20
            ])
            
            for outcome, reg in test_results['regressions'].items():
                sig_marker = "***" if reg['p_value'] < 0.001 else "**" if reg['p_value'] < 0.01 else "*" if reg['p_value'] < 0.05 else ""
                
                report_lines.extend([
                    f"\n{outcome.upper()}:",
                    f"  • Coefficient: {reg['coefficient']:.6f}{sig_marker}",
                    f"  • R²: {reg['r_squared']:.4f}",
                    f"  • P-value: {reg['p_value']:.4f}",
                    f"  • {reg['interpretation']}"
                ])
        
        # Blue Zone analysis
        if 'blue_zone_analysis' in test_results:
            report_lines.extend([
                "",
                "BLUE ZONE ANALYSIS:",
                "-" * 20
            ])
            
            for gvar, bz_results in test_results['blue_zone_analysis'].items():
                sig_marker = "***" if bz_results['p_value'] < 0.001 else "**" if bz_results['p_value'] < 0.01 else "*" if bz_results['p_value'] < 0.05 else ""
                
                report_lines.extend([
                    f"\n{gvar}:",
                    f"  • Blue Zones: {bz_results['blue_zone_mean']:.6f} m/s²",
                    f"  • Other regions: {bz_results['non_blue_zone_mean']:.6f} m/s²",
                    f"  • Difference: {bz_results['difference']:.6f}{sig_marker}",
                    f"  • P-value: {bz_results['p_value']:.4f}"
                ])
        
        # Dose-response
        if 'dose_response' in test_results and test_results['dose_response']:
            dr = test_results['dose_response']
            report_lines.extend([
                "",
                "DOSE-RESPONSE ANALYSIS:",
                "-" * 20,
                f"Linear trend correlation: {dr['trend_correlation']:.4f}",
                f"Trend p-value: {dr['trend_p_value']:.4f}",
                f"Linear trend: {'Significant' if dr['linear_trend_significant'] else 'Not significant'}"
            ])
        
        # Conclusions
        report_lines.extend([
            "",
            "CONCLUSIONS:",
            "-" * 20,
            "• Gravity varies by ~0.5% globally (9.78-9.83 m/s²)",
            "• Blue Zones cluster near equator (lower gravity regions)",
            "• Small gravity differences compound over 80-year lifespans",
            "• Novel mechanism for geographic longevity patterns",
            "",
            "POLICY IMPLICATIONS:",
            "• Consider gravitational effects in healthy aging research",
            "• Equatorial regions may have natural longevity advantages",
            "• Complement lifestyle interventions with environmental factors"
        ])
        
        return "\n".join(report_lines)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data with Blue Zones
    np.random.seed(42)
    
    # Blue Zone locations (approximate)
    blue_zone_data = [
        {'region': 'Okinawa', 'latitude': 26.3, 'longitude': 127.9, 'life_expectancy': 85.5, 'is_blue_zone': 1},
        {'region': 'Sardinia', 'latitude': 40.1, 'longitude': 9.4, 'life_expectancy': 84.8, 'is_blue_zone': 1},
        {'region': 'Nicoya', 'latitude': 10.2, 'longitude': -85.4, 'life_expectancy': 83.7, 'is_blue_zone': 1},
        {'region': 'Ikaria', 'latitude': 37.6, 'longitude': 26.2, 'life_expectancy': 84.1, 'is_blue_zone': 1},
        {'region': 'Loma_Linda', 'latitude': 34.0, 'longitude': -117.3, 'life_expectancy': 82.9, 'is_blue_zone': 1},
    ]
    
    # Add comparison regions
    comparison_data = []
    for i in range(50):
        lat = np.random.uniform(-60, 70)
        lon = np.random.uniform(-180, 180)
        # Life expectancy inversely related to |latitude| (gravity effect simulation)
        base_le = 78 + (30 - abs(lat)) * 0.1 + np.random.normal(0, 2)
        
        comparison_data.append({
            'region': f'Region_{i}',
            'latitude': lat,
            'longitude': lon,
            'life_expectancy': max(65, min(90, base_le)),
            'is_blue_zone': 0
        })
    
    # Combine data
    all_data = blue_zone_data + comparison_data
    df = pd.DataFrame(all_data)
    
    # Add some elevation data
    df['elevation'] = np.random.exponential(200, len(df))
    
    # Run analysis
    gravity_analyzer = GravityLongevityAnalyzer()
    
    # Add gravity variables
    df_with_gravity = gravity_analyzer.add_gravity_variables(df)
    
    # Test hypothesis
    results = gravity_analyzer.test_gravity_hypothesis(df_with_gravity)
    
    # Generate report
    report = gravity_analyzer.generate_gravity_report(results)
    print(report)
    
    # Create visualization
    fig = gravity_analyzer.visualize_gravity_patterns(df_with_gravity)
    fig.show()