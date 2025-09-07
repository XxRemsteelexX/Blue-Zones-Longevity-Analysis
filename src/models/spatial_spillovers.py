"""
Spatial Spillover Effects Analysis
Test how neighboring regions' characteristics affect local outcomes
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


class SpatialSpilloverAnalyzer:
    """Analyze spatial spillover effects in health outcomes"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.spatial_weights = {}
        
    def create_spatial_weights_matrix(self, data: pd.DataFrame, 
                                     distance_threshold: float = 100,
                                     k_neighbors: Optional[int] = None,
                                     method: str = 'distance') -> np.ndarray:
        """
        Create spatial weights matrix
        
        Args:
            data: DataFrame with latitude/longitude
            distance_threshold: Max distance (km) for neighbors
            k_neighbors: Number of nearest neighbors (alternative to distance)
            method: 'distance', 'knn', or 'inverse_distance'
            
        Returns:
            Spatial weights matrix (n x n)
        """
        
        # Extract coordinates
        coords = data[['latitude', 'longitude']].values
        n = len(coords)
        
        # Calculate distance matrix (in kilometers)
        # Using haversine formula for Earth distances
        distances = self._haversine_distance_matrix(coords)
        
        # Create weights matrix
        W = np.zeros((n, n))
        
        if method == 'distance':
            # Binary weights: 1 if within threshold, 0 otherwise
            W = (distances <= distance_threshold).astype(float)
            np.fill_diagonal(W, 0)  # Remove self-connections
            
        elif method == 'knn' and k_neighbors:
            # K-nearest neighbors
            for i in range(n):
                neighbor_indices = np.argsort(distances[i])[1:k_neighbors+1]  # Skip self (index 0)
                W[i, neighbor_indices] = 1
                
        elif method == 'inverse_distance':
            # Inverse distance weights
            W = 1 / (distances + 0.1)  # Add small constant to avoid division by zero
            np.fill_diagonal(W, 0)
            
            # Apply distance threshold
            W[distances > distance_threshold] = 0
        
        # Row-normalize weights (each row sums to 1)
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero for isolated units
        W = W / row_sums[:, np.newaxis]
        
        self.spatial_weights[f"{method}_{distance_threshold}"] = W
        
        self.logger.info(f"Created {method} weights matrix: {n}x{n}")
        self.logger.info(f"Average neighbors per unit: {(W > 0).sum(axis=1).mean():.1f}")
        
        return W
    
    def _haversine_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Calculate haversine distances between all coordinate pairs"""
        
        def haversine_vectorized(lat1, lon1, lat2, lon2):
            """Vectorized haversine distance calculation"""
            R = 6371  # Earth's radius in km
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        n = len(coords)
        distances = np.zeros((n, n))
        
        for i in range(n):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[:, 0], coords[:, 1]
            distances[i, :] = haversine_vectorized(lat1, lon1, lat2, lon2)
        
        return distances
    
    def calculate_spatial_lags(self, data: pd.DataFrame, 
                              variables: List[str],
                              weights_matrix: np.ndarray) -> pd.DataFrame:
        """
        Calculate spatial lags for specified variables
        
        Args:
            data: DataFrame with variables
            variables: List of variables to create spatial lags for
            weights_matrix: Spatial weights matrix
            
        Returns:
            DataFrame with original data plus spatial lag variables
        """
        
        data_with_lags = data.copy()
        
        for var in variables:
            if var in data.columns:
                # Calculate spatial lag: W * X
                spatial_lag = weights_matrix @ data[var].fillna(0).values
                data_with_lags[f'{var}_spatial_lag'] = spatial_lag
                
                # Also calculate spatial difference (X - spatial lag)
                data_with_lags[f'{var}_spatial_diff'] = data[var] - spatial_lag
                
                self.logger.info(f"Created spatial lag for {var}")
        
        return data_with_lags
    
    def test_spillover_effects(self, data: pd.DataFrame,
                              outcome_var: str,
                              treatment_vars: List[str],
                              control_vars: List[str] = None,
                              weights_matrix: np.ndarray = None) -> Dict[str, Any]:
        """
        Test for spatial spillover effects using various methods
        
        Args:
            data: DataFrame with outcomes, treatments, and spatial lags
            outcome_var: Dependent variable
            treatment_vars: Treatment variables (both direct and spatial lags)
            control_vars: Control variables
            weights_matrix: Spatial weights matrix for additional tests
            
        Returns:
            Spillover test results
        """
        
        if control_vars is None:
            control_vars = []
        
        results = {}
        
        # 1. Direct vs Spillover Effects Comparison
        self.logger.info("Testing direct vs spillover effects...")
        
        spillover_results = {}
        
        for treat_var in treatment_vars:
            direct_var = treat_var
            spillover_var = f'{treat_var}_spatial_lag'
            
            if spillover_var in data.columns:
                # Regression with both direct and spillover effects
                reg_vars = [direct_var, spillover_var] + control_vars
                clean_data = data[[outcome_var] + reg_vars].dropna()
                
                if len(clean_data) > len(reg_vars) + 5:  # Minimum observations
                    from sklearn.linear_model import LinearRegression
                    from scipy import stats
                    
                    X = clean_data[reg_vars].values
                    y = clean_data[outcome_var].values
                    
                    reg = LinearRegression()
                    reg.fit(X, y)
                    
                    # Get coefficients
                    direct_coef = reg.coef_[0]
                    spillover_coef = reg.coef_[1]
                    
                    # Calculate standard errors (simplified)
                    y_pred = reg.predict(X)
                    residuals = y - y_pred
                    n, k = len(y), X.shape[1]
                    mse = np.sum(residuals**2) / (n - k)
                    
                    try:
                        var_coef = mse * np.linalg.inv(X.T @ X)
                        std_errors = np.sqrt(np.diag(var_coef))
                        
                        direct_se = std_errors[0]
                        spillover_se = std_errors[1]
                        
                        # T-tests
                        direct_t = direct_coef / direct_se
                        spillover_t = spillover_coef / spillover_se
                        
                        direct_p = 2 * (1 - stats.t.cdf(np.abs(direct_t), n - k))
                        spillover_p = 2 * (1 - stats.t.cdf(np.abs(spillover_t), n - k))
                        
                    except:
                        direct_se = spillover_se = np.nan
                        direct_p = spillover_p = np.nan
                    
                    spillover_results[treat_var] = {
                        'direct_effect': direct_coef,
                        'direct_se': direct_se,
                        'direct_p_value': direct_p,
                        'spillover_effect': spillover_coef,
                        'spillover_se': spillover_se,
                        'spillover_p_value': spillover_p,
                        'spillover_ratio': abs(spillover_coef / direct_coef) if direct_coef != 0 else np.inf,
                        'total_effect': direct_coef + spillover_coef,
                        'r_squared': reg.score(X, y),
                        'n_obs': len(clean_data)
                    }
        
        results['spillover_effects'] = spillover_results
        
        # 2. Moran's I test for spatial autocorrelation
        if weights_matrix is not None:
            self.logger.info("Testing spatial autocorrelation...")
            
            morans_results = {}
            test_vars = [outcome_var] + treatment_vars
            
            for var in test_vars:
                if var in data.columns:
                    clean_values = data[var].dropna().values
                    if len(clean_values) == weights_matrix.shape[0]:
                        morans_i = self._calculate_morans_i(clean_values, weights_matrix)
                        morans_results[var] = morans_i
            
            results['spatial_autocorrelation'] = morans_results
        
        # 3. Range of neighbor influence
        self.logger.info("Analyzing range of spatial influence...")
        
        if 'latitude' in data.columns and 'longitude' in data.columns:
            influence_ranges = {}
            
            for treat_var in treatment_vars:
                if treat_var in data.columns:
                    # Test different distance thresholds
                    distances = [25, 50, 100, 200, 500]  # km
                    range_effects = []
                    
                    for dist in distances:
                        try:
                            # Create weights for this distance
                            temp_weights = self.create_spatial_weights_matrix(
                                data, distance_threshold=dist, method='distance'
                            )
                            
                            # Calculate spatial lag
                            spatial_lag = temp_weights @ data[treat_var].fillna(0).values
                            
                            # Quick correlation with outcome
                            corr = np.corrcoef(data[outcome_var].fillna(0), spatial_lag)[0, 1]
                            range_effects.append({'distance': dist, 'correlation': corr})
                            
                        except:
                            continue
                    
                    influence_ranges[treat_var] = range_effects
            
            results['influence_ranges'] = influence_ranges
        
        return results
    
    def _calculate_morans_i(self, values: np.ndarray, weights_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate Moran's I statistic for spatial autocorrelation"""
        
        n = len(values)
        mean_val = np.mean(values)
        
        # Numerator: sum of spatial weights * products of deviations
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += weights_matrix[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
        
        # Denominator: sum of squared deviations
        denominator = np.sum((values - mean_val)**2)
        
        # Sum of all weights
        W = np.sum(weights_matrix)
        
        # Moran's I
        if W > 0 and denominator > 0:
            morans_i = (n / W) * (numerator / denominator)
        else:
            morans_i = 0
        
        # Expected value under null hypothesis of no spatial correlation
        expected_i = -1 / (n - 1)
        
        # Simplified significance test (z-score approximation)
        variance_i = (n**2 - 3*n + 3) * W - n * np.sum(weights_matrix**2) + 3 * W**2
        variance_i = variance_i / ((n - 1) * (n - 2) * (n - 3) * W**2)
        
        if variance_i > 0:
            z_score = (morans_i - expected_i) / np.sqrt(variance_i)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0
            p_value = 1
        
        return {
            'morans_i': morans_i,
            'expected_i': expected_i,
            'variance': variance_i,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def visualize_spillovers(self, data: pd.DataFrame, 
                           treatment_var: str,
                           outcome_var: str,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Create spillover effect visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Spatial Spillover Analysis: {treatment_var} → {outcome_var}', fontsize=16)
        
        # 1. Direct vs Spillover scatter
        if f'{treatment_var}_spatial_lag' in data.columns:
            ax1 = axes[0, 0]
            
            # Direct effect
            clean_data = data[[outcome_var, treatment_var]].dropna()
            ax1.scatter(clean_data[treatment_var], clean_data[outcome_var], 
                       alpha=0.6, color='blue', label='Direct Effect')
            
            # Add trendline
            z = np.polyfit(clean_data[treatment_var], clean_data[outcome_var], 1)
            p = np.poly1d(z)
            ax1.plot(clean_data[treatment_var].sort_values(), 
                    p(clean_data[treatment_var].sort_values()), 
                    "r--", alpha=0.8)
            
            ax1.set_xlabel(f'{treatment_var} (Direct)')
            ax1.set_ylabel(outcome_var)
            ax1.set_title('Direct Effect')
            ax1.legend()
        
        # 2. Spillover effect
        if f'{treatment_var}_spatial_lag' in data.columns:
            ax2 = axes[0, 1]
            
            spillover_data = data[[outcome_var, f'{treatment_var}_spatial_lag']].dropna()
            ax2.scatter(spillover_data[f'{treatment_var}_spatial_lag'], 
                       spillover_data[outcome_var], 
                       alpha=0.6, color='green', label='Spillover Effect')
            
            # Add trendline
            if len(spillover_data) > 1:
                z = np.polyfit(spillover_data[f'{treatment_var}_spatial_lag'], 
                              spillover_data[outcome_var], 1)
                p = np.poly1d(z)
                ax2.plot(spillover_data[f'{treatment_var}_spatial_lag'].sort_values(), 
                        p(spillover_data[f'{treatment_var}_spatial_lag'].sort_values()), 
                        "r--", alpha=0.8)
            
            ax2.set_xlabel(f'{treatment_var} (Neighbors)')
            ax2.set_ylabel(outcome_var)
            ax2.set_title('Spillover Effect')
            ax2.legend()
        
        # 3. Geographic map if coordinates available
        if all(col in data.columns for col in ['latitude', 'longitude']):
            ax3 = axes[1, 0]
            
            scatter = ax3.scatter(data['longitude'], data['latitude'], 
                                c=data[treatment_var], cmap='viridis', 
                                s=50, alpha=0.7)
            
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.set_title(f'Geographic Distribution: {treatment_var}')
            plt.colorbar(scatter, ax=ax3)
        
        # 4. Spatial correlation by distance
        ax4 = axes[1, 1]
        
        if 'latitude' in data.columns and 'longitude' in data.columns:
            # Calculate correlations at different distance bands
            distances = [25, 50, 100, 200, 500]
            correlations = []
            
            coords = data[['latitude', 'longitude']].values
            distance_matrix = self._haversine_distance_matrix(coords)
            
            for max_dist in distances:
                # Find pairs within distance band
                mask = (distance_matrix > 0) & (distance_matrix <= max_dist)
                
                if mask.sum() > 0:
                    # Extract values for pairs within distance
                    values = data[treatment_var].fillna(0).values
                    corr_sum = 0
                    pair_count = 0
                    
                    for i in range(len(values)):
                        for j in range(i+1, len(values)):
                            if mask[i, j]:
                                corr_sum += values[i] * values[j]
                                pair_count += 1
                    
                    if pair_count > 0:
                        avg_correlation = corr_sum / pair_count
                        correlations.append(avg_correlation)
                    else:
                        correlations.append(0)
                else:
                    correlations.append(0)
            
            ax4.plot(distances, correlations, 'o-', linewidth=2, markersize=8)
            ax4.set_xlabel('Maximum Distance (km)')
            ax4.set_ylabel('Spatial Correlation')
            ax4.set_title('Spatial Correlation by Distance')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Spillover visualization saved to {save_path}")
        
        return fig
    
    def generate_spillover_report(self, spillover_results: Dict[str, Any]) -> str:
        """Generate comprehensive spillover analysis report"""
        
        report_lines = [
            "🗺️ SPATIAL SPILLOVER EFFECTS ANALYSIS",
            "=" * 50,
            "",
            "HYPOTHESIS: Neighboring regions' characteristics affect local health outcomes.",
            "Tests whether 'your neighbor's walkability affects your longevity'.",
            ""
        ]
        
        # Spillover effects results
        if 'spillover_effects' in spillover_results:
            report_lines.extend([
                "SPILLOVER EFFECTS ANALYSIS:",
                "-" * 30
            ])
            
            for var, results in spillover_results['spillover_effects'].items():
                direct_sig = "***" if results['direct_p_value'] < 0.001 else "**" if results['direct_p_value'] < 0.01 else "*" if results['direct_p_value'] < 0.05 else ""
                spillover_sig = "***" if results['spillover_p_value'] < 0.001 else "**" if results['spillover_p_value'] < 0.01 else "*" if results['spillover_p_value'] < 0.05 else ""
                
                report_lines.extend([
                    f"\n{var.upper()}:",
                    f"  • Direct Effect: {results['direct_effect']:.4f}{direct_sig}",
                    f"  • Spillover Effect: {results['spillover_effect']:.4f}{spillover_sig}",
                    f"  • Spillover Ratio: {results['spillover_ratio']:.2f}",
                    f"  • Total Effect: {results['total_effect']:.4f}",
                    f"  • R²: {results['r_squared']:.4f}",
                    f"  • Observations: {results['n_obs']}"
                ])
                
                # Interpretation
                if results['spillover_p_value'] < 0.05:
                    spillover_strength = "strong" if results['spillover_ratio'] > 0.5 else "moderate" if results['spillover_ratio'] > 0.2 else "weak"
                    report_lines.append(f"  • Interpretation: {spillover_strength} spatial spillover effects detected")
        
        # Spatial autocorrelation results
        if 'spatial_autocorrelation' in spillover_results:
            report_lines.extend([
                "",
                "SPATIAL AUTOCORRELATION (MORAN'S I):",
                "-" * 30
            ])
            
            for var, moran in spillover_results['spatial_autocorrelation'].items():
                sig_marker = "***" if moran['p_value'] < 0.001 else "**" if moran['p_value'] < 0.01 else "*" if moran['p_value'] < 0.05 else ""
                
                autocorr_type = "positive clustering" if moran['morans_i'] > moran['expected_i'] else "negative clustering" if moran['morans_i'] < moran['expected_i'] else "random"
                
                report_lines.extend([
                    f"\n{var}:",
                    f"  • Moran's I: {moran['morans_i']:.4f}{sig_marker}",
                    f"  • Expected I: {moran['expected_i']:.4f}",
                    f"  • Z-score: {moran['z_score']:.4f}",
                    f"  • P-value: {moran['p_value']:.4f}",
                    f"  • Pattern: {autocorr_type}"
                ])
        
        # Range of influence
        if 'influence_ranges' in spillover_results:
            report_lines.extend([
                "",
                "SPATIAL INFLUENCE RANGE:",
                "-" * 30
            ])
            
            for var, ranges in spillover_results['influence_ranges'].items():
                report_lines.append(f"\n{var}:")
                
                max_corr = 0
                optimal_distance = 0
                
                for range_data in ranges:
                    dist = range_data['distance']
                    corr = range_data['correlation']
                    report_lines.append(f"  • {dist}km: correlation = {corr:.4f}")
                    
                    if abs(corr) > abs(max_corr):
                        max_corr = corr
                        optimal_distance = dist
                
                if ranges:
                    report_lines.append(f"  • Optimal range: {optimal_distance}km (r = {max_corr:.4f})")
        
        # Conclusions
        report_lines.extend([
            "",
            "KEY FINDINGS:",
            "-" * 20,
            "• Spatial spillovers quantify 'neighborhood effects' on health",
            "• Significant spillovers suggest policy coordination benefits",
            "• Range analysis identifies optimal intervention scales",
            "",
            "POLICY IMPLICATIONS:",
            "• Health interventions have regional multiplier effects",
            "• Coordinate policies across neighboring jurisdictions",
            "• Consider spillover benefits in cost-benefit analysis"
        ])
        
        return "\n".join(report_lines)


if __name__ == "__main__":
    # Example usage
    from scipy import stats
    
    # Create sample spatial data
    np.random.seed(42)
    n_locations = 100
    
    # Generate coordinates (scattered around globe)
    latitudes = np.random.uniform(-60, 70, n_locations)
    longitudes = np.random.uniform(-180, 180, n_locations)
    
    # Create spatial structure in treatment (clusters of high/low values)
    treatment = np.random.normal(0, 1, n_locations)
    
    # Add spatial correlation to treatment
    coords = np.column_stack([latitudes, longitudes])
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                # Haversine distance
                lat1, lon1 = np.radians(coords[i])
                lat2, lon2 = np.radians(coords[j])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance = 6371 * c  # km
                
                if distance < 500:  # Nearby locations influence each other
                    treatment[i] += 0.1 * treatment[j] * (500 - distance) / 500
    
    # Generate outcome with both direct and spillover effects
    outcome = np.zeros(n_locations)
    
    for i in range(n_locations):
        # Direct effect
        outcome[i] += 2.0 * treatment[i]
        
        # Spillover effect from neighbors
        neighbor_effect = 0
        neighbor_count = 0
        
        for j in range(n_locations):
            if i != j:
                lat1, lon1 = np.radians(coords[i])
                lat2, lon2 = np.radians(coords[j])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance = 6371 * c
                
                if distance < 200:  # Spillover from nearby neighbors
                    weight = (200 - distance) / 200
                    neighbor_effect += weight * treatment[j]
                    neighbor_count += weight
        
        if neighbor_count > 0:
            outcome[i] += 0.8 * (neighbor_effect / neighbor_count)  # Spillover effect
        
        # Add noise
        outcome[i] += np.random.normal(0, 0.5)
    
    # Create DataFrame
    data = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'treatment': treatment,
        'outcome': outcome,
        'control1': np.random.normal(0, 1, n_locations)
    })
    
    # Run spillover analysis
    spillover_analyzer = SpatialSpilloverAnalyzer()
    
    # Create spatial weights
    weights_matrix = spillover_analyzer.create_spatial_weights_matrix(
        data, distance_threshold=200, method='distance'
    )
    
    # Calculate spatial lags
    data_with_lags = spillover_analyzer.calculate_spatial_lags(
        data, ['treatment', 'control1'], weights_matrix
    )
    
    # Test spillover effects
    spillover_results = spillover_analyzer.test_spillover_effects(
        data_with_lags, 'outcome', ['treatment'], ['control1'], weights_matrix
    )
    
    # Generate report
    report = spillover_analyzer.generate_spillover_report(spillover_results)
    print(report)
    
    # Create visualization
    fig = spillover_analyzer.visualize_spillovers(
        data_with_lags, 'treatment', 'outcome'
    )
    plt.show()