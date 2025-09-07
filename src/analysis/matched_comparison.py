"""
Matched comparison analysis for Blue Zones vs control regions
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


class MatchedComparison:
    """Matched comparison analysis using propensity score matching"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def estimate_treatment_effects(self, data: pd.DataFrame,
                                 treatment_col: str = 'is_blue_zone',
                                 outcome_col: str = 'val',
                                 feature_cols: List[str] = None) -> Dict[str, Any]:
        """
        Estimate average treatment effect of Blue Zones
        
        Args:
            data: Analysis dataset
            treatment_col: Blue Zone indicator column
            outcome_col: Life expectancy column
            feature_cols: Features for matching
            
        Returns:
            Dictionary with treatment effect estimates
        """
        if feature_cols is None:
            feature_cols = self._select_matching_features(data)
            
        self.logger.info(f"Estimating treatment effects using {len(feature_cols)} features")
        
        # Prepare data
        analysis_data = data.dropna(subset=[treatment_col, outcome_col] + feature_cols)
        
        if analysis_data[treatment_col].sum() == 0:
            self.logger.warning("No Blue Zone observations found")
            return self._empty_results()
            
        self.logger.info(f"Analysis dataset: {len(analysis_data)} observations, "
                        f"{analysis_data[treatment_col].sum()} Blue Zone cells")
        
        # Run different matching methods
        results = {}
        
        # 1. Propensity Score Matching
        ps_results = self._propensity_score_matching(
            analysis_data, treatment_col, outcome_col, feature_cols
        )
        results['propensity_score'] = ps_results
        
        # 2. Nearest Neighbor Matching
        nn_results = self._nearest_neighbor_matching(
            analysis_data, treatment_col, outcome_col, feature_cols
        )
        results['nearest_neighbor'] = nn_results
        
        # 3. Exact Matching (if feasible)
        exact_results = self._exact_matching(
            analysis_data, treatment_col, outcome_col, feature_cols
        )
        results['exact_matching'] = exact_results
        
        # 4. Simple difference in means (no matching)
        naive_results = self._naive_comparison(
            analysis_data, treatment_col, outcome_col
        )
        results['naive'] = naive_results
        
        # Summary
        results['summary'] = self._summarize_results(results)
        
        return results
        
    def _select_matching_features(self, data: pd.DataFrame) -> List[str]:
        """Select features for matching"""
        # Exclude outcome and treatment variables
        exclude_patterns = [
            'geo_id', 'year', 'location', 'val', 'lower', 'upper',
            'is_blue_zone', 'is_sardinia', 'is_okinawa', 'is_nicoya', 'is_ikaria', 'is_loma_linda'
        ]
        
        candidate_features = []
        for col in data.columns:
            if (data[col].dtype in [np.number] and 
                not any(pattern in col.lower() for pattern in exclude_patterns)):
                candidate_features.append(col)
                
        # Remove features with too many missing values
        valid_features = []
        for col in candidate_features:
            missing_pct = data[col].isnull().sum() / len(data)
            if missing_pct < 0.5:  # Less than 50% missing
                valid_features.append(col)
                
        self.logger.info(f"Selected {len(valid_features)} features for matching")
        return valid_features
        
    def _propensity_score_matching(self, data: pd.DataFrame,
                                 treatment_col: str, outcome_col: str,
                                 feature_cols: List[str]) -> Dict[str, Any]:
        """Propensity score matching"""
        try:
            # Prepare features
            X = data[feature_cols].fillna(data[feature_cols].median())
            X_scaled = self.scaler.fit_transform(X)
            y_treatment = data[treatment_col]
            y_outcome = data[outcome_col]
            
            # Estimate propensity scores
            ps_model = LogisticRegression(random_state=42, max_iter=1000)
            ps_model.fit(X_scaled, y_treatment)
            propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
            
            # Match treated to control units
            treated_idx = data[data[treatment_col] == 1].index
            control_idx = data[data[treatment_col] == 0].index
            
            if len(treated_idx) == 0 or len(control_idx) == 0:
                return {'att': np.nan, 'se': np.nan, 'n_matched': 0}
            
            matches = []
            caliper = self.config['models']['matching']['caliper']
            
            for idx in treated_idx:
                treated_ps = propensity_scores[idx]
                
                # Find control units within caliper
                control_ps = propensity_scores[control_idx]
                distances = np.abs(control_ps - treated_ps)
                
                valid_matches = control_idx[distances <= caliper]
                
                if len(valid_matches) > 0:
                    # Use closest match
                    best_match = valid_matches[np.argmin(distances[distances <= caliper])]
                    matches.append({
                        'treated_idx': idx,
                        'control_idx': best_match,
                        'treated_outcome': y_outcome[idx],
                        'control_outcome': y_outcome[best_match],
                        'difference': y_outcome[idx] - y_outcome[best_match]
                    })
                    
            if not matches:
                return {'att': np.nan, 'se': np.nan, 'n_matched': 0}
                
            # Calculate ATT
            matches_df = pd.DataFrame(matches)
            att = matches_df['difference'].mean()
            se = matches_df['difference'].std() / np.sqrt(len(matches_df))
            
            return {
                'att': att,
                'se': se,
                'n_matched': len(matches_df),
                'matches': matches_df
            }
            
        except Exception as e:
            self.logger.error(f"Propensity score matching failed: {e}")
            return {'att': np.nan, 'se': np.nan, 'n_matched': 0}
            
    def _nearest_neighbor_matching(self, data: pd.DataFrame,
                                 treatment_col: str, outcome_col: str,
                                 feature_cols: List[str]) -> Dict[str, Any]:
        """Nearest neighbor matching"""
        try:
            # Prepare features
            X = data[feature_cols].fillna(data[feature_cols].median())
            X_scaled = self.scaler.fit_transform(X)
            
            treated_mask = data[treatment_col] == 1
            control_mask = data[treatment_col] == 0
            
            if treated_mask.sum() == 0 or control_mask.sum() == 0:
                return {'att': np.nan, 'se': np.nan, 'n_matched': 0}
                
            treated_X = X_scaled[treated_mask]
            control_X = X_scaled[control_mask]
            treated_outcomes = data[outcome_col][treated_mask]
            control_outcomes = data[outcome_col][control_mask]
            
            # Find nearest neighbors
            nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_model.fit(control_X)
            
            distances, indices = nn_model.kneighbors(treated_X)
            
            # Calculate treatment effects
            matched_control_outcomes = control_outcomes.iloc[indices.flatten()]
            differences = treated_outcomes.values - matched_control_outcomes.values
            
            att = np.mean(differences)
            se = np.std(differences) / np.sqrt(len(differences))
            
            return {
                'att': att,
                'se': se,
                'n_matched': len(differences)
            }
            
        except Exception as e:
            self.logger.error(f"Nearest neighbor matching failed: {e}")
            return {'att': np.nan, 'se': np.nan, 'n_matched': 0}
            
    def _exact_matching(self, data: pd.DataFrame,
                       treatment_col: str, outcome_col: str,
                       feature_cols: List[str]) -> Dict[str, Any]:
        """Exact matching on categorical variables"""
        try:
            # Select categorical features for exact matching
            categorical_features = []
            for col in feature_cols:
                if data[col].nunique() < 20:  # Treat as categorical if <20 unique values
                    categorical_features.append(col)
                    
            if not categorical_features:
                return {'att': np.nan, 'se': np.nan, 'n_matched': 0}
                
            # Create matching strata
            data_copy = data.copy()
            for col in categorical_features:
                data_copy[col] = pd.cut(data_copy[col], bins=5, labels=False)
                
            data_copy['stratum'] = data_copy[categorical_features].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
            
            # Match within strata
            differences = []
            
            for stratum in data_copy['stratum'].unique():
                stratum_data = data_copy[data_copy['stratum'] == stratum]
                
                treated = stratum_data[stratum_data[treatment_col] == 1][outcome_col]
                control = stratum_data[stratum_data[treatment_col] == 0][outcome_col]
                
                if len(treated) > 0 and len(control) > 0:
                    # Simple difference in means within stratum
                    stratum_diff = treated.mean() - control.mean()
                    differences.append(stratum_diff)
                    
            if not differences:
                return {'att': np.nan, 'se': np.nan, 'n_matched': 0}
                
            att = np.mean(differences)
            se = np.std(differences) / np.sqrt(len(differences))
            
            return {
                'att': att,
                'se': se,
                'n_matched': len(differences)
            }
            
        except Exception as e:
            self.logger.error(f"Exact matching failed: {e}")
            return {'att': np.nan, 'se': np.nan, 'n_matched': 0}
            
    def _naive_comparison(self, data: pd.DataFrame,
                         treatment_col: str, outcome_col: str) -> Dict[str, Any]:
        """Simple difference in means without matching"""
        try:
            treated = data[data[treatment_col] == 1][outcome_col]
            control = data[data[treatment_col] == 0][outcome_col]
            
            if len(treated) == 0 or len(control) == 0:
                return {'att': np.nan, 'se': np.nan, 'n_treated': 0, 'n_control': 0}
                
            att = treated.mean() - control.mean()
            
            # Standard error for difference in means
            se_treated = treated.std() / np.sqrt(len(treated))
            se_control = control.std() / np.sqrt(len(control))
            se = np.sqrt(se_treated**2 + se_control**2)
            
            return {
                'att': att,
                'se': se,
                'n_treated': len(treated),
                'n_control': len(control),
                'treated_mean': treated.mean(),
                'control_mean': control.mean()
            }
            
        except Exception as e:
            self.logger.error(f"Naive comparison failed: {e}")
            return {'att': np.nan, 'se': np.nan, 'n_treated': 0, 'n_control': 0}
            
    def _summarize_results(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Summarize matching results"""
        summary = {
            'method_comparison': {},
            'best_estimate': np.nan,
            'robustness_check': False
        }
        
        # Extract ATT estimates
        for method, result in results.items():
            if method != 'summary' and 'att' in result:
                att = result['att']
                se = result['se']
                n_matched = result.get('n_matched', 0)
                
                summary['method_comparison'][method] = {
                    'att': att,
                    'se': se,
                    'ci_lower': att - 1.96 * se if not np.isnan(att) and not np.isnan(se) else np.nan,
                    'ci_upper': att + 1.96 * se if not np.isnan(att) and not np.isnan(se) else np.nan,
                    'n_matched': n_matched
                }
                
        # Select best estimate (prefer propensity score if available)
        if 'propensity_score' in summary['method_comparison']:
            ps_result = summary['method_comparison']['propensity_score']
            if not np.isnan(ps_result['att']):
                summary['best_estimate'] = ps_result['att']
                
        # Robustness check: are estimates consistent across methods?
        valid_estimates = [
            result['att'] for result in summary['method_comparison'].values()
            if not np.isnan(result['att'])
        ]
        
        if len(valid_estimates) > 1:
            estimate_range = max(valid_estimates) - min(valid_estimates)
            summary['robustness_check'] = estimate_range < 2.0  # Within 2 years
            
        return summary
        
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'propensity_score': {'att': np.nan, 'se': np.nan, 'n_matched': 0},
            'nearest_neighbor': {'att': np.nan, 'se': np.nan, 'n_matched': 0},
            'exact_matching': {'att': np.nan, 'se': np.nan, 'n_matched': 0},
            'naive': {'att': np.nan, 'se': np.nan, 'n_treated': 0, 'n_control': 0},
            'summary': {'best_estimate': np.nan, 'robustness_check': False}
        }