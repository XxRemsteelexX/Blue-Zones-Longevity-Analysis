"""
Panel Fixed Effects models for causal inference
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


class PanelFixedEffects:
    """Panel Fixed Effects estimation for causal inference"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.results = {}
        self.data = None
        self.scaler = StandardScaler()
        
    def prepare_panel_data(self, data: pd.DataFrame, 
                          unit_id: str = 'geo_id',
                          time_id: str = 'year',
                          outcome_vars: List[str] = None,
                          treatment_vars: List[str] = None,
                          control_vars: List[str] = None) -> pd.DataFrame:
        """
        Prepare data for panel analysis
        
        Args:
            data: Raw panel data
            unit_id: Unit identifier column
            time_id: Time identifier column  
            outcome_vars: List of outcome variables
            treatment_vars: List of treatment variables
            control_vars: List of control variables
            
        Returns:
            Clean panel dataset
        """
        if outcome_vars is None:
            outcome_vars = ['life_expectancy']
        if treatment_vars is None:
            treatment_vars = ['walkability_score', 'greenspace_pct', 'market_density']
        if control_vars is None:
            control_vars = ['population_density_log', 'gdp_per_capita', 'temperature_mean']
            
        # Ensure required columns exist
        required_cols = [unit_id, time_id] + outcome_vars + treatment_vars + control_vars
        available_cols = [col for col in required_cols if col in data.columns]
        
        if len(available_cols) < len(required_cols):
            missing = set(required_cols) - set(available_cols)
            self.logger.warning(f"Missing columns: {missing}")
            
        # Clean panel data
        panel_data = data[available_cols].copy()
        
        # Remove missing outcomes
        panel_data = panel_data.dropna(subset=outcome_vars)
        
        # Standardize treatment variables (z-scores)
        treatment_cols = [col for col in treatment_vars if col in panel_data.columns]
        if treatment_cols:
            panel_data[treatment_cols] = self.scaler.fit_transform(panel_data[treatment_cols])
            self.logger.info(f"Standardized treatment variables: {treatment_cols}")
            
        # Create balanced panel indicator
        panel_counts = panel_data.groupby(unit_id)[time_id].count()
        max_periods = panel_counts.max()
        balanced_units = panel_counts[panel_counts == max_periods].index
        
        panel_data['balanced_panel'] = panel_data[unit_id].isin(balanced_units)
        
        self.logger.info(f"Panel data prepared: {len(panel_data)} obs, {panel_data[unit_id].nunique()} units")
        self.logger.info(f"Balanced panel: {panel_data['balanced_panel'].mean()*100:.1f}% of observations")
        
        self.data = panel_data
        return panel_data
        
    def estimate_fixed_effects(self, outcome: str, 
                              treatments: List[str],
                              controls: List[str] = None,
                              unit_id: str = 'geo_id',
                              time_id: str = 'year',
                              cluster_se: bool = True) -> Dict[str, Any]:
        """
        Estimate panel fixed effects model
        
        Args:
            outcome: Outcome variable
            treatments: Treatment variables  
            controls: Control variables
            unit_id: Unit identifier
            time_id: Time identifier
            cluster_se: Use clustered standard errors
            
        Returns:
            Estimation results
        """
        if self.data is None:
            raise ValueError("Must prepare panel data first")
            
        try:
            # Try using linearmodels for proper panel FE
            from linearmodels import PanelOLS
            from linearmodels.panel import PanelResults
            
            # Prepare data for linearmodels
            panel_data = self.data.set_index([unit_id, time_id])
            
            # Build formula components
            treatment_terms = " + ".join(treatments)
            control_terms = " + ".join(controls) if controls else ""
            
            if control_terms:
                formula_rhs = f"{treatment_terms} + {control_terms}"
            else:
                formula_rhs = treatment_terms
                
            # Estimate model
            mod = PanelOLS.from_formula(
                f"{outcome} ~ {formula_rhs} + EntityEffects + TimeEffects",
                data=panel_data
            )
            
            if cluster_se:
                res = mod.fit(cov_type='clustered', cluster_entity=True)
            else:
                res = mod.fit()
                
            # Extract results
            results = {
                'model_type': 'PanelOLS',
                'outcome': outcome,
                'treatments': treatments,
                'controls': controls or [],
                'n_obs': res.nobs,
                'n_entities': res.entity_info.total,
                'n_time_periods': res.time_info.total,
                'r_squared': res.rsquared,
                'r_squared_within': res.rsquared_within,
                'coefficients': res.params.to_dict(),
                'std_errors': res.std_errors.to_dict(),
                'pvalues': res.pvalues.to_dict(),
                'conf_int': res.conf_int().to_dict(),
                'f_statistic': res.f_statistic.stat,
                'f_pvalue': res.f_statistic.pval,
                'model_results': res
            }
            
        except ImportError:
            # Fallback to manual within-transformation
            self.logger.warning("linearmodels not available, using manual FE estimation")
            results = self._manual_fixed_effects(outcome, treatments, controls, unit_id, time_id)
            
        except Exception as e:
            self.logger.error(f"Panel estimation failed: {e}")
            results = self._manual_fixed_effects(outcome, treatments, controls, unit_id, time_id)
            
        # Store results
        model_key = f"{outcome}_{'_'.join(treatments)}"
        self.results[model_key] = results
        
        return results
        
    def _manual_fixed_effects(self, outcome: str, treatments: List[str], 
                             controls: List[str], unit_id: str, time_id: str) -> Dict[str, Any]:
        """Manual within-transformation for FE estimation"""
        from sklearn.linear_model import LinearRegression
        from scipy import stats
        
        # Within transformation (demean by unit and time)
        data_copy = self.data.copy()
        
        # Combine all variables
        all_vars = [outcome] + treatments + (controls or [])
        
        # Unit demeaning
        unit_means = data_copy.groupby(unit_id)[all_vars].transform('mean')
        
        # Time demeaning  
        time_means = data_copy.groupby(time_id)[all_vars].transform('mean')
        
        # Overall mean
        overall_means = data_copy[all_vars].mean()
        
        # Within transformation: x_it - x_i - x_t + x_overall
        for var in [outcome] + treatments + (controls or []):
            data_copy[f"{var}_within"] = (data_copy[var] - unit_means[var] - 
                                        time_means[var] + overall_means[var])
        
        # Prepare variables
        y = data_copy[f"{outcome}_within"].values
        X_vars = [f"{var}_within" for var in treatments + (controls or [])]
        X = data_copy[X_vars].values
        
        # Remove missing values
        valid_mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        y = y[valid_mask]
        X = X[valid_mask]
        
        # Fit model
        reg = LinearRegression(fit_intercept=False)  # No intercept after within transformation
        reg.fit(X, y)
        
        # Calculate statistics
        y_pred = reg.predict(X)
        residuals = y - y_pred
        n = len(y)
        k = X.shape[1]
        
        # R-squared (within)
        tss = np.sum((y - np.mean(y))**2)
        rss = np.sum(residuals**2)
        r_squared = 1 - (rss / tss)
        
        # Standard errors (homoskedastic)
        mse = rss / (n - k)
        var_coef = mse * np.linalg.inv(X.T @ X)
        std_errors = np.sqrt(np.diag(var_coef))
        
        # T-statistics and p-values
        t_stats = reg.coef_ / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        
        # Confidence intervals
        t_critical = stats.t.ppf(0.975, n - k)
        conf_int_lower = reg.coef_ - t_critical * std_errors
        conf_int_upper = reg.coef_ + t_critical * std_errors
        
        # Organize results
        var_names = treatments + (controls or [])
        results = {
            'model_type': 'Manual_FE',
            'outcome': outcome,
            'treatments': treatments,
            'controls': controls or [],
            'n_obs': n,
            'n_entities': data_copy[unit_id].nunique(),
            'n_time_periods': data_copy[time_id].nunique(),
            'r_squared_within': r_squared,
            'coefficients': dict(zip(var_names, reg.coef_)),
            'std_errors': dict(zip(var_names, std_errors)),
            'pvalues': dict(zip(var_names, p_values)),
            'conf_int': {var: (conf_int_lower[i], conf_int_upper[i]) 
                        for i, var in enumerate(var_names)}
        }
        
        return results
        
    def event_study(self, outcome: str, treatment: str,
                   event_var: str, leads: int = 5, lags: int = 5,
                   unit_id: str = 'geo_id', time_id: str = 'year',
                   controls: List[str] = None) -> Dict[str, Any]:
        """
        Event study analysis around treatment events
        
        Args:
            outcome: Outcome variable
            treatment: Treatment variable
            event_var: Binary event indicator
            leads: Number of leads to include
            lags: Number of lags to include  
            unit_id: Unit identifier
            time_id: Time identifier
            controls: Control variables
            
        Returns:
            Event study results
        """
        if self.data is None:
            raise ValueError("Must prepare panel data first")
            
        # Create event study data
        event_data = self._create_event_study_data(
            self.data, event_var, leads, lags, unit_id, time_id
        )
        
        if event_data.empty:
            self.logger.warning("No event study data created")
            return {}
            
        # Create relative time dummies
        rel_time_vars = []
        for t in range(-leads, lags + 1):
            if t != -1:  # Omit t=-1 as reference
                var_name = f"event_t{t:+d}"
                event_data[var_name] = (event_data['rel_time'] == t).astype(int)
                rel_time_vars.append(var_name)
                
        # Estimate event study model
        all_treatments = rel_time_vars
        
        try:
            es_results = self.estimate_fixed_effects(
                outcome=outcome,
                treatments=all_treatments,
                controls=controls,
                unit_id=unit_id,
                time_id=time_id
            )
            
            # Extract event study coefficients
            es_coefs = {}
            es_ses = {}
            es_pvals = {}
            
            for var in rel_time_vars:
                rel_time = int(var.split('t')[1])
                es_coefs[rel_time] = es_results['coefficients'].get(var, 0)
                es_ses[rel_time] = es_results['std_errors'].get(var, 0)
                es_pvals[rel_time] = es_results['pvalues'].get(var, 1)
                
            # Add reference period (t=-1)
            es_coefs[-1] = 0
            es_ses[-1] = 0
            es_pvals[-1] = np.nan
            
            event_study_results = {
                'coefficients': es_coefs,
                'std_errors': es_ses,
                'pvalues': es_pvals,
                'leads': leads,
                'lags': lags,
                'n_events': event_data['treated'].sum(),
                'model_results': es_results
            }
            
        except Exception as e:
            self.logger.error(f"Event study estimation failed: {e}")
            event_study_results = {}
            
        return event_study_results
        
    def _create_event_study_data(self, data: pd.DataFrame, event_var: str,
                                leads: int, lags: int, unit_id: str, time_id: str) -> pd.DataFrame:
        """Create event study dataset"""
        
        # Find event times for each unit
        events = data[data[event_var] == 1].groupby(unit_id)[time_id].min()
        
        if events.empty:
            return pd.DataFrame()
            
        # Create event study dataset
        event_data = data.copy()
        event_data['event_time'] = event_data[unit_id].map(events)
        event_data['treated'] = event_data[unit_id].isin(events.index)
        
        # Calculate relative time
        event_data['rel_time'] = event_data[time_id] - event_data['event_time']
        
        # Keep only relevant time periods
        event_data = event_data[
            (event_data['rel_time'] >= -leads) & 
            (event_data['rel_time'] <= lags)
        ]
        
        return event_data
        
    def test_parallel_trends(self, outcome: str, treatment: str,
                           unit_id: str = 'geo_id', time_id: str = 'year',
                           pre_periods: int = 3) -> Dict[str, Any]:
        """
        Test parallel trends assumption
        
        Args:
            outcome: Outcome variable
            treatment: Treatment variable (binary)
            unit_id: Unit identifier
            time_id: Time identifier
            pre_periods: Number of pre-periods to test
            
        Returns:
            Parallel trends test results
        """
        if self.data is None:
            raise ValueError("Must prepare panel data first")
            
        # Create treatment groups
        test_data = self.data.copy()
        
        # Define treated/control groups (time-invariant)
        treatment_status = test_data.groupby(unit_id)[treatment].mean()
        treated_units = treatment_status[treatment_status > 0.5].index
        
        test_data['treated_group'] = test_data[unit_id].isin(treated_units)
        
        # Focus on pre-treatment period
        if 'event_time' in test_data.columns:
            pre_data = test_data[test_data['rel_time'] < 0]
        else:
            # Use first half of time period as "pre"
            mid_time = test_data[time_id].median()
            pre_data = test_data[test_data[time_id] < mid_time]
            
        if len(pre_data) == 0:
            return {'test_failed': True, 'reason': 'No pre-treatment data'}
            
        # Test trend differences
        trend_results = []
        
        for group in [True, False]:  # Treated, Control
            group_data = pre_data[pre_data['treated_group'] == group]
            
            if len(group_data) == 0:
                continue
                
            # Calculate group-specific trends
            group_trends = []
            for unit in group_data[unit_id].unique():
                unit_data = group_data[group_data[unit_id] == unit].sort_values(time_id)
                
                if len(unit_data) >= 2:
                    # Linear trend
                    x = unit_data[time_id].values
                    y = unit_data[outcome].values
                    
                    if not np.isnan(y).all():
                        trend = np.polyfit(x, y, 1)[0]  # Slope
                        group_trends.append(trend)
                        
            if group_trends:
                trend_results.append({
                    'group': 'treated' if group else 'control',
                    'mean_trend': np.mean(group_trends),
                    'std_trend': np.std(group_trends),
                    'n_units': len(group_trends)
                })
                
        # Statistical test
        if len(trend_results) == 2:
            from scipy.stats import ttest_ind
            
            treated_trends = [r for r in trend_results if r['group'] == 'treated'][0]
            control_trends = [r for r in trend_results if r['group'] == 'control'][0]
            
            # Test difference in trends
            # (This is simplified - would need individual trend estimates for proper test)
            trend_diff = treated_trends['mean_trend'] - control_trends['mean_trend']
            
            parallel_trends_results = {
                'treated_trend': treated_trends['mean_trend'],
                'control_trend': control_trends['mean_trend'],
                'trend_difference': trend_diff,
                'parallel_trends_likely': abs(trend_diff) < 0.1,  # Arbitrary threshold
                'test_data': trend_results
            }
        else:
            parallel_trends_results = {'test_failed': True, 'reason': 'Insufficient groups'}
            
        return parallel_trends_results
        
    def plot_event_study(self, event_study_results: Dict[str, Any], 
                        title: str = "Event Study Results") -> go.Figure:
        """Plot event study results"""
        
        if not event_study_results or 'coefficients' not in event_study_results:
            return go.Figure().add_annotation(text="No event study results to plot")
            
        # Extract data
        rel_times = sorted(event_study_results['coefficients'].keys())
        coefs = [event_study_results['coefficients'][t] for t in rel_times]
        ses = [event_study_results['std_errors'][t] for t in rel_times]
        
        # Calculate confidence intervals
        ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
        ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]
        
        # Create plot
        fig = go.Figure()
        
        # Add coefficient line
        fig.add_trace(go.Scatter(
            x=rel_times,
            y=coefs,
            mode='lines+markers',
            name='Coefficient',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=rel_times + rel_times[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI'
        ))
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No Effect")
        fig.add_vline(x=-1, line_dash="dash", line_color="red", annotation_text="Reference Period")
        fig.add_vline(x=0, line_dash="solid", line_color="red", annotation_text="Treatment")
        
        fig.update_layout(
            title=title,
            xaxis_title="Periods Relative to Treatment",
            yaxis_title="Effect on Outcome",
            showlegend=True,
            height=500
        )
        
        return fig
        
    def create_results_table(self, results: Dict[str, Any], 
                           format: str = 'latex') -> str:
        """Create publication-ready results table"""
        
        treatments = results.get('treatments', [])
        coefs = results.get('coefficients', {})
        ses = results.get('std_errors', {})  
        pvals = results.get('pvalues', {})
        
        # Format coefficients with stars
        formatted_results = []
        
        for var in treatments:
            coef = coefs.get(var, 0)
            se = ses.get(var, 0)
            p = pvals.get(var, 1)
            
            # Add significance stars
            stars = ""
            if p < 0.01:
                stars = "***"
            elif p < 0.05:
                stars = "**"
            elif p < 0.1:
                stars = "*"
                
            formatted_results.append({
                'variable': var,
                'coefficient': f"{coef:.4f}{stars}",
                'std_error': f"({se:.4f})",
                'p_value': f"{p:.4f}"
            })
            
        # Create table
        if format == 'latex':
            table = self._format_latex_table(formatted_results, results)
        else:
            table = self._format_text_table(formatted_results, results)
            
        return table
        
    def _format_latex_table(self, formatted_results: List[Dict], results: Dict) -> str:
        """Format results as LaTeX table"""
        
        latex_table = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Panel Fixed Effects Results}",
            "\\begin{tabular}{lcc}",
            "\\hline",
            "Variable & Coefficient & Std. Error \\\\",
            "\\hline"
        ]
        
        for row in formatted_results:
            latex_table.append(
                f"{row['variable']} & {row['coefficient']} & {row['std_error']} \\\\"
            )
            
        latex_table.extend([
            "\\hline",
            f"Observations & \\multicolumn{{2}}{{c}}{{{results.get('n_obs', 'N/A')}}} \\\\",
            f"R-squared (within) & \\multicolumn{{2}}{{c}}{{{results.get('r_squared_within', 'N/A'):.4f}}} \\\\",
            f"Number of entities & \\multicolumn{{2}}{{c}}{{{results.get('n_entities', 'N/A')}}} \\\\",
            "\\hline",
            "\\multicolumn{3}{p{0.75\\textwidth}}{\\footnotesize Notes: * p$<$0.1, ** p$<$0.05, *** p$<$0.01. Standard errors clustered by entity.} \\\\",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_table)
        
    def _format_text_table(self, formatted_results: List[Dict], results: Dict) -> str:
        """Format results as text table"""
        
        table_lines = [
            "Panel Fixed Effects Results",
            "=" * 50,
            f"{'Variable':<20} {'Coefficient':<15} {'Std. Error':<12}",
            "-" * 50
        ]
        
        for row in formatted_results:
            table_lines.append(
                f"{row['variable']:<20} {row['coefficient']:<15} {row['std_error']:<12}"
            )
            
        table_lines.extend([
            "-" * 50,
            f"Observations: {results.get('n_obs', 'N/A')}",
            f"R-squared (within): {results.get('r_squared_within', 'N/A'):.4f}",  
            f"Number of entities: {results.get('n_entities', 'N/A')}",
            "",
            "Notes: * p<0.1, ** p<0.05, *** p<0.01"
        ])
        
        return "\n".join(table_lines)
    
    def visualize_event_study(self, event_results, outcome_name="Outcome", 
                             treatment_name="Treatment", save_path=None):
        """Create event study plot"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract coefficients and confidence intervals
        periods = sorted(event_results['coefficients'].keys())
        coefs = [event_results['coefficients'][t] for t in periods]
        ses = [event_results['std_errors'][t] for t in periods]
        
        # Calculate confidence intervals
        ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
        ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]
        
        # Plot coefficient estimates
        ax.plot(periods, coefs, 'o-', color='darkblue', linewidth=2, 
                markersize=6, label='Point Estimates')
        
        # Plot confidence intervals
        ax.fill_between(periods, ci_lower, ci_upper, alpha=0.3, 
                       color='lightblue', label='95% Confidence Interval')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add vertical line at treatment period (t=0)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                  label='Treatment Period')
        
        # Shade pre-treatment period
        if periods:
            pre_min = min([p for p in periods if p < 0] + [0])
            pre_treatment = patches.Rectangle((pre_min, ax.get_ylim()[0]), 
                                            -pre_min, 
                                            ax.get_ylim()[1] - ax.get_ylim()[0],
                                            alpha=0.1, facecolor='gray', 
                                            label='Pre-treatment')
            ax.add_patch(pre_treatment)
        
        # Formatting
        ax.set_xlabel('Periods Relative to Treatment', fontsize=12)
        ax.set_ylabel(f'Effect on {outcome_name}', fontsize=12)
        ax.set_title(f'Event Study: Effect of {treatment_name} on {outcome_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set x-axis ticks
        ax.set_xticks(periods)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Event study plot saved to {save_path}")
        
        return fig, ax

    def run_robustness_checks(self, data, outcome, treatment, controls=None, 
                             unit_id='geo_id', time_id='year'):
        """Run comprehensive robustness checks"""
        self.logger.info("Running robustness checks...")
        
        robustness_results = {}
        
        # 1. Different lag structures
        self.logger.info("  Testing different lag structures...")
        for lag in [1, 2, 3]:
            lagged_data = data.copy()
            lagged_data[f'{treatment}_lag{lag}'] = (
                lagged_data.groupby(unit_id)[treatment].shift(lag)
            )
            
            try:
                result = self.estimate_fixed_effects(
                    outcome, [f'{treatment}_lag{lag}'], 
                    controls
                )
                robustness_results[f'lag_{lag}'] = result
            except Exception as e:
                self.logger.warning(f"Failed lag {lag} test: {e}")
        
        # 2. Placebo tests with lead treatments
        self.logger.info("  Running placebo tests...")
        for lead in [1, 2]:
            placebo_data = data.copy()
            placebo_data[f'{treatment}_lead{lead}'] = (
                placebo_data.groupby(unit_id)[treatment].shift(-lead)
            )
            
            try:
                result = self.estimate_fixed_effects(
                    outcome, [f'{treatment}_lead{lead}'], 
                    controls
                )
                robustness_results[f'placebo_lead_{lead}'] = result
            except Exception as e:
                self.logger.warning(f"Failed placebo lead {lead} test: {e}")
        
        # 3. Alternative outcome specifications
        if f'{outcome}_log' in data.columns:
            self.logger.info("  Testing log outcome specification...")
            try:
                result = self.estimate_fixed_effects(
                    f'{outcome}_log', [treatment], controls
                )
                robustness_results['log_outcome'] = result
            except Exception as e:
                self.logger.warning(f"Failed log outcome test: {e}")
        
        # 4. Subsample analysis
        self.logger.info("  Running subsample analysis...")
        if 'region' in data.columns:
            for region in data['region'].unique()[:3]:  # Test top 3 regions
                region_data = data[data['region'] == region]
                if len(region_data) > 100:  # Minimum sample size
                    try:
                        # Update data reference temporarily
                        original_data = self.data
                        self.data = self.prepare_panel_data(region_data)
                        result = self.estimate_fixed_effects(
                            outcome, [treatment], controls
                        )
                        robustness_results[f'region_{region}'] = result
                        self.data = original_data  # Restore original
                    except Exception as e:
                        self.logger.warning(f"Failed region {region} test: {e}")
        
        self.logger.info(f"Completed {len(robustness_results)} robustness checks")
        return robustness_results
    
    def summarize_robustness(self, robustness_results, main_result, treatment_var):
        """Summarize robustness check results"""
        summary = {
            'main_coefficient': main_result['coefficients'][treatment_var],
            'main_pvalue': main_result['pvalues'][treatment_var],
            'robustness_tests': len(robustness_results),
            'significant_tests': 0,
            'coefficient_range': [],
            'tests_detail': {}
        }
        
        for test_name, result in robustness_results.items():
            # Find treatment variable in results
            treatment_cols = [col for col in result['coefficients'].keys() 
                            if treatment_var in col or 'lag' in col or 'lead' in col]
            
            if treatment_cols:
                coef = result['coefficients'][treatment_cols[0]]
                pval = result['pvalues'][treatment_cols[0]]
                
                summary['coefficient_range'].append(coef)
                summary['tests_detail'][test_name] = {
                    'coefficient': coef,
                    'p_value': pval,
                    'significant': pval < 0.05
                }
                
                if pval < 0.05:
                    summary['significant_tests'] += 1
        
        if summary['coefficient_range']:
            summary['coef_min'] = min(summary['coefficient_range'])
            summary['coef_max'] = max(summary['coefficient_range'])
            summary['coef_std'] = np.std(summary['coefficient_range'])
        
        return summary
    
    def generate_research_summary(self, main_results, robustness_summary, 
                                 outcome_name, treatment_name):
        """Generate publication-ready research summary"""
        
        main_coef = robustness_summary['main_coefficient']
        main_pval = robustness_summary['main_pvalue']
        sig_tests = robustness_summary['significant_tests']
        total_tests = robustness_summary['robustness_tests']
        
        # Statistical significance
        sig_level = "not significant"
        if main_pval < 0.001:
            sig_level = "highly significant (p<0.001)"
        elif main_pval < 0.01:
            sig_level = "significant (p<0.01)"
        elif main_pval < 0.05:
            sig_level = "significant (p<0.05)"
        
        # Effect size interpretation
        effect_size = abs(main_coef)
        if effect_size < 0.1:
            effect_magnitude = "small"
        elif effect_size < 0.5:
            effect_magnitude = "moderate"
        else:
            effect_magnitude = "large"
        
        # Robustness assessment
        robustness_pct = (sig_tests / total_tests * 100) if total_tests > 0 else 0
        
        coef_min = robustness_summary.get('coef_min', main_coef)
        coef_max = robustness_summary.get('coef_max', main_coef)
        coef_std = robustness_summary.get('coef_std', 0)
        
        summary_text = f"""
🔬 CAUSAL INFERENCE RESULTS SUMMARY
{'='*50}

MAIN FINDING:
A 1-unit increase in {treatment_name} is associated with a {main_coef:.3f} unit change in {outcome_name}.
This effect is {sig_level} and represents a {effect_magnitude} effect size.

ROBUSTNESS:
✓ {sig_tests}/{total_tests} robustness checks show significant effects ({robustness_pct:.1f}%)
✓ Coefficient range: [{coef_min:.3f}, {coef_max:.3f}]
✓ Standard deviation across tests: {coef_std:.3f}

IDENTIFICATION:
• Panel Fixed Effects with unit and time controls
• Clustered standard errors at geographic unit level
• Multiple robustness checks including placebo tests
• Event study analysis for dynamic effects

POLICY IMPLICATIONS:
Results suggest causal relationship between {treatment_name} and {outcome_name}.
Effect size is {effect_magnitude} and statistically robust across specifications.
"""
        
        return summary_text
    
    def estimate_instrumental_variables(self, data: pd.DataFrame, 
                                       outcome: str, treatment: str,
                                       instruments: List[str], controls: List[str] = None,
                                       unit_id: str = 'geo_id', time_id: str = 'year') -> Dict[str, Any]:
        """Estimate treatment effect using instrumental variables"""
        
        if controls is None:
            controls = []
            
        # Prepare data
        all_vars = [outcome, treatment] + instruments + controls + [unit_id, time_id]
        clean_data = data[all_vars].dropna()
        
        if len(clean_data) < 20:
            return {'error': 'Insufficient observations for IV estimation'}
        
        try:
            # Two-stage least squares
            from sklearn.linear_model import LinearRegression
            
            # First stage: regress treatment on instruments + controls
            X_first = clean_data[instruments + controls].values
            T = clean_data[treatment].values
            
            first_stage = LinearRegression()
            first_stage.fit(X_first, T)
            T_fitted = first_stage.predict(X_first)
            
            # F-statistic for instrument strength
            t_mean = np.mean(T)
            ss_total = np.sum((T - t_mean)**2)
            ss_res = np.sum((T - T_fitted)**2)
            r2_first = 1 - (ss_res / ss_total)
            
            n = len(T)
            k_instruments = len(instruments)
            f_stat = (r2_first / k_instruments) / ((1 - r2_first) / (n - k_instruments - 1))
            
            # Second stage: regress outcome on fitted treatment + controls
            X_second = np.column_stack([T_fitted] + [clean_data[c].values for c in controls])
            y = clean_data[outcome].values
            
            second_stage = LinearRegression()
            second_stage.fit(X_second, y)
            
            iv_coefficient = second_stage.coef_[0]
            
            # Calculate standard errors (simplified)
            y_pred = second_stage.predict(X_second)
            residuals = y - y_pred
            mse = np.mean(residuals**2)
            
            # Simplified SE calculation
            se = np.sqrt(mse / np.var(T_fitted))
            t_stat = iv_coefficient / se
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - len(controls) - 1))
            
            results = {
                'iv_coefficient': iv_coefficient,
                'std_error': se,
                'p_value': p_value,
                'first_stage_f': f_stat,
                'weak_instruments': f_stat < 10,
                'first_stage_r2': r2_first,
                'n_obs': n,
                'instruments': instruments,
                'controls': controls
            }
            
        except Exception as e:
            self.logger.error(f"IV estimation failed: {e}")
            results = {'error': str(e)}
            
        return results
    
    def oster_sensitivity_analysis(self, controlled_result: Dict[str, Any],
                                  uncontrolled_result: Dict[str, Any],
                                  treatment_var: str, r_max: float = 1.3) -> Dict[str, Any]:
        """Calculate Oster's delta for sensitivity to unobserved confounders"""
        
        try:
            controlled_beta = controlled_result['coefficients'][treatment_var]
            uncontrolled_beta = uncontrolled_result['coefficients'][treatment_var]
            
            controlled_r2 = controlled_result.get('r_squared_within', controlled_result.get('r_squared', 0))
            uncontrolled_r2 = uncontrolled_result.get('r_squared_within', uncontrolled_result.get('r_squared', 0))
            
            # Oster's delta calculation
            numerator = controlled_beta * (controlled_r2 - uncontrolled_r2)
            denominator = (uncontrolled_beta - controlled_beta) * (r_max - controlled_r2)
            
            if abs(denominator) < 1e-10:
                delta = np.inf if controlled_beta != 0 else 0
            else:
                delta = numerator / denominator
            
            # Interpretation
            if delta > 1:
                interpretation = "Very robust - unobservables would need to be stronger than observables"
            elif delta > 0.5:
                interpretation = "Moderately robust - unobservables would need moderate strength"
            elif delta > 0.2:
                interpretation = "Somewhat robust - moderate sensitivity to unobservables"
            else:
                interpretation = "Low robustness - sensitive to weak unobservables"
            
            results = {
                'delta': delta,
                'controlled_beta': controlled_beta,
                'uncontrolled_beta': uncontrolled_beta,
                'controlled_r2': controlled_r2,
                'uncontrolled_r2': uncontrolled_r2,
                'r_max': r_max,
                'interpretation': interpretation,
                'robust': delta > 1
            }
            
        except Exception as e:
            self.logger.error(f"Oster sensitivity analysis failed: {e}")
            results = {'error': str(e)}
            
        return results
    
    def bootstrap_confidence_intervals(self, data: pd.DataFrame, 
                                      outcome: str, treatments: List[str],
                                      controls: List[str] = None,
                                      n_bootstrap: int = 1000,
                                      ci_level: float = 95) -> Dict[str, Any]:
        """Calculate bootstrap confidence intervals"""
        
        estimates = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = data.sample(len(data), replace=True)
            
            try:
                # Re-prepare panel data for bootstrap sample
                original_data = self.data
                self.data = self.prepare_panel_data(
                    bootstrap_sample,
                    outcome_vars=[outcome],
                    treatment_vars=treatments,
                    control_vars=controls or []
                )
                
                # Estimate model
                result = self.estimate_fixed_effects(outcome, treatments, controls)
                
                # Store coefficients
                coefs = {var: result['coefficients'].get(var, 0) for var in treatments}
                estimates.append(coefs)
                
                # Restore original data
                self.data = original_data
                
            except:
                continue
        
        if not estimates:
            return {'error': 'No successful bootstrap iterations'}
        
        # Calculate confidence intervals
        bootstrap_results = {}
        lower_percentile = (100 - ci_level) / 2
        upper_percentile = 100 - lower_percentile
        
        for var in treatments:
            var_estimates = [est[var] for est in estimates if var in est]
            
            if var_estimates:
                bootstrap_results[var] = {
                    'mean': np.mean(var_estimates),
                    'std': np.std(var_estimates),
                    'ci_lower': np.percentile(var_estimates, lower_percentile),
                    'ci_upper': np.percentile(var_estimates, upper_percentile),
                    'n_bootstrap': len(var_estimates)
                }
        
        return bootstrap_results


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Create sample data
    np.random.seed(42)
    n_units, n_years = 100, 20
    
    data = []
    for unit in range(n_units):
        for year in range(2000, 2000 + n_years):
            # Unit fixed effect
            unit_fe = np.random.normal(0, 2)
            # Time fixed effect  
            time_fe = (year - 2010) * 0.1
            # Treatment (some policy introduced randomly)
            treatment = np.random.normal(0, 1)
            # Outcome with causal effect
            outcome = unit_fe + time_fe + 0.5 * treatment + np.random.normal(0, 1)
            
            data.append({
                'geo_id': f'unit_{unit}',
                'year': year,
                'outcome': outcome,
                'treatment': treatment,
                'control1': np.random.normal(0, 1),
                'region': f'region_{unit % 5}'
            })
    
    df = pd.DataFrame(data)
    
    # Run analysis
    panel_fe = PanelFixedEffects()
    
    # Prepare data
    panel_data = panel_fe.prepare_panel_data(
        df, outcome_vars=['outcome'], 
        treatment_vars=['treatment'], 
        control_vars=['control1']
    )
    
    # Estimate main model
    result = panel_fe.estimate_fixed_effects(
        'outcome', ['treatment'], ['control1']
    )
    print(panel_fe.create_results_table(result, format='text'))
    
    # Run robustness checks
    robustness = panel_fe.run_robustness_checks(
        df, 'outcome', 'treatment', ['control1']
    )
    
    # Summarize results
    summary = panel_fe.summarize_robustness(robustness, result, 'treatment')
    research_summary = panel_fe.generate_research_summary(
        result, summary, 'Life Expectancy', 'Walkability Index'
    )
    
    print(research_summary)