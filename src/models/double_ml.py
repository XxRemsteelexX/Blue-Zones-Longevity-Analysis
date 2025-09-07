"""
Double Machine Learning for heterogeneous treatment effects
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

class DoubleMachineLearning:
    """Double ML estimator for causal inference with ML"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.results = {}
        
    def estimate_ate(self, data: pd.DataFrame, 
                     outcome: str, treatment: str, confounders: List[str],
                     ml_model_y=None, ml_model_t=None, 
                     cv_folds: int = 5) -> Dict[str, Any]:
        """
        Estimate Average Treatment Effect using Double ML
        
        Args:
            data: Panel data
            outcome: Outcome variable
            treatment: Treatment variable
            confounders: List of confounding variables
            ml_model_y: ML model for outcome (default: RandomForest)
            ml_model_t: ML model for treatment (default: RandomForest)
            cv_folds: Cross-validation folds
            
        Returns:
            ATE estimation results
        """
        
        # Default models
        if ml_model_y is None:
            ml_model_y = RandomForestRegressor(n_estimators=100, random_state=42)
        if ml_model_t is None:
            ml_model_t = RandomForestRegressor(n_estimators=100, random_state=42)
            
        # Prepare data
        clean_data = data[[outcome, treatment] + confounders].dropna()
        Y = clean_data[outcome].values
        T = clean_data[treatment].values
        X = clean_data[confounders].values
        
        # Step 1: Predict outcome using confounders (cross-fitted)
        Y_pred = cross_val_predict(ml_model_y, X, Y, cv=cv_folds)
        Y_residual = Y - Y_pred
        
        # Step 2: Predict treatment using confounders (cross-fitted)
        T_pred = cross_val_predict(ml_model_t, X, T, cv=cv_folds)
        T_residual = T - T_pred
        
        # Step 3: Estimate ATE using residualized variables
        ate_reg = LinearRegression()
        ate_reg.fit(T_residual.reshape(-1, 1), Y_residual)
        
        ate = ate_reg.coef_[0]
        
        # Calculate standard errors (simplified)
        n = len(Y)
        residuals = Y_residual - ate * T_residual
        mse = np.mean(residuals**2)
        var_t = np.var(T_residual)
        se = np.sqrt(mse / (n * var_t))
        
        # Confidence intervals
        t_critical = 1.96  # 95% CI
        ci_lower = ate - t_critical * se
        ci_upper = ate + t_critical * se
        
        results = {
            'ate': ate,
            'std_error': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': 2 * (1 - norm.cdf(abs(ate / se))),
            'n_obs': n,
            'cv_folds': cv_folds,
            'outcome': outcome,
            'treatment': treatment,
            'confounders': confounders
        }
        
        return results
    
    def estimate_cate(self, data: pd.DataFrame, outcome: str, treatment: str, 
                      confounders: List[str], effect_modifiers: List[str] = None,
                      ml_model=None) -> Dict[str, Any]:
        """
        Estimate Conditional Average Treatment Effects (CATE)
        
        Args:
            data: Panel data
            outcome: Outcome variable  
            treatment: Treatment variable
            confounders: Confounding variables
            effect_modifiers: Variables that modify treatment effect
            ml_model: ML model for CATE estimation
            
        Returns:
            CATE estimation results
        """
        
        if effect_modifiers is None:
            effect_modifiers = confounders[:3]  # Use first 3 confounders
            
        if ml_model is None:
            from sklearn.ensemble import GradientBoostingRegressor
            ml_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
        # Prepare data
        all_vars = [outcome, treatment] + confounders + effect_modifiers
        clean_data = data[all_vars].dropna().reset_index(drop=True)
        
        # Create interaction terms
        for modifier in effect_modifiers:
            if modifier in clean_data.columns:
                clean_data[f'{treatment}_x_{modifier}'] = (
                    clean_data[treatment] * clean_data[modifier]
                )
        
        # Features for CATE model
        cate_features = confounders + effect_modifiers + [f'{treatment}_x_{modifier}' for modifier in effect_modifiers]
        
        X = clean_data[cate_features]
        Y = clean_data[outcome]
        
        # Fit CATE model
        ml_model.fit(X, Y)
        
        # Predict individual treatment effects
        cate_pred = []
        for idx in clean_data.index:
            # Predict with treatment = 1
            X_treat = X.loc[idx:idx].copy()
            for modifier in effect_modifiers:
                X_treat[f'{treatment}_x_{modifier}'] = X_treat[modifier]
            y1_pred = ml_model.predict(X_treat)[0]
            
            # Predict with treatment = 0  
            X_control = X.loc[idx:idx].copy()
            for modifier in effect_modifiers:
                X_control[f'{treatment}_x_{modifier}'] = 0
            y0_pred = ml_model.predict(X_control)[0]
            
            cate_pred.append(y1_pred - y0_pred)
        
        clean_data['cate_pred'] = cate_pred
        
        # Summary statistics
        cate_mean = np.mean(cate_pred)
        cate_std = np.std(cate_pred)
        cate_min = np.min(cate_pred)
        cate_max = np.max(cate_pred)
        
        results = {
            'cate_predictions': cate_pred,
            'cate_mean': cate_mean,
            'cate_std': cate_std,
            'cate_min': cate_min,
            'cate_max': cate_max,
            'data_with_cate': clean_data,
            'model': ml_model,
            'effect_modifiers': effect_modifiers
        }
        
        return results

if __name__ == "__main__":
    # Example usage
    from scipy.stats import norm
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    
    # Confounders
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    
    # Treatment (depends on confounders)
    treatment_prob = 1 / (1 + np.exp(-(x1 + x2 + x3)))
    treatment = np.random.binomial(1, treatment_prob, n)
    
    # Outcome (with heterogeneous treatment effects)
    outcome = (2 + x1 + x2 + x3 + 
              treatment * (1 + 0.5 * x1) +  # Treatment effect varies with x1
              np.random.normal(0, 1, n))
    
    data = pd.DataFrame({
        'outcome': outcome,
        'treatment': treatment,
        'x1': x1,
        'x2': x2, 
        'x3': x3
    })
    
    # Run Double ML
    dml = DoubleMachineLearning()
    
    # Estimate ATE
    ate_results = dml.estimate_ate(
        data, 'outcome', 'treatment', ['x1', 'x2', 'x3']
    )
    
    print("Double ML ATE Results:")
    print(f"ATE: {ate_results['ate']:.3f} (SE: {ate_results['std_error']:.3f})")
    print(f"95% CI: [{ate_results['ci_lower']:.3f}, {ate_results['ci_upper']:.3f}]")
    
    # Estimate CATE
    cate_results = dml.estimate_cate(
        data, 'outcome', 'treatment', ['x1', 'x2', 'x3'], ['x1']
    )
    
    print(f"\nCATE Results:")
    print(f"Mean CATE: {cate_results['cate_mean']:.3f}")
    print(f"CATE Range: [{cate_results['cate_min']:.3f}, {cate_results['cate_max']:.3f}]")