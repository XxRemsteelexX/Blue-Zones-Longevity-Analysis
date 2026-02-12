#!/usr/bin/env python3
"""
ML-Based Life Expectancy Prediction and Country Overperformance Analysis.

Trains regularized ML models on cross-sectional country features to predict
life expectancy, then uses residuals (actual - predicted) to identify countries
that significantly outperform or underperform their predicted life expectancy.

Countries with large positive residuals are "hidden Blue Zones" -- places where
unmeasured factors (culture, diet, social cohesion) may boost longevity beyond
what standard indicators predict.

All evaluation uses LOOCV (Leave-One-Out Cross-Validation), appropriate for n=93.
"""

import pandas as pd
import numpy as np
import os
import logging
import warnings

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLUE_ZONE_ISOS = {'USA', 'JPN', 'ITA', 'GRC', 'CRI'}

# Target variable
TARGET = 'life_expectancy'

# Columns that must never be used as features
META_COLS = [
    'iso_code', 'country_name', 'is_blue_zone', 'latitude', 'longitude',
    'data_year', 'life_expectancy', 'life_expectancy_female',
    'life_expectancy_male', 'life_expectancy_who',
]


# ---------------------------------------------------------------------------
# Feature Selection Pipeline
# ---------------------------------------------------------------------------
def coverage_filter(df, feature_cols, threshold=0.50):
    """Drop features with less than threshold non-null coverage."""
    n = len(df)
    report = []
    kept = []

    for col in feature_cols:
        pct = df[col].notna().sum() / n
        if pct < threshold:
            report.append({
                'feature': col,
                'stage_dropped': 'coverage',
                'reason': f'Only {pct:.1%} non-null (threshold: {threshold:.0%})',
                'coverage_pct': round(pct * 100, 1),
            })
            logger.info(f"  Dropped (coverage): {col} ({pct:.1%})")
        else:
            kept.append(col)
            report.append({
                'feature': col,
                'stage_dropped': 'kept',
                'reason': 'Passed coverage filter',
                'coverage_pct': round(pct * 100, 1),
            })

    return kept, report


def correlation_filter(df, feature_cols, target_col, threshold=0.85):
    """Among highly correlated pairs, drop the one less correlated with target."""
    corr_matrix = df[feature_cols].corr().abs()
    target_corr = df[feature_cols].corrwith(df[target_col]).abs()

    dropped = set()
    report = []

    for i in range(len(feature_cols)):
        if feature_cols[i] in dropped:
            continue
        for j in range(i + 1, len(feature_cols)):
            if feature_cols[j] in dropped:
                continue
            pair_corr = corr_matrix.iloc[i, j]
            if pair_corr > threshold:
                fi, fj = feature_cols[i], feature_cols[j]
                # Drop the one with lower univariate correlation to target
                if target_corr.get(fi, 0) < target_corr.get(fj, 0):
                    drop = fi
                else:
                    drop = fj
                dropped.add(drop)
                keep = fi if drop == fj else fj
                report.append({
                    'feature': drop,
                    'stage_dropped': 'correlation',
                    'reason': f'|r|={pair_corr:.3f} with {keep}; lower target corr',
                    'coverage_pct': np.nan,
                })
                logger.info(f"  Dropped (corr): {drop} (|r|={pair_corr:.3f} with {keep})")

    kept = [c for c in feature_cols if c not in dropped]
    return kept, report


def vif_filter(df, feature_cols, max_vif=10, max_iterations=20):
    """Iteratively drop highest-VIF feature until all VIF < max_vif."""
    current = list(feature_cols)
    report = []

    # Need complete cases for VIF
    subset = df[current].dropna()
    if len(subset) < len(current) + 2:
        logger.warning("Not enough complete cases for VIF; skipping VIF filter")
        return current, report

    for iteration in range(max_iterations):
        subset = df[current].dropna()
        if len(subset) < len(current) + 2:
            break

        X = subset.values
        try:
            vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        except (np.linalg.LinAlgError, ValueError):
            logger.warning("VIF computation failed; stopping VIF filter")
            break

        max_vif_val = max(vifs)
        if max_vif_val <= max_vif:
            break

        worst_idx = vifs.index(max_vif_val)
        worst_col = current[worst_idx]
        report.append({
            'feature': worst_col,
            'stage_dropped': 'vif',
            'reason': f'VIF={max_vif_val:.1f} (threshold: {max_vif})',
            'coverage_pct': np.nan,
        })
        logger.info(f"  Dropped (VIF): {worst_col} (VIF={max_vif_val:.1f})")
        current.pop(worst_idx)

    return current, report


def run_feature_selection(df, target_col=TARGET):
    """Run the full feature selection pipeline."""
    logger.info("=" * 60)
    logger.info("FEATURE SELECTION PIPELINE")
    logger.info("=" * 60)

    # Identify candidate features
    all_cols = [c for c in df.columns if c not in META_COLS]
    logger.info(f"Starting features: {len(all_cols)}")

    # Stage 1: Coverage filter
    logger.info("\nStage 1: Coverage filter (>50%)")
    kept, report1 = coverage_filter(df, all_cols)
    logger.info(f"  After coverage: {len(kept)} features")

    # Stage 2: Correlation filter
    logger.info("\nStage 2: Correlation filter (|r| < 0.85)")
    kept, report2 = correlation_filter(df, kept, target_col)
    logger.info(f"  After correlation: {len(kept)} features")

    # Stage 3: VIF filter
    logger.info("\nStage 3: VIF filter (VIF < 10)")
    kept, report3 = vif_filter(df, kept)
    logger.info(f"  After VIF: {len(kept)} features")

    # Combine reports
    full_report = report1 + report2 + report3
    report_df = pd.DataFrame(full_report)

    logger.info(f"\nFinal feature set ({len(kept)}): {kept}")
    return kept, report_df


# ---------------------------------------------------------------------------
# Model Training and Evaluation
# ---------------------------------------------------------------------------
def prepare_data(df, feature_cols, target_col=TARGET):
    """Impute missing values and scale features. Returns X, y, scaler, imputer."""
    X_raw = df[feature_cols].values
    y = df[target_col].values

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y, scaler, imputer


def train_ols_extended(X, y, feature_names):
    """OLS baseline with all features (statsmodels)."""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    # LOOCV predictions
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_train = sm.add_constant(X[train_idx])
        # add_constant may fail on single-row arrays; add manually
        X_test = np.hstack([np.ones((len(test_idx), 1)), X[test_idx]])
        fold_model = sm.OLS(y[train_idx], X_train).fit()
        preds[test_idx] = fold_model.predict(X_test)

    loocv_r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
    loocv_rmse = np.sqrt(np.mean((y - preds) ** 2))
    loocv_mae = np.mean(np.abs(y - preds))

    return {
        'name': 'OLS Extended',
        'model': model,
        'loocv_preds': preds,
        'loocv_r2': loocv_r2,
        'loocv_rmse': loocv_rmse,
        'loocv_mae': loocv_mae,
        'n_features': X.shape[1],
        'coefs': dict(zip(feature_names, model.params[1:])),  # skip intercept
    }


def train_ridge(X, y):
    """Ridge regression with built-in CV for alpha selection."""
    alphas = np.logspace(-2, 4, 100)
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X, y)

    loo = LeaveOneOut()
    preds = cross_val_predict(model, X, y, cv=loo)

    loocv_r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
    loocv_rmse = np.sqrt(np.mean((y - preds) ** 2))
    loocv_mae = np.mean(np.abs(y - preds))

    return {
        'name': 'Ridge',
        'model': model,
        'loocv_preds': preds,
        'loocv_r2': loocv_r2,
        'loocv_rmse': loocv_rmse,
        'loocv_mae': loocv_mae,
        'n_features': X.shape[1],
        'alpha': model.alpha_,
    }


def train_lasso(X, y):
    """Lasso regression with built-in CV for alpha selection."""
    alphas = np.logspace(-3, 2, 100)
    model = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
    model.fit(X, y)

    loo = LeaveOneOut()
    preds = cross_val_predict(model, X, y, cv=loo)

    loocv_r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
    loocv_rmse = np.sqrt(np.mean((y - preds) ** 2))
    loocv_mae = np.mean(np.abs(y - preds))

    n_nonzero = np.sum(model.coef_ != 0)

    return {
        'name': 'Lasso',
        'model': model,
        'loocv_preds': preds,
        'loocv_r2': loocv_r2,
        'loocv_rmse': loocv_rmse,
        'loocv_mae': loocv_mae,
        'n_features': n_nonzero,
        'alpha': model.alpha_,
    }


def train_elasticnet(X, y):
    """ElasticNet with built-in CV."""
    alphas = np.logspace(-3, 2, 50)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    model = ElasticNetCV(
        alphas=alphas, l1_ratio=l1_ratios, cv=5,
        max_iter=10000, random_state=42
    )
    model.fit(X, y)

    loo = LeaveOneOut()
    preds = cross_val_predict(model, X, y, cv=loo)

    loocv_r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
    loocv_rmse = np.sqrt(np.mean((y - preds) ** 2))
    loocv_mae = np.mean(np.abs(y - preds))

    n_nonzero = np.sum(model.coef_ != 0)

    return {
        'name': 'ElasticNet',
        'model': model,
        'loocv_preds': preds,
        'loocv_r2': loocv_r2,
        'loocv_rmse': loocv_rmse,
        'loocv_mae': loocv_mae,
        'n_features': n_nonzero,
        'alpha': model.alpha_,
        'l1_ratio': model.l1_ratio_,
    }


def train_random_forest(X, y):
    """Random Forest with conservative hyperparameters for n=93."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    loo = LeaveOneOut()
    preds = cross_val_predict(model, X, y, cv=loo)

    loocv_r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
    loocv_rmse = np.sqrt(np.mean((y - preds) ** 2))
    loocv_mae = np.mean(np.abs(y - preds))

    return {
        'name': 'Random Forest',
        'model': model,
        'loocv_preds': preds,
        'loocv_r2': loocv_r2,
        'loocv_rmse': loocv_rmse,
        'loocv_mae': loocv_mae,
        'n_features': X.shape[1],
    }


def train_all_models(X, y, feature_names):
    """Train all 5 models and return results."""
    logger.info("=" * 60)
    logger.info("MODEL TRAINING (LOOCV)")
    logger.info("=" * 60)

    results = []

    logger.info("\n1. OLS Extended...")
    ols = train_ols_extended(X, y, feature_names)
    results.append(ols)
    logger.info(f"   LOOCV R2={ols['loocv_r2']:.4f}, RMSE={ols['loocv_rmse']:.2f}")

    logger.info("\n2. Ridge...")
    ridge = train_ridge(X, y)
    results.append(ridge)
    logger.info(f"   LOOCV R2={ridge['loocv_r2']:.4f}, RMSE={ridge['loocv_rmse']:.2f}, alpha={ridge['alpha']:.4f}")

    logger.info("\n3. Lasso...")
    lasso = train_lasso(X, y)
    results.append(lasso)
    logger.info(f"   LOOCV R2={lasso['loocv_r2']:.4f}, RMSE={lasso['loocv_rmse']:.2f}, alpha={lasso['alpha']:.4f}, features={lasso['n_features']}")

    logger.info("\n4. ElasticNet...")
    enet = train_elasticnet(X, y)
    results.append(enet)
    logger.info(f"   LOOCV R2={enet['loocv_r2']:.4f}, RMSE={enet['loocv_rmse']:.2f}, alpha={enet['alpha']:.4f}, l1_ratio={enet['l1_ratio']:.2f}")

    logger.info("\n5. Random Forest...")
    rf = train_random_forest(X, y)
    results.append(rf)
    logger.info(f"   LOOCV R2={rf['loocv_r2']:.4f}, RMSE={rf['loocv_rmse']:.2f}")

    # Sort by LOOCV R2
    results.sort(key=lambda x: x['loocv_r2'], reverse=True)
    logger.info(f"\nBest model: {results[0]['name']} (LOOCV R2={results[0]['loocv_r2']:.4f})")

    return results


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------
def compute_feature_importance(results, X, y, feature_names):
    """Compute feature importance from multiple models."""
    logger.info("\nComputing feature importance...")

    importance_data = []

    # Get Lasso coefficients
    lasso_result = next((r for r in results if r['name'] == 'Lasso'), None)
    lasso_coefs = {}
    if lasso_result:
        lasso_coefs = dict(zip(feature_names, lasso_result['model'].coef_))

    # Get RF permutation importance
    rf_result = next((r for r in results if r['name'] == 'Random Forest'), None)
    rf_imp = {}
    if rf_result:
        perm = permutation_importance(
            rf_result['model'], X, y,
            n_repeats=30, random_state=42, n_jobs=-1
        )
        rf_imp = dict(zip(feature_names, perm.importances_mean))

    # Univariate correlations (on scaled data -- same ranking as unscaled)
    for i, feat in enumerate(feature_names):
        mask = ~np.isnan(X[:, i])
        if mask.sum() > 5:
            r = np.corrcoef(X[mask, i], y[mask])[0, 1]
        else:
            r = 0.0

        importance_data.append({
            'feature': feat,
            'lasso_coef': lasso_coefs.get(feat, 0.0),
            'lasso_abs_coef': abs(lasso_coefs.get(feat, 0.0)),
            'rf_importance': rf_imp.get(feat, 0.0),
            'univariate_r': r,
            'abs_univariate_r': abs(r),
        })

    imp_df = pd.DataFrame(importance_data)

    # Combined ranking: average of (Lasso rank + RF rank + univariate rank)
    imp_df['lasso_rank'] = imp_df['lasso_abs_coef'].rank(ascending=False)
    imp_df['rf_rank'] = imp_df['rf_importance'].rank(ascending=False)
    imp_df['univariate_rank'] = imp_df['abs_univariate_r'].rank(ascending=False)
    imp_df['combined_rank'] = (
        (imp_df['lasso_rank'] + imp_df['rf_rank'] + imp_df['univariate_rank']) / 3
    ).rank()

    imp_df = imp_df.sort_values('combined_rank')

    # Drop helper columns
    imp_df = imp_df.drop(columns=['lasso_abs_coef', 'abs_univariate_r',
                                   'lasso_rank', 'rf_rank', 'univariate_rank'])

    return imp_df


# ---------------------------------------------------------------------------
# Residual Analysis
# ---------------------------------------------------------------------------
def residual_analysis(df, best_result):
    """Analyze residuals to identify overperformers and underperformers."""
    logger.info("\n" + "=" * 60)
    logger.info("RESIDUAL ANALYSIS (Overperformance Detection)")
    logger.info("=" * 60)

    preds = best_result['loocv_preds']
    actual = df[TARGET].values
    residuals = actual - preds

    # Standardize residuals
    res_mean = np.mean(residuals)
    res_std = np.std(residuals, ddof=1)
    z_scores = (residuals - res_mean) / res_std

    # Classification
    classifications = []
    for z in z_scores:
        if z > 1.0:
            classifications.append('overperformer')
        elif z < -1.0:
            classifications.append('underperformer')
        else:
            classifications.append('as_expected')

    # Build output dataframe
    residual_df = pd.DataFrame({
        'iso_code': df['iso_code'].values,
        'country_name': df['country_name'].values,
        'is_blue_zone': df['is_blue_zone'].values,
        'actual_le': actual,
        'predicted_le': np.round(preds, 2),
        'residual': np.round(residuals, 2),
        'residual_zscore': np.round(z_scores, 3),
        'classification': classifications,
    })

    residual_df['residual_rank'] = residual_df['residual'].rank(ascending=False).astype(int)
    residual_df = residual_df.sort_values('residual', ascending=False)

    n_over = sum(1 for c in classifications if c == 'overperformer')
    n_under = sum(1 for c in classifications if c == 'underperformer')
    n_expected = sum(1 for c in classifications if c == 'as_expected')

    logger.info(f"  Overperformers (z > 1): {n_over}")
    logger.info(f"  As expected (-1 < z < 1): {n_expected}")
    logger.info(f"  Underperformers (z < -1): {n_under}")
    logger.info(f"  Residual std: {res_std:.2f} years")

    return residual_df


def validate_blue_zones(residual_df):
    """Check where known Blue Zone countries appear in residual rankings."""
    logger.info("\n" + "=" * 60)
    logger.info("BLUE ZONE VALIDATION")
    logger.info("=" * 60)

    bz_rows = residual_df[residual_df['iso_code'].isin(BLUE_ZONE_ISOS)]

    for _, row in bz_rows.iterrows():
        logger.info(
            f"  {row['country_name']:20s} rank={row['residual_rank']:3d}/93  "
            f"residual={row['residual']:+.2f}  z={row['residual_zscore']:+.3f}  "
            f"[{row['classification']}]"
        )

    # Summary
    bz_mean_residual = bz_rows['residual'].mean()
    bz_mean_z = bz_rows['residual_zscore'].mean()
    bz_median_rank = bz_rows['residual_rank'].median()

    logger.info(f"\n  BZ mean residual: {bz_mean_residual:+.2f} years")
    logger.info(f"  BZ mean z-score: {bz_mean_z:+.3f}")
    logger.info(f"  BZ median rank: {bz_median_rank:.0f}/93")

    # Check: are BZ countries (excluding USA) mostly in top half?
    bz_no_usa = bz_rows[bz_rows['iso_code'] != 'USA']
    n_top_half = (bz_no_usa['residual_rank'] <= 47).sum()
    logger.info(f"  BZ countries (excl. USA) in top half: {n_top_half}/{len(bz_no_usa)}")


# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------
def save_model_comparison(results):
    """Save model comparison table."""
    rows = []
    for r in results:
        rows.append({
            'model': r['name'],
            'loocv_r2': round(r['loocv_r2'], 4),
            'loocv_rmse': round(r['loocv_rmse'], 3),
            'loocv_mae': round(r['loocv_mae'], 3),
            'n_features': r['n_features'],
        })

    comp_df = pd.DataFrame(rows)

    # Also compute 5-fold CV for comparison
    # (already captured by LOOCV which is more rigorous)
    path = os.path.join(OUTPUT_DIR, 'ml_model_comparison.csv')
    comp_df.to_csv(path, index=False)
    logger.info(f"Saved: {path}")
    return comp_df


def save_outputs(residual_df, importance_df, selection_report_df, model_comparison_df):
    """Save all ML output files."""
    # Feature selection report
    path = os.path.join(OUTPUT_DIR, 'ml_feature_selection_report.csv')
    selection_report_df.to_csv(path, index=False)
    logger.info(f"Saved: {path}")

    # Feature importance
    path = os.path.join(OUTPUT_DIR, 'ml_feature_importance.csv')
    importance_df.to_csv(path, index=False)
    logger.info(f"Saved: {path}")

    # Full residual analysis
    path = os.path.join(OUTPUT_DIR, 'ml_residual_analysis.csv')
    residual_df.to_csv(path, index=False)
    logger.info(f"Saved: {path}")

    # Top 15 overperformers ("Hidden Blue Zones")
    overperformers = residual_df.head(15).copy()
    path = os.path.join(OUTPUT_DIR, 'ml_hidden_blue_zones.csv')
    overperformers.to_csv(path, index=False)
    logger.info(f"Saved: {path}")

    # Bottom 15 underperformers
    underperformers = residual_df.tail(15).copy()
    path = os.path.join(OUTPUT_DIR, 'ml_underperformers.csv')
    underperformers.to_csv(path, index=False)
    logger.info(f"Saved: {path}")


def print_summary(model_comparison_df, residual_df, importance_df):
    """Print final summary to console."""
    print("\n" + "=" * 70)
    print("ML LIFE EXPECTANCY PREDICTION -- RESULTS SUMMARY")
    print("=" * 70)

    print("\nMODEL COMPARISON (sorted by LOOCV R2):")
    print(model_comparison_df.to_string(index=False))

    print("\nTOP 10 FEATURES (by combined importance):")
    top_feats = importance_df.head(10)[['feature', 'lasso_coef', 'rf_importance', 'univariate_r', 'combined_rank']]
    print(top_feats.to_string(index=False))

    print("\nTOP 10 OVERPERFORMERS ('Hidden Blue Zone' candidates):")
    top_over = residual_df.head(10)[['residual_rank', 'country_name', 'is_blue_zone', 'actual_le', 'predicted_le', 'residual', 'residual_zscore']]
    print(top_over.to_string(index=False))

    print("\nBOTTOM 10 UNDERPERFORMERS:")
    bottom = residual_df.tail(10)[['residual_rank', 'country_name', 'is_blue_zone', 'actual_le', 'predicted_le', 'residual', 'residual_zscore']]
    print(bottom.to_string(index=False))

    print("\nBLUE ZONE COUNTRIES:")
    bz = residual_df[residual_df['iso_code'].isin(BLUE_ZONE_ISOS)][
        ['residual_rank', 'country_name', 'actual_le', 'predicted_le', 'residual', 'classification']
    ]
    print(bz.to_string(index=False))

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    """Run the full ML prediction pipeline."""
    # 1. Load feature matrix
    features_path = os.path.join(PROJECT_DIR, 'data', 'features', 'ml_feature_matrix.csv')
    df = pd.read_csv(features_path)
    logger.info(f"Loaded feature matrix: {df.shape}")

    # Verify target exists
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in feature matrix")

    # Drop rows without target
    df = df[df[TARGET].notna()].reset_index(drop=True)
    logger.info(f"Rows with valid LE: {len(df)}")

    # 2. Feature selection
    selected_features, selection_report = run_feature_selection(df)

    if len(selected_features) < 3:
        raise ValueError(f"Only {len(selected_features)} features survived selection; need at least 3")

    # 3. Prepare data
    X, y, scaler, imputer = prepare_data(df, selected_features)
    logger.info(f"\nPrepared data: X={X.shape}, y={y.shape}")

    # 4. Train all models
    results = train_all_models(X, y, selected_features)

    # 5. Save model comparison
    model_comp = save_model_comparison(results)

    # 6. Feature importance (uses Lasso + RF results)
    importance_df = compute_feature_importance(results, X, y, selected_features)

    # 7. Residual analysis using best model
    best = results[0]
    logger.info(f"\nUsing best model for residuals: {best['name']}")
    residual_df = residual_analysis(df, best)

    # 8. Blue Zone validation
    validate_blue_zones(residual_df)

    # 9. Save all outputs
    save_outputs(residual_df, importance_df, selection_report, model_comp)

    # 10. Print summary
    print_summary(model_comp, residual_df, importance_df)

    return results, residual_df, importance_df


if __name__ == '__main__':
    main()
