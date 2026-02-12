#!/usr/bin/env python3
"""
Statistical Significance Tests for Blue Zones Longevity Analysis

Covers:
  - Bootstrap confidence intervals on the BZ-global gap
  - P-values on beta convergence (Pearson + Bonferroni)
  - Sigma convergence OLS trend test + Bartlett's test
  - COVID acceleration test (prediction interval)
  - Decade improvement independent t-tests with Cohen's d
  - Partial correlations controlling for GDP
  - Multiple regression with VIF
  - Population-weighted averages
  - Drop-one-country sensitivity analysis
  - Time-series tests (ADF, Chow structural break, ARIMA)
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, bartlett, ttest_ind
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore', category=FutureWarning)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLUE_ZONE_ISOS = {'USA', 'JPN', 'ITA', 'GRC', 'CRI'}


def load_panel():
    path = os.path.join(PROJECT_DIR, 'data', 'historical', 'merged_historical_panel.csv')
    df = pd.read_csv(path)
    df['is_blue_zone'] = df['iso_code'].isin(BLUE_ZONE_ISOS).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2a. Bootstrap Confidence Intervals on the Blue Zone Gap
# ---------------------------------------------------------------------------
def gap_confidence_intervals(df, n_bootstrap=1000, seed=42):
    """For each year, compute 95% CI on (BZ_mean - Global_mean) via bootstrap."""
    rng = np.random.RandomState(seed)
    results = []

    for year in sorted(df['year'].unique()):
        yr = df[df['year'] == year].dropna(subset=['life_expectancy'])
        bz = yr[yr['is_blue_zone'] == 1]['life_expectancy'].values
        non_bz = yr[yr['is_blue_zone'] == 0]['life_expectancy'].values

        if len(bz) < 3 or len(non_bz) < 5:
            continue

        observed_gap = bz.mean() - non_bz.mean()

        # Bootstrap: resample 5 countries from non-BZ pool, compute gap
        boot_gaps = []
        for _ in range(n_bootstrap):
            sample = rng.choice(non_bz, size=len(bz), replace=True)
            boot_gaps.append(sample.mean() - non_bz.mean())

        boot_gaps = np.array(boot_gaps)
        # P-value: fraction of bootstrap samples where random gap >= observed gap
        p_value = (boot_gaps >= observed_gap).sum() / n_bootstrap

        ci_lower = np.percentile(boot_gaps, 2.5)
        ci_upper = np.percentile(boot_gaps, 97.5)

        # CI on the actual gap using bootstrap of BZ countries
        boot_actual = []
        for _ in range(n_bootstrap):
            bz_s = rng.choice(bz, size=len(bz), replace=True)
            nb_s = rng.choice(non_bz, size=len(non_bz), replace=True)
            boot_actual.append(bz_s.mean() - nb_s.mean())
        boot_actual = np.array(boot_actual)

        results.append({
            'year': year,
            'bz_mean': bz.mean(),
            'global_mean': non_bz.mean(),
            'gap': observed_gap,
            'ci_lower': np.percentile(boot_actual, 2.5),
            'ci_upper': np.percentile(boot_actual, 97.5),
            'p_value': p_value,
            'significant_at_05': p_value < 0.05,
        })

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUTPUT_DIR, 'gap_confidence_intervals.csv'), index=False)
    print(f"[2a] Gap CIs: {len(out)} years, gap range {out['gap'].min():.2f} to {out['gap'].max():.2f}")
    sig_years = out['significant_at_05'].sum()
    print(f"     Significant at p<0.05: {sig_years}/{len(out)} years")
    return out


# ---------------------------------------------------------------------------
# 2b. P-values on Beta Convergence
# ---------------------------------------------------------------------------
def beta_convergence_pvalues(df):
    """Compute Pearson r and p-value for each decade's beta convergence."""
    results = []
    decades = [(1960, 1970), (1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2019)]

    for start, end in decades:
        d_start = df[df['year'] == start].dropna(subset=['life_expectancy'])
        d_end = df[df['year'] == end].dropna(subset=['life_expectancy'])

        merged = d_start[['iso_code', 'life_expectancy']].merge(
            d_end[['iso_code', 'life_expectancy']],
            on='iso_code', suffixes=('_start', '_end')
        )
        if len(merged) < 10:
            continue

        merged['gain'] = merged['life_expectancy_end'] - merged['life_expectancy_start']
        r, p = pearsonr(merged['life_expectancy_start'], merged['gain'])

        results.append({
            'decade': f"{start}-{end}",
            'n_countries': len(merged),
            'beta_r': r,
            'p_value': p,
        })

    out = pd.DataFrame(results)
    n_tests = len(out)
    out['p_value_bonferroni'] = np.minimum(out['p_value'] * n_tests, 1.0)
    out['significant_at_05'] = out['p_value_bonferroni'] < 0.05

    # Merge into existing beta_convergence CSVs
    for subdir in ['pre_covid', 'full_period']:
        beta_path = os.path.join(OUTPUT_DIR, subdir, 'beta_convergence.csv')
        if os.path.exists(beta_path):
            existing = pd.read_csv(beta_path)
            if 'p_value' not in existing.columns:
                existing = existing.merge(
                    out[['decade', 'p_value', 'p_value_bonferroni', 'significant_at_05']],
                    on='decade', how='left'
                )
                existing.to_csv(beta_path, index=False)

    out.to_csv(os.path.join(OUTPUT_DIR, 'beta_convergence_pvalues.csv'), index=False)
    print(f"[2b] Beta convergence: {n_tests} decades tested")
    for _, row in out.iterrows():
        sig = "***" if row['significant_at_05'] else ""
        print(f"     {row['decade']}: r={row['beta_r']:.3f}, p={row['p_value']:.4f}, "
              f"p_bonf={row['p_value_bonferroni']:.4f} {sig}")
    return out


# ---------------------------------------------------------------------------
# 2c. Sigma Convergence Trend Test
# ---------------------------------------------------------------------------
def sigma_convergence_test(df):
    """OLS regression of LE standard deviation on year. Bartlett's test on start vs end."""
    results = []

    for label, end_year in [('pre_covid', 2019), ('full_period', 2023)]:
        sub = df[(df['year'] <= end_year)].copy()
        sigma_by_year = sub.groupby('year')['life_expectancy'].std().dropna()

        if len(sigma_by_year) < 10:
            continue

        X = sm.add_constant(sigma_by_year.index.values.astype(float))
        y = sigma_by_year.values
        model = sm.OLS(y, X).fit()

        # Bartlett's test: first decade vs last decade variance
        first_decade = sub[sub['year'] <= sub['year'].min() + 10]
        last_decade = sub[sub['year'] >= end_year - 10]
        first_le = first_decade.groupby('iso_code')['life_expectancy'].mean().dropna()
        last_le = last_decade.groupby('iso_code')['life_expectancy'].mean().dropna()

        try:
            bart_stat, bart_p = bartlett(first_le.values, last_le.values)
        except Exception:
            bart_stat, bart_p = np.nan, np.nan

        results.append({
            'period': label,
            'n_years': len(sigma_by_year),
            'slope': model.params[1],
            'slope_se': model.bse[1],
            'slope_p_value': model.pvalues[1],
            'r_squared': model.rsquared,
            'bartlett_stat': bart_stat,
            'bartlett_p_value': bart_p,
            'sigma_declining': model.params[1] < 0 and model.pvalues[1] < 0.05,
        })

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUTPUT_DIR, 'sigma_convergence_test.csv'), index=False)
    print(f"[2c] Sigma convergence test:")
    for _, row in out.iterrows():
        print(f"     {row['period']}: slope={row['slope']:.4f}/yr, p={row['slope_p_value']:.6f}, "
              f"R2={row['r_squared']:.3f}, Bartlett p={row['bartlett_p_value']:.4f}")
    return out


# ---------------------------------------------------------------------------
# 2d. COVID Acceleration Test
# ---------------------------------------------------------------------------
def covid_acceleration_test(df):
    """Test whether 2023 gap is significantly below pre-COVID trend extrapolation."""
    gap_data = df.groupby(['year', 'is_blue_zone'])['life_expectancy'].mean().unstack()
    gap_data.columns = ['non_bz', 'bz']
    gap_data['gap'] = gap_data['bz'] - gap_data['non_bz']
    gap_data = gap_data.dropna()

    # Fit OLS on 1990-2019
    pre = gap_data.loc[1990:2019]
    X = sm.add_constant(pre.index.values.astype(float))
    y = pre['gap'].values
    model = sm.OLS(y, X).fit()

    # Predict 2020-2023 with prediction intervals
    results = []
    for year in [2020, 2021, 2022, 2023]:
        if year not in gap_data.index:
            continue
        x_pred = np.array([[1.0, float(year)]])
        predicted = model.predict(x_pred)[0]
        actual = gap_data.loc[year, 'gap']

        # Prediction interval
        pred_summary = model.get_prediction(x_pred).summary_frame(alpha=0.05)
        pi_lower = pred_summary['obs_ci_lower'].values[0]
        pi_upper = pred_summary['obs_ci_upper'].values[0]

        results.append({
            'year': year,
            'predicted_gap': predicted,
            'actual_gap': actual,
            'difference': actual - predicted,
            'pi_lower': pi_lower,
            'pi_upper': pi_upper,
            'outside_pi': actual < pi_lower or actual > pi_upper,
        })

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUTPUT_DIR, 'covid_acceleration_test.csv'), index=False)
    print(f"[2d] COVID acceleration test (trend 1990-2019):")
    print(f"     Trend slope: {model.params[1]:.4f} years/year")
    for _, row in out.iterrows():
        flag = " *** OUTSIDE PI" if row['outside_pi'] else ""
        print(f"     {row['year']}: predicted={row['predicted_gap']:.2f}, "
              f"actual={row['actual_gap']:.2f}, diff={row['difference']:.2f}{flag}")
    return out


# ---------------------------------------------------------------------------
# 2e. Decade Improvement Differences
# ---------------------------------------------------------------------------
def decade_improvement_tests(df):
    """Independent t-test: BZ gains vs non-BZ gains per decade. Cohen's d."""
    decades = [(1960, 1970), (1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2019)]
    results = []

    for start, end in decades:
        d_start = df[df['year'] == start].dropna(subset=['life_expectancy'])
        d_end = df[df['year'] == end].dropna(subset=['life_expectancy'])

        merged = d_start[['iso_code', 'life_expectancy', 'is_blue_zone']].merge(
            d_end[['iso_code', 'life_expectancy']],
            on='iso_code', suffixes=('_start', '_end')
        )
        merged['gain'] = merged['life_expectancy_end'] - merged['life_expectancy_start']

        bz_gains = merged[merged['is_blue_zone'] == 1]['gain'].values
        non_bz_gains = merged[merged['is_blue_zone'] == 0]['gain'].values

        if len(bz_gains) < 2 or len(non_bz_gains) < 5:
            continue

        t_stat, p_val = ttest_ind(bz_gains, non_bz_gains, equal_var=False)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(bz_gains) - 1) * bz_gains.std(ddof=1)**2 +
             (len(non_bz_gains) - 1) * non_bz_gains.std(ddof=1)**2) /
            (len(bz_gains) + len(non_bz_gains) - 2)
        )
        cohens_d = (bz_gains.mean() - non_bz_gains.mean()) / pooled_std if pooled_std > 0 else 0

        results.append({
            'decade': f"{start}-{end}",
            'bz_mean_gain': bz_gains.mean(),
            'non_bz_mean_gain': non_bz_gains.mean(),
            'bz_n': len(bz_gains),
            'non_bz_n': len(non_bz_gains),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
        })

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUTPUT_DIR, 'decade_improvement_tests.csv'), index=False)
    print(f"[2e] Decade improvement t-tests:")
    for _, row in out.iterrows():
        sig = "sig" if row['p_value'] < 0.05 else "ns"
        print(f"     {row['decade']}: BZ={row['bz_mean_gain']:.2f}, nonBZ={row['non_bz_mean_gain']:.2f}, "
              f"d={row['cohens_d']:.2f} ({row['effect_size']}), p={row['p_value']:.3f} [{sig}]")
    return out


# ---------------------------------------------------------------------------
# 3a. Partial Correlations (controlling for GDP)
# ---------------------------------------------------------------------------
def partial_correlations(df):
    """Compute partial correlation of each indicator with LE, controlling for GDP."""
    # Use most recent data with best coverage (2015+)
    recent = df[df['year'] >= 2015].copy()

    indicators = [
        'physicians_per_1000', 'urban_population_pct', 'health_expenditure_pc',
        'fertility_rate', 'population_65plus_pct', 'death_rate',
        'clean_water_access_pct', 'alcohol_per_capita', 'pm25_air_pollution',
        'forest_area_pct', 'gini_index',
    ]

    # Cross-sectional: most recent non-null value per country
    rows = []
    for iso, grp in recent.groupby('iso_code'):
        valid = grp.dropna(subset=['life_expectancy', 'gdp_per_capita'])
        if len(valid) > 0:
            rows.append(valid.sort_values('year').iloc[-1])
    cross = pd.DataFrame(rows)

    if cross.empty:
        print("[3a] Not enough data for partial correlations")
        return pd.DataFrame()

    results = []
    for ind in indicators:
        valid = cross.dropna(subset=['life_expectancy', 'gdp_per_capita', ind])
        if len(valid) < 15:
            continue

        le = valid['life_expectancy'].values
        x = valid[ind].values
        gdp = valid['gdp_per_capita'].values

        # Raw correlation
        r_raw, p_raw = pearsonr(x, le)

        # Partial correlation (controlling for GDP)
        r_xz, _ = pearsonr(x, gdp)
        r_yz, _ = pearsonr(le, gdp)

        denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denom > 0:
            r_partial = (r_raw - r_xz * r_yz) / denom
        else:
            r_partial = np.nan

        # P-value for partial correlation (approximate using t-distribution)
        n = len(valid)
        if not np.isnan(r_partial) and abs(r_partial) < 1:
            t_stat = r_partial * np.sqrt((n - 3) / (1 - r_partial**2))
            p_partial = 2 * stats.t.sf(abs(t_stat), df=n - 3)
        else:
            p_partial = np.nan

        results.append({
            'indicator': ind,
            'n_countries': n,
            'raw_correlation': r_raw,
            'raw_p_value': p_raw,
            'partial_correlation_gdp_controlled': r_partial,
            'partial_p_value': p_partial,
            'correlation_change': r_partial - r_raw if not np.isnan(r_partial) else np.nan,
            'gdp_explains_relationship': abs(r_raw) > 0.3 and abs(r_partial) < 0.15,
        })

    out = pd.DataFrame(results)
    out = out.sort_values('partial_correlation_gdp_controlled', key=abs, ascending=False)
    out.to_csv(os.path.join(OUTPUT_DIR, 'partial_correlations.csv'), index=False)
    print(f"[3a] Partial correlations (GDP-controlled), {len(out)} indicators:")
    for _, row in out.iterrows():
        flag = " [GDP explains]" if row['gdp_explains_relationship'] else ""
        print(f"     {row['indicator']}: raw r={row['raw_correlation']:.3f}, "
              f"partial r={row['partial_correlation_gdp_controlled']:.3f}{flag}")
    return out


# ---------------------------------------------------------------------------
# 3b. Multiple Regression
# ---------------------------------------------------------------------------
def multiple_regression(df):
    """OLS: life_expectancy ~ gdp + physicians + urban + health_exp. VIF check."""
    recent = df[df['year'] >= 2015].copy()

    predictors = ['gdp_per_capita', 'physicians_per_1000', 'urban_population_pct',
                  'health_expenditure_pc']

    # Cross-sectional: most recent non-null per country
    rows = []
    for iso, grp in recent.groupby('iso_code'):
        valid = grp.dropna(subset=['life_expectancy'] + predictors)
        if len(valid) > 0:
            rows.append(valid.sort_values('year').iloc[-1])
    cross = pd.DataFrame(rows)

    if len(cross) < 20:
        print("[3b] Not enough complete-case countries for regression")
        return None

    y = cross['life_expectancy'].values
    X = cross[predictors].values
    X_const = sm.add_constant(X)

    model = sm.OLS(y, X_const).fit()

    # VIF
    vif_data = []
    for i, pred in enumerate(predictors):
        vif_val = variance_inflation_factor(X_const, i + 1)  # +1 because const is at 0
        vif_data.append({'predictor': pred, 'vif': vif_val})
    vif_df = pd.DataFrame(vif_data)

    # Save regression summary
    reg_results = pd.DataFrame({
        'predictor': ['(intercept)'] + predictors,
        'coefficient': model.params,
        'std_error': model.bse,
        'p_value': model.pvalues,
        't_statistic': model.tvalues,
    })
    reg_results['significant_at_05'] = reg_results['p_value'] < 0.05

    meta = pd.DataFrame([{
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_p_value': model.f_pvalue,
        'n_observations': int(model.nobs),
        'aic': model.aic,
        'bic': model.bic,
    }])

    reg_results.to_csv(os.path.join(OUTPUT_DIR, 'regression_results.csv'), index=False)
    vif_df.to_csv(os.path.join(OUTPUT_DIR, 'regression_vif.csv'), index=False)
    meta.to_csv(os.path.join(OUTPUT_DIR, 'regression_meta.csv'), index=False)

    print(f"[3b] Multiple regression (n={int(model.nobs)}):")
    print(f"     R2={model.rsquared:.3f}, Adj R2={model.rsquared_adj:.3f}, F p={model.f_pvalue:.6f}")
    for _, row in reg_results.iterrows():
        sig = "*" if row['significant_at_05'] else ""
        print(f"     {row['predictor']}: coef={row['coefficient']:.4f}, p={row['p_value']:.4f} {sig}")
    print("     VIF values:")
    for _, row in vif_df.iterrows():
        flag = " HIGH" if row['vif'] > 10 else ""
        print(f"       {row['predictor']}: {row['vif']:.1f}{flag}")
    return model


# ---------------------------------------------------------------------------
# 5. Population-Weighted Averages
# ---------------------------------------------------------------------------
def population_weighted_averages(df):
    """Compute pop-weighted global average LE and recalculate BZ gap."""
    results = []

    for year in sorted(df['year'].unique()):
        yr = df[(df['year'] == year)].dropna(subset=['life_expectancy', 'population_total'])
        if len(yr) < 10:
            continue

        bz = yr[yr['is_blue_zone'] == 1]
        non_bz = yr[yr['is_blue_zone'] == 0]

        # Unweighted
        global_mean = non_bz['life_expectancy'].mean()
        bz_mean = bz['life_expectancy'].mean()

        # Population-weighted
        if non_bz['population_total'].sum() > 0:
            weighted_global = np.average(
                non_bz['life_expectancy'], weights=non_bz['population_total'])
        else:
            weighted_global = global_mean

        if bz['population_total'].sum() > 0:
            weighted_bz = np.average(
                bz['life_expectancy'], weights=bz['population_total'])
        else:
            weighted_bz = bz_mean

        results.append({
            'year': year,
            'bz_mean_unweighted': bz_mean,
            'bz_mean_weighted': weighted_bz,
            'global_mean_unweighted': global_mean,
            'global_mean_weighted': weighted_global,
            'gap_unweighted': bz_mean - global_mean,
            'gap_weighted': weighted_bz - weighted_global,
            'gap_difference': (bz_mean - global_mean) - (weighted_bz - weighted_global),
        })

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUTPUT_DIR, 'population_weighted_gap.csv'), index=False)

    # Update the existing gap CSVs with weighted columns
    for subdir in ['pre_covid', 'full_period']:
        gap_path = os.path.join(OUTPUT_DIR, subdir, 'blue_zone_vs_global.csv')
        if os.path.exists(gap_path):
            existing = pd.read_csv(gap_path)
            if 'weighted_global_mean' not in existing.columns:
                merged = existing.merge(
                    out[['year', 'global_mean_weighted', 'bz_mean_weighted', 'gap_weighted']],
                    on='year', how='left'
                )
                merged.rename(columns={
                    'global_mean_weighted': 'weighted_global_mean',
                    'bz_mean_weighted': 'weighted_bz_mean',
                    'gap_weighted': 'weighted_gap',
                }, inplace=True)
                merged.to_csv(gap_path, index=False)

    recent = out[out['year'] >= 2000]
    print(f"[5] Population-weighted averages: {len(out)} years")
    if not recent.empty:
        print(f"     Recent avg unweighted gap: {recent['gap_unweighted'].mean():.2f}")
        print(f"     Recent avg weighted gap: {recent['gap_weighted'].mean():.2f}")
        print(f"     Weighting effect: {recent['gap_difference'].mean():.2f} years")
    return out


# ---------------------------------------------------------------------------
# 6. Sensitivity Analysis (Drop-One-Country)
# ---------------------------------------------------------------------------
def sensitivity_drop_one(df):
    """Remove each BZ country one at a time, recalculate gap and convergence."""
    bz_countries = sorted(BLUE_ZONE_ISOS)
    country_names = {
        'USA': 'United States', 'JPN': 'Japan', 'ITA': 'Italy',
        'GRC': 'Greece', 'CRI': 'Costa Rica'
    }
    results = []

    # Baseline: all 5 BZ countries, year 2019
    yr2019 = df[df['year'] == 2019].dropna(subset=['life_expectancy'])
    bz_all = yr2019[yr2019['is_blue_zone'] == 1]['life_expectancy']
    non_bz = yr2019[yr2019['is_blue_zone'] == 0]['life_expectancy']
    baseline_gap = bz_all.mean() - non_bz.mean()
    baseline_bz_mean = bz_all.mean()

    # Baseline beta (2010-2019)
    d2010 = df[df['year'] == 2010].dropna(subset=['life_expectancy'])
    d2019 = df[df['year'] == 2019].dropna(subset=['life_expectancy'])
    beta_merged = d2010[['iso_code', 'life_expectancy']].merge(
        d2019[['iso_code', 'life_expectancy']], on='iso_code', suffixes=('_start', '_end'))
    beta_merged['gain'] = beta_merged['life_expectancy_end'] - beta_merged['life_expectancy_start']
    baseline_beta_r, _ = pearsonr(beta_merged['life_expectancy_start'], beta_merged['gain'])

    for iso in bz_countries:
        # Recalculate with this country removed
        remaining_bz = yr2019[(yr2019['is_blue_zone'] == 1) & (yr2019['iso_code'] != iso)]
        new_bz_mean = remaining_bz['life_expectancy'].mean()
        new_gap = new_bz_mean - non_bz.mean()

        # Recalculate beta without this country
        beta_without = beta_merged[beta_merged['iso_code'] != iso]
        new_beta_r, _ = pearsonr(beta_without['life_expectancy_start'], beta_without['gain'])

        country_le = yr2019[yr2019['iso_code'] == iso]['life_expectancy'].values[0]

        results.append({
            'dropped_country': country_names[iso],
            'dropped_iso': iso,
            'dropped_le_2019': country_le,
            'remaining_bz_mean': new_bz_mean,
            'gap_2019': new_gap,
            'gap_change': new_gap - baseline_gap,
            'gap_change_pct': ((new_gap - baseline_gap) / baseline_gap) * 100,
            'beta_r': new_beta_r,
            'beta_r_change': new_beta_r - baseline_beta_r,
        })

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUTPUT_DIR, 'sensitivity_drop_one.csv'), index=False)
    print(f"[6] Sensitivity analysis (baseline gap={baseline_gap:.2f}):")
    for _, row in out.iterrows():
        print(f"     Drop {row['dropped_country']} (LE={row['dropped_le_2019']:.1f}): "
              f"gap={row['gap_2019']:.2f} ({row['gap_change']:+.2f}), "
              f"beta r change={row['beta_r_change']:+.3f}")
    return out


# ---------------------------------------------------------------------------
# 10. Time-Series Tests
# ---------------------------------------------------------------------------
def time_series_tests(df):
    """ADF test, structural break at 2020, ARIMA on BZ gap."""
    results = []

    # Global average LE series
    global_le = df.groupby('year')['life_expectancy'].mean().dropna()

    # ADF test on global LE
    adf_result = adfuller(global_le.values, maxlag=5, autolag='AIC')
    results.append({
        'test': 'ADF_global_LE',
        'description': 'Augmented Dickey-Fuller test on global average LE',
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'conclusion': 'Non-stationary (trending)' if adf_result[1] > 0.05 else 'Stationary',
    })

    # ADF on first differences
    global_le_diff = global_le.diff().dropna()
    adf_diff = adfuller(global_le_diff.values, maxlag=5, autolag='AIC')
    results.append({
        'test': 'ADF_global_LE_diff',
        'description': 'ADF on first differences of global LE',
        'statistic': adf_diff[0],
        'p_value': adf_diff[1],
        'conclusion': 'Stationary after differencing' if adf_diff[1] < 0.05 else 'Still non-stationary',
    })

    # BZ gap series
    gap_data = df.groupby(['year', 'is_blue_zone'])['life_expectancy'].mean().unstack()
    gap_data.columns = ['non_bz', 'bz']
    gap_data['gap'] = gap_data['bz'] - gap_data['non_bz']
    gap_series = gap_data['gap'].dropna()

    # ADF on gap
    adf_gap = adfuller(gap_series.values, maxlag=5, autolag='AIC')
    results.append({
        'test': 'ADF_BZ_gap',
        'description': 'ADF test on BZ-global gap series',
        'statistic': adf_gap[0],
        'p_value': adf_gap[1],
        'conclusion': 'Non-stationary' if adf_gap[1] > 0.05 else 'Stationary',
    })

    # Chow-type structural break test at 2020
    pre_2020 = gap_series[gap_series.index < 2020]
    post_2020 = gap_series[gap_series.index >= 2020]

    if len(pre_2020) > 5 and len(post_2020) > 2:
        # Full model
        X_full = sm.add_constant(gap_series.index.values.astype(float))
        model_full = sm.OLS(gap_series.values, X_full).fit()

        # Pre-2020 model
        X_pre = sm.add_constant(pre_2020.index.values.astype(float))
        model_pre = sm.OLS(pre_2020.values, X_pre).fit()

        # Post-2020 model
        X_post = sm.add_constant(post_2020.index.values.astype(float))
        model_post = sm.OLS(post_2020.values, X_post).fit()

        # Chow F-statistic
        rss_full = model_full.ssr
        rss_pre = model_pre.ssr
        rss_post = model_post.ssr
        k = 2  # number of parameters
        n = len(gap_series)
        n1 = len(pre_2020)
        n2 = len(post_2020)

        if (rss_pre + rss_post) > 0 and (n - 2 * k) > 0:
            f_stat = ((rss_full - rss_pre - rss_post) / k) / ((rss_pre + rss_post) / (n - 2 * k))
            f_p = 1 - stats.f.cdf(f_stat, k, n - 2 * k)
        else:
            f_stat, f_p = np.nan, np.nan

        results.append({
            'test': 'Chow_structural_break_2020',
            'description': 'Structural break test at 2020 on BZ gap series',
            'statistic': f_stat,
            'p_value': f_p,
            'conclusion': 'Structural break detected' if f_p < 0.05 else 'No structural break',
        })

    # Simple ARIMA(1,1,0) on BZ gap
    try:
        from statsmodels.tsa.arima.model import ARIMA
        arima_model = ARIMA(gap_series.values, order=(1, 1, 0)).fit()

        # Check if COVID period residuals are outliers
        residuals = arima_model.resid
        covid_idx = gap_series.index.get_loc(2020) if 2020 in gap_series.index else None
        if covid_idx is not None:
            covid_resid = residuals[covid_idx]
            resid_std = residuals[:covid_idx].std()
            z_score = covid_resid / resid_std if resid_std > 0 else 0

            results.append({
                'test': 'ARIMA_COVID_shock',
                'description': 'ARIMA(1,1,0) residual at 2020 as z-score',
                'statistic': z_score,
                'p_value': 2 * stats.norm.sf(abs(z_score)),
                'conclusion': 'Significant shock' if abs(z_score) > 1.96 else 'Within normal variation',
            })
    except Exception as e:
        results.append({
            'test': 'ARIMA_COVID_shock',
            'description': f'ARIMA failed: {str(e)}',
            'statistic': np.nan,
            'p_value': np.nan,
            'conclusion': 'Test could not be completed',
        })

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUTPUT_DIR, 'time_series_tests.csv'), index=False)
    print(f"[10] Time-series tests:")
    for _, row in out.iterrows():
        print(f"     {row['test']}: stat={row['statistic']:.3f}, p={row['p_value']:.4f} -> {row['conclusion']}")
    return out


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def compile_summary(gap_ci, beta_p, sigma_t, covid_t, decade_t, partial_c, sens):
    """Compile all test results into one summary CSV."""
    rows = []

    # Gap significance (most recent year)
    if gap_ci is not None and len(gap_ci) > 0:
        last = gap_ci.iloc[-1]
        rows.append({
            'test_name': 'BZ Gap (most recent year)',
            'statistic_type': 'gap',
            'statistic_value': last['gap'],
            'p_value': last['p_value'],
            'ci_lower': last['ci_lower'],
            'ci_upper': last['ci_upper'],
            'conclusion': f"Gap of {last['gap']:.1f} years, {'significant' if last['significant_at_05'] else 'not significant'} at p<0.05",
        })

    # Beta convergence (each decade)
    if beta_p is not None:
        for _, row in beta_p.iterrows():
            rows.append({
                'test_name': f"Beta Convergence {row['decade']}",
                'statistic_type': 'pearson_r',
                'statistic_value': row['beta_r'],
                'p_value': row['p_value_bonferroni'],
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'conclusion': f"r={row['beta_r']:.3f}, {'significant' if row['significant_at_05'] else 'not significant'} after Bonferroni",
            })

    # Sigma convergence
    if sigma_t is not None:
        for _, row in sigma_t.iterrows():
            rows.append({
                'test_name': f"Sigma Convergence ({row['period']})",
                'statistic_type': 'OLS_slope',
                'statistic_value': row['slope'],
                'p_value': row['slope_p_value'],
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'conclusion': f"Slope={row['slope']:.4f}/yr, {'declining' if row['sigma_declining'] else 'not declining'}",
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'statistical_tests_summary.csv'), index=False)
    print(f"\nSummary: {len(summary)} test results saved to statistical_tests_summary.csv")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS - BLUE ZONES ANALYSIS")
    print("=" * 70)

    df = load_panel()
    print(f"Loaded: {len(df)} rows, {df['iso_code'].nunique()} countries\n")

    # Step 2: Statistical tests
    gap_ci = gap_confidence_intervals(df)
    print()
    beta_p = beta_convergence_pvalues(df)
    print()
    sigma_t = sigma_convergence_test(df)
    print()
    covid_t = covid_acceleration_test(df)
    print()
    decade_t = decade_improvement_tests(df)
    print()

    # Step 3: GDP controls
    partial_c = partial_correlations(df)
    print()
    multiple_regression(df)
    print()

    # Step 5: Population-weighted
    population_weighted_averages(df)
    print()

    # Step 6: Sensitivity
    sens = sensitivity_drop_one(df)
    print()

    # Step 10: Time-series
    time_series_tests(df)
    print()

    # Summary
    compile_summary(gap_ci, beta_p, sigma_t, covid_t, decade_t, partial_c, sens)

    print("\nAll statistical tests complete.")


if __name__ == '__main__':
    main()
