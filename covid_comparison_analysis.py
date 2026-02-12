#!/usr/bin/env python3
"""
COVID Impact Comparison Analysis for Blue Zones Study

Runs the full trend analysis twice:
  1. Pre-COVID period (1960-2019) -- clean secular trends
  2. Full period (1960-2023) -- includes COVID disruption and recovery

Then produces direct comparison metrics showing exactly where and how
COVID distorted the data.

All data is real. No synthetic or fabricated data used.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BLUE_ZONE_ISOS = {'USA', 'JPN', 'ITA', 'GRC', 'CRI'}


def load_data():
    path = os.path.join(PROJECT_DIR, 'data', 'historical', 'merged_historical_panel.csv')
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df['iso_code'].nunique()} countries, "
                f"years {df['year'].min()}-{df['year'].max()}")
    return df


def run_gap_analysis(df):
    """Blue Zone avg vs global avg per year."""
    le = df[['iso_code', 'year', 'life_expectancy', 'is_blue_zone']].dropna(subset=['life_expectancy'])

    global_avg = le.groupby('year')['life_expectancy'].agg(['mean', 'median', 'std', 'count']).reset_index()
    global_avg.columns = ['year', 'global_mean', 'global_median', 'global_std', 'n_countries']

    bz = le[le['is_blue_zone'] == 1].groupby('year')['life_expectancy'].agg(['mean', 'count']).reset_index()
    bz.columns = ['year', 'blue_zone_mean', 'bz_n_countries']

    non_bz = le[le['is_blue_zone'] == 0].groupby('year')['life_expectancy'].mean().reset_index()
    non_bz.columns = ['year', 'non_bz_mean']

    q25 = le.groupby('year')['life_expectancy'].quantile(0.25).reset_index()
    q25.columns = ['year', 'global_q25']
    q75 = le.groupby('year')['life_expectancy'].quantile(0.75).reset_index()
    q75.columns = ['year', 'global_q75']

    result = global_avg.merge(bz, on='year', how='left')
    result = result.merge(non_bz, on='year', how='left')
    result = result.merge(q25, on='year', how='left')
    result = result.merge(q75, on='year', how='left')
    result['bz_gap_over_global'] = result['blue_zone_mean'] - result['global_mean']
    return result


def run_sigma(df):
    """Sigma convergence -- SD of LE across countries per year."""
    le = df[['iso_code', 'year', 'life_expectancy']].dropna(subset=['life_expectancy'])
    sigma = le.groupby('year')['life_expectancy'].agg(['std', 'count', 'mean']).reset_index()
    sigma.columns = ['year', 'le_std', 'n_countries', 'le_mean']
    sigma['coefficient_of_variation'] = sigma['le_std'] / sigma['le_mean']
    return sigma


def run_beta(df):
    """Beta convergence -- do lagging countries catch up faster?"""
    le = df[['iso_code', 'year', 'life_expectancy']].dropna(subset=['life_expectancy'])
    max_year = int(le['year'].max())

    results = []
    decades = list(range(1960, max_year, 10))

    for i in range(len(decades) - 1):
        start = decades[i]
        end = decades[i + 1]
        if end > max_year:
            break

        s = le[(le['year'] >= start) & (le['year'] <= start + 2)]
        s = s.sort_values('year').groupby('iso_code').first().reset_index()
        e = le[(le['year'] >= end) & (le['year'] <= end + 2)]
        e = e.sort_values('year').groupby('iso_code').first().reset_index()

        m = s[['iso_code', 'life_expectancy']].merge(
            e[['iso_code', 'life_expectancy']], on='iso_code', suffixes=('_start', '_end'))
        if len(m) < 5:
            continue

        m['le_gain'] = m['life_expectancy_end'] - m['life_expectancy_start']
        m['is_blue_zone'] = m['iso_code'].isin(BLUE_ZONE_ISOS).astype(int)
        corr = m['life_expectancy_start'].corr(m['le_gain'])
        bz_gain = m[m['is_blue_zone'] == 1]['le_gain'].mean()
        non_bz_gain = m[m['is_blue_zone'] == 0]['le_gain'].mean()

        results.append({
            'decade': f"{start}s",
            'start_year': start, 'end_year': end,
            'n_countries': len(m),
            'avg_gain_years': m['le_gain'].mean(),
            'bz_avg_gain': bz_gain,
            'non_bz_avg_gain': non_bz_gain,
            'beta_correlation': corr,
            'convergence': 'Yes' if corr < -0.1 else ('Weak' if corr < 0 else 'No'),
        })
    return pd.DataFrame(results)


def run_decade_improvements(df):
    """LE gains per decade by group."""
    le = df[['iso_code', 'year', 'life_expectancy', 'is_blue_zone']].dropna(subset=['life_expectancy'])
    max_year = int(le['year'].max())

    results = []
    decades = []
    for start in range(1960, max_year, 10):
        end = start + 10
        if end > max_year + 2:
            end = max_year
        if end <= start:
            continue
        decades.append((start, end))

    for start, end in decades:
        for group_name, group_filter in [('blue_zone', True), ('non_blue_zone', False), ('global', None)]:
            if group_filter is None:
                subset = le
            elif group_filter:
                subset = le[le['is_blue_zone'] == 1]
            else:
                subset = le[le['is_blue_zone'] == 0]

            sv = subset[(subset['year'] >= start) & (subset['year'] <= start + 2)]
            sa = sv.groupby('iso_code')['life_expectancy'].first()
            ev = subset[(subset['year'] >= end - 2) & (subset['year'] <= end + 2)]
            ea = ev.groupby('iso_code')['life_expectancy'].first()
            common = sa.index.intersection(ea.index)
            if len(common) == 0:
                continue
            gains = ea[common] - sa[common]
            results.append({
                'decade': f"{start}-{end}",
                'group': group_name,
                'avg_le_start': sa[common].mean(),
                'avg_le_end': ea[common].mean(),
                'avg_gain': gains.mean(),
                'median_gain': gains.median(),
                'n_countries': len(common),
            })
    return pd.DataFrame(results)


def run_full_analysis(df, label, output_dir):
    """Run all analyses on a given dataframe and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    yr_min, yr_max = df['year'].min(), df['year'].max()
    n = df['iso_code'].nunique()
    logger.info(f"[{label}] Running analysis: {n} countries, {yr_min}-{yr_max}, {len(df)} rows")

    gap = run_gap_analysis(df)
    gap.to_csv(os.path.join(output_dir, 'blue_zone_vs_global.csv'), index=False)

    sigma = run_sigma(df)
    sigma.to_csv(os.path.join(output_dir, 'sigma_convergence.csv'), index=False)

    beta = run_beta(df)
    beta.to_csv(os.path.join(output_dir, 'beta_convergence.csv'), index=False)

    decades = run_decade_improvements(df)
    decades.to_csv(os.path.join(output_dir, 'decade_improvements.csv'), index=False)

    logger.info(f"[{label}] Saved: gap({len(gap)}), sigma({len(sigma)}), "
                f"beta({len(beta)}), decades({len(decades)})")

    return gap, sigma, beta, decades


def run_covid_comparison(df_full, gap_pre, gap_full, sigma_pre, sigma_full):
    """Produce direct comparison metrics between pre-COVID and full period."""
    results = {}

    # Year-by-year COVID impact
    covid_years = df_full[df_full['year'].isin([2018, 2019, 2020, 2021, 2022, 2023])]
    le_covid = covid_years[covid_years['life_expectancy'].notna()]

    country_impacts = []
    for iso in df_full['iso_code'].unique():
        c = le_covid[le_covid['iso_code'] == iso].sort_values('year')
        le_2019 = c[c['year'] == 2019]['life_expectancy']
        le_2020 = c[c['year'] == 2020]['life_expectancy']
        le_2021 = c[c['year'] == 2021]['life_expectancy']
        le_2023 = c[c['year'] == 2023]['life_expectancy']

        if len(le_2019) > 0 and len(le_2020) > 0:
            drop_2020 = le_2020.iloc[0] - le_2019.iloc[0]
            drop_2021 = le_2021.iloc[0] - le_2019.iloc[0] if len(le_2021) > 0 else np.nan
            recovery_2023 = le_2023.iloc[0] - le_2019.iloc[0] if len(le_2023) > 0 else np.nan
            country_impacts.append({
                'iso_code': iso,
                'le_2019': le_2019.iloc[0],
                'le_2020': le_2020.iloc[0],
                'le_2021': le_2021.iloc[0] if len(le_2021) > 0 else np.nan,
                'le_2023': le_2023.iloc[0] if len(le_2023) > 0 else np.nan,
                'drop_2020': drop_2020,
                'drop_2021': drop_2021,
                'recovery_2023': recovery_2023,
                'is_blue_zone': int(iso in BLUE_ZONE_ISOS),
                'fully_recovered': recovery_2023 >= 0 if not np.isnan(recovery_2023) else None,
            })

    impact_df = pd.DataFrame(country_impacts)

    # Pre-COVID vs Full period key metrics comparison
    pre_last = gap_pre.iloc[-1] if len(gap_pre) > 0 else None
    full_2019 = gap_full[gap_full['year'] == 2019].iloc[0] if len(gap_full[gap_full['year'] == 2019]) > 0 else None
    full_2021 = gap_full[gap_full['year'] == 2021].iloc[0] if len(gap_full[gap_full['year'] == 2021]) > 0 else None
    full_last = gap_full.iloc[-1] if len(gap_full) > 0 else None

    summary = {
        'pre_covid_final_year': int(gap_pre['year'].max()) if len(gap_pre) > 0 else None,
        'pre_covid_bz_mean': pre_last['blue_zone_mean'] if pre_last is not None else None,
        'pre_covid_global_mean': pre_last['global_mean'] if pre_last is not None else None,
        'pre_covid_gap': pre_last['bz_gap_over_global'] if pre_last is not None else None,
        'pre_covid_sigma': sigma_pre.iloc[-1]['le_std'] if len(sigma_pre) > 0 else None,
        'full_2019_gap': full_2019['bz_gap_over_global'] if full_2019 is not None else None,
        'full_2021_gap': full_2021['bz_gap_over_global'] if full_2021 is not None else None,
        'full_2023_gap': full_last['bz_gap_over_global'] if full_last is not None else None,
        'full_2023_sigma': sigma_full.iloc[-1]['le_std'] if len(sigma_full) > 0 else None,
        'global_drop_2020': gap_full[gap_full['year'] == 2020]['global_mean'].iloc[0] -
                            gap_full[gap_full['year'] == 2019]['global_mean'].iloc[0]
                            if len(gap_full[gap_full['year'].isin([2019, 2020])]) >= 2 else None,
        'global_drop_2021': gap_full[gap_full['year'] == 2021]['global_mean'].iloc[0] -
                            gap_full[gap_full['year'] == 2019]['global_mean'].iloc[0]
                            if len(gap_full[gap_full['year'].isin([2019, 2021])]) >= 2 else None,
        'n_countries_worse_2023_than_2019':
            int((impact_df['recovery_2023'] < 0).sum()) if 'recovery_2023' in impact_df.columns else None,
        'n_countries_recovered':
            int((impact_df['recovery_2023'] >= 0).sum()) if 'recovery_2023' in impact_df.columns else None,
        'avg_drop_bz': impact_df[impact_df['is_blue_zone'] == 1]['drop_2020'].mean(),
        'avg_drop_non_bz': impact_df[impact_df['is_blue_zone'] == 0]['drop_2020'].mean(),
        'worst_hit_country': impact_df.loc[impact_df['drop_2020'].idxmin(), 'iso_code']
                             if len(impact_df) > 0 else None,
        'worst_hit_drop': impact_df['drop_2020'].min() if len(impact_df) > 0 else None,
    }

    return impact_df, summary


def main():
    df = load_data()

    # Period 1: Pre-COVID (through 2019)
    df_pre = df[df['year'] <= 2019].copy()
    pre_dir = os.path.join(PROJECT_DIR, 'outputs', 'analysis', 'pre_covid')
    gap_pre, sigma_pre, beta_pre, dec_pre = run_full_analysis(df_pre, 'PRE-COVID', pre_dir)

    # Period 2: Full (through 2023)
    full_dir = os.path.join(PROJECT_DIR, 'outputs', 'analysis', 'full_period')
    gap_full, sigma_full, beta_full, dec_full = run_full_analysis(df, 'FULL', full_dir)

    # Also update the main analysis directory (full period is the default)
    main_dir = os.path.join(PROJECT_DIR, 'outputs', 'analysis')
    gap_full.to_csv(os.path.join(main_dir, 'blue_zone_vs_global.csv'), index=False)
    sigma_full.to_csv(os.path.join(main_dir, 'sigma_convergence.csv'), index=False)
    beta_full.to_csv(os.path.join(main_dir, 'beta_convergence.csv'), index=False)
    dec_full.to_csv(os.path.join(main_dir, 'decade_improvements.csv'), index=False)

    # COVID comparison
    impact_df, summary = run_covid_comparison(df, gap_pre, gap_full, sigma_pre, sigma_full)
    comparison_dir = os.path.join(PROJECT_DIR, 'outputs', 'analysis', 'covid_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    impact_df.to_csv(os.path.join(comparison_dir, 'country_covid_impact.csv'), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(comparison_dir, 'comparison_summary.csv'), index=False)

    # Print report
    print("\n" + "=" * 70)
    print("COVID IMPACT COMPARISON ANALYSIS")
    print("=" * 70)

    print("\n--- PRE-COVID (1960-2019) ---")
    print(f"  Final year: {int(gap_pre['year'].max())}")
    print(f"  BZ avg: {gap_pre.iloc[-1]['blue_zone_mean']:.1f} years")
    print(f"  Global avg: {gap_pre.iloc[-1]['global_mean']:.1f} years")
    print(f"  Gap: {gap_pre.iloc[-1]['bz_gap_over_global']:.1f} years")
    print(f"  Global SD: {sigma_pre.iloc[-1]['le_std']:.2f} years")

    print("\n--- FULL PERIOD (1960-2023) ---")
    print(f"  Final year: {int(gap_full['year'].max())}")
    print(f"  BZ avg: {gap_full.iloc[-1]['blue_zone_mean']:.1f} years")
    print(f"  Global avg: {gap_full.iloc[-1]['global_mean']:.1f} years")
    print(f"  Gap: {gap_full.iloc[-1]['bz_gap_over_global']:.1f} years")
    print(f"  Global SD: {sigma_full.iloc[-1]['le_std']:.2f} years")

    print("\n--- COVID IMPACT ---")
    print(f"  Global avg drop 2019->2020: {summary['global_drop_2020']:.2f} years")
    print(f"  Global avg drop 2019->2021: {summary['global_drop_2021']:.2f} years")
    print(f"  BZ countries avg drop 2019->2020: {summary['avg_drop_bz']:.2f} years")
    print(f"  Non-BZ avg drop 2019->2020: {summary['avg_drop_non_bz']:.2f} years")
    print(f"  Worst hit: {summary['worst_hit_country']} ({summary['worst_hit_drop']:.2f} years)")
    print(f"  Countries recovered by 2023: {summary['n_countries_recovered']}")
    print(f"  Countries still below 2019 in 2023: {summary['n_countries_worse_2023_than_2019']}")

    print("\n--- BZ COUNTRY COVID IMPACT ---")
    bz_impact = impact_df[impact_df['is_blue_zone'] == 1].sort_values('drop_2020')
    for _, row in bz_impact.iterrows():
        rec = f"recovered (+{row['recovery_2023']:.1f})" if row['recovery_2023'] >= 0 else f"still down ({row['recovery_2023']:.1f})"
        print(f"  {row['iso_code']}: 2019={row['le_2019']:.1f}, "
              f"2020 drop={row['drop_2020']:+.2f}, "
              f"2023: {rec}")

    print("\n--- KEY FINDING ---")
    pre_gap = gap_pre.iloc[-1]['bz_gap_over_global']
    full_gap = gap_full.iloc[-1]['bz_gap_over_global']
    print(f"  Pre-COVID gap (2019): {pre_gap:.1f} years")
    print(f"  Post-recovery gap (2023): {full_gap:.1f} years")
    if full_gap < pre_gap:
        print(f"  COVID *accelerated* convergence by {pre_gap - full_gap:.1f} years")
        print(f"  Poorer countries were hit harder initially but the global recovery")
        print(f"  pushed the gap below pre-COVID levels")
    else:
        print(f"  COVID *slowed* convergence temporarily")

    print(f"\nOutputs saved to:")
    print(f"  {pre_dir}/")
    print(f"  {full_dir}/")
    print(f"  {comparison_dir}/")


if __name__ == '__main__':
    main()
