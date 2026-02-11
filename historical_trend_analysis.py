#!/usr/bin/env python3
"""
Historical Trend Analysis for Blue Zones Longevity Study
Analyzes 60+ years of real World Bank/WHO data to answer:
1. How have Blue Zone countries changed over time vs global averages?
2. Is the world converging toward Blue Zone longevity levels?
3. What are the decade-by-decade improvement rates?

All analysis uses REAL data only.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

BLUE_ZONE_ISOS = {'USA', 'JPN', 'ITA', 'GRC', 'CRI'}

# Regional groupings for peer comparison
REGIONS = {
    'East Asia': ['JPN', 'KOR', 'CHN', 'SGP', 'THA', 'VNM', 'MYS', 'IDN', 'PHL', 'MMR'],
    'Western Europe': ['ITA', 'FRA', 'DEU', 'ESP', 'GBR', 'NLD', 'BEL', 'AUT', 'CHE', 'PRT'],
    'Southern Europe': ['GRC', 'ITA', 'ESP', 'PRT', 'ALB', 'HRV', 'SVN', 'MNE', 'MKD', 'BIH', 'SRB', 'BGR'],
    'Central America & Caribbean': ['CRI', 'MEX', 'COL', 'VEN', 'ECU', 'PER'],
    'North America': ['USA', 'CAN', 'MEX'],
    'Nordics': ['NOR', 'SWE', 'DNK', 'FIN', 'ISL'],
    'Sub-Saharan Africa': ['NGA', 'KEN', 'ETH', 'GHA', 'TZA', 'UGA', 'ZAF', 'MOZ', 'MDG', 'BWA', 'NAM', 'ZMB', 'ZWE'],
}


def load_historical_data():
    """Load the merged historical panel data."""
    path = os.path.join(PROJECT_DIR, 'data', 'historical', 'merged_historical_panel.csv')
    if not os.path.exists(path):
        logger.error(f"Historical data not found at {path}")
        logger.error("Run historical_data_collector.py first")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded historical data: {len(df)} rows, {df['iso_code'].nunique()} countries, years {df['year'].min()}-{df['year'].max()}")
    return df


def load_projections():
    """Load UN projection data."""
    path = os.path.join(PROJECT_DIR, 'data', 'projections', 'un_life_expectancy_projections.csv')
    if not os.path.exists(path):
        logger.warning(f"Projections not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded projections: {len(df)} rows")
    return df


def analyze_blue_zone_vs_global(df):
    """
    Analysis 1: Blue Zone countries vs global average life expectancy over time.
    Returns a DataFrame with yearly averages.
    """
    logger.info("Analyzing Blue Zone countries vs global averages...")

    le_data = df[['iso_code', 'year', 'life_expectancy', 'is_blue_zone']].dropna(subset=['life_expectancy'])

    # Global average per year (unweighted - each country counts equally)
    global_avg = le_data.groupby('year')['life_expectancy'].agg(['mean', 'median', 'std', 'count']).reset_index()
    global_avg.columns = ['year', 'global_mean', 'global_median', 'global_std', 'n_countries']

    # Blue Zone country average per year
    bz_data = le_data[le_data['is_blue_zone'] == 1]
    bz_avg = bz_data.groupby('year')['life_expectancy'].agg(['mean', 'count']).reset_index()
    bz_avg.columns = ['year', 'blue_zone_mean', 'bz_n_countries']

    # Non-Blue Zone average
    non_bz = le_data[le_data['is_blue_zone'] == 0]
    non_bz_avg = non_bz.groupby('year')['life_expectancy'].agg(['mean']).reset_index()
    non_bz_avg.columns = ['year', 'non_bz_mean']

    # Top quartile and bottom quartile per year
    q25 = le_data.groupby('year')['life_expectancy'].quantile(0.25).reset_index()
    q25.columns = ['year', 'global_q25']
    q75 = le_data.groupby('year')['life_expectancy'].quantile(0.75).reset_index()
    q75.columns = ['year', 'global_q75']

    # Merge all
    result = global_avg.merge(bz_avg, on='year', how='left')
    result = result.merge(non_bz_avg, on='year', how='left')
    result = result.merge(q25, on='year', how='left')
    result = result.merge(q75, on='year', how='left')

    # Calculate gap
    result['bz_gap_over_global'] = result['blue_zone_mean'] - result['global_mean']
    result['bz_gap_over_median'] = result['blue_zone_mean'] - result['global_median']

    return result


def analyze_convergence(df):
    """
    Analysis 2: Sigma-convergence and beta-convergence.
    Sigma: Is the global spread of life expectancy shrinking?
    Beta: Are initially poorer countries improving faster?
    """
    logger.info("Analyzing convergence patterns...")

    le_data = df[['iso_code', 'year', 'life_expectancy']].dropna(subset=['life_expectancy'])

    # Sigma-convergence: standard deviation of LE across countries per year
    sigma = le_data.groupby('year')['life_expectancy'].agg(['std', 'count', 'mean']).reset_index()
    sigma.columns = ['year', 'le_std', 'n_countries', 'le_mean']
    sigma['coefficient_of_variation'] = sigma['le_std'] / sigma['le_mean']

    # Beta-convergence: for each decade pair, regress growth rate on initial level
    beta_results = []
    decades = list(range(1960, 2021, 10))

    for i in range(len(decades) - 1):
        start_year = decades[i]
        end_year = decades[i + 1]

        # Get countries with data in both years (allow +/- 2 year window)
        start_data = le_data[(le_data['year'] >= start_year) & (le_data['year'] <= start_year + 2)]
        start_data = start_data.sort_values('year').groupby('iso_code').first().reset_index()

        end_data = le_data[(le_data['year'] >= end_year) & (le_data['year'] <= end_year + 2)]
        end_data = end_data.sort_values('year').groupby('iso_code').first().reset_index()

        merged = start_data[['iso_code', 'life_expectancy']].merge(
            end_data[['iso_code', 'life_expectancy']],
            on='iso_code', suffixes=('_start', '_end')
        )

        if len(merged) < 5:
            continue

        merged['le_gain'] = merged['life_expectancy_end'] - merged['life_expectancy_start']
        merged['is_blue_zone'] = merged['iso_code'].isin(BLUE_ZONE_ISOS).astype(int)

        # Simple correlation: do countries that started lower gain more?
        corr = merged['life_expectancy_start'].corr(merged['le_gain'])

        # Average gain for Blue Zone vs non-Blue Zone
        bz_gain = merged[merged['is_blue_zone'] == 1]['le_gain'].mean()
        non_bz_gain = merged[merged['is_blue_zone'] == 0]['le_gain'].mean()

        beta_results.append({
            'decade': f"{start_year}s",
            'start_year': start_year,
            'end_year': end_year,
            'n_countries': len(merged),
            'avg_gain_years': merged['le_gain'].mean(),
            'bz_avg_gain': bz_gain,
            'non_bz_avg_gain': non_bz_gain,
            'beta_correlation': corr,
            'convergence': 'Yes' if corr < -0.1 else ('Weak' if corr < 0 else 'No'),
        })

    sigma_df = sigma
    beta_df = pd.DataFrame(beta_results)
    return sigma_df, beta_df


def analyze_country_profiles(df):
    """
    Analysis 3: Individual profiles for each Blue Zone country
    with comparison to regional peers.
    """
    logger.info("Building Blue Zone country profiles...")

    le_data = df[['iso_code', 'year', 'life_expectancy', 'gdp_per_capita',
                   'physicians_per_1000', 'country_name']].copy()

    profiles = {}

    bz_regional_peers = {
        'JPN': 'East Asia',
        'ITA': 'Western Europe',
        'GRC': 'Southern Europe',
        'CRI': 'Central America & Caribbean',
        'USA': 'North America',
    }

    for iso, region_name in bz_regional_peers.items():
        country_data = le_data[le_data['iso_code'] == iso].copy()
        country_data = country_data.sort_values('year')

        # Regional peer average (excluding the Blue Zone country itself)
        peer_isos = [c for c in REGIONS.get(region_name, []) if c != iso]
        peer_data = le_data[le_data['iso_code'].isin(peer_isos)]
        peer_avg = peer_data.groupby('year')['life_expectancy'].mean().reset_index()
        peer_avg.columns = ['year', 'peer_avg_le']

        # Merge country + peer data
        merged = country_data.merge(peer_avg, on='year', how='outer')
        merged['iso_code'] = iso
        merged['country_name'] = merged['country_name'].fillna(
            df[df['iso_code'] == iso]['country_name'].iloc[0] if len(df[df['iso_code'] == iso]) > 0 else iso
        )
        merged['region'] = region_name
        merged['le_advantage_over_peers'] = merged['life_expectancy'] - merged['peer_avg_le']
        merged = merged.sort_values('year')

        profiles[iso] = merged

    all_profiles = pd.concat(profiles.values(), ignore_index=True)
    return all_profiles


def analyze_decade_improvements(df):
    """
    Analysis 4: Life expectancy improvement rates by decade.
    How many years of LE were gained per decade?
    """
    logger.info("Calculating decade-by-decade improvement rates...")

    le_data = df[['iso_code', 'year', 'life_expectancy', 'is_blue_zone']].dropna(subset=['life_expectancy'])

    results = []
    decades = [(1960, 1970), (1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2020)]

    for start, end in decades:
        for group_name, group_filter in [('blue_zone', True), ('non_blue_zone', False), ('global', None)]:
            if group_filter is None:
                subset = le_data
            elif group_filter:
                subset = le_data[le_data['is_blue_zone'] == 1]
            else:
                subset = le_data[le_data['is_blue_zone'] == 0]

            # Get values at start and end of decade (with 2-year window)
            start_vals = subset[(subset['year'] >= start) & (subset['year'] <= start + 2)]
            start_avg = start_vals.groupby('iso_code')['life_expectancy'].first()

            end_vals = subset[(subset['year'] >= end) & (subset['year'] <= end + 2)]
            end_avg = end_vals.groupby('iso_code')['life_expectancy'].first()

            common = start_avg.index.intersection(end_avg.index)
            if len(common) == 0:
                continue

            gains = end_avg[common] - start_avg[common]

            results.append({
                'decade': f"{start}-{end}",
                'group': group_name,
                'avg_le_start': start_avg[common].mean(),
                'avg_le_end': end_avg[common].mean(),
                'avg_gain': gains.mean(),
                'median_gain': gains.median(),
                'max_gain': gains.max(),
                'min_gain': gains.min(),
                'n_countries': len(common),
            })

    return pd.DataFrame(results)


def main():
    """Run all analyses and save results."""
    df = load_historical_data()
    if df.empty:
        return

    output_dir = os.path.join(PROJECT_DIR, 'outputs', 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Analysis 1: Blue Zone vs Global
    bz_vs_global = analyze_blue_zone_vs_global(df)
    bz_vs_global.to_csv(os.path.join(output_dir, 'blue_zone_vs_global.csv'), index=False)

    # Analysis 2: Convergence
    sigma_df, beta_df = analyze_convergence(df)
    sigma_df.to_csv(os.path.join(output_dir, 'sigma_convergence.csv'), index=False)
    beta_df.to_csv(os.path.join(output_dir, 'beta_convergence.csv'), index=False)

    # Analysis 3: Country profiles
    profiles = analyze_country_profiles(df)
    profiles.to_csv(os.path.join(output_dir, 'blue_zone_country_profiles.csv'), index=False)

    # Analysis 4: Decade improvements
    decade_improvements = analyze_decade_improvements(df)
    decade_improvements.to_csv(os.path.join(output_dir, 'decade_improvements.csv'), index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("HISTORICAL TREND ANALYSIS COMPLETE")
    print("=" * 70)

    print("\n--- BLUE ZONE vs GLOBAL LIFE EXPECTANCY ---")
    key_years = bz_vs_global[bz_vs_global['year'].isin([1960, 1970, 1980, 1990, 2000, 2010, 2020])]
    if not key_years.empty:
        for _, row in key_years.iterrows():
            bz = row.get('blue_zone_mean', float('nan'))
            gl = row.get('global_mean', float('nan'))
            gap = row.get('bz_gap_over_global', float('nan'))
            n = int(row.get('n_countries', 0))
            print(f"  {int(row['year'])}: BZ countries={bz:.1f}, Global={gl:.1f}, Gap={gap:+.1f} years (n={n} countries)")

    print("\n--- CONVERGENCE (Beta by Decade) ---")
    if not beta_df.empty:
        for _, row in beta_df.iterrows():
            print(f"  {row['decade']}: Avg gain={row['avg_gain_years']:.1f}yr, "
                  f"BZ gain={row['bz_avg_gain']:.1f}yr, "
                  f"Non-BZ gain={row['non_bz_avg_gain']:.1f}yr, "
                  f"Convergence={row['convergence']} (r={row['beta_correlation']:.3f})")

    print("\n--- SIGMA CONVERGENCE (Global LE Spread Over Time) ---")
    for yr in [1960, 1980, 2000, 2020]:
        row = sigma_df[sigma_df['year'] == yr]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {yr}: SD={r['le_std']:.2f} years, CV={r['coefficient_of_variation']:.4f}, n={int(r['n_countries'])}")

    print("\n--- DECADE IMPROVEMENT RATES ---")
    if not decade_improvements.empty:
        for _, row in decade_improvements[decade_improvements['group'] == 'global'].iterrows():
            print(f"  {row['decade']}: Global avg gain = {row['avg_gain']:.1f} years")

    print(f"\nAll results saved to: {output_dir}/")
    return bz_vs_global, sigma_df, beta_df, profiles, decade_improvements


if __name__ == '__main__':
    main()
