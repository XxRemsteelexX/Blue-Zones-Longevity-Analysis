#!/usr/bin/env python3
"""
Regional Peer Comparison and Income Group Analysis for Blue Zones

Covers:
  - Each BZ country compared to its regional peers over time
  - Statistical tests on regional outlier status
  - World Bank income group classification
  - Convergence analysis within income groups
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'analysis')
FIGURE_DIR = os.path.join(PROJECT_DIR, 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

BLUE_ZONE_ISOS = {'USA', 'JPN', 'ITA', 'GRC', 'CRI'}

# Regional groupings (each BZ country mapped to its geographic region peers)
BZ_REGIONAL_PEERS = {
    'JPN': {
        'region_name': 'East Asia & Pacific',
        'peers': ['JPN', 'CHN', 'KOR', 'SGP', 'THA', 'VNM', 'MYS', 'IDN', 'PHL', 'MMR', 'AUS', 'NZL', 'PNG', 'FJI'],
    },
    'ITA': {
        'region_name': 'Southern Europe',
        'peers': ['ITA', 'ESP', 'GRC', 'PRT', 'HRV', 'SVN', 'ALB', 'MNE', 'MKD', 'BIH', 'SRB'],
    },
    'GRC': {
        'region_name': 'Southern Europe',
        'peers': ['ITA', 'ESP', 'GRC', 'PRT', 'HRV', 'SVN', 'ALB', 'MNE', 'MKD', 'BIH', 'SRB'],
    },
    'CRI': {
        'region_name': 'Central & South America',
        'peers': ['CRI', 'MEX', 'COL', 'ECU', 'PER', 'CHL', 'ARG', 'BRA', 'BOL', 'PRY', 'URY', 'VEN'],
    },
    'USA': {
        'region_name': 'North America & Western Europe',
        'peers': ['USA', 'CAN', 'GBR', 'FRA', 'DEU', 'NLD', 'BEL', 'CHE', 'AUT', 'IRL', 'LUX'],
    },
}

# World Bank income group classification (2024 thresholds applied to 2019 GDP per capita)
# Low: <$1,145; Lower-middle: $1,146-$4,515; Upper-middle: $4,516-$14,005; High: >$14,005
INCOME_THRESHOLDS = {
    'Low income': 1145,
    'Lower-middle income': 4515,
    'Upper-middle income': 14005,
}


def load_panel():
    path = os.path.join(PROJECT_DIR, 'data', 'historical', 'merged_historical_panel.csv')
    df = pd.read_csv(path)
    df['is_blue_zone'] = df['iso_code'].isin(BLUE_ZONE_ISOS).astype(int)
    return df


def classify_income_group(gdp):
    """Classify a GDP per capita value into World Bank income group."""
    if pd.isna(gdp):
        return 'Unknown'
    if gdp <= INCOME_THRESHOLDS['Low income']:
        return 'Low income'
    elif gdp <= INCOME_THRESHOLDS['Lower-middle income']:
        return 'Lower-middle income'
    elif gdp <= INCOME_THRESHOLDS['Upper-middle income']:
        return 'Upper-middle income'
    else:
        return 'High income'


# ---------------------------------------------------------------------------
# 7. Regional Peer Comparison
# ---------------------------------------------------------------------------
def regional_peer_comparison(df):
    """Compare each BZ country to its regional peer average over time."""
    all_results = []

    for bz_iso, config in BZ_REGIONAL_PEERS.items():
        region_name = config['region_name']
        peer_isos = config['peers']

        for year in sorted(df['year'].unique()):
            yr = df[(df['year'] == year) & (df['iso_code'].isin(peer_isos))].dropna(subset=['life_expectancy'])
            bz_row = yr[yr['iso_code'] == bz_iso]
            peers = yr[yr['iso_code'] != bz_iso]

            if len(bz_row) == 0 or len(peers) < 3:
                continue

            bz_le = bz_row['life_expectancy'].values[0]
            peer_mean = peers['life_expectancy'].mean()
            peer_std = peers['life_expectancy'].std()
            peer_max = peers['life_expectancy'].max()

            all_results.append({
                'bz_country': bz_iso,
                'region': region_name,
                'year': year,
                'bz_le': bz_le,
                'regional_mean': peer_mean,
                'regional_std': peer_std,
                'regional_max': peer_max,
                'advantage_over_region': bz_le - peer_mean,
                'n_peers': len(peers),
                'rank_in_region': 1 + (peers['life_expectancy'] > bz_le).sum(),
                'total_in_region': len(peers) + 1,
            })

    out = pd.DataFrame(all_results)

    # Statistical test: t-test on decade means (BZ country vs regional avg)
    test_results = []
    for bz_iso in BZ_REGIONAL_PEERS:
        country_data = out[out['bz_country'] == bz_iso]
        if len(country_data) < 20:
            continue

        # Use 2000-2019 for the test
        recent = country_data[(country_data['year'] >= 2000) & (country_data['year'] <= 2019)]
        if len(recent) < 10:
            continue

        advantage = recent['advantage_over_region'].values
        # Test if advantage is significantly different from 0
        t_stat, p_val = ttest_ind(advantage, np.zeros(len(advantage)))

        test_results.append({
            'bz_country': bz_iso,
            'region': BZ_REGIONAL_PEERS[bz_iso]['region_name'],
            'mean_advantage_2000_2019': advantage.mean(),
            'std_advantage': advantage.std(),
            't_statistic': t_stat,
            'p_value': p_val,
            'is_regional_outlier': p_val < 0.05 and advantage.mean() > 0,
        })

    test_df = pd.DataFrame(test_results)

    out.to_csv(os.path.join(OUTPUT_DIR, 'regional_peer_comparison.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'regional_outlier_tests.csv'), index=False)

    print(f"[7] Regional peer comparison: {len(out)} country-year rows")
    for _, row in test_df.iterrows():
        sig = "OUTLIER" if row['is_regional_outlier'] else "not outlier"
        print(f"     {row['bz_country']} vs {row['region']}: advantage={row['mean_advantage_2000_2019']:.1f} yr, "
              f"p={row['p_value']:.4f} -> {sig}")

    return out, test_df


def plot_regional_peers(comparison_df):
    """5-panel chart: each BZ country vs its regional peers."""
    bz_countries = ['JPN', 'ITA', 'GRC', 'CRI', 'USA']
    bz_names = {'JPN': 'Japan', 'ITA': 'Italy', 'GRC': 'Greece', 'CRI': 'Costa Rica', 'USA': 'United States'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, bz_iso in enumerate(bz_countries):
        ax = axes[idx]
        data = comparison_df[comparison_df['bz_country'] == bz_iso]
        if data.empty:
            continue

        region = BZ_REGIONAL_PEERS[bz_iso]['region_name']

        ax.plot(data['year'], data['bz_le'], 'b-', linewidth=2.5, label=bz_names[bz_iso])
        ax.plot(data['year'], data['regional_mean'], 'gray', linewidth=2, label=f'{region} Avg')
        ax.fill_between(data['year'],
                         data['regional_mean'] - data['regional_std'],
                         data['regional_mean'] + data['regional_std'],
                         alpha=0.15, color='gray', label='Regional +/- 1 SD')
        ax.axvspan(2020, 2023, alpha=0.08, color='red')
        ax.set_title(f"{bz_names[bz_iso]} vs {region}", fontsize=12)
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Life Expectancy', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[5].set_visible(False)

    plt.suptitle('Blue Zone Countries vs Regional Peers', fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'nb07_regional_peers.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 8. Income Group Subanalysis
# ---------------------------------------------------------------------------
def income_group_analysis(df):
    """Classify countries by income, run convergence within each group."""
    # Use 2019 GDP to classify
    gdp_2019 = df[df['year'] == 2019][['iso_code', 'gdp_per_capita']].dropna()
    gdp_2019['income_group'] = gdp_2019['gdp_per_capita'].apply(classify_income_group)

    # For countries without 2019 GDP, try nearest year
    classified_isos = set(gdp_2019['iso_code'])
    all_isos = set(df['iso_code'].unique())
    missing = all_isos - classified_isos

    for iso in missing:
        country_gdp = df[(df['iso_code'] == iso) & (df['gdp_per_capita'].notna())].sort_values('year', ascending=False)
        if not country_gdp.empty:
            latest_gdp = country_gdp.iloc[0]['gdp_per_capita']
            gdp_2019 = pd.concat([gdp_2019, pd.DataFrame([{
                'iso_code': iso,
                'gdp_per_capita': latest_gdp,
                'income_group': classify_income_group(latest_gdp),
            }])], ignore_index=True)

    income_map = dict(zip(gdp_2019['iso_code'], gdp_2019['income_group']))
    df['income_group'] = df['iso_code'].map(income_map)

    print(f"\n[8] Income group classification:")
    for group in ['Low income', 'Lower-middle income', 'Upper-middle income', 'High income', 'Unknown']:
        n = gdp_2019[gdp_2019['income_group'] == group]['iso_code'].nunique()
        if n > 0:
            print(f"     {group}: {n} countries")

    # Convergence within each income group
    convergence_results = []
    groups = ['Low income', 'Lower-middle income', 'Upper-middle income', 'High income']

    for group in groups:
        group_df = df[df['income_group'] == group]
        n_countries = group_df['iso_code'].nunique()

        if n_countries < 5:
            continue

        # Sigma convergence: SD of LE over time
        sigma = group_df.groupby('year')['life_expectancy'].agg(['mean', 'std', 'count']).dropna()
        sigma = sigma[sigma['count'] >= 5]

        if len(sigma) < 10:
            continue

        # Trend in SD
        from scipy.stats import pearsonr
        years = sigma.index.values.astype(float)
        stds = sigma['std'].values
        r, p = pearsonr(years, stds)

        # LE improvement (first to last decade)
        first = sigma.iloc[:10]['mean'].mean()
        last = sigma.iloc[-10:]['mean'].mean()

        convergence_results.append({
            'income_group': group,
            'n_countries': n_countries,
            'le_1960s': first,
            'le_recent': last,
            'le_improvement': last - first,
            'sigma_trend_r': r,
            'sigma_trend_p': p,
            'sigma_converging': r < 0 and p < 0.05,
            'sigma_start': sigma.iloc[0]['std'],
            'sigma_end': sigma.iloc[-1]['std'],
        })

    conv_df = pd.DataFrame(convergence_results)
    conv_df.to_csv(os.path.join(OUTPUT_DIR, 'income_group_convergence.csv'), index=False)

    print(f"\n     Convergence by income group:")
    for _, row in conv_df.iterrows():
        conv = "converging" if row['sigma_converging'] else "not converging"
        print(f"     {row['income_group']}: {row['n_countries']} countries, "
              f"LE improvement={row['le_improvement']:.1f} yr, sigma {conv} (r={row['sigma_trend_r']:.3f})")

    # Save income classification for later use
    income_class = gdp_2019[['iso_code', 'income_group']].copy()
    income_class.to_csv(os.path.join(OUTPUT_DIR, 'country_income_groups.csv'), index=False)

    return conv_df


def plot_income_convergence(df):
    """Plot LE trends by income group."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'High income': '#2166ac',
        'Upper-middle income': '#67a9cf',
        'Lower-middle income': '#ef8a62',
        'Low income': '#b2182b',
    }

    income_map_path = os.path.join(OUTPUT_DIR, 'country_income_groups.csv')
    if not os.path.exists(income_map_path):
        return

    income_map = pd.read_csv(income_map_path)
    imap = dict(zip(income_map['iso_code'], income_map['income_group']))
    df['income_group'] = df['iso_code'].map(imap)

    for group in ['High income', 'Upper-middle income', 'Lower-middle income', 'Low income']:
        group_data = df[df['income_group'] == group]
        yearly = group_data.groupby('year')['life_expectancy'].mean()
        if len(yearly) > 5:
            ax.plot(yearly.index, yearly.values, color=colors.get(group, 'gray'),
                    linewidth=2.5, label=group)

    ax.axvspan(2020, 2023, alpha=0.1, color='red')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Life Expectancy (years)', fontsize=12)
    ax.set_title('Life Expectancy Trends by Income Group', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'nb07_income_convergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 70)
    print("REGIONAL PEER COMPARISON & INCOME GROUP ANALYSIS")
    print("=" * 70)

    df = load_panel()
    print(f"Loaded: {len(df)} rows, {df['iso_code'].nunique()} countries\n")

    # Step 7: Regional peer comparison
    comparison_df, test_df = regional_peer_comparison(df)
    print()
    plot_regional_peers(comparison_df)

    # Step 8: Income group analysis
    income_df = income_group_analysis(df)
    print()
    plot_income_convergence(df)

    print("\nRegional and income group analysis complete.")


if __name__ == '__main__':
    main()
