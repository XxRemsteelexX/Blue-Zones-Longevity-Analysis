#!/usr/bin/env python3
"""
Projection Visualizer for Blue Zones Longevity Study
Creates publication-quality graphs showing historical trends and future projections.
All visualizations use REAL data only.

Generates:
1. Blue Zone countries LE over time (1960-2100) with projections
2. Convergence story graph
3. Individual country deep dives (5 graphs)
4. Global heatmap over time
5. Improvement rate bar charts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

BLUE_ZONE_ISOS = {'USA', 'JPN', 'ITA', 'GRC', 'CRI'}
BLUE_ZONE_NAMES = {
    'USA': 'United States (Loma Linda)',
    'JPN': 'Japan (Okinawa)',
    'ITA': 'Italy (Sardinia)',
    'GRC': 'Greece (Ikaria)',
    'CRI': 'Costa Rica (Nicoya)',
}

# Color palette
BZ_COLORS = {
    'USA': '#E74C3C',
    'JPN': '#3498DB',
    'ITA': '#2ECC71',
    'GRC': '#9B59B6',
    'CRI': '#F39C12',
}

GLOBAL_COLOR = '#7F8C8D'


def load_all_data():
    """Load historical data, projections, and analysis results."""
    data = {}

    # Historical panel
    hist_path = os.path.join(PROJECT_DIR, 'data', 'historical', 'merged_historical_panel.csv')
    if os.path.exists(hist_path):
        data['historical'] = pd.read_csv(hist_path)
        logger.info(f"Historical: {len(data['historical'])} rows")
    else:
        logger.error(f"Missing: {hist_path}")
        return data

    # Projections
    proj_path = os.path.join(PROJECT_DIR, 'data', 'projections', 'un_life_expectancy_projections.csv')
    if os.path.exists(proj_path):
        data['projections'] = pd.read_csv(proj_path)
        logger.info(f"Projections: {len(data['projections'])} rows")

    # Analysis results
    analysis_dir = os.path.join(PROJECT_DIR, 'outputs', 'analysis')
    for fname in ['blue_zone_vs_global.csv', 'sigma_convergence.csv', 'beta_convergence.csv',
                   'blue_zone_country_profiles.csv', 'decade_improvements.csv']:
        fpath = os.path.join(analysis_dir, fname)
        if os.path.exists(fpath):
            key = fname.replace('.csv', '')
            data[key] = pd.read_csv(fpath)
            logger.info(f"Analysis [{key}]: {len(data[key])} rows")

    return data


def plot_blue_zones_historical_projections(data, output_dir):
    """
    Graph 1: Blue Zone countries life expectancy 1960-2100.
    Solid lines for historical, dashed for projections, shaded uncertainty.
    """
    logger.info("Creating: Blue Zones Historical + Projections timeline...")

    fig, ax = plt.subplots(figsize=(16, 9))

    hist = data.get('historical', pd.DataFrame())
    proj = data.get('projections', pd.DataFrame())

    # Plot each Blue Zone country's historical data
    for iso in sorted(BLUE_ZONE_ISOS):
        country_hist = hist[hist['iso_code'] == iso].sort_values('year')
        le = country_hist[['year', 'life_expectancy']].dropna()

        if not le.empty:
            ax.plot(le['year'], le['life_expectancy'],
                    color=BZ_COLORS[iso], linewidth=2.5, label=BLUE_ZONE_NAMES[iso],
                    solid_capstyle='round')

        # Plot projections
        if not proj.empty:
            country_proj = proj[proj['iso_code'] == iso].sort_values('year')
            if not country_proj.empty and 'le_medium' in country_proj.columns:
                # Connect historical to projection
                if not le.empty:
                    last_hist_year = le['year'].max()
                    last_hist_le = le[le['year'] == last_hist_year]['life_expectancy'].iloc[0]
                    # Add connection point
                    proj_with_connect = pd.concat([
                        pd.DataFrame({'year': [last_hist_year], 'le_medium': [last_hist_le]}),
                        country_proj[['year', 'le_medium']].dropna()
                    ])
                else:
                    proj_with_connect = country_proj[['year', 'le_medium']].dropna()

                ax.plot(proj_with_connect['year'], proj_with_connect['le_medium'],
                        color=BZ_COLORS[iso], linewidth=2, linestyle='--', alpha=0.7)

                # Uncertainty band
                if 'le_high' in country_proj.columns and 'le_low' in country_proj.columns:
                    proj_clean = country_proj[['year', 'le_high', 'le_low']].dropna()
                    if not proj_clean.empty:
                        ax.fill_between(proj_clean['year'],
                                       proj_clean['le_low'], proj_clean['le_high'],
                                       color=BZ_COLORS[iso], alpha=0.1)

    # Global average line (historical)
    if not hist.empty:
        global_avg = hist.groupby('year')['life_expectancy'].mean().dropna()
        ax.plot(global_avg.index, global_avg.values,
                color=GLOBAL_COLOR, linewidth=3, linestyle='-', alpha=0.6,
                label='Global Average', zorder=1)

        # Global average projection
        if not proj.empty and 'le_medium' in proj.columns:
            global_proj = proj.groupby('year')['le_medium'].mean()
            if not global_proj.empty:
                last_year = global_avg.index.max()
                last_val = global_avg.iloc[-1]
                proj_line = pd.concat([pd.Series({last_year: last_val}), global_proj])
                ax.plot(proj_line.index, proj_line.values,
                        color=GLOBAL_COLOR, linewidth=3, linestyle='--', alpha=0.4, zorder=1)

    # Vertical line at present
    ax.axvline(x=2023, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(2024, ax.get_ylim()[0] + 1, 'Projections', fontsize=9, color='gray', alpha=0.7)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Life Expectancy at Birth (years)', fontsize=12)
    ax.set_title('Blue Zone Countries: Life Expectancy Over Time (1960-2100)\nHistorical Data + UN Projections',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1960, 2100)

    plt.tight_layout()
    path = os.path.join(output_dir, 'blue_zones_historical_projections.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_convergence_story(data, output_dir):
    """
    Graph 2: The convergence story - gap between Blue Zone countries and global average.
    """
    logger.info("Creating: Convergence Story graph...")

    bz_global = data.get('blue_zone_vs_global', pd.DataFrame())
    sigma = data.get('sigma_convergence', pd.DataFrame())

    if bz_global.empty:
        logger.warning("No blue_zone_vs_global data")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 2]})

    # Top: Blue Zone gap over global average
    years = bz_global['year']
    gap = bz_global['bz_gap_over_global']

    ax1.fill_between(years, 0, gap, where=(gap > 0), color='#2ECC71', alpha=0.3, label='BZ advantage')
    ax1.fill_between(years, 0, gap, where=(gap <= 0), color='#E74C3C', alpha=0.3, label='BZ disadvantage')
    ax1.plot(years, gap, color='#2C3E50', linewidth=2)
    ax1.axhline(y=0, color='gray', linewidth=0.5)

    # Also show Q25 and Q75 bands relative to BZ
    if 'global_q25' in bz_global.columns:
        bz_q25_gap = bz_global['blue_zone_mean'] - bz_global['global_q25']
        bz_q75_gap = bz_global['blue_zone_mean'] - bz_global['global_q75']
        ax1.plot(years, bz_q25_gap, color='#E74C3C', linewidth=1, linestyle=':', alpha=0.5, label='Gap vs Bottom 25%')
        ax1.plot(years, bz_q75_gap, color='#3498DB', linewidth=1, linestyle=':', alpha=0.5, label='Gap vs Top 25%')

    ax1.set_ylabel('Life Expectancy Gap (years)', fontsize=11)
    ax1.set_title('Blue Zone Countries vs Global Average: Life Expectancy Gap Over Time',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Bottom: Sigma convergence (global spread of LE)
    if not sigma.empty:
        ax2.plot(sigma['year'], sigma['le_std'], color='#8E44AD', linewidth=2)
        ax2.fill_between(sigma['year'], 0, sigma['le_std'], color='#8E44AD', alpha=0.15)
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Std Dev of Life Expectancy\nacross countries (years)', fontsize=10)
        ax2.set_title('Sigma Convergence: Is the World Becoming More Equal in Longevity?',
                      fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Annotate trend
        if len(sigma) > 10:
            first_std = sigma.iloc[0]['le_std']
            last_std = sigma.iloc[-1]['le_std']
            direction = "CONVERGING" if last_std < first_std else "DIVERGING"
            ax2.annotate(f'{direction}\n({first_std:.1f} -> {last_std:.1f} years)',
                        xy=(sigma['year'].iloc[-1], last_std),
                        fontsize=10, fontweight='bold',
                        color='#2ECC71' if last_std < first_std else '#E74C3C',
                        ha='right')

    plt.tight_layout()
    path = os.path.join(output_dir, 'convergence_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_country_deep_dives(data, output_dir):
    """
    Graph 3: Individual deep dive for each Blue Zone country.
    """
    logger.info("Creating: Country deep dive graphs...")

    profiles = data.get('blue_zone_country_profiles', pd.DataFrame())
    proj = data.get('projections', pd.DataFrame())
    hist = data.get('historical', pd.DataFrame())

    country_dir = os.path.join(output_dir, 'country_deep_dives')
    os.makedirs(country_dir, exist_ok=True)

    for iso in sorted(BLUE_ZONE_ISOS):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Get historical data for this country
        country_hist = hist[hist['iso_code'] == iso].sort_values('year') if not hist.empty else pd.DataFrame()

        # Top: Life expectancy + regional peer comparison
        if not profiles.empty:
            country_profile = profiles[profiles['iso_code'] == iso].sort_values('year')
            if not country_profile.empty:
                le = country_profile[['year', 'life_expectancy']].dropna()
                peer = country_profile[['year', 'peer_avg_le']].dropna()
                region = country_profile['region'].iloc[0] if 'region' in country_profile.columns else ''

                if not le.empty:
                    ax1.plot(le['year'], le['life_expectancy'],
                            color=BZ_COLORS[iso], linewidth=2.5, label=BLUE_ZONE_NAMES[iso])
                if not peer.empty:
                    ax1.plot(peer['year'], peer['peer_avg_le'],
                            color=GLOBAL_COLOR, linewidth=2, linestyle='--',
                            label=f'{region} regional avg')

                # Gap fill
                if not le.empty and not peer.empty:
                    merged_yrs = le.merge(peer, on='year')
                    if not merged_yrs.empty:
                        ax1.fill_between(merged_yrs['year'],
                                        merged_yrs['life_expectancy'], merged_yrs['peer_avg_le'],
                                        alpha=0.15, color=BZ_COLORS[iso])
        elif not country_hist.empty:
            le = country_hist[['year', 'life_expectancy']].dropna()
            if not le.empty:
                ax1.plot(le['year'], le['life_expectancy'],
                        color=BZ_COLORS[iso], linewidth=2.5, label=BLUE_ZONE_NAMES[iso])

        # Add projections to top graph
        if not proj.empty:
            country_proj = proj[proj['iso_code'] == iso].sort_values('year')
            if not country_proj.empty and 'le_medium' in country_proj.columns:
                ax1.plot(country_proj['year'], country_proj['le_medium'],
                        color=BZ_COLORS[iso], linewidth=2, linestyle='--', alpha=0.6,
                        label='Projected (Medium)')
                if 'le_high' in country_proj.columns and 'le_low' in country_proj.columns:
                    ax1.fill_between(country_proj['year'],
                                    country_proj['le_low'], country_proj['le_high'],
                                    color=BZ_COLORS[iso], alpha=0.1, label='Projection range')

        ax1.axvline(x=2023, color='gray', linestyle=':', alpha=0.4)
        ax1.set_ylabel('Life Expectancy (years)', fontsize=11)
        ax1.set_title(f'{BLUE_ZONE_NAMES[iso]}: Life Expectancy Trajectory',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Bottom: GDP per capita over time
        if not country_hist.empty and 'gdp_per_capita' in country_hist.columns:
            gdp = country_hist[['year', 'gdp_per_capita']].dropna()
            if not gdp.empty:
                ax2.plot(gdp['year'], gdp['gdp_per_capita'],
                        color='#F39C12', linewidth=2)
                ax2.fill_between(gdp['year'], 0, gdp['gdp_per_capita'],
                                color='#F39C12', alpha=0.15)
                ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('GDP per Capita (USD)', fontsize=11)
        ax2.set_title(f'{BLUE_ZONE_NAMES[iso]}: Economic Trajectory',
                      fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(country_dir, f'{iso}_deep_dive.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved: {path}")


def plot_global_heatmap(data, output_dir):
    """
    Graph 4: Heatmap of life expectancy across all countries and years.
    """
    logger.info("Creating: Global heatmap...")

    hist = data.get('historical', pd.DataFrame())
    if hist.empty:
        return

    le_data = hist[['iso_code', 'year', 'life_expectancy', 'country_name', 'is_blue_zone']].dropna(subset=['life_expectancy'])

    # Pivot to matrix: countries x years
    pivot = le_data.pivot_table(index='iso_code', columns='year', values='life_expectancy', aggfunc='first')

    if pivot.empty:
        logger.warning("Not enough data for heatmap")
        return

    # Sort by most recent life expectancy (descending)
    last_year_col = pivot.columns.max()
    pivot = pivot.sort_values(by=last_year_col, ascending=True, na_position='first')

    # Create country labels with Blue Zone markers
    iso_to_name = dict(zip(hist['iso_code'], hist['country_name']))
    labels = []
    for iso in pivot.index:
        name = iso_to_name.get(iso, iso)
        if name and len(str(name)) > 20:
            name = str(name)[:20]
        marker = ' *' if iso in BLUE_ZONE_ISOS else ''
        labels.append(f"{name}{marker}")

    fig, ax = plt.subplots(figsize=(20, max(12, len(pivot) * 0.25)))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                   vmin=35, vmax=85, interpolation='nearest')

    # Set ticks
    year_cols = pivot.columns.tolist()
    tick_positions = [i for i, y in enumerate(year_cols) if y % 10 == 0]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([year_cols[i] for i in tick_positions], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    # Highlight Blue Zone countries
    for i, iso in enumerate(pivot.index):
        if iso in BLUE_ZONE_ISOS:
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_color(BZ_COLORS.get(iso, 'black'))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Life Expectancy at Birth (years)', fontsize=11)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_title('Global Life Expectancy Heatmap (1960-2023)\n* = Blue Zone Country',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'global_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_improvement_rates(data, output_dir):
    """
    Graph 5: Decade-by-decade improvement rates.
    """
    logger.info("Creating: Improvement rates bar chart...")

    dec = data.get('decade_improvements', pd.DataFrame())
    if dec.empty:
        logger.warning("No decade_improvements data")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    decades = dec[dec['group'] == 'global']['decade'].tolist()
    x = np.arange(len(decades))
    width = 0.25

    for i, (group, color, label) in enumerate([
        ('blue_zone', '#3498DB', 'Blue Zone Countries'),
        ('non_blue_zone', '#95A5A6', 'Non-Blue Zone Countries'),
        ('global', '#2C3E50', 'Global Average'),
    ]):
        group_data = dec[dec['group'] == group]
        if group_data.empty:
            continue
        # Align with decades list
        vals = []
        for d in decades:
            row = group_data[group_data['decade'] == d]
            vals.append(row['avg_gain'].iloc[0] if not row.empty else 0)

        bars = ax.bar(x + i * width, vals, width, color=color, alpha=0.8, label=label)

        # Add value labels
        for bar, val in zip(bars, vals):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Average Life Expectancy Gain (years)', fontsize=12)
    ax.set_title('Life Expectancy Improvement by Decade:\nBlue Zone Countries vs Rest of World',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(decades, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'improvement_rates.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_blue_zone_ranking_over_time(data, output_dir):
    """
    Bonus Graph: How Blue Zone countries rank globally over time.
    """
    logger.info("Creating: Blue Zone ranking over time...")

    hist = data.get('historical', pd.DataFrame())
    if hist.empty:
        return

    le_data = hist[['iso_code', 'year', 'life_expectancy']].dropna(subset=['life_expectancy'])

    # For each year, rank all countries
    rankings = []
    for year in sorted(le_data['year'].unique()):
        year_data = le_data[le_data['year'] == year].copy()
        year_data['rank'] = year_data['life_expectancy'].rank(ascending=False)
        total = len(year_data)
        year_data['percentile'] = (1 - year_data['rank'] / total) * 100

        for iso in BLUE_ZONE_ISOS:
            row = year_data[year_data['iso_code'] == iso]
            if not row.empty:
                rankings.append({
                    'year': year,
                    'iso_code': iso,
                    'rank': int(row['rank'].iloc[0]),
                    'percentile': row['percentile'].iloc[0],
                    'total_countries': total,
                })

    if not rankings:
        return

    rank_df = pd.DataFrame(rankings)

    fig, ax = plt.subplots(figsize=(14, 8))

    for iso in sorted(BLUE_ZONE_ISOS):
        country_ranks = rank_df[rank_df['iso_code'] == iso].sort_values('year')
        if not country_ranks.empty:
            ax.plot(country_ranks['year'], country_ranks['percentile'],
                    color=BZ_COLORS[iso], linewidth=2, marker='o', markersize=3,
                    label=BLUE_ZONE_NAMES[iso])

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='Median')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Percentile Ranking (higher = longer-lived)', fontsize=12)
    ax.set_title('Blue Zone Countries: Global Life Expectancy Ranking Over Time',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'blue_zone_ranking.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def main():
    """Generate all visualizations."""
    data = load_all_data()

    if 'historical' not in data:
        logger.error("Cannot visualize without historical data. Run historical_data_collector.py first.")
        return

    output_dir = os.path.join(PROJECT_DIR, 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Generate all graphs
    plot_blue_zones_historical_projections(data, output_dir)
    plot_convergence_story(data, output_dir)
    plot_country_deep_dives(data, output_dir)
    plot_global_heatmap(data, output_dir)
    plot_improvement_rates(data, output_dir)
    plot_blue_zone_ranking_over_time(data, output_dir)

    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print(f"Graphs saved to: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            fpath = os.path.join(output_dir, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")

    country_dir = os.path.join(output_dir, 'country_deep_dives')
    if os.path.exists(country_dir):
        for f in sorted(os.listdir(country_dir)):
            if f.endswith('.png'):
                fpath = os.path.join(country_dir, f)
                size_kb = os.path.getsize(fpath) / 1024
                print(f"  country_deep_dives/{f} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
