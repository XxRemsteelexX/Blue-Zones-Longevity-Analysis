#!/usr/bin/env python3
"""
Gender Life Expectancy Analysis for Blue Zones

Computes:
  - Male LE gap (BZ vs global) and Female LE gap separately
  - Gender gap (female - male) within BZ countries vs global
  - Tests whether BZ advantage is stronger for males or females
  - Generates figures for notebook 07
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'analysis')
FIGURE_DIR = os.path.join(PROJECT_DIR, 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

BLUE_ZONE_ISOS = {'USA', 'JPN', 'ITA', 'GRC', 'CRI'}
COUNTRY_NAMES = {
    'USA': 'United States', 'JPN': 'Japan', 'ITA': 'Italy',
    'GRC': 'Greece', 'CRI': 'Costa Rica'
}


def load_panel():
    path = os.path.join(PROJECT_DIR, 'data', 'historical', 'merged_historical_panel.csv')
    df = pd.read_csv(path)
    df['is_blue_zone'] = df['iso_code'].isin(BLUE_ZONE_ISOS).astype(int)
    return df


def gender_gap_analysis(df):
    """Compute male and female LE gaps (BZ vs global) and gender gap over time."""
    results = []

    for year in sorted(df['year'].unique()):
        yr = df[df['year'] == year].dropna(subset=['life_expectancy_male', 'life_expectancy_female'])
        bz = yr[yr['is_blue_zone'] == 1]
        non_bz = yr[yr['is_blue_zone'] == 0]

        if len(bz) < 3 or len(non_bz) < 10:
            continue

        # Male LE
        bz_male = bz['life_expectancy_male'].mean()
        global_male = non_bz['life_expectancy_male'].mean()
        male_gap = bz_male - global_male

        # Female LE
        bz_female = bz['life_expectancy_female'].mean()
        global_female = non_bz['life_expectancy_female'].mean()
        female_gap = bz_female - global_female

        # Gender gap (female - male)
        bz_gender_gap = bz_female - bz_male
        global_gender_gap = global_female - global_male

        # Overall LE
        bz_overall = bz['life_expectancy'].mean() if 'life_expectancy' in bz.columns else np.nan
        global_overall = non_bz['life_expectancy'].mean() if 'life_expectancy' in non_bz.columns else np.nan

        results.append({
            'year': year,
            'bz_male_le': bz_male,
            'bz_female_le': bz_female,
            'global_male_le': global_male,
            'global_female_le': global_female,
            'male_gap': male_gap,
            'female_gap': female_gap,
            'bz_gender_gap': bz_gender_gap,
            'global_gender_gap': global_gender_gap,
            'bz_advantage_stronger_for': 'males' if male_gap > female_gap else 'females',
            'male_female_gap_difference': male_gap - female_gap,
        })

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUTPUT_DIR, 'gender_gap_analysis.csv'), index=False)
    return out


def plot_gender_trends(gender_df):
    """Plot male vs female LE trends for BZ and global."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Male and Female LE over time
    ax = axes[0]
    ax.plot(gender_df['year'], gender_df['bz_female_le'], 'b-', linewidth=2, label='BZ Female')
    ax.plot(gender_df['year'], gender_df['bz_male_le'], 'b--', linewidth=2, label='BZ Male')
    ax.plot(gender_df['year'], gender_df['global_female_le'], 'gray', linewidth=1.5, label='Global Female')
    ax.plot(gender_df['year'], gender_df['global_male_le'], 'gray', linestyle='--', linewidth=1.5, label='Global Male')
    ax.axvspan(2020, 2023, alpha=0.1, color='red', label='COVID period')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Life Expectancy (years)', fontsize=12)
    ax.set_title('Male vs Female Life Expectancy: Blue Zones and Global', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: BZ advantage by gender
    ax = axes[1]
    ax.plot(gender_df['year'], gender_df['male_gap'], 'steelblue', linewidth=2, label='Male BZ Advantage')
    ax.plot(gender_df['year'], gender_df['female_gap'], 'coral', linewidth=2, label='Female BZ Advantage')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvspan(2020, 2023, alpha=0.1, color='red', label='COVID period')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('BZ Advantage (years)', fontsize=12)
    ax.set_title('Blue Zone Advantage by Gender', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'nb07_gender_le_trends.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_gender_gap_comparison(gender_df):
    """Plot gender gap (F-M) in BZ countries vs global over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(gender_df['year'], gender_df['bz_gender_gap'], 'steelblue', linewidth=2.5,
            label='Blue Zone Countries')
    ax.plot(gender_df['year'], gender_df['global_gender_gap'], 'gray', linewidth=2,
            label='Global Average')
    ax.fill_between(gender_df['year'], gender_df['bz_gender_gap'], gender_df['global_gender_gap'],
                     alpha=0.15, color='steelblue')
    ax.axvspan(2020, 2023, alpha=0.1, color='red')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Gender Gap: Female - Male LE (years)', fontsize=12)
    ax.set_title('Gender Gap in Life Expectancy: Blue Zones vs Global', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'nb07_gender_gap_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 70)
    print("GENDER LIFE EXPECTANCY ANALYSIS")
    print("=" * 70)

    df = load_panel()
    print(f"Loaded: {len(df)} rows")

    # Check coverage
    male_coverage = df['life_expectancy_male'].notna().sum()
    female_coverage = df['life_expectancy_female'].notna().sum()
    print(f"Male LE records: {male_coverage}, Female LE records: {female_coverage}")

    gender_df = gender_gap_analysis(df)
    print(f"\nGender analysis: {len(gender_df)} years")

    if len(gender_df) > 0:
        recent = gender_df[gender_df['year'] >= 2010]
        if len(recent) > 0:
            print(f"\nRecent findings (2010+):")
            print(f"  Male BZ advantage: {recent['male_gap'].mean():.2f} years")
            print(f"  Female BZ advantage: {recent['female_gap'].mean():.2f} years")
            print(f"  BZ advantage stronger for: {'males' if recent['male_gap'].mean() > recent['female_gap'].mean() else 'females'}")
            print(f"  BZ gender gap (F-M): {recent['bz_gender_gap'].mean():.2f} years")
            print(f"  Global gender gap (F-M): {recent['global_gender_gap'].mean():.2f} years")

        # 2019 snapshot
        y2019 = gender_df[gender_df['year'] == 2019]
        if len(y2019) > 0:
            r = y2019.iloc[0]
            print(f"\n2019 Snapshot:")
            print(f"  BZ Male LE: {r['bz_male_le']:.1f}, Female LE: {r['bz_female_le']:.1f}")
            print(f"  Global Male LE: {r['global_male_le']:.1f}, Female LE: {r['global_female_le']:.1f}")
            print(f"  Male gap: {r['male_gap']:.1f}, Female gap: {r['female_gap']:.1f}")

        plot_gender_trends(gender_df)
        plot_gender_gap_comparison(gender_df)

    print("\nGender analysis complete.")


if __name__ == '__main__':
    main()
