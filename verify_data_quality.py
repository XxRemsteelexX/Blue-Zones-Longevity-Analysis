#!/usr/bin/env python3
"""
Data Quality Verification for Blue Zones Historical Analysis
Runs automated checks to ensure data integrity.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BLUE_ZONE_ISOS = {'USA', 'JPN', 'ITA', 'GRC', 'CRI'}


def check_historical_data():
    """Verify the historical panel data."""
    results = []
    path = os.path.join(PROJECT_DIR, 'data', 'historical', 'merged_historical_panel.csv')

    if not os.path.exists(path):
        results.append(("FAIL", "Historical data file not found"))
        return results

    df = pd.read_csv(path)

    # Check 1: Blue Zone countries have data
    for iso in sorted(BLUE_ZONE_ISOS):
        country = df[df['iso_code'] == iso]
        le = country['life_expectancy'].dropna() if 'life_expectancy' in country.columns else pd.Series()
        n_years = len(le)
        total_years = 64  # 1960-2023
        pct = (n_years / total_years) * 100

        if n_years == 0:
            results.append(("FAIL", f"{iso}: No life expectancy data at all"))
        elif pct >= 80:
            results.append(("PASS", f"{iso}: {n_years} years of LE data ({pct:.0f}% coverage)"))
        elif pct >= 50:
            results.append(("WARN", f"{iso}: {n_years} years of LE data ({pct:.0f}% coverage) - below 80% target"))
        else:
            results.append(("FAIL", f"{iso}: Only {n_years} years of LE data ({pct:.0f}% coverage)"))

    # Check 2: Life expectancy values in reasonable range
    if 'life_expectancy' in df.columns:
        le = df['life_expectancy'].dropna()
        out_of_range = le[(le < 25) | (le > 95)]
        if len(out_of_range) == 0:
            results.append(("PASS", f"All {len(le)} LE values in range [25, 95]"))
        else:
            results.append(("WARN", f"{len(out_of_range)} LE values outside [25, 95]: "
                           f"min={le.min():.1f}, max={le.max():.1f}"))

    # Check 3: GDP values positive
    if 'gdp_per_capita' in df.columns:
        gdp = df['gdp_per_capita'].dropna()
        negatives = gdp[gdp <= 0]
        if len(negatives) == 0:
            results.append(("PASS", f"All {len(gdp)} GDP values are positive"))
        else:
            results.append(("FAIL", f"{len(negatives)} negative GDP values found"))

    # Check 4: No impossible jumps in life expectancy
    if 'life_expectancy' in df.columns:
        big_jumps = 0
        for iso in df['iso_code'].unique():
            c = df[df['iso_code'] == iso].sort_values('year')
            le = c['life_expectancy'].dropna()
            if len(le) > 1:
                diffs = le.diff().abs()
                jumps = diffs[diffs > 5]
                big_jumps += len(jumps)

        if big_jumps == 0:
            results.append(("PASS", "No life expectancy jumps > 5 years between consecutive data points"))
        else:
            results.append(("WARN", f"{big_jumps} instances of LE changing >5 years between consecutive points"))

    # Check 5: Year range
    yr_min = df['year'].min()
    yr_max = df['year'].max()
    if yr_min <= 1965 and yr_max >= 2020:
        results.append(("PASS", f"Year range: {yr_min}-{yr_max} (covers 1960s to 2020s)"))
    else:
        results.append(("WARN", f"Year range only {yr_min}-{yr_max}"))

    # Check 6: Country count
    n_countries = df['iso_code'].nunique()
    if n_countries >= 80:
        results.append(("PASS", f"{n_countries} countries with data (target: 80+)"))
    else:
        results.append(("WARN", f"Only {n_countries} countries (target: 80+)"))

    # Check 7: Total data volume
    total_rows = len(df)
    if total_rows >= 1000:
        results.append(("PASS", f"{total_rows:,} total country-year rows"))
    else:
        results.append(("WARN", f"Only {total_rows} rows - expected 1000+"))

    return results


def check_projections():
    """Verify projection data."""
    results = []
    path = os.path.join(PROJECT_DIR, 'data', 'projections', 'un_life_expectancy_projections.csv')

    if not os.path.exists(path):
        results.append(("WARN", "Projections file not found - run un_projections_collector.py"))
        return results

    df = pd.read_csv(path)

    # Check 1: Has Blue Zone countries
    for iso in sorted(BLUE_ZONE_ISOS):
        country = df[df['iso_code'] == iso]
        if len(country) > 0:
            results.append(("PASS", f"Projections exist for {iso} ({len(country)} data points)"))
        else:
            results.append(("FAIL", f"No projections for Blue Zone country {iso}"))

    # Check 2: Has future years
    if 'year' in df.columns:
        max_year = df['year'].max()
        if max_year >= 2050:
            results.append(("PASS", f"Projections extend to {max_year}"))
        else:
            results.append(("WARN", f"Projections only extend to {max_year} (target: 2050+)"))

    # Check 3: Medium variant exists
    if 'le_medium' in df.columns:
        n_valid = df['le_medium'].notna().sum()
        results.append(("PASS", f"Medium variant: {n_valid} data points"))
    else:
        results.append(("FAIL", "No le_medium column in projections"))

    # Check 4: Values reasonable
    if 'le_medium' in df.columns:
        le = df['le_medium'].dropna()
        if le.min() >= 30 and le.max() <= 100:
            results.append(("PASS", f"Projection values in range [{le.min():.1f}, {le.max():.1f}]"))
        else:
            results.append(("WARN", f"Projection range [{le.min():.1f}, {le.max():.1f}] - check extremes"))

    return results


def check_analysis_outputs():
    """Verify analysis output files exist and are non-empty."""
    results = []
    analysis_dir = os.path.join(PROJECT_DIR, 'outputs', 'analysis')

    expected_files = [
        'blue_zone_vs_global.csv',
        'sigma_convergence.csv',
        'beta_convergence.csv',
        'blue_zone_country_profiles.csv',
        'decade_improvements.csv',
    ]

    for fname in expected_files:
        fpath = os.path.join(analysis_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            if len(df) > 0:
                results.append(("PASS", f"{fname}: {len(df)} rows"))
            else:
                results.append(("FAIL", f"{fname}: exists but empty"))
        else:
            results.append(("WARN", f"{fname}: not found"))

    return results


def check_visualizations():
    """Verify visualization files exist."""
    results = []
    fig_dir = os.path.join(PROJECT_DIR, 'outputs', 'figures')

    expected = [
        'blue_zones_historical_projections.png',
        'convergence_analysis.png',
        'global_heatmap.png',
        'improvement_rates.png',
        'blue_zone_ranking.png',
    ]

    for fname in expected:
        fpath = os.path.join(fig_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            if size_kb > 10:
                results.append(("PASS", f"{fname}: {size_kb:.0f} KB"))
            else:
                results.append(("WARN", f"{fname}: only {size_kb:.0f} KB (might be empty)"))
        else:
            results.append(("WARN", f"{fname}: not found"))

    # Country deep dives
    dd_dir = os.path.join(fig_dir, 'country_deep_dives')
    for iso in sorted(BLUE_ZONE_ISOS):
        fpath = os.path.join(dd_dir, f'{iso}_deep_dive.png')
        if os.path.exists(fpath):
            results.append(("PASS", f"country_deep_dives/{iso}_deep_dive.png"))
        else:
            results.append(("WARN", f"country_deep_dives/{iso}_deep_dive.png: not found"))

    return results


def main():
    """Run all checks and produce a report."""
    output_dir = os.path.join(PROJECT_DIR, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    print("=" * 70)
    print("DATA QUALITY VERIFICATION REPORT")
    print("=" * 70)

    sections = [
        ("HISTORICAL DATA", check_historical_data),
        ("PROJECTIONS", check_projections),
        ("ANALYSIS OUTPUTS", check_analysis_outputs),
        ("VISUALIZATIONS", check_visualizations),
    ]

    for section_name, check_func in sections:
        print(f"\n--- {section_name} ---")
        results = check_func()
        all_results.extend(results)
        for status, msg in results:
            icon = {'PASS': 'OK', 'WARN': '!!', 'FAIL': 'XX'}[status]
            print(f"  [{icon}] {msg}")

    # Summary
    passes = sum(1 for s, _ in all_results if s == 'PASS')
    warns = sum(1 for s, _ in all_results if s == 'WARN')
    fails = sum(1 for s, _ in all_results if s == 'FAIL')

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {passes} passed, {warns} warnings, {fails} failures")
    print(f"{'=' * 70}")

    if fails == 0 and warns <= 3:
        print("VERDICT: Data quality is GOOD")
    elif fails == 0:
        print("VERDICT: Data quality is ACCEPTABLE (some warnings)")
    else:
        print("VERDICT: Data quality NEEDS ATTENTION (failures detected)")

    # Save report
    report_path = os.path.join(output_dir, 'data_quality_report.txt')
    with open(report_path, 'w') as f:
        f.write("DATA QUALITY VERIFICATION REPORT\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write("=" * 70 + "\n\n")
        for section_name, check_func in sections:
            f.write(f"\n--- {section_name} ---\n")
            results = check_func()
            for status, msg in results:
                f.write(f"  [{status}] {msg}\n")
        f.write(f"\nSUMMARY: {passes} passed, {warns} warnings, {fails} failures\n")

    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
