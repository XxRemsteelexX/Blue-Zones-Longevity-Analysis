#!/usr/bin/env python3
"""
Extended Feature Collector for ML-Based Life Expectancy Prediction.

Pulls 14 additional World Bank indicators beyond the existing 16, adds static
geography/climate data (temperature, elevation), computes derived features,
and builds a cross-sectional feature matrix for ML modeling.

All data is REAL -- World Bank API and standard geographic references.
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import shared constants and functions from existing collector
from historical_data_collector import (
    COUNTRY_ISO_MAP, VALID_ISOS, BLUE_ZONE_ISOS,
    COUNTRY_COORDS, fetch_wb_indicator_all_years,
)

# ---------------------------------------------------------------------------
# New World Bank indicators (not already in the panel)
# ---------------------------------------------------------------------------
NEW_WB_INDICATORS = {
    'SE.ADT.LITR.ZS': 'adult_literacy_rate',
    'SE.TER.ENRR': 'tertiary_enrollment',
    'EG.ELC.ACCS.ZS': 'electricity_access',
    'IT.NET.USER.ZS': 'internet_users_pct',
    'EN.ATM.CO2E.PC': 'co2_per_capita',
    'AG.LND.TOTL.K2': 'land_area_km2',
    'AG.LND.ARBL.ZS': 'arable_land_pct',
    'SP.DYN.CBRT.IN': 'birth_rate',
    'SP.POP.DPND': 'age_dependency_ratio',
    'SH.DYN.NCOM.ZS': 'ncd_mortality_rate',
    'SH.IMM.MEAS': 'measles_immunization',
    'SH.STA.OWAD.ZS': 'obesity_prevalence',
    'SH.STA.SUIC.P5': 'suicide_rate',
    'NY.GNP.PCAP.CD': 'gni_per_capita',
}

# ---------------------------------------------------------------------------
# Static data: Mean annual temperature (Celsius)
# Sources: World Bank Climate Change Knowledge Portal, standard references
# ---------------------------------------------------------------------------
COUNTRY_TEMPERATURE = {
    'AFG': 12.6, 'ALB': 15.2, 'DZA': 22.5, 'ARG': 14.8, 'AUS': 21.7,
    'AUT': 6.3, 'BGD': 25.5, 'BEL': 9.8, 'BIH': 9.8, 'BOL': 21.3,
    'BWA': 22.2, 'BRA': 25.0, 'BGR': 10.6, 'CAN': -5.4, 'CHL': 8.5,
    'CHN': 7.0, 'COL': 24.5, 'CRI': 25.1, 'HRV': 10.7, 'CZE': 7.5,
    'DNK': 7.7, 'ECU': 22.0, 'EGY': 22.1, 'EST': 5.2, 'ETH': 22.2,
    'FJI': 25.6, 'FIN': 1.7, 'FRA': 10.7, 'DEU': 8.5, 'GHA': 27.6,
    'GRC': 15.5, 'HUN': 9.8, 'ISL': 1.0, 'IND': 24.0, 'IDN': 26.6,
    'IRN': 17.0, 'IRQ': 22.4, 'IRL': 9.3, 'ISR': 19.9, 'ITA': 13.5,
    'JPN': 11.8, 'JOR': 18.3, 'KEN': 24.7, 'KOR': 11.5, 'LVA': 5.9,
    'LBN': 15.3, 'LBY': 20.5, 'LTU': 6.1, 'LUX': 8.7, 'MKD': 11.5,
    'MDG': 23.0, 'MYS': 27.0, 'MAR': 17.1, 'MEX': 21.0, 'MNE': 10.6,
    'MOZ': 24.0, 'MMR': 25.0, 'NAM': 19.5, 'NLD': 9.3, 'NZL': 10.5,
    'NGA': 27.0, 'NOR': 1.5, 'PAK': 20.2, 'PNG': 25.3, 'PRY': 23.0,
    'PER': 18.0, 'PHL': 26.6, 'POL': 7.8, 'PRT': 15.2, 'ROU': 8.8,
    'RUS': -5.1, 'SAU': 25.1, 'SRB': 10.6, 'SGP': 27.0, 'SVK': 8.1,
    'SVN': 8.3, 'ZAF': 17.5, 'ESP': 13.3, 'SWE': 2.1, 'CHE': 5.5,
    'SYR': 17.5, 'TZA': 23.0, 'THA': 27.2, 'TUN': 19.2, 'TUR': 11.1,
    'UGA': 22.3, 'URY': 17.5, 'USA': 8.5, 'VEN': 25.3, 'VNM': 24.4,
    'ZMB': 21.4, 'ZWE': 20.3,
}

# ---------------------------------------------------------------------------
# Static data: Mean elevation (meters above sea level)
# Sources: CIESIN/SEDAC, CIA World Factbook, standard geographic references
# ---------------------------------------------------------------------------
COUNTRY_ELEVATION = {
    'AFG': 1884, 'ALB': 708, 'DZA': 800, 'ARG': 595, 'AUS': 330,
    'AUT': 910, 'BGD': 85, 'BEL': 181, 'BIH': 500, 'BOL': 1192,
    'BWA': 1013, 'BRA': 320, 'BGR': 472, 'CAN': 487, 'CHL': 1871,
    'CHN': 1840, 'COL': 593, 'CRI': 746, 'HRV': 331, 'CZE': 433,
    'DNK': 34, 'ECU': 1117, 'EGY': 321, 'EST': 61, 'ETH': 1330,
    'FJI': 146, 'FIN': 164, 'FRA': 375, 'DEU': 263, 'GHA': 190,
    'GRC': 498, 'HUN': 143, 'ISL': 557, 'IND': 621, 'IDN': 367,
    'IRN': 1305, 'IRQ': 312, 'IRL': 118, 'ISR': 508, 'ITA': 538,
    'JPN': 438, 'JOR': 812, 'KEN': 762, 'KOR': 282, 'LVA': 87,
    'LBN': 1250, 'LBY': 423, 'LTU': 110, 'LUX': 325, 'MKD': 741,
    'MDG': 615, 'MYS': 538, 'MAR': 909, 'MEX': 1111, 'MNE': 1086,
    'MOZ': 345, 'MMR': 702, 'NAM': 1141, 'NLD': 30, 'NZL': 388,
    'NGA': 380, 'NOR': 460, 'PAK': 900, 'PNG': 667, 'PRY': 178,
    'PER': 1555, 'PHL': 442, 'POL': 173, 'PRT': 372, 'ROU': 414,
    'RUS': 600, 'SAU': 665, 'SRB': 473, 'SGP': 15, 'SVK': 458,
    'SVN': 492, 'ZAF': 1034, 'ESP': 660, 'SWE': 320, 'CHE': 1350,
    'SYR': 514, 'TZA': 1018, 'THA': 287, 'TUN': 246, 'TUR': 1132,
    'UGA': 1100, 'URY': 109, 'USA': 760, 'VEN': 450, 'VNM': 398,
    'ZMB': 1138, 'ZWE': 961,
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def fetch_new_wb_indicators(session):
    """Fetch 14 additional WB indicators using existing API pattern."""
    all_dfs = []

    for wb_code, col_name in NEW_WB_INDICATORS.items():
        logger.info(f"Fetching {col_name} ({wb_code})...")
        records = fetch_wb_indicator_all_years(wb_code, col_name, session)

        if records:
            indicator_df = pd.DataFrame(records)
            indicator_df = indicator_df.pivot_table(
                index=['iso_code', 'year'], columns='indicator',
                values='value', aggfunc='first'
            ).reset_index()
            all_dfs.append(indicator_df)
            logger.info(f"  {col_name}: {len(indicator_df)} records")
        else:
            logger.warning(f"  {col_name}: no data returned")

        time.sleep(0.5)

    if not all_dfs:
        logger.error("No new WB indicators fetched")
        return pd.DataFrame(columns=['iso_code', 'year'])

    # Merge all indicator DataFrames
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = pd.merge(merged, df, on=['iso_code', 'year'], how='outer')

    logger.info(f"New WB indicators merged: {merged.shape}")
    return merged


def add_static_features(df):
    """Add temperature and elevation from lookup tables."""
    df = df.copy()
    df['mean_temperature'] = df['iso_code'].map(COUNTRY_TEMPERATURE)
    df['mean_elevation'] = df['iso_code'].map(COUNTRY_ELEVATION)
    return df


def compute_derived_features(df):
    """Compute features derived from existing and new data."""
    df = df.copy()

    # Distance from equator (climate proxy)
    if 'latitude' in df.columns:
        df['abs_latitude'] = df['latitude'].abs()

    # Population density
    if 'population_total' in df.columns and 'land_area_km2' in df.columns:
        mask = (df['population_total'].notna()) & (df['land_area_km2'].notna())
        mask = mask & (df['land_area_km2'] > 0)
        df.loc[mask, 'population_density'] = (
            df.loc[mask, 'population_total'] / df.loc[mask, 'land_area_km2']
        )

    # Female LE advantage
    if 'life_expectancy_female' in df.columns and 'life_expectancy_male' in df.columns:
        df['female_le_advantage'] = (
            df['life_expectancy_female'] - df['life_expectancy_male']
        )

    # Log GDP
    if 'gdp_per_capita' in df.columns:
        df['log_gdp_per_capita'] = np.log1p(df['gdp_per_capita'])

    return df


def build_cross_sectional_matrix(panel_df):
    """
    Build a cross-sectional feature matrix (93 rows x ~35 columns).

    For each country, for each feature, take the most recent non-null value
    from 2015-2023. This maximizes coverage.
    """
    recent = panel_df[panel_df['year'] >= 2015].copy()

    # Identify feature columns (exclude metadata)
    meta_cols = ['iso_code', 'year', 'country_name', 'is_blue_zone',
                 'blue_zone_region', 'latitude', 'longitude']
    feature_cols = [c for c in recent.columns if c not in meta_cols]

    rows = []
    for iso in sorted(recent['iso_code'].unique()):
        country_data = recent[recent['iso_code'] == iso].sort_values('year', ascending=False)
        row = {'iso_code': iso}

        # Get metadata from first available row
        first = country_data.iloc[0]
        row['country_name'] = first.get('country_name', iso)
        row['is_blue_zone'] = first.get('is_blue_zone', 0)
        row['latitude'] = first.get('latitude', np.nan)
        row['longitude'] = first.get('longitude', np.nan)

        # For each feature, take the most recent non-null value
        for col in feature_cols:
            valid = country_data[country_data[col].notna()]
            if len(valid) > 0:
                row[col] = valid[col].iloc[0]
                if col == 'life_expectancy':
                    row['data_year'] = int(valid['year'].iloc[0])
            else:
                row[col] = np.nan

        rows.append(row)

    cross = pd.DataFrame(rows)
    logger.info(f"Cross-sectional matrix: {cross.shape}")
    return cross


def print_coverage_report(cross_df):
    """Print coverage statistics for the feature matrix."""
    meta_cols = ['iso_code', 'country_name', 'is_blue_zone', 'latitude',
                 'longitude', 'data_year']
    feature_cols = [c for c in cross_df.columns if c not in meta_cols]

    print("\n" + "=" * 60)
    print("FEATURE MATRIX COVERAGE REPORT")
    print("=" * 60)
    print(f"Countries: {len(cross_df)}")
    print(f"Total columns: {len(cross_df.columns)}")
    print(f"Feature columns: {len(feature_cols)}")
    print()

    coverage_data = []
    for col in sorted(feature_cols):
        n_valid = cross_df[col].notna().sum()
        pct = n_valid / len(cross_df) * 100
        coverage_data.append((col, n_valid, pct))
        status = "OK" if pct >= 50 else "LOW"
        print(f"  {col:35s} {n_valid:3d}/93  ({pct:5.1f}%)  [{status}]")

    n_good = sum(1 for _, _, p in coverage_data if p >= 50)
    print(f"\nFeatures with >=50% coverage: {n_good}/{len(feature_cols)}")
    print("=" * 60)


def main():
    """Main feature collection pipeline."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Blue-Zones-Research/2.0 (Educational Research)'
    })

    # 1. Load existing panel
    panel_path = os.path.join(PROJECT_DIR, 'data', 'historical',
                              'merged_historical_panel.csv')
    panel = pd.read_csv(panel_path)
    logger.info(f"Loaded existing panel: {panel.shape}")

    # 2. Fetch new WB indicators
    new_features = fetch_new_wb_indicators(session)
    logger.info(f"New features fetched: {new_features.shape}")

    # 3. Merge new features into panel
    if not new_features.empty:
        merged = pd.merge(panel, new_features, on=['iso_code', 'year'], how='left')
    else:
        merged = panel.copy()
    logger.info(f"Merged panel: {merged.shape}")

    # 4. Add static features (temperature, elevation)
    merged = add_static_features(merged)

    # 5. Compute derived features
    merged = compute_derived_features(merged)
    logger.info(f"After derived features: {merged.shape}")

    # 6. Build cross-sectional matrix
    cross_section = build_cross_sectional_matrix(merged)

    # 7. Save outputs
    features_dir = os.path.join(PROJECT_DIR, 'data', 'features')
    os.makedirs(features_dir, exist_ok=True)

    cross_path = os.path.join(features_dir, 'ml_feature_matrix.csv')
    cross_section.to_csv(cross_path, index=False)
    logger.info(f"Saved: {cross_path}")

    expanded_path = os.path.join(features_dir, 'expanded_panel.csv')
    merged.to_csv(expanded_path, index=False)
    logger.info(f"Saved: {expanded_path}")

    # 8. Print coverage report
    print_coverage_report(cross_section)

    return cross_section


if __name__ == '__main__':
    main()
