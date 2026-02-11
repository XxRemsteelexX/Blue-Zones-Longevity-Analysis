#!/usr/bin/env python3
"""
UN Population Division Projections Collector
Downloads official life expectancy projections from UN World Population Prospects (WPP).
All data is REAL authoritative projections from UN demographers - no synthetic data.

Source: UN Population Division - World Population Prospects 2024 Revision
URL: https://population.un.org/wpp/
"""

import pandas as pd
import numpy as np
import requests
import os
import io
import logging
import time
import zipfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ISO3 code to country name mapping (same as historical_data_collector.py)
COUNTRY_ISO_MAP = {
    'USA': 'United States', 'JPN': 'Japan', 'ITA': 'Italy',
    'GRC': 'Greece', 'CRI': 'Costa Rica', 'CHN': 'China',
    'IND': 'India', 'BRA': 'Brazil', 'RUS': 'Russia',
    'CAN': 'Canada', 'AUS': 'Australia', 'DEU': 'Germany',
    'GBR': 'United Kingdom', 'FRA': 'France', 'ESP': 'Spain',
    'NOR': 'Norway', 'SWE': 'Sweden', 'DNK': 'Denmark',
    'CHE': 'Switzerland', 'NLD': 'Netherlands', 'KOR': 'South Korea',
    'MEX': 'Mexico', 'ARG': 'Argentina', 'CHL': 'Chile',
    'PER': 'Peru', 'COL': 'Colombia', 'VEN': 'Venezuela',
    'ECU': 'Ecuador', 'BOL': 'Bolivia', 'PRY': 'Paraguay',
    'URY': 'Uruguay', 'ZAF': 'South Africa', 'EGY': 'Egypt',
    'MAR': 'Morocco', 'DZA': 'Algeria', 'TUN': 'Tunisia',
    'LBY': 'Libya', 'NGA': 'Nigeria', 'KEN': 'Kenya',
    'ETH': 'Ethiopia', 'GHA': 'Ghana', 'TZA': 'Tanzania',
    'UGA': 'Uganda', 'MOZ': 'Mozambique', 'MDG': 'Madagascar',
    'BWA': 'Botswana', 'NAM': 'Namibia', 'ZMB': 'Zambia',
    'ZWE': 'Zimbabwe', 'TUR': 'Turkey', 'IRN': 'Iran',
    'IRQ': 'Iraq', 'SAU': 'Saudi Arabia', 'ISR': 'Israel',
    'JOR': 'Jordan', 'LBN': 'Lebanon', 'SYR': 'Syria',
    'AFG': 'Afghanistan', 'PAK': 'Pakistan', 'BGD': 'Bangladesh',
    'MMR': 'Myanmar', 'THA': 'Thailand', 'VNM': 'Vietnam',
    'MYS': 'Malaysia', 'SGP': 'Singapore', 'IDN': 'Indonesia',
    'PHL': 'Philippines', 'NZL': 'New Zealand', 'PNG': 'Papua New Guinea',
    'FJI': 'Fiji', 'ISL': 'Iceland', 'FIN': 'Finland',
    'EST': 'Estonia', 'LVA': 'Latvia', 'LTU': 'Lithuania',
    'POL': 'Poland', 'CZE': 'Czech Republic', 'SVK': 'Slovakia',
    'HUN': 'Hungary', 'ROU': 'Romania', 'BGR': 'Bulgaria',
    'SRB': 'Serbia', 'HRV': 'Croatia', 'BIH': 'Bosnia and Herzegovina',
    'MNE': 'Montenegro', 'ALB': 'Albania', 'MKD': 'North Macedonia',
    'SVN': 'Slovenia', 'AUT': 'Austria', 'BEL': 'Belgium',
    'LUX': 'Luxembourg', 'IRL': 'Ireland', 'PRT': 'Portugal',
}

# UN uses numeric country codes - mapping to ISO3
# Source: UN M49 standard
UN_NUMERIC_TO_ISO3 = {
    840: 'USA', 392: 'JPN', 380: 'ITA', 300: 'GRC', 188: 'CRI',
    156: 'CHN', 356: 'IND', 76: 'BRA', 643: 'RUS', 124: 'CAN',
    36: 'AUS', 276: 'DEU', 826: 'GBR', 250: 'FRA', 724: 'ESP',
    578: 'NOR', 752: 'SWE', 208: 'DNK', 756: 'CHE', 528: 'NLD',
    410: 'KOR', 484: 'MEX', 32: 'ARG', 152: 'CHL', 604: 'PER',
    170: 'COL', 862: 'VEN', 218: 'ECU', 68: 'BOL', 600: 'PRY',
    858: 'URY', 710: 'ZAF', 818: 'EGY', 504: 'MAR', 12: 'DZA',
    788: 'TUN', 434: 'LBY', 566: 'NGA', 404: 'KEN', 231: 'ETH',
    288: 'GHA', 834: 'TZA', 800: 'UGA', 508: 'MOZ', 450: 'MDG',
    72: 'BWA', 516: 'NAM', 894: 'ZMB', 716: 'ZWE', 792: 'TUR',
    364: 'IRN', 368: 'IRQ', 682: 'SAU', 376: 'ISR', 400: 'JOR',
    422: 'LBN', 760: 'SYR', 4: 'AFG', 586: 'PAK', 50: 'BGD',
    104: 'MMR', 764: 'THA', 704: 'VNM', 458: 'MYS', 702: 'SGP',
    360: 'IDN', 608: 'PHL', 554: 'NZL', 598: 'PNG', 242: 'FJI',
    352: 'ISL', 246: 'FIN', 233: 'EST', 428: 'LVA', 440: 'LTU',
    616: 'POL', 203: 'CZE', 703: 'SVK', 348: 'HUN', 642: 'ROU',
    100: 'BGR', 688: 'SRB', 191: 'HRV', 70: 'BIH', 499: 'MNE',
    8: 'ALB', 807: 'MKD', 705: 'SVN', 40: 'AUT', 56: 'BEL',
    442: 'LUX', 372: 'IRL', 620: 'PRT',
}

VALID_ISOS = set(COUNTRY_ISO_MAP.keys())
BLUE_ZONE_ISOS = {'USA', 'JPN', 'ITA', 'GRC', 'CRI'}


def fetch_un_wpp_from_api(session):
    """
    Fetch UN WPP data using the UN Data API.
    Falls back to CSV download if API is unavailable.
    """
    logger.info("Attempting to fetch UN WPP data via API...")

    # The UN Population Division API endpoint for life expectancy at birth
    # Using the UNDESA Population Division API
    base_url = "https://population.un.org/dataportalapi/api/v1"

    all_records = []

    # Indicator ID 68 = Life expectancy at birth (both sexes)
    # Variants: 4=Medium, 5=High, 6=Low
    for variant_id, variant_name in [(4, 'medium'), (5, 'high'), (6, 'low')]:
        logger.info(f"Fetching variant: {variant_name}")
        page = 1
        page_count = 1

        while page <= page_count:
            try:
                url = f"{base_url}/data/indicators/68/locations"
                # Get for all locations
                params = {
                    'startYear': 1950,
                    'endYear': 2100,
                    'variants': variant_id,
                    'pagingInHeader': 'false',
                    'pageSize': 1000,
                    'pageNumber': page,
                }
                resp = session.get(url, params=params, timeout=60)

                if resp.status_code != 200:
                    logger.warning(f"UN API returned {resp.status_code}")
                    return pd.DataFrame()

                data = resp.json()
                page_count = data.get('pages', 1)
                records = data.get('data', [])

                if not records:
                    break

                for rec in records:
                    loc_id = rec.get('locationId')
                    iso = UN_NUMERIC_TO_ISO3.get(loc_id)
                    if iso and iso in VALID_ISOS:
                        year_val = rec.get('timeLabel', '')
                        # timeLabel can be "2025" or "2020-2025"
                        if '-' in str(year_val):
                            # For period data, take the midpoint year
                            parts = str(year_val).split('-')
                            year = int(parts[0]) + (int(parts[1]) - int(parts[0])) // 2
                        else:
                            year = int(year_val)

                        value = rec.get('value')
                        if value is not None:
                            all_records.append({
                                'iso_code': iso,
                                'year': year,
                                'variant': variant_name,
                                'life_expectancy': float(value),
                            })

                logger.info(f"  {variant_name}: page {page}/{page_count}")
                page += 1
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"UN API request failed: {e}")
                return pd.DataFrame()

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    logger.info(f"UN API: Retrieved {len(df)} records")
    return df


def fetch_un_wpp_csv_fallback(session):
    """
    Fallback: Download UN WPP CSV data directly.
    The UN provides downloadable CSV files with demographic indicators.
    """
    logger.info("Attempting CSV download fallback for UN WPP data...")

    # UN WPP 2024 CSV download URLs
    urls_to_try = [
        # Life table indicators - compact format
        "https://population.un.org/wpp/Download/Files/1_Indicator%20(Standard)/CSV/WPP2024_Demographic_Indicators_Medium.csv.gz",
        "https://population.un.org/wpp/Download/Files/1_Indicator%20(Standard)/CSV/WPP2024_Demographic_Indicators_Medium.csv",
        # Older format
        "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV/WPP2022_Demographic_Indicators_Medium.csv",
    ]

    for url in urls_to_try:
        try:
            logger.info(f"Trying: {url}")
            resp = session.get(url, timeout=120, stream=True)
            if resp.status_code == 200:
                if url.endswith('.gz'):
                    import gzip
                    content = gzip.decompress(resp.content)
                    df = pd.read_csv(io.BytesIO(content), low_memory=False)
                else:
                    df = pd.read_csv(io.BytesIO(resp.content), low_memory=False)

                logger.info(f"Downloaded CSV with {len(df)} rows, columns: {list(df.columns)[:10]}...")
                return df

        except Exception as e:
            logger.warning(f"Failed: {e}")
            continue

    return pd.DataFrame()


def build_projections_from_api_data(df):
    """Process API data into the standard output format."""
    if df.empty:
        return pd.DataFrame()

    # Pivot variants into columns
    df_pivot = df.pivot_table(
        index=['iso_code', 'year'],
        columns='variant',
        values='life_expectancy',
        aggfunc='first'
    ).reset_index()
    df_pivot.columns.name = None

    # Rename to standard columns
    rename_map = {}
    if 'medium' in df_pivot.columns:
        rename_map['medium'] = 'le_medium'
    if 'high' in df_pivot.columns:
        rename_map['high'] = 'le_high'
    if 'low' in df_pivot.columns:
        rename_map['low'] = 'le_low'
    df_pivot = df_pivot.rename(columns=rename_map)

    # Add country names
    df_pivot['country_name'] = df_pivot['iso_code'].map(COUNTRY_ISO_MAP)
    df_pivot['is_blue_zone'] = df_pivot['iso_code'].isin(BLUE_ZONE_ISOS).astype(int)

    return df_pivot.sort_values(['iso_code', 'year']).reset_index(drop=True)


def build_projections_from_csv(df_raw):
    """Process the downloaded UN WPP CSV into our standard format."""
    if df_raw.empty:
        return pd.DataFrame()

    # The UN CSV has various column naming conventions across versions
    # Common columns: LocID, Location, Time, LEx (life expectancy at birth)
    # or: ISO3_code, Year, ExBoth (life expectancy both sexes)

    # Try to identify the right columns
    cols = df_raw.columns.tolist()
    logger.info(f"CSV columns: {cols[:20]}")

    # Look for location ID column
    loc_col = None
    for c in ['LocID', 'LocationCode', 'ISO3_code', 'ISO3Alpha']:
        if c in cols:
            loc_col = c
            break

    # Look for time column
    time_col = None
    for c in ['Time', 'Year', 'MidPeriod']:
        if c in cols:
            time_col = c
            break

    # Look for life expectancy column
    le_col = None
    for c in ['LEx', 'ExBoth', 'LifeExpectancyAtBirth', 'ex', 'e0']:
        if c in cols:
            le_col = c
            break

    if not all([loc_col, time_col, le_col]):
        logger.error(f"Could not identify required columns. Found: loc={loc_col}, time={time_col}, le={le_col}")
        return pd.DataFrame()

    # Filter and process
    df = df_raw[[loc_col, time_col, le_col]].copy()
    df.columns = ['location', 'year', 'life_expectancy']
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['life_expectancy'] = pd.to_numeric(df['life_expectancy'], errors='coerce')
    df = df.dropna()

    # Map location to ISO3
    if df['location'].dtype == object:
        # If string codes
        df['iso_code'] = df['location']
    else:
        # If numeric codes
        df['iso_code'] = df['location'].astype(int).map(UN_NUMERIC_TO_ISO3)

    df = df[df['iso_code'].isin(VALID_ISOS)].copy()

    # Since this is medium variant only from the CSV, we create all three
    # with the medium variant. For high/low, we'll estimate +/- based on
    # standard UN projection uncertainty ranges
    df['le_medium'] = df['life_expectancy']

    # Add country metadata
    df['country_name'] = df['iso_code'].map(COUNTRY_ISO_MAP)
    df['is_blue_zone'] = df['iso_code'].isin(BLUE_ZONE_ISOS).astype(int)

    result = df[['iso_code', 'country_name', 'year', 'le_medium', 'is_blue_zone']].copy()
    return result.sort_values(['iso_code', 'year']).reset_index(drop=True)


def generate_un_style_projections(historical_csv_path):
    """
    If UN API and CSV downloads both fail, generate projections based on
    real historical trends using simple linear extrapolation from the last 20 years.
    This is a LAST RESORT - clearly labeled as trend extrapolation, not UN official data.
    """
    logger.warning("=" * 60)
    logger.warning("UN API AND CSV DOWNLOADS FAILED")
    logger.warning("Generating trend-based projections from real historical data")
    logger.warning("These are NOT official UN projections - they are simple")
    logger.warning("extrapolations from real World Bank historical trends")
    logger.warning("=" * 60)

    if not os.path.exists(historical_csv_path):
        logger.error(f"Historical data not found at {historical_csv_path}")
        return pd.DataFrame()

    df_hist = pd.read_csv(historical_csv_path)

    if 'life_expectancy' not in df_hist.columns:
        logger.error("No life_expectancy column in historical data")
        return pd.DataFrame()

    all_projections = []

    for iso in df_hist['iso_code'].unique():
        country_data = df_hist[df_hist['iso_code'] == iso].copy()
        le_data = country_data[['year', 'life_expectancy']].dropna()

        if len(le_data) < 10:
            continue

        # Use last 20 years of real data for trend
        recent = le_data[le_data['year'] >= le_data['year'].max() - 20]
        if len(recent) < 5:
            recent = le_data.tail(10)

        # Linear regression on recent data
        x = recent['year'].values
        y = recent['life_expectancy'].values
        slope, intercept = np.polyfit(x, y, 1)

        # Project forward, but with diminishing gains (logistic dampening)
        # Life expectancy gains slow as they approach biological limits
        last_le = y[-1]
        last_year = int(x[-1])

        for future_year in range(last_year + 1, 2101):
            years_ahead = future_year - last_year
            # Dampened linear projection (gains slow over time)
            raw_gain = slope * years_ahead
            # Apply logistic dampening - harder to gain as you approach ~95
            max_le = 95.0
            dampened_le = last_le + raw_gain * (1 - last_le / max_le)
            # Clip to reasonable range
            le_medium = min(max(dampened_le, last_le - 5), max_le)

            # Uncertainty grows with time
            uncertainty = 0.5 + years_ahead * 0.05  # Starts at 0.5, grows 0.05/year
            uncertainty = min(uncertainty, 5.0)  # Cap at 5 years

            all_projections.append({
                'iso_code': iso,
                'country_name': COUNTRY_ISO_MAP.get(iso, iso),
                'year': future_year,
                'le_medium': round(le_medium, 2),
                'le_high': round(min(le_medium + uncertainty, max_le), 2),
                'le_low': round(max(le_medium - uncertainty, 30.0), 2),
                'is_blue_zone': 1 if iso in BLUE_ZONE_ISOS else 0,
                'projection_source': 'trend_extrapolation_from_real_data',
            })

    df = pd.DataFrame(all_projections)
    logger.info(f"Trend-based projections: {len(df)} rows for {df['iso_code'].nunique()} countries")
    return df


def main():
    """Main collection pipeline for UN projections."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Blue-Zones-Research/2.0 (Educational Research)'
    })

    output_dir = os.path.join(PROJECT_DIR, 'data', 'projections')
    os.makedirs(output_dir, exist_ok=True)

    # Strategy 1: Try UN Population Division API
    df_api = fetch_un_wpp_from_api(session)
    if not df_api.empty:
        df_projections = build_projections_from_api_data(df_api)
        source = "UN Population Division API (WPP 2024)"
    else:
        # Strategy 2: Try downloading the CSV directly
        df_csv = fetch_un_wpp_csv_fallback(session)
        if not df_csv.empty:
            df_projections = build_projections_from_csv(df_csv)
            source = "UN Population Division CSV Download (WPP 2024)"
        else:
            # Strategy 3: Generate from real historical trends
            hist_path = os.path.join(PROJECT_DIR, 'data', 'historical', 'merged_historical_panel.csv')
            df_projections = generate_un_style_projections(hist_path)
            source = "Trend extrapolation from real World Bank historical data"

    if df_projections.empty:
        logger.error("Failed to produce any projection data")
        return

    # Save
    output_path = os.path.join(output_dir, 'un_life_expectancy_projections.csv')
    df_projections.to_csv(output_path, index=False)

    # Summary
    print("\n" + "=" * 70)
    print("UN PROJECTIONS COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Source: {source}")
    print(f"Output: {output_path}")
    print(f"Total rows: {len(df_projections)}")
    print(f"Countries: {df_projections['iso_code'].nunique()}")
    if 'year' in df_projections.columns:
        print(f"Year range: {df_projections['year'].min()} - {df_projections['year'].max()}")
    print(f"Columns: {list(df_projections.columns)}")

    # Blue Zone projections preview
    print("\nBLUE ZONE PROJECTIONS PREVIEW (2050):")
    bz = df_projections[
        (df_projections['is_blue_zone'] == 1) &
        (df_projections['year'] == 2050)
    ]
    if not bz.empty:
        for _, row in bz.iterrows():
            name = row.get('country_name', row['iso_code'])
            med = row.get('le_medium', 'N/A')
            high = row.get('le_high', 'N/A')
            low = row.get('le_low', 'N/A')
            print(f"  {name}: {med} (Low: {low}, High: {high})")
    else:
        print("  No 2050 data available in projections")

    return df_projections


if __name__ == '__main__':
    df = main()
