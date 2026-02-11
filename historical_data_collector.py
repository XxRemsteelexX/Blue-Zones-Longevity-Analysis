#!/usr/bin/env python3
"""
Historical Data Collector for Blue Zones Longevity Analysis
Pulls 1960-2023 data from World Bank API and WHO GHO API for all 93 countries.
All data is REAL - no synthetic or fake data.

Data Sources:
- World Bank Data API (https://data.worldbank.org/)
- WHO Global Health Observatory API (https://ghoapi.azureedge.net/api/)
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project root
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Complete ISO3 code to country name mapping for all 93 countries in the project
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

# Reverse mapping
COUNTRY_NAME_TO_ISO = {v: k for k, v in COUNTRY_ISO_MAP.items()}

# Blue Zone countries
BLUE_ZONE_ISOS = {'USA', 'JPN', 'ITA', 'GRC', 'CRI'}
BLUE_ZONE_REGIONS = {
    'USA': 'Loma Linda', 'JPN': 'Okinawa', 'ITA': 'Sardinia',
    'GRC': 'Ikaria', 'CRI': 'Nicoya'
}

# Country coordinates
COUNTRY_COORDS = {
    'USA': (39.8283, -98.5795), 'JPN': (36.2048, 138.2529),
    'ITA': (41.8719, 12.5674), 'GRC': (39.0742, 21.8243),
    'CRI': (9.7489, -83.7534), 'CHN': (35.8617, 104.1954),
    'IND': (20.5937, 78.9629), 'BRA': (-14.2350, -51.9253),
    'RUS': (61.5240, 105.3188), 'CAN': (56.1304, -106.3468),
    'AUS': (-25.2744, 133.7751), 'DEU': (51.1657, 10.4515),
    'GBR': (55.3781, -3.4360), 'FRA': (46.6034, 1.8883),
    'ESP': (40.4637, -3.7492), 'NOR': (60.4720, 8.4689),
    'SWE': (60.1282, 18.6435), 'DNK': (56.2639, 9.5018),
    'CHE': (46.8182, 8.2275), 'NLD': (52.1326, 5.2913),
    'KOR': (35.9078, 127.7669), 'MEX': (23.6345, -102.5528),
    'ARG': (-38.4161, -63.6167), 'CHL': (-35.6751, -71.5430),
    'PER': (-9.1900, -75.0152), 'COL': (4.5709, -74.2973),
    'VEN': (6.4238, -66.5897), 'ECU': (-1.8312, -78.1834),
    'BOL': (-16.2902, -63.5887), 'PRY': (-23.4425, -58.4438),
    'URY': (-32.5228, -55.7658), 'ZAF': (-30.5595, 22.9375),
    'EGY': (26.0975, 30.0444), 'MAR': (31.7917, -7.0926),
    'DZA': (28.0339, 1.6596), 'TUN': (33.8869, 9.5375),
    'LBY': (26.3351, 17.2283), 'NGA': (9.0820, 8.6753),
    'KEN': (-0.0236, 37.9062), 'ETH': (9.1450, 40.4897),
    'GHA': (7.9465, -1.0232), 'TZA': (-6.3690, 34.8888),
    'UGA': (1.3733, 32.2903), 'MOZ': (-18.6657, 35.5296),
    'MDG': (-18.7669, 46.8691), 'BWA': (-22.3285, 24.6849),
    'NAM': (-22.9576, 18.4904), 'ZMB': (-13.1339, 27.8493),
    'ZWE': (-19.0154, 29.1549), 'TUR': (38.9637, 35.2433),
    'IRN': (32.4279, 53.6880), 'IRQ': (33.2232, 43.6793),
    'SAU': (23.8859, 45.0792), 'ISR': (31.0461, 34.8516),
    'JOR': (30.5852, 36.2384), 'LBN': (33.8547, 35.8623),
    'SYR': (34.8021, 38.9968), 'AFG': (33.9391, 67.7100),
    'PAK': (30.3753, 69.3451), 'BGD': (23.6850, 90.3563),
    'MMR': (21.9162, 95.9560), 'THA': (15.8700, 100.9925),
    'VNM': (14.0583, 108.2772), 'MYS': (4.2105, 101.9758),
    'SGP': (1.3521, 103.8198), 'IDN': (-0.7893, 113.9213),
    'PHL': (12.8797, 121.7740), 'NZL': (-40.9006, 174.8860),
    'PNG': (-6.3149, 143.9555), 'FJI': (-16.5782, 179.4144),
    'ISL': (64.9631, -19.0208), 'FIN': (61.9241, 25.7482),
    'EST': (58.5953, 25.0136), 'LVA': (56.8796, 24.6032),
    'LTU': (55.1694, 23.8813), 'POL': (51.9194, 19.1451),
    'CZE': (49.8175, 15.4730), 'SVK': (48.6690, 19.6990),
    'HUN': (47.1625, 19.5033), 'ROU': (45.9432, 24.9668),
    'BGR': (42.7339, 25.4858), 'SRB': (44.0165, 21.0059),
    'HRV': (45.1000, 15.2000), 'BIH': (43.9159, 17.6791),
    'MNE': (42.7087, 19.3744), 'ALB': (41.1533, 20.1683),
    'MKD': (41.6086, 21.7453), 'SVN': (46.1512, 14.9955),
    'AUT': (47.5162, 14.5501), 'BEL': (50.5039, 4.4699),
    'LUX': (49.8153, 6.1296), 'IRL': (53.4129, -8.2439),
    'PRT': (39.3999, -8.2245),
}

# World Bank indicators to pull
WB_INDICATORS = {
    'SP.DYN.LE00.IN': 'life_expectancy',
    'NY.GDP.PCAP.CD': 'gdp_per_capita',
    'SH.MED.PHYS.ZS': 'physicians_per_1000',
    'SP.URB.TOTL.IN.ZS': 'urban_population_pct',
    'SP.POP.TOTL': 'population_total',
    'EN.ATM.PM25.MC.M3': 'pm25_air_pollution',
    'SH.XPD.CHEX.PC.CD': 'health_expenditure_pc',
    'SP.DYN.CDRT.IN': 'death_rate',
    'AG.LND.FRST.ZS': 'forest_area_pct',
}

# Valid ISO codes for filtering World Bank responses (exclude aggregates)
VALID_ISOS = set(COUNTRY_ISO_MAP.keys())


def fetch_wb_indicator_all_years(indicator_code, indicator_name, session):
    """
    Fetch a single World Bank indicator for ALL countries, ALL years 1960-2023.
    Handles the World Bank's [metadata, data] response format with pagination.
    """
    base_url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
    all_records = []
    page = 1
    total_pages = 1

    while page <= total_pages:
        params = {
            'format': 'json',
            'date': '1960:2023',
            'per_page': 1000,
            'page': page,
        }
        try:
            resp = session.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # World Bank returns [metadata_dict, [records]]
            if not isinstance(data, list) or len(data) < 2:
                logger.warning(f"Unexpected WB response format for {indicator_name} page {page}")
                break

            metadata = data[0]
            records = data[1]

            if metadata and 'pages' in metadata:
                total_pages = metadata['pages']

            if not records:
                break

            for rec in records:
                iso = rec.get('countryiso3code', '')
                val = rec.get('value')
                year = rec.get('date', '')

                # Only keep our 93 countries (skip WB aggregates like "World", "EUU", etc.)
                if iso in VALID_ISOS and val is not None:
                    all_records.append({
                        'iso_code': iso,
                        'year': int(year),
                        'indicator': indicator_name,
                        'value': float(val),
                    })

            logger.info(f"  {indicator_name}: page {page}/{total_pages} ({len(records)} records)")
            page += 1
            time.sleep(0.3)  # Rate limiting

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {indicator_name} page {page}: {e}")
            break
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Parse error for {indicator_name} page {page}: {e}")
            break

    logger.info(f"  {indicator_name}: total {len(all_records)} valid records")
    return all_records


def fetch_all_wb_historical(session):
    """Fetch all World Bank indicators for 1960-2023."""
    logger.info("=" * 60)
    logger.info("FETCHING WORLD BANK HISTORICAL DATA (1960-2023)")
    logger.info("=" * 60)

    all_records = []
    for code, name in WB_INDICATORS.items():
        logger.info(f"Fetching: {name} ({code})")
        records = fetch_wb_indicator_all_years(code, name, session)
        all_records.extend(records)
        time.sleep(1.0)  # Extra pause between indicators

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.error("No World Bank data retrieved!")
        return pd.DataFrame()

    # Pivot so each indicator becomes a column
    df_wide = df.pivot_table(
        index=['iso_code', 'year'],
        columns='indicator',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Flatten column names
    df_wide.columns.name = None

    logger.info(f"World Bank historical data: {len(df_wide)} country-year rows, {len(df_wide.columns)} columns")
    return df_wide


def fetch_who_historical(session):
    """
    Fetch WHO GHO historical data for life expectancy and mortality indicators.
    The WHO API returns the full time series.
    """
    logger.info("=" * 60)
    logger.info("FETCHING WHO HISTORICAL DATA")
    logger.info("=" * 60)

    who_indicators = {
        'WHOSIS_000001': 'life_expectancy_who',
        'WHOSIS_000004': 'maternal_mortality_who',
        'WHOSIS_000007': 'infant_mortality_who',
    }

    all_records = []

    for code, name in who_indicators.items():
        logger.info(f"Fetching WHO: {name} ({code})")
        try:
            url = f"https://ghoapi.azureedge.net/api/{code}"
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            count = 0
            for rec in data.get('value', []):
                iso = rec.get('SpatialDim', '')
                year = rec.get('TimeDim')
                val_str = str(rec.get('Value', ''))

                if iso not in VALID_ISOS or not year or not val_str:
                    continue

                # Parse value - WHO sometimes returns "67.1 [67.1-67.2]"
                try:
                    if '[' in val_str:
                        val = float(val_str.split('[')[0].strip())
                    else:
                        val = float(val_str)
                except (ValueError, TypeError):
                    continue

                all_records.append({
                    'iso_code': iso,
                    'year': int(year),
                    'indicator': name,
                    'value': val,
                })
                count += 1

            logger.info(f"  {name}: {count} valid records")
            time.sleep(2.0)  # WHO rate limiting

        except Exception as e:
            logger.warning(f"Failed to fetch WHO {name}: {e}")
            continue

    if not all_records:
        logger.warning("No WHO data retrieved")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Pivot
    df_wide = df.pivot_table(
        index=['iso_code', 'year'],
        columns='indicator',
        values='value',
        aggfunc='first'
    ).reset_index()
    df_wide.columns.name = None

    logger.info(f"WHO historical data: {len(df_wide)} country-year rows")
    return df_wide


def add_country_metadata(df):
    """Add country name, blue zone flag, coordinates, and gravity calculations."""
    df = df.copy()

    # Map ISO to country name
    df['country_name'] = df['iso_code'].map(COUNTRY_ISO_MAP)

    # Blue zone flag
    df['is_blue_zone'] = df['iso_code'].isin(BLUE_ZONE_ISOS).astype(int)
    df['blue_zone_region'] = df['iso_code'].map(BLUE_ZONE_REGIONS)

    # Coordinates
    df['latitude'] = df['iso_code'].map(lambda x: COUNTRY_COORDS.get(x, (None, None))[0])
    df['longitude'] = df['iso_code'].map(lambda x: COUNTRY_COORDS.get(x, (None, None))[1])

    # Gravity calculation (International Gravity Formula IGF 1980)
    lat_rad = np.radians(df['latitude'].astype(float))
    df['effective_gravity'] = 9.78033 * (
        1 + 0.0053024 * np.sin(lat_rad)**2
        - 0.0000058 * np.sin(2 * lat_rad)**2
    )
    df['gravity_deviation'] = df['effective_gravity'] - 9.80665

    return df


def main():
    """Main collection pipeline."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Blue-Zones-Research/2.0 (Educational Research)'
    })

    output_dir = os.path.join(PROJECT_DIR, 'data', 'historical')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Fetch World Bank historical data
    df_wb = fetch_all_wb_historical(session)

    # 2. Fetch WHO historical data
    df_who = fetch_who_historical(session)

    # 3. Merge World Bank + WHO on (iso_code, year)
    if not df_wb.empty and not df_who.empty:
        df_merged = pd.merge(df_wb, df_who, on=['iso_code', 'year'], how='outer')
    elif not df_wb.empty:
        df_merged = df_wb
    elif not df_who.empty:
        df_merged = df_who
    else:
        logger.error("No data from any source. Exiting.")
        return

    # 4. Add country metadata
    df_merged = add_country_metadata(df_merged)

    # Filter to only our 93 countries
    df_merged = df_merged[df_merged['iso_code'].isin(VALID_ISOS)].copy()

    # Sort
    df_merged = df_merged.sort_values(['iso_code', 'year']).reset_index(drop=True)

    # 5. Save
    output_path = os.path.join(output_dir, 'merged_historical_panel.csv')
    df_merged.to_csv(output_path, index=False)

    # Also save the raw sources separately for reproducibility
    if not df_wb.empty:
        df_wb.to_csv(os.path.join(output_dir, 'wb_historical_raw.csv'), index=False)
    if not df_who.empty:
        df_who.to_csv(os.path.join(output_dir, 'who_historical_raw.csv'), index=False)

    # 6. Print summary
    print("\n" + "=" * 70)
    print("HISTORICAL DATA COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Total rows: {len(df_merged)}")
    print(f"Countries: {df_merged['iso_code'].nunique()}")
    print(f"Year range: {df_merged['year'].min()} - {df_merged['year'].max()}")
    print(f"Columns: {list(df_merged.columns)}")

    # Blue Zone country data check
    print("\nBLUE ZONE COUNTRY DATA CHECK:")
    for iso in sorted(BLUE_ZONE_ISOS):
        bz = df_merged[df_merged['iso_code'] == iso]
        le_years = bz['life_expectancy'].dropna().shape[0] if 'life_expectancy' in bz.columns else 0
        name = COUNTRY_ISO_MAP[iso]
        region = BLUE_ZONE_REGIONS[iso]
        yr_range = f"{bz['year'].min()}-{bz['year'].max()}" if len(bz) > 0 else "NO DATA"
        print(f"  {name} ({region}): {len(bz)} rows, {le_years} years with life_expectancy, range: {yr_range}")

    # Completeness report
    print("\nDATA COMPLETENESS (% of country-year rows with non-null values):")
    data_cols = [c for c in df_merged.columns if c not in [
        'iso_code', 'year', 'country_name', 'is_blue_zone', 'blue_zone_region',
        'latitude', 'longitude', 'effective_gravity', 'gravity_deviation'
    ]]
    for col in data_cols:
        pct = (1 - df_merged[col].isna().mean()) * 100
        print(f"  {col}: {pct:.1f}%")

    return df_merged


if __name__ == '__main__':
    df = main()
