#!/usr/bin/env python3
"""
Improved Real World Data Collector for Blue Zones Research
Uses multiple APIs and fallback methods for comprehensive real-world data
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedRealDataCollector:
    """
    Improved real-world data collector with multiple fallback sources
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Blue-Zones-Research/1.0 (Educational Research)'
        })
        
        # Known Blue Zone locations (ground truth)
        self.blue_zones = {
            'Italy': {'region': 'Sardinia', 'latitude': 40.1209, 'longitude': 9.0129},
            'Japan': {'region': 'Okinawa', 'latitude': 26.5012, 'longitude': 127.9688},
            'Costa Rica': {'region': 'Nicoya', 'latitude': 10.1484, 'longitude': -85.4526},
            'Greece': {'region': 'Ikaria', 'latitude': 37.6047, 'longitude': 26.1698},
            'United States': {'region': 'Loma Linda', 'latitude': 34.0522, 'longitude': -117.2437}
        }
        
        # Country coordinates (major countries)
        self.country_coords = {
            'United States': {'latitude': 39.8283, 'longitude': -98.5795},
            'Japan': {'latitude': 36.2048, 'longitude': 138.2529},
            'Italy': {'latitude': 41.8719, 'longitude': 12.5674},
            'Greece': {'latitude': 39.0742, 'longitude': 21.8243},
            'Costa Rica': {'latitude': 9.7489, 'longitude': -83.7534},
            'China': {'latitude': 35.8617, 'longitude': 104.1954},
            'India': {'latitude': 20.5937, 'longitude': 78.9629},
            'Brazil': {'latitude': -14.2350, 'longitude': -51.9253},
            'Russia': {'latitude': 61.5240, 'longitude': 105.3188},
            'Canada': {'latitude': 56.1304, 'longitude': -106.3468},
            'Australia': {'latitude': -25.2744, 'longitude': 133.7751},
            'Germany': {'latitude': 51.1657, 'longitude': 10.4515},
            'United Kingdom': {'latitude': 55.3781, 'longitude': -3.4360},
            'France': {'latitude': 46.6034, 'longitude': 1.8883},
            'Spain': {'latitude': 40.4637, 'longitude': -3.7492},
            'Norway': {'latitude': 60.4720, 'longitude': 8.4689},
            'Sweden': {'latitude': 60.1282, 'longitude': 18.6435},
            'Denmark': {'latitude': 56.2639, 'longitude': 9.5018},
            'Switzerland': {'latitude': 46.8182, 'longitude': 8.2275},
            'Netherlands': {'latitude': 52.1326, 'longitude': 5.2913},
            'South Korea': {'latitude': 35.9078, 'longitude': 127.7669},
            'Mexico': {'latitude': 23.6345, 'longitude': -102.5528},
            'Argentina': {'latitude': -38.4161, 'longitude': -63.6167},
            'Chile': {'latitude': -35.6751, 'longitude': -71.5430},
            'Peru': {'latitude': -9.1900, 'longitude': -75.0152},
            'Colombia': {'latitude': 4.5709, 'longitude': -74.2973},
            'Venezuela': {'latitude': 6.4238, 'longitude': -66.5897},
            'Ecuador': {'latitude': -1.8312, 'longitude': -78.1834},
            'Bolivia': {'latitude': -16.2902, 'longitude': -63.5887},
            'Paraguay': {'latitude': -23.4425, 'longitude': -58.4438},
            'Uruguay': {'latitude': -32.5228, 'longitude': -55.7658},
            'South Africa': {'latitude': -30.5595, 'longitude': 22.9375},
            'Egypt': {'latitude': 26.0975, 'longitude': 30.0444},
            'Morocco': {'latitude': 31.7917, 'longitude': -7.0926},
            'Algeria': {'latitude': 28.0339, 'longitude': 1.6596},
            'Tunisia': {'latitude': 33.8869, 'longitude': 9.5375},
            'Libya': {'latitude': 26.3351, 'longitude': 17.2283},
            'Nigeria': {'latitude': 9.0820, 'longitude': 8.6753},
            'Kenya': {'latitude': -0.0236, 'longitude': 37.9062},
            'Ethiopia': {'latitude': 9.1450, 'longitude': 40.4897},
            'Ghana': {'latitude': 7.9465, 'longitude': -1.0232},
            'Tanzania': {'latitude': -6.3690, 'longitude': 34.8888},
            'Uganda': {'latitude': 1.3733, 'longitude': 32.2903},
            'Mozambique': {'latitude': -18.6657, 'longitude': 35.5296},
            'Madagascar': {'latitude': -18.7669, 'longitude': 46.8691},
            'Botswana': {'latitude': -22.3285, 'longitude': 24.6849},
            'Namibia': {'latitude': -22.9576, 'longitude': 18.4904},
            'Zambia': {'latitude': -13.1339, 'longitude': 27.8493},
            'Zimbabwe': {'latitude': -19.0154, 'longitude': 29.1549},
            'Turkey': {'latitude': 38.9637, 'longitude': 35.2433},
            'Iran': {'latitude': 32.4279, 'longitude': 53.6880},
            'Iraq': {'latitude': 33.2232, 'longitude': 43.6793},
            'Saudi Arabia': {'latitude': 23.8859, 'longitude': 45.0792},
            'Israel': {'latitude': 31.0461, 'longitude': 34.8516},
            'Jordan': {'latitude': 30.5852, 'longitude': 36.2384},
            'Lebanon': {'latitude': 33.8547, 'longitude': 35.8623},
            'Syria': {'latitude': 34.8021, 'longitude': 38.9968},
            'Afghanistan': {'latitude': 33.9391, 'longitude': 67.7100},
            'Pakistan': {'latitude': 30.3753, 'longitude': 69.3451},
            'Bangladesh': {'latitude': 23.6850, 'longitude': 90.3563},
            'Myanmar': {'latitude': 21.9162, 'longitude': 95.9560},
            'Thailand': {'latitude': 15.8700, 'longitude': 100.9925},
            'Vietnam': {'latitude': 14.0583, 'longitude': 108.2772},
            'Malaysia': {'latitude': 4.2105, 'longitude': 101.9758},
            'Singapore': {'latitude': 1.3521, 'longitude': 103.8198},
            'Indonesia': {'latitude': -0.7893, 'longitude': 113.9213},
            'Philippines': {'latitude': 12.8797, 'longitude': 121.7740},
            'New Zealand': {'latitude': -40.9006, 'longitude': 174.8860},
            'Papua New Guinea': {'latitude': -6.3149, 'longitude': 143.9555},
            'Fiji': {'latitude': -16.5782, 'longitude': 179.4144},
            'Iceland': {'latitude': 64.9631, 'longitude': -19.0208},
            'Finland': {'latitude': 61.9241, 'longitude': 25.7482},
            'Estonia': {'latitude': 58.5953, 'longitude': 25.0136},
            'Latvia': {'latitude': 56.8796, 'longitude': 24.6032},
            'Lithuania': {'latitude': 55.1694, 'longitude': 23.8813},
            'Poland': {'latitude': 51.9194, 'longitude': 19.1451},
            'Czech Republic': {'latitude': 49.8175, 'longitude': 15.4730},
            'Slovakia': {'latitude': 48.6690, 'longitude': 19.6990},
            'Hungary': {'latitude': 47.1625, 'longitude': 19.5033},
            'Romania': {'latitude': 45.9432, 'longitude': 24.9668},
            'Bulgaria': {'latitude': 42.7339, 'longitude': 25.4858},
            'Serbia': {'latitude': 44.0165, 'longitude': 21.0059},
            'Croatia': {'latitude': 45.1000, 'longitude': 15.2000},
            'Bosnia and Herzegovina': {'latitude': 43.9159, 'longitude': 17.6791},
            'Montenegro': {'latitude': 42.7087, 'longitude': 19.3744},
            'Albania': {'latitude': 41.1533, 'longitude': 20.1683},
            'North Macedonia': {'latitude': 41.6086, 'longitude': 21.7453},
            'Slovenia': {'latitude': 46.1512, 'longitude': 14.9955},
            'Austria': {'latitude': 47.5162, 'longitude': 14.5501},
            'Belgium': {'latitude': 50.5039, 'longitude': 4.4699},
            'Luxembourg': {'latitude': 49.8153, 'longitude': 6.1296},
            'Ireland': {'latitude': 53.4129, 'longitude': -8.2439},
            'Portugal': {'latitude': 39.3999, 'longitude': -8.2245},
        }
        
    def collect_world_bank_simple(self) -> pd.DataFrame:
        """Collect basic World Bank data using simple REST API"""
        logger.info("Fetching World Bank data using REST API...")
        
        # Key indicators we want
        indicators = {
            'SP.DYN.LE00.IN': 'life_expectancy',
            'NY.GDP.PCAP.CD': 'gdp_per_capita', 
            'SH.MED.PHYS.ZS': 'physicians_per_1000',
            'SP.URB.TOTL.IN.ZS': 'urban_population_pct',
            'SP.POP.TOTL': 'population_total',
            'AG.LND.FRST.ZS': 'forest_area_pct',
            'EN.ATM.PM25.MC.M3': 'pm25_air_pollution'
        }
        
        all_data = []
        
        for indicator_code, indicator_name in indicators.items():
            try:
                url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}?format=json&date=2019:2022&per_page=500"
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1 and data[1]:  # Check if data exists
                        for record in data[1]:
                            if record.get('value') is not None:
                                all_data.append({
                                    'country_name': record['country']['value'],
                                    'iso_code': record['countryiso3code'],
                                    'indicator': indicator_name,
                                    'value': float(record['value']),
                                    'year': int(record['date']),
                                    'source': 'WorldBank'
                                })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to fetch World Bank indicator {indicator_name}: {e}")
                continue
        
        if not all_data:
            logger.warning("No World Bank data retrieved")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)
        
        # Pivot to have indicators as columns
        df_pivot = df.pivot_table(
            index=['country_name', 'iso_code', 'year'],
            columns='indicator',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Get most recent data for each country
        df_latest = df_pivot.sort_values('year').groupby(['country_name', 'iso_code']).last().reset_index()
        
        logger.info(f"Collected World Bank data for {len(df_latest)} countries")
        return df_latest
    
    def collect_who_simple(self) -> pd.DataFrame:
        """Collect WHO data using their API"""
        logger.info("Fetching WHO health data...")
        
        who_data = []
        
        # Basic WHO health indicators
        indicators = {
            'WHOSIS_000001': 'life_expectancy_who',
            'WHOSIS_000004': 'maternal_mortality',
            'WHOSIS_000007': 'infant_mortality'
        }
        
        for code, name in indicators.items():
            try:
                url = f"https://ghoapi.azureedge.net/api/{code}"
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for record in data.get('value', []):
                        if record.get('SpatialDim') and record.get('Value'):
                            try:
                                # Handle values like "67.1 [67.1-67.2]"
                                value_str = str(record.get('Value', ''))
                                if '[' in value_str:
                                    value = float(value_str.split('[')[0].strip())
                                else:
                                    value = float(value_str)
                                
                                who_data.append({
                                    'iso_code': record.get('SpatialDim'),
                                    'indicator': name,
                                    'value': value,
                                    'year': record.get('TimeDim'),
                                    'source': 'WHO'
                                })
                            except (ValueError, TypeError):
                                continue
                
                time.sleep(2.0)  # WHO rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to fetch WHO indicator {name}: {e}")
                continue
        
        if not who_data:
            return pd.DataFrame()
            
        df_who = pd.DataFrame(who_data)
        df_who_pivot = df_who.pivot_table(
            index=['iso_code', 'year'],
            columns='indicator', 
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Get most recent data
        df_who_latest = df_who_pivot.groupby('iso_code').last().reset_index()
        
        logger.info(f"Collected WHO data for {len(df_who_latest)} countries")
        return df_who_latest
    
    def create_country_base(self) -> pd.DataFrame:
        """Create base country dataset with coordinates"""
        logger.info("Creating base country dataset...")
        
        countries_data = []
        
        for country_name, coords in self.country_coords.items():
            countries_data.append({
                'country_name': country_name,
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'is_blue_zone': 1 if country_name in self.blue_zones else 0,
                'blue_zone_region': self.blue_zones.get(country_name, {}).get('region', None)
            })
        
        df = pd.DataFrame(countries_data)
        logger.info(f"Created base dataset with {len(df)} countries")
        
        return df
    
    def calculate_gravity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gravity variations"""
        logger.info("Calculating gravity variations...")
        
        if 'latitude' not in df.columns:
            logger.warning("No latitude data available for gravity calculations")
            return df
            
        df = df.copy()
        
        # Physical constants
        STANDARD_GRAVITY = 9.80665  # m/s²
        EQUATORIAL_GRAVITY = 9.78033  # m/s²
        
        # International Gravity Formula
        lat_rad = np.radians(df['latitude'])
        gravity_lat = (EQUATORIAL_GRAVITY * 
                      (1 + 0.0053024 * np.sin(lat_rad)**2 - 
                       0.0000058 * np.sin(2 * lat_rad)**2))
        
        df['effective_gravity'] = gravity_lat
        df['gravity_deviation'] = gravity_lat - STANDARD_GRAVITY
        df['gravity_deviation_pct'] = (df['gravity_deviation'] / STANDARD_GRAVITY) * 100
        
        return df
    
    def collect_comprehensive_real_data(self) -> pd.DataFrame:
        """Collect comprehensive real-world dataset"""
        logger.info("Starting comprehensive real-world data collection...")
        
        # Create base country dataset
        df_base = self.create_country_base()
        
        # Collect World Bank data
        df_wb = self.collect_world_bank_simple()
        
        # Collect WHO data  
        df_who = self.collect_who_simple()
        
        # Merge datasets
        logger.info("Merging datasets...")
        
        # Start with base
        df_final = df_base.copy()
        
        # Add World Bank data
        if not df_wb.empty:
            # Match countries by name (simplified)
            df_final = df_final.merge(
                df_wb[['country_name'] + [col for col in df_wb.columns if col not in ['country_name', 'iso_code', 'year']]],
                on='country_name',
                how='left'
            )
        
        # Add WHO data (will need ISO code mapping)
        if not df_who.empty:
            # Create simple ISO code mapping
            iso_mapping = {
                'USA': 'United States', 'JPN': 'Japan', 'ITA': 'Italy',
                'GRC': 'Greece', 'CRI': 'Costa Rica', 'CHN': 'China',
                'IND': 'India', 'BRA': 'Brazil', 'RUS': 'Russia',
                'CAN': 'Canada', 'AUS': 'Australia', 'DEU': 'Germany',
                'GBR': 'United Kingdom', 'FRA': 'France', 'ESP': 'Spain',
                'NOR': 'Norway', 'SWE': 'Sweden', 'DNK': 'Denmark',
                'CHE': 'Switzerland', 'NLD': 'Netherlands', 'KOR': 'South Korea',
                'MEX': 'Mexico', 'ARG': 'Argentina', 'CHL': 'Chile',
                'PER': 'Peru', 'COL': 'Colombia', 'VEN': 'Venezuela',
                'ZAF': 'South Africa', 'EGY': 'Egypt', 'MAR': 'Morocco',
                'NGA': 'Nigeria', 'KEN': 'Kenya', 'ETH': 'Ethiopia',
                'TUR': 'Turkey', 'IRN': 'Iran', 'IRQ': 'Iraq',
                'SAU': 'Saudi Arabia', 'ISR': 'Israel', 'JOR': 'Jordan',
                'AFG': 'Afghanistan', 'PAK': 'Pakistan', 'BGD': 'Bangladesh',
                'THA': 'Thailand', 'VNM': 'Vietnam', 'MYS': 'Malaysia',
                'SGP': 'Singapore', 'IDN': 'Indonesia', 'PHL': 'Philippines',
                'NZL': 'New Zealand', 'ISL': 'Iceland', 'FIN': 'Finland'
            }
            
            df_who['country_name'] = df_who['iso_code'].map(iso_mapping)
            df_who_clean = df_who.dropna(subset=['country_name'])
            
            if not df_who_clean.empty:
                df_final = df_final.merge(
                    df_who_clean[['country_name'] + [col for col in df_who_clean.columns if col not in ['iso_code', 'year', 'country_name']]],
                    on='country_name',
                    how='left'
                )
        
        # Calculate gravity
        df_final = self.calculate_gravity(df_final)
        
        # Add metadata
        df_final['data_collection_date'] = datetime.now()
        df_final['data_source'] = 'Real-World-Multi-API'
        
        # Clean up
        df_final = df_final.dropna(thresh=len(df_final.columns) * 0.3)  # Keep rows with at least 30% data
        
        logger.info(f"Final real-world dataset: {len(df_final)} countries, {len(df_final.columns)} features")
        logger.info(f"Blue Zones identified: {df_final['is_blue_zone'].sum()}")
        
        return df_final


def main():
    """Main function"""
    collector = ImprovedRealDataCollector()
    
    # Collect real-world data
    df = collector.collect_comprehensive_real_data()
    
    # Save to CSV
    output_file = "real_world_blue_zones_comprehensive.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print("REAL WORLD DATA COLLECTION COMPLETE")
    print(f"{'='*80}")
    print(f"Dataset saved to: {output_file}")
    print(f"Total countries: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Blue Zones identified: {df['is_blue_zone'].sum()}")
    
    # Show available columns
    print(f"\nAvailable features:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\nSample data (first 5 countries):")
    display_cols = ['country_name', 'is_blue_zone', 'latitude', 'longitude']
    available_display_cols = [col for col in display_cols if col in df.columns]
    print(df[available_display_cols].head())
    
    print(f"\nData completeness by feature:")
    completeness = (1 - df.isnull().mean()) * 100
    for col in df.columns:
        print(f"{col}: {completeness[col]:.1f}%")

if __name__ == "__main__":
    main()
