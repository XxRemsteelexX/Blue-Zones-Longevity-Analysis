#!/usr/bin/env python3
"""
fetch real world data from public apis and datasets
no synthetic data - only real country and city data
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime
import sys

# configuration
WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2"
REST_COUNTRIES_URL = "https://restcountries.com/v3.1/all?fields=name,cca3,latlng,population,area,region,subregion"
GEONAMES_USERNAME = "demo"  # need to register for free account

def fetch_world_bank_indicator(indicator_code, countries="all"):
    """
    fetch real data from world bank api
    """
    url = f"{WORLD_BANK_BASE_URL}/country/{countries}/indicator/{indicator_code}"
    params = {
        "format": "json",
        "per_page": 500,
        "date": "2015:2023",
        "source": 2
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if len(data) > 1 and data[1]:
            return pd.DataFrame(data[1])
        return pd.DataFrame()
    
    except Exception as e:
        print(f"error fetching {indicator_code}: {e}")
        return pd.DataFrame()

def fetch_country_coordinates():
    """
    fetch real country coordinates and basic info
    """
    try:
        response = requests.get(REST_COUNTRIES_URL)
        response.raise_for_status()
        countries = response.json()
        
        country_data = []
        for country in countries:
            if 'latlng' in country and len(country['latlng']) == 2:
                country_data.append({
                    'country_name': country.get('name', {}).get('common', ''),
                    'country_code': country.get('cca3', ''),
                    'latitude': country['latlng'][0],
                    'longitude': country['latlng'][1],
                    'population': country.get('population', 0),
                    'area': country.get('area', 0),
                    'region': country.get('region', ''),
                    'subregion': country.get('subregion', '')
                })
        
        return pd.DataFrame(country_data)
    
    except Exception as e:
        print(f"error fetching country data: {e}")
        return pd.DataFrame()

def fetch_life_expectancy_data():
    """
    fetch real life expectancy data from world bank
    """
    print("fetching life expectancy data...")
    
    # life expectancy at birth, total (years)
    life_exp_df = fetch_world_bank_indicator("SP.DYN.LE00.IN")
    
    if not life_exp_df.empty:
        # clean and reshape data
        life_exp_df = life_exp_df[['country', 'date', 'value']].copy()
        life_exp_df.columns = ['country_name', 'year', 'life_expectancy']
        life_exp_df['year'] = pd.to_numeric(life_exp_df['year'], errors='coerce')
        life_exp_df['life_expectancy'] = pd.to_numeric(life_exp_df['life_expectancy'], errors='coerce')
        life_exp_df = life_exp_df.dropna()
        
        return life_exp_df
    
    return pd.DataFrame()

def fetch_economic_data():
    """
    fetch real gdp per capita data
    """
    print("fetching gdp per capita data...")
    
    # gdp per capita (current us$)
    gdp_df = fetch_world_bank_indicator("NY.GDP.PCAP.CD")
    
    if not gdp_df.empty:
        gdp_df = gdp_df[['country', 'date', 'value']].copy()
        gdp_df.columns = ['country_name', 'year', 'gdp_per_capita']
        gdp_df['year'] = pd.to_numeric(gdp_df['year'], errors='coerce')
        gdp_df['gdp_per_capita'] = pd.to_numeric(gdp_df['gdp_per_capita'], errors='coerce')
        gdp_df = gdp_df.dropna()
        
        return gdp_df
    
    return pd.DataFrame()

def fetch_health_indicators():
    """
    fetch real health data from world bank
    """
    print("fetching health indicators...")
    
    indicators = {
        'mortality_rate': 'SP.DYN.CDRT.IN',  # death rate, crude
        'infant_mortality': 'SP.DYN.IMRT.IN',  # infant mortality rate
        'health_expenditure': 'SH.XPD.CHEX.PC.CD',  # health expenditure per capita
        'physicians_per_1000': 'SH.MED.PHYS.ZS',  # physicians per 1000 people
        'hospital_beds_per_1000': 'SH.MED.BEDS.ZS'  # hospital beds per 1000
    }
    
    health_data = {}
    for name, code in indicators.items():
        df = fetch_world_bank_indicator(code)
        if not df.empty:
            df = df[['country', 'date', 'value']].copy()
            df.columns = ['country_name', 'year', name]
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df[name] = pd.to_numeric(df[name], errors='coerce')
            health_data[name] = df.dropna()
    
    return health_data

def fetch_environmental_data():
    """
    fetch real environmental indicators
    """
    print("fetching environmental data...")
    
    indicators = {
        'co2_emissions': 'EN.ATM.CO2E.PC',  # co2 emissions per capita
        'forest_area': 'AG.LND.FRST.ZS',  # forest area percentage
        'pm25_exposure': 'EN.ATM.PM25.MC.M3',  # pm2.5 air pollution
        'renewable_energy': 'EG.FEC.RNEW.ZS',  # renewable energy consumption
        'urban_population': 'SP.URB.TOTL.IN.ZS'  # urban population percentage
    }
    
    env_data = {}
    for name, code in indicators.items():
        df = fetch_world_bank_indicator(code)
        if not df.empty:
            df = df[['country', 'date', 'value']].copy()
            df.columns = ['country_name', 'year', name]
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df[name] = pd.to_numeric(df[name], errors='coerce')
            env_data[name] = df.dropna()
            time.sleep(0.5)  # rate limiting
    
    return env_data

def calculate_gravity_fields(latitude):
    """
    calculate real gravity based on latitude
    uses the international gravity formula
    """
    # convert to radians
    lat_rad = np.radians(latitude)
    
    # international gravity formula
    g_equator = 9.780318
    g_pole = 9.832177
    
    # gravity at latitude
    gravity = g_equator * (1 + 0.0053024 * np.sin(lat_rad)**2 - 
                           0.0000058 * np.sin(2 * lat_rad)**2)
    
    # standard gravity
    g_standard = 9.80665
    
    return {
        'effective_gravity': gravity,
        'gravity_deviation': gravity - g_standard,
        'gravity_deviation_pct': (gravity - g_standard) / g_standard * 100,
        'equatorial_distance': abs(latitude)
    }

def identify_blue_zones(df):
    """
    mark real blue zone countries/regions
    """
    # real blue zone locations
    blue_zone_countries = {
        'JPN': True,  # japan (okinawa)
        'ITA': True,  # italy (sardinia)
        'GRC': True,  # greece (ikaria)
        'CRI': True,  # costa rica (nicoya)
        'USA': True,  # usa (loma linda)
    }
    
    df['is_blue_zone'] = df['country_code'].map(blue_zone_countries).fillna(False).astype(int)
    
    return df

def fetch_city_data(n_cities=500):
    """
    fetch real city data from geonames
    """
    print(f"fetching top {n_cities} cities data...")
    
    # use geonames api for city data
    url = "http://api.geonames.org/searchJSON"
    
    cities = []
    
    # fetch major cities
    params = {
        'username': GEONAMES_USERNAME,
        'maxRows': min(n_cities, 1000),
        'cities': 'cities15000',  # cities with population > 15000
        'orderby': 'population',
        'featureClass': 'P'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'geonames' in data:
            for city in data['geonames']:
                city_data = {
                    'city_name': city.get('name', ''),
                    'country_code': city.get('countryCode', ''),
                    'latitude': float(city.get('lat', 0)),
                    'longitude': float(city.get('lng', 0)),
                    'population': int(city.get('population', 0)),
                    'elevation': int(city.get('elevation', 0)) if city.get('elevation') else 0
                }
                
                # add gravity calculations
                gravity_data = calculate_gravity_fields(city_data['latitude'])
                city_data.update(gravity_data)
                
                cities.append(city_data)
        
        return pd.DataFrame(cities)
    
    except Exception as e:
        print(f"error fetching city data: {e}")
        return pd.DataFrame()

def merge_all_data():
    """
    combine all real data sources
    """
    print("\nmerging all data sources...")
    
    # fetch all data
    countries_df = fetch_country_coordinates()
    life_exp_df = fetch_life_expectancy_data()
    gdp_df = fetch_economic_data()
    health_data = fetch_health_indicators()
    env_data = fetch_environmental_data()
    
    # start with countries and coordinates
    if countries_df.empty:
        print("error: no country data available")
        return pd.DataFrame()
    
    # add gravity calculations
    gravity_cols = countries_df['latitude'].apply(lambda lat: pd.Series(calculate_gravity_fields(lat)))
    countries_df = pd.concat([countries_df, gravity_cols], axis=1)
    
    # merge life expectancy (most recent year)
    if not life_exp_df.empty:
        recent_life_exp = life_exp_df.groupby('country_name')['life_expectancy'].last().reset_index()
        countries_df = countries_df.merge(recent_life_exp, on='country_name', how='left')
    
    # merge gdp
    if not gdp_df.empty:
        recent_gdp = gdp_df.groupby('country_name')['gdp_per_capita'].last().reset_index()
        countries_df = countries_df.merge(recent_gdp, on='country_name', how='left')
    
    # merge health indicators
    for name, df in health_data.items():
        if not df.empty:
            recent = df.groupby('country_name')[name].last().reset_index()
            countries_df = countries_df.merge(recent, on='country_name', how='left')
    
    # merge environmental indicators
    for name, df in env_data.items():
        if not df.empty:
            recent = df.groupby('country_name')[name].last().reset_index()
            countries_df = countries_df.merge(recent, on='country_name', how='left')
    
    # identify blue zones
    countries_df = identify_blue_zones(countries_df)
    
    # add year column
    countries_df['year'] = datetime.now().year
    
    return countries_df

def main():
    print("fetching real world data for blue zones analysis")
    print("=" * 50)
    
    # fetch and merge all data
    countries_df = merge_all_data()
    
    if countries_df.empty:
        print("error: no data fetched")
        return
    
    print(f"\nfetched data for {len(countries_df)} countries")
    print(f"blue zone countries: {countries_df['is_blue_zone'].sum()}")
    
    # fetch city data for more granular analysis
    cities_df = fetch_city_data(n_cities=500)
    
    if not cities_df.empty:
        print(f"fetched data for {len(cities_df)} cities")
        
        # merge country data with cities
        city_country_df = cities_df.merge(
            countries_df[['country_code', 'life_expectancy', 'gdp_per_capita', 'is_blue_zone']],
            on='country_code',
            how='left'
        )
        
        # save city-level data
        city_output = '../outputs/real_cities_data.csv'
        city_country_df.to_csv(city_output, index=False)
        print(f"saved city data to {city_output}")
    
    # save country-level data
    country_output = '../outputs/real_countries_data.csv'
    countries_df.to_csv(country_output, index=False)
    print(f"saved country data to {country_output}")
    
    # create panel data with historical values
    print("\nfetching historical panel data...")
    
    years = range(2015, 2024)
    panel_data = []
    
    for year in years:
        print(f"fetching year {year}...")
        
        # fetch life expectancy for this year
        life_exp_year = fetch_world_bank_indicator("SP.DYN.LE00.IN")
        if not life_exp_year.empty:
            year_data = life_exp_year[life_exp_year['date'] == str(year)].copy()
            if not year_data.empty:
                year_data['year'] = year
                panel_data.append(year_data)
        
        time.sleep(1)  # rate limiting
    
    if panel_data:
        panel_df = pd.concat(panel_data, ignore_index=True)
        panel_output = '../outputs/real_panel_data.csv'
        panel_df.to_csv(panel_output, index=False)
        print(f"saved panel data to {panel_output}")
    
    # print summary statistics
    print("\nsummary of real data:")
    print("-" * 40)
    
    if 'life_expectancy' in countries_df.columns:
        le_stats = countries_df['life_expectancy'].describe()
        print(f"life expectancy: {le_stats['mean']:.1f} ± {le_stats['std']:.1f}")
        print(f"  range: {le_stats['min']:.1f} - {le_stats['max']:.1f}")
    
    if 'gdp_per_capita' in countries_df.columns:
        gdp_stats = countries_df['gdp_per_capita'].describe()
        print(f"gdp per capita: ${gdp_stats['mean']:.0f} ± ${gdp_stats['std']:.0f}")
    
    if 'effective_gravity' in countries_df.columns:
        grav_stats = countries_df['effective_gravity'].describe()
        print(f"gravity range: {grav_stats['min']:.4f} - {grav_stats['max']:.4f} m/s²")
    
    print("\ndata fetching complete!")
    
    return countries_df

if __name__ == '__main__':
    df = main()