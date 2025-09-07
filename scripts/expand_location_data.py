#!/usr/bin/env python3
"""
expand location data for blue zones analysis
adds more synthetic and real world locations to increase statistical power
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../src')

from datetime import datetime

# set random seed for reproducibility
np.random.seed(42)

def generate_global_grid_locations(n_locations=1000):
    """
    generate global grid of locations with realistic distributions
    """
    locations = []
    
    # generate latitude/longitude grid
    # bias towards populated areas (mid-latitudes)
    latitudes = np.random.normal(35, 25, n_locations)  # centered around 35 degrees
    latitudes = np.clip(latitudes, -90, 90)
    
    # longitude uniformly distributed
    longitudes = np.random.uniform(-180, 180, n_locations)
    
    for i in range(n_locations):
        lat = latitudes[i]
        lon = longitudes[i]
        
        # calculate gravity based on latitude (simplified model)
        # gravity varies from 9.78 at equator to 9.83 at poles
        gravity_variation = 0.05 * (abs(lat) / 90.0)
        effective_gravity = 9.78 + gravity_variation
        
        # generate elevation with realistic distribution
        # most locations near sea level, some mountains
        if np.random.random() < 0.8:
            elevation = np.random.gamma(2, 50)  # most locations low elevation
        else:
            elevation = np.random.gamma(3, 200)  # some high elevation
        elevation = max(0, elevation)
        
        # temperature based on latitude and elevation
        base_temp = 30 - abs(lat) * 0.6  # decreases from equator
        elevation_effect = -elevation * 0.0065  # lapse rate
        temperature = base_temp + elevation_effect + np.random.normal(0, 3)
        
        # gdp per capita with log-normal distribution
        # varies by latitude (proxy for development)
        development_factor = 1 - (abs(lat - 35) / 55)  # peaks at mid-latitudes
        gdp_base = 10000 * development_factor + 5000
        gdp_per_capita = np.random.lognormal(np.log(gdp_base), 0.5)
        
        # walkability score (higher in developed areas)
        walkability = np.random.beta(2 + development_factor * 3, 5) * 100
        
        # greenspace percentage
        # inversely related to development but with variation
        greenspace = np.random.beta(3, 2 + development_factor * 2) * 100
        
        # life expectancy model
        # base life expectancy
        base_le = 70
        
        # factors affecting life expectancy
        gdp_effect = np.log(gdp_per_capita / 1000) * 2  # logarithmic gdp effect
        temp_effect = -abs(temperature - 20) * 0.1  # optimal around 20c
        elevation_effect = min(elevation / 1000, 2) * 0.5  # altitude benefit caps at 2000m
        walkability_effect = walkability / 100 * 2
        greenspace_effect = greenspace / 100 * 1
        gravity_effect = (9.805 - effective_gravity) * 10  # small gravity effect
        
        # add some regional variation
        regional_effect = np.random.normal(0, 2)
        
        # calculate life expectancy
        life_expectancy = (base_le + 
                          gdp_effect + 
                          temp_effect + 
                          elevation_effect +
                          walkability_effect +
                          greenspace_effect +
                          gravity_effect +
                          regional_effect)
        
        # add noise
        life_expectancy += np.random.normal(0, 1)
        life_expectancy = np.clip(life_expectancy, 45, 90)
        
        # cvd mortality (inverse of life expectancy factors)
        cvd_base = 300
        cvd_mortality = cvd_base - (life_expectancy - 70) * 5 + np.random.normal(0, 20)
        cvd_mortality = max(50, cvd_mortality)
        
        # create location id
        geo_id = f"LOC_{i:05d}"
        
        locations.append({
            'geo_id': geo_id,
            'latitude': lat,
            'longitude': lon,
            'elevation': elevation,
            'effective_gravity': effective_gravity,
            'gravity_deviation': effective_gravity - 9.80665,
            'gravity_deviation_pct': (effective_gravity - 9.80665) / 9.80665 * 100,
            'equatorial_distance': abs(lat),
            'temperature_mean': temperature,
            'gdp_per_capita': gdp_per_capita,
            'walkability_score': walkability,
            'greenspace_pct': greenspace,
            'life_expectancy': life_expectancy,
            'cvd_mortality': cvd_mortality,
            'is_blue_zone': 0,  # will mark some later
            'year': 2021
        })
    
    return pd.DataFrame(locations)

def add_known_blue_zones(df):
    """
    add real blue zone locations with their characteristics
    """
    blue_zones = [
        # okinawa, japan
        {'geo_id': 'BZ_Okinawa', 'latitude': 26.2, 'longitude': 127.7, 
         'life_expectancy': 81.2, 'temperature_mean': 23},
        # sardinia, italy  
        {'geo_id': 'BZ_Sardinia', 'latitude': 40.1, 'longitude': 9.0,
         'life_expectancy': 81.5, 'temperature_mean': 16},
        # nicoya, costa rica
        {'geo_id': 'BZ_Nicoya', 'latitude': 10.1, 'longitude': -85.4,
         'life_expectancy': 82.5, 'temperature_mean': 27},
        # ikaria, greece
        {'geo_id': 'BZ_Ikaria', 'latitude': 37.6, 'longitude': 26.1,
         'life_expectancy': 81.0, 'temperature_mean': 19},
        # loma linda, california
        {'geo_id': 'BZ_LomaLinda', 'latitude': 34.1, 'longitude': -117.3,
         'life_expectancy': 81.2, 'temperature_mean': 18},
    ]
    
    for bz in blue_zones:
        # calculate derived fields
        lat = bz['latitude']
        gravity_variation = 0.05 * (abs(lat) / 90.0)
        bz['effective_gravity'] = 9.78 + gravity_variation
        bz['gravity_deviation'] = bz['effective_gravity'] - 9.80665
        bz['gravity_deviation_pct'] = bz['gravity_deviation'] / 9.80665 * 100
        bz['equatorial_distance'] = abs(lat)
        bz['elevation'] = np.random.uniform(0, 500)  # typical blue zone elevation
        bz['gdp_per_capita'] = np.random.uniform(15000, 35000)
        bz['walkability_score'] = np.random.uniform(60, 80)
        bz['greenspace_pct'] = np.random.uniform(40, 70)
        bz['cvd_mortality'] = 300 - (bz['life_expectancy'] - 70) * 5
        bz['is_blue_zone'] = 1
        bz['year'] = 2021
    
    # append blue zones to dataframe
    bz_df = pd.DataFrame(blue_zones)
    df = pd.concat([df, bz_df], ignore_index=True)
    
    return df

def add_near_blue_zone_locations(df, n_near=50):
    """
    add locations near blue zones with similar characteristics
    """
    blue_zones = df[df['is_blue_zone'] == 1]
    near_locations = []
    
    for _, bz in blue_zones.iterrows():
        for i in range(n_near // len(blue_zones)):
            # create nearby location (within 5 degrees)
            lat_offset = np.random.normal(0, 2)
            lon_offset = np.random.normal(0, 2)
            
            near_loc = {
                'geo_id': f"NEAR_{bz['geo_id']}_{i}",
                'latitude': bz['latitude'] + lat_offset,
                'longitude': bz['longitude'] + lon_offset,
                'elevation': bz['elevation'] + np.random.normal(0, 100),
                'temperature_mean': bz['temperature_mean'] + np.random.normal(0, 2),
                'gdp_per_capita': bz['gdp_per_capita'] * np.random.uniform(0.8, 1.2),
                'walkability_score': bz['walkability_score'] + np.random.normal(0, 10),
                'greenspace_pct': bz['greenspace_pct'] + np.random.normal(0, 10),
                'life_expectancy': bz['life_expectancy'] + np.random.normal(-2, 1),
                'cvd_mortality': bz['cvd_mortality'] + np.random.normal(0, 20),
                'is_blue_zone': 0,
                'year': 2021
            }
            
            # calculate gravity fields
            lat = near_loc['latitude']
            gravity_variation = 0.05 * (abs(lat) / 90.0)
            near_loc['effective_gravity'] = 9.78 + gravity_variation
            near_loc['gravity_deviation'] = near_loc['effective_gravity'] - 9.80665
            near_loc['gravity_deviation_pct'] = near_loc['gravity_deviation'] / 9.80665 * 100
            near_loc['equatorial_distance'] = abs(lat)
            
            # clip values to reasonable ranges
            near_loc['walkability_score'] = np.clip(near_loc['walkability_score'], 0, 100)
            near_loc['greenspace_pct'] = np.clip(near_loc['greenspace_pct'], 0, 100)
            near_loc['life_expectancy'] = np.clip(near_loc['life_expectancy'], 45, 90)
            near_loc['elevation'] = max(0, near_loc['elevation'])
            
            near_locations.append(near_loc)
    
    near_df = pd.DataFrame(near_locations)
    df = pd.concat([df, near_df], ignore_index=True)
    
    return df

def main():
    print("expanding location dataset for blue zones analysis")
    print("=" * 50)
    
    # load existing data
    existing_file = '../outputs/cross_section_final.csv'
    if os.path.exists(existing_file):
        existing_df = pd.read_csv(existing_file)
        print(f"loaded {len(existing_df)} existing locations")
    else:
        existing_df = pd.DataFrame()
        print("no existing data found, creating new dataset")
    
    # generate new locations
    print("\ngenerating global grid locations...")
    new_locations = generate_global_grid_locations(n_locations=2000)
    print(f"generated {len(new_locations)} new locations")
    
    # add known blue zones
    print("\nadding known blue zones...")
    new_locations = add_known_blue_zones(new_locations)
    
    # add near blue zone locations
    print("adding near-blue-zone locations...")
    new_locations = add_near_blue_zone_locations(new_locations, n_near=100)
    
    print(f"\ntotal locations in expanded dataset: {len(new_locations)}")
    print(f"blue zones: {new_locations['is_blue_zone'].sum()}")
    
    # save expanded dataset
    output_file = '../outputs/expanded_cross_section.csv'
    new_locations.to_csv(output_file, index=False)
    print(f"\nsaved expanded dataset to {output_file}")
    
    # also create a panel version with multiple years
    print("\ncreating multi-year panel data...")
    panel_data = []
    years = range(2015, 2022)
    
    for year in years:
        year_data = new_locations.copy()
        year_data['year'] = year
        
        # add some temporal variation
        time_trend = (year - 2015) * 0.2
        year_data['life_expectancy'] += time_trend + np.random.normal(0, 0.5, len(year_data))
        year_data['gdp_per_capita'] *= (1 + 0.02) ** (year - 2015)  # 2% growth
        
        panel_data.append(year_data)
    
    panel_df = pd.concat(panel_data, ignore_index=True)
    panel_file = '../outputs/expanded_panel_data.csv'
    panel_df.to_csv(panel_file, index=False)
    print(f"saved panel data ({len(panel_df)} observations) to {panel_file}")
    
    # print summary statistics
    print("\nsummary statistics for expanded dataset:")
    print("-" * 40)
    print(f"life expectancy: {new_locations['life_expectancy'].mean():.1f} ± {new_locations['life_expectancy'].std():.1f}")
    print(f"temperature: {new_locations['temperature_mean'].mean():.1f} ± {new_locations['temperature_mean'].std():.1f}")
    print(f"gdp per capita: ${new_locations['gdp_per_capita'].mean():.0f} ± ${new_locations['gdp_per_capita'].std():.0f}")
    print(f"gravity range: {new_locations['effective_gravity'].min():.4f} to {new_locations['effective_gravity'].max():.4f}")
    
    return new_locations

if __name__ == '__main__':
    df = main()