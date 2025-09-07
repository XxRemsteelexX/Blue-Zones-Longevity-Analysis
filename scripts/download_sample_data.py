#!/usr/bin/env python3
"""
Download REAL data for Blue Zones Quantified project - No Synthetic Data
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import requests
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, load_config
from data_acquisition.world_bank_api import WorldBankAPI
from data_acquisition.climate_data_acquirer import OpenWeatherAPI
from data_acquisition.geospatial_data_acquirer import OSMDataAcquirer


def main():
    """Download REAL data only - no synthetic fallbacks"""
    
    # Setup
    logger = setup_logging("INFO")
    config = load_config("config/config.yaml")
    
    logger.info("🌍 Starting REAL data download for Blue Zones project (No Synthetic Data)")
    
    # Create output directories
    raw_data_dir = Path("data/raw")
    for subdir in ["life_expectancy", "climate", "population", "socioeconomic", "amenities", "elevation"]:
        (raw_data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_sources = 3
    
    # 1. Download World Bank data (REAL ONLY)
    logger.info("📊 Downloading World Bank data...")
    if download_world_bank_data(config, logger):
        success_count += 1
    
    # 2. Download real climate data
    logger.info("🌡️ Downloading real climate data...")
    if download_real_climate_data(config, logger):
        success_count += 1
    
    # 3. Download amenities data for Blue Zone regions (REAL ONLY)
    logger.info("🏥 Downloading real amenities data...")
    if download_real_amenities_data(config, logger):
        success_count += 1
    
    logger.info(f"📈 Data acquisition summary: {success_count}/{total_sources} sources successful")
    
    if success_count == 0:
        logger.error("❌ No real data sources were available. Check your internet connection and API access.")
        return 1
    elif success_count < total_sources:
        logger.warning(f"⚠️ Only {success_count}/{total_sources} data sources succeeded. Analysis may be limited.")
    else:
        logger.info("✅ All real data sources downloaded successfully!")
    
    logger.info("🚀 Ready to run analysis with real data only!")
    return 0


def download_world_bank_data(config, logger):
    """Download World Bank data - REAL ONLY"""
    try:
        wb_api = WorldBankAPI(config, logger)
        
        # Key indicators for Blue Zones analysis
        indicators = [
            'life_expectancy', 'gdp_per_capita', 'education_years', 
            'pm25', 'urban_population', 'health_expenditure'
        ]
        
        # Focus on countries with known Blue Zones and some comparison countries
        focus_countries = [
            'ITA', 'JPN', 'CRI', 'GRC', 'USA',  # Blue Zone countries
            'CHE', 'MCO', 'SGP', 'AUS', 'DEU', 'GBR'  # High longevity comparison
        ]
        
        # Download data
        wb_data = wb_api.fetch_data(
            indicators=indicators,
            countries=focus_countries,
            start_year=2000,
            end_year=2023
        )
        
        if wb_data is None or wb_data.empty:
            logger.error("❌ World Bank API returned no data")
            return False
        
        # Save to CSV
        output_file = Path("data/raw/socioeconomic/world_bank_real.csv")
        wb_data.to_csv(output_file, index=False)
        
        logger.info(f"✅ World Bank REAL data saved: {len(wb_data)} records")
        
        # Also get country metadata
        metadata = wb_api.get_country_metadata()
        if metadata is not None and not metadata.empty:
            metadata.to_csv("data/raw/socioeconomic/country_metadata.csv", index=False)
            logger.info(f"✅ Country metadata saved: {len(metadata)} countries")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download World Bank data: {e}")
        logger.error("❌ No synthetic fallback - real data only mode")
        return False


def download_real_climate_data(config, logger):
    """Download real climate data using OpenWeatherAPI or other real sources"""
    try:
        # Blue Zone coordinates
        blue_zone_locations = [
            {'name': 'Sardinia', 'lat': 40.1, 'lon': 9.4, 'country': 'Italy'},
            {'name': 'Okinawa', 'lat': 26.3, 'lon': 127.9, 'country': 'Japan'},
            {'name': 'Nicoya', 'lat': 10.2, 'lon': -85.4, 'country': 'Costa Rica'},
            {'name': 'Ikaria', 'lat': 37.6, 'lon': 26.2, 'country': 'Greece'},
            {'name': 'Loma_Linda', 'lat': 34.0, 'lon': -117.3, 'country': 'United States'}
        ]
        
        # Try to get real climate data from OpenWeather API
        try:
            climate_api = OpenWeatherAPI(config, logger)
            all_climate_data = []
            
            for location in blue_zone_locations:
                try:
                    climate_data = climate_api.fetch_historical_weather(
                        lat=location['lat'],
                        lon=location['lon'],
                        start_year=2020,  # Recent years for available data
                        end_year=2023
                    )
                    
                    if climate_data is not None and not climate_data.empty:
                        climate_data['location'] = location['name']
                        climate_data['country'] = location['country']
                        all_climate_data.append(climate_data)
                        logger.info(f"  ✅ {location['name']}: {len(climate_data)} climate records")
                    else:
                        logger.warning(f"  ⚠️ No climate data for {location['name']}")
                        
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to fetch climate data for {location['name']}: {e}")
            
            if all_climate_data:
                combined_climate = pd.concat(all_climate_data, ignore_index=True)
                combined_climate.to_csv("data/raw/climate/real_climate_data.csv", index=False)
                logger.info(f"✅ Real climate data saved: {len(combined_climate)} records")
                return True
            else:
                logger.error("❌ No real climate data could be retrieved")
                return False
                
        except Exception as e:
            logger.error(f"❌ Climate API failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to download real climate data: {e}")
        return False


def download_real_amenities_data(config, logger):
    """Download real amenities data using OSM - NO SYNTHETIC FALLBACK"""
    
    # Blue Zone regions with bounding boxes (expanded for more comprehensive coverage)
    regions = {
        'Sardinia_Ogliastra': [39.6, 8.8, 40.4, 9.8],
        'Okinawa_Main': [26.0, 127.6, 26.9, 128.4], 
        'Nicoya_Peninsula': [9.8, -85.8, 10.6, -85.0],
        'Ikaria_Island': [37.5, 26.0, 37.7, 26.4],
        'Loma_Linda': [34.0, -117.3, 34.1, -117.2]
    }
    
    all_amenities = []
    success_regions = 0
    
    try:
        osm_acquirer = OSMDataAcquirer(config, logger)
        
        for region_name, bbox in regions.items():
            try:
                logger.info(f"  🔍 Fetching amenities for {region_name}...")
                amenities = osm_acquirer.fetch_amenities(
                    bbox=bbox,
                    amenity_types=['hospital', 'clinic', 'pharmacy', 'market', 'school', 'place_of_worship']
                )
                
                if amenities is not None and not amenities.empty:
                    amenities['region'] = region_name
                    all_amenities.append(amenities)
                    success_regions += 1
                    logger.info(f"  ✅ {region_name}: {len(amenities)} real amenities found")
                else:
                    logger.warning(f"  ⚠️ No amenities found for {region_name}")
                    
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to fetch {region_name}: {e}")
        
        # Save combined real data only
        if all_amenities:
            combined_amenities = pd.concat(all_amenities, ignore_index=True)
            combined_amenities.to_csv("data/raw/amenities/real_amenities_data.csv", index=False)
            logger.info(f"✅ Real amenities data saved: {len(combined_amenities)} records from {success_regions} regions")
            return True
        else:
            logger.error("❌ No real amenities data could be retrieved from any region")
            logger.error("❌ No synthetic fallback available - real data only mode")
            return False
            
    except Exception as e:
        logger.error(f"❌ OSM API completely failed: {e}")
        logger.error("❌ No synthetic fallback available - real data only mode")
        return False






if __name__ == "__main__":
    sys.exit(main())