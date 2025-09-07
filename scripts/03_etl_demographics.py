#!/usr/bin/env python3
"""
Extract, transform, and load demographic and socioeconomic data
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, load_config, load_intermediate_data
from etl.demographics_etl import DemographicsETL, SocioeconomicETL, NightLightsETL


def main():
    """Run demographics ETL pipeline"""
    
    # Setup
    logger = setup_logging("INFO")
    config = load_config("config/config.yaml")
    
    logger.info("Starting demographics ETL pipeline")
    
    # Load grid
    try:
        grid_df = load_intermediate_data("global_grid_5km", "processed")
        logger.info(f"Loaded grid with {len(grid_df)} cells")
    except Exception as e:
        logger.error(f"Could not load grid: {e}")
        return 1
    
    # Run population ETL
    logger.info("Processing population data")
    pop_etl = DemographicsETL(config, grid_df, logger)
    
    try:
        pop_etl.run("population_features")
        logger.info("Population ETL completed")
    except Exception as e:
        logger.error(f"Population ETL failed: {e}")
        
    # Run socioeconomic ETL
    logger.info("Processing socioeconomic data")
    socio_etl = SocioeconomicETL(config, grid_df, logger)
    
    try:
        socio_etl.run("socioeconomic_features")
        logger.info("Socioeconomic ETL completed")
    except Exception as e:
        logger.error(f"Socioeconomic ETL failed: {e}")
        
    # Run nightlights ETL
    logger.info("Processing nighttime lights data")
    ntl_etl = NightLightsETL(config, grid_df, logger)
    
    try:
        ntl_etl.run("nightlights_features")
        logger.info("Nightlights ETL completed")
    except Exception as e:
        logger.error(f"Nightlights ETL failed: {e}")
    
    logger.info("Demographics ETL pipeline completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())