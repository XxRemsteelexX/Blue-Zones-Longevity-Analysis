#!/usr/bin/env python3
"""
Extract, transform, and load life expectancy data
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, load_config, load_intermediate_data
from etl.life_expectancy_etl import LifeExpectancyETL


def main():
    """Run life expectancy data ETL"""
    
    # Setup
    logger = setup_logging("INFO")
    config = load_config("config/config.yaml")
    
    logger.info("Starting life expectancy data ETL")
    
    # Load grid
    try:
        grid_df = load_intermediate_data("global_grid_5km", "processed")
        logger.info(f"Loaded grid with {len(grid_df)} cells")
    except Exception as e:
        logger.error(f"Could not load grid: {e}")
        return 1
    
    # Initialize ETL
    le_etl = LifeExpectancyETL(config, grid_df, logger)
    
    # Run ETL
    try:
        le_etl.run("life_expectancy")
        logger.info("Life expectancy ETL completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Life expectancy ETL failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())