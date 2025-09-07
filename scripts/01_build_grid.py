#!/usr/bin/env python3
"""
Build global grid system for Blue Zones Quantified project
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, load_config, create_global_grid, get_blue_zone_masks
from utils import save_intermediate_data, validate_data_quality


def main():
    """Build global grid system"""
    
    # Setup
    logger = setup_logging("INFO")
    config = load_config("config/config.yaml")
    
    logger.info("Starting global grid creation")
    
    # Create primary grid (5km)
    primary_resolution = config['spatial']['primary_resolution_km']
    bounds = config['spatial']['global_bounds']
    
    logger.info(f"Creating {primary_resolution}km global grid")
    grid_df = create_global_grid(
        resolution_km=primary_resolution,
        bounds=bounds
    )
    
    logger.info(f"Created grid with {len(grid_df)} cells")
    
    # Add Blue Zone masks
    logger.info("Adding Blue Zone labels")
    blue_zone_masks = get_blue_zone_masks(grid_df, config)
    
    # Combine grid with masks
    grid_with_labels = grid_df.merge(
        blue_zone_masks.reset_index(), 
        left_index=True, 
        right_index=True
    )
    
    # Validate
    required_cols = ['geo_id', 'lat', 'lon', 'geometry', 'is_blue_zone']
    if not validate_data_quality(grid_with_labels, required_cols, logger):
        logger.error("Grid validation failed")
        return 1
        
    # Save primary grid
    save_intermediate_data(grid_with_labels, "global_grid_5km", "processed")
    
    logger.info(f"Blue Zone cells: {blue_zone_masks['is_blue_zone'].sum()}")
    logger.info("Grid saved successfully")
    
    # Create sensitivity grids
    for resolution in config['spatial']['sensitivity_resolutions_km']:
        logger.info(f"Creating {resolution}km sensitivity grid")
        
        sens_grid = create_global_grid(
            resolution_km=resolution,
            bounds=bounds
        )
        
        sens_masks = get_blue_zone_masks(sens_grid, config)
        sens_grid_labeled = sens_grid.merge(
            sens_masks.reset_index(),
            left_index=True,
            right_index=True
        )
        
        save_intermediate_data(sens_grid_labeled, f"global_grid_{resolution}km", "processed")
        logger.info(f"Sensitivity grid {resolution}km saved ({len(sens_grid)} cells)")
    
    logger.info("Grid creation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())