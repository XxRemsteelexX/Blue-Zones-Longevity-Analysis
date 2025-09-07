#!/usr/bin/env python3
"""
GPU-Accelerated Global Grid Builder for Blue Zones Analysis
Uses RAPIDS cuDF for massive speed improvements on RTX 5090
"""
import sys
sys.path.append('../src')

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import geopandas as gpd
from utils import setup_logging, save_intermediate_data, load_config
import time

def create_gpu_grid(resolution_km: float = 5.0) -> cudf.DataFrame:
    """Create global grid using GPU acceleration"""
    logger = setup_logging()
    
    # Earth parameters
    earth_circumference = 40075.017  # km at equator
    
    # Calculate grid steps
    lat_step = resolution_km / 111.32  # degrees latitude per km
    
    logger.info(f"Creating {resolution_km}km GPU grid")
    start_time = time.time()
    
    # Create coordinate arrays on GPU
    lats = cp.arange(-90, 90, lat_step, dtype=cp.float32)
    lons = cp.arange(-180, 180, lat_step, dtype=cp.float32)
    
    # Create meshgrid on GPU
    lat_grid, lon_grid = cp.meshgrid(lats, lons, indexing='ij')
    
    # Flatten and create cuDF DataFrame
    grid = cudf.DataFrame({
        'latitude': lat_grid.flatten(),
        'longitude': lon_grid.flatten()
    })
    
    # Add grid ID
    grid['grid_id'] = cudf.Series(range(len(grid)), dtype='int32')
    
    logger.info(f"Created GPU grid with {len(grid):,} cells in {time.time() - start_time:.2f}s")
    return grid

def add_blue_zone_labels_gpu(grid: cudf.DataFrame) -> cudf.DataFrame:
    """Add Blue Zone labels using GPU operations"""
    logger = setup_logging()
    
    # Blue Zone coordinates (lat, lon, radius_km)
    blue_zones = [
        (10.2, -85.4, 50, 'Nicoya'),      # Costa Rica
        (26.3, 127.9, 50, 'Okinawa'),     # Japan  
        (40.1, 9.4, 50, 'Sardinia'),     # Italy
        (37.6, 26.2, 50, 'Ikaria'),      # Greece
        (34.0, -117.3, 50, 'Loma_Linda') # California
    ]
    
    logger.info("Adding Blue Zone labels with GPU acceleration")
    
    # Initialize is_blue_zone column
    grid['is_blue_zone'] = cudf.Series([0] * len(grid), dtype='int8')
    grid['blue_zone_name'] = cudf.Series([''] * len(grid), dtype='str')
    
    for bz_lat, bz_lon, radius_km, name in blue_zones:
        # Calculate distance using GPU vectorized operations
        lat_diff = grid['latitude'] - bz_lat
        lon_diff = grid['longitude'] - bz_lon
        
        # Haversine approximation for small distances
        distance_km = cp.sqrt(
            (lat_diff * 111.32) ** 2 + 
            (lon_diff * 111.32 * cp.cos(cp.radians(grid['latitude']))) ** 2
        )
        
        # Mark cells within radius
        mask = distance_km <= radius_km
        grid.loc[mask, 'is_blue_zone'] = 1
        grid.loc[mask, 'blue_zone_name'] = name
    
    blue_zone_count = int(grid['is_blue_zone'].sum())
    logger.info(f"Blue Zone cells: {blue_zone_count:,}")
    
    return grid

def main():
    """Main execution with GPU acceleration"""
    try:
        logger = setup_logging()
        logger.info("Starting GPU-accelerated grid creation")
        
        # Create 5km grid on GPU
        grid = create_gpu_grid(5.0)
        
        # Add Blue Zone labels
        grid_with_labels = add_blue_zone_labels_gpu(grid)
        
        # Convert to pandas for compatibility with existing pipeline
        logger.info("Converting to pandas for compatibility")
        grid_pandas = grid_with_labels.to_pandas()
        
        # Save grid
        save_intermediate_data(grid_pandas, "global_grid_5km_gpu", "processed")
        logger.info("GPU grid saved successfully")
        
        # Quick sensitivity test with 1km grid for smaller region
        logger.info("Creating 1km sensitivity grid for Blue Zones only")
        sensitivity_grid = create_gpu_grid(1.0)
        
        # Filter to Blue Zone regions only (±2 degrees around each)
        bz_regions = [
            (10.2, -85.4), (26.3, 127.9), (40.1, 9.4), 
            (37.6, 26.2), (34.0, -117.3)
        ]
        
        region_mask = cudf.Series([False] * len(sensitivity_grid))
        for lat, lon in bz_regions:
            mask = (
                (sensitivity_grid['latitude'].between(lat-2, lat+2)) & 
                (sensitivity_grid['longitude'].between(lon-2, lon+2))
            )
            region_mask = region_mask | mask
        
        sensitivity_filtered = sensitivity_grid[region_mask]
        sensitivity_labeled = add_blue_zone_labels_gpu(sensitivity_filtered)
        
        # Save sensitivity grid
        sensitivity_pandas = sensitivity_labeled.to_pandas()
        save_intermediate_data(sensitivity_pandas, "sensitivity_grid_1km_gpu", "processed")
        
        logger.info("GPU-accelerated grid creation complete")
        return 0
        
    except Exception as e:
        logger.error(f"GPU grid creation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())