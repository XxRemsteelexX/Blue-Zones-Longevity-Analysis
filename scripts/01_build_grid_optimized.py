#!/usr/bin/env python3
"""
Optimized Global Grid Builder for Blue Zones Analysis
Uses vectorized NumPy operations for maximum CPU performance
"""
import sys
sys.path.append('../src')

import numpy as np
import pandas as pd
import geopandas as gpd
from utils import setup_logging, save_intermediate_data, load_config
import time
from numba import jit

@jit(nopython=True)
def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation with Numba JIT compilation"""
    R = 6371.0  # Earth radius in kilometers
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def create_optimized_grid(resolution_km: float = 5.0) -> pd.DataFrame:
    """Create global grid using optimized vectorized operations"""
    logger = setup_logging()
    
    # Calculate grid steps
    lat_step = resolution_km / 111.32  # degrees latitude per km
    
    logger.info(f"Creating {resolution_km}km optimized grid")
    start_time = time.time()
    
    # Create coordinate arrays
    lats = np.arange(-90, 90, lat_step, dtype=np.float32)
    lons = np.arange(-180, 180, lat_step, dtype=np.float32)
    
    # Create meshgrid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Flatten and create DataFrame
    grid = pd.DataFrame({
        'latitude': lat_grid.flatten(),
        'longitude': lon_grid.flatten()
    })
    
    # Add grid ID
    grid['grid_id'] = range(len(grid))
    
    logger.info(f"Created grid with {len(grid):,} cells in {time.time() - start_time:.2f}s")
    return grid

def add_blue_zone_labels_optimized(grid: pd.DataFrame) -> pd.DataFrame:
    """Add Blue Zone labels using vectorized operations"""
    logger = setup_logging()
    
    # Blue Zone coordinates (lat, lon, radius_km)
    blue_zones = [
        (10.2, -85.4, 50, 'Nicoya'),      # Costa Rica
        (26.3, 127.9, 50, 'Okinawa'),     # Japan  
        (40.1, 9.4, 50, 'Sardinia'),     # Italy
        (37.6, 26.2, 50, 'Ikaria'),      # Greece
        (34.0, -117.3, 50, 'Loma_Linda') # California
    ]
    
    logger.info("Adding Blue Zone labels with vectorized operations")
    start_time = time.time()
    
    # Initialize columns
    grid['is_blue_zone'] = 0
    grid['blue_zone_name'] = ''
    
    # Convert to numpy arrays for speed
    grid_lats = grid['latitude'].values
    grid_lons = grid['longitude'].values
    
    for bz_lat, bz_lon, radius_km, name in blue_zones:
        # Calculate distances using JIT-compiled function
        distances = haversine_distance_vectorized(
            grid_lats, grid_lons, 
            np.full_like(grid_lats, bz_lat), 
            np.full_like(grid_lons, bz_lon)
        )
        
        # Mark cells within radius
        mask = distances <= radius_km
        grid.loc[mask, 'is_blue_zone'] = 1
        grid.loc[mask, 'blue_zone_name'] = name
    
    blue_zone_count = int(grid['is_blue_zone'].sum())
    logger.info(f"Blue Zone cells: {blue_zone_count:,} in {time.time() - start_time:.2f}s")
    
    return grid

def main():
    """Main execution with optimized performance"""
    try:
        logger = setup_logging()
        logger.info("Starting optimized grid creation")
        
        # Create 5km grid
        grid = create_optimized_grid(5.0)
        
        # Add Blue Zone labels
        grid_with_labels = add_blue_zone_labels_optimized(grid)
        
        # Data validation
        logger.info("Validating grid data")
        assert not grid_with_labels.isnull().any().any(), "Grid contains null values"
        assert len(grid_with_labels) > 0, "Grid is empty"
        
        # Save grid
        save_intermediate_data(grid_with_labels, "global_grid_5km_optimized", "processed")
        logger.info("Optimized grid saved successfully")
        
        # Create 1km sensitivity grid for Blue Zone regions only
        logger.info("Creating 1km sensitivity grid for Blue Zones")
        
        # Filter to regions around Blue Zones (±2 degrees)
        bz_regions = [
            (10.2, -85.4), (26.3, 127.9), (40.1, 9.4), 
            (37.6, 26.2), (34.0, -117.3)
        ]
        
        sensitivity_grids = []
        for lat, lon in bz_regions:
            # Create small regional grid
            regional_lats = np.arange(lat-2, lat+2, 1.0/111.32, dtype=np.float32)  # 1km resolution
            regional_lons = np.arange(lon-2, lon+2, 1.0/111.32, dtype=np.float32)
            
            lat_grid, lon_grid = np.meshgrid(regional_lats, regional_lons, indexing='ij')
            
            regional_grid = pd.DataFrame({
                'latitude': lat_grid.flatten(),
                'longitude': lon_grid.flatten()
            })
            
            sensitivity_grids.append(regional_grid)
        
        # Combine all regional grids
        sensitivity_grid = pd.concat(sensitivity_grids, ignore_index=True)
        sensitivity_grid['grid_id'] = range(len(sensitivity_grid))
        
        # Add Blue Zone labels to sensitivity grid
        sensitivity_labeled = add_blue_zone_labels_optimized(sensitivity_grid)
        
        # Save sensitivity grid
        save_intermediate_data(sensitivity_labeled, "sensitivity_grid_1km_optimized", "processed")
        
        logger.info("Optimized grid creation complete")
        return 0
        
    except Exception as e:
        logger.error(f"Optimized grid creation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())