#!/usr/bin/env python3
"""
GPU-Accelerated Feature Engineering using CUDA and Numba
"""
import sys
sys.path.append('../src')

import os
import numpy as np
import pandas as pd
import cupy as cp
from numba import cuda
from utils import setup_logging, save_intermediate_data, load_config
import time

# Force CUDA initialization
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@cuda.jit
def calculate_gravity_gpu(latitudes, elevations, results):
    """Calculate effective gravity on GPU"""
    idx = cuda.grid(1)
    if idx < latitudes.size:
        lat_rad = latitudes[idx] * 3.14159265359 / 180.0
        # Standard gravity formula
        g0 = 9.80665
        # Latitude correction (simplified)
        lat_correction = -0.0053024 * (cuda.cmath.sin(lat_rad) ** 2) + 0.0000058 * (cuda.cmath.sin(2 * lat_rad) ** 2)
        # Elevation correction (simplified)
        elevation_correction = -3.086e-6 * elevations[idx]
        results[idx] = g0 + lat_correction + elevation_correction

def create_gpu_features():
    """Create features using GPU acceleration"""
    logger = setup_logging()
    
    try:
        # Initialize CUDA
        cuda.select_device(0)
        logger.info(f"Using GPU: {cuda.get_current_device().name.decode()}")
    except Exception as e:
        logger.error(f"GPU initialization failed: {e}")
        logger.info("Falling back to CuPy for vectorized operations")
    
    logger.info("Starting GPU-accelerated feature engineering")
    
    # Create sample data for Blue Zone regions
    blue_zones = [
        {'name': 'Nicoya', 'lat': 10.2, 'lon': -85.4, 'elev': 200},
        {'name': 'Okinawa', 'lat': 26.3, 'lon': 127.9, 'elev': 50},
        {'name': 'Sardinia', 'lat': 40.1, 'lon': 9.4, 'elev': 300},
        {'name': 'Ikaria', 'lat': 37.6, 'lon': 26.2, 'elev': 400},
        {'name': 'Loma Linda', 'lat': 34.0, 'lon': -117.3, 'elev': 350}
    ]
    
    # Generate high-resolution grids around each Blue Zone
    all_features = []
    
    for bz in blue_zones:
        logger.info(f"Processing {bz['name']} with GPU acceleration")
        
        # Create 0.1-degree grid around Blue Zone (high resolution)
        lat_range = np.arange(bz['lat'] - 2, bz['lat'] + 2, 0.1, dtype=np.float32)
        lon_range = np.arange(bz['lon'] - 2, bz['lon'] + 2, 0.1, dtype=np.float32)
        
        lat_grid, lon_grid = np.meshgrid(lat_range, lon_range, indexing='ij')
        
        # Flatten for processing
        lats_flat = lat_grid.flatten()
        lons_flat = lon_grid.flatten()
        elevs_flat = np.full_like(lats_flat, bz['elev'])  # Simplified elevation
        
        logger.info(f"Processing {len(lats_flat):,} points for {bz['name']}")
        
        try:
            # GPU acceleration with CUDA
            lats_gpu = cuda.to_device(lats_flat)
            elevs_gpu = cuda.to_device(elevs_flat)
            results_gpu = cuda.device_array(len(lats_flat), dtype=np.float32)
            
            # Configure CUDA grid
            threads_per_block = 256
            blocks_per_grid = (len(lats_flat) + threads_per_block - 1) // threads_per_block
            
            # Run GPU kernel
            calculate_gravity_gpu[blocks_per_grid, threads_per_block](lats_gpu, elevs_gpu, results_gpu)
            
            # Copy back to host
            gravity_values = results_gpu.copy_to_host()
            logger.info(f"GPU processing completed for {bz['name']}")
            
        except Exception as e:
            logger.warning(f"GPU processing failed for {bz['name']}: {e}")
            logger.info("Using CuPy for vectorized computation")
            
            # Fallback to CuPy
            try:
                import cupy as cp
                lats_cp = cp.asarray(lats_flat)
                elevs_cp = cp.asarray(elevs_flat)
                
                # Vectorized gravity calculation on GPU
                lat_rad = lats_cp * cp.pi / 180.0
                g0 = 9.80665
                lat_correction = -0.0053024 * (cp.sin(lat_rad) ** 2) + 0.0000058 * (cp.sin(2 * lat_rad) ** 2)
                elevation_correction = -3.086e-6 * elevs_cp
                gravity_values = cp.asnumpy(g0 + lat_correction + elevation_correction)
                
            except Exception as e2:
                logger.warning(f"CuPy also failed: {e2}, using NumPy")
                # Final fallback to NumPy
                lat_rad = lats_flat * np.pi / 180.0
                g0 = 9.80665
                lat_correction = -0.0053024 * (np.sin(lat_rad) ** 2) + 0.0000058 * (np.sin(2 * lat_rad) ** 2)
                elevation_correction = -3.086e-6 * elevs_flat
                gravity_values = g0 + lat_correction + elevation_correction
        
        # Create feature DataFrame
        features = pd.DataFrame({
            'latitude': lats_flat,
            'longitude': lons_flat,
            'elevation': elevs_flat,
            'effective_gravity': gravity_values,
            'blue_zone_name': bz['name'],
            'is_blue_zone': 1,
            'distance_to_center': np.sqrt((lats_flat - bz['lat'])**2 + (lons_flat - bz['lon'])**2)
        })
        
        # Add derived features
        features['gravity_deviation'] = features['effective_gravity'] - 9.80665
        features['gravity_deviation_pct'] = (features['gravity_deviation'] / 9.80665) * 100
        features['equatorial_distance'] = np.abs(features['latitude'])
        
        all_features.append(features)
    
    # Combine all features
    final_features = pd.concat(all_features, ignore_index=True)
    final_features['grid_id'] = range(len(final_features))
    
    logger.info(f"Generated {len(final_features):,} feature points across all Blue Zones")
    
    return final_features

def main():
    """Main execution with GPU acceleration"""
    try:
        logger = setup_logging()
        logger.info("Starting GPU-accelerated comprehensive feature engineering")
        
        # Generate features
        features = create_gpu_features()
        
        # Save features
        save_intermediate_data(features, "gpu_blue_zone_features", "processed")
        logger.info("GPU-accelerated features saved successfully")
        
        # Quick statistics
        logger.info(f"Feature statistics:")
        logger.info(f"  Total points: {len(features):,}")
        logger.info(f"  Gravity range: {features['effective_gravity'].min():.6f} - {features['effective_gravity'].max():.6f} m/s²")
        logger.info(f"  Blue Zones covered: {features['blue_zone_name'].nunique()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"GPU feature engineering failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())