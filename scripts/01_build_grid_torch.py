#!/usr/bin/env python3
"""
GPU-Accelerated Grid Builder using PyTorch tensors on RTX 5090
"""
import sys
sys.path.append('../src')

import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from utils import setup_logging, save_intermediate_data
import time
import os

def check_gpu():
    """Check GPU availability and initialize"""
    # Clear any CUDA context issues
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        return device
    else:
        print("GPU not available, falling back to CPU")
        return torch.device('cpu')

def create_gpu_grid_torch(resolution_km: float = 5.0) -> pd.DataFrame:
    """Create global grid using PyTorch tensors on GPU"""
    logger = setup_logging()
    device = check_gpu()
    
    # Calculate grid steps
    lat_step = resolution_km / 111.32  # degrees latitude per km
    
    logger.info(f"Creating {resolution_km}km grid on {device}")
    start_time = time.time()
    
    # Create coordinate tensors on GPU
    lats = torch.arange(-90, 90, lat_step, device=device, dtype=torch.float32)
    lons = torch.arange(-180, 180, lat_step, device=device, dtype=torch.float32)
    
    # Create meshgrid on GPU
    lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing='ij')
    
    # Flatten tensors
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    
    # Move to CPU for DataFrame creation
    grid_data = {
        'latitude': lat_flat.cpu().numpy(),
        'longitude': lon_flat.cpu().numpy()
    }
    
    grid = pd.DataFrame(grid_data)
    grid['grid_id'] = range(len(grid))
    
    logger.info(f"Created grid with {len(grid):,} cells in {time.time() - start_time:.2f}s")
    return grid

def add_blue_zone_labels_torch(grid: pd.DataFrame) -> pd.DataFrame:
    """Add Blue Zone labels using GPU tensor operations"""
    logger = setup_logging()
    device = check_gpu()
    
    # Blue Zone coordinates
    blue_zones = [
        (10.2, -85.4, 50, 'Nicoya'),
        (26.3, 127.9, 50, 'Okinawa'),
        (40.1, 9.4, 50, 'Sardinia'),
        (37.6, 26.2, 50, 'Ikaria'),
        (34.0, -117.3, 50, 'Loma_Linda')
    ]
    
    logger.info(f"Adding Blue Zone labels on {device}")
    start_time = time.time()
    
    # Convert grid to GPU tensors
    grid_lats = torch.tensor(grid['latitude'].values, device=device, dtype=torch.float32)
    grid_lons = torch.tensor(grid['longitude'].values, device=device, dtype=torch.float32)
    
    # Initialize result tensors
    is_blue_zone = torch.zeros(len(grid), device=device, dtype=torch.int8)
    blue_zone_names = [''] * len(grid)
    
    for bz_lat, bz_lon, radius_km, name in blue_zones:
        # Haversine distance calculation on GPU
        R = 6371.0  # Earth radius in km
        
        # Convert to radians
        lat1 = torch.deg2rad(grid_lats)
        lon1 = torch.deg2rad(grid_lons)
        lat2 = torch.deg2rad(torch.tensor(bz_lat, device=device))
        lon2 = torch.deg2rad(torch.tensor(bz_lon, device=device))
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (torch.sin(dlat/2)**2 + 
             torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2)
        c = 2 * torch.asin(torch.sqrt(a))
        distances = R * c
        
        # Mark cells within radius
        mask = distances <= radius_km
        is_blue_zone = torch.where(mask, torch.ones_like(is_blue_zone), is_blue_zone)
        
        # Update names (CPU operation)
        mask_cpu = mask.cpu().numpy()
        for i in range(len(blue_zone_names)):
            if mask_cpu[i]:
                blue_zone_names[i] = name
    
    # Add results to grid
    grid['is_blue_zone'] = is_blue_zone.cpu().numpy()
    grid['blue_zone_name'] = blue_zone_names
    
    blue_zone_count = int(grid['is_blue_zone'].sum())
    logger.info(f"Blue Zone cells: {blue_zone_count:,} in {time.time() - start_time:.2f}s")
    
    return grid

def main():
    """Main execution"""
    try:
        logger = setup_logging()
        logger.info("Starting PyTorch GPU grid creation")
        
        # Create 5km grid
        grid = create_gpu_grid_torch(5.0)
        
        # Add Blue Zone labels
        grid_with_labels = add_blue_zone_labels_torch(grid)
        
        # Save grid
        save_intermediate_data(grid_with_labels, "global_grid_5km_torch", "processed")
        logger.info("PyTorch GPU grid saved successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"PyTorch GPU grid creation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())