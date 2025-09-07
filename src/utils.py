"""
Utility functions for Blue Zones Quantified project
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('blue_zones.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent

def create_global_grid(resolution_km: int = 5, 
                      bounds: Dict[str, float] = None) -> gpd.GeoDataFrame:
    """
    Create global grid at specified resolution
    
    Args:
        resolution_km: Grid resolution in kilometers
        bounds: Dict with min_lat, max_lat, min_lon, max_lon
        
    Returns:
        GeoDataFrame with grid cells and geo_ids
    """
    if bounds is None:
        bounds = {
            'min_lat': -60, 'max_lat': 80,
            'min_lon': -180, 'max_lon': 180
        }
    
    # Convert km to degrees (approximate)
    degree_res = resolution_km / 111.32  # km per degree at equator
    
    # Create coordinate arrays
    lats = np.arange(bounds['min_lat'], bounds['max_lat'], degree_res)
    lons = np.arange(bounds['min_lon'], bounds['max_lon'], degree_res)
    
    # Create grid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Create GeoDataFrame
    from shapely.geometry import Point
    
    grid_cells = []
    geo_ids = []
    
    for i, (lat, lon) in enumerate(zip(lat_grid.flatten(), lon_grid.flatten())):
        geo_id = f"grid_{resolution_km}km_{i:08d}"
        geo_ids.append(geo_id)
        grid_cells.append(Point(lon, lat))
    
    gdf = gpd.GeoDataFrame({
        'geo_id': geo_ids,
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten(),
        'geometry': grid_cells
    }, crs='EPSG:4326')
    
    return gdf

def calculate_effective_gravity(lat: float, elevation_m: float) -> float:
    """
    Calculate effective gravity based on latitude and elevation
    
    Args:
        lat: Latitude in degrees
        elevation_m: Elevation in meters
        
    Returns:
        Effective gravity in m/s²
    """
    # Standard gravity at sea level
    g0 = 9.80665
    
    # Latitude correction (centrifugal force)
    lat_rad = np.radians(lat)
    lat_correction = -0.5 * (1.293e-3) * np.sin(2 * lat_rad)
    
    # Elevation correction (free air)
    elev_correction = -3.086e-6 * elevation_m
    
    return g0 + lat_correction + elev_correction

def haversine_distance(lat1: float, lon1: float, 
                      lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def get_blue_zone_masks(grid_df: gpd.GeoDataFrame, 
                       config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create Blue Zone masks for grid cells
    
    Args:
        grid_df: Grid GeoDataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with Blue Zone labels
    """
    masks = pd.DataFrame(index=grid_df.index)
    
    for zone_name, zone_info in config['blue_zones']['canonical'].items():
        lat_min, lat_max = zone_info['lat_range']
        lon_min, lon_max = zone_info['lon_range']
        
        mask = (
            (grid_df['lat'] >= lat_min) & 
            (grid_df['lat'] <= lat_max) &
            (grid_df['lon'] >= lon_min) & 
            (grid_df['lon'] <= lon_max)
        )
        
        masks[f'is_{zone_name}'] = mask
    
    masks['is_blue_zone'] = masks.any(axis=1)
    
    return masks

def validate_data_quality(df: pd.DataFrame, 
                         required_cols: list,
                         logger: logging.Logger = None) -> bool:
    """
    Validate data quality
    
    Args:
        df: DataFrame to validate
        required_cols: Required columns
        logger: Logger instance
        
    Returns:
        True if validation passes
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check required columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for excessive missing values
    missing_pct = df.isnull().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 50]
    
    if not high_missing.empty:
        logger.warning(f"Columns with >50% missing: {high_missing.to_dict()}")
    
    # Check for duplicate geo_ids
    if 'geo_id' in df.columns:
        duplicates = df['geo_id'].duplicated().sum()
        if duplicates > 0:
            logger.error(f"Found {duplicates} duplicate geo_ids")
            return False
    
    logger.info("Data quality validation passed")
    return True

def save_intermediate_data(df: pd.DataFrame, 
                          filename: str, 
                          subdir: str = "processed") -> None:
    """Save intermediate data with compression"""
    output_dir = get_project_root() / "data" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / f"{filename}.parquet"
    df.to_parquet(filepath, compression='snappy')
    
def load_intermediate_data(filename: str, 
                          subdir: str = "processed") -> pd.DataFrame:
    """Load intermediate data"""
    filepath = get_project_root() / "data" / subdir / f"{filename}.parquet"
    return pd.read_parquet(filepath)