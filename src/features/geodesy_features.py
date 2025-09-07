"""
Geodesy and terrain feature engineering
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, Any, Optional
import logging
from pathlib import Path


class GeodesyFeatures:
    """Generate geodesy and terrain features"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
    def generate_basic_geodesy(self, grid_df: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Generate basic geodesy features
        
        Args:
            grid_df: Grid GeoDataFrame with lat/lon
            
        Returns:
            DataFrame with geodesy features
        """
        features = pd.DataFrame({
            'geo_id': grid_df['geo_id'],
            'latitude': grid_df['lat'],
            'longitude': grid_df['lon'],
            'abs_latitude': np.abs(grid_df['lat']),
            'distance_to_equator': np.abs(grid_df['lat']) * 111.32  # km
        })
        
        # Distance to prime meridian
        features['distance_to_prime_meridian'] = np.abs(features['longitude']) * 111.32
        
        return features
        
    def add_coastline_distance(self, features: pd.DataFrame, 
                             coastline_data: Optional[gpd.GeoDataFrame] = None) -> pd.DataFrame:
        """
        Add distance to coastline feature
        
        Args:
            features: Feature DataFrame
            coastline_data: Coastline geometries (optional)
            
        Returns:
            DataFrame with coastline distance
        """
        if coastline_data is None:
            # Use simplified ocean/land mask approach
            self.logger.info("Using simplified land/ocean classification")
            features['distance_to_coast'] = self._estimate_coast_distance(features)
        else:
            # Calculate actual distance to coastline
            features['distance_to_coast'] = self._calculate_coast_distance(features, coastline_data)
            
        return features
        
    def _estimate_coast_distance(self, features: pd.DataFrame) -> pd.Series:
        """Estimate distance to coast using simple heuristics"""
        lat = features['latitude'].values
        lon = features['longitude'].values
        
        # Simple approximation: minimum distance to major water bodies
        # This is a placeholder - would need actual coastline data for precision
        
        distances = []
        for i, (lat_val, lon_val) in enumerate(zip(lat, lon)):
            # Distance to nearest major water body (simplified)
            min_dist = float('inf')
            
            # Major water bodies (approximate centers)
            water_bodies = [
                (0, 0, 6371),      # Atlantic (center, radius)
                (0, 150, 5500),    # Pacific
                (20, 75, 3000),    # Indian Ocean
                (70, 0, 1800),     # Arctic Ocean
            ]
            
            for water_lat, water_lon, radius in water_bodies:
                dist = self._haversine_distance(lat_val, lon_val, water_lat, water_lon)
                coast_dist = max(0, dist - radius)
                min_dist = min(min_dist, coast_dist)
                
            distances.append(min_dist)
            
        return pd.Series(distances)
        
    def _calculate_coast_distance(self, features: pd.DataFrame, 
                                coastline: gpd.GeoDataFrame) -> pd.Series:
        """Calculate actual distance to coastline"""
        from shapely.geometry import Point
        
        distances = []
        for _, row in features.iterrows():
            point = Point(row['longitude'], row['latitude'])
            min_dist = coastline.distance(point).min()
            # Convert to km (assuming degrees to km conversion)
            distances.append(min_dist * 111.32)
            
        return pd.Series(distances)
        
    def add_elevation_features(self, features: pd.DataFrame,
                             elevation_file: Optional[str] = None) -> pd.DataFrame:
        """
        Add elevation and terrain features
        
        Args:
            features: Feature DataFrame
            elevation_file: Path to elevation raster
            
        Returns:
            DataFrame with elevation features
        """
        if elevation_file is None:
            self.logger.warning("No elevation data provided, using placeholder values")
            features['elevation'] = 0
            features['slope'] = 0
            features['ruggedness'] = 0
        else:
            features = self._extract_elevation_features(features, elevation_file)
            
        return features
        
    def _extract_elevation_features(self, features: pd.DataFrame, 
                                  elevation_file: str) -> pd.DataFrame:
        """Extract elevation features from raster"""
        try:
            import rasterio
            from rasterio.transform import rowcol
            
            with rasterio.open(elevation_file) as src:
                elevation_data = src.read(1)
                transform = src.transform
                
                elevations = []
                slopes = []
                ruggedness_vals = []
                
                for _, row in features.iterrows():
                    lat, lon = row['latitude'], row['longitude']
                    
                    try:
                        # Get pixel coordinates
                        col, row_idx = rowcol(transform, lon, lat)
                        
                        if (0 <= row_idx < elevation_data.shape[0] and 
                            0 <= col < elevation_data.shape[1]):
                            
                            elevation = elevation_data[row_idx, col]
                            
                            # Calculate slope and ruggedness from neighborhood
                            slope, ruggedness = self._calculate_terrain_metrics(
                                elevation_data, row_idx, col
                            )
                            
                        else:
                            elevation, slope, ruggedness = 0, 0, 0
                            
                    except Exception:
                        elevation, slope, ruggedness = 0, 0, 0
                        
                    elevations.append(elevation)
                    slopes.append(slope)
                    ruggedness_vals.append(ruggedness)
                    
                features['elevation'] = elevations
                features['slope'] = slopes
                features['ruggedness'] = ruggedness_vals
                
        except ImportError:
            self.logger.warning("rasterio not available, using zero elevation")
            features['elevation'] = 0
            features['slope'] = 0
            features['ruggedness'] = 0
        except Exception as e:
            self.logger.error(f"Error processing elevation: {e}")
            features['elevation'] = 0
            features['slope'] = 0
            features['ruggedness'] = 0
            
        return features
        
    def _calculate_terrain_metrics(self, elevation_data: np.ndarray,
                                 row: int, col: int, 
                                 window_size: int = 3) -> tuple:
        """Calculate slope and ruggedness from elevation neighborhood"""
        half_window = window_size // 2
        
        # Extract neighborhood
        min_row = max(0, row - half_window)
        max_row = min(elevation_data.shape[0], row + half_window + 1)
        min_col = max(0, col - half_window)
        max_col = min(elevation_data.shape[1], col + half_window + 1)
        
        neighborhood = elevation_data[min_row:max_row, min_col:max_col]
        
        if neighborhood.size < 4:  # Too small for calculation
            return 0, 0
            
        # Calculate slope (maximum rate of change)
        if neighborhood.shape[0] > 1 and neighborhood.shape[1] > 1:
            dy, dx = np.gradient(neighborhood)
            slope = np.sqrt(dx**2 + dy**2)
            max_slope = np.max(slope)
            
            # Ruggedness (standard deviation of elevation)
            ruggedness = np.std(neighborhood)
        else:
            max_slope = 0
            ruggedness = 0
            
        return max_slope, ruggedness
        
    def add_effective_gravity(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add effective gravity based on latitude and elevation"""
        features['effective_gravity'] = self._calculate_effective_gravity(
            features['latitude'], 
            features['elevation']
        )
        
        return features
        
    def _calculate_effective_gravity(self, lat: pd.Series, elevation: pd.Series) -> pd.Series:
        """Calculate effective gravity"""
        # Standard gravity at sea level
        g0 = 9.80665
        
        # Latitude correction (centrifugal force)
        lat_rad = np.radians(lat)
        lat_correction = -0.5 * (1.293e-3) * np.sin(2 * lat_rad)
        
        # Elevation correction (free air)
        elev_correction = -3.086e-6 * elevation
        
        return g0 + lat_correction + elev_correction
        
    def _haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c