"""
Geospatial data acquirer for elevation, population, and satellite data
"""
import pandas as pd
import numpy as np
import requests
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_acquirer import BaseDataAcquirer, FileDownloader
from pathlib import Path
import zipfile
import io


class ElevationDataAcquirer(FileDownloader):
    """SRTM elevation data acquirer"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, download_dir="data/raw/elevation", logger=logger)
        self.srtm_base_url = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTMGL1"
        
    def fetch_srtm_tiles(self, bounds: Dict[str, float]) -> List[Path]:
        """
        Download SRTM tiles for given bounding box
        
        Args:
            bounds: Dictionary with 'north', 'south', 'east', 'west' keys
            
        Returns:
            List of downloaded file paths
        """
        tiles_needed = self._calculate_srtm_tiles(bounds)
        downloaded_files = []
        
        for tile in tiles_needed:
            file_url = f"{self.srtm_base_url}/{tile}.zip"
            
            try:
                filepath = self.download_file(file_url, f"{tile}.zip")
                
                # Extract the zip file
                extracted_file = self._extract_srtm_zip(filepath, tile)
                if extracted_file:
                    downloaded_files.append(extracted_file)
                    
            except Exception as e:
                self.logger.warning(f"Failed to download SRTM tile {tile}: {e}")
                
        return downloaded_files
        
    def _calculate_srtm_tiles(self, bounds: Dict[str, float]) -> List[str]:
        """Calculate which SRTM tiles are needed for bounds"""
        north, south = int(np.ceil(bounds['north'])), int(np.floor(bounds['south']))
        east, west = int(np.ceil(bounds['east'])), int(np.floor(bounds['west']))
        
        tiles = []
        for lat in range(south, north):
            for lon in range(west, east):
                # SRTM tile naming convention
                lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
                lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
                tiles.append(f"{lat_str}{lon_str}")
                
        return tiles
        
    def _extract_srtm_zip(self, zip_path: Path, tile_name: str) -> Optional[Path]:
        """Extract SRTM zip file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Look for .hgt file
                hgt_files = [f for f in zip_ref.namelist() if f.endswith('.hgt')]
                
                if hgt_files:
                    hgt_file = hgt_files[0]
                    extracted_path = self.download_dir / f"{tile_name}.hgt"
                    
                    with zip_ref.open(hgt_file) as source, open(extracted_path, 'wb') as target:
                        target.write(source.read())
                        
                    return extracted_path
                    
        except Exception as e:
            self.logger.error(f"Failed to extract {zip_path}: {e}")
            
        return None


class PopulationDataAcquirer(FileDownloader):
    """WorldPop population data acquirer"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, download_dir="data/raw/population", logger=logger)
        self.worldpop_base_url = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj"
        
    def fetch_population_data(self, countries: List[str] = None, 
                             years: List[int] = None) -> List[Path]:
        """
        Download WorldPop population data
        
        Args:
            countries: List of ISO3 country codes
            years: List of years to download
            
        Returns:
            List of downloaded file paths
        """
        if years is None:
            years = [2020]  # Latest available
            
        if countries is None:
            countries = ['global']  # Download global dataset
            
        downloaded_files = []
        
        for year in years:
            for country in countries:
                if country.lower() == 'global':
                    file_url = f"{self.worldpop_base_url}/{year}/0_Mosaicked/ppp_{year}_1km_Aggregated_UNadj.tif"
                    filename = f"worldpop_global_{year}.tif"
                else:
                    file_url = f"{self.worldpop_base_url}/{year}/{country.upper()}/{country.lower()}_ppp_{year}_1km_Aggregated_UNadj.tif"
                    filename = f"worldpop_{country}_{year}.tif"
                
                try:
                    filepath = self.download_file(file_url, filename)
                    downloaded_files.append(filepath)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to download population data for {country} {year}: {e}")
                    
        return downloaded_files


class NightLightsAcquirer(FileDownloader):
    """VIIRS nighttime lights data acquirer"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, download_dir="data/raw/nightlights", logger=logger)
        self.viirs_base_url = "https://eogdata.mines.edu/nighttime_light/annual/v21"
        
    def fetch_viirs_data(self, years: List[int] = None) -> List[Path]:
        """
        Download VIIRS nighttime lights data
        
        Args:
            years: List of years to download
            
        Returns:
            List of downloaded file paths
        """
        if years is None:
            years = list(range(2012, 2024))  # VIIRS available from 2012
            
        downloaded_files = []
        
        for year in years:
            file_url = f"{self.viirs_base_url}/{year}/VNL_v21_npp_{year}_global_vcmslcfg_c202102150000.average_masked.tif.gz"
            filename = f"viirs_nightlights_{year}.tif.gz"
            
            try:
                filepath = self.download_file(file_url, filename)
                
                # Extract gzipped file
                extracted_file = self._extract_gzip(filepath)
                if extracted_file:
                    downloaded_files.append(extracted_file)
                    
            except Exception as e:
                self.logger.warning(f"Failed to download VIIRS data for {year}: {e}")
                
        return downloaded_files
        
    def _extract_gzip(self, gz_path: Path) -> Optional[Path]:
        """Extract gzipped file"""
        import gzip
        import shutil
        
        try:
            extracted_path = gz_path.with_suffix('')  # Remove .gz
            
            with gzip.open(gz_path, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            return extracted_path
            
        except Exception as e:
            self.logger.error(f"Failed to extract {gz_path}: {e}")
            return None


class LandCoverAcquirer(BaseDataAcquirer):
    """Land cover and vegetation data acquirer"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        
    def fetch_modis_ndvi(self, locations: List[Tuple[float, float]], 
                        start_date: str = "2020-01-01", 
                        end_date: str = "2020-12-31") -> pd.DataFrame:
        """
        Fetch MODIS NDVI data for locations
        (Note: This would typically use Google Earth Engine or NASA APIs)
        
        Args:
            locations: List of (lat, lon) tuples
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with NDVI values
        """
        # For demonstration, generate synthetic NDVI data
        # In production, this would call Google Earth Engine or NASA APIs
        
        ndvi_data = []
        np.random.seed(42)
        
        for i, (lat, lon) in enumerate(locations):
            # Simulate NDVI based on latitude (vegetation patterns)
            base_ndvi = 0.7 - abs(lat) / 90 * 0.5  # Less vegetation near poles
            base_ndvi += np.random.normal(0, 0.1)  # Add noise
            base_ndvi = np.clip(base_ndvi, -1, 1)  # NDVI range
            
            ndvi_record = {
                'location_id': i,
                'latitude': lat,
                'longitude': lon,
                'ndvi_mean': base_ndvi,
                'ndvi_std': abs(np.random.normal(0, 0.05)),
                'ndvi_min': base_ndvi - abs(np.random.normal(0, 0.1)),
                'ndvi_max': base_ndvi + abs(np.random.normal(0, 0.1)),
                'start_date': start_date,
                'end_date': end_date
            }
            
            ndvi_data.append(ndvi_record)
            
        return pd.DataFrame(ndvi_data)


class OSMDataAcquirer(BaseDataAcquirer):
    """OpenStreetMap data acquirer via Overpass API"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        
    def fetch_amenities(self, bbox: List[float], 
                       amenity_types: List[str] = None) -> pd.DataFrame:
        """
        Fetch amenities from OpenStreetMap
        
        Args:
            bbox: [south, west, north, east] bounding box
            amenity_types: List of amenity types to fetch
            
        Returns:
            DataFrame with amenity data
        """
        if amenity_types is None:
            amenity_types = [
                'hospital', 'clinic', 'pharmacy', 'school', 'university',
                'restaurant', 'market', 'place_of_worship', 'park'
            ]
            
        all_amenities = []
        
        for amenity_type in amenity_types:
            cache_key = f"osm_amenities_{amenity_type}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            cached_data = self.load_cached_data(cache_key, max_age_hours=168)  # Cache for a week
            
            if cached_data is not None:
                all_amenities.extend(cached_data if isinstance(cached_data, list) else [cached_data])
                continue
                
            query = self._build_overpass_query(bbox, amenity_type)
            
            try:
                response = requests.post(self.overpass_url, data=query, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                amenities = self._parse_osm_response(data, amenity_type)
                all_amenities.extend(amenities)
                
                # Cache the results
                self.cache_data(amenities, cache_key)
                
                self.logger.info(f"Fetched {len(amenities)} {amenity_type} amenities")
                
                # Rate limiting
                import time
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Failed to fetch {amenity_type} amenities: {e}")
                
        return pd.DataFrame(all_amenities)
        
    def _build_overpass_query(self, bbox: List[float], amenity_type: str) -> str:
        """Build Overpass API query"""
        south, west, north, east = bbox
        
        query = f"""
        [out:json][timeout:60];
        (
          node["amenity"="{amenity_type}"]({south},{west},{north},{east});
          way["amenity"="{amenity_type}"]({south},{west},{north},{east});
        );
        out center geom;
        """
        
        return query
        
    def _parse_osm_response(self, data: Dict, amenity_type: str) -> List[Dict]:
        """Parse OSM Overpass API response"""
        amenities = []
        
        for element in data.get('elements', []):
            try:
                amenity = {
                    'amenity_type': amenity_type,
                    'osm_id': element.get('id'),
                    'osm_type': element.get('type')
                }
                
                # Get coordinates
                if element['type'] == 'node':
                    amenity['latitude'] = element.get('lat')
                    amenity['longitude'] = element.get('lon')
                elif element['type'] == 'way' and 'center' in element:
                    amenity['latitude'] = element['center'].get('lat')
                    amenity['longitude'] = element['center'].get('lon')
                else:
                    continue
                    
                # Get tags
                tags = element.get('tags', {})
                amenity['name'] = tags.get('name', '')
                amenity['operator'] = tags.get('operator', '')
                amenity['opening_hours'] = tags.get('opening_hours', '')
                
                # Specific attributes by amenity type
                if amenity_type in ['hospital', 'clinic']:
                    amenity['healthcare'] = tags.get('healthcare', '')
                    amenity['emergency'] = tags.get('emergency', '')
                elif amenity_type == 'school':
                    amenity['school_type'] = tags.get('school:type', '')
                elif amenity_type == 'place_of_worship':
                    amenity['religion'] = tags.get('religion', '')
                    
                amenities.append(amenity)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse OSM element: {e}")
                
        return amenities


class ProtectedAreasAcquirer(FileDownloader):
    """World Database on Protected Areas (WDPA) acquirer"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, download_dir="data/raw/protected_areas", logger=logger)
        
    def fetch_wdpa_data(self, countries: List[str] = None) -> List[Path]:
        """
        Download WDPA protected areas data
        
        Note: WDPA requires registration and direct download.
        This is a placeholder implementation.
        
        Args:
            countries: List of ISO3 country codes
            
        Returns:
            List of downloaded file paths
        """
        # WDPA requires manual download with registration
        # This function would download from stored URLs after user registration
        
        self.logger.info("WDPA data requires manual download from https://www.protectedplanet.net/")
        self.logger.info("Please register and download country-specific data")
        
        return []


class GeospatialDataProcessor:
    """Process geospatial data and extract features"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def process_elevation_data(self, elevation_files: List[Path], 
                              grid_points: pd.DataFrame) -> pd.DataFrame:
        """Process elevation data to grid points"""
        
        try:
            import rasterio
            from rasterio.transform import rowcol
        except ImportError:
            self.logger.error("rasterio required: pip install rasterio")
            return pd.DataFrame()
            
        elevation_features = []
        
        for _, point in grid_points.iterrows():
            lat, lon = point['latitude'], point['longitude']
            
            elevation_data = {
                'geo_id': point.get('geo_id'),
                'latitude': lat,
                'longitude': lon,
                'elevation': 0,
                'slope': 0,
                'aspect': 0,
                'roughness': 0
            }
            
            # Find relevant elevation file(s)
            for elev_file in elevation_files:
                try:
                    with rasterio.open(elev_file) as src:
                        # Check if point is within raster bounds
                        bounds = src.bounds
                        if (bounds.left <= lon <= bounds.right and 
                            bounds.bottom <= lat <= bounds.top):
                            
                            # Get pixel coordinates
                            row, col = rowcol(src.transform, lon, lat)
                            
                            # Read elevation
                            if 0 <= row < src.height and 0 <= col < src.width:
                                elevation = src.read(1)[row, col]
                                
                                if elevation != src.nodata:
                                    elevation_data['elevation'] = float(elevation)
                                    
                                    # Calculate terrain metrics from neighborhood
                                    terrain_metrics = self._calculate_terrain_metrics(
                                        src, row, col
                                    )
                                    elevation_data.update(terrain_metrics)
                                    
                            break  # Found the right file
                            
                except Exception as e:
                    self.logger.warning(f"Error processing elevation for {lat}, {lon}: {e}")
                    
            elevation_features.append(elevation_data)
            
        return pd.DataFrame(elevation_features)
        
    def _calculate_terrain_metrics(self, raster_src, row: int, col: int, 
                                 window_size: int = 3) -> Dict[str, float]:
        """Calculate terrain metrics from elevation raster"""
        
        try:
            # Extract neighborhood
            half_window = window_size // 2
            window = raster_src.read(
                1,
                window=((max(0, row - half_window), 
                        min(raster_src.height, row + half_window + 1)),
                       (max(0, col - half_window), 
                        min(raster_src.width, col + half_window + 1)))
            )
            
            if window.size < 4:
                return {'slope': 0, 'aspect': 0, 'roughness': 0}
                
            # Calculate slope and aspect using gradient
            dy, dx = np.gradient(window.astype(float))
            
            if dy.size > 0 and dx.size > 0:
                # Get center values
                center_dy = dy[dy.shape[0]//2, dy.shape[1]//2] if dy.size > 1 else 0
                center_dx = dx[dx.shape[0]//2, dx.shape[1]//2] if dx.size > 1 else 0
                
                slope = np.sqrt(center_dx**2 + center_dy**2)
                aspect = np.arctan2(center_dy, center_dx) * 180 / np.pi
                
                # Roughness as standard deviation
                roughness = np.std(window[window != raster_src.nodata])
                
                return {
                    'slope': float(slope),
                    'aspect': float(aspect),
                    'roughness': float(roughness) if not np.isnan(roughness) else 0
                }
                
        except Exception as e:
            self.logger.warning(f"Error calculating terrain metrics: {e}")
            
        return {'slope': 0, 'aspect': 0, 'roughness': 0}
        
    def process_osm_amenities(self, amenities_df: pd.DataFrame,
                             grid_points: pd.DataFrame,
                             buffer_km: float = 5.0) -> pd.DataFrame:
        """Process OSM amenities to grid-based features"""
        
        if amenities_df.empty or grid_points.empty:
            return pd.DataFrame()
            
        # Calculate amenity densities within buffer
        amenity_features = []
        
        for _, point in grid_points.iterrows():
            point_lat, point_lon = point['latitude'], point['longitude']
            
            features = {
                'geo_id': point.get('geo_id'),
                'latitude': point_lat,
                'longitude': point_lon
            }
            
            # Calculate distances to all amenities
            if 'latitude' in amenities_df.columns and 'longitude' in amenities_df.columns:
                distances = self._haversine_distance(
                    point_lat, point_lon,
                    amenities_df['latitude'], amenities_df['longitude']
                )
                
                # Filter amenities within buffer
                nearby_amenities = amenities_df[distances <= buffer_km]
                
                # Count by amenity type
                amenity_counts = nearby_amenities['amenity_type'].value_counts()
                
                # Convert to density (per km²)
                buffer_area = np.pi * (buffer_km ** 2)
                
                for amenity_type, count in amenity_counts.items():
                    features[f'{amenity_type}_density'] = count / buffer_area
                    
                # Calculate access scores
                features['healthcare_access'] = (
                    amenity_counts.get('hospital', 0) * 3 + 
                    amenity_counts.get('clinic', 0) * 2 + 
                    amenity_counts.get('pharmacy', 0)
                ) / buffer_area
                
                features['education_access'] = (
                    amenity_counts.get('university', 0) * 3 + 
                    amenity_counts.get('school', 0) * 2
                ) / buffer_area
                
                features['social_amenities'] = (
                    amenity_counts.get('place_of_worship', 0) +
                    amenity_counts.get('park', 0) +
                    amenity_counts.get('market', 0)
                ) / buffer_area
                
            amenity_features.append(features)
            
        return pd.DataFrame(amenity_features)
        
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between points"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
        
    def create_accessibility_features(self, grid_points: pd.DataFrame,
                                    hospitals_df: pd.DataFrame = None,
                                    cities_df: pd.DataFrame = None) -> pd.DataFrame:
        """Create accessibility features"""
        
        accessibility_features = []
        
        for _, point in grid_points.iterrows():
            point_lat, point_lon = point['latitude'], point['longitude']
            
            features = {
                'geo_id': point.get('geo_id'),
                'latitude': point_lat,
                'longitude': point_lon,
                'travel_time_to_hospital': np.inf,
                'travel_time_to_city': np.inf,
                'isolation_index': 0
            }
            
            # Distance to nearest hospital
            if hospitals_df is not None and not hospitals_df.empty:
                if 'latitude' in hospitals_df.columns:
                    hospital_distances = self._haversine_distance(
                        point_lat, point_lon,
                        hospitals_df['latitude'], hospitals_df['longitude']
                    )
                    min_hospital_distance = hospital_distances.min()
                    
                    # Approximate travel time (assuming 50 km/h average speed)
                    features['travel_time_to_hospital'] = min_hospital_distance / 50 * 60  # minutes
                    
            # Distance to nearest major city
            if cities_df is not None and not cities_df.empty:
                if 'latitude' in cities_df.columns:
                    city_distances = self._haversine_distance(
                        point_lat, point_lon,
                        cities_df['latitude'], cities_df['longitude']
                    )
                    min_city_distance = city_distances.min()
                    
                    # Approximate travel time
                    features['travel_time_to_city'] = min_city_distance / 60 * 60  # minutes (highway speed)
                    
            # Isolation index (inverse of population density in surrounding area)
            features['isolation_index'] = 1.0  # Placeholder - would calculate from population data
            
            accessibility_features.append(features)
            
        return pd.DataFrame(accessibility_features)