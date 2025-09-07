"""
Lifestyle and social environment feature engineering using OpenStreetMap data
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import requests
import time


class LifestyleFeatures:
    """Generate lifestyle and social environment features from OSM"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
    def generate_osm_features(self, grid_df: gpd.GeoDataFrame, 
                            buffer_km: float = 2.0) -> pd.DataFrame:
        """
        Generate OpenStreetMap-based lifestyle features
        
        Args:
            grid_df: Grid GeoDataFrame
            buffer_km: Buffer radius for feature extraction
            
        Returns:
            DataFrame with lifestyle features
        """
        features = pd.DataFrame({'geo_id': grid_df['geo_id']})
        
        # Process in batches to avoid overwhelming Overpass API
        batch_size = 100
        batches = [grid_df[i:i+batch_size] for i in range(0, len(grid_df), batch_size)]
        
        all_features = []
        
        for i, batch in enumerate(batches):
            self.logger.info(f"Processing batch {i+1}/{len(batches)}")
            
            try:
                batch_features = self._process_grid_batch(batch, buffer_km)
                all_features.append(batch_features)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error processing batch {i+1}: {e}")
                # Create empty features for failed batch
                empty_features = pd.DataFrame({
                    'geo_id': batch['geo_id'],
                    **{col: 0 for col in self._get_feature_columns()}
                })
                all_features.append(empty_features)
                
        if all_features:
            result = pd.concat(all_features, ignore_index=True)
        else:
            result = pd.DataFrame({
                'geo_id': grid_df['geo_id'],
                **{col: 0 for col in self._get_feature_columns()}
            })
            
        return result
        
    def _process_grid_batch(self, grid_batch: gpd.GeoDataFrame, 
                           buffer_km: float) -> pd.DataFrame:
        """Process a batch of grid cells"""
        # Get bounding box for batch
        bounds = grid_batch.total_bounds
        buffer_deg = buffer_km / 111.32  # Convert km to degrees
        bbox = [
            bounds[1] - buffer_deg,  # south
            bounds[0] - buffer_deg,  # west  
            bounds[3] + buffer_deg,  # north
            bounds[2] + buffer_deg   # east
        ]
        
        # Query OSM data for the region
        osm_data = self._query_osm_region(bbox)
        
        # Extract features for each grid cell
        features_list = []
        for _, grid_cell in grid_batch.iterrows():
            cell_features = self._extract_cell_features(grid_cell, osm_data, buffer_km)
            features_list.append(cell_features)
            
        return pd.DataFrame(features_list)
        
    def _query_osm_region(self, bbox: List[float]) -> Dict[str, List]:
        """Query OSM data for a region using Overpass API"""
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        # Build Overpass query
        query = self._build_overpass_query(bbox)
        
        try:
            response = requests.post(overpass_url, data=query, timeout=60)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.warning(f"OSM query failed: {e}")
            return {"elements": []}
            
    def _build_overpass_query(self, bbox: List[float]) -> str:
        """Build Overpass API query for lifestyle features"""
        south, west, north, east = bbox
        
        query = f"""
        [out:json][timeout:60];
        (
          // Religious sites
          node["amenity"~"^(place_of_worship)$"]({south},{west},{north},{east});
          way["amenity"~"^(place_of_worship)$"]({south},{west},{north},{east});
          
          // Markets and food access
          node["amenity"~"^(marketplace|farmers_market)$"]({south},{west},{north},{east});
          node["shop"~"^(supermarket|greengrocer|butcher|fishmonger|bakery)$"]({south},{west},{north},{east});
          way["amenity"~"^(marketplace|farmers_market)$"]({south},{west},{north},{east});
          
          // Healthcare
          node["amenity"~"^(hospital|clinic|doctors|pharmacy)$"]({south},{west},{north},{east});
          way["amenity"~"^(hospital|clinic)$"]({south},{west},{north},{east});
          
          // Transportation and walkability
          node["amenity"~"^(bus_station|bus_stop)$"]({south},{west},{north},{east});
          way["highway"~"^(footway|cycleway|path|pedestrian)$"]({south},{west},{north},{east});
          
          // Recreation and community
          node["amenity"~"^(community_centre|social_centre)$"]({south},{west},{north},{east});
          node["leisure"~"^(park|garden|playground)$"]({south},{west},{north},{east});
          way["leisure"~"^(park|garden)$"]({south},{west},{north},{east});
          
          // Mediterranean diet proxies
          node["landuse"="orchard"]({south},{west},{north},{east});
          way["landuse"="orchard"]({south},{west},{north},{east});
          node["natural"="coastline"]({south},{west},{north},{east});
          way["natural"="coastline"]({south},{west},{north},{east});
        );
        out geom;
        """
        
        return query
        
    def _extract_cell_features(self, grid_cell: gpd.GeoSeries, 
                             osm_data: Dict, buffer_km: float) -> Dict[str, Any]:
        """Extract features for a single grid cell"""
        from shapely.geometry import Point, Polygon
        from shapely.ops import transform
        import pyproj
        
        # Create buffer around grid cell
        cell_point = grid_cell['geometry']
        
        # Convert to local projection for accurate distance
        local_proj = pyproj.Proj(
            proj='aeqd',
            lat_0=grid_cell['lat'],
            lon_0=grid_cell['lon'],
            datum='WGS84'
        )
        wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
        
        # Transform to local coordinates and create buffer
        project_to_local = pyproj.Transformer.from_proj(wgs84, local_proj).transform
        project_to_wgs84 = pyproj.Transformer.from_proj(local_proj, wgs84).transform
        
        local_point = transform(project_to_local, cell_point)
        buffer_local = local_point.buffer(buffer_km * 1000)  # Convert km to m
        buffer_wgs84 = transform(project_to_wgs84, buffer_local)
        
        # Initialize feature counts
        features = {
            'geo_id': grid_cell['geo_id'],
            'religious_site_density': 0,
            'market_density': 0,
            'healthcare_density': 0,
            'public_transport_density': 0,
            'walkability_score': 0,
            'community_facility_density': 0,
            'green_space_density': 0,
            'mediterranean_diet_proxy': 0,
            'cycling_infrastructure': 0
        }
        
        # Count features within buffer
        for element in osm_data.get("elements", []):
            if not self._element_in_buffer(element, buffer_wgs84):
                continue
                
            tags = element.get("tags", {})
            
            # Categorize based on tags
            if tags.get("amenity") == "place_of_worship":
                features['religious_site_density'] += 1
                
            elif tags.get("amenity") in ["marketplace", "farmers_market"]:
                features['market_density'] += 1
                
            elif tags.get("shop") in ["supermarket", "greengrocer", "butcher", "fishmonger", "bakery"]:
                features['market_density'] += 0.5  # Weight shops less than markets
                
            elif tags.get("amenity") in ["hospital", "clinic", "doctors", "pharmacy"]:
                features['healthcare_density'] += 1
                
            elif tags.get("amenity") in ["bus_station", "bus_stop"]:
                features['public_transport_density'] += 1
                
            elif tags.get("highway") in ["footway", "cycleway", "path", "pedestrian"]:
                features['walkability_score'] += 1
                if tags.get("highway") == "cycleway":
                    features['cycling_infrastructure'] += 1
                    
            elif tags.get("amenity") in ["community_centre", "social_centre"]:
                features['community_facility_density'] += 1
                
            elif tags.get("leisure") in ["park", "garden", "playground"]:
                features['green_space_density'] += 1
                
            elif tags.get("landuse") == "orchard":
                features['mediterranean_diet_proxy'] += 1
                
            elif tags.get("natural") == "coastline":
                features['mediterranean_diet_proxy'] += 0.5
                
        # Normalize by buffer area (features per km²)
        buffer_area_km2 = np.pi * (buffer_km ** 2)
        
        for key in features:
            if key != 'geo_id' and isinstance(features[key], (int, float)):
                features[key] = features[key] / buffer_area_km2
                
        return features
        
    def _element_in_buffer(self, element: Dict, buffer_polygon: Polygon) -> bool:
        """Check if OSM element is within buffer"""
        from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
        
        try:
            if element["type"] == "node":
                point = Point(element["lon"], element["lat"])
                return buffer_polygon.contains(point)
                
            elif element["type"] == "way":
                if "geometry" in element:
                    coords = [(node["lon"], node["lat"]) for node in element["geometry"]]
                    if len(coords) > 1:
                        line = LineString(coords)
                        return buffer_polygon.intersects(line)
                        
            return False
            
        except Exception:
            return False
            
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        return [
            'religious_site_density',
            'market_density', 
            'healthcare_density',
            'public_transport_density',
            'walkability_score',
            'community_facility_density',
            'green_space_density',
            'mediterranean_diet_proxy',
            'cycling_infrastructure'
        ]
        
    def create_lifestyle_proxies(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create derived lifestyle proxy features"""
        result = features.copy()
        
        # Social cohesion proxy
        result['social_cohesion_proxy'] = (
            result['religious_site_density'] + 
            result['community_facility_density']
        )
        
        # Food security proxy
        result['food_security_proxy'] = result['market_density']
        
        # Active lifestyle proxy
        result['active_lifestyle_proxy'] = (
            result['walkability_score'] + 
            result['cycling_infrastructure'] +
            result['green_space_density']
        )
        
        # Healthcare access proxy
        result['healthcare_access_proxy'] = (
            result['healthcare_density'] + 
            result['public_transport_density'] * 0.1  # Transit improves access
        )
        
        return result