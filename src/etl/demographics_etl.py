"""
Demographics and socioeconomic data ETL
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from .base_etl import RasterETL, VectorETL, APIExtractor


class DemographicsETL(RasterETL):
    """ETL for demographic data (WorldPop, etc.)"""
    
    def __init__(self, config: Dict[str, Any], grid_df, logger: Optional[logging.Logger] = None):
        super().__init__(config, grid_df, logger)
        self.pop_config = config['data_sources']['population']
        
    def extract(self) -> Dict[str, Any]:
        """Extract population data"""
        pop_data = {}
        pop_dir = self.data_dir / "raw" / "population"
        
        # Find population raster files
        for file in pop_dir.glob("*.tif"):
            # Extract year from filename if possible
            year_match = None
            for year in range(1990, 2025):
                if str(year) in file.name:
                    year_match = year
                    break
                    
            if year_match:
                try:
                    with rasterio.open(file) as src:
                        pop_data[year_match] = {
                            'raster': src.read(1),
                            'transform': src.transform,
                            'crs': src.crs,
                            'bounds': src.bounds
                        }
                        self.logger.info(f"Loaded population data for {year_match}")
                except Exception as e:
                    self.logger.error(f"Error loading {file}: {e}")
                    
        return pop_data
        
    def transform(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Transform population data to demographic features"""
        features_list = []
        
        for year, raster_info in data.items():
            self.logger.info(f"Processing population data for {year}")
            
            # Extract population density to grid
            pop_values = self._extract_population_to_grid(raster_info)
            
            # Calculate demographic features
            demo_features = self._calculate_demographic_features(pop_values, year)
            features_list.append(demo_features)
            
        if not features_list:
            return pd.DataFrame()
            
        # Combine temporal data
        result = pd.concat(features_list, ignore_index=True)
        
        # Calculate additional temporal features
        result = self._add_temporal_features(result)
        
        return result
        
    def _extract_population_to_grid(self, raster_info: Dict) -> pd.DataFrame:
        """Extract population values to grid points"""
        from rasterio.warp import transform
        from rasterio.transform import rowcol
        
        raster = raster_info['raster']
        transform_matrix = raster_info['transform']
        
        values = []
        for _, row in self.grid_df.iterrows():
            try:
                # Convert lat/lon to raster coordinates
                x, y = row['geometry'].x, row['geometry'].y
                col, row_idx = rowcol(transform_matrix, x, y)
                
                # Extract value (handle out of bounds)
                if 0 <= row_idx < raster.shape[0] and 0 <= col < raster.shape[1]:
                    value = raster[row_idx, col]
                    # Handle NoData values
                    if value < 0 or value > 1e6:  # Reasonable population bounds
                        value = 0
                else:
                    value = 0
                    
            except Exception:
                value = 0
                
            values.append(value)
            
        return pd.DataFrame({
            'geo_id': self.grid_df['geo_id'],
            'population': values
        })
        
    def _calculate_demographic_features(self, pop_data: pd.DataFrame, 
                                      year: int) -> pd.DataFrame:
        """Calculate demographic features"""
        features = pop_data.copy()
        features['year'] = year
        
        # Population density (log transformed)
        grid_area_km2 = (self.config['spatial']['primary_resolution_km'] ** 2)
        features['population_density'] = features['population'] / grid_area_km2
        features['population_density_log'] = np.log1p(features['population_density'])
        
        # Urban/rural classification (simplified)
        urban_threshold = 300  # people per km2
        features['is_urban'] = (features['population_density'] > urban_threshold).astype(int)
        
        return features
        
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""
        if 'year' not in data.columns or data['year'].nunique() < 2:
            return data
            
        # Sort by geo_id and year
        data = data.sort_values(['geo_id', 'year'])
        
        # Calculate population growth rates
        data['population_growth_rate'] = data.groupby('geo_id')['population'].pct_change()
        
        # Calculate urbanization trends
        data['urbanization_trend'] = data.groupby('geo_id')['is_urban'].diff()
        
        return data
        
    def load(self, data: pd.DataFrame, output_path: str) -> None:
        """Load demographic features"""
        output_file = self.data_dir / "processed" / f"{output_path}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_parquet(output_file, compression='snappy')
        self.logger.info(f"Demographic features saved to {output_file}")


class SocioeconomicETL(VectorETL):
    """ETL for socioeconomic data from World Bank API"""
    
    def __init__(self, config: Dict[str, Any], grid_df, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.grid_df = grid_df
        self.wb_indicators = {
            'GDP_PPP': 'NY.GDP.PCAP.PP.CD',
            'poverty_headcount': 'SI.POV.DDAY',
            'education_years': 'BAR.SCHL.15UP',
            'life_expectancy': 'SP.DYN.LE00.IN'
        }
        
    def extract(self) -> pd.DataFrame:
        """Extract World Bank data via API"""
        try:
            import wbdata
        except ImportError:
            self.logger.warning("wbdata not available, using cached data if available")
            return self._load_cached_data()
            
        all_data = []
        
        for indicator_name, indicator_code in self.wb_indicators.items():
            self.logger.info(f"Fetching {indicator_name} data")
            
            try:
                # Fetch data for all countries
                data = wbdata.get_dataframe({indicator_code: indicator_name})
                data = data.reset_index()
                
                # Clean and reshape
                data = data.melt(
                    id_vars=['country'], 
                    var_name='year', 
                    value_name=indicator_name
                )
                data['year'] = pd.to_datetime(data['year']).dt.year
                
                all_data.append(data)
                
            except Exception as e:
                self.logger.error(f"Error fetching {indicator_name}: {e}")
                
        if not all_data:
            return pd.DataFrame()
            
        # Merge all indicators
        result = all_data[0]
        for data in all_data[1:]:
            result = result.merge(data, on=['country', 'year'], how='outer')
            
        return result
        
    def _load_cached_data(self) -> pd.DataFrame:
        """Load cached socioeconomic data"""
        cache_file = self.data_dir / "raw" / "socioeconomic" / "wb_data.csv"
        if cache_file.exists():
            return pd.read_csv(cache_file)
        return pd.DataFrame()
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform socioeconomic data to grid"""
        if data.empty:
            return pd.DataFrame()
            
        # Load country geometries
        countries = self._get_country_geometries()
        
        # Join with country data
        socio_data = countries.merge(data, left_on='NAME', right_on='country', how='inner')
        
        # Spatial join to grid
        grid_socio = gpd.sjoin(
            self.grid_df, 
            socio_data[['geometry'] + list(self.wb_indicators.keys()) + ['year']], 
            how='inner'
        )
        
        # Clean up
        result = grid_socio.drop(columns=['geometry', 'index_right'])
        
        return result
        
    def _get_country_geometries(self) -> gpd.GeoDataFrame:
        """Get country geometries"""
        try:
            # Try to load from Natural Earth data
            import geopandas as gpd
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            return world[['name', 'geometry']].rename(columns={'name': 'NAME'})
        except:
            self.logger.warning("Could not load country geometries")
            return gpd.GeoDataFrame()
            
    def load(self, data: pd.DataFrame, output_path: str) -> None:
        """Load socioeconomic features"""
        if data.empty:
            self.logger.warning("No socioeconomic data to save")
            return
            
        output_file = self.data_dir / "processed" / f"{output_path}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_parquet(output_file, compression='snappy')
        self.logger.info(f"Socioeconomic features saved to {output_file}")


class NightLightsETL(RasterETL):
    """ETL for VIIRS nighttime lights data"""
    
    def __init__(self, config: Dict[str, Any], grid_df, logger: Optional[logging.Logger] = None):
        super().__init__(config, grid_df, logger)
        
    def extract(self) -> Dict[str, Any]:
        """Extract nighttime lights data"""
        ntl_data = {}
        ntl_dir = self.data_dir / "raw" / "nightlights"
        
        for file in ntl_dir.glob("*.tif"):
            # Extract year from filename
            year_match = None
            for year in range(2012, 2025):  # VIIRS available from 2012
                if str(year) in file.name:
                    year_match = year
                    break
                    
            if year_match:
                try:
                    with rasterio.open(file) as src:
                        ntl_data[year_match] = {
                            'raster': src.read(1),
                            'transform': src.transform,
                            'crs': src.crs
                        }
                        self.logger.info(f"Loaded nightlights for {year_match}")
                except Exception as e:
                    self.logger.error(f"Error loading {file}: {e}")
                    
        return ntl_data
        
    def transform(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Transform nightlights to features"""
        features_list = []
        
        for year, raster_info in data.items():
            self.logger.info(f"Processing nightlights for {year}")
            
            # Extract values to grid
            ntl_values = self._extract_nightlights_to_grid(raster_info)
            ntl_values['year'] = year
            
            features_list.append(ntl_values)
            
        if not features_list:
            return pd.DataFrame()
            
        result = pd.concat(features_list, ignore_index=True)
        
        # Add temporal features
        result = self._add_nightlights_features(result)
        
        return result
        
    def _extract_nightlights_to_grid(self, raster_info: Dict) -> pd.DataFrame:
        """Extract nightlights values to grid"""
        from rasterio.transform import rowcol
        
        raster = raster_info['raster']
        transform_matrix = raster_info['transform']
        
        values = []
        for _, row in self.grid_df.iterrows():
            try:
                x, y = row['geometry'].x, row['geometry'].y
                col, row_idx = rowcol(transform_matrix, x, y)
                
                if 0 <= row_idx < raster.shape[0] and 0 <= col < raster.shape[1]:
                    value = raster[row_idx, col]
                    # Clean negative values (masked pixels)
                    value = max(0, value) if not np.isnan(value) else 0
                else:
                    value = 0
                    
            except Exception:
                value = 0
                
            values.append(value)
            
        return pd.DataFrame({
            'geo_id': self.grid_df['geo_id'],
            'nighttime_lights': values
        })
        
    def _add_nightlights_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add nightlights-derived features"""
        # Log transform (nightlights are highly skewed)
        data['nighttime_lights_log'] = np.log1p(data['nighttime_lights'])
        
        # Development proxy (normalized)
        data['development_index'] = (
            data.groupby('year')['nighttime_lights']
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        )
        
        return data
        
    def load(self, data: pd.DataFrame, output_path: str) -> None:
        """Load nightlights features"""
        output_file = self.data_dir / "processed" / f"{output_path}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_parquet(output_file, compression='snappy')
        self.logger.info(f"Nightlights features saved to {output_file}")