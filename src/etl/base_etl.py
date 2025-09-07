"""
Base ETL classes for Blue Zones Quantified project
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import geopandas as gpd
import xarray as xr
import logging
from pathlib import Path


class BaseETL(ABC):
    """Base class for all ETL operations"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.data_dir = Path("data")
        
    @abstractmethod
    def extract(self) -> Any:
        """Extract data from source"""
        pass
        
    @abstractmethod  
    def transform(self, data: Any) -> Any:
        """Transform data"""
        pass
        
    @abstractmethod
    def load(self, data: Any, output_path: str) -> None:
        """Load data to destination"""
        pass
        
    def run(self, output_path: str) -> None:
        """Run full ETL pipeline"""
        self.logger.info(f"Starting ETL pipeline: {self.__class__.__name__}")
        
        # Extract
        raw_data = self.extract()
        self.logger.info("Data extraction completed")
        
        # Transform
        processed_data = self.transform(raw_data)
        self.logger.info("Data transformation completed")
        
        # Load
        self.load(processed_data, output_path)
        self.logger.info(f"Data loaded to: {output_path}")


class RasterETL(BaseETL):
    """Base class for raster data ETL"""
    
    def __init__(self, config: Dict[str, Any], grid_df: gpd.GeoDataFrame, 
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.grid_df = grid_df
        
    def extract_to_grid(self, raster_data: xr.DataArray) -> pd.DataFrame:
        """
        Extract raster values to grid points
        
        Args:
            raster_data: xarray DataArray with raster data
            
        Returns:
            DataFrame with grid_id and extracted values
        """
        from rasterio.transform import from_bounds
        import rasterio.features
        
        # Get grid coordinates
        coords = [(geom.x, geom.y) for geom in self.grid_df.geometry]
        
        # Sample raster at grid points
        values = []
        for x, y in coords:
            try:
                value = raster_data.sel(x=x, y=y, method='nearest').values
                values.append(value)
            except (KeyError, IndexError):
                values.append(None)
                
        return pd.DataFrame({
            'geo_id': self.grid_df['geo_id'],
            'value': values
        })
        
    def aggregate_temporal(self, data: xr.Dataset, 
                          method: str = 'mean') -> xr.DataArray:
        """
        Aggregate temporal data
        
        Args:
            data: xarray Dataset with time dimension
            method: Aggregation method ('mean', 'sum', 'max', etc.)
            
        Returns:
            Aggregated DataArray
        """
        if method == 'mean':
            return data.mean(dim='time')
        elif method == 'sum':
            return data.sum(dim='time')
        elif method == 'max':
            return data.max(dim='time')
        elif method == 'min':
            return data.min(dim='time')
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")


class VectorETL(BaseETL):
    """Base class for vector data ETL"""
    
    def spatial_join_to_grid(self, vector_data: gpd.GeoDataFrame,
                           grid_df: gpd.GeoDataFrame,
                           how: str = 'inner') -> gpd.GeoDataFrame:
        """
        Spatial join vector data to grid
        
        Args:
            vector_data: Vector GeoDataFrame
            grid_df: Grid GeoDataFrame  
            how: Join method
            
        Returns:
            Joined GeoDataFrame
        """
        return gpd.sjoin(grid_df, vector_data, how=how, predicate='intersects')
        
    def aggregate_by_grid(self, joined_data: gpd.GeoDataFrame,
                         value_cols: List[str],
                         agg_method: str = 'mean') -> pd.DataFrame:
        """
        Aggregate values by grid cell
        
        Args:
            joined_data: Spatially joined data
            value_cols: Columns to aggregate
            agg_method: Aggregation method
            
        Returns:
            Aggregated DataFrame
        """
        agg_dict = {col: agg_method for col in value_cols}
        return joined_data.groupby('geo_id').agg(agg_dict).reset_index()


class APIExtractor:
    """Helper class for API data extraction"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        
    def make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """Make API request"""
        import requests
        
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
            
        if params is None:
            params = {}
            
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        return response.json()


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_grid_coverage(data: pd.DataFrame, 
                             expected_grid_size: int,
                             logger: logging.Logger) -> bool:
        """Validate grid data coverage"""
        actual_size = len(data)
        coverage_pct = (actual_size / expected_grid_size) * 100
        
        logger.info(f"Grid coverage: {coverage_pct:.1f}% ({actual_size}/{expected_grid_size})")
        
        if coverage_pct < 80:
            logger.warning("Grid coverage below 80%")
            return False
            
        return True
        
    @staticmethod
    def validate_temporal_coverage(data: pd.DataFrame,
                                 time_col: str,
                                 expected_years: List[int],
                                 logger: logging.Logger) -> bool:
        """Validate temporal data coverage"""
        if time_col not in data.columns:
            logger.error(f"Time column '{time_col}' not found")
            return False
            
        actual_years = sorted(data[time_col].unique())
        missing_years = set(expected_years) - set(actual_years)
        
        if missing_years:
            logger.warning(f"Missing years: {sorted(missing_years)}")
            
        coverage_pct = (len(set(actual_years) & set(expected_years)) / len(expected_years)) * 100
        logger.info(f"Temporal coverage: {coverage_pct:.1f}%")
        
        return coverage_pct >= 80
        
    @staticmethod
    def validate_value_ranges(data: pd.DataFrame,
                            value_ranges: Dict[str, tuple],
                            logger: logging.Logger) -> bool:
        """Validate that values are within expected ranges"""
        all_valid = True
        
        for col, (min_val, max_val) in value_ranges.items():
            if col not in data.columns:
                continue
                
            invalid_count = ((data[col] < min_val) | (data[col] > max_val)).sum()
            if invalid_count > 0:
                invalid_pct = (invalid_count / len(data)) * 100
                logger.warning(f"{col}: {invalid_count} values ({invalid_pct:.1f}%) outside range [{min_val}, {max_val}]")
                
                if invalid_pct > 5:  # More than 5% invalid
                    all_valid = False
                    
        return all_valid