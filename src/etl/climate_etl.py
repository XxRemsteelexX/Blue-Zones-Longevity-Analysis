"""
Climate data ETL for ERA5 reanalysis data
"""
import pandas as pd
import xarray as xr
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from .base_etl import RasterETL


class ClimateETL(RasterETL):
    """ETL for ERA5 climate data"""
    
    def __init__(self, config: Dict[str, Any], grid_df, logger: Optional[logging.Logger] = None):
        super().__init__(config, grid_df, logger)
        self.variables = ['temperature', 'precipitation', 'relative_humidity']
        self.climate_config = config['data_sources']['climate']
        
    def extract(self) -> Dict[str, xr.Dataset]:
        """
        Extract ERA5 climate data
        
        Returns:
            Dictionary of datasets by variable
        """
        climate_data = {}
        climate_dir = self.data_dir / "raw" / "climate"
        
        for variable in self.variables:
            self.logger.info(f"Loading {variable} data")
            
            # Look for NetCDF files
            pattern = f"*{variable}*.nc"
            files = list(climate_dir.glob(pattern))
            
            if not files:
                self.logger.warning(f"No {variable} files found in {climate_dir}")
                continue
                
            try:
                # Load and concatenate files
                datasets = []
                for file in sorted(files):
                    ds = xr.open_dataset(file)
                    datasets.append(ds)
                    
                if datasets:
                    climate_data[variable] = xr.concat(datasets, dim='time')
                    
            except Exception as e:
                self.logger.error(f"Error loading {variable} data: {e}")
                
        return climate_data
        
    def transform(self, data: Dict[str, xr.Dataset]) -> pd.DataFrame:
        """
        Transform climate data to grid features
        
        Args:
            data: Dictionary of climate datasets
            
        Returns:
            DataFrame with climate features
        """
        features_list = []
        
        for variable, dataset in data.items():
            self.logger.info(f"Processing {variable}")
            
            # Get main variable (assumes single data variable per dataset)
            if len(dataset.data_vars) == 1:
                var_name = list(dataset.data_vars)[0]
                da = dataset[var_name]
            else:
                # Try common variable names
                possible_names = [variable, variable.split('_')[0], 
                                'temperature' if 'temp' in variable else variable]
                var_name = None
                for name in possible_names:
                    if name in dataset.data_vars:
                        var_name = name
                        break
                        
                if var_name is None:
                    self.logger.warning(f"Could not identify variable in {variable} dataset")
                    continue
                    
                da = dataset[var_name]
            
            # Generate features for this variable
            var_features = self._generate_climate_features(da, variable)
            features_list.append(var_features)
            
        if not features_list:
            return pd.DataFrame()
            
        # Combine all features
        result = features_list[0]
        for features in features_list[1:]:
            result = result.merge(features, on='geo_id', how='outer')
            
        return result
        
    def _generate_climate_features(self, data_array: xr.DataArray, 
                                  variable: str) -> pd.DataFrame:
        """
        Generate climate features from data array
        
        Args:
            data_array: xarray DataArray with climate data
            variable: Variable name
            
        Returns:
            DataFrame with climate features
        """
        features = {'geo_id': self.grid_df['geo_id']}
        
        # Annual mean
        annual_mean = data_array.groupby('time.year').mean('time')
        overall_mean = annual_mean.mean('year')
        mean_values = self.extract_to_grid(overall_mean)
        features[f'{variable}_mean'] = mean_values['value']
        
        # Seasonality (coefficient of variation across months)
        monthly_mean = data_array.groupby('time.month').mean('time')
        monthly_std = monthly_mean.std('month')
        monthly_cv = monthly_std / monthly_mean.mean('month')
        seasonality_values = self.extract_to_grid(monthly_cv)
        features[f'{variable}_seasonality'] = seasonality_values['value']
        
        # Long-term trends and moving averages
        if 'time' in data_array.dims:
            years = pd.to_datetime(data_array.time.values).year
            unique_years = sorted(set(years))
            
            # 20-year moving averages
            if len(unique_years) >= 20:
                moving_avg_20y = self._calculate_moving_averages(
                    annual_mean, window=20
                )
                avg_20y_values = self.extract_to_grid(moving_avg_20y.mean('year'))
                features[f'{variable}_20y_avg'] = avg_20y_values['value']
                
            # 10-year trends
            if len(unique_years) >= 10:
                trend_10y = self._calculate_trends(annual_mean, window=10)
                trend_values = self.extract_to_grid(trend_10y)
                features[f'{variable}_10y_trend'] = trend_values['value']
        
        # Variable-specific features
        if variable == 'temperature':
            # Diurnal temperature range (if available)
            if 'hour' in data_array.dims:
                daily_max = data_array.groupby('time.date').max()
                daily_min = data_array.groupby('time.date').min()
                diurnal_range = (daily_max - daily_min).mean('date')
                dtr_values = self.extract_to_grid(diurnal_range)
                features[f'{variable}_diurnal_range'] = dtr_values['value']
                
        elif variable == 'precipitation':
            # Precipitation intensity and dry days
            annual_sum = data_array.groupby('time.year').sum('time')
            precip_sum = self.extract_to_grid(annual_sum.mean('year'))
            features[f'{variable}_annual_sum'] = precip_sum['value']
            
        return pd.DataFrame(features)
        
    def _calculate_moving_averages(self, data: xr.DataArray, 
                                  window: int) -> xr.DataArray:
        """Calculate rolling moving averages"""
        return data.rolling(year=window, center=True).mean()
        
    def _calculate_trends(self, data: xr.DataArray, window: int) -> xr.DataArray:
        """Calculate linear trends over specified window"""
        # Get last 'window' years
        recent_data = data.isel(year=slice(-window, None))
        years = recent_data.year.values
        
        # Calculate linear trend for each grid cell
        def calc_trend(values):
            if np.isnan(values).all():
                return np.nan
            valid_mask = ~np.isnan(values)
            if valid_mask.sum() < 3:  # Need at least 3 points
                return np.nan
            return np.polyfit(years[valid_mask], values[valid_mask], 1)[0]
        
        trends = xr.apply_ufunc(
            calc_trend,
            recent_data,
            input_core_dims=[['year']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
        )
        
        return trends
        
    def load(self, data: pd.DataFrame, output_path: str) -> None:
        """Load climate features to parquet"""
        output_file = self.data_dir / "processed" / f"{output_path}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_parquet(output_file, compression='snappy')
        self.logger.info(f"Climate features saved to {output_file}")
        
        # Save summary statistics
        summary_file = output_file.with_suffix('.summary.csv')
        summary = data.describe()
        summary.to_csv(summary_file)


class CDSAPIExtractor:
    """Climate Data Store API extractor for ERA5"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def download_era5_data(self, variable: str, years: List[int], 
                          area: List[float], output_path: str) -> None:
        """
        Download ERA5 data via CDS API
        
        Args:
            variable: ERA5 variable name
            years: List of years to download
            area: [north, west, south, east] bounding box
            output_path: Output file path
        """
        try:
            import cdsapi
        except ImportError:
            raise ImportError("cdsapi package required. Install with: pip install cdsapi")
            
        c = cdsapi.Client()
        
        # Map variable names to CDS variable names
        variable_mapping = {
            'temperature': '2m_temperature',
            'precipitation': 'total_precipitation', 
            'relative_humidity': 'relative_humidity'
        }
        
        cds_variable = variable_mapping.get(variable, variable)
        
        request = {
            'product_type': 'reanalysis',
            'variable': cds_variable,
            'year': [str(y) for y in years],
            'month': [f'{i:02d}' for i in range(1, 13)],
            'day': [f'{i:02d}' for i in range(1, 32)],
            'time': '12:00',  # Daily at noon
            'area': area,  # [north, west, south, east]
            'format': 'netcdf',
        }
        
        c.retrieve('reanalysis-era5-single-levels', request, output_path)