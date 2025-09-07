"""
Climate data acquirer for ERA5 and other climate datasets
"""
import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_acquirer import BaseDataAcquirer, FileDownloader
import requests
import json
from pathlib import Path


class ERA5Acquirer(BaseDataAcquirer):
    """ERA5 climate data acquirer using Copernicus CDS API"""
    
    def __init__(self, config: Dict[str, Any], 
                 api_key: str = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.api_key = api_key
        self.cds_url = "https://cds.climate.copernicus.eu/api/v2"
        
        # ERA5 variables mapping
        self.era5_variables = {
            'temperature': '2m_temperature',
            'precipitation': 'total_precipitation',
            'humidity': 'relative_humidity',
            'wind_speed': '10m_wind_speed',
            'pressure': 'surface_pressure',
            'solar_radiation': 'surface_solar_radiation_downwards'
        }
        
    def fetch_data(self, variables: List[str] = None, 
                  years: List[int] = None,
                  months: List[int] = None,
                  area: List[float] = None) -> Dict[str, xr.Dataset]:
        """
        Fetch ERA5 climate data
        
        Args:
            variables: List of climate variables
            years: List of years to download
            months: List of months (1-12)
            area: [north, west, south, east] bounding box
            
        Returns:
            Dictionary of xarray datasets by variable
        """
        if not self.api_key:
            self.logger.error("CDS API key required for ERA5 data")
            return {}
            
        if variables is None:
            variables = ['temperature', 'precipitation', 'humidity']
            
        if years is None:
            years = list(range(1990, 2024))
            
        if months is None:
            months = list(range(1, 13))
            
        if area is None:
            area = [90, -180, -90, 180]  # Global
            
        datasets = {}
        
        for var in variables:
            if var not in self.era5_variables:
                self.logger.warning(f"Unknown variable: {var}")
                continue
                
            self.logger.info(f"Fetching ERA5 {var}")
            
            # Check cache first
            cache_key = f"era5_{var}_{min(years)}_{max(years)}"
            cached_data = self._load_cached_netcdf(cache_key)
            
            if cached_data is not None:
                datasets[var] = cached_data
            else:
                # Download from CDS
                dataset = self._download_era5_variable(var, years, months, area)
                if dataset is not None:
                    datasets[var] = dataset
                    self._cache_netcdf(dataset, cache_key)
                    
        return datasets
        
    def _download_era5_variable(self, variable: str, years: List[int],
                               months: List[int], area: List[float]) -> Optional[xr.Dataset]:
        """Download single ERA5 variable"""
        
        try:
            import cdsapi
        except ImportError:
            self.logger.error("cdsapi package required: pip install cdsapi")
            return None
            
        # Setup CDS client
        c = cdsapi.Client()
        
        era5_var = self.era5_variables[variable]
        
        # Create download request
        request = {
            'product_type': 'reanalysis',
            'variable': era5_var,
            'year': [str(y) for y in years],
            'month': [f'{m:02d}' for m in months],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': '12:00',  # Daily at noon
            'area': area,
            'format': 'netcdf',
        }
        
        # Download file
        temp_file = self.cache_dir / f"era5_{variable}_temp.nc"
        
        try:
            c.retrieve('reanalysis-era5-single-levels', request, str(temp_file))
            
            # Load as xarray dataset
            dataset = xr.open_dataset(temp_file)
            
            # Clean up temp file
            temp_file.unlink()
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"ERA5 download failed for {variable}: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return None
            
    def _load_cached_netcdf(self, cache_key: str) -> Optional[xr.Dataset]:
        """Load cached NetCDF data"""
        cache_file = self.cache_dir / f"{cache_key}.nc"
        
        if not cache_file.exists():
            return None
            
        try:
            return xr.open_dataset(cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to load cached NetCDF {cache_key}: {e}")
            return None
            
    def _cache_netcdf(self, dataset: xr.Dataset, cache_key: str) -> None:
        """Cache NetCDF dataset"""
        cache_file = self.cache_dir / f"{cache_key}.nc"
        
        try:
            dataset.to_netcdf(cache_file)
            self.logger.info(f"Cached NetCDF to {cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to cache NetCDF {cache_key}: {e}")


class OpenWeatherAPI(BaseDataAcquirer):
    """OpenWeatherMap API client for current and historical weather data"""
    
    def __init__(self, config: Dict[str, Any], 
                 api_key: str,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
    def fetch_current_weather(self, locations: List[Tuple[float, float]]) -> pd.DataFrame:
        """Fetch current weather for locations"""
        
        weather_data = []
        
        for i, (lat, lon) in enumerate(locations):
            cache_key = f"owm_current_{lat:.2f}_{lon:.2f}"
            cached_data = self.load_cached_data(cache_key, max_age_hours=1)
            
            if cached_data is not None:
                weather_data.extend(cached_data if isinstance(cached_data, list) else [cached_data])
                continue
                
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Parse weather data
                weather_point = {
                    'latitude': lat,
                    'longitude': lon,
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data.get('wind', {}).get('speed', 0),
                    'weather_description': data['weather'][0]['description'],
                    'timestamp': pd.Timestamp.now()
                }
                
                weather_data.append(weather_point)
                self.cache_data([weather_point], cache_key)
                
                # Rate limiting
                if i % 60 == 0 and i > 0:  # OpenWeatherMap free tier: 60 calls/min
                    import time
                    time.sleep(60)
                    
            except Exception as e:
                self.logger.error(f"Failed to fetch weather for {lat}, {lon}: {e}")
                
        return pd.DataFrame(weather_data)


class NOAADataAcquirer(FileDownloader):
    """NOAA climate data acquirer"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, download_dir="data/raw/climate", logger=logger)
        self.base_url = "https://www.ncei.noaa.gov/data"
        
    def fetch_ghcn_data(self, start_year: int = 1990, end_year: int = 2023) -> pd.DataFrame:
        """Fetch GHCN (Global Historical Climatology Network) data"""
        
        ghcn_url = f"{self.base_url}/ghcn/daily/by_year"
        climate_data = []
        
        for year in range(start_year, end_year + 1):
            cache_key = f"ghcn_{year}"
            cached_data = self.load_cached_data(cache_key, max_age_hours=168)  # Cache for a week
            
            if cached_data is not None:
                climate_data.append(cached_data)
                continue
                
            file_url = f"{ghcn_url}/{year}.csv.gz"
            
            try:
                # Download year data
                response = requests.get(file_url)
                response.raise_for_status()
                
                # Parse CSV data
                import io
                import gzip
                
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
                    year_data = pd.read_csv(f, names=[
                        'station_id', 'date', 'element', 'value', 
                        'measurement_flag', 'quality_flag', 'source_flag', 'observation_time'
                    ])
                    
                # Filter for temperature and precipitation
                year_data = year_data[year_data['element'].isin(['TMAX', 'TMIN', 'PRCP'])]
                year_data['year'] = year
                
                climate_data.append(year_data)
                self.cache_data(year_data, cache_key)
                
                self.logger.info(f"Downloaded GHCN data for {year}")
                
            except Exception as e:
                self.logger.error(f"Failed to download GHCN data for {year}: {e}")
                
        if climate_data:
            return pd.concat(climate_data, ignore_index=True)
        else:
            return pd.DataFrame()


class ClimateDataProcessor:
    """Process and clean climate data"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def process_era5_data(self, datasets: Dict[str, xr.Dataset], 
                         grid_points: pd.DataFrame) -> pd.DataFrame:
        """Process ERA5 datasets and extract to grid points"""
        
        if not datasets or grid_points.empty:
            return pd.DataFrame()
            
        all_features = []
        
        for variable, dataset in datasets.items():
            self.logger.info(f"Processing ERA5 {variable}")
            
            # Extract to grid points
            features = self._extract_era5_to_points(dataset, grid_points, variable)
            all_features.append(features)
            
        # Combine all variables
        if all_features:
            result = all_features[0]
            for features in all_features[1:]:
                result = result.merge(features, on=['geo_id', 'latitude', 'longitude'], how='outer')
            return result
        else:
            return pd.DataFrame()
            
    def _extract_era5_to_points(self, dataset: xr.Dataset, 
                               grid_points: pd.DataFrame, 
                               variable: str) -> pd.DataFrame:
        """Extract ERA5 data to specific grid points"""
        
        features = []
        
        for _, point in grid_points.iterrows():
            lat, lon = point['latitude'], point['longitude']
            
            try:
                # Select nearest grid point
                point_data = dataset.sel(latitude=lat, longitude=lon, method='nearest')
                
                # Calculate statistics
                if 'time' in point_data.dims:
                    # Time series statistics
                    data_values = point_data.values.flatten()
                    data_values = data_values[~np.isnan(data_values)]
                    
                    if len(data_values) > 0:
                        stats = {
                            'geo_id': point.get('geo_id'),
                            'latitude': lat,
                            'longitude': lon,
                            f'{variable}_mean': np.mean(data_values),
                            f'{variable}_std': np.std(data_values),
                            f'{variable}_min': np.min(data_values),
                            f'{variable}_max': np.max(data_values),
                            f'{variable}_p25': np.percentile(data_values, 25),
                            f'{variable}_p75': np.percentile(data_values, 75)
                        }
                        
                        # Seasonal statistics if we have monthly data
                        if len(data_values) >= 12:
                            monthly_means = []
                            for month in range(12):
                                month_data = data_values[month::12]
                                if len(month_data) > 0:
                                    monthly_means.append(np.mean(month_data))
                                    
                            if len(monthly_means) == 12:
                                stats[f'{variable}_seasonality'] = np.std(monthly_means)
                                
                        features.append(stats)
                        
            except Exception as e:
                self.logger.warning(f"Failed to extract {variable} for point {lat}, {lon}: {e}")
                
        return pd.DataFrame(features)
        
    def calculate_climate_indices(self, climate_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate climate indices and derived variables"""
        
        if climate_data.empty:
            return climate_data
            
        result = climate_data.copy()
        
        # Temperature-based indices
        if 'temperature_mean' in result.columns:
            # Growing degree days (base 10°C)
            result['growing_degree_days'] = np.maximum(0, result['temperature_mean'] - 10)
            
            # Heat index approximation
            if 'humidity_mean' in result.columns:
                result['heat_index'] = self._calculate_heat_index(
                    result['temperature_mean'], 
                    result['humidity_mean']
                )
                
        # Precipitation-based indices
        if 'precipitation_mean' in result.columns:
            # Aridity index (simple approximation)
            if 'temperature_mean' in result.columns:
                result['aridity_index'] = result['precipitation_mean'] / (result['temperature_mean'] + 10)
                
        # Comfort indices
        comfort_vars = ['temperature_mean', 'humidity_mean', 'wind_speed_mean']
        if all(col in result.columns for col in comfort_vars):
            result['comfort_index'] = self._calculate_comfort_index(
                result['temperature_mean'],
                result['humidity_mean'], 
                result['wind_speed_mean']
            )
            
        return result
        
    def _calculate_heat_index(self, temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index"""
        # Convert to Fahrenheit for heat index formula
        temp_f = temp_c * 9/5 + 32
        
        # Heat index formula (approximation)
        hi = (-42.379 + 
              2.04901523 * temp_f + 
              10.14333127 * humidity - 
              0.22475541 * temp_f * humidity)
        
        # Convert back to Celsius
        return (hi - 32) * 5/9
        
    def _calculate_comfort_index(self, temp: pd.Series, humidity: pd.Series, 
                               wind_speed: pd.Series) -> pd.Series:
        """Calculate human comfort index"""
        # Simplified comfort index based on temperature, humidity, and wind
        
        # Optimal ranges
        temp_comfort = np.exp(-0.5 * ((temp - 22) / 5) ** 2)  # Optimal around 22°C
        humidity_comfort = np.exp(-0.5 * ((humidity - 50) / 20) ** 2)  # Optimal around 50%
        wind_comfort = np.exp(-0.5 * ((wind_speed - 2) / 3) ** 2)  # Light breeze optimal
        
        return temp_comfort * humidity_comfort * wind_comfort
        
    def add_temporal_features(self, climate_data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal climate features"""
        
        if 'year' not in climate_data.columns:
            return climate_data
            
        result = climate_data.copy()
        
        # Sort by location and year
        result = result.sort_values(['geo_id', 'year'])
        
        # Climate variables to process
        climate_vars = [col for col in result.columns 
                       if any(var in col for var in ['temperature', 'precipitation', 'humidity'])]
        
        for var in climate_vars:
            if var in result.columns:
                # 5-year moving average
                result[f'{var}_5y_avg'] = result.groupby('geo_id')[var].rolling(
                    window=5, center=True
                ).mean().reset_index(level=0, drop=True)
                
                # 10-year moving average  
                result[f'{var}_10y_avg'] = result.groupby('geo_id')[var].rolling(
                    window=10, center=True
                ).mean().reset_index(level=0, drop=True)
                
                # Linear trend over past 10 years
                result[f'{var}_10y_trend'] = result.groupby('geo_id')[var].rolling(
                    window=10
                ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
                ).reset_index(level=0, drop=True)
                
        return result