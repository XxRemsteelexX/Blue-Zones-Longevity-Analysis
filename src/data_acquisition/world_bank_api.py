"""
World Bank API data acquirer for socioeconomic and life expectancy data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from .base_acquirer import APIAcquirer


class WorldBankAPI(APIAcquirer):
    """World Bank Open Data API client"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(
            base_url='https://api.worldbank.org/v2',
            config=config,
            rate_limit=0.5,  # World Bank allows 2 requests per second
            logger=logger
        )
        
        # World Bank indicators
        self.indicators = {
            'life_expectancy': 'SP.DYN.LE00.IN',
            'gdp_per_capita': 'NY.GDP.PCAP.PP.CD',
            'poverty_headcount': 'SI.POV.DDAY',
            'education_years': 'BAR.SCHL.15UP',
            'pm25': 'EN.ATM.PM25.MC.M3',
            'population': 'SP.POP.TOTL',
            'population_density': 'EN.POP.DNST',
            'urban_population': 'SP.URB.TOTL.IN.ZS',
            'internet_users': 'IT.NET.USER.ZS',
            'electricity_access': 'EG.ELC.ACCS.ZS',
            'co2_emissions': 'EN.ATM.CO2E.PC',
            'health_expenditure': 'SH.XPD.CHEX.GD.ZS',
            'hospital_beds': 'SH.MED.BEDS.ZS'
        }
        
    def fetch_data(self, indicators: List[str] = None, 
                  countries: List[str] = None,
                  start_year: int = 1990, 
                  end_year: int = 2023) -> pd.DataFrame:
        """
        Fetch World Bank data for specified indicators and countries
        
        Args:
            indicators: List of indicator keys from self.indicators
            countries: List of country codes (ISO3) or 'all' for all countries
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            DataFrame with World Bank data
        """
        if indicators is None:
            indicators = list(self.indicators.keys())
            
        if countries is None:
            countries = 'all'
            
        all_data = []
        
        for indicator_key in indicators:
            if indicator_key not in self.indicators:
                self.logger.warning(f"Unknown indicator: {indicator_key}")
                continue
                
            indicator_code = self.indicators[indicator_key]
            self.logger.info(f"Fetching {indicator_key} ({indicator_code})")
            
            # Check cache first
            cache_key = f"wb_{indicator_code}_{start_year}_{end_year}"
            cached_data = self.load_cached_data(cache_key, max_age_hours=24)
            
            if cached_data is not None:
                self.logger.info(f"Using cached data for {indicator_key}")
                indicator_data = cached_data
            else:
                # Fetch from API
                indicator_data = self._fetch_indicator_data(
                    indicator_code, countries, start_year, end_year
                )
                
                # Cache the data
                self.cache_data(indicator_data, cache_key)
            
            if not indicator_data.empty:
                indicator_data['indicator'] = indicator_key
                all_data.append(indicator_data)
                
        if not all_data:
            self.logger.warning("No data fetched")
            return pd.DataFrame()
            
        # Combine all indicators
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Reshape data
        final_data = self._reshape_data(combined_data)
        
        return final_data
        
    def _fetch_indicator_data(self, indicator_code: str, countries: str,
                             start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch data for a single indicator"""
        
        if isinstance(countries, list):
            countries = ';'.join(countries)
        elif countries == 'all':
            countries = 'all'
            
        endpoint = f"country/{countries}/indicator/{indicator_code}"
        params = {
            'date': f'{start_year}:{end_year}',
            'format': 'json',
            'per_page': 1000
        }
        
        try:
            # Make paginated request
            all_data = self.paginated_request(
                endpoint=endpoint,
                params=params,
                page_param='page',
                per_page_param='per_page'
            )
            
            if not all_data:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Clean and standardize
            df = self._clean_wb_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {indicator_code}: {e}")
            return pd.DataFrame()
            
    def _clean_wb_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean World Bank API response data"""
        
        if df.empty:
            return df
            
        # Standardize column names
        column_mapping = {
            'date': 'year',
            'value': 'value',
            'country': 'country_info',
            'countryiso3code': 'country_code'
        }
        
        # Extract country information
        if 'country' in df.columns and isinstance(df['country'].iloc[0], dict):
            df['country_name'] = df['country'].apply(lambda x: x.get('value', '') if isinstance(x, dict) else str(x))
            df['country_code'] = df['country'].apply(lambda x: x.get('id', '') if isinstance(x, dict) else '')
        
        # Clean year column
        if 'date' in df.columns:
            df['year'] = pd.to_numeric(df['date'], errors='coerce')
            
        # Clean value column
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
        # Remove rows with missing values
        df = df.dropna(subset=['year', 'value'])
        
        # Select relevant columns
        keep_cols = ['country_name', 'country_code', 'year', 'value']
        available_cols = [col for col in keep_cols if col in df.columns]
        
        return df[available_cols]
        
    def _reshape_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reshape long format data to wide format"""
        
        if df.empty:
            return df
            
        # Pivot data to have indicators as columns
        pivot_data = df.pivot_table(
            index=['country_name', 'country_code', 'year'],
            columns='indicator',
            values='value',
            aggfunc='first'  # In case of duplicates
        ).reset_index()
        
        # Flatten column names
        pivot_data.columns.name = None
        
        return pivot_data
        
    def get_country_metadata(self) -> pd.DataFrame:
        """Fetch country metadata including regions and income levels"""
        
        cache_key = "wb_countries_metadata"
        cached_data = self.load_cached_data(cache_key, max_age_hours=168)  # Cache for a week
        
        if cached_data is not None:
            return cached_data
            
        endpoint = "country"
        params = {
            'format': 'json',
            'per_page': 500
        }
        
        try:
            countries = self.paginated_request(endpoint, params)
            
            if not countries:
                return pd.DataFrame()
                
            df = pd.DataFrame(countries)
            
            # Extract nested information
            df['country_name'] = df['name']
            df['country_code'] = df['id']
            df['region'] = df['region'].apply(lambda x: x.get('value', '') if isinstance(x, dict) else '')
            df['income_level'] = df['incomeLevel'].apply(lambda x: x.get('value', '') if isinstance(x, dict) else '')
            df['capital_city'] = df.get('capitalCity', '')
            df['longitude'] = pd.to_numeric(df.get('longitude', np.nan), errors='coerce')
            df['latitude'] = pd.to_numeric(df.get('latitude', np.nan), errors='coerce')
            
            # Select relevant columns
            metadata_cols = [
                'country_name', 'country_code', 'region', 'income_level',
                'capital_city', 'longitude', 'latitude'
            ]
            
            result = df[metadata_cols]
            
            # Cache the data
            self.cache_data(result, cache_key)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to fetch country metadata: {e}")
            return pd.DataFrame()
            
    def get_available_indicators(self) -> pd.DataFrame:
        """Get list of available indicators"""
        
        endpoint = "indicator"
        params = {
            'format': 'json',
            'per_page': 1000
        }
        
        try:
            indicators = self.paginated_request(endpoint, params, max_pages=20)
            
            if not indicators:
                return pd.DataFrame()
                
            df = pd.DataFrame(indicators)
            
            # Clean indicator information
            df['indicator_id'] = df['id']
            df['indicator_name'] = df['name']
            df['source'] = df['source'].apply(lambda x: x.get('value', '') if isinstance(x, dict) else '')
            df['topic'] = df['topics'].apply(
                lambda x: ', '.join([t.get('value', '') for t in x]) if isinstance(x, list) else ''
            )
            
            return df[['indicator_id', 'indicator_name', 'source', 'topic']]
            
        except Exception as e:
            self.logger.error(f"Failed to fetch indicators: {e}")
            return pd.DataFrame()
            
    def fetch_subnational_data(self, country_code: str, indicators: List[str]) -> pd.DataFrame:
        """Fetch subnational data where available"""
        
        # World Bank has limited subnational data
        # This is a placeholder for future enhancement
        self.logger.info("Subnational data fetching not yet implemented")
        return pd.DataFrame()


class WorldBankDataProcessor:
    """Process and clean World Bank data"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def process_data(self, raw_data: pd.DataFrame, 
                    country_metadata: pd.DataFrame = None) -> pd.DataFrame:
        """Process raw World Bank data"""
        
        if raw_data.empty:
            return raw_data
            
        # Merge with country metadata if provided
        if country_metadata is not None and not country_metadata.empty:
            processed_data = raw_data.merge(
                country_metadata, 
                on=['country_code'], 
                how='left',
                suffixes=('', '_meta')
            )
        else:
            processed_data = raw_data.copy()
            
        # Data quality improvements
        processed_data = self._interpolate_missing_years(processed_data)
        processed_data = self._add_derived_indicators(processed_data)
        processed_data = self._filter_outliers(processed_data)
        
        return processed_data
        
    def _interpolate_missing_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing years for time series data"""
        
        if 'year' not in df.columns:
            return df
            
        interpolated_data = []
        
        for country in df['country_code'].unique():
            country_data = df[df['country_code'] == country].sort_values('year')
            
            if len(country_data) < 2:
                interpolated_data.append(country_data)
                continue
                
            # Create complete year range
            year_range = range(int(country_data['year'].min()), 
                             int(country_data['year'].max()) + 1)
            
            # Reindex and interpolate
            country_data = country_data.set_index('year').reindex(year_range)
            
            # Forward fill country metadata
            for col in ['country_name', 'country_code', 'region', 'income_level']:
                if col in country_data.columns:
                    country_data[col] = country_data[col].fillna(method='ffill')
                    country_data[col] = country_data[col].fillna(method='bfill')
                    
            # Interpolate numeric indicators
            numeric_cols = country_data.select_dtypes(include=[np.number]).columns
            country_data[numeric_cols] = country_data[numeric_cols].interpolate(method='linear')
            
            country_data = country_data.reset_index()
            interpolated_data.append(country_data)
            
        return pd.concat(interpolated_data, ignore_index=True)
        
    def _add_derived_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived indicators"""
        
        # GDP per capita growth rate
        if 'gdp_per_capita' in df.columns:
            df['gdp_growth_rate'] = df.groupby('country_code')['gdp_per_capita'].pct_change()
            
        # Life expectancy trend
        if 'life_expectancy' in df.columns:
            df['life_expectancy_trend'] = df.groupby('country_code')['life_expectancy'].diff()
            
        # Development index (normalized composite)
        development_indicators = ['gdp_per_capita', 'education_years', 'life_expectancy']
        available_indicators = [col for col in development_indicators if col in df.columns]
        
        if len(available_indicators) >= 2:
            # Normalize indicators
            normalized_data = df[available_indicators].copy()
            for col in available_indicators:
                normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                                     (normalized_data[col].max() - normalized_data[col].min())
            
            df['development_index'] = normalized_data.mean(axis=1)
            
        return df
        
    def _filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter extreme outliers"""
        
        outlier_bounds = {
            'life_expectancy': (30, 90),
            'gdp_per_capita': (100, 200000),
            'pm25': (0, 200),
            'population_density': (0, 30000),
            'urban_population': (0, 100)
        }
        
        for col, (min_val, max_val) in outlier_bounds.items():
            if col in df.columns:
                original_len = len(df)
                df = df[(df[col].between(min_val, max_val)) | df[col].isna()]
                removed = original_len - len(df)
                
                if removed > 0:
                    self.logger.info(f"Removed {removed} outliers from {col}")
                    
        return df
        
    def create_country_profiles(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Create country profiles with latest indicators"""
        
        if df.empty or 'year' not in df.columns:
            return {}
            
        profiles = {}
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        for _, row in latest_data.iterrows():
            country_code = row.get('country_code')
            if not country_code:
                continue
                
            profile = {
                'country_name': row.get('country_name', ''),
                'latest_year': latest_year,
                'indicators': {}
            }
            
            # Add all numeric indicators
            for col in df.columns:
                if col in ['country_name', 'country_code', 'year']:
                    continue
                    
                value = row.get(col)
                if pd.notna(value) and isinstance(value, (int, float)):
                    profile['indicators'][col] = value
                    
            profiles[country_code] = profile
            
        return profiles