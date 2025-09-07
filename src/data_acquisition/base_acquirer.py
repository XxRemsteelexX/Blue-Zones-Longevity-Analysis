"""
Base classes for data acquisition from various sources
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
import time
import json
from pathlib import Path
from abc import ABC, abstractmethod
import urllib.parse
from datetime import datetime, timedelta


class BaseDataAcquirer(ABC):
    """Base class for data acquisition"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.session = requests.Session()
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """Fetch data from source"""
        pass
        
    def setup_session(self, headers: Dict[str, str] = None) -> None:
        """Setup requests session with headers"""
        if headers:
            self.session.headers.update(headers)
            
    def cache_data(self, data: Union[pd.DataFrame, Dict], cache_key: str) -> None:
        """Cache data to disk"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if isinstance(data, pd.DataFrame):
            data.to_json(cache_file, orient='records', date_format='iso')
        else:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        self.logger.info(f"Cached data to {cache_file}")
        
    def load_cached_data(self, cache_key: str, max_age_hours: int = 24) -> Optional[Union[pd.DataFrame, Dict]]:
        """Load cached data if recent enough"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > timedelta(hours=max_age_hours):
            self.logger.info(f"Cache expired for {cache_key}")
            return None
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
            # Try to convert to DataFrame if it looks like tabular data
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                return pd.DataFrame(data)
            else:
                return data
                
        except Exception as e:
            self.logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None


class APIAcquirer(BaseDataAcquirer):
    """Base class for API-based data acquisition"""
    
    def __init__(self, base_url: str, config: Dict[str, Any], 
                 api_key: str = None, rate_limit: float = 1.0,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
    def make_request(self, endpoint: str, params: Dict[str, Any] = None,
                    method: str = 'GET', data: Dict[str, Any] = None) -> requests.Response:
        """Make API request with rate limiting"""
        
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
            
        # Build URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Add API key if configured
        if self.api_key:
            if params is None:
                params = {}
            params['api_key'] = self.api_key
            
        try:
            response = self.session.request(method, url, params=params, json=data)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
            
    def paginated_request(self, endpoint: str, params: Dict[str, Any] = None,
                         page_param: str = 'page', per_page_param: str = 'per_page',
                         per_page: int = 100, max_pages: int = None) -> List[Dict]:
        """Handle paginated API requests"""
        
        if params is None:
            params = {}
            
        params[per_page_param] = per_page
        
        all_data = []
        page = 1
        
        while True:
            if max_pages and page > max_pages:
                break
                
            params[page_param] = page
            
            try:
                response = self.make_request(endpoint, params)
                data = response.json()
                
                # Handle different response formats
                if isinstance(data, list):
                    page_data = data
                elif isinstance(data, dict):
                    # Common patterns
                    for key in ['data', 'results', 'items']:
                        if key in data:
                            page_data = data[key]
                            break
                    else:
                        page_data = [data]  # Single item response
                else:
                    break
                    
                if not page_data:
                    break
                    
                all_data.extend(page_data)
                self.logger.info(f"Fetched page {page}, {len(page_data)} items")
                page += 1
                
            except Exception as e:
                self.logger.error(f"Failed to fetch page {page}: {e}")
                break
                
        return all_data


class WebScraper(BaseDataAcquirer):
    """Base class for web scraping"""
    
    def __init__(self, config: Dict[str, Any], 
                 user_agent: str = None,
                 delay: float = 1.0,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.delay = delay
        self.last_request_time = 0
        
        # Setup user agent
        if user_agent is None:
            user_agent = "Blue Zones Research Bot 1.0 (Educational/Research Purpose)"
            
        self.setup_session({'User-Agent': user_agent})
        
    def fetch_page(self, url: str, params: Dict[str, Any] = None) -> requests.Response:
        """Fetch web page with rate limiting"""
        
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
            
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Web request failed: {e}")
            raise
            
    def parse_html(self, html_content: str) -> 'BeautifulSoup':
        """Parse HTML content"""
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(html_content, 'html.parser')
        except ImportError:
            raise ImportError("BeautifulSoup4 required for web scraping: pip install beautifulsoup4")


class FileDownloader(BaseDataAcquirer):
    """Base class for file downloads"""
    
    def __init__(self, config: Dict[str, Any], 
                 download_dir: str = "data/downloads",
                 chunk_size: int = 8192,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        
    def download_file(self, url: str, filename: str = None, 
                     force_download: bool = False) -> Path:
        """Download file from URL"""
        
        if filename is None:
            filename = Path(urllib.parse.urlparse(url).path).name
            
        filepath = self.download_dir / filename
        
        # Check if file already exists
        if filepath.exists() and not force_download:
            self.logger.info(f"File {filename} already exists, skipping download")
            return filepath
            
        self.logger.info(f"Downloading {url} to {filepath}")
        
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = downloaded / total_size * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                self.logger.info(f"Download progress: {progress:.1f}%")
                                
            self.logger.info(f"Download completed: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            if filepath.exists():
                filepath.unlink()  # Clean up partial file
            raise


class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_coordinates(df: pd.DataFrame, lat_col: str = 'latitude', 
                           lon_col: str = 'longitude') -> pd.DataFrame:
        """Validate and clean coordinate data"""
        original_len = len(df)
        
        # Remove invalid coordinates
        df = df[
            (df[lat_col].between(-90, 90)) & 
            (df[lon_col].between(-180, 180)) &
            df[lat_col].notna() & 
            df[lon_col].notna()
        ]
        
        removed = original_len - len(df)
        if removed > 0:
            print(f"Removed {removed} rows with invalid coordinates")
            
        return df
        
    @staticmethod
    def validate_temporal_data(df: pd.DataFrame, date_col: str = 'date',
                              start_year: int = 1990, end_year: int = 2024) -> pd.DataFrame:
        """Validate and clean temporal data"""
        original_len = len(df)
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
        # Filter by date range
        df = df[
            (df[date_col].dt.year >= start_year) & 
            (df[date_col].dt.year <= end_year) &
            df[date_col].notna()
        ]
        
        removed = original_len - len(df)
        if removed > 0:
            print(f"Removed {removed} rows with invalid dates")
            
        return df
        
    @staticmethod
    def validate_numeric_ranges(df: pd.DataFrame, 
                               range_checks: Dict[str, tuple]) -> pd.DataFrame:
        """Validate numeric columns against expected ranges"""
        original_len = len(df)
        
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                df = df[df[col].between(min_val, max_val, na_action='ignore')]
                
        removed = original_len - len(df)
        if removed > 0:
            print(f"Removed {removed} rows with values outside expected ranges")
            
        return df


class DataCleaner:
    """Utility class for data cleaning"""
    
    @staticmethod
    def standardize_country_names(df: pd.DataFrame, country_col: str = 'country') -> pd.DataFrame:
        """Standardize country names"""
        country_mapping = {
            'USA': 'United States',
            'US': 'United States', 
            'United States of America': 'United States',
            'UK': 'United Kingdom',
            'Great Britain': 'United Kingdom',
            'Russia': 'Russian Federation',
            'South Korea': 'Korea, Republic of',
            'North Korea': 'Korea, Democratic People\'s Republic of'
        }
        
        df[country_col] = df[country_col].replace(country_mapping)
        return df
        
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method"""
        original_len = len(df)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df = df[df[col].between(lower, upper, na_action='ignore')]
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
                
        removed = original_len - len(df)
        if removed > 0:
            print(f"Removed {removed} outliers")
            
        return df
        
    @staticmethod
    def fill_missing_values(df: pd.DataFrame, strategies: Dict[str, str]) -> pd.DataFrame:
        """Fill missing values using different strategies"""
        for col, strategy in strategies.items():
            if col not in df.columns:
                continue
                
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)
            elif strategy == 'forward':
                df[col] = df[col].fillna(method='ffill')
            elif strategy == 'backward':
                df[col] = df[col].fillna(method='bfill')
            elif strategy == 'zero':
                df[col] = df[col].fillna(0)
                
        return df