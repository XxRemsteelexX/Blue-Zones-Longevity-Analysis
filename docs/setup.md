# Blue Zones Quantified - Setup Guide

## Environment Setup

### 1. Python Environment
```bash
# Create virtual environment
python -m venv blue_zones_env
source blue_zones_env/bin/activate  # Linux/Mac
# or
blue_zones_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Optional: Google Earth Engine Setup
```bash
# Install Earth Engine API
pip install earthengine-api

# Authenticate (requires Google account)
earthengine authenticate
```

## Data Acquisition

### Core Datasets Required

1. **Life Expectancy**: IHME Global Burden of Disease
   - Download subnational estimates 1990-2021
   - Place in `data/raw/life_expectancy/`

2. **Climate Data**: ERA5 Reanalysis  
   - Temperature, precipitation, humidity (monthly, 1990-2021)
   - Use Copernicus CDS API
   - Place in `data/raw/climate/`

3. **Air Quality**: Van Donkelaar PM2.5
   - Annual global grids 1990-2021
   - Download from NASA
   - Place in `data/raw/air_quality/`

4. **Population**: WorldPop
   - 100m resolution population grids
   - Download for all countries
   - Place in `data/raw/population/`

5. **Elevation**: NASA SRTM
   - 30m DEM (or aggregated versions)
   - Download global coverage
   - Place in `data/raw/elevation/`

6. **Socioeconomic**: World Bank Open Data
   - GDP PPP, poverty, education indicators
   - API access recommended
   - Cache in `data/raw/socioeconomic/`

7. **Night Lights**: VIIRS
   - Annual composites 2012-2021
   - Download from NOAA
   - Place in `data/raw/nightlights/`

### API Setup

Create `config/secrets.yaml` (not tracked in git):

```yaml
apis:
  world_bank:
    base_url: "https://api.worldbank.org/v2/"
  
  copernicus_cds:
    api_key: "YOUR_CDS_API_KEY"
    
  google_earth_engine:
    service_account: "path/to/service_account.json"
```

## Project Initialization

Run the setup script to create initial grid and validate data:

```bash
python scripts/00_setup_project.py
```

This will:
- Create the global 5km grid system
- Validate data directory structure  
- Run basic data quality checks
- Generate initial configuration

## Development Workflow

1. **Phase 1**: Grid + ETL
   ```bash
   python scripts/01_build_grid.py
   python scripts/02_etl_climate.py
   python scripts/03_etl_demographics.py
   ```

2. **Phase 2**: Feature Engineering
   ```bash
   python scripts/04_engineer_features.py
   python scripts/05_add_life_expectancy.py
   ```

3. **Phase 3**: Analysis
   ```bash
   python scripts/06_matched_comparison.py
   python scripts/07_blue_zone_classifier.py
   ```

4. **Phase 4**: Modeling
   ```bash
   python scripts/08_forecasting_models.py
   python scripts/09_uncertainty_analysis.py
   ```

5. **Phase 5**: Visualization
   ```bash
   python scripts/10_create_maps.py
   python scripts/11_build_dashboard.py
   ```

## Testing

Run unit tests:
```bash
pytest tests/
```

## Troubleshooting

### Memory Issues
- Use `dask` for large raster operations
- Process data in chunks for global grids
- Consider reducing spatial resolution for testing

### Data Access Issues  
- Check API keys and authentication
- Verify data URLs are still active
- Use alternative data sources if needed

### Performance Optimization
- Use multiprocessing for embarrassingly parallel tasks
- Optimize raster I/O with appropriate chunk sizes
- Cache intermediate results

## Contributing

1. Create feature branch from main
2. Add tests for new functionality  
3. Update documentation
4. Submit pull request with clear description