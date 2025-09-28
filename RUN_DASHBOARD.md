# 🌿 Blue Zones Longevity Dashboard - Running Instructions

## Overview
This professional interactive dashboard analyzes longevity patterns in Blue Zones vs. global populations using real-world health, environmental, and economic data.

## Prerequisites
- Python 3.8+ with virtual environment activated
- Required packages: `streamlit`, `pandas`, `plotly`, `numpy`
- Blue Zones dataset CSV files in the project directory

## Quick Start

### 1. Activate Virtual Environment
```bash
# Navigate to project directory
cd ~/Desktop/PULLED_PROJECTS/Blue-Zones-Longevity-Analysis

# Activate virtual environment
source .venv/bin/activate
```

### 2. Install Required Packages (if not already installed)
```bash
pip install streamlit plotly pandas numpy
```

### 3. Run the Dashboard
```bash
streamlit run blue_zones_dashboard.py
```

### 4. Access the Dashboard
- The dashboard will automatically open in your default browser
- URL: http://localhost:8501
- If it doesn't open automatically, click the URL shown in the terminal

## Dashboard Features

### 🔍 Interactive Controls
- **Zone Type Filter**: Choose between Blue Zones, Regular Zones, or both
- **Country Filter**: Select specific countries for focused analysis
- **Data Explorer**: Toggle between all zones or Blue Zones only

### 📈 Key Performance Indicators
- Blue Zones count and coverage
- Life expectancy comparisons
- Infant mortality rates
- Forest coverage analysis

### 🗺️ Visualizations
- **World Map**: Interactive global view of Blue Zones with life expectancy data
- **Longevity Comparison**: Box plots showing life expectancy distributions
- **Radar Chart**: Multi-dimensional health metrics comparison
- **Scatter Plot**: GDP vs Life Expectancy correlation
- **Correlation Heatmap**: Health and environmental factor relationships

### 📋 Data Explorer
- Sortable and filterable data table
- Blue Zones specific filtering
- Key metrics display

### 🔍 Insights Section
- Blue Zones advantages summary
- Statistical findings and correlations

## Technical Details

### Data Sources
- `real_world_blue_zones_comprehensive.csv` (primary)
- `real_world_analysis_results.csv` (fallback)

### Key Technologies
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation and analysis
- **Custom CSS**: Professional styling

### Performance Features
- Data caching for improved load times
- Responsive design for various screen sizes
- Professional color scheme and styling

## Troubleshooting

### Dashboard Won't Start
```bash
# Check if streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install --upgrade streamlit
```

### Data File Errors
- Ensure CSV files are in the same directory as `blue_zones_dashboard.py`
- Check file names match exactly:
  - `real_world_blue_zones_comprehensive.csv`
  - `real_world_analysis_results.csv`

### Browser Issues
- Try a different browser (Chrome, Firefox, Safari)
- Clear browser cache
- Use incognito/private browsing mode

### Port Conflicts
```bash
# Run on a different port
streamlit run blue_zones_dashboard.py --server.port 8502
```

## Advanced Usage

### Custom Configuration
Create a `.streamlit/config.toml` file for custom settings:
```toml
[server]
port = 8501
headless = false

[theme]
primaryColor = "#2E8B57"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F8FF"
```

### Running in Production
```bash
# For production deployment
streamlit run blue_zones_dashboard.py --server.headless true --server.port 8501
```

## Support

### Common Issues
1. **Virtual Environment**: Ensure the correct virtual environment is activated
2. **Dependencies**: All required packages must be installed in the active environment
3. **Data Files**: CSV files must be in the same directory as the dashboard script

### Getting Help
- Check the terminal for error messages
- Verify all prerequisites are met
- Ensure data files are properly formatted

## Next Steps

After running the dashboard:
1. Explore different filter combinations
2. Analyze the correlation patterns
3. Export insights for reports
4. Consider additional data sources for enhanced analysis

---

**Enjoy exploring the secrets of longevity through data! 🌿📊**
