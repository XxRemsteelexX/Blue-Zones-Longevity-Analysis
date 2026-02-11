# Blue Zones Longevity Analysis

## Historical Trends (1960-2023) and Future Projections (2024-2100)

[![Countries](https://img.shields.io/badge/Countries-93-blue)](./data)
[![Years](https://img.shields.io/badge/Years-1960--2023-green)](./data/historical)
[![Data Points](https://img.shields.io/badge/Observations-5%2C952-informational)](./data/historical/merged_historical_panel.csv)

---

## Project Overview

This project analyzes **60+ years of real-world health and development data** for 93 countries to answer: How have Blue Zone countries changed over time, and what does the future look like?

Blue Zones are five geographic regions where people live measurably longer lives:

| Blue Zone Region | Country | Latest LE (country-level) |
|-----------------|---------|---------------------------|
| Okinawa | Japan | ~84 years |
| Sardinia | Italy | ~83 years |
| Ikaria | Greece | ~80 years |
| Nicoya Peninsula | Costa Rica | ~79 years |
| Loma Linda, CA | United States | ~78 years |

**Important limitation:** Blue Zones are specific *regions* within countries, not entire countries. All public API data is country-level. Japan's national life expectancy does not equal Okinawa's specifically. This analysis tracks "countries containing Blue Zones" because sub-national historical time series are not available from public APIs.

### Key Findings

1. **Blue Zone countries currently lead the global average by ~6.4 years** in life expectancy
2. **The gap is narrowing** -- from ~10 years in the 1960s to ~6.4 years today (global convergence)
3. **Global convergence confirmed** -- the standard deviation of life expectancy across countries decreased from 11.4 to 6.8 years
4. **All 5 Blue Zone countries now have 64 years of complete life expectancy data** (the previous version had data for only 1 of 5)

---

## Data Sources

All data is real -- no synthetic or fake data used.

| Source | What | Coverage |
|--------|------|----------|
| **World Bank API** | Life expectancy, GDP, physicians, air quality, urbanization, health expenditure, death rate, forest area, population | 93 countries, 1960-2023 |
| **WHO Global Health Observatory** | Infant mortality, maternal mortality, life expectancy (cross-validation) | 93 countries, variable years |
| **Trend Extrapolation** | Life expectancy projections (Medium/High/Low scenarios) | 93 countries, 2024-2100 |

---

## Project Structure

```
Blue-Zones-Longevity-Analysis/
│
├── README.md
├── Blue_Zones_Historical_Projection_Report.md     # Full analysis report
├── Blue_Zones_Real_World_Analysis_Report.md        # Original cross-sectional report
├── REAL_DATA_RESEARCH_SUMMARY.md                   # Research summary
│
├── historical_data_collector.py                    # Pulls 1960-2023 data from WB + WHO APIs
├── un_projections_collector.py                     # Generates projections to 2100
├── historical_trend_analysis.py                    # Convergence and trend analysis
├── projection_visualizer.py                        # Static matplotlib figures
├── generate_report.py                              # Markdown report generator
├── verify_data_quality.py                          # Automated data quality checks
├── blue_zones_dashboard.py                         # Interactive Streamlit dashboard
│
├── data/
│   ├── historical/
│   │   ├── merged_historical_panel.csv             # 5,952 rows -- main dataset
│   │   ├── wb_historical_raw.csv                   # Raw World Bank data
│   │   └── who_historical_raw.csv                  # Raw WHO data
│   └── projections/
│       └── un_life_expectancy_projections.csv      # 7,161 rows -- projections to 2100
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb                   # Dataset overview and completeness
│   ├── 02_Historical_Trends.ipynb                  # BZ vs global trends over time
│   ├── 03_Convergence_Analysis.ipynb               # Sigma and beta convergence
│   ├── 04_Country_Deep_Dives.ipynb                 # Individual BZ country profiles
│   └── 05_Projections_Analysis.ipynb               # Future projections to 2100
│
└── outputs/
    ├── analysis/
    │   ├── blue_zone_vs_global.csv                 # BZ avg vs global avg per year
    │   ├── sigma_convergence.csv                   # Global LE spread per year
    │   ├── beta_convergence.csv                    # Decade-level convergence
    │   ├── blue_zone_country_profiles.csv          # BZ countries vs regional peers
    │   └── decade_improvements.csv                 # LE gains per decade by group
    └── figures/                                    # 25 PNG charts
```

---

## Getting Started

### Prerequisites

```bash
python --version  # Python 3.8+ required
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn requests streamlit
```

### Reproduce the Analysis

```bash
# Clone the repository
git clone https://github.com/XxRemsteelexX/Blue-Zones-Longevity-Analysis.git
cd Blue-Zones-Longevity-Analysis

# Step 1: Collect historical data (calls World Bank + WHO APIs)
python historical_data_collector.py

# Step 2: Generate projections
python un_projections_collector.py

# Step 3: Run trend analysis
python historical_trend_analysis.py

# Step 4: Generate static figures
python projection_visualizer.py

# Step 5: Generate report
python generate_report.py

# Step 6: Verify data quality
python verify_data_quality.py
```

### Run the Interactive Dashboard

```bash
streamlit run blue_zones_dashboard.py
```

The dashboard includes:
- Overview with key metrics
- Historical trends (BZ countries vs global average, 1960-2023)
- Convergence analysis (sigma and beta convergence)
- Country deep dives (each BZ country vs regional peers)
- Future projections (2024-2100 with uncertainty bands)
- Data explorer (filter, download, correlation matrix)

### Run the Notebooks

All notebooks are pre-executed with outputs embedded. To re-run:

```bash
jupyter notebook notebooks/
```

---

## Analysis Methods

| Method | Question Answered |
|--------|-------------------|
| **Gap Analysis** | How far ahead are Blue Zone countries? (BZ avg minus global avg per year) |
| **Sigma Convergence** | Is the world becoming more equal? (SD of LE across countries over time) |
| **Beta Convergence** | Do lagging countries catch up faster? (initial LE vs subsequent gains) |
| **Decade Improvements** | How many years of LE gained per decade by group? |
| **Regional Peer Comparison** | How does each BZ country compare to its regional neighbors? |
| **Trend Extrapolation** | What do projections to 2100 look like? (logistic dampening, uncertainty bands) |

---

## Results Summary

### Blue Zone Gap Over Time

| Decade | BZ Country Avg | Global Avg | Gap |
|--------|---------------|------------|-----|
| 1960s  | ~68 years     | ~58 years  | ~10 |
| 1980s  | ~75 years     | ~65 years  | ~10 |
| 2000s  | ~79 years     | ~71 years  | ~8  |
| 2020s  | ~82 years     | ~75 years  | ~6  |

The world is converging toward Blue Zone levels. The gap narrowed by ~4 years over 60 years, driven by faster improvements in developing countries.

### Data Quality

- 33 automated checks passed, 1 warning, 0 failures
- All 5 Blue Zone countries: 100% life expectancy coverage (64 years each)
- Life expectancy values all within expected range (25-95 years)
- No impossible year-to-year jumps detected

---

## Limitations

1. **Country-level vs region-level:** Blue Zones are specific communities, not countries. Okinawa differs from Japan's national average.
2. **Data gaps:** Some indicators (PM2.5, health expenditure) only available from ~2000 onward.
3. **Projection uncertainty:** Trend extrapolations are model outputs, not predictions. The UN Population Division API was unavailable at collection time.
4. **No causal claims:** All analysis is observational. Correlations do not prove causation.
5. **Sample bias:** The 93-country sample skews toward larger countries with better statistical infrastructure.

---

## Acknowledgments

- **Dan Buettner** and National Geographic for Blue Zones identification
- **World Bank** for socioeconomic indicator APIs
- **World Health Organization** for global health statistics
- **UN Population Division** for demographic methodology

---

*Last Updated: February 2026*
