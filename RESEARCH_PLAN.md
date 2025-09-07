# Blue Zones Quantified: Research Plan
*Pre-registered analysis plan - Timestamp: 2024-08-26*

## Research Questions

**RQ1 (Causal)**: Do "Blue Zone" lifestyle/ecological proxies measurably increase life expectancy after controlling for place and time effects?

**RQ2 (Heterogeneity)**: Which mechanisms dominate (walkability, greenspace, social proximity, diet access, climate comfort), and where do they matter most?

**RQ3 (Policy)**: If a region increases amenity X by Δ, what's the expected life expectancy gain in 5-10 years, with uncertainty bounds?

## Outcomes (Pre-specified)
- **Primary**: Life expectancy at birth
- **Secondary**: Life expectancy at 60 (LE60)  
- **Secondary**: Healthy Life Expectancy (HALE)
- **Mechanism**: CVD mortality rate (if available)

## Identification Strategy

### Primary: Panel Fixed Effects + Event Studies
- Unit: 5km grid cells globally
- Time: Annual panel 2000-2021
- Treatment: Standardized lifestyle proxies (z-scores)
- Model: LE_it = α_i + γ_t + β·Lifestyle_it + θ·Controls_it + ε_it
- Key assumption: Parallel trends (testable)

### Secondary: Geographic Regression Discontinuity
- Borders where lifestyle proxies change sharply
- Compare cells just inside/outside treatment boundaries

### Tertiary: Causal Machine Learning
- Double ML / Causal Forests for heterogeneous treatment effects
- Control for confounders flexibly with ML

## Treatment Variables (Lifestyle Proxies)
1. **Greenspace**: NDVI, park area per capita
2. **Food Access**: Market density, fresh food POIs  
3. **Walkability**: Road connectivity, intersection density
4. **Social Infrastructure**: Religious sites, community centers
5. **Climate Comfort**: Temperature variability, heat stress days

## Control Variables  
- Population density, age structure
- GDP per capita, education proxy
- Climate (mean temp, precipitation, heat days)
- Elevation, slope
- PM2.5 air pollution
- Spatial lags (spillover controls)

## Robustness Checks
1. **Spatial**: Moran's I test, spatial error models
2. **Temporal**: Lagged treatments (1,3,5 years)
3. **Placebo**: Lead treatments, shuffled years
4. **Alternative**: Different grid sizes (5km vs 10km)
5. **Negative controls**: Outcomes that shouldn't respond
6. **Sensitivity**: Rosenbaum bounds, Oster delta

## Expected Results Format
- "1 SD increase in walkability leads to +0.4 years LE (95% CI: 0.2-0.6)"
- "Effects 2x larger in temperate climates" 
- "Policy simulation: 5% road to pedestrian conversion leads to +0.15-0.30 years in 10 years"

## Robustness Threshold
- Effects survive spatial clustering
- No pre-trends in event studies
- Rosenbaum Gamma > 1.5 (moderate unobserved confounder robustness)
- Consistent across outcome measures

## Target Journals
- Primary: International Journal of Epidemiology
- Alternative: PLOS Global Public Health, Health & Place
- Reach: Nature Cities (if mechanisms strong)