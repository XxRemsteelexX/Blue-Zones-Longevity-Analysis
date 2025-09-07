#!/usr/bin/env python3
"""
Train life expectancy forecasting models with uncertainty quantification
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, load_config, load_intermediate_data, save_intermediate_data
from models.forecasting_models import LifeExpectancyForecaster


def main():
    """Train forecasting models and generate predictions"""
    
    # Setup
    logger = setup_logging("INFO")
    config = load_config("config/config.yaml")
    
    logger.info("Starting life expectancy forecasting model development")
    
    # Load data
    try:
        # Load combined features
        features = load_intermediate_data("combined_features", "features")
        logger.info(f"Loaded features: {len(features)} observations")
        
        # Load life expectancy with temporal data
        life_expectancy = load_intermediate_data("life_expectancy", "processed")
        logger.info(f"Loaded life expectancy: {len(life_expectancy)} observations")
        
    except Exception as e:
        logger.error(f"Could not load required data: {e}")
        return 1
    
    # Merge features with life expectancy data
    if 'year' in life_expectancy.columns:
        # Temporal analysis - merge on geo_id and year
        logger.info("Performing temporal forecasting analysis")
        
        # For this example, use most recent features with all years of life expectancy
        latest_features = features.copy()  # Assume features represent most recent state
        
        # Create full temporal dataset
        temporal_data = []
        for year in life_expectancy['year'].unique():
            year_le = life_expectancy[life_expectancy['year'] == year]
            year_data = latest_features.merge(year_le, on='geo_id', how='inner')
            temporal_data.append(year_data)
            
        if temporal_data:
            modeling_data = pd.concat(temporal_data, ignore_index=True)
        else:
            modeling_data = latest_features.merge(life_expectancy, on='geo_id', how='inner')
    else:
        # Cross-sectional analysis
        logger.info("Performing cross-sectional analysis (no temporal dimension)")
        modeling_data = features.merge(life_expectancy, on='geo_id', how='inner')
    
    logger.info(f"Modeling dataset: {len(modeling_data)} observations")
    
    if len(modeling_data) == 0:
        logger.error("No data available after merging features and life expectancy")
        return 1
    
    # Initialize forecaster
    forecaster = LifeExpectancyForecaster(config, logger)
    
    # Train forecasting models
    logger.info("Training ensemble of forecasting models")
    training_results = forecaster.train_forecasting_models(
        modeling_data, 
        target_col='val'  # Life expectancy column
    )
    
    if not training_results:
        logger.error("Forecasting model training failed")
        return 1
    
    # Generate forecasts
    forecast_years = config['temporal']['forecast_years']
    logger.info(f"Generating forecasts for years: {forecast_years}")
    
    # Create scenario forecasts
    scenario_forecasts = forecaster.create_scenario_forecasts(
        features,  # Use base features for forecasting
        forecast_years
    )
    
    # Process and combine scenario results
    all_forecasts = []
    for scenario, forecasts in scenario_forecasts.items():
        forecasts['scenario'] = scenario
        all_forecasts.append(forecasts)
        
    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
    
    # Analyze forecast uncertainty
    uncertainty_analysis = analyze_forecast_uncertainty(combined_forecasts, logger)
    
    # Save results
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trained forecaster
    with open(model_dir / "life_expectancy_forecaster.pkl", "wb") as f:
        pickle.dump(forecaster, f)
    
    # Save forecasts
    combined_forecasts.to_parquet(output_dir / "life_expectancy_forecasts.parquet")
    combined_forecasts.to_csv(output_dir / "life_expectancy_forecasts.csv", index=False)
    
    # Save training results
    with open(output_dir / "forecasting_training_results.json", "w") as f:
        json_results = convert_numpy_types(training_results)
        json.dump(json_results, f, indent=2)
    
    # Save uncertainty analysis
    with open(output_dir / "forecast_uncertainty_analysis.json", "w") as f:
        json_uncertainty = convert_numpy_types(uncertainty_analysis)
        json.dump(json_uncertainty, f, indent=2)
    
    # Create summary reports
    create_forecast_summaries(combined_forecasts, scenario_forecasts, logger, output_dir)
    
    # Create regional forecasts (if we have Blue Zone data)
    try:
        grid_df = load_intermediate_data("global_grid_5km", "processed")
        blue_zone_forecasts = analyze_blue_zone_forecasts(
            combined_forecasts, grid_df, logger
        )
        blue_zone_forecasts.to_csv(output_dir / "blue_zone_forecasts.csv", index=False)
    except Exception as e:
        logger.warning(f"Could not create Blue Zone forecast analysis: {e}")
    
    # Log key results
    logger.info("=== FORECASTING RESULTS ===")
    
    if 'lightgbm' in training_results:
        lgb_results = training_results['lightgbm']
        logger.info(f"LightGBM CV RMSE: {lgb_results.get('cv_rmse_mean', 'N/A'):.3f}")
        logger.info(f"LightGBM CV R²: {lgb_results.get('cv_r2_mean', 'N/A'):.3f}")
    
    for year in forecast_years:
        year_data = combined_forecasts[combined_forecasts['year'] == year]
        if not year_data.empty:
            baseline_data = year_data[year_data['scenario'] == 'baseline']
            if not baseline_data.empty:
                mean_pred = baseline_data['predicted_life_expectancy'].mean()
                mean_uncertainty = baseline_data['prediction_interval_width'].mean()
                logger.info(f"{year} forecast: {mean_pred:.1f} years (±{mean_uncertainty/2:.1f})")
    
    logger.info("Forecasting model development completed successfully")
    return 0


def analyze_forecast_uncertainty(forecasts: pd.DataFrame, 
                               logger: logging.Logger) -> Dict[str, Any]:
    """Analyze forecast uncertainty patterns"""
    
    uncertainty_analysis = {}
    
    # Overall uncertainty statistics
    uncertainty_analysis['overall'] = {
        'mean_interval_width': forecasts['prediction_interval_width'].mean(),
        'median_interval_width': forecasts['prediction_interval_width'].median(),
        'max_interval_width': forecasts['prediction_interval_width'].max(),
        'min_interval_width': forecasts['prediction_interval_width'].min()
    }
    
    # Uncertainty by year
    yearly_uncertainty = forecasts.groupby('year')['prediction_interval_width'].agg([
        'mean', 'median', 'std'
    ]).to_dict('index')
    uncertainty_analysis['by_year'] = yearly_uncertainty
    
    # Uncertainty by scenario
    scenario_uncertainty = forecasts.groupby('scenario')['prediction_interval_width'].agg([
        'mean', 'median', 'std'
    ]).to_dict('index')
    uncertainty_analysis['by_scenario'] = scenario_uncertainty
    
    # High uncertainty regions
    high_uncertainty_threshold = forecasts['prediction_interval_width'].quantile(0.9)
    high_uncertainty_count = (forecasts['prediction_interval_width'] > high_uncertainty_threshold).sum()
    
    uncertainty_analysis['high_uncertainty'] = {
        'threshold': high_uncertainty_threshold,
        'count': int(high_uncertainty_count),
        'percentage': (high_uncertainty_count / len(forecasts)) * 100
    }
    
    logger.info(f"Mean forecast uncertainty: ±{uncertainty_analysis['overall']['mean_interval_width']/2:.1f} years")
    logger.info(f"High uncertainty regions: {uncertainty_analysis['high_uncertainty']['percentage']:.1f}%")
    
    return uncertainty_analysis


def create_forecast_summaries(all_forecasts: pd.DataFrame, 
                            scenario_forecasts: Dict[str, pd.DataFrame],
                            logger: logging.Logger,
                            output_dir: Path) -> None:
    """Create forecast summary reports"""
    
    # Summary by year and scenario
    summary_stats = all_forecasts.groupby(['year', 'scenario']).agg({
        'predicted_life_expectancy': ['mean', 'median', 'std'],
        'prediction_interval_width': ['mean', 'median']
    }).round(2)
    
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats.to_csv(output_dir / "forecast_summary_by_year_scenario.csv")
    
    # Global trends summary
    baseline_forecasts = all_forecasts[all_forecasts['scenario'] == 'baseline']
    
    trend_summary = []
    for year in sorted(baseline_forecasts['year'].unique()):
        year_data = baseline_forecasts[baseline_forecasts['year'] == year]
        trend_summary.append({
            'year': year,
            'global_mean_forecast': year_data['predicted_life_expectancy'].mean(),
            'global_median_forecast': year_data['predicted_life_expectancy'].median(),
            'forecast_std': year_data['predicted_life_expectancy'].std(),
            'mean_uncertainty': year_data['prediction_interval_width'].mean(),
            'n_regions': len(year_data)
        })
    
    trend_df = pd.DataFrame(trend_summary)
    trend_df.to_csv(output_dir / "global_forecast_trends.csv", index=False)
    
    # Scenario comparison
    scenario_comparison = []
    for year in sorted(all_forecasts['year'].unique()):
        year_data = all_forecasts[all_forecasts['year'] == year]
        for scenario in ['optimistic', 'baseline', 'pessimistic']:
            scenario_data = year_data[year_data['scenario'] == scenario]
            if not scenario_data.empty:
                scenario_comparison.append({
                    'year': year,
                    'scenario': scenario,
                    'mean_forecast': scenario_data['predicted_life_expectancy'].mean(),
                    'mean_uncertainty': scenario_data['prediction_interval_width'].mean()
                })
    
    scenario_df = pd.DataFrame(scenario_comparison)
    scenario_pivot = scenario_df.pivot(index='year', columns='scenario', values='mean_forecast')
    scenario_pivot.to_csv(output_dir / "scenario_comparison.csv")
    
    logger.info("Forecast summary reports created")


def analyze_blue_zone_forecasts(forecasts: pd.DataFrame, 
                               grid_df: pd.DataFrame,
                               logger: logging.Logger) -> pd.DataFrame:
    """Analyze forecasts for Blue Zone regions"""
    
    # Get Blue Zone cells
    blue_zone_cols = [col for col in grid_df.columns if col.startswith('is_')]
    blue_zone_data = grid_df[grid_df['is_blue_zone'] == True]
    
    if len(blue_zone_data) == 0:
        logger.warning("No Blue Zone cells found")
        return pd.DataFrame()
    
    # Get forecasts for Blue Zone cells
    blue_zone_forecasts = forecasts[
        forecasts['geo_id'].isin(blue_zone_data['geo_id'])
    ].copy()
    
    if len(blue_zone_forecasts) == 0:
        logger.warning("No forecasts found for Blue Zone cells")
        return pd.DataFrame()
    
    # Add Blue Zone labels
    blue_zone_forecasts = blue_zone_forecasts.merge(
        blue_zone_data[['geo_id'] + blue_zone_cols],
        on='geo_id',
        how='left'
    )
    
    # Calculate Blue Zone-specific statistics
    blue_zone_summary = blue_zone_forecasts.groupby(['year', 'scenario']).agg({
        'predicted_life_expectancy': ['mean', 'median', 'std'],
        'prediction_interval_width': 'mean'
    }).round(2)
    
    logger.info(f"Blue Zone forecast analysis: {len(blue_zone_forecasts)} forecast points")
    
    return blue_zone_forecasts


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif hasattr(obj, 'dtype'):  # numpy types
        if pd.isna(obj):
            return None
        else:
            return obj.item()
    else:
        return obj


if __name__ == "__main__":
    sys.exit(main())