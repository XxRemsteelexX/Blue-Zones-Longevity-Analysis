#!/usr/bin/env python3
"""
Run matched comparison analysis for Blue Zones
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, load_config, load_intermediate_data, save_intermediate_data
from analysis.matched_comparison import MatchedComparison


def main():
    """Run matched comparison analysis"""
    
    # Setup
    logger = setup_logging("INFO")
    config = load_config("config/config.yaml")
    
    logger.info("Starting matched comparison analysis")
    
    # Load combined dataset
    try:
        # Load features
        features = load_intermediate_data("combined_features", "features")
        logger.info(f"Loaded features: {len(features)} observations, {len(features.columns)} columns")
        
        # Load life expectancy data
        life_expectancy = load_intermediate_data("life_expectancy", "processed")
        logger.info(f"Loaded life expectancy: {len(life_expectancy)} observations")
        
        # Load grid with Blue Zone labels
        grid_df = load_intermediate_data("global_grid_5km", "processed")
        
    except Exception as e:
        logger.error(f"Could not load required data: {e}")
        return 1
    
    # Merge datasets
    logger.info("Merging datasets for analysis")
    
    # Get most recent year of life expectancy data
    if 'year' in life_expectancy.columns:
        latest_year = life_expectancy['year'].max()
        le_latest = life_expectancy[life_expectancy['year'] == latest_year]
        logger.info(f"Using life expectancy data from {latest_year}")
    else:
        le_latest = life_expectancy
    
    # Merge all data
    analysis_data = features.merge(
        le_latest[['geo_id', 'val', 'lower', 'upper']], 
        on='geo_id', 
        how='inner'
    )
    
    # Add Blue Zone labels from grid
    blue_zone_cols = [col for col in grid_df.columns if col.startswith('is_')]
    analysis_data = analysis_data.merge(
        grid_df[['geo_id'] + blue_zone_cols],
        on='geo_id',
        how='left'
    )
    
    # Fill missing Blue Zone labels with False
    for col in blue_zone_cols:
        analysis_data[col] = analysis_data[col].fillna(False)
    
    logger.info(f"Analysis dataset: {len(analysis_data)} observations")
    logger.info(f"Blue Zone observations: {analysis_data['is_blue_zone'].sum()}")
    
    # Run matched comparison
    matcher = MatchedComparison(config, logger)
    
    # Overall Blue Zone effect
    logger.info("Estimating overall Blue Zone treatment effect")
    overall_results = matcher.estimate_treatment_effects(
        analysis_data, 
        treatment_col='is_blue_zone',
        outcome_col='val'
    )
    
    # Individual Blue Zone effects
    individual_results = {}
    blue_zones = ['sardinia', 'okinawa', 'nicoya', 'ikaria', 'loma_linda']
    
    for zone in blue_zones:
        zone_col = f'is_{zone}'
        if zone_col in analysis_data.columns:
            logger.info(f"Estimating treatment effect for {zone}")
            
            zone_results = matcher.estimate_treatment_effects(
                analysis_data,
                treatment_col=zone_col,
                outcome_col='val'
            )
            
            individual_results[zone] = zone_results
    
    # Compile results
    results = {
        'overall_blue_zones': overall_results,
        'individual_blue_zones': individual_results,
        'analysis_metadata': {
            'n_observations': len(analysis_data),
            'n_blue_zone_cells': int(analysis_data['is_blue_zone'].sum()),
            'n_features_used': len([col for col in analysis_data.columns 
                                  if analysis_data[col].dtype in ['int64', 'float64']]) - 1,
            'data_year': latest_year if 'year' in life_expectancy.columns else 'unknown'
        }
    }
    
    # Save results
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "matched_comparison_results.json", "w") as f:
        # Convert numpy types for JSON serialization
        json_results = convert_numpy_types(results)
        json.dump(json_results, f, indent=2)
    
    # Save summary table
    summary_data = []
    
    # Overall results
    if 'summary' in overall_results:
        summary = overall_results['summary']
        if 'method_comparison' in summary:
            for method, result in summary['method_comparison'].items():
                summary_data.append({
                    'blue_zone': 'Overall',
                    'method': method,
                    'att_estimate': result['att'],
                    'standard_error': result['se'],
                    'ci_lower': result.get('ci_lower'),
                    'ci_upper': result.get('ci_upper'),
                    'n_matched': result.get('n_matched', 0)
                })
    
    # Individual zone results
    for zone, zone_results in individual_results.items():
        if 'summary' in zone_results:
            summary = zone_results['summary']
            if 'method_comparison' in summary:
                for method, result in summary['method_comparison'].items():
                    summary_data.append({
                        'blue_zone': zone.title(),
                        'method': method,
                        'att_estimate': result['att'],
                        'standard_error': result['se'],
                        'ci_lower': result.get('ci_lower'),
                        'ci_upper': result.get('ci_upper'),
                        'n_matched': result.get('n_matched', 0)
                    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "matched_comparison_summary.csv", index=False)
    
    # Log key findings
    logger.info("=== MATCHED COMPARISON RESULTS ===")
    
    if 'summary' in overall_results and 'best_estimate' in overall_results['summary']:
        best_estimate = overall_results['summary']['best_estimate']
        if not pd.isna(best_estimate):
            logger.info(f"Overall Blue Zone effect: {best_estimate:.2f} years")
        else:
            logger.info("Could not estimate overall Blue Zone effect")
    
    for zone, zone_results in individual_results.items():
        if 'summary' in zone_results and 'best_estimate' in zone_results['summary']:
            best_estimate = zone_results['summary']['best_estimate']
            if not pd.isna(best_estimate):
                logger.info(f"{zone.title()} effect: {best_estimate:.2f} years")
    
    logger.info("Matched comparison analysis completed successfully")
    return 0


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