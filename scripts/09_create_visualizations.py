#!/usr/bin/env python3
"""
Create comprehensive visualizations and dashboard for Blue Zones analysis
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, load_config, load_intermediate_data
from viz.dashboard import BlueZonesDashboard


def main():
    """Create comprehensive visualizations and dashboard"""
    
    # Setup
    logger = setup_logging("INFO")
    config = load_config("config/config.yaml")
    
    logger.info("Starting comprehensive visualization creation")
    
    # Load all analysis results
    data_sources = load_all_data_sources(logger)
    
    if not data_sources:
        logger.error("Could not load required data sources")
        return 1
    
    # Initialize dashboard
    dashboard = BlueZonesDashboard(config, logger)
    
    # Create maps
    logger.info("Creating interactive maps")
    try:
        maps = dashboard.create_global_maps(
            features=data_sources.get('features', pd.DataFrame()),
            predictions=data_sources.get('predictions', pd.DataFrame()),
            forecasts=data_sources.get('forecasts', pd.DataFrame())
        )
        logger.info(f"Created {len(maps)} interactive maps")
    except Exception as e:
        logger.error(f"Error creating maps: {e}")
        maps = {}
    
    # Create analysis plots
    logger.info("Creating analysis plots")
    try:
        plots = dashboard.create_analysis_plots(
            matched_results=data_sources.get('matched_results', {}),
            classifier_results=data_sources.get('classifier_results', {}),
            forecasts=data_sources.get('forecasts', pd.DataFrame())
        )
        logger.info(f"Created {len(plots)} analysis plots")
    except Exception as e:
        logger.error(f"Error creating plots: {e}")
        plots = {}
    
    # Save individual visualizations
    output_dir = Path("data/outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save maps as HTML files
    logger.info("Saving interactive maps")
    for map_name, map_obj in maps.items():
        if hasattr(map_obj, 'save'):
            map_obj.save(output_dir / f"{map_name}_map.html")
        elif isinstance(map_obj, dict):
            # Handle nested maps
            for sub_name, sub_map in map_obj.items():
                if hasattr(sub_map, 'save'):
                    sub_map.save(output_dir / f"{map_name}_{sub_name}_map.html")
    
    # Save plots as HTML files
    logger.info("Saving analysis plots")
    for plot_name, plot_obj in plots.items():
        if hasattr(plot_obj, 'write_html'):
            plot_obj.write_html(output_dir / f"{plot_name}_plot.html")
    
    # Create comprehensive dashboard
    logger.info("Creating comprehensive HTML dashboard")
    
    all_data = {
        'maps': maps,
        'plots': plots,
        'analysis_metadata': extract_analysis_metadata(data_sources)
    }
    
    try:
        dashboard_html = dashboard.create_summary_dashboard(all_data)
        
        # Save dashboard
        dashboard_path = output_dir / "blue_zones_dashboard.html"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
            
        logger.info(f"Dashboard saved to: {dashboard_path}")
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
    
    # Create summary report
    logger.info("Creating summary analysis report")
    create_summary_report(data_sources, output_dir, logger)
    
    # Create data export for external use
    logger.info("Creating data exports")
    create_data_exports(data_sources, output_dir, logger)
    
    logger.info("Visualization creation completed successfully")
    logger.info(f"All outputs saved to: {output_dir}")
    
    return 0


def load_all_data_sources(logger: logging.Logger) -> Dict[str, Any]:
    """Load all data sources for visualization"""
    
    data_sources = {}
    
    # Load features
    try:
        features = load_intermediate_data("combined_features", "features")
        data_sources['features'] = features
        logger.info(f"Loaded features: {len(features)} observations")
    except Exception as e:
        logger.warning(f"Could not load features: {e}")
        data_sources['features'] = pd.DataFrame()
    
    # Load predictions
    try:
        predictions = pd.read_parquet("data/outputs/blue_zone_predictions.parquet")
        data_sources['predictions'] = predictions
        logger.info(f"Loaded predictions: {len(predictions)} observations")
    except Exception as e:
        logger.warning(f"Could not load predictions: {e}")
        data_sources['predictions'] = pd.DataFrame()
    
    # Load forecasts
    try:
        forecasts = pd.read_parquet("data/outputs/life_expectancy_forecasts.parquet")
        data_sources['forecasts'] = forecasts
        logger.info(f"Loaded forecasts: {len(forecasts)} observations")
    except Exception as e:
        logger.warning(f"Could not load forecasts: {e}")
        data_sources['forecasts'] = pd.DataFrame()
    
    # Load matched comparison results
    try:
        with open("data/outputs/matched_comparison_results.json", "r") as f:
            matched_results = json.load(f)
        data_sources['matched_results'] = matched_results
        logger.info("Loaded matched comparison results")
    except Exception as e:
        logger.warning(f"Could not load matched comparison results: {e}")
        data_sources['matched_results'] = {}
    
    # Load classifier results
    try:
        with open("data/outputs/classifier_training_results.json", "r") as f:
            classifier_results = json.load(f)
        data_sources['classifier_results'] = classifier_results
        logger.info("Loaded classifier training results")
    except Exception as e:
        logger.warning(f"Could not load classifier results: {e}")
        data_sources['classifier_results'] = {}
    
    return data_sources


def extract_analysis_metadata(data_sources: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from analysis results"""
    
    metadata = {}
    
    # Features metadata
    features = data_sources.get('features', pd.DataFrame())
    if not features.empty:
        metadata['n_observations'] = len(features)
        metadata['n_features'] = len(features.columns) - 1  # Exclude geo_id
    
    # Predictions metadata
    predictions = data_sources.get('predictions', pd.DataFrame())
    if not predictions.empty:
        metadata['n_predictions'] = len(predictions)
        metadata['high_score_regions'] = len(predictions[predictions['blue_zone_decile'] >= 8])
    
    # Forecasts metadata
    forecasts = data_sources.get('forecasts', pd.DataFrame())
    if not forecasts.empty:
        metadata['n_forecasts'] = len(forecasts)
        metadata['forecast_years'] = sorted(forecasts['year'].unique()) if 'year' in forecasts.columns else []
        metadata['forecast_scenarios'] = sorted(forecasts['scenario'].unique()) if 'scenario' in forecasts.columns else []
    
    # Analysis results metadata
    matched_results = data_sources.get('matched_results', {})
    if 'analysis_metadata' in matched_results:
        metadata.update(matched_results['analysis_metadata'])
    
    return metadata


def create_summary_report(data_sources: Dict[str, Any], 
                         output_dir: Path, 
                         logger: logging.Logger) -> None:
    """Create comprehensive summary report"""
    
    report_lines = []
    
    # Header
    report_lines.extend([
        "# Blue Zones Quantified - Analysis Summary Report",
        "",
        "## Executive Summary",
        "",
        "This report presents the results of a comprehensive analysis of global longevity patterns",
        "and Blue Zone characteristics using machine learning and geospatial analysis.",
        ""
    ])
    
    # Data Overview
    report_lines.append("## Data Overview")
    report_lines.append("")
    
    features = data_sources.get('features', pd.DataFrame())
    if not features.empty:
        report_lines.extend([
            f"- **Total grid cells analyzed:** {len(features):,}",
            f"- **Features engineered:** {len(features.columns) - 1}",
            f"- **Spatial resolution:** 5km global grid",
        ])
    
    # Blue Zone Analysis
    report_lines.append("")
    report_lines.append("## Blue Zone Analysis Results")
    report_lines.append("")
    
    matched_results = data_sources.get('matched_results', {})
    if 'overall_blue_zones' in matched_results:
        overall = matched_results['overall_blue_zones']
        if 'summary' in overall and 'best_estimate' in overall['summary']:
            best_estimate = overall['summary']['best_estimate']
            if not pd.isna(best_estimate):
                report_lines.append(f"- **Overall Blue Zone effect:** {best_estimate:.1f} years of additional life expectancy")
    
    # Individual Blue Zone effects
    if 'individual_blue_zones' in matched_results:
        report_lines.append("- **Individual Blue Zone effects:**")
        for zone, zone_results in matched_results['individual_blue_zones'].items():
            if 'summary' in zone_results and 'best_estimate' in zone_results['summary']:
                best_estimate = zone_results['summary']['best_estimate']
                if not pd.isna(best_estimate):
                    report_lines.append(f"  - {zone.title()}: {best_estimate:.1f} years")
    
    # Classification Results
    report_lines.append("")
    report_lines.append("## Blue Zone Classification")
    report_lines.append("")
    
    classifier_results = data_sources.get('classifier_results', {})
    if 'evaluation_metrics' in classifier_results:
        metrics = classifier_results['evaluation_metrics']
        auc_score = metrics.get('auc_score')
        if auc_score:
            report_lines.append(f"- **Classification AUC:** {auc_score:.3f}")
    
    predictions = data_sources.get('predictions', pd.DataFrame())
    if not predictions.empty:
        high_score_count = len(predictions[predictions['blue_zone_decile'] >= 8])
        total_count = len(predictions)
        report_lines.extend([
            f"- **High-scoring regions identified:** {high_score_count:,}",
            f"- **Percentage of global grid:** {(high_score_count/total_count*100):.1f}%"
        ])
    
    # Top Features
    if 'feature_importance' in classifier_results:
        top_features = classifier_results['feature_importance'].get('top_10_features', [])
        if top_features:
            report_lines.append("- **Top predictive features:**")
            for i, feature in enumerate(top_features[:5], 1):
                report_lines.append(f"  {i}. {feature}")
    
    # Forecasting Results
    report_lines.append("")
    report_lines.append("## Life Expectancy Forecasts")
    report_lines.append("")
    
    forecasts = data_sources.get('forecasts', pd.DataFrame())
    if not forecasts.empty:
        forecast_years = sorted(forecasts['year'].unique()) if 'year' in forecasts.columns else []
        if forecast_years:
            report_lines.append(f"- **Forecast horizon:** {min(forecast_years)} - {max(forecast_years)}")
            
            # Baseline scenario trends
            baseline = forecasts[forecasts['scenario'] == 'baseline'] if 'scenario' in forecasts.columns else forecasts
            if not baseline.empty:
                global_trends = baseline.groupby('year')['predicted_life_expectancy'].mean()
                if len(global_trends) > 1:
                    trend = global_trends.iloc[-1] - global_trends.iloc[0]
                    report_lines.append(f"- **Projected global trend:** {trend:+.1f} years over forecast period")
    
    # Methodology
    report_lines.extend([
        "",
        "## Methodology",
        "",
        "- **Spatial Analysis:** Global 5km grid system",
        "- **Feature Engineering:** Climate, demographics, infrastructure, lifestyle proxies",
        "- **Matching Analysis:** Propensity score matching for causal inference",
        "- **Machine Learning:** LightGBM classification with SHAP explanations",
        "- **Forecasting:** Ensemble models with uncertainty quantification",
        "- **Validation:** Spatial and temporal cross-validation",
        ""
    ])
    
    # Data Sources
    report_lines.extend([
        "## Data Sources",
        "",
        "- Life Expectancy: IHME Global Burden of Disease",
        "- Climate: ERA5 Reanalysis",
        "- Air Quality: Van Donkelaar PM2.5",
        "- Demographics: WorldPop",
        "- Socioeconomic: World Bank Open Data",
        "- Infrastructure: OpenStreetMap",
        "- Elevation: NASA SRTM",
        ""
    ])
    
    # Save report
    report_path = output_dir / "analysis_summary_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Summary report saved to: {report_path}")


def create_data_exports(data_sources: Dict[str, Any],
                       output_dir: Path,
                       logger: logging.Logger) -> None:
    """Create data exports for external use"""
    
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(exist_ok=True)
    
    # Export key results as CSV
    exports = {
        'blue_zone_predictions': data_sources.get('predictions', pd.DataFrame()),
        'life_expectancy_forecasts': data_sources.get('forecasts', pd.DataFrame()),
        'combined_features': data_sources.get('features', pd.DataFrame())
    }
    
    for name, data in exports.items():
        if not data.empty:
            # Limit size for CSV export
            if len(data) > 10000:
                export_data = data.sample(n=10000, random_state=42)
                logger.info(f"Sampling {name} to 10,000 rows for CSV export")
            else:
                export_data = data
                
            csv_path = exports_dir / f"{name}.csv"
            export_data.to_csv(csv_path, index=False)
            logger.info(f"Exported {name}: {len(export_data)} rows to {csv_path}")
    
    # Export summary statistics
    summary_stats = {}
    
    for name, data in exports.items():
        if not data.empty:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_stats[name] = data[numeric_cols].describe().to_dict()
    
    if summary_stats:
        with open(exports_dir / "summary_statistics.json", "w") as f:
            json.dump(summary_stats, f, indent=2)
        logger.info("Summary statistics exported")


if __name__ == "__main__":
    sys.exit(main())