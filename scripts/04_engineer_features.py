#!/usr/bin/env python3
"""
Engineer comprehensive features for Blue Zones analysis
"""
import sys
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, load_config, load_intermediate_data, save_intermediate_data
from features.geodesy_features import GeodesyFeatures
from features.lifestyle_features import LifestyleFeatures


def main():
    """Run comprehensive feature engineering"""
    
    # Setup
    logger = setup_logging("INFO")
    config = load_config("config/config.yaml")
    
    logger.info("Starting comprehensive feature engineering")
    
    # Load grid
    try:
        grid_df = load_intermediate_data("global_grid_5km", "processed")
        logger.info(f"Loaded grid with {len(grid_df)} cells")
    except Exception as e:
        logger.error(f"Could not load grid: {e}")
        return 1
    
    # Initialize feature engineering classes
    geodesy_fe = GeodesyFeatures(config, logger)
    lifestyle_fe = LifestyleFeatures(config, logger)
    
    # 1. Generate basic geodesy features
    logger.info("Generating geodesy features")
    geodesy_features = geodesy_fe.generate_basic_geodesy(grid_df)
    
    # Add coastline distance (simplified version)
    geodesy_features = geodesy_fe.add_coastline_distance(geodesy_features)
    
    # Add elevation features (if data available)
    elevation_file = None  # Would be path to SRTM data
    geodesy_features = geodesy_fe.add_elevation_features(geodesy_features, elevation_file)
    
    # Add effective gravity
    geodesy_features = geodesy_fe.add_effective_gravity(geodesy_features)
    
    logger.info(f"Generated {len(geodesy_features.columns)-1} geodesy features")
    
    # 2. Generate lifestyle features (OSM-based)
    logger.info("Generating lifestyle features (this may take a while)")
    
    # Sample grid for development/testing
    if len(grid_df) > 1000:
        logger.info("Using sample of grid for lifestyle feature development")
        sample_grid = grid_df.sample(n=1000, random_state=42)
    else:
        sample_grid = grid_df
        
    lifestyle_features = lifestyle_fe.generate_osm_features(sample_grid, buffer_km=2.0)
    lifestyle_features = lifestyle_fe.create_lifestyle_proxies(lifestyle_features)
    
    logger.info(f"Generated {len(lifestyle_features.columns)-1} lifestyle features")
    
    # 3. Load existing processed features
    feature_datasets = {}
    
    try:
        climate_features = load_intermediate_data("climate_features", "processed")
        feature_datasets['climate'] = climate_features
        logger.info(f"Loaded climate features: {len(climate_features.columns)} columns")
    except Exception as e:
        logger.warning(f"Could not load climate features: {e}")
        
    try:
        pop_features = load_intermediate_data("population_features", "processed")
        feature_datasets['population'] = pop_features
        logger.info(f"Loaded population features: {len(pop_features.columns)} columns")
    except Exception as e:
        logger.warning(f"Could not load population features: {e}")
        
    try:
        socio_features = load_intermediate_data("socioeconomic_features", "processed")
        feature_datasets['socioeconomic'] = socio_features  
        logger.info(f"Loaded socioeconomic features: {len(socio_features.columns)} columns")
    except Exception as e:
        logger.warning(f"Could not load socioeconomic features: {e}")
        
    try:
        ntl_features = load_intermediate_data("nightlights_features", "processed")
        feature_datasets['nightlights'] = ntl_features
        logger.info(f"Loaded nightlights features: {len(ntl_features.columns)} columns")
    except Exception as e:
        logger.warning(f"Could not load nightlights features: {e}")
    
    # 4. Combine all features
    logger.info("Combining all feature sets")
    
    # Start with geodesy features
    combined_features = geodesy_features
    
    # Merge lifestyle features
    combined_features = combined_features.merge(
        lifestyle_features, on='geo_id', how='left'
    )
    
    # Merge other feature sets
    for name, features in feature_datasets.items():
        logger.info(f"Merging {name} features")
        
        # Handle temporal features (take most recent year)
        if 'year' in features.columns:
            latest_year = features['year'].max()
            features_latest = features[features['year'] == latest_year]
            features_latest = features_latest.drop(columns=['year'])
        else:
            features_latest = features
            
        combined_features = combined_features.merge(
            features_latest, on='geo_id', how='left'
        )
        
    logger.info(f"Combined feature set: {len(combined_features)} rows, {len(combined_features.columns)} columns")
    
    # 5. Feature validation and cleaning
    logger.info("Cleaning and validating features")
    combined_features = clean_features(combined_features, logger)
    
    # 6. Save combined features
    save_intermediate_data(combined_features, "combined_features", "features")
    logger.info("Combined features saved successfully")
    
    # 7. Generate feature summary
    generate_feature_summary(combined_features, logger)
    
    logger.info("Feature engineering completed successfully")
    return 0


def clean_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Clean and validate feature dataset"""
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Fill with 0 for counts/densities, median for continuous variables
    for col in numeric_cols:
        if col == 'geo_id':
            continue
            
        # Use 0 for density/count features
        if any(keyword in col.lower() for keyword in ['density', 'count', 'score']):
            df[col] = df[col].fillna(0)
        else:
            # Use median for other continuous features
            df[col] = df[col].fillna(df[col].median())
    
    # Remove features with too many missing values
    missing_pct = df.isnull().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 70]
    
    if not high_missing.empty:
        logger.warning(f"Removing features with >70% missing: {list(high_missing.index)}")
        df = df.drop(columns=high_missing.index)
    
    # Remove constant features
    constant_features = []
    for col in numeric_cols:
        if col in df.columns and df[col].nunique() <= 1:
            constant_features.append(col)
            
    if constant_features:
        logger.warning(f"Removing constant features: {constant_features}")
        df = df.drop(columns=constant_features)
    
    return df


def generate_feature_summary(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Generate feature summary statistics"""
    
    summary_stats = df.describe()
    
    # Save summary
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_stats.to_csv(output_dir / "feature_summary.csv")
    
    # Generate feature documentation
    feature_doc = []
    feature_doc.append("# Blue Zones Feature Documentation\n")
    feature_doc.append("## Feature Summary\n")
    feature_doc.append(f"- Total features: {len(df.columns) - 1}")  # Exclude geo_id
    feature_doc.append(f"- Total observations: {len(df)}")
    feature_doc.append(f"- Missing data: {df.isnull().sum().sum()} values\n")
    
    # Feature categories
    categories = {
        'Geodesy': ['latitude', 'longitude', 'elevation', 'slope', 'gravity'],
        'Climate': ['temperature', 'precipitation', 'humidity'],
        'Demographics': ['population', 'urban', 'density'],
        'Socioeconomic': ['gdp', 'poverty', 'education', 'lights'],
        'Environment': ['pm25', 'ndvi', 'tree'],
        'Lifestyle': ['religious', 'market', 'healthcare', 'walkability'],
        'Infrastructure': ['transport', 'road', 'access']
    }
    
    for category, keywords in categories.items():
        matching_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in keywords):
                matching_cols.append(col)
                
        if matching_cols:
            feature_doc.append(f"\n### {category} Features ({len(matching_cols)})")
            for col in matching_cols[:10]:  # Show first 10
                feature_doc.append(f"- {col}")
            if len(matching_cols) > 10:
                feature_doc.append(f"- ... and {len(matching_cols) - 10} more")
    
    # Write documentation
    with open(output_dir / "feature_documentation.md", "w") as f:
        f.write("\n".join(feature_doc))
    
    logger.info("Feature summary and documentation generated")


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())