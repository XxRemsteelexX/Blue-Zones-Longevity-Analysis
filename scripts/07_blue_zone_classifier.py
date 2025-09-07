#!/usr/bin/env python3
"""
Build and evaluate Blue Zone classifier
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
from models.blue_zone_classifier import BlueZoneClassifier


def main():
    """Build Blue Zone classifier"""
    
    # Setup
    logger = setup_logging("INFO")
    config = load_config("config/config.yaml")
    
    logger.info("Starting Blue Zone classifier development")
    
    # Load data
    try:
        # Load features
        features = load_intermediate_data("combined_features", "features")
        logger.info(f"Loaded features: {len(features)} observations")
        
        # Load grid with Blue Zone labels
        grid_df = load_intermediate_data("global_grid_5km", "processed")
        
    except Exception as e:
        logger.error(f"Could not load required data: {e}")
        return 1
    
    # Merge features with Blue Zone labels
    blue_zone_cols = [col for col in grid_df.columns if col.startswith('is_')]
    training_data = features.merge(
        grid_df[['geo_id'] + blue_zone_cols],
        on='geo_id',
        how='left'
    )
    
    # Fill missing Blue Zone labels
    for col in blue_zone_cols:
        training_data[col] = training_data[col].fillna(False)
    
    logger.info(f"Training data: {len(training_data)} observations")
    logger.info(f"Blue Zone cells: {training_data['is_blue_zone'].sum()}")
    
    if training_data['is_blue_zone'].sum() == 0:
        logger.error("No Blue Zone observations found for training")
        return 1
    
    # Initialize classifier
    classifier = BlueZoneClassifier(config, logger)
    
    # Train classifier
    logger.info("Training Blue Zone classifier")
    training_results = classifier.train_classifier(
        training_data, 
        treatment_col='is_blue_zone'
    )
    
    if not training_results:
        logger.error("Classifier training failed")
        return 1
    
    # Generate predictions for all data
    logger.info("Generating Blue Zone likelihood scores")
    predictions = classifier.predict_blue_zone_likelihood(training_data)
    
    # Add geographic information
    predictions = predictions.merge(
        training_data[['geo_id', 'latitude', 'longitude']],
        on='geo_id',
        how='left'
    )
    
    # Explain predictions (sample)
    logger.info("Generating prediction explanations")
    explanations = classifier.explain_predictions(training_data, n_samples=200)
    
    # Save results
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trained classifier
    with open(model_dir / "blue_zone_classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)
    
    # Save predictions
    predictions.to_parquet(output_dir / "blue_zone_predictions.parquet")
    predictions.to_csv(output_dir / "blue_zone_predictions.csv", index=False)
    
    # Save training results
    with open(output_dir / "classifier_training_results.json", "w") as f:
        json_results = convert_numpy_types(training_results)
        json.dump(json_results, f, indent=2)
    
    # Save feature importance
    if 'feature_importance' in training_results:
        feat_importance = training_results['feature_importance']['feature_importance']
        feat_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    
    # Create candidate regions report
    logger.info("Identifying candidate Blue Zone regions")
    candidate_regions = identify_candidate_regions(predictions, logger)
    candidate_regions.to_csv(output_dir / "candidate_blue_zones.csv", index=False)
    
    # Validation: check known Blue Zones
    validate_known_blue_zones(predictions, training_data, logger)
    
    # Log results
    logger.info("=== CLASSIFIER RESULTS ===")
    
    if 'evaluation_metrics' in training_results:
        metrics = training_results['evaluation_metrics']
        logger.info(f"Cross-validation AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
        logger.info(f"Test AUC: {metrics['auc_score']:.3f}")
    
    if 'feature_importance' in training_results:
        top_features = training_results['feature_importance']['top_10_features']
        logger.info("Top 10 features:")
        for i, feature in enumerate(top_features[:10], 1):
            logger.info(f"  {i}. {feature}")
    
    high_score_regions = len(predictions[predictions['blue_zone_decile'] >= 9])
    logger.info(f"Identified {high_score_regions} high-scoring regions (top 20%)")
    
    logger.info("Blue Zone classifier completed successfully")
    return 0


def identify_candidate_regions(predictions: pd.DataFrame, 
                             logger: logging.Logger,
                             top_n: int = 50) -> pd.DataFrame:
    """Identify top candidate Blue Zone regions"""
    
    # Sort by Blue Zone score
    candidates = predictions.sort_values('blue_zone_score', ascending=False).head(top_n)
    
    # Add ranking
    candidates = candidates.copy()
    candidates['rank'] = range(1, len(candidates) + 1)
    
    # Classify confidence levels
    candidates['confidence'] = pd.cut(
        candidates['blue_zone_score'],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    logger.info(f"Top candidate region score: {candidates.iloc[0]['blue_zone_score']:.4f}")
    logger.info(f"Score range in top {top_n}: {candidates['blue_zone_score'].min():.4f} - {candidates['blue_zone_score'].max():.4f}")
    
    return candidates[['rank', 'geo_id', 'latitude', 'longitude', 'blue_zone_score', 'blue_zone_decile', 'confidence']]


def validate_known_blue_zones(predictions: pd.DataFrame, 
                            training_data: pd.DataFrame,
                            logger: logging.Logger) -> None:
    """Validate classifier performance on known Blue Zones"""
    
    # Get known Blue Zone cells
    known_zones = training_data[training_data['is_blue_zone'] == True]['geo_id']
    
    if len(known_zones) == 0:
        logger.warning("No known Blue Zone cells found for validation")
        return
    
    # Get predictions for known zones
    known_predictions = predictions[predictions['geo_id'].isin(known_zones)]
    
    if len(known_predictions) == 0:
        logger.warning("No predictions found for known Blue Zone cells")
        return
    
    logger.info("=== VALIDATION ON KNOWN BLUE ZONES ===")
    logger.info(f"Known Blue Zone cells: {len(known_zones)}")
    logger.info(f"Mean score for known zones: {known_predictions['blue_zone_score'].mean():.4f}")
    logger.info(f"Median score for known zones: {known_predictions['blue_zone_score'].median():.4f}")
    
    # Check decile distribution
    decile_dist = known_predictions['blue_zone_decile'].value_counts().sort_index()
    logger.info("Decile distribution for known Blue Zones:")
    for decile, count in decile_dist.items():
        logger.info(f"  Decile {decile}: {count} cells")
    
    # Check how many are in top deciles
    top_deciles = known_predictions['blue_zone_decile'] >= 8
    logger.info(f"Known zones in top 30% (deciles 8-10): {top_deciles.sum()}/{len(known_predictions)} ({top_deciles.mean()*100:.1f}%)")


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