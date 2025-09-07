#!/usr/bin/env python3
"""
Longevity Prediction Tool - Working Script
Uses actual data from cross_section_final.csv to create a functional prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import warnings
# Suppress specific warnings only when necessary
# warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set light blue background for plots (user preference)
plt.rcParams['axes.facecolor'] = '#E5ECF6'

def load_and_prepare_data():
    """Load and prepare the real data"""
    data_file = '../outputs/cross_section_final.csv'
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return None, None
    
    df = pd.read_csv(data_file)
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # Define features available in the actual dataset
    features = [
        'gdp_per_capita',
        'walkability_score', 
        'greenspace_pct',
        'cvd_mortality',
        'population_density_log',
        'temperature_mean',
        'elevation'
    ]
    
    # Check which features are available
    available_features = [f for f in features if f in df.columns]
    print(f"Available features: {', '.join(available_features)}")
    
    # Create clean dataset
    required_cols = available_features + ['life_expectancy']
    clean_df = df[required_cols].dropna()
    
    print(f"Clean dataset: {len(clean_df)} observations")
    print(f"Blue zones: {df['is_blue_zone'].sum() if 'is_blue_zone' in df.columns else 'N/A'}")
    
    return clean_df, available_features

def train_model(df, features):
    """Train the longevity prediction model"""
    print("\nTraining longevity prediction model...")
    
    # Prepare data
    X = df[features]
    y = df['life_expectancy']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"\nModel Performance:")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f} years")
    print(f"Cross-Validation R²: {cv_mean:.4f} (±{cv_std:.4f})")
    
    return model, scaler, {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std
    }

def analyze_feature_importance(model, features):
    """Analyze feature importance"""
    print(f"\nFeature Importance Analysis:")
    print("=" * 40)
    
    feature_descriptions = {
        'gdp_per_capita': 'GDP per Capita (USD)',
        'walkability_score': 'Walkability Score (0-100)',
        'greenspace_pct': 'Green Space Percentage',
        'cvd_mortality': 'CVD Mortality Rate',
        'population_density_log': 'Population Density (log)',
        'temperature_mean': 'Mean Temperature (°C)',
        'elevation': 'Elevation (meters)'
    }
    
    importance_data = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance (from gradient boosting model):")
    for _, row in importance_data.iterrows():
        feature_desc = feature_descriptions.get(row['feature'], row['feature'])
        print(f"{feature_desc:<40}: {row['importance']:.4f}")
    
    return importance_data

def test_predictions(model, scaler, features, original_df):
    """Test predictions on actual data scenarios"""
    print(f"\nTesting Predictions on Actual Data:")
    print("=" * 50)
    
    feature_descriptions = {
        'gdp_per_capita': 'GDP per Capita (USD)',
        'walkability_score': 'Walkability Score (0-100)', 
        'greenspace_pct': 'Green Space Percentage',
        'cvd_mortality': 'CVD Mortality Rate',
        'population_density_log': 'Population Density (log)',
        'temperature_mean': 'Mean Temperature (°C)',
        'elevation': 'Elevation (meters)'
    }
    
    def predict(inputs):
        """Make a prediction"""
        input_vector = []
        for feature in features:
            input_vector.append(inputs.get(feature, 0))
        
        input_array = np.array(input_vector).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        return model.predict(input_scaled)[0]
    
    # Test scenarios based on actual data
    test_scenarios = {}
    
    # Get Ikaria (Blue Zone) if it exists
    if 'is_blue_zone' in original_df.columns:
        blue_zones = original_df[original_df['is_blue_zone'] == 1]
        if len(blue_zones) > 0:
            ikaria = blue_zones.iloc[0]
            test_scenarios['Ikaria Blue Zone (Actual)'] = {
                feature: ikaria[feature] for feature in features if feature in ikaria.index
            }
    
    # Get high/low GDP examples
    high_gdp = original_df.nlargest(1, 'gdp_per_capita').iloc[0]
    low_gdp = original_df.nsmallest(1, 'gdp_per_capita').iloc[0]
    
    test_scenarios['High GDP Location (Actual)'] = {
        feature: high_gdp[feature] for feature in features if feature in high_gdp.index
    }
    
    test_scenarios['Low GDP Location (Actual)'] = {
        feature: low_gdp[feature] for feature in features if feature in low_gdp.index
    }
    
    # Dataset average
    test_scenarios['Dataset Average Profile'] = {
        feature: original_df[feature].mean() for feature in features
    }
    
    print(f"Prediction Results:")
    print("=" * 60)
    print(f"{'Scenario':<30} {'Predicted Life Expectancy':<25}")
    print("=" * 60)
    
    results = {}
    for scenario_name, inputs in test_scenarios.items():
        try:
            prediction = predict(inputs)
            results[scenario_name] = prediction
            print(f"{scenario_name:<30} {prediction:.1f} years")
        except Exception as e:
            print(f"{scenario_name:<30} Error: {e}")
    
    # Show detailed breakdown for first scenario
    if results and test_scenarios:
        first_scenario_name = list(test_scenarios.keys())[0]
        first_scenario_inputs = test_scenarios[first_scenario_name]
        
        print(f"\nDetailed Prediction Breakdown ({first_scenario_name}):")
        print("-" * 50)
        
        for feature, value in first_scenario_inputs.items():
            description = feature_descriptions.get(feature, feature)
            print(f"{description:<35}: {value:>8.2f}")
        
        prediction = results[first_scenario_name]
        print(f"\nPredicted Life Expectancy: {prediction:.1f} years")
    
    return results

def create_visualization(original_df, features):
    """Create a simple visualization"""
    print(f"\nCreating data visualization...")
    
    # Plot feature correlations with life expectancy
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            
            # Clean data
            mask = original_df[feature].notna() & original_df['life_expectancy'].notna()
            if mask.sum() > 0:
                x = original_df[mask][feature]
                y = original_df[mask]['life_expectancy']
                
                # Color by blue zone status if available
                if 'is_blue_zone' in original_df.columns:
                    colors = ['red' if bz == 1 else 'steelblue' for bz in original_df[mask]['is_blue_zone']]
                else:
                    colors = 'steelblue'
                
                ax.scatter(x, y, c=colors, alpha=0.7)
                ax.set_xlabel(feature.replace('_', ' ').title())
                ax.set_ylabel('Life Expectancy (years)')
                
                # Add trend line
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_line, p(x_line), 'darkgreen', linestyle='--', alpha=0.8)
                except:
                    pass
                
                ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(features), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Feature Relationships with Life Expectancy', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_dir = '../outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'longevity_features_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_dir}/longevity_features_analysis.png")
    
    plt.show()

def save_results(results, model_stats, feature_importance):
    """Save results to CSV files"""
    print(f"\nSaving results...")
    
    output_dir = '../outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prediction results
    if results:
        results_df = pd.DataFrame([
            {'scenario': scenario, 'predicted_life_expectancy': prediction}
            for scenario, prediction in results.items()
        ])
        
        results_file = os.path.join(output_dir, 'longevity_prediction_scenarios.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Prediction scenarios saved to: {results_file}")
    
    # Save feature importance
    if feature_importance is not None:
        importance_file = os.path.join(output_dir, 'longevity_feature_importance.csv')
        feature_importance.to_csv(importance_file, index=False)
        print(f"Feature importance saved to: {importance_file}")
    
    # Save model statistics
    if model_stats:
        stats_df = pd.DataFrame([model_stats])
        stats_file = os.path.join(output_dir, 'longevity_model_performance.csv')
        stats_df.to_csv(stats_file, index=False)
        print(f"Model performance saved to: {stats_file}")

def main():
    """Main function to run the longevity prediction analysis"""
    print("Longevity Prediction Tool - Real Data Analysis")
    print("=" * 60)
    
    # Load data
    df, features = load_and_prepare_data()
    if df is None:
        print("Could not load data. Exiting.")
        return
    
    # Train model
    model, scaler, model_stats = train_model(df, features)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, features)
    
    # Test predictions
    results = test_predictions(model, scaler, features, df)
    
    # Create visualization
    create_visualization(df, features)
    
    # Save results
    save_results(results, model_stats, feature_importance)
    
    print(f"\n" + "=" * 60)
    print("LONGEVITY PREDICTION ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"✓ Model trained successfully (R² = {model_stats['cv_r2_mean']:.4f})")
    print(f"✓ Prediction accuracy: ±{model_stats['test_mae']:.1f} years MAE")
    print(f"✓ Feature importance analyzed")
    print(f"✓ Predictions generated for real data scenarios")
    print(f"✓ Visualizations created")
    print(f"✓ Results saved to ../outputs/")

if __name__ == "__main__":
    main()
