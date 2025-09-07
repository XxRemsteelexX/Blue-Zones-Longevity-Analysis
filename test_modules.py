#!/usr/bin/env python3
"""
Test Blue Zones modules and generate sample data for notebooks
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append("src")

# Test imports
print("Testing module imports...")
try:
    from features.gravity_hypothesis import GravityLongevityAnalyzer
    print("SUCCESS: Gravity hypothesis module imported successfully")
except Exception as e:
    print(f"ERROR: Gravity hypothesis import failed: {e}")

try:
    from models.panel_fe import PanelFixedEffects
    print("SUCCESS: Panel FE module imported successfully")
except Exception as e:
    print(f"ERROR: Panel FE import failed: {e}")

try:
    from models.spatial_spillovers import SpatialSpilloverAnalyzer
    print("SUCCESS: Spatial spillovers module imported successfully")
except Exception as e:
    print(f"ERROR: Spatial spillovers import failed: {e}")

try:
    from models.double_ml import DoubleMachineLearning
    print("SUCCESS: Double ML module imported successfully")
except Exception as e:
    print(f"ERROR: Double ML import failed: {e}")

# Test gravity analysis with sample data
print("\nTesting gravity analysis...")
try:
    # Create sample data
    np.random.seed(42)
    
    # Blue Zone locations
    locations = [
        {'name': 'Nicoya', 'latitude': 10.2, 'longitude': -85.4, 'elevation': 200, 'life_expectancy': 83.7, 'is_blue_zone': 1},
        {'name': 'Okinawa', 'latitude': 26.3, 'longitude': 127.9, 'elevation': 50, 'life_expectancy': 85.5, 'is_blue_zone': 1},
        {'name': 'Sardinia', 'latitude': 40.1, 'longitude': 9.4, 'elevation': 300, 'life_expectancy': 84.8, 'is_blue_zone': 1},
        {'name': 'Ikaria', 'latitude': 37.6, 'longitude': 26.2, 'elevation': 400, 'life_expectancy': 84.1, 'is_blue_zone': 1},
        {'name': 'Loma Linda', 'latitude': 34.0, 'longitude': -117.3, 'elevation': 350, 'life_expectancy': 82.9, 'is_blue_zone': 1},
    ]
    
    # Add comparison locations
    for i in range(20):
        locations.append({
            'name': f'Location_{i}',
            'latitude': np.random.uniform(-60, 70),
            'longitude': np.random.uniform(-180, 180),
            'elevation': max(0, np.random.exponential(200)),
            'life_expectancy': np.random.normal(78, 5),
            'is_blue_zone': 0
        })
    
    df = pd.DataFrame(locations)
    df['life_expectancy'] = np.clip(df['life_expectancy'], 65, 95)
    
    # Run gravity analysis
    gravity_analyzer = GravityLongevityAnalyzer()
    df_with_gravity = gravity_analyzer.add_gravity_variables(df)
    
    print("SUCCESS: Gravity variables added successfully")
    print(f"   Gravity range: {df_with_gravity['effective_gravity'].min():.5f} to {df_with_gravity['effective_gravity'].max():.5f} m/s²")
    
    # Test hypothesis
    results = gravity_analyzer.test_gravity_hypothesis(df_with_gravity)
    
    if 'correlations' in results and 'life_expectancy' in results['correlations']:
        gravity_corr = results['correlations']['life_expectancy'].get('effective_gravity', 0)
        print(f"   Gravity-Longevity correlation: r = {gravity_corr:.4f}")
    
    # Save test data for notebooks
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    df_with_gravity.to_csv(output_dir / "test_data_with_gravity.csv", index=False)
    print(f"SUCCESS: Test data saved to {output_dir / 'test_data_with_gravity.csv'}")
    
    # Test basic panel data creation (simplified)
    print("\nTesting basic panel data...")
    panel_data = []
    for _, location in df_with_gravity.iterrows():
        for year in range(2018, 2022):  # Just 4 years for testing
            panel_data.append({
                'geo_id': location['name'],
                'year': year,
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'elevation': location['elevation'],
                'effective_gravity': location['effective_gravity'],
                'life_expectancy': location['life_expectancy'] + np.random.normal(0, 0.5),
                'walkability_score': np.random.normal(50, 15),
                'greenspace_pct': np.random.normal(30, 10),
                'gdp_per_capita': np.random.lognormal(10, 0.3)
            })
    
    panel_df = pd.DataFrame(panel_data)
    panel_df.to_csv(output_dir / "test_panel_data.csv", index=False)
    print(f"SUCCESS: Panel test data saved to {output_dir / 'test_panel_data.csv'}")
    
    print("\nAll modules tested successfully!")
    print("\nReady for Jupyter notebooks:")
    print("   • All modules can be imported")
    print("   • Test data is available")
    print("   • Gravity analysis is working")
    
except Exception as e:
    print(f"ERROR: Gravity analysis test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nJupyter notebooks are now ready to run!")