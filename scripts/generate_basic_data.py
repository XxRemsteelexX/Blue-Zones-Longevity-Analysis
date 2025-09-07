#!/usr/bin/env python3
"""
Generate basic data for Blue Zones analysis
Simple data generation without complex causal inference
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.gravity_hypothesis import GravityLongevityAnalyzer

def main():
    """Generate basic data for notebooks"""
    print("Generating basic Blue Zones data...")
    
    # Create output directories
    output_dir = Path("outputs")
    for subdir in ["figures", "tables", "reports"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    gravity_analyzer = GravityLongevityAnalyzer()
    
    # Create comprehensive sample data
    np.random.seed(42)
    
    # Blue Zone locations
    blue_zones = [
        {'name': 'Nicoya', 'latitude': 10.2, 'longitude': -85.4, 'elevation': 200, 'is_blue_zone': 1},
        {'name': 'Okinawa', 'latitude': 26.3, 'longitude': 127.9, 'elevation': 50, 'is_blue_zone': 1},
        {'name': 'Sardinia', 'latitude': 40.1, 'longitude': 9.4, 'elevation': 300, 'is_blue_zone': 1},
        {'name': 'Ikaria', 'latitude': 37.6, 'longitude': 26.2, 'elevation': 400, 'is_blue_zone': 1},
        {'name': 'Loma Linda', 'latitude': 34.0, 'longitude': -117.3, 'elevation': 350, 'is_blue_zone': 1},
    ]
    
    # Generate additional global locations
    additional_locations = []
    for i in range(95):
        latitude = np.random.uniform(-60, 70)
        longitude = np.random.uniform(-180, 180)
        elevation = max(0, np.random.exponential(200))
        
        additional_locations.append({
            'name': f'Location_{i}',
            'latitude': latitude,
            'longitude': longitude,
            'elevation': elevation,
            'is_blue_zone': 0
        })
    
    # Combine locations
    all_locations = blue_zones + additional_locations
    
    # Create panel data (multiple years per location)
    panel_data = []
    years = range(2000, 2021)
    
    for location in all_locations:
        for year in years:
            # Calculate effective gravity
            effective_gravity = gravity_analyzer.calculate_effective_gravity(
                location['latitude'], location['elevation']
            )
            
            # Simulate life expectancy with gravity effect
            base_le = 75 + np.random.normal(0, 3)
            gravity_effect = (9.80665 - effective_gravity) * 100  # Negative gravity effect
            blue_zone_bonus = 3 if location['is_blue_zone'] else 0
            
            life_expectancy = base_le + gravity_effect + blue_zone_bonus + np.random.normal(0, 1)
            life_expectancy = np.clip(life_expectancy, 60, 95)
            
            # Other variables
            walkability_score = np.random.normal(50, 15)
            greenspace_pct = np.random.normal(30, 10)
            gdp_per_capita = np.random.lognormal(10, 0.5)
            population_density = np.random.lognormal(6, 1)
            temperature_mean = 20 - abs(location['latitude']) / 3 + np.random.normal(0, 5)
            
            # CVD mortality (inversely related to life expectancy)
            cvd_mortality = max(0, (90 - life_expectancy) * 10 + np.random.normal(0, 50))
            
            panel_data.append({
                'geo_id': location['name'],
                'year': year,
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'elevation': location['elevation'],
                'is_blue_zone': location['is_blue_zone'],
                'life_expectancy': life_expectancy,
                'cvd_mortality': cvd_mortality,
                'walkability_score': walkability_score,
                'greenspace_pct': greenspace_pct,
                'gdp_per_capita': gdp_per_capita,
                'population_density_log': np.log(population_density),
                'temperature_mean': temperature_mean,
                'effective_gravity': effective_gravity
            })
    
    # Create DataFrame
    df = pd.DataFrame(panel_data)
    
    # Add gravity variables
    df = gravity_analyzer.add_gravity_variables(df)
    
    # Test gravity hypothesis
    test_data = df.groupby('geo_id').first().reset_index()
    gravity_results = gravity_analyzer.test_gravity_hypothesis(
        test_data, outcome_vars=['life_expectancy', 'cvd_mortality']
    )
    
    # Save data
    df.to_csv(output_dir / "comprehensive_panel_data.csv", index=False)
    test_data.to_csv(output_dir / "cross_section_data.csv", index=False)
    
    # Generate basic report
    report = gravity_analyzer.generate_gravity_report(gravity_results)
    with open(output_dir / "reports" / "basic_gravity_analysis.txt", "w") as f:
        f.write(report)
    
    # Create visualization
    fig = gravity_analyzer.visualize_gravity_patterns(
        test_data, save_path=output_dir / "figures" / "gravity_analysis.html"
    )
    
    print("SUCCESS: Data generation complete!")
    print(f"Generated {len(df)} panel observations ({len(test_data)} unique locations)")
    print(f"Blue Zones: {test_data['is_blue_zone'].sum()}")
    print(f"Years: {df['year'].min()}-{df['year'].max()}")
    print()
    print("Files created:")
    print(f"- {output_dir}/comprehensive_panel_data.csv - Full panel dataset")
    print(f"- {output_dir}/cross_section_data.csv - Cross-sectional data")
    print(f"- {output_dir}/figures/gravity_analysis.html - Interactive visualization")
    print(f"- {output_dir}/reports/basic_gravity_analysis.txt - Analysis report")
    print()
    print("Data is ready for Jupyter notebooks!")

if __name__ == "__main__":
    main()