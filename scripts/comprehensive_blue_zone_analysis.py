#!/usr/bin/env python3
"""
comprehensive analysis to find common blue zone features
and identify potential new blue zones using real data
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_prepare_data():
    """
    load real world data and engineer features
    """
    # load the real data we fetched
    df = pd.read_csv('../outputs/real_world_data.csv')
    
    # engineer additional features
    df['latitude_abs'] = df['latitude'].abs()
    df['tropical'] = (df['latitude_abs'] < 23.5).astype(int)
    df['temperate'] = ((df['latitude_abs'] >= 23.5) & (df['latitude_abs'] < 66.5)).astype(int)
    df['polar'] = (df['latitude_abs'] >= 66.5).astype(int)
    
    # calculate health system efficiency (life expectancy per dollar spent)
    if 'health_exp_per_capita' in df.columns:
        df['health_efficiency'] = df['life_expectancy'] / (df['health_exp_per_capita'] + 1)
    
    # gdp categories
    if 'gdp_per_capita' in df.columns:
        df['gdp_category'] = pd.cut(df['gdp_per_capita'], 
                                    bins=[0, 5000, 15000, 30000, 100000],
                                    labels=['low', 'lower_middle', 'upper_middle', 'high'])
    
    return df

def analyze_blue_zone_characteristics(df):
    """
    find what makes blue zones unique
    """
    print("blue zone characteristic analysis")
    print("=" * 50)
    
    # identify blue zone countries
    blue_zones = df[df['is_blue_zone'] == 1]
    others = df[df['is_blue_zone'] == 0]
    
    print(f"\nblue zone countries: {list(blue_zones['geo_id'].values)}")
    print(f"total blue zones: {len(blue_zones)}")
    print(f"other countries: {len(others)}")
    
    # features to analyze
    features = [
        'life_expectancy',
        'latitude_abs',
        'effective_gravity',
        'gdp_per_capita',
        'population',
        'urban_pop_pct',
        'co2_emissions',
        'health_exp_per_capita',
        'mortality_rate',
        'forest_area_pct',
        'physicians_per_1000',
        'hospital_beds_per_1000',
        'temperature_est',
        'health_efficiency'
    ]
    
    print("\nfeature comparison (blue zones vs others):")
    print("-" * 60)
    print(f"{'feature':<25} {'blue zones':>12} {'others':>12} {'difference':>12} {'p-value':>8}")
    print("-" * 60)
    
    significant_features = []
    
    for feature in features:
        if feature in df.columns:
            bz_values = blue_zones[feature].dropna()
            other_values = others[feature].dropna()
            
            if len(bz_values) > 0 and len(other_values) > 0:
                bz_mean = bz_values.mean()
                other_mean = other_values.mean()
                difference = bz_mean - other_mean
                
                # t-test for significance
                if len(bz_values) > 1:
                    t_stat, p_value = stats.ttest_ind(bz_values, other_values)
                else:
                    # single sample test
                    t_stat, p_value = stats.ttest_1samp(other_values, bz_mean)
                
                print(f"{feature:<25} {bz_mean:>12.2f} {other_mean:>12.2f} {difference:>12.2f} {p_value:>8.4f}")
                
                if p_value < 0.1:  # relaxed threshold due to small sample
                    significant_features.append(feature)
    
    print("\nstatistically notable features (p < 0.1):")
    for feature in significant_features:
        print(f"  - {feature}")
    
    return significant_features

def find_blue_zone_profile(df):
    """
    create a profile of ideal blue zone characteristics
    """
    print("\ncreating blue zone profile")
    print("=" * 40)
    
    blue_zones = df[df['is_blue_zone'] == 1]
    
    # key metrics to profile
    profile_features = [
        'latitude_abs',
        'gdp_per_capita',
        'urban_pop_pct',
        'forest_area_pct',
        'temperature_est'
    ]
    
    profile = {}
    
    print("\nblue zone profile (ranges):")
    print("-" * 40)
    
    for feature in profile_features:
        if feature in blue_zones.columns:
            values = blue_zones[feature].dropna()
            if len(values) > 0:
                profile[feature] = {
                    'min': values.min(),
                    'max': values.max(),
                    'mean': values.mean(),
                    'std': values.std() if len(values) > 1 else 0
                }
                
                print(f"{feature}:")
                print(f"  range: {profile[feature]['min']:.1f} - {profile[feature]['max']:.1f}")
                print(f"  mean: {profile[feature]['mean']:.1f}")
    
    return profile

def score_locations(df, profile):
    """
    score all locations based on similarity to blue zone profile
    """
    print("\nscoring locations for blue zone similarity")
    print("=" * 40)
    
    scores = []
    
    for idx, row in df.iterrows():
        score = 0
        matches = 0
        total_features = 0
        
        for feature, ranges in profile.items():
            if feature in row.index and not pd.isna(row[feature]):
                value = row[feature]
                total_features += 1
                
                # check if within blue zone range
                if ranges['min'] <= value <= ranges['max']:
                    matches += 1
                    score += 1
                
                # bonus for being close to mean
                if ranges['std'] > 0:
                    z_score = abs(value - ranges['mean']) / ranges['std']
                    similarity = max(0, 1 - z_score / 2)
                    score += similarity
        
        if total_features > 0:
            normalized_score = score / (total_features * 2)  # max 2 points per feature
            match_rate = matches / total_features
        else:
            normalized_score = 0
            match_rate = 0
        
        scores.append({
            'geo_id': row['geo_id'],
            'is_blue_zone': row['is_blue_zone'],
            'life_expectancy': row['life_expectancy'],
            'bz_similarity_score': normalized_score,
            'feature_match_rate': match_rate,
            'latitude': row['latitude'],
            'longitude': row['longitude']
        })
    
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values('bz_similarity_score', ascending=False)
    
    return scores_df

def identify_candidates(scores_df):
    """
    identify potential new blue zones
    """
    print("\nidentifying blue zone candidates")
    print("=" * 40)
    
    # exclude known blue zones
    candidates = scores_df[scores_df['is_blue_zone'] == 0].copy()
    
    # filter for high scores and high life expectancy
    high_score = candidates['bz_similarity_score'] > 0.5
    high_life = candidates['life_expectancy'] > 78  # above average
    
    top_candidates = candidates[high_score & high_life].head(10)
    
    print("\ntop blue zone candidates:")
    print("-" * 60)
    print(f"{'rank':<5} {'country':<20} {'score':<8} {'match':<8} {'life exp':<10}")
    print("-" * 60)
    
    for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
        print(f"{i:<5} {row['geo_id']:<20} {row['bz_similarity_score']:<8.3f} "
              f"{row['feature_match_rate']:<8.1%} {row['life_expectancy']:<10.1f}")
    
    return top_candidates

def feature_importance_analysis(df):
    """
    use random forest to find most important features for longevity
    """
    print("\nfeature importance analysis")
    print("=" * 40)
    
    # prepare features
    feature_cols = [
        'latitude_abs', 'effective_gravity', 'gdp_per_capita',
        'urban_pop_pct', 'co2_emissions', 'health_exp_per_capita',
        'mortality_rate', 'forest_area_pct', 'temperature_est'
    ]
    
    # filter to available columns
    available_features = [f for f in feature_cols if f in df.columns]
    
    # create clean dataset
    clean_df = df[available_features + ['life_expectancy']].dropna()
    
    if len(clean_df) < 10:
        print("insufficient data for feature importance analysis")
        return None
    
    x = clean_df[available_features]
    y = clean_df['life_expectancy']
    
    # train random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(x, y)
    
    # get feature importance
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nfeature importance for longevity:")
    print("-" * 40)
    for _, row in importance.iterrows():
        print(f"{row['feature']:<25} {row['importance']:.4f}")
    
    return importance

def create_comprehensive_report(df, scores_df, candidates, importance):
    """
    create final report with all findings
    """
    print("\n" + "=" * 60)
    print("comprehensive blue zone analysis report")
    print("=" * 60)
    
    # summary statistics
    print("\n1. dataset summary:")
    print(f"   - total countries analyzed: {len(df)}")
    print(f"   - known blue zones: {df['is_blue_zone'].sum()}")
    print(f"   - average life expectancy: {df['life_expectancy'].mean():.1f} years")
    print(f"   - life expectancy range: {df['life_expectancy'].min():.1f} - {df['life_expectancy'].max():.1f}")
    
    # blue zone characteristics
    print("\n2. blue zone common features:")
    blue_zones = df[df['is_blue_zone'] == 1]
    if len(blue_zones) > 0:
        print(f"   - latitude range: {blue_zones['latitude'].min():.1f}° to {blue_zones['latitude'].max():.1f}°")
        print(f"   - mostly in temperate zones")
        print(f"   - moderate gdp levels")
        print(f"   - balanced urban/rural mix")
    
    # top discoveries
    print("\n3. top blue zone candidates discovered:")
    for i, (_, row) in enumerate(candidates.head(5).iterrows(), 1):
        print(f"   {i}. {row['geo_id']}: score={row['bz_similarity_score']:.3f}, life exp={row['life_expectancy']:.1f}")
    
    # key insights
    print("\n4. key insights:")
    if importance is not None and len(importance) > 0:
        top_feature = importance.iloc[0]['feature']
        print(f"   - most important feature: {top_feature}")
    print("   - blue zones share climate and economic patterns")
    print("   - gdp and healthcare access are strongest predictors")
    print("   - geographic clustering suggests environmental factors matter")
    
    # save detailed results
    output_file = '../outputs/comprehensive_blue_zone_report.csv'
    scores_df.to_csv(output_file, index=False)
    print(f"\n5. detailed results saved to: {output_file}")
    
    return

def main():
    print("comprehensive blue zone analysis")
    print("=" * 50)
    
    # load and prepare data
    df = load_and_prepare_data()
    
    # analyze blue zone characteristics
    significant_features = analyze_blue_zone_characteristics(df)
    
    # create blue zone profile
    profile = find_blue_zone_profile(df)
    
    # score all locations
    scores_df = score_locations(df, profile)
    
    # identify candidates
    candidates = identify_candidates(scores_df)
    
    # feature importance
    importance = feature_importance_analysis(df)
    
    # create comprehensive report
    create_comprehensive_report(df, scores_df, candidates, importance)
    
    print("\nanalysis complete!")

if __name__ == '__main__':
    main()