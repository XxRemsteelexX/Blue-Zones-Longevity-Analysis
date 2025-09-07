#!/usr/bin/env python3
"""
fix blue zone identification - they should be rare exceptional places
not 65% of all locations!
"""

import pandas as pd
import numpy as np
from scipy import stats

def load_expanded_data():
    """
    load the expanded dataset
    """
    try:
        df = pd.read_csv('../outputs/expanded_cities_dataset.csv')
        return df
    except:
        print("error: run expand_sample_size.py first")
        return pd.DataFrame()

def analyze_life_expectancy_distribution(df):
    """
    understand the actual distribution
    """
    print("analyzing life expectancy distribution")
    print("=" * 50)
    
    le = df['life_expectancy']
    
    print(f"total cities: {len(df)}")
    print(f"life expectancy statistics:")
    print(f"  mean: {le.mean():.1f} years")
    print(f"  median: {le.median():.1f} years") 
    print(f"  std dev: {le.std():.1f} years")
    print(f"  min: {le.min():.1f} years")
    print(f"  max: {le.max():.1f} years")
    
    # percentiles
    percentiles = [90, 95, 98, 99, 99.5, 99.9]
    print(f"\npercentiles:")
    for p in percentiles:
        value = np.percentile(le, p)
        count = (le >= value).sum()
        print(f"  {p}th percentile: {value:.1f} years ({count} cities, {count/len(df)*100:.1f}%)")
    
    return le

def set_realistic_blue_zone_criteria(df):
    """
    define realistic criteria for true blue zones
    """
    print("\nsetting realistic blue zone criteria")
    print("=" * 50)
    
    # real blue zones are extremely rare - maybe top 1% globally
    le_99th = np.percentile(df['life_expectancy'], 99)
    le_995th = np.percentile(df['life_expectancy'], 99.5)
    le_999th = np.percentile(df['life_expectancy'], 99.9)
    
    print(f"potential thresholds:")
    print(f"  99th percentile (1%): {le_99th:.1f} years")
    print(f"  99.5th percentile (0.5%): {le_995th:.1f} years")
    print(f"  99.9th percentile (0.1%): {le_999th:.1f} years")
    
    # use 99.5th percentile - top 0.5% of locations
    blue_zone_threshold = le_995th
    
    # recalculate blue zones
    df['is_blue_zone'] = (df['life_expectancy'] >= blue_zone_threshold).astype(int)
    
    blue_zone_count = df['is_blue_zone'].sum()
    print(f"\nrevised blue zone identification:")
    print(f"  threshold: {blue_zone_threshold:.1f} years")
    print(f"  blue zones found: {blue_zone_count} ({blue_zone_count/len(df)*100:.2f}%)")
    
    return df, blue_zone_threshold

def analyze_true_blue_zone_characteristics(df):
    """
    analyze what makes the top 0.5% special
    """
    print("\nanalyzing true blue zone characteristics")
    print("=" * 50)
    
    blue_zones = df[df['is_blue_zone'] == 1]
    others = df[df['is_blue_zone'] == 0]
    
    features = [
        'physicians_per_1000',
        'gdp_per_capita', 
        'urban_pop_pct',
        'health_exp_per_capita',
        'hospital_beds_per_1000',
        'effective_gravity',
        'forest_area_pct',
        'temperature_est'
    ]
    
    print(f"comparing {len(blue_zones)} blue zones vs {len(others)} others:")
    print("-" * 70)
    print(f"{'feature':<25} {'blue zones':<12} {'others':<12} {'difference':<12} {'effect size':<12}")
    print("-" * 70)
    
    significant_differences = []
    
    for feature in features:
        if feature in df.columns:
            bz_mean = blue_zones[feature].mean()
            other_mean = others[feature].mean()
            difference = bz_mean - other_mean
            
            # effect size (cohen's d)
            pooled_std = np.sqrt(((len(blue_zones)-1) * blue_zones[feature].var() + 
                                 (len(others)-1) * others[feature].var()) / 
                                (len(blue_zones) + len(others) - 2))
            
            if pooled_std > 0:
                cohens_d = difference / pooled_std
            else:
                cohens_d = 0
            
            # t-test
            t_stat, p_value = stats.ttest_ind(blue_zones[feature], others[feature])
            
            print(f"{feature:<25} {bz_mean:<12.2f} {other_mean:<12.2f} {difference:<12.2f} {cohens_d:<12.2f}")
            
            # large effect size = cohen's d > 0.8
            if abs(cohens_d) > 0.5 and p_value < 0.001:
                significant_differences.append((feature, cohens_d, p_value))
    
    print(f"\nfeatures with large effect sizes (cohen's d > 0.5):")
    for feature, d, p in significant_differences:
        direction = "higher" if d > 0 else "lower"
        print(f"  {feature}: {direction} in blue zones (d={d:.2f}, p<0.001)")
    
    return significant_differences

def create_blue_zone_profile(df):
    """
    create realistic profile for identifying new blue zones
    """
    print("\ncreating blue zone identification profile")
    print("=" * 50)
    
    blue_zones = df[df['is_blue_zone'] == 1]
    
    # key features that distinguish blue zones
    profile_features = [
        'physicians_per_1000',
        'gdp_per_capita',
        'health_exp_per_capita',
        'urban_pop_pct'
    ]
    
    profile = {}
    
    for feature in profile_features:
        values = blue_zones[feature]
        profile[feature] = {
            'min': values.quantile(0.25),  # 25th percentile of blue zones
            'max': values.quantile(0.75),  # 75th percentile of blue zones  
            'mean': values.mean(),
            'median': values.median()
        }
    
    print("blue zone profile ranges (IQR of top 0.5%):")
    for feature, stats in profile.items():
        print(f"{feature}:")
        print(f"  optimal range: {stats['min']:.1f} - {stats['max']:.1f}")
        print(f"  median: {stats['median']:.1f}")
    
    return profile

def identify_realistic_candidates(df, profile):
    """
    identify realistic blue zone candidates
    """
    print("\nidentifying realistic blue zone candidates")
    print("=" * 50)
    
    # exclude existing blue zones
    candidates = df[df['is_blue_zone'] == 0].copy()
    
    # score based on similarity to blue zone profile
    def calculate_similarity_score(row):
        score = 0
        max_score = 0
        
        for feature, ranges in profile.items():
            if feature in row.index and not pd.isna(row[feature]):
                value = row[feature]
                
                # score based on how close to blue zone range
                if ranges['min'] <= value <= ranges['max']:
                    # perfect score if in range
                    feature_score = 1.0
                else:
                    # partial score based on distance
                    center = (ranges['min'] + ranges['max']) / 2
                    range_size = ranges['max'] - ranges['min']
                    if range_size > 0:
                        distance = abs(value - center) / range_size
                        feature_score = max(0, 1 - distance)
                    else:
                        feature_score = 0
                
                score += feature_score
                max_score += 1
        
        return score / max_score if max_score > 0 else 0
    
    candidates['similarity_score'] = candidates.apply(calculate_similarity_score, axis=1)
    
    # filter for high-scoring candidates with decent life expectancy
    high_score = candidates['similarity_score'] > 0.7
    high_life = candidates['life_expectancy'] > np.percentile(df['life_expectancy'], 85)  # top 15%
    
    top_candidates = candidates[high_score & high_life].sort_values('life_expectancy', ascending=False)
    
    print(f"found {len(top_candidates)} realistic blue zone candidates")
    print("\ntop 20 candidates:")
    print("-" * 80)
    print(f"{'rank':<5} {'city_id':<12} {'life_exp':<10} {'score':<8} {'physicians':<11} {'gdp':<8}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(top_candidates.head(20).iterrows(), 1):
        print(f"{i:<5} {row['city_id']:<12} {row['life_expectancy']:<10.1f} "
              f"{row['similarity_score']:<8.2f} {row['physicians_per_1000']:<11.1f} "
              f"{row['gdp_per_capita']:<8.0f}")
    
    return top_candidates

def save_corrected_analysis(df, candidates):
    """
    save the corrected analysis
    """
    # save corrected dataset
    df.to_csv('../outputs/corrected_cities_dataset.csv', index=False)
    
    # save candidates
    candidates.to_csv('../outputs/realistic_blue_zone_candidates.csv', index=False)
    
    print(f"\nsaved corrected analysis:")
    print(f"  full dataset: ../outputs/corrected_cities_dataset.csv")
    print(f"  candidates: ../outputs/realistic_blue_zone_candidates.csv")

def main():
    print("fixing blue zone identification criteria")
    print("=" * 60)
    
    # load data
    df = load_expanded_data()
    if df.empty:
        return
    
    # analyze distribution
    le_dist = analyze_life_expectancy_distribution(df)
    
    # set realistic criteria
    df_corrected, threshold = set_realistic_blue_zone_criteria(df)
    
    # analyze true characteristics
    significant_diffs = analyze_true_blue_zone_characteristics(df_corrected)
    
    # create profile
    profile = create_blue_zone_profile(df_corrected)
    
    # identify candidates
    candidates = identify_realistic_candidates(df_corrected, profile)
    
    # save results
    save_corrected_analysis(df_corrected, candidates)
    
    print(f"\n" + "=" * 60)
    print("corrected blue zone analysis summary")
    print("=" * 60)
    
    blue_zone_count = df_corrected['is_blue_zone'].sum()
    candidate_count = len(candidates)
    
    print(f"total cities analyzed: {len(df_corrected):,}")
    print(f"true blue zones (top 0.5%): {blue_zone_count}")
    print(f"realistic candidates identified: {candidate_count}")
    print(f"life expectancy threshold: {threshold:.1f} years")
    print(f"\nthis makes much more sense - blue zones are rare!")
    
    return df_corrected, candidates

if __name__ == '__main__':
    corrected_df, candidates = main()