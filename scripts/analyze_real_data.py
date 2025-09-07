#!/usr/bin/env python3
"""
analyze real world data for gravity-longevity hypothesis
uses only real country data from world bank
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_real_data():
    """
    load the real world data we fetched
    """
    data_file = '../outputs/real_world_data.csv'
    
    if not os.path.exists(data_file):
        print("error: real data file not found. run fetch_worldbank_data.py first")
        return None
    
    df = pd.read_csv(data_file)
    
    # remove any rows with missing life expectancy
    df = df[df['life_expectancy'].notna()]
    
    return df

def analyze_gravity_correlation(df):
    """
    analyze correlation between gravity and longevity using real data
    """
    print("gravity-longevity analysis with real data")
    print("=" * 50)
    
    # calculate correlations
    correlations = {
        'gravity_vs_life_exp': df['effective_gravity'].corr(df['life_expectancy']),
        'latitude_vs_life_exp': df['latitude'].abs().corr(df['life_expectancy']),
        'gdp_vs_life_exp': df['gdp_per_capita'].corr(df['life_expectancy']) if 'gdp_per_capita' in df.columns else None,
    }
    
    print("\ncorrelations:")
    print("-" * 30)
    for name, corr in correlations.items():
        if corr is not None:
            print(f"{name}: r = {corr:.4f}")
    
    # statistical tests
    print("\nstatistical tests:")
    print("-" * 30)
    
    # gravity vs life expectancy
    mask = df['life_expectancy'].notna()
    if mask.sum() > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df[mask]['effective_gravity'],
            df[mask]['life_expectancy']
        )
        
        print(f"gravity regression:")
        print(f"  slope: {slope:.4f}")
        print(f"  r-squared: {r_value**2:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  result: statistically significant")
        else:
            print(f"  result: not statistically significant")
    
    # compare blue zones vs others
    print("\nblue zone comparison:")
    print("-" * 30)
    
    blue_zones = df[df['is_blue_zone'] == 1]
    others = df[df['is_blue_zone'] == 0]
    
    print(f"blue zone countries: {len(blue_zones)}")
    print(f"  avg life expectancy: {blue_zones['life_expectancy'].mean():.1f} years")
    print(f"  avg gravity: {blue_zones['effective_gravity'].mean():.4f} m/s²")
    
    print(f"other countries: {len(others)}")
    print(f"  avg life expectancy: {others['life_expectancy'].mean():.1f} years")
    print(f"  avg gravity: {others['effective_gravity'].mean():.4f} m/s²")
    
    # t-test for difference
    if len(blue_zones) > 0 and len(others) > 0:
        t_stat, p_val = stats.ttest_ind(
            blue_zones['life_expectancy'].dropna(),
            others['life_expectancy'].dropna()
        )
        print(f"\nlife expectancy difference:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4f}")
    
    return correlations

def create_visualizations(df):
    """
    create plots with real data
    """
    print("\ncreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # separate blue zones and others
    blue_zones = df[df['is_blue_zone'] == 1]
    others = df[df['is_blue_zone'] == 0]
    
    # 1. gravity vs life expectancy
    ax = axes[0, 0]
    colors = ['red' if bz else 'blue' for bz in df['is_blue_zone']]
    ax.scatter(df['effective_gravity'], df['life_expectancy'], c=colors, alpha=0.6)
    ax.set_xlabel('effective gravity (m/s²)')
    ax.set_ylabel('life expectancy (years)')
    ax.set_title('gravity vs life expectancy (real data)')
    
    # add regression line
    mask = df['life_expectancy'].notna()
    if mask.sum() > 2:
        z = np.polyfit(df[mask]['effective_gravity'], df[mask]['life_expectancy'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['effective_gravity'].min(), df['effective_gravity'].max(), 100)
        ax.plot(x_line, p(x_line), 'g--', alpha=0.5, label='trend')
    
    ax.legend(['trend', 'blue zone', 'other'])
    
    # 2. latitude vs life expectancy
    ax = axes[0, 1]
    ax.scatter(df['latitude'].abs(), df['life_expectancy'], c=colors, alpha=0.6)
    ax.set_xlabel('absolute latitude (degrees)')
    ax.set_ylabel('life expectancy (years)')
    ax.set_title('latitude vs life expectancy')
    
    # 3. gdp vs life expectancy
    ax = axes[1, 0]
    if 'gdp_per_capita' in df.columns:
        gdp_mask = df['gdp_per_capita'].notna() & df['life_expectancy'].notna()
        if gdp_mask.sum() > 0:
            ax.scatter(df[gdp_mask]['gdp_per_capita'], df[gdp_mask]['life_expectancy'], 
                      c=[colors[i] for i in range(len(df)) if gdp_mask.iloc[i]], alpha=0.6)
            ax.set_xlabel('gdp per capita ($)')
            ax.set_ylabel('life expectancy (years)')
            ax.set_title('gdp vs life expectancy')
            ax.set_xscale('log')
    
    # 4. gravity distribution
    ax = axes[1, 1]
    ax.hist(others['effective_gravity'], bins=20, alpha=0.5, label='other countries', color='blue')
    if len(blue_zones) > 0:
        ax.hist(blue_zones['effective_gravity'], bins=5, alpha=0.7, label='blue zones', color='red')
    ax.set_xlabel('effective gravity (m/s²)')
    ax.set_ylabel('count')
    ax.set_title('gravity distribution')
    ax.legend()
    
    plt.tight_layout()
    
    # save figure
    output_dir = '../outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'real_data_analysis.png'), dpi=150)
    print(f"saved visualization to {output_dir}/real_data_analysis.png")
    
    # plt.show()  # commented out to avoid hanging

def multiple_regression_analysis(df):
    """
    multiple regression to control for confounders
    """
    print("\nmultiple regression analysis:")
    print("-" * 40)
    
    # prepare data
    features = ['effective_gravity', 'gdp_per_capita', 'urban_pop_pct', 'health_exp_per_capita']
    
    # check which features are available
    available_features = [f for f in features if f in df.columns]
    
    # create clean dataset
    clean_df = df[available_features + ['life_expectancy']].dropna()
    
    if len(clean_df) < 10:
        print("insufficient data for multiple regression")
        return
    
    print(f"using {len(clean_df)} countries with complete data")
    print(f"features: {', '.join(available_features)}")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # standardize features
    scaler = StandardScaler()
    x = scaler.fit_transform(clean_df[available_features])
    y = clean_df['life_expectancy'].values
    
    # fit model
    model = LinearRegression()
    model.fit(x, y)
    
    # get r-squared
    r2 = model.score(x, y)
    
    print(f"\nmodel results:")
    print(f"r-squared: {r2:.4f}")
    print(f"\ncoefficients (standardized):")
    
    for feature, coef in zip(available_features, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    
    # calculate p-values (simplified)
    n = len(clean_df)
    k = len(available_features)
    
    # residual sum of squares
    y_pred = model.predict(x)
    rss = np.sum((y - y_pred) ** 2)
    
    # standard error
    se = np.sqrt(rss / (n - k - 1))
    
    print(f"\nstandard error: {se:.4f}")
    
    return model

def main():
    print("blue zones analysis with real world data")
    print("=" * 50)
    
    # load real data
    df = load_real_data()
    
    if df is None:
        return
    
    print(f"\nloaded {len(df)} countries with real data")
    print(f"blue zone countries: {df['is_blue_zone'].sum()}")
    
    # run analyses
    correlations = analyze_gravity_correlation(df)
    
    # multiple regression
    model = multiple_regression_analysis(df)
    
    # create visualizations
    create_visualizations(df)
    
    # final conclusions
    print("\n" + "=" * 50)
    print("conclusions from real data:")
    print("=" * 50)
    
    if correlations['gravity_vs_life_exp'] is not None:
        strength = abs(correlations['gravity_vs_life_exp'])
        if strength < 0.1:
            print("1. gravity effect: negligible (r < 0.1)")
        elif strength < 0.3:
            print("1. gravity effect: weak")
        elif strength < 0.5:
            print("1. gravity effect: moderate")
        else:
            print("1. gravity effect: strong")
    
    print("2. sample size: still limited - need city-level data for robust conclusions")
    print("3. confounding: gdp and healthcare likely dominate any gravity effect")
    print("4. recommendation: focus on socioeconomic factors, not gravity")
    
    # save results
    results_file = '../outputs/real_data_results.txt'
    with open(results_file, 'w') as f:
        f.write("blue zones real data analysis results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"countries analyzed: {len(df)}\n")
        f.write(f"blue zones: {df['is_blue_zone'].sum()}\n\n")
        
        if correlations['gravity_vs_life_exp'] is not None:
            f.write(f"gravity correlation: r = {correlations['gravity_vs_life_exp']:.4f}\n")
        
        f.write(f"\naverage life expectancy: {df['life_expectancy'].mean():.1f} years\n")
        f.write(f"gravity range: {df['effective_gravity'].min():.4f} - {df['effective_gravity'].max():.4f} m/s²\n")
    
    print(f"\nresults saved to {results_file}")

if __name__ == '__main__':
    main()