#!/usr/bin/env python3
"""
analyze all features systematically to find what actually works
create visualizations and build predictive tools
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
# Suppress specific warnings only when necessary
# warnings.filterwarnings('ignore', category=DeprecationWarning)

def load_all_data():
    """
    load all available data sources
    """
    data_sources = []
    
    # load real world data
    try:
        real_data = pd.read_csv('../outputs/real_world_data.csv')
        print(f"loaded {len(real_data)} countries from real world data")
        data_sources.append(real_data)
    except:
        pass
    
    # check for any other data files
    import os
    output_dir = '../outputs'
    for file in os.listdir(output_dir):
        if file.endswith('.csv') and 'real' in file:
            try:
                df = pd.read_csv(os.path.join(output_dir, file))
                if 'life_expectancy' in df.columns:
                    print(f"found additional data: {file} ({len(df)} rows)")
            except:
                pass
    
    # combine all data
    if data_sources:
        combined = pd.concat(data_sources, ignore_index=True)
        # remove duplicates
        if 'geo_id' in combined.columns:
            combined = combined.drop_duplicates(subset=['geo_id'])
        return combined
    
    return pd.DataFrame()

def comprehensive_correlation_analysis(df):
    """
    analyze correlations for all features
    """
    print("\ncomprehensive correlation analysis")
    print("=" * 60)
    
    # get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # remove id-like columns
    exclude = ['year', 'is_blue_zone']
    feature_cols = [col for col in numeric_cols if col not in exclude]
    
    # calculate correlations with life expectancy
    correlations = {}
    p_values = {}
    
    for col in feature_cols:
        if col != 'life_expectancy':
            # clean data
            mask = df[col].notna() & df['life_expectancy'].notna()
            if mask.sum() > 3:
                corr, p_val = stats.pearsonr(df[mask][col], df[mask]['life_expectancy'])
                correlations[col] = corr
                p_values[col] = p_val
    
    # sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\ntop correlations with life expectancy:")
    print("-" * 60)
    print(f"{'feature':<30} {'correlation':>12} {'p-value':>12} {'significance':>15}")
    print("-" * 60)
    
    significant_features = []
    
    for feature, corr in sorted_corr[:20]:
        p_val = p_values[feature]
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = ""
        
        print(f"{feature:<30} {corr:>12.4f} {p_val:>12.4f} {sig:>15}")
        
        if p_val < 0.05:
            significant_features.append((feature, corr))
    
    return significant_features

def create_feature_visualizations(df, significant_features):
    """
    create comprehensive visualizations
    """
    print("\ncreating visualizations...")
    
    # select top features
    top_features = [f[0] for f in significant_features[:8]]
    
    # create figure with subplots
    n_features = len(top_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        # clean data
        mask = df[feature].notna() & df['life_expectancy'].notna()
        x = df[mask][feature]
        y = df[mask]['life_expectancy']
        
        # color by blue zone status
        colors = ['red' if bz else 'blue' for bz in df[mask]['is_blue_zone']]
        
        # scatter plot
        ax.scatter(x, y, c=colors, alpha=0.6, s=50)
        
        # add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'g--', alpha=0.5, linewidth=2)
        
        # labels
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('life expectancy', fontsize=10)
        ax.set_title(f'r = {significant_features[i][1]:.3f}', fontsize=11)
        
        # add r-squared
        r2 = significant_features[i][1] ** 2
        ax.text(0.05, 0.95, f'r² = {r2:.3f}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=9)
    
    # remove empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('significant features for longevity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # save figure
    output_dir = '../outputs/figures'
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'significant_features.png'), dpi=150)
    print(f"saved visualization to {output_dir}/significant_features.png")
    plt.close()
    
    # create correlation heatmap
    create_correlation_heatmap(df, top_features)

def create_correlation_heatmap(df, features):
    """
    create correlation heatmap for top features
    """
    import os
    
    # prepare data
    feature_subset = features + ['life_expectancy']
    clean_df = df[feature_subset].dropna()
    
    if len(clean_df) < 5:
        print("insufficient data for heatmap")
        return
    
    # calculate correlation matrix
    corr_matrix = clean_df.corr()
    
    # create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    
    plt.title('feature correlation matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # save
    output_dir = '../outputs/figures'
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    print(f"saved heatmap to {output_dir}/correlation_heatmap.png")
    plt.close()

def build_prediction_models(df, significant_features):
    """
    build and evaluate prediction models
    """
    print("\nbuilding prediction models")
    print("=" * 60)
    
    # prepare features
    feature_names = [f[0] for f in significant_features[:10]]  # top 10 features
    
    # filter to available columns
    available_features = [f for f in feature_names if f in df.columns]
    
    # create clean dataset
    clean_df = df[available_features + ['life_expectancy']].dropna()
    
    if len(clean_df) < 10:
        print("insufficient data for modeling")
        return None
    
    print(f"using {len(clean_df)} samples with {len(available_features)} features")
    print(f"features: {', '.join(available_features)}")
    
    x = clean_df[available_features]
    y = clean_df['life_expectancy']
    
    # standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # test multiple models
    models = {
        'linear regression': LinearRegression(),
        'ridge regression': Ridge(alpha=1.0),
        'random forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'gradient boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    
    print("\nmodel evaluation (5-fold cross-validation):")
    print("-" * 50)
    
    kfold = KFold(n_splits=min(5, len(clean_df)), shuffle=True, random_state=42)
    
    for name, model in models.items():
        # cross-validation
        scores = cross_val_score(model, x_scaled, y, cv=kfold, 
                                scoring='r2', n_jobs=-1)
        
        # fit on full data for feature importance
        model.fit(x_scaled, y)
        y_pred = model.predict(x_scaled)
        mae = mean_absolute_error(y, y_pred)
        
        results[name] = {
            'cv_r2': scores.mean(),
            'cv_std': scores.std(),
            'mae': mae,
            'model': model
        }
        
        print(f"{name:<20} r² = {scores.mean():.4f} (±{scores.std():.4f}), mae = {mae:.2f} years")
        
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = model
    
    # feature importance from best model
    print("\nfeature importance (from best model):")
    print("-" * 40)
    
    if hasattr(best_model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': available_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importance.head(10).iterrows():
            print(f"{row['feature']:<25} {row['importance']:.4f}")
    elif hasattr(best_model, 'coef_'):
        # for linear models
        importance = pd.DataFrame({
            'feature': available_features,
            'coefficient': best_model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        for _, row in importance.head(10).iterrows():
            print(f"{row['feature']:<25} {row['coefficient']:.4f}")
    
    return best_model, scaler, available_features

def create_prediction_tool(model, scaler, features):
    """
    create a practical prediction tool
    """
    print("\ncreating prediction tool")
    print("=" * 60)
    
    def predict_life_expectancy(**kwargs):
        """
        predict life expectancy based on key features
        """
        # prepare input
        input_data = []
        for feature in features:
            if feature in kwargs:
                input_data.append(kwargs[feature])
            else:
                # use mean value if not provided
                input_data.append(0)  # scaled mean is 0
        
        # scale input
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # predict
        prediction = model.predict(input_scaled)[0]
        
        return prediction
    
    # test the tool
    print("\ntesting prediction tool:")
    print("-" * 40)
    
    # example predictions
    test_cases = [
        {'gdp_per_capita': 50000, 'health_exp_per_capita': 5000, 'urban_pop_pct': 80},
        {'gdp_per_capita': 20000, 'health_exp_per_capita': 1000, 'urban_pop_pct': 60},
        {'gdp_per_capita': 5000, 'health_exp_per_capita': 200, 'urban_pop_pct': 40},
    ]
    
    for i, test in enumerate(test_cases, 1):
        # fill missing features with means
        full_test = {f: 0 for f in features}
        full_test.update(test)
        
        prediction = predict_life_expectancy(**full_test)
        print(f"test {i}: gdp=${test['gdp_per_capita']:,} → life expectancy = {prediction:.1f} years")
    
    return predict_life_expectancy

def identify_actionable_features(df, significant_features):
    """
    identify which features are actionable for policy
    """
    print("\nactionable features for policy")
    print("=" * 60)
    
    # categorize features
    actionable = {
        'economic': ['gdp_per_capita', 'health_exp_per_capita'],
        'environmental': ['forest_area_pct', 'co2_emissions', 'urban_pop_pct'],
        'healthcare': ['physicians_per_1000', 'hospital_beds_per_1000'],
        'social': ['mortality_rate']
    }
    
    # fixed features (not actionable)
    fixed = ['latitude', 'longitude', 'effective_gravity', 'temperature_est', 'latitude_abs']
    
    print("\nactionable features with strong correlations:")
    print("-" * 50)
    
    for category, features in actionable.items():
        print(f"\n{category.upper()}:")
        for feature, corr in significant_features:
            if feature in features:
                direction = "increase" if corr > 0 else "decrease"
                impact = abs(corr)
                if impact > 0.5:
                    strength = "strong"
                elif impact > 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                print(f"  {feature}: {direction} for longevity ({strength} effect, r={corr:.3f})")

def create_blue_zone_scorer(df):
    """
    create a scoring system for blue zone potential
    """
    print("\nblue zone scoring system")
    print("=" * 60)
    
    # define scoring criteria based on analysis
    criteria = {
        'gdp_per_capita': {'range': (10000, 30000), 'weight': 2},
        'forest_area_pct': {'range': (30, 70), 'weight': 3},
        'urban_pop_pct': {'range': (60, 90), 'weight': 2},
        'health_exp_per_capita': {'range': (500, 2000), 'weight': 2},
        'mortality_rate': {'range': (4, 7), 'weight': 1}
    }
    
    def calculate_blue_zone_score(row):
        """
        calculate blue zone potential score
        """
        score = 0
        max_score = 0
        
        for feature, params in criteria.items():
            if feature in row.index and not pd.isna(row[feature]):
                value = row[feature]
                min_val, max_val = params['range']
                weight = params['weight']
                
                # calculate score for this feature
                if min_val <= value <= max_val:
                    # perfect score if in range
                    feature_score = 1.0
                else:
                    # partial score based on distance
                    if value < min_val:
                        distance = (min_val - value) / min_val
                    else:
                        distance = (value - max_val) / max_val
                    feature_score = max(0, 1 - distance)
                
                score += feature_score * weight
                max_score += weight
        
        if max_score > 0:
            return score / max_score * 100
        return 0
    
    # apply scoring to all countries
    df['blue_zone_score'] = df.apply(calculate_blue_zone_score, axis=1)
    
    # rank countries
    ranked = df.sort_values('blue_zone_score', ascending=False)
    
    print("\ntop blue zone candidates by score:")
    print("-" * 60)
    print(f"{'rank':<5} {'country':<25} {'score':<10} {'life exp':<10}")
    print("-" * 60)
    
    for i, (_, row) in enumerate(ranked.head(10).iterrows(), 1):
        print(f"{i:<5} {row['geo_id']:<25} {row['blue_zone_score']:<10.1f} {row['life_expectancy']:<10.1f}")
    
    return ranked

def main():
    print("finding what works for longevity")
    print("=" * 60)
    
    # load all available data
    df = load_all_data()
    
    if df.empty:
        print("no data available")
        return
    
    print(f"\ntotal samples: {len(df)}")
    
    # comprehensive correlation analysis
    significant_features = comprehensive_correlation_analysis(df)
    
    # create visualizations
    if significant_features:
        create_feature_visualizations(df, significant_features)
    
    # build prediction models
    model, scaler, features = build_prediction_models(df, significant_features)
    
    if model is not None:
        # create prediction tool
        predict_tool = create_prediction_tool(model, scaler, features)
    
    # identify actionable features
    identify_actionable_features(df, significant_features)
    
    # create blue zone scoring system
    scored_df = create_blue_zone_scorer(df)
    
    # save results
    output_file = '../outputs/longevity_analysis_results.csv'
    scored_df.to_csv(output_file, index=False)
    print(f"\nresults saved to {output_file}")
    
    print("\n" + "=" * 60)
    print("analysis complete!")
    print("=" * 60)
    
    return scored_df

if __name__ == '__main__':
    results = main()