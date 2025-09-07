# PCA to understand patterns in blue zones - FIXED VERSION
print("Principal Component Analysis")
print("=" * 50)

# prepare data for PCA using ACTUAL available columns from the dataset
# Available columns: 'geo_id', 'year', 'latitude', 'longitude', 'elevation', 
#                   'is_blue_zone', 'life_expectancy', 'cvd_mortality', 
#                   'walkability_score', 'greenspace_pct', 'gdp_per_capita', 
#                   'population_density_log', 'temperature_mean', 'effective_gravity', 
#                   'gravity_deviation', 'gravity_deviation_pct', 'equatorial_distance', 
#                   'gravity_x_walkability_score', 'lifetime_gravity_exposure'

pca_features = ['gdp_per_capita', 'walkability_score', 'greenspace_pct', 
               'cvd_mortality', 'population_density_log', 'temperature_mean', 'elevation']

# combine all data for PCA
all_data = df[pca_features + ['is_blue_zone', 'life_expectancy']].dropna()

print(f"\nUsing features for PCA analysis:")
for feature in pca_features:
    print(f"  - {feature}")

print(f"\nData shape after removing missing values: {all_data.shape}")

# standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(all_data[pca_features])

# perform PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(features_scaled)

# explained variance
print(f"\nExplained Variance Ratio:")
for i, var_ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var_ratio:.3f} ({var_ratio*100:.1f}%)")
print(f"  Total: {pca.explained_variance_ratio_.sum():.3f} ({pca.explained_variance_ratio_.sum()*100:.1f}%)")

# feature loadings
print(f"\nFeature Loadings (PC1 and PC2):")
print("-" * 40)
print(f"{'Feature':<25} {'PC1':<8} {'PC2':<8}")
print("-" * 40)
for i, feature in enumerate(pca_features):
    print(f"{feature:<25} {pca.components_[0][i]:<8.3f} {pca.components_[1][i]:<8.3f}")

# add PCA results to dataframe
all_data['PC1'] = pca_result[:, 0]
all_data['PC2'] = pca_result[:, 1]
all_data['PC3'] = pca_result[:, 2]
