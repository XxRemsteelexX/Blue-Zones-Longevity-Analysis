#!/usr/bin/env python3
"""
practical longevity prediction tool based on real data analysis
uses the strongest predictive features identified
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os

class LongevityPredictor:
    """
    predict life expectancy based on key country characteristics
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = [
            'physicians_per_1000',      # strongest predictor (r=0.856)
            'urban_pop_pct',            # strong predictor (r=0.791) 
            'gdp_per_capita',           # strong predictor (r=0.785)
            'health_exp_per_capita',    # strong predictor (r=0.756)
            'hospital_beds_per_1000',   # moderate predictor (r=0.535)
        ]
        self.feature_descriptions = {
            'physicians_per_1000': 'physicians per 1000 people',
            'urban_pop_pct': 'urban population percentage',
            'gdp_per_capita': 'gdp per capita (usd)',
            'health_exp_per_capita': 'health expenditure per capita (usd)',
            'hospital_beds_per_1000': 'hospital beds per 1000 people'
        }
        self.trained = False
    
    def train(self, data_file='../outputs/real_world_data.csv'):
        """
        train the model on real world data
        """
        try:
            df = pd.read_csv(data_file)
        except:
            print("error: cannot load training data")
            return False
        
        # prepare training data
        feature_data = []
        target_data = []
        
        for _, row in df.iterrows():
            # check if all features are available
            if all(f in row and not pd.isna(row[f]) for f in self.features):
                if not pd.isna(row['life_expectancy']):
                    feature_data.append([row[f] for f in self.features])
                    target_data.append(row['life_expectancy'])
        
        if len(feature_data) < 5:
            print(f"error: insufficient training data ({len(feature_data)} samples)")
            return False
        
        # convert to arrays
        x = np.array(feature_data)
        y = np.array(target_data)
        
        print(f"training on {len(x)} samples")
        
        # standardize features
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)
        
        # train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(x_scaled, y)
        
        # calculate training accuracy
        y_pred = self.model.predict(x_scaled)
        mae = np.mean(np.abs(y - y_pred))
        r2 = self.model.score(x_scaled, y)
        
        print(f"training completed:")
        print(f"  r-squared: {r2:.3f}")
        print(f"  mean absolute error: {mae:.1f} years")
        
        self.trained = True
        return True
    
    def predict(self, **features):
        """
        predict life expectancy from input features
        """
        if not self.trained:
            print("error: model not trained")
            return None
        
        # prepare input
        input_data = []
        missing_features = []
        
        for feature in self.features:
            if feature in features:
                input_data.append(features[feature])
            else:
                missing_features.append(feature)
        
        if missing_features:
            print(f"warning: missing features: {', '.join(missing_features)}")
            return None
        
        # scale input
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = self.scaler.transform(input_array)
        
        # predict
        prediction = self.model.predict(input_scaled)[0]
        
        return prediction
    
    def analyze_feature_importance(self):
        """
        show which features matter most
        """
        if not self.trained:
            print("error: model not trained")
            return
        
        importance = self.model.feature_importances_
        
        print("\nfeature importance:")
        print("-" * 50)
        for i, feature in enumerate(self.features):
            desc = self.feature_descriptions[feature]
            print(f"{desc:<35} {importance[i]:.3f}")
    
    def save_model(self, filename='longevity_model.pkl'):
        """
        save trained model to file
        """
        if not self.trained:
            print("error: no trained model to save")
            return False
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'descriptions': self.feature_descriptions
        }
        
        output_dir = '../outputs'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"model saved to {filepath}")
        return True
    
    def load_model(self, filename='longevity_model.pkl'):
        """
        load trained model from file
        """
        filepath = os.path.join('../outputs', filename)
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            self.feature_descriptions = model_data['descriptions']
            self.trained = True
            
            print(f"model loaded from {filepath}")
            return True
        except:
            print(f"error: cannot load model from {filepath}")
            return False

def create_blue_zone_finder():
    """
    create a tool to identify potential blue zones
    """
    print("\ncreating blue zone finder tool")
    print("=" * 50)
    
    # load data
    try:
        df = pd.read_csv('../outputs/real_world_data.csv')
    except:
        print("error: cannot load data")
        return None
    
    # define blue zone criteria based on analysis
    criteria = {
        'physicians_per_1000': {'min': 2.0, 'ideal': 4.0, 'max': 6.0},
        'urban_pop_pct': {'min': 60, 'ideal': 80, 'max': 95},
        'gdp_per_capita': {'min': 15000, 'ideal': 25000, 'max': 40000},
        'health_exp_per_capita': {'min': 1000, 'ideal': 3000, 'max': 6000},
        'forest_area_pct': {'min': 20, 'ideal': 50, 'max': 80}
    }
    
    def score_location(data):
        """
        score a location's blue zone potential
        """
        total_score = 0
        max_possible = 0
        
        for feature, ranges in criteria.items():
            if feature in data and not pd.isna(data[feature]):
                value = data[feature]
                
                # calculate score for this feature
                if ranges['min'] <= value <= ranges['max']:
                    # within acceptable range
                    if value <= ranges['ideal']:
                        # below ideal - score based on distance from min
                        score = (value - ranges['min']) / (ranges['ideal'] - ranges['min'])
                    else:
                        # above ideal - score based on distance from max
                        score = (ranges['max'] - value) / (ranges['max'] - ranges['ideal'])
                    
                    total_score += max(0, score)
                
                max_possible += 1
        
        if max_possible > 0:
            return (total_score / max_possible) * 100
        return 0
    
    # score all locations
    scores = []
    for _, row in df.iterrows():
        score = score_location(row)
        if score > 0:
            scores.append({
                'country': row['geo_id'],
                'score': score,
                'life_expectancy': row['life_expectancy'],
                'is_blue_zone': row['is_blue_zone']
            })
    
    # sort by score
    scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    
    print("\ntop blue zone candidates:")
    print("-" * 60)
    print(f"{'rank':<5} {'country':<25} {'score':<10} {'life exp':<12} {'known bz':<10}")
    print("-" * 60)
    
    for i, location in enumerate(scores[:15], 1):
        bz_status = "yes" if location['is_blue_zone'] else "no"
        print(f"{i:<5} {location['country']:<25} {location['score']:<10.1f} "
              f"{location['life_expectancy']:<12.1f} {bz_status:<10}")
    
    return scores

def demo_tool():
    """
    demonstrate the longevity prediction tool
    """
    print("\nlongevity prediction tool demo")
    print("=" * 50)
    
    # create predictor
    predictor = LongevityPredictor()
    
    # train on real data
    if not predictor.train():
        return
    
    # show feature importance
    predictor.analyze_feature_importance()
    
    # save model
    predictor.save_model()
    
    print("\ndemonstration predictions:")
    print("-" * 50)
    
    # test scenarios
    scenarios = [
        {
            'name': 'high-income developed country',
            'physicians_per_1000': 4.5,
            'urban_pop_pct': 85,
            'gdp_per_capita': 45000,
            'health_exp_per_capita': 5000,
            'hospital_beds_per_1000': 4.0
        },
        {
            'name': 'middle-income country',
            'physicians_per_1000': 2.0,
            'urban_pop_pct': 65,
            'gdp_per_capita': 15000,
            'health_exp_per_capita': 1500,
            'hospital_beds_per_1000': 2.0
        },
        {
            'name': 'low-income country',
            'physicians_per_1000': 0.5,
            'urban_pop_pct': 40,
            'gdp_per_capita': 3000,
            'health_exp_per_capita': 200,
            'hospital_beds_per_1000': 0.8
        },
        {
            'name': 'optimal blue zone profile',
            'physicians_per_1000': 3.0,
            'urban_pop_pct': 75,
            'gdp_per_capita': 25000,
            'health_exp_per_capita': 2000,
            'hospital_beds_per_1000': 3.0
        }
    ]
    
    for scenario in scenarios:
        name = scenario.pop('name')
        prediction = predictor.predict(**scenario)
        if prediction:
            print(f"{name}: {prediction:.1f} years")
    
    return predictor

def main():
    print("longevity prediction and blue zone identification tools")
    print("=" * 60)
    
    # create and demo prediction tool
    predictor = demo_tool()
    
    # create blue zone finder
    scores = create_blue_zone_finder()
    
    print("\n" + "=" * 60)
    print("key findings from real data analysis:")
    print("=" * 60)
    
    print("\n1. strongest predictors of longevity:")
    print("   - physicians per 1000 people (r = 0.856)")
    print("   - urban population percentage (r = 0.791)")
    print("   - gdp per capita (r = 0.785)")
    print("   - health expenditure per capita (r = 0.756)")
    
    print("\n2. actionable policy recommendations:")
    print("   - increase physician density")
    print("   - improve urban planning and services")
    print("   - invest in healthcare infrastructure")
    print("   - focus on economic development")
    
    print("\n3. gravity effect:")
    print("   - weak correlation (r = 0.616) but likely spurious")
    print("   - dominated by economic and healthcare factors")
    print("   - not a practical target for intervention")
    
    print("\n4. practical tools created:")
    print("   - longevity prediction model (saved as longevity_model.pkl)")
    print("   - blue zone identification scoring system")
    print("   - feature importance analysis")
    
    return predictor, scores

if __name__ == '__main__':
    predictor, scores = main()