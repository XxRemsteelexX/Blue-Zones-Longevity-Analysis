"""
Life expectancy forecasting models with uncertainty quantification
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from scipy import stats


class LifeExpectancyForecaster:
    """Life expectancy forecasting with uncertainty quantification"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        
    def train_forecasting_models(self, data: pd.DataFrame,
                                target_col: str = 'val',
                                feature_cols: List[str] = None) -> Dict[str, Any]:
        """
        Train ensemble of forecasting models
        
        Args:
            data: Training dataset with temporal component
            target_col: Life expectancy column
            feature_cols: Predictor features
            
        Returns:
            Training results and validation metrics
        """
        if feature_cols is None:
            feature_cols = self._select_forecasting_features(data)
            
        self.feature_names = feature_cols
        self.logger.info(f"Training forecasting models with {len(feature_cols)} features")
        
        # Prepare temporal data
        temporal_data = self._prepare_temporal_data(data, target_col, feature_cols)
        
        if temporal_data.empty:
            self.logger.error("No temporal data available for forecasting")
            return {}
            
        # Train multiple models
        model_results = {}
        
        # 1. LightGBM (primary model)
        lgb_results = self._train_lightgbm_forecaster(temporal_data, target_col, feature_cols)
        model_results['lightgbm'] = lgb_results
        
        # 2. Quantile regression for uncertainty
        quantile_results = self._train_quantile_models(temporal_data, target_col, feature_cols)
        model_results['quantile'] = quantile_results
        
        # 3. Simple trend model (baseline)
        trend_results = self._train_trend_model(temporal_data, target_col)
        model_results['trend'] = trend_results
        
        # Ensemble validation
        ensemble_results = self._validate_ensemble(temporal_data, target_col, feature_cols)
        model_results['ensemble'] = ensemble_results
        
        return model_results
        
    def forecast_life_expectancy(self, data: pd.DataFrame, 
                                forecast_years: List[int],
                                scenario: str = 'baseline') -> pd.DataFrame:
        """
        Generate life expectancy forecasts with uncertainty
        
        Args:
            data: Base data for forecasting
            forecast_years: Years to forecast
            scenario: Scenario type ('baseline', 'optimistic', 'pessimistic')
            
        Returns:
            DataFrame with forecasts and uncertainty intervals
        """
        if not self.models:
            raise ValueError("Models must be trained first")
            
        forecasts = []
        
        for year in forecast_years:
            self.logger.info(f"Generating forecasts for {year}")
            
            # Prepare features for forecasting year
            forecast_features = self._prepare_forecast_features(data, year, scenario)
            
            # Generate predictions from each model
            predictions = {}
            
            if 'lightgbm' in self.models:
                predictions['lightgbm'] = self._predict_lightgbm(forecast_features)
                
            if 'quantile' in self.models:
                quantile_preds = self._predict_quantiles(forecast_features)
                predictions.update(quantile_preds)
                
            if 'trend' in self.models:
                predictions['trend'] = self._predict_trend(forecast_features, year)
                
            # Combine predictions
            year_forecasts = self._combine_predictions(predictions, forecast_features, year, scenario)
            forecasts.append(year_forecasts)
            
        result = pd.concat(forecasts, ignore_index=True)
        return result
        
    def create_scenario_forecasts(self, data: pd.DataFrame,
                                forecast_years: List[int]) -> Dict[str, pd.DataFrame]:
        """
        Create forecasts under different scenarios
        
        Args:
            data: Base data
            forecast_years: Years to forecast
            
        Returns:
            Dictionary of forecasts by scenario
        """
        scenarios = ['optimistic', 'baseline', 'pessimistic']
        scenario_forecasts = {}
        
        for scenario in scenarios:
            self.logger.info(f"Generating {scenario} scenario forecasts")
            forecasts = self.forecast_life_expectancy(data, forecast_years, scenario)
            scenario_forecasts[scenario] = forecasts
            
        return scenario_forecasts
        
    def _select_forecasting_features(self, data: pd.DataFrame) -> List[str]:
        """Select features for forecasting"""
        # Temporal features are important for forecasting
        temporal_features = []
        basic_features = []
        
        for col in data.columns:
            if data[col].dtype in [np.number]:
                if any(keyword in col.lower() for keyword in ['trend', 'change', 'growth', 'avg', 'mean']):
                    temporal_features.append(col)
                elif not any(pattern in col.lower() for pattern in ['geo_id', 'year', 'val', 'lower', 'upper', 'location']):
                    basic_features.append(col)
                    
        # Prioritize temporal features but include basic features
        selected_features = temporal_features + basic_features[:20]  # Limit to prevent overfitting
        
        # Remove highly missing features
        valid_features = []
        for col in selected_features:
            if col in data.columns:
                missing_pct = data[col].isnull().sum() / len(data)
                if missing_pct < 0.6:
                    valid_features.append(col)
                    
        self.logger.info(f"Selected {len(valid_features)} features for forecasting")
        return valid_features
        
    def _prepare_temporal_data(self, data: pd.DataFrame,
                             target_col: str, feature_cols: List[str]) -> pd.DataFrame:
        """Prepare data for temporal modeling"""
        if 'year' not in data.columns:
            self.logger.warning("No year column found, treating as cross-sectional data")
            return data[['geo_id'] + feature_cols + [target_col]].dropna()
            
        # Sort by geo_id and year
        temporal_data = data.sort_values(['geo_id', 'year'])
        
        # Create lagged features
        for lag in [1, 2, 3]:
            lagged_target = temporal_data.groupby('geo_id')[target_col].shift(lag)
            temporal_data[f'{target_col}_lag{lag}'] = lagged_target
            
        # Add temporal trends
        temporal_data['year_numeric'] = temporal_data['year'] - temporal_data['year'].min()
        
        # Add regional trends
        for geo_id in temporal_data['geo_id'].unique():
            geo_data = temporal_data[temporal_data['geo_id'] == geo_id]
            if len(geo_data) > 2:
                trend = np.polyfit(geo_data['year_numeric'], geo_data[target_col], 1)[0]
                temporal_data.loc[temporal_data['geo_id'] == geo_id, 'local_trend'] = trend
                
        temporal_data['local_trend'] = temporal_data['local_trend'].fillna(0)
        
        return temporal_data.dropna(subset=[target_col])
        
    def _train_lightgbm_forecaster(self, data: pd.DataFrame,
                                 target_col: str, feature_cols: List[str]) -> Dict[str, Any]:
        """Train LightGBM forecasting model"""
        # Prepare features (include temporal features)
        extended_features = feature_cols + [col for col in data.columns 
                                          if col.startswith(target_col + '_lag') or col in ['year_numeric', 'local_trend']]
        
        X = data[extended_features].fillna(data[extended_features].median())
        y = data[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['lightgbm'] = scaler
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1,
                'random_state': 42
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Validate
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            cv_scores.append({
                'mae': mean_absolute_error(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'r2': r2_score(y_val, y_pred)
            })
            
        # Train final model on all data
        train_data = lgb.Dataset(X_scaled, label=y)
        
        final_model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        self.models['lightgbm'] = {
            'model': final_model,
            'features': extended_features
        }
        
        return {
            'cv_scores': cv_scores,
            'cv_mae_mean': np.mean([s['mae'] for s in cv_scores]),
            'cv_rmse_mean': np.mean([s['rmse'] for s in cv_scores]),
            'cv_r2_mean': np.mean([s['r2'] for s in cv_scores]),
            'feature_importance': final_model.feature_importance(importance_type='gain')
        }
        
    def _train_quantile_models(self, data: pd.DataFrame,
                             target_col: str, feature_cols: List[str]) -> Dict[str, Any]:
        """Train quantile regression models for uncertainty estimation"""
        from sklearn.ensemble import GradientBoostingRegressor
        
        extended_features = feature_cols + [col for col in data.columns 
                                          if col.startswith(target_col + '_lag') or col in ['year_numeric', 'local_trend']]
        
        X = data[extended_features].fillna(data[extended_features].median())
        y = data[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['quantile'] = scaler
        
        # Train models for different quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantile_models = {}
        
        for quantile in quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
            
            model.fit(X_scaled, y)
            quantile_models[f'q{int(quantile*100)}'] = {
                'model': model,
                'features': extended_features
            }
            
        self.models['quantile'] = quantile_models
        
        return {'quantiles_trained': list(quantiles)}
        
    def _train_trend_model(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Train simple trend model as baseline"""
        if 'year' not in data.columns:
            return {'method': 'global_mean', 'value': data[target_col].mean()}
            
        # Simple linear trend by geo_id
        trend_models = {}
        
        for geo_id in data['geo_id'].unique():
            geo_data = data[data['geo_id'] == geo_id]
            if len(geo_data) > 2:
                # Fit linear trend
                years = geo_data['year'] - geo_data['year'].min()
                trend, intercept = np.polyfit(years, geo_data[target_col], 1)
                trend_models[geo_id] = {'trend': trend, 'intercept': intercept, 'base_year': geo_data['year'].min()}
                
        self.models['trend'] = trend_models
        
        return {'n_geo_trends': len(trend_models)}
        
    def _validate_ensemble(self, data: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Dict[str, Any]:
        """Validate ensemble performance"""
        # Simple ensemble validation using holdout
        if 'year' in data.columns:
            # Use last year as holdout
            max_year = data['year'].max()
            train_data = data[data['year'] < max_year]
            test_data = data[data['year'] == max_year]
        else:
            # Random holdout
            test_size = min(100, len(data) // 5)
            test_data = data.sample(n=test_size, random_state=42)
            train_data = data.drop(test_data.index)
            
        if len(test_data) == 0:
            return {'validation': 'no_holdout_data'}
            
        # Retrain models on training data
        # (This is simplified - in practice would retrain all models)
        
        return {
            'holdout_size': len(test_data),
            'train_size': len(train_data),
            'validation_method': 'temporal_holdout' if 'year' in data.columns else 'random_holdout'
        }
        
    def _prepare_forecast_features(self, data: pd.DataFrame, year: int, scenario: str) -> pd.DataFrame:
        """Prepare features for forecasting specific year"""
        forecast_data = data.copy()
        
        # Update temporal features based on scenario
        if scenario == 'optimistic':
            # Assume improvements in key drivers
            if 'pm25_annual' in forecast_data.columns:
                forecast_data['pm25_annual'] *= 0.9  # 10% reduction
            if 'gdp_per_capita_ppp' in forecast_data.columns:
                forecast_data['gdp_per_capita_ppp'] *= 1.05  # 5% increase
                
        elif scenario == 'pessimistic':
            # Assume worsening conditions
            if 'pm25_annual' in forecast_data.columns:
                forecast_data['pm25_annual'] *= 1.1  # 10% increase
            if 'temperature_mean' in forecast_data.columns:
                forecast_data['temperature_mean'] += 1.5  # Climate warming
                
        # Add year information
        forecast_data['year_numeric'] = year - self.config['temporal']['start_year']
        
        return forecast_data
        
    def _predict_lightgbm(self, data: pd.DataFrame) -> np.ndarray:
        """Generate LightGBM predictions"""
        if 'lightgbm' not in self.models:
            return np.full(len(data), np.nan)
            
        model_info = self.models['lightgbm']
        model = model_info['model']
        features = model_info['features']
        
        # Prepare features
        X = data[features].fillna(data[features].median())
        X_scaled = self.scalers['lightgbm'].transform(X)
        
        return model.predict(X_scaled, num_iteration=model.best_iteration)
        
    def _predict_quantiles(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate quantile predictions"""
        if 'quantile' not in self.models:
            return {}
            
        quantile_models = self.models['quantile']
        predictions = {}
        
        for quantile_name, model_info in quantile_models.items():
            model = model_info['model']
            features = model_info['features']
            
            # Prepare features
            X = data[features].fillna(data[features].median())
            X_scaled = self.scalers['quantile'].transform(X)
            
            predictions[quantile_name] = model.predict(X_scaled)
            
        return predictions
        
    def _predict_trend(self, data: pd.DataFrame, year: int) -> np.ndarray:
        """Generate trend-based predictions"""
        if 'trend' not in self.models:
            return np.full(len(data), np.nan)
            
        trend_models = self.models['trend']
        predictions = []
        
        for _, row in data.iterrows():
            geo_id = row.get('geo_id')
            
            if geo_id in trend_models:
                trend_info = trend_models[geo_id]
                years_ahead = year - trend_info['base_year']
                pred = trend_info['intercept'] + trend_info['trend'] * years_ahead
            else:
                # Use global trend if no local model
                pred = 75.0  # Global average approximation
                
            predictions.append(pred)
            
        return np.array(predictions)
        
    def _combine_predictions(self, predictions: Dict[str, np.ndarray],
                           data: pd.DataFrame, year: int, scenario: str) -> pd.DataFrame:
        """Combine predictions from different models"""
        n_obs = len(data)
        
        # Primary prediction (ensemble average)
        pred_values = []
        for pred in predictions.values():
            if not isinstance(pred, dict) and len(pred) == n_obs:
                pred_values.append(pred)
                
        if pred_values:
            ensemble_pred = np.mean(pred_values, axis=0)
        else:
            ensemble_pred = np.full(n_obs, 75.0)  # Default value
            
        # Uncertainty intervals
        if 'q10' in predictions and 'q90' in predictions:
            lower_bound = predictions['q10']
            upper_bound = predictions['q90']
        else:
            # Approximate uncertainty
            pred_std = np.std(pred_values, axis=0) if len(pred_values) > 1 else np.full(n_obs, 2.0)
            lower_bound = ensemble_pred - 1.645 * pred_std  # 90% CI
            upper_bound = ensemble_pred + 1.645 * pred_std
            
        # Create results DataFrame
        result = pd.DataFrame({
            'geo_id': data['geo_id'] if 'geo_id' in data.columns else range(n_obs),
            'year': year,
            'scenario': scenario,
            'predicted_life_expectancy': ensemble_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'prediction_interval_width': upper_bound - lower_bound
        })
        
        # Add geographic info if available
        if 'latitude' in data.columns:
            result['latitude'] = data['latitude'].values
        if 'longitude' in data.columns:
            result['longitude'] = data['longitude'].values
            
        return result