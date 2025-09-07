"""
Blue Zone classification model
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import shap


class BlueZoneClassifier:
    """Machine learning classifier for Blue Zone identification"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.shap_explainer = None
        
    def train_classifier(self, data: pd.DataFrame,
                        treatment_col: str = 'is_blue_zone',
                        feature_cols: List[str] = None) -> Dict[str, Any]:
        """
        Train Blue Zone classifier
        
        Args:
            data: Training dataset
            treatment_col: Blue Zone indicator column
            feature_cols: Features for classification
            
        Returns:
            Training results and evaluation metrics
        """
        if feature_cols is None:
            feature_cols = self._select_classification_features(data)
            
        self.feature_names = feature_cols
        self.logger.info(f"Training classifier with {len(feature_cols)} features")
        
        # Prepare data
        X, y = self._prepare_training_data(data, treatment_col, feature_cols)
        
        if y.sum() == 0:
            self.logger.error("No Blue Zone observations found")
            return {}
            
        # Handle class imbalance
        class_weights = self._calculate_class_weights(y)
        
        # Train model
        model_results = self._train_lightgbm(X, y, class_weights)
        
        # Evaluate model
        evaluation_results = self._evaluate_model(X, y)
        
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance(X, y)
        
        # Create SHAP explainer
        self._create_shap_explainer(X)
        
        results = {
            'model_performance': model_results,
            'evaluation_metrics': evaluation_results,
            'feature_importance': feature_importance,
            'class_weights': class_weights,
            'n_features': len(feature_cols),
            'n_samples': len(X),
            'n_blue_zones': y.sum()
        }
        
        return results
        
    def predict_blue_zone_likelihood(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict Blue Zone likelihood for new data
        
        Args:
            data: Dataset to score
            
        Returns:
            DataFrame with Blue Zone scores
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        # Prepare features
        X = self._prepare_prediction_data(data)
        
        # Generate predictions
        probabilities = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Create results dataframe
        results = pd.DataFrame({
            'geo_id': data['geo_id'] if 'geo_id' in data.columns else range(len(data)),
            'blue_zone_score': probabilities,
            'blue_zone_decile': pd.qcut(probabilities, q=10, labels=range(1, 11))
        })
        
        return results
        
    def explain_predictions(self, data: pd.DataFrame, 
                          n_samples: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions
        
        Args:
            data: Data to explain
            n_samples: Number of samples to explain
            
        Returns:
            SHAP values and explanations
        """
        if self.shap_explainer is None:
            self.logger.warning("SHAP explainer not available")
            return {}
            
        # Sample data for explanation
        if len(data) > n_samples:
            sample_data = data.sample(n=n_samples, random_state=42)
        else:
            sample_data = data
            
        # Prepare features
        X_sample = self._prepare_prediction_data(sample_data)
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        # Create explanations
        explanations = {
            'shap_values': shap_values,
            'feature_names': self.feature_names,
            'base_value': self.shap_explainer.expected_value,
            'sample_data': X_sample,
            'predictions': self.model.predict(X_sample, num_iteration=self.model.best_iteration)
        }
        
        return explanations
        
    def _select_classification_features(self, data: pd.DataFrame) -> List[str]:
        """Select features for classification"""
        # Exclude target and ID variables
        exclude_patterns = [
            'geo_id', 'year', 'location', 'val', 'lower', 'upper',
            'is_blue_zone', 'is_sardinia', 'is_okinawa', 'is_nicoya', 'is_ikaria', 'is_loma_linda'
        ]
        
        candidate_features = []
        for col in data.columns:
            if (data[col].dtype in [np.number] and 
                not any(pattern in col.lower() for pattern in exclude_patterns)):
                candidate_features.append(col)
                
        # Remove features with high missing rate
        valid_features = []
        for col in candidate_features:
            missing_pct = data[col].isnull().sum() / len(data)
            if missing_pct < 0.7:  # Less than 70% missing
                valid_features.append(col)
                
        # Remove highly correlated features
        if len(valid_features) > 10:
            valid_features = self._remove_correlated_features(data[valid_features])
            
        self.logger.info(f"Selected {len(valid_features)} features for classification")
        return valid_features
        
    def _remove_correlated_features(self, data: pd.DataFrame, 
                                  threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features"""
        corr_matrix = data.corr().abs()
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                    
        # Remove one feature from each pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # Keep feature with less missing data
            missing1 = data[feat1].isnull().sum()
            missing2 = data[feat2].isnull().sum()
            
            if missing1 > missing2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
                
        remaining_features = [col for col in data.columns if col not in features_to_remove]
        
        if features_to_remove:
            self.logger.info(f"Removed {len(features_to_remove)} highly correlated features")
            
        return remaining_features
        
    def _prepare_training_data(self, data: pd.DataFrame, treatment_col: str,
                             feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data"""
        # Remove rows with missing target
        clean_data = data.dropna(subset=[treatment_col])
        
        # Prepare features
        X = clean_data[feature_cols].fillna(clean_data[feature_cols].median())
        y = clean_data[treatment_col].astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
        
    def _prepare_prediction_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for prediction"""
        X = data[self.feature_names].fillna(data[self.feature_names].median())
        X_scaled = self.scaler.transform(X)
        return X_scaled
        
    def _calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced data"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
        
    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, 
                       class_weights: Dict[int, float]) -> Dict[str, Any]:
        """Train LightGBM classifier"""
        # Convert class weights to sample weights
        sample_weights = np.array([class_weights[label] for label in y])
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': self.config['models']['classification']['params'].get('max_depth', 6) * 2,
            'learning_rate': self.config['models']['classification']['params']['learning_rate'],
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Create dataset
        train_data = lgb.Dataset(X, label=y, weight=sample_weights)
        
        # Train with cross-validation
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=self.config['models']['classification']['params']['n_estimators'],
            nfold=5,
            stratified=True,
            shuffle=True,
            seed=42,
            return_cvbooster=True,
            eval_train_metric=True
        )
        
        # Train final model
        best_iteration = len(cv_results['valid auc-mean'])
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=best_iteration,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        return {
            'best_iteration': best_iteration,
            'cv_auc_mean': cv_results['valid auc-mean'][-1],
            'cv_auc_std': cv_results['valid auc-stdv'][-1]
        }
        
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, cv=StratifiedKFold(n_splits=5), scoring='roc_auc'
        )
        
        # Predictions
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        auc_score = roc_auc_score(y, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        
        return {
            'cv_auc_scores': cv_scores,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'auc_score': auc_score,
            'precision_recall_auc': np.trapz(precision, recall),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
    def _analyze_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze feature importance"""
        # LightGBM feature importance
        importance_gain = self.model.feature_importance(importance_type='gain')
        importance_split = self.model.feature_importance(importance_type='split')
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_gain': importance_gain,
            'importance_split': importance_split
        }).sort_values('importance_gain', ascending=False)
        
        return {
            'feature_importance': feature_importance_df,
            'top_10_features': feature_importance_df.head(10)['feature'].tolist()
        }
        
    def _create_shap_explainer(self, X: np.ndarray) -> None:
        """Create SHAP explainer"""
        try:
            # Sample data for SHAP background
            if len(X) > 100:
                background_data = X[np.random.choice(len(X), 100, replace=False)]
            else:
                background_data = X
                
            self.shap_explainer = shap.TreeExplainer(self.model, background_data)
            
        except Exception as e:
            self.logger.warning(f"Could not create SHAP explainer: {e}")
            self.shap_explainer = None