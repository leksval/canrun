#!/usr/bin/env python3
"""
Comprehensive ML Data Science Pipeline for FPS Prediction
Professional-grade feature engineering, model training, and validation
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re
from datetime import datetime
from enum import Enum

# Optional advanced ML libraries (graceful fallback if not available)
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelType(Enum):
    """Available ML model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    use_gpu_features: bool = True
    use_game_features: bool = True
    use_resolution_features: bool = True
    use_vram_features: bool = True
    use_passmark_features: bool = True
    use_interaction_features: bool = True
    use_derived_features: bool = True


@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_type: ModelType = ModelType.RANDOM_FOREST
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    use_grid_search: bool = True
    n_jobs: int = -1


@dataclass
class PipelineResult:
    """Results from ML pipeline execution"""
    model: Any
    features: List[str]
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    model_config: ModelConfig
    feature_config: FeatureConfig


class AdvancedFeatureEngineer:
    """Advanced feature engineering for FPS prediction"""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.logger = logging.getLogger(__name__)
        self.gpu_hierarchy = self._load_gpu_hierarchy()
        self.game_complexity_scores = self._load_game_complexity()
        
    def _load_gpu_hierarchy(self) -> Dict:
        """Load GPU hierarchy for feature engineering"""
        try:
            hierarchy_path = Path(__file__).parent.parent / "data" / "gpu_hierarchy.json"
            if hierarchy_path.exists():
                with open(hierarchy_path, 'r') as f:
                    data = json.load(f)
                    return data.get('nvidia', {})
            return {}
        except Exception as e:
            self.logger.warning(f"Could not load GPU hierarchy: {e}")
            return {}
    
    def _load_game_complexity(self) -> Dict[str, float]:
        """Define game complexity scores for feature engineering"""
        return {
            'cyberpunk_2077': 0.95,      # Very demanding
            'red_dead_redemption_2': 0.92,
            'microsoft_flight_simulator': 0.90,
            'metro_exodus': 0.85,
            'call_of_duty_warzone': 0.75,
            'battlefield_2042': 0.73,
            'apex_legends': 0.70,
            'fortnite': 0.65,
            'overwatch_2': 0.60,
            'league_of_legends': 0.45,
            'counter_strike_2': 0.40,
            'valorant': 0.35,            # Highly optimized
            'rocket_league': 0.30
        }
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive features from raw data"""
        features_df = data.copy()
        
        if self.config.use_gpu_features:
            features_df = self._add_gpu_features(features_df)
            
        if self.config.use_game_features:
            features_df = self._add_game_features(features_df)
            
        if self.config.use_resolution_features:
            features_df = self._add_resolution_features(features_df)
            
        if self.config.use_vram_features:
            features_df = self._add_vram_features(features_df)
            
        if self.config.use_passmark_features:
            features_df = self._add_passmark_features(features_df)
            
        if self.config.use_interaction_features:
            features_df = self._add_interaction_features(features_df)
            
        if self.config.use_derived_features:
            features_df = self._add_derived_features(features_df)
        
        return features_df
    
    def _add_gpu_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add GPU-related features"""
        # GPU generation
        df['gpu_generation'] = df['gpu_name'].apply(self._extract_gpu_generation)
        
        # GPU tier (entry, mid, high, flagship)
        df['gpu_tier'] = df['passmark_score'].apply(self._categorize_gpu_tier)
        
        # GPU brand
        df['gpu_brand'] = df['gpu_name'].apply(lambda x: 'nvidia' if 'rtx' in x.lower() or 'gtx' in x.lower() else 'amd')
        
        # Architecture features
        df['is_rtx'] = df['gpu_name'].str.lower().str.contains('rtx').astype(int)
        df['is_super'] = df['gpu_name'].str.lower().str.contains('super').astype(int)
        df['is_ti'] = df['gpu_name'].str.lower().str.contains('ti').astype(int)
        
        return df
    
    def _add_game_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add game-related features"""
        # Game complexity score
        df['game_complexity'] = df['game_name'].apply(self._get_game_complexity)
        
        # Game category
        df['game_category'] = df['game_name'].apply(self._categorize_game)
        
        # Engine type (estimated)
        df['engine_type'] = df['game_name'].apply(self._estimate_engine_type)
        
        return df
    
    def _add_resolution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add resolution-related features"""
        # Pixel count
        resolution_pixels = {
            '720p': 1280 * 720,
            '1080p': 1920 * 1080,
            '1440p': 2560 * 1440,
            '4k': 3840 * 2160
        }
        df['pixel_count'] = df['resolution'].map(resolution_pixels)
        
        # Resolution tier
        df['resolution_tier'] = df['resolution'].map({
            '720p': 1, '1080p': 2, '1440p': 3, '4k': 4
        })
        
        # Aspect ratio (assuming 16:9 for all)
        df['aspect_ratio'] = 16/9
        
        return df
    
    def _add_vram_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VRAM-related features"""
        # VRAM per pixel ratio
        df['vram_per_pixel'] = (df['memory_gb'] * 1024**3) / df['pixel_count']
        
        # VRAM tier
        df['vram_tier'] = df['memory_gb'].apply(lambda x: 
            1 if x <= 4 else 2 if x <= 8 else 3 if x <= 12 else 4)
        
        # VRAM adequacy for resolution
        vram_requirements = {'720p': 2, '1080p': 4, '1440p': 8, '4k': 12}
        df['vram_adequacy'] = df.apply(lambda row: 
            row['memory_gb'] / vram_requirements.get(row['resolution'], 4), axis=1)
        
        return df
    
    def _add_passmark_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add PassMark-related features"""
        # PassMark per pixel
        df['passmark_per_pixel'] = df['passmark_score'] / df['pixel_count']
        
        # PassMark tier
        df['passmark_tier'] = df['passmark_score'].apply(lambda x:
            1 if x < 10000 else 2 if x < 20000 else 3 if x < 30000 else 4)
        
        # Performance efficiency (PassMark / TDP estimation)
        df['performance_efficiency'] = df['passmark_score'] / self._estimate_tdp(df['gpu_name'])
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different components"""
        # GPU-Game complexity interaction
        df['gpu_game_balance'] = df['passmark_score'] * (1 - df['game_complexity'])
        
        # VRAM-Resolution balance
        df['vram_resolution_balance'] = df['vram_adequacy'] * df['resolution_tier']
        
        # Performance per pixel per complexity
        df['performance_density'] = (df['passmark_per_pixel'] * 
                                    (1 + df['game_complexity']))
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from domain knowledge"""
        # Theoretical max FPS (simplified)
        df['theoretical_max_fps'] = df['passmark_score'] / 100
        
        # Bottleneck indicators
        df['likely_gpu_bottleneck'] = (df['game_complexity'] > 0.7).astype(int)
        df['likely_vram_bottleneck'] = (df['vram_adequacy'] < 1.0).astype(int)
        
        # Performance headroom
        df['performance_headroom'] = np.log1p(df['passmark_score'] / 
                                            (df['game_complexity'] * 20000))
        
        return df
    
    def _extract_gpu_generation(self, gpu_name: str) -> int:
        """Extract GPU generation number"""
        gpu_lower = gpu_name.lower()
        if 'rtx 50' in gpu_lower:
            return 50
        elif 'rtx 40' in gpu_lower:
            return 40
        elif 'rtx 30' in gpu_lower:
            return 30
        elif 'rtx 20' in gpu_lower:
            return 20
        elif 'gtx 16' in gpu_lower:
            return 16
        elif 'gtx 10' in gpu_lower:
            return 10
        else:
            # Extract numbers
            numbers = re.findall(r'\d+', gpu_name)
            if numbers:
                first_num = int(numbers[0])
                if first_num > 500:  # Like 1080, 2080
                    return first_num // 10
            return 0
    
    def _categorize_gpu_tier(self, passmark: float) -> str:
        """Categorize GPU tier based on PassMark score"""
        if passmark >= 30000:
            return 'flagship'
        elif passmark >= 20000:
            return 'high_end'
        elif passmark >= 12000:
            return 'mid_range'
        elif passmark >= 6000:
            return 'entry_level'
        else:
            return 'budget'
    
    def _get_game_complexity(self, game_name: str) -> float:
        """Get game complexity score"""
        game_key = game_name.lower().replace(' ', '_').replace(':', '').replace("'", "")
        return self.game_complexity_scores.get(game_key, 0.6)  # Default moderate complexity
    
    def _categorize_game(self, game_name: str) -> str:
        """Categorize game by type"""
        name_lower = game_name.lower()
        if any(word in name_lower for word in ['counter-strike', 'valorant', 'overwatch']):
            return 'esports'
        elif any(word in name_lower for word in ['fortnite', 'apex', 'warzone', 'pubg']):
            return 'battle_royale'
        elif any(word in name_lower for word in ['cyberpunk', 'witcher', 'metro', 'red_dead']):
            return 'aaa_single_player'
        elif any(word in name_lower for word in ['battlefield', 'call_of_duty']):
            return 'aaa_multiplayer'
        else:
            return 'general'
    
    def _estimate_engine_type(self, game_name: str) -> str:
        """Estimate game engine type"""
        name_lower = game_name.lower()
        if 'unreal' in name_lower or any(word in name_lower for word in ['fortnite', 'rocket_league']):
            return 'unreal'
        elif any(word in name_lower for word in ['battlefield', 'apex']):
            return 'source'
        elif 'cyberpunk' in name_lower:
            return 'red_engine'
        else:
            return 'unknown'
    
    def _estimate_tdp(self, gpu_name: str) -> float:
        """Estimate GPU TDP for efficiency calculation"""
        gpu_lower = gpu_name.lower()
        if '4090' in gpu_lower:
            return 450
        elif '4080' in gpu_lower:
            return 320
        elif '4070' in gpu_lower:
            return 200
        elif '4060' in gpu_lower:
            return 115
        elif '3080' in gpu_lower:
            return 320
        elif '3070' in gpu_lower:
            return 220
        else:
            return 200  # Default


class MLPipelineDataScience:
    """Comprehensive ML pipeline for FPS prediction"""
    
    def __init__(self, feature_config: FeatureConfig = None, model_config: ModelConfig = None):
        self.feature_config = feature_config or FeatureConfig()
        self.model_config = model_config or ModelConfig()
        self.logger = logging.getLogger(__name__)
        self.feature_engineer = AdvancedFeatureEngineer(self.feature_config)
        self.scaler = None
        self.encoder = None
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available. Limited functionality.")
    
    def load_training_data(self) -> pd.DataFrame:
        """Load and prepare training data from multiple sources"""
        # Load ML model training data
        ml_data = self._load_ml_training_data()
        
        # Load benchmark data
        benchmark_data = self._load_benchmark_data()
        
        # Combine and clean data
        combined_data = pd.concat([ml_data, benchmark_data], ignore_index=True)
        cleaned_data = self._clean_data(combined_data)
        
        return cleaned_data
    
    def _load_ml_training_data(self) -> pd.DataFrame:
        """Load existing ML model data"""
        try:
            model_path = Path(__file__).parent.parent / "data" / "ml_fps_model.json"
            if model_path.exists():
                with open(model_path, 'r') as f:
                    data = json.load(f)
                    
                lookup_table = data.get('lookup_table', {})
                rows = []
                
                for key, fps in lookup_table.items():
                    parts = key.split('|')
                    if len(parts) == 3:
                        gpu_name, game_name, resolution = parts
                        rows.append({
                            'gpu_name': gpu_name,
                            'game_name': game_name,
                            'resolution': resolution,
                            'fps': fps,
                            'passmark_score': self._get_gpu_passmark(gpu_name),
                            'memory_gb': self._get_gpu_memory(gpu_name),
                            'source': 'ml_model'
                        })
                
                return pd.DataFrame(rows)
        except Exception as e:
            self.logger.warning(f"Could not load ML training data: {e}")
            
        return pd.DataFrame()
    
    def _load_benchmark_data(self) -> pd.DataFrame:
        """Load additional benchmark data"""
        try:
            benchmark_path = Path(__file__).parent.parent.parent / "data" / "training_benchmarks.json"
            if benchmark_path.exists():
                with open(benchmark_path, 'r') as f:
                    data = json.load(f)
                    
                rows = []
                for benchmark in data.get('benchmarks', []):
                    rows.append({
                        'gpu_name': benchmark.get('gpu'),
                        'game_name': benchmark.get('game'),
                        'resolution': benchmark.get('resolution'),
                        'fps': benchmark.get('fps'),
                        'passmark_score': benchmark.get('passmark_score'),
                        'memory_gb': benchmark.get('memory_gb'),
                        'source': 'benchmark'
                    })
                
                return pd.DataFrame(rows)
        except Exception as e:
            self.logger.warning(f"Could not load benchmark data: {e}")
            
        return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate training data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['gpu_name', 'game_name', 'resolution'])
        
        # Remove invalid FPS values
        df = df[(df['fps'] > 0) & (df['fps'] < 1000)]
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['gpu_name', 'game_name', 'resolution', 'fps'])
        
        # Fill missing PassMark scores
        df['passmark_score'] = df.apply(lambda row: 
            self._get_gpu_passmark(row['gpu_name']) if pd.isna(row['passmark_score']) 
            else row['passmark_score'], axis=1)
        
        # Fill missing memory
        df['memory_gb'] = df.apply(lambda row:
            self._get_gpu_memory(row['gpu_name']) if pd.isna(row['memory_gb'])
            else row['memory_gb'], axis=1)
        
        return df.reset_index(drop=True)
    
    def train_pipeline(self, data: pd.DataFrame = None) -> PipelineResult:
        """Train complete ML pipeline"""
        if data is None:
            data = self.load_training_data()
        
        if data.empty:
            raise ValueError("No training data available")
        
        # Feature engineering
        self.logger.info("Starting feature engineering...")
        features_df = self.feature_engineer.extract_features(data)
        
        # Prepare features and target
        feature_columns = [col for col in features_df.columns 
                          if col not in ['fps', 'gpu_name', 'game_name', 'source']]
        
        X = features_df[feature_columns]
        y = features_df['fps']
        
        # Handle categorical variables
        X = self._prepare_features(X)
        
        if not SKLEARN_AVAILABLE:
            return self._create_fallback_result(X, y, feature_columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.model_config.test_size, 
            random_state=self.model_config.random_state
        )
        
        # Train model
        model = self._train_model(X_train, y_train)
        
        # Evaluate model
        predictions = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, predictions)
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, feature_columns)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=self.model_config.cv_folds, 
                                   scoring='neg_mean_absolute_error')
        metrics['cv_mae'] = -cv_scores.mean()
        metrics['cv_mae_std'] = cv_scores.std()
        
        self.logger.info(f"Pipeline training completed. MAE: {metrics['mae']:.2f}")
        
        return PipelineResult(
            model=model,
            features=feature_columns,
            metrics=metrics,
            feature_importance=feature_importance,
            predictions=predictions,
            actuals=y_test.values,
            model_config=self.model_config,
            feature_config=self.feature_config
        )
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training"""
        # Handle categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        if len(categorical_columns) > 0:
            # Simple label encoding for categorical variables
            for col in categorical_columns:
                X[col] = pd.Categorical(X[col]).codes
        
        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        if SKLEARN_AVAILABLE and len(numerical_columns) > 0:
            if self.scaler is None:
                self.scaler = RobustScaler()
                X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
            else:
                X[numerical_columns] = self.scaler.transform(X[numerical_columns])
        
        return X
    
    def _train_model(self, X_train, y_train):
        """Train the ML model"""
        model_params = self._get_model_params()
        
        if self.model_config.model_type == ModelType.RANDOM_FOREST:
            model = RandomForestRegressor(**model_params)
        elif self.model_config.model_type == ModelType.GRADIENT_BOOSTING:
            model = GradientBoostingRegressor(**model_params)
        elif self.model_config.model_type == ModelType.RIDGE:
            model = Ridge(**model_params)
        elif self.model_config.model_type == ModelType.LASSO:
            model = Lasso(**model_params)
        elif self.model_config.model_type == ModelType.ELASTIC_NET:
            model = ElasticNet(**model_params)
        else:
            model = RandomForestRegressor(**model_params)
        
        # Grid search if enabled
        if self.model_config.use_grid_search:
            param_grid = self._get_param_grid()
            if param_grid:
                grid_search = GridSearchCV(
                    model, param_grid, cv=3, scoring='neg_mean_absolute_error',
                    n_jobs=self.model_config.n_jobs
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                self.logger.info(f"Best parameters: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def _get_model_params(self) -> Dict:
        """Get default model parameters"""
        if self.model_config.model_type == ModelType.RANDOM_FOREST:
            return {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.model_config.random_state,
                'n_jobs': self.model_config.n_jobs
            }
        elif self.model_config.model_type == ModelType.GRADIENT_BOOSTING:
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 8,
                'random_state': self.model_config.random_state
            }
        else:
            return {'random_state': self.model_config.random_state}
    
    def _get_param_grid(self) -> Dict:
        """Get parameter grid for grid search"""
        if self.model_config.model_type == ModelType.RANDOM_FOREST:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_config.model_type == ModelType.GRADIENT_BOOSTING:
            return {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10]
            }
        return {}
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate comprehensive model metrics"""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Add percentile-based metrics
        errors = np.abs(y_true - y_pred)
        metrics['p50_error'] = np.percentile(errors, 50)
        metrics['p90_error'] = np.percentile(errors, 90)
        metrics['p95_error'] = np.percentile(errors, 95)
        
        return metrics
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance_values = model.feature_importances_
            return dict(zip(feature_names, importance_values))
        elif hasattr(model, 'coef_'):
            importance_values = np.abs(model.coef_)
            return dict(zip(feature_names, importance_values))
        else:
            return {}
    
    def _create_fallback_result(self, X, y, feature_columns) -> PipelineResult:
        """Create fallback result when sklearn is not available"""
        # Simple mean prediction
        mean_fps = y.mean()
        predictions = np.full(len(y), mean_fps)
        
        metrics = {
            'mae': np.mean(np.abs(y - predictions)),
            'mse': np.mean((y - predictions) ** 2),
            'r2': 0.0,
            'fallback_mode': True
        }
        
        return PipelineResult(
            model=None,
            features=feature_columns,
            metrics=metrics,
            feature_importance={},
            predictions=predictions,
            actuals=y.values,
            model_config=self.model_config,
            feature_config=self.feature_config
        )
    
    def _get_gpu_passmark(self, gpu_name: str) -> float:
        """Get GPU PassMark score"""
        # This would integrate with the existing GPU hierarchy
        # For now, return a default value
        return 20000.0
    
    def _get_gpu_memory(self, gpu_name: str) -> float:
        """Get GPU memory"""
        # This would integrate with the existing GPU hierarchy
        # For now, return a default value
        return 8.0
    
    def save_model(self, result: PipelineResult, path: str):
        """Save trained model and metadata"""
        if SKLEARN_AVAILABLE and result.model:
            import joblib
            joblib.dump(result.model, f"{path}_model.pkl")
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'features': result.features,
            'metrics': result.metrics,
            'feature_importance': result.feature_importance,
            'model_config': {
                'model_type': result.model_config.model_type.value,
                'test_size': result.model_config.test_size,
                'cv_folds': result.model_config.cv_folds
            },
            'feature_config': {
                'use_gpu_features': result.feature_config.use_gpu_features,
                'use_game_features': result.feature_config.use_game_features,
                'use_resolution_features': result.feature_config.use_resolution_features,
                'use_vram_features': result.feature_config.use_vram_features,
                'use_interaction_features': result.feature_config.use_interaction_features
            }
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {path}")


def main():
    """Example usage of the ML pipeline"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure feature engineering
    feature_config = FeatureConfig(
        use_gpu_features=True,
        use_game_features=True,
        use_resolution_features=True,
        use_vram_features=True,
        use_passmark_features=True,
        use_interaction_features=True,
        use_derived_features=True
    )
    
    # Configure model training
    model_config = ModelConfig(
        model_type=ModelType.RANDOM_FOREST,
        test_size=0.2,
        cv_folds=5,
        use_grid_search=True
    )
    
    # Create and run pipeline
    pipeline = MLPipelineDataScience(feature_config, model_config)
    
    try:
        result = pipeline.train_pipeline()
        
        print("\n=== ML Pipeline Results ===")
        print(f"Model Type: {result.model_config.model_type.value}")
        print(f"Features Used: {len(result.features)}")
        print(f"MAE: {result.metrics['mae']:.2f} FPS")
        print(f"R²: {result.metrics['r2']:.3f}")
        
        if result.feature_importance:
            print("\nTop 10 Most Important Features:")
            sorted_importance = sorted(result.feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_importance[:10]:
                print(f"  {feature}: {importance:.4f}")
        
        # Save model
        pipeline.save_model(result, "canrun/models/advanced_fps_model")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()