#!/usr/bin/env python3
"""
Test suite for the advanced ML data science pipeline
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_pipeline_datascience import (
    MLPipelineDataScience,
    AdvancedFeatureEngineer,
    FeatureConfig,
    ModelConfig,
    ModelType
)

def create_sample_data() -> pd.DataFrame:
    """Create sample training data for testing"""
    sample_data = [
        # High-end GPU, demanding game
        {
            'gpu_name': 'RTX 4090',
            'game_name': 'Cyberpunk 2077',
            'resolution': '1080p',
            'fps': 98.5,
            'passmark_score': 38192,
            'memory_gb': 24,
            'source': 'test'
        },
        {
            'gpu_name': 'RTX 4090',
            'game_name': 'Cyberpunk 2077',
            'resolution': '4k',
            'fps': 42.3,
            'passmark_score': 38192,
            'memory_gb': 24,
            'source': 'test'
        },
        # High-end GPU, optimized game
        {
            'gpu_name': 'RTX 4090',
            'game_name': 'Counter-Strike 2',
            'resolution': '1080p',
            'fps': 487.2,
            'passmark_score': 38192,
            'memory_gb': 24,
            'source': 'test'
        },
        {
            'gpu_name': 'RTX 4090',
            'game_name': 'Valorant',
            'resolution': '1080p',
            'fps': 512.8,
            'passmark_score': 38192,
            'memory_gb': 24,
            'source': 'test'
        },
        # Mid-range GPU, various games
        {
            'gpu_name': 'RTX 4070',
            'game_name': 'Fortnite',
            'resolution': '1080p',
            'fps': 156.7,
            'passmark_score': 26925,
            'memory_gb': 12,
            'source': 'test'
        },
        {
            'gpu_name': 'RTX 4070',
            'game_name': 'Apex Legends',
            'resolution': '1080p',
            'fps': 142.3,
            'passmark_score': 26925,
            'memory_gb': 12,
            'source': 'test'
        },
        {
            'gpu_name': 'RTX 4070',
            'game_name': 'Cyberpunk 2077',
            'resolution': '1080p',
            'fps': 78.9,
            'passmark_score': 26925,
            'memory_gb': 12,
            'source': 'test'
        },
        # Lower-end GPU
        {
            'gpu_name': 'RTX 4060',
            'game_name': 'Counter-Strike 2',
            'resolution': '1080p',
            'fps': 298.4,
            'passmark_score': 19542,
            'memory_gb': 8,
            'source': 'test'
        },
        {
            'gpu_name': 'RTX 4060',
            'game_name': 'Cyberpunk 2077',
            'resolution': '1080p',
            'fps': 45.6,
            'passmark_score': 19542,
            'memory_gb': 8,
            'source': 'test'
        },
        # Different resolutions
        {
            'gpu_name': 'RTX 4080',
            'game_name': 'Fortnite',
            'resolution': '1440p',
            'fps': 198.7,
            'passmark_score': 34453,
            'memory_gb': 16,
            'source': 'test'
        }
    ]
    
    return pd.DataFrame(sample_data)

def test_feature_engineering():
    """Test the advanced feature engineering"""
    print("=== Testing Feature Engineering ===")
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize feature engineer
    config = FeatureConfig(
        use_gpu_features=True,
        use_game_features=True,
        use_resolution_features=True,
        use_vram_features=True,
        use_passmark_features=True,
        use_interaction_features=True,
        use_derived_features=True
    )
    
    engineer = AdvancedFeatureEngineer(config)
    
    # Extract features
    features_df = engineer.extract_features(data)
    
    # Check that features were added
    original_columns = set(data.columns)
    new_columns = set(features_df.columns) - original_columns
    
    print(f"Original columns: {len(original_columns)}")
    print(f"New feature columns: {len(new_columns)}")
    print(f"Total columns: {len(features_df.columns)}")
    
    # Verify key features exist
    expected_features = [
        'gpu_generation', 'gpu_tier', 'game_complexity', 'pixel_count',
        'vram_per_pixel', 'passmark_per_pixel', 'gpu_game_balance'
    ]
    
    missing_features = [f for f in expected_features if f not in features_df.columns]
    if missing_features:
        print(f"[WARNING] Missing expected features: {missing_features}")
    else:
        print("[SUCCESS] All expected features present")
    
    # Check for reasonable values
    print(f"\nSample feature values:")
    print(f"GPU generations: {sorted(features_df['gpu_generation'].unique())}")
    print(f"GPU tiers: {sorted(features_df['gpu_tier'].unique())}")
    print(f"Game complexity range: {features_df['game_complexity'].min():.2f} - {features_df['game_complexity'].max():.2f}")
    print(f"Pixel count range: {features_df['pixel_count'].min():,} - {features_df['pixel_count'].max():,}")
    
    return features_df

def test_pipeline_training():
    """Test the complete ML pipeline training"""
    print("\n=== Testing ML Pipeline Training ===")
    
    # Create sample data
    data = create_sample_data()
    
    # Configure pipeline
    feature_config = FeatureConfig(
        use_gpu_features=True,
        use_game_features=True,
        use_resolution_features=True,
        use_vram_features=True,
        use_passmark_features=True,
        use_interaction_features=True,
        use_derived_features=True
    )
    
    model_config = ModelConfig(
        model_type=ModelType.RANDOM_FOREST,
        test_size=0.3,  # Larger test size for small dataset
        cv_folds=3,     # Smaller CV for small dataset
        use_grid_search=False  # Skip grid search for speed
    )
    
    # Initialize pipeline
    pipeline = MLPipelineDataScience(feature_config, model_config)
    
    try:
        # Train pipeline
        result = pipeline.train_pipeline(data)
        
        print(f"[SUCCESS] Pipeline training completed")
        print(f"Model type: {result.model_config.model_type.value}")
        print(f"Features used: {len(result.features)}")
        
        # Check metrics
        if 'mae' in result.metrics:
            print(f"MAE: {result.metrics['mae']:.2f} FPS")
        if 'r2' in result.metrics:
            print(f"R²: {result.metrics['r2']:.3f}")
        if 'mape' in result.metrics:
            print(f"MAPE: {result.metrics['mape']:.1f}%")
        
        # Check feature importance
        if result.feature_importance:
            print(f"\nTop 5 most important features:")
            sorted_importance = sorted(result.feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_importance[:5]):
                print(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Check predictions vs actuals
        if len(result.predictions) > 0 and len(result.actuals) > 0:
            print(f"\nPrediction examples:")
            for i in range(min(3, len(result.predictions))):
                actual = result.actuals[i]
                pred = result.predictions[i]
                error = abs(actual - pred)
                print(f"  Actual: {actual:.1f} FPS, Predicted: {pred:.1f} FPS, Error: {error:.1f} FPS")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Pipeline training failed: {e}")
        return None

def test_different_models():
    """Test different ML model types"""
    print("\n=== Testing Different Model Types ===")
    
    data = create_sample_data()
    feature_config = FeatureConfig()
    
    model_types = [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING, ModelType.RIDGE]
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type.value}...")
        
        model_config = ModelConfig(
            model_type=model_type,
            test_size=0.3,
            cv_folds=3,
            use_grid_search=False
        )
        
        pipeline = MLPipelineDataScience(feature_config, model_config)
        
        try:
            result = pipeline.train_pipeline(data)
            results[model_type.value] = result.metrics.get('mae', float('inf'))
            print(f"  MAE: {result.metrics.get('mae', 'N/A')}")
            print(f"  R²: {result.metrics.get('r2', 'N/A')}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[model_type.value] = float('inf')
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k])
    print(f"\n[RESULT] Best model: {best_model} (MAE: {results[best_model]:.2f})")
    
    return results

def test_feature_importance_analysis():
    """Test feature importance analysis"""
    print("\n=== Testing Feature Importance Analysis ===")
    
    data = create_sample_data()
    
    # Train with all features
    pipeline = MLPipelineDataScience()
    result = pipeline.train_pipeline(data)
    
    if not result.feature_importance:
        print("[WARNING] No feature importance available")
        return
    
    # Analyze feature importance
    sorted_features = sorted(result.feature_importance.items(), 
                           key=lambda x: x[1], reverse=True)
    
    print(f"Feature importance analysis ({len(sorted_features)} features):")
    
    # Group features by category
    gpu_features = [f for f, _ in sorted_features if any(term in f.lower() for term in ['gpu', 'rtx', 'passmark'])]
    game_features = [f for f, _ in sorted_features if any(term in f.lower() for term in ['game', 'complexity'])]
    resolution_features = [f for f, _ in sorted_features if any(term in f.lower() for term in ['resolution', 'pixel'])]
    vram_features = [f for f, _ in sorted_features if any(term in f.lower() for term in ['vram', 'memory'])]
    
    print(f"  GPU-related features: {len(gpu_features)}")
    print(f"  Game-related features: {len(game_features)}")
    print(f"  Resolution-related features: {len(resolution_features)}")
    print(f"  VRAM-related features: {len(vram_features)}")
    
    # Show top features overall
    print(f"\nTop 10 most important features:")
    for i, (feature, importance) in enumerate(sorted_features[:10]):
        print(f"  {i+1:2d}. {feature:25s}: {importance:.4f}")

def test_cross_validation():
    """Test cross-validation functionality"""
    print("\n=== Testing Cross-Validation ===")
    
    data = create_sample_data()
    
    model_config = ModelConfig(
        model_type=ModelType.RANDOM_FOREST,
        cv_folds=3,
        use_grid_search=False
    )
    
    pipeline = MLPipelineDataScience(model_config=model_config)
    result = pipeline.train_pipeline(data)
    
    if 'cv_mae' in result.metrics:
        cv_mae = result.metrics['cv_mae']
        cv_std = result.metrics.get('cv_mae_std', 0)
        print(f"Cross-validation MAE: {cv_mae:.2f} ± {cv_std:.2f} FPS")
        
        # Compare with test set MAE
        test_mae = result.metrics.get('mae', 0)
        print(f"Test set MAE: {test_mae:.2f} FPS")
        
        if abs(cv_mae - test_mae) < 20:  # Within 20 FPS
            print("[SUCCESS] CV and test MAE are consistent")
        else:
            print("[WARNING] Large difference between CV and test MAE")
    else:
        print("[WARNING] Cross-validation metrics not available")

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("Starting comprehensive ML pipeline test...\n")
    
    try:
        # Test 1: Feature Engineering
        features_df = test_feature_engineering()
        
        # Test 2: Basic Pipeline Training
        result = test_pipeline_training()
        
        if result is None:
            print("\n[CRITICAL] Basic pipeline training failed. Stopping tests.")
            return False
        
        # Test 3: Different Models
        model_results = test_different_models()
        
        # Test 4: Feature Importance
        test_feature_importance_analysis()
        
        # Test 5: Cross-Validation
        test_cross_validation()
        
        print("\n=== Summary ===")
        print("[SUCCESS] All ML pipeline tests completed")
        print(f"Features engineered: {len(features_df.columns)}")
        print(f"Models tested: {len(model_results)}")
        print(f"Best model performance: {min(model_results.values()):.2f} MAE")
        
        return True
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n[PASS] ML pipeline test suite completed successfully")
    else:
        print("\n[FAIL] ML pipeline test suite encountered errors")