"""
Utility Functions
=================
Helper functions for model creation, training, evaluation, and output formatting.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from .config import RANDOM_STATE, OVERFIT_THRESHOLD


def print_section(title):
    """Print a formatted section header for console output.
    
    Args:
        title: Section title to display.
    """
    print("\n" + "="*70)
    print(title)
    print("="*70)


def create_model(model_type, params=None):
    """Factory function to create model instances with default or custom parameters.
    
    Supports CatBoost, XGBoost, HistGradientBoosting, and ElasticNet.
    Custom params override defaults but preserve required settings (random_state, verbose).
    
    Args:
        model_type: One of 'CatBoost', 'XGBoost', 'HistGB', 'ElasticNet'.
        params: Optional dict of hyperparameters to override defaults.
    
    Returns:
        Configured sklearn-compatible regressor instance.
    
    Raises:
        ValueError: If model_type is not recognized.
    """
    if params is None:
        params = {}
    
    if model_type == 'CatBoost':
        defaults = {'iterations': 500, 'learning_rate': 0.05, 'depth': 6, 
                    'random_seed': RANDOM_STATE, 'verbose': 0}
        defaults.update(params)
        return CatBoostRegressor(**defaults)
    
    elif model_type == 'XGBoost':
        defaults = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6, 
                    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0}
        defaults.update(params)
        return XGBRegressor(**defaults)
    
    elif model_type == 'HistGB':
        defaults = {'max_iter': 500, 'learning_rate': 0.05, 'max_depth': 8, 
                    'random_state': RANDOM_STATE, 'verbose': 0}
        defaults.update(params)
        return HistGradientBoostingRegressor(**defaults)
    
    elif model_type == 'ElasticNet':
        defaults = {'alpha': 0.01, 'l1_ratio': 0.5, 'random_state': RANDOM_STATE, 
                    'max_iter': 10000}
        defaults.update(params)
        return ElasticNet(**defaults)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(model, name, X_train, X_val, y_train, y_val):
    """Train a model and compute evaluation metrics with overfitting detection.
    
    Fits the model on training data, predicts on both train and validation sets,
    computes R2 and RMSE metrics, and flags potential overfitting based on the 
    train-validation gap.
    
    Args:
        model: Sklearn-compatible regressor (unfitted).
        name: Display name for logging.
        X_train: Training features (numpy array or DataFrame).
        X_val: Validation features.
        y_train: Training target.
        y_val: Validation target.
    
    Returns:
        Dict containing: model name, train_r2, val_r2, train_rmse, val_rmse, gap.
    """
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    gap = train_r2 - val_r2
    
    print(f"\n{name}:")
    print(f"  Train R2: {train_r2:.4f}  |  Train RMSE: {train_rmse:.4f}")
    print(f"  Val R2:   {val_r2:.4f}  |  Val RMSE:   {val_rmse:.4f}")
    print(f"  Gap:      {gap:.4f}")
    if gap > OVERFIT_THRESHOLD:
        print(f"  Potential overfitting detected!")
    
    return {
        'model': name, 
        'train_r2': train_r2, 
        'val_r2': val_r2, 
        'train_rmse': train_rmse, 
        'val_rmse': val_rmse, 
        'gap': gap
    }
