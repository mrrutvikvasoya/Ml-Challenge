"""
Preprocessing Functions
=======================
Data loading, splitting, scaling, feature selection, and polynomial feature generation.
All functions ensure proper train/val/test separation to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from .config import RANDOM_STATE, DATA_PATH, TARGET_PATH, TARGET_COLUMN


def load_data():
    """Load features and target data from CSV files.
    
    Returns:
        Tuple of (X, y, feature_names) where X is features DataFrame,
        y is target Series, and feature_names is list of column names.
    """
    X = pd.read_csv(DATA_PATH)
    y = pd.read_csv(TARGET_PATH)[TARGET_COLUMN]
    feature_names = X.columns.tolist()
    return X, y, feature_names


def split_data(X, y):
    """Split data into train/validation/test sets with 70/15/15 ratio.
    
    Uses stratified random sampling and verifies no index overlap between splits.
    
    Args:
        X: Features DataFrame.
        y: Target Series.
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    
    Raises:
        AssertionError: If any index overlap is detected between splits.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=RANDOM_STATE, shuffle=True
    )
    
    # Verify no data leakage through index overlap
    assert len(set(X_train.index) & set(X_val.index)) == 0, "Train/Val overlap!"
    assert len(set(X_train.index) & set(X_test.index)) == 0, "Train/Test overlap!"
    assert len(set(X_val.index) & set(X_test.index)) == 0, "Val/Test overlap!"
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test, feature_names):
    """Scale features using RobustScaler fitted only on training data.
    
    RobustScaler is less sensitive to outliers than StandardScaler.
    
    Args:
        X_train, X_val, X_test: Feature DataFrames.
        feature_names: List of column names for output DataFrames.
    
    Returns:
        Tuple of (X_train_df, X_val_df, X_test_df, scaler) where DataFrames
        contain scaled values and scaler is the fitted RobustScaler.
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    return X_train_df, X_val_df, X_test_df, scaler


def select_features(X_train_df, y_train, feature_names):
    """Select top features using Random Forest and XGBoost importance.
    
    Trains both RF and XGBoost on training data only, extracts feature importances,
    and creates three feature sets: RF top 10, XGBoost top 10, and their union.
    
    Args:
w        y_train: Training target.
        feature_names: List of all feature names.
    
    Returns:
        Tuple of (feature_combinations dict, rf_importance DataFrame, xgb_importance DataFrame).
    """
    # Random Forest importance
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, 
                               random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_df, y_train)
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    RF_TOP10 = rf_importance.head(10)['feature'].tolist()
    
    # XGBoost importance
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, 
                       random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    xgb.fit(X_train_df, y_train)
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    XGB_TOP10 = xgb_importance.head(10)['feature'].tolist()
    
    # Combined set (union of RF and XGB top features)
    COMBINED_TOP10 = list(set(RF_TOP10) | set(XGB_TOP10))
    
    feature_combinations = {
        'RF_TOP10': RF_TOP10,
        'XGB_TOP10': XGB_TOP10,
        'COMBINED_TOP10': COMBINED_TOP10
    }
    
    return feature_combinations, rf_importance, xgb_importance


def test_feature_combinations(feature_combinations, X_train_df, X_val_df, y_train, y_val):
    """Evaluate all feature combinations using XGBoost with polynomial features.
    
    For each combination, creates degree-2 polynomial features, trains XGBoost,
    and computes train/validation R2 scores to identify best feature set.
    
    Args:
        feature_combinations: Dict mapping combo names to feature lists.
        X_train_df, X_val_df: Scaled feature DataFrames.
        y_train, y_val: Target Series.
    
    Returns:
        DataFrame with columns: combination, n_features, n_poly_features, 
        train_r2, val_r2, gap. Sorted by val_r2 descending.
    """
    results = []
    
    for combo_name, selected_features in feature_combinations.items():
        pipe = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False))])
        X_train_p = pipe.fit_transform(X_train_df[selected_features])
        X_val_p = pipe.transform(X_val_df[selected_features])
        
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, 
                            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        model.fit(X_train_p, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train_p))
        val_r2 = r2_score(y_val, model.predict(X_val_p))
        
        results.append({
            'combination': combo_name,
            'n_features': len(selected_features),
            'n_poly_features': X_train_p.shape[1],
            'train_r2': train_r2,
            'val_r2': val_r2,
            'gap': train_r2 - val_r2
        })
    
    return pd.DataFrame(results).sort_values('val_r2', ascending=False)


def create_polynomial_features(X_train_df, X_val_df, X_test_df, selected_features):
    """Generate degree-2 polynomial features for selected feature subset.
    
    Fits PolynomialFeatures on training data only to prevent data leakage.
    
    Args:
        X_train_df, X_val_df, X_test_df: Scaled feature DataFrames.
        selected_features: List of feature names to include.
    
    Returns:
        Tuple of (X_train_poly, X_val_poly, X_test_poly, pipeline) where
        arrays contain polynomial features and pipeline is the fitted transformer.
    """
    pipeline = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False))])
    
    X_train_poly = pipeline.fit_transform(X_train_df[selected_features])
    X_val_poly = pipeline.transform(X_val_df[selected_features])
    X_test_poly = pipeline.transform(X_test_df[selected_features])
    
    return X_train_poly, X_val_poly, X_test_poly, pipeline
