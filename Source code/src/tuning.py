"""
Optuna Hyperparameter Tuning
============================
Functions for CatBoost hyperparameter optimization using Optuna with TPE sampler.
Objective: minimize validation RMSE.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import optuna

from .config import RANDOM_STATE, N_OPTUNA_TRIALS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_optuna_params(trial):
    """Define the hyperparameter search space for CatBoost.
    
    Uses Optuna trial object to sample hyperparameters within defined ranges.
    Learning rate uses log scale for better exploration of smaller values.
    
    Args:
        trial: Optuna Trial object.
    
    Returns:
        Dict of CatBoost hyperparameters for this trial.
    """
    return {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'random_seed': RANDOM_STATE,
        'verbose': 0
    }


def run_optuna_tuning(X_train, X_val, y_train, y_val, n_trials=N_OPTUNA_TRIALS):
    """Run Optuna hyperparameter optimization for CatBoost.
    
    Creates an Optuna study that minimizes validation RMSE using TPE sampler.
    Each trial trains a CatBoost model with sampled hyperparameters and 
    evaluates on the validation set.
    
    Args:
        X_train: Training features (numpy array).
        X_val: Validation features.
        y_train: Training target.
        y_val: Validation target.
        n_trials: Number of optimization trials (default: N_OPTUNA_TRIALS).
    
    Returns:
        optuna.Study object containing best parameters and trial history.
    """
    def objective(trial):
        params = get_optuna_params(trial)
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        return rmse
    
    study = optuna.create_study(
        direction='minimize', 
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study
