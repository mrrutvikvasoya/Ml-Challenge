#!/usr/bin/env python3
"""
ML Pipeline - Main Script
=========================
Modular feature selection and model training pipeline.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

# Import from src modules
from src.config import RANDOM_STATE, OVERFIT_THRESHOLD, N_OPTUNA_TRIALS, OUTPUT_DIR
from src.utils import print_section, create_model, train_and_evaluate
from src.tuning import run_optuna_tuning
from src.preprocessing import (
    load_data, split_data, scale_data,
    select_features, test_feature_combinations,
    create_polynomial_features
)

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("Loading data...")
X, y, feature_names = load_data()
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

print("\nSplitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("[OK] No index overlap between splits")

X_train_df, X_val_df, X_test_df, scaler = scale_data(X_train, X_val, X_test, feature_names)

# =============================================================================
# 2. FEATURE SELECTION
# =============================================================================
print("\nFeature selection with Random Forest...")
print("Feature selection with XGBoost...")
feature_combinations, rf_importance, xgb_importance = select_features(X_train_df, y_train, feature_names)

print(f"\nFeature combinations to test:")
for name, features in feature_combinations.items():
    print(f"  {name}: {len(features)} features")

# Save feature importance plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
rf_plot = rf_importance.head(10).sort_values('importance')
ax1.barh(rf_plot['feature'], rf_plot['importance'], color='steelblue')
ax1.set_title('Random Forest Top 10')
xgb_plot = xgb_importance.head(10).sort_values('importance')
ax2.barh(xgb_plot['feature'], xgb_plot['importance'], color='coral')
ax2.set_title('XGBoost Top 10')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150)
plt.close()

# =============================================================================
# 3. TEST FEATURE COMBINATIONS
# =============================================================================
print_section("TESTING ALL FEATURE COMBINATIONS WITH XGBOOST")

combo_df = test_feature_combinations(feature_combinations, X_train_df, X_val_df, y_train, y_val)

for _, row in combo_df.iterrows():
    print(f"\n--- {row['combination']} ({row['n_features']} features) ---")
    print(f"  Poly features: {row['n_poly_features']}")
    print(f"  Train R2: {row['train_r2']:.4f}")
    print(f"  Val R2:   {row['val_r2']:.4f}")
    print(f"  Gap:      {row['gap']:.4f}")

print_section("FEATURE COMBINATION COMPARISON (selected by Val R2)")
print("\n" + combo_df.to_string(index=False))

best_combo = combo_df.iloc[0]
print(f"\n>>> Best: {best_combo['combination']} with Val R2={best_combo['val_r2']:.4f}")

SELECTED_FEATURES = feature_combinations[best_combo['combination']]
print(f"\nUsing {best_combo['combination']} ({len(SELECTED_FEATURES)} features) for Optuna tuning...")

# =============================================================================
# 4. POLYNOMIAL FEATURES
# =============================================================================
print_section("PREPROCESSING PIPELINE")

X_train_poly, X_val_poly, X_test_poly, pipeline = create_polynomial_features(
    X_train_df, X_val_df, X_test_df, SELECTED_FEATURES
)
print(f"Original features: {len(SELECTED_FEATURES)}")
print(f"After polynomial: {X_train_poly.shape[1]}")
print("[OK] Pipeline fitted on train, transformed val/test")

# =============================================================================
# 5. BASELINE MODEL TRAINING
# =============================================================================
print_section("MODEL TRAINING WITH POLYNOMIAL FEATURES")

model_types = ['CatBoost', 'XGBoost', 'HistGB', 'ElasticNet']
results = []
for model_type in model_types:
    model = create_model(model_type)
    result = train_and_evaluate(model, model_type, X_train_poly, X_val_poly, y_train, y_val)
    results.append(result)

results_df = pd.DataFrame(results).sort_values('val_r2', ascending=False)
print_section("RESULTS SUMMARY (selected by Val R2)")
print("\n" + results_df.to_string(index=False))

# =============================================================================
# 6. OPTUNA HYPERPARAMETER TUNING
# =============================================================================
print_section("OPTUNA HYPERPARAMETER TUNING (CatBoost)")

print(f"\nTuning CatBoost ({N_OPTUNA_TRIALS} trials)...")
study = run_optuna_tuning(X_train_poly, X_val_poly, y_train, y_val)
print(f"Best CatBoost Val RMSE: {study.best_value:.4f}")

# =============================================================================
# 7. BEST PARAMETERS
# =============================================================================
print_section("BEST PARAMETERS FROM OPTUNA")

best_params = study.best_params.copy()
best_params['random_seed'] = RANDOM_STATE
best_params['verbose'] = 0

print("\nCatBoost best params:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# =============================================================================
# 8. MERGE TRAIN + VAL
# =============================================================================
print_section("MERGING TRAIN + VALIDATION SETS")

X_trainval_poly = np.vstack([X_train_poly, X_val_poly])
y_trainval = pd.concat([y_train, y_val], ignore_index=True)

print(f"Train samples: {len(X_train_poly)}")
print(f"Val samples: {len(X_val_poly)}")
print(f"Merged samples: {len(X_trainval_poly)}")

# =============================================================================
# 9. CROSS-VALIDATION
# =============================================================================
print_section("5-FOLD CROSS-VALIDATION (on Train+Val)")

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []
cv_rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval_poly)):
    X_cv_train = X_trainval_poly[train_idx]
    X_cv_val = X_trainval_poly[val_idx]
    y_cv_train = y_trainval.iloc[train_idx]
    y_cv_val = y_trainval.iloc[val_idx]
    
    cv_model = CatBoostRegressor(**best_params)
    cv_model.fit(X_cv_train, y_cv_train)
    cv_pred = cv_model.predict(X_cv_val)
    
    cv_r2 = r2_score(y_cv_val, cv_pred)
    cv_rmse = np.sqrt(mean_squared_error(y_cv_val, cv_pred))
    cv_scores.append(cv_r2)
    cv_rmses.append(cv_rmse)
    print(f"  Fold {fold+1}: R2 = {cv_r2:.4f}, RMSE = {cv_rmse:.4f}")

cv_scores = np.array(cv_scores)
cv_rmses = np.array(cv_rmses)
print(f"\nCV Mean R2:   {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
print(f"CV Mean RMSE: {cv_rmses.mean():.4f} (+/-{cv_rmses.std():.4f})")

# =============================================================================
# 10. TRAIN FINAL MODEL
# =============================================================================
print_section("TRAINING FINAL MODEL ON TRAIN+VAL")

final_model = CatBoostRegressor(**best_params)
final_model.fit(X_trainval_poly, y_trainval)

trainval_pred = final_model.predict(X_trainval_poly)
trainval_r2 = r2_score(y_trainval, trainval_pred)
trainval_rmse = np.sqrt(mean_squared_error(y_trainval, trainval_pred))

print(f"\nTrain+Val R2:   {trainval_r2:.4f}")
print(f"Train+Val RMSE: {trainval_rmse:.4f}")

# =============================================================================
# 11. FINAL TEST EVALUATION
# =============================================================================
print_section("FINAL TEST EVALUATION")

test_pred = final_model.predict(X_test_poly)
test_r2 = r2_score(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_mae = mean_absolute_error(y_test, test_pred)

print(f"\n[FINAL RESULTS]")
print(f"  CV Mean R2:   {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
print(f"  CV Mean RMSE: {cv_rmses.mean():.4f} (+/-{cv_rmses.std():.4f})")
print(f"  Test R2:      {test_r2:.4f}")
print(f"  Test RMSE:    {test_rmse:.4f}")
print(f"  Test MAE:     {test_mae:.4f}")

# =============================================================================
# 12. LEARNING CURVES
# =============================================================================
print_section("LEARNING CURVES")

train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
train_rmses_curve = []
val_rmses_curve = []

for size in train_sizes:
    n_samples = int(len(X_trainval_poly) * size)
    X_subset = X_trainval_poly[:n_samples]
    y_subset = y_trainval.iloc[:n_samples]
    
    model = CatBoostRegressor(**best_params)
    model.fit(X_subset, y_subset)
    
    train_pred = model.predict(X_subset)
    test_pred_curve = model.predict(X_test_poly)
    
    train_rmse = np.sqrt(mean_squared_error(y_subset, train_pred))
    test_rmse_curve = np.sqrt(mean_squared_error(y_test, test_pred_curve))
    
    train_rmses_curve.append(train_rmse)
    val_rmses_curve.append(test_rmse_curve)
    print(f"  {int(size*100)}% data: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse_curve:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
train_size_labels = [f"{int(s*100)}%" for s in train_sizes]
ax.plot(train_size_labels, train_rmses_curve, 'o-', label='Training RMSE', color='blue', linewidth=2, markersize=8)
ax.plot(train_size_labels, val_rmses_curve, 'o-', label='Test RMSE', color='red', linewidth=2, markersize=8)
ax.set_xlabel('Training Set Size (Train+Val)', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('Learning Curves - CatBoost', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/learning_curves.png', dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/learning_curves.png")

# =============================================================================
# 13. RESIDUAL ANALYSIS
# =============================================================================
print_section("RESIDUAL ANALYSIS")

residuals = y_test.values - test_pred
print(f"Residual Statistics:")
print(f"  Mean: {np.mean(residuals):.4f}")
print(f"  Std:  {np.std(residuals):.4f}")
print(f"  Min:  {np.min(residuals):.4f}")
print(f"  Max:  {np.max(residuals):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(y_test, test_pred, alpha=0.5, s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Predicted vs Actual')

axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution')

axes[2].scatter(test_pred, residuals, alpha=0.5, s=20)
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Residual')
axes[2].set_title('Residuals vs Predicted')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/residual_analysis.png', dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/residual_analysis.png")

# =============================================================================
# 14. FINAL SUMMARY
# =============================================================================
print_section("FINAL SUMMARY")

print(f"""
Model: CatBoost with Optuna-tuned hyperparameters
Features: {len(SELECTED_FEATURES)} original -> {X_trainval_poly.shape[1]} polynomial

Training:
  Train+Val samples: {len(X_trainval_poly)}
  Train+Val R2: {trainval_r2:.4f}
  Train+Val RMSE: {trainval_rmse:.4f}

Cross-Validation (5-Fold on Train+Val):
  CV Mean R2: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})
  CV Mean RMSE: {cv_rmses.mean():.4f} (+/-{cv_rmses.std():.4f})

Test Set (n={len(y_test)}):
  Test R2: {test_r2:.4f}
  Test RMSE: {test_rmse:.4f}
  Test MAE: {test_mae:.4f}
""")

print_section("CONCLUSION")
print(f"\n[RESULT] Final Model: CatBoost")
print(f"[RESULT] Test R2: {test_r2:.4f}")
print(f"[RESULT] Test RMSE: {test_rmse:.4f}")
print(f"[RESULT] CV Mean R2: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")

gap = trainval_r2 - test_r2
if gap > OVERFIT_THRESHOLD:
    print(f"\n[WARNING] Possible overfitting (Train-Test gap: {gap:.4f})")
else:
    print(f"\n[OK] No significant overfitting (Train-Test gap: {gap:.4f})")

print("\n[DONE] Pipeline complete!")

# =============================================================================
# 15. GENERATE PREDICTIONS ON EVALUATION DATASET
# =============================================================================
print_section("GENERATING PREDICTIONS ON EVALUATION DATASET")

# Load evaluation dataset
EVAL_PATH = 'problem_1/EVAL_1.csv'
print(f"\nLoading: {EVAL_PATH}")
X_eval = pd.read_csv(EVAL_PATH)
print(f"Evaluation samples: {X_eval.shape[0]}")
print(f"Evaluation features: {X_eval.shape[1]}")

# Apply same preprocessing (scaler fitted on training data)
print("\nApplying preprocessing...")
X_eval_scaled = scaler.transform(X_eval)
X_eval_df = pd.DataFrame(X_eval_scaled, columns=feature_names)

# Select same features and apply polynomial transformation
X_eval_poly = pipeline.transform(X_eval_df[SELECTED_FEATURES])
print(f"Selected features: {len(SELECTED_FEATURES)}")
print(f"Polynomial features: {X_eval_poly.shape[1]}")

# Predict using final model (predictions maintain input row order)
print("\nGenerating predictions with final_model...")
eval_predictions = final_model.predict(X_eval_poly)

# Save predictions in required format (same row order as EVAL_1.csv)
# Row 1 in EVAL_1.csv -> Row 1 in output, Row 2 -> Row 2, etc.
OUTPUT_FILE = f'{OUTPUT_DIR}/EVAL_target01_1.csv'
predictions_df = pd.DataFrame({'target01': eval_predictions})
predictions_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n[SAVED] Predictions saved to: {OUTPUT_FILE}")
print(f"  Rows: {len(predictions_df)}")
print(f"  Prediction range: [{eval_predictions.min():.4f}, {eval_predictions.max():.4f}]")
print(f"  Prediction mean: {eval_predictions.mean():.4f}")

print("\n[DONE] Evaluation predictions complete!")

