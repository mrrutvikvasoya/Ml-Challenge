# ML Challenge - Part 1: Regression Pipeline

## Project Overview

This project implements a complete machine learning pipeline for regression prediction on a dataset with 273 features and 10,000 samples. The pipeline includes exploratory data analysis, feature selection, hyperparameter tuning, and model evaluation.

## Results Summary

| Metric | Value |
|--------|-------|
| **Test R²** | **0.9686** |
| **Test RMSE** | **0.0424** |
| Test MAE | 0.0281 |
| CV Mean R² | 0.9643 (±0.0020) |

## Project Structure

```
ml/
├── main.py                 # Main training pipeline
├── eda_analysis.py         # Exploratory data analysis
├── requirements.txt        # Dependencies
├── src/                    # Source modules
│   ├── config.py           # Configuration constants
│   ├── utils.py            # Helper functions
│   ├── tuning.py           # Optuna hyperparameter tuning
│   ├── preprocessing.py    # Data preprocessing
│   ├── data_loader.py      # Data loading utilities
│   ├── eda_analyzer.py     # EDA analysis functions
│   └── visualization.py    # Plotting functions
└── outputs/                # Generated outputs
```

## Methodology

### 1. Exploratory Data Analysis
- Data quality checks (no missing values, no duplicates)
- Outlier detection using Z-score method (47% of rows contain outliers)
- Feature correlation analysis
- Feature importance ranking using Random Forest

### 2. Feature Selection
- Random Forest feature importance
- XGBoost feature importance
- Combined top features (union of RF and XGB top 10)
- **Final selection: 14 features**

### 3. Feature Engineering
- RobustScaler normalization (fitted on training data only)
- Polynomial features (degree=2): 14 → 119 features

### 4. Model Selection
Tested models with polynomial features:
| Model | Val R² | Val RMSE |
|-------|--------|----------|
| CatBoost | 0.9465 | 0.0556 |
| HistGradientBoosting | 0.9457 | 0.0560 |
| XGBoost | 0.9357 | 0.0609 |
| ElasticNet | 0.1448 | 0.2222 |

**Selected: CatBoost** (best validation performance)

### 5. Hyperparameter Tuning
- Optuna optimization (150 trials)
- Objective: Minimize RMSE
- Best parameters:
  - iterations: 454
  - learning_rate: 0.053
  - depth: 8
  - l2_leaf_reg: 2.22
  - min_data_in_leaf: 90

### 6. Model Validation
- 5-Fold Cross-Validation on Train+Val set
- Final model trained on combined Train+Val data
- Test set used only once for final evaluation

## Data Split
- Training: 70% (7,004 samples)
- Validation: 15% (1,496 samples)
- Test: 15% (1,500 samples)

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run EDA Analysis
```bash
python eda_analysis.py
```

### Run Training Pipeline
```bash
python main.py
```

This will:
1. Load and preprocess data
2. Perform feature selection
3. Train and tune CatBoost model
4. Evaluate on test set
5. Generate predictions on EVAL_1.csv
6. Save predictions to `outputs/EVAL_target01_1.csv`

## Output Files

- **EVAL_target01_1.csv**: Predictions on evaluation dataset (10,000 rows)
- **feature_importance.png**: RF and XGB feature importance plots
- **learning_curves.png**: Training vs Test RMSE across data sizes
- **residual_analysis.png**: Residual distribution and scatter plots

## Key Design Decisions

1. **No Data Leakage**: All preprocessing (scaling, polynomial features) fitted on training data only
2. **RobustScaler**: Chosen for robustness to outliers
3. **Polynomial Features**: Captures non-linear relationships
4. **CatBoost**: Best balance of performance and generalization
5. **Optuna**: Efficient hyperparameter search with TPE sampler

## Dependencies

- Python 3.8+
- numpy, pandas
- scikit-learn
- catboost, xgboost
- optuna
- matplotlib, seaborn
