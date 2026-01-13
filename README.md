# 🤖 ML Prediction: Continuous & Rule-Based Regression

Advanced machine learning project implementing ensemble methods and rule discovery for dual-target prediction on high-dimensional data.

## 🎯 Key Results

- **96.86% R² Score** - Continuous target prediction with optimized CatBoost
- **Perfect Prediction (R² = 1.0)** - Rule-based system using only 4 features
- **Edge-Ready** - Simple if-else rules for deployment without ML libraries

## 📊 Project Overview

This project tackles two prediction challenges on a dataset with 10,000 samples and 273 features:
dataset : https://drive.google.com/drive/folders/1GXB-f_2PaOfgUCTWCociwLuEO8StXsFN?usp=sharing

**Part 1: Continuous Target Prediction**
- Ensemble learning with polynomial feature engineering
- Hyperparameter optimization using Optuna
- Achieved 96.86% R² with strict data leakage prevention

**Part 2: Rule-Based Prediction**
- Discovered deterministic piecewise-linear structure
- Perfect prediction using only 4 features and simple thresholds
- Suitable for edge devices without ML libraries

## 🛠️ Tech Stack

- **Python** - scikit-learn, XGBoost, CatBoost
- **Optimization** - Optuna (150 trials)
- **Analysis** - NumPy, Pandas, Matplotlib

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/ml-prediction-project.git

# Install dependencies
pip install -r requirements.txt

# Run prediction
python src/predict.py --input data.csv
```

## 📈 Methodology

### Continuous Target
1. **Preprocessing** - RobustScaler for outlier handling
2. **Feature Selection** - Combined RF + XGBoost importance (14 features)
3. **Feature Engineering** - Polynomial features (14 → 119)
4. **Model Training** - CatBoost with Optuna optimization
5. **Validation** - 5-fold CV (R² = 0.9627 ± 0.0019)

### Rule-Based Target
1. **Analysis** - Correlation + Decision Tree exploration
2. **Discovery** - Piecewise-linear structure identification
3. **Rules** - 4 regions based on feat_121 thresholds
4. 
## 🔬 Key Features

- **Zero Data Leakage** - Strict train/val/test separation
- **Robust Preprocessing** - Outlier-resistant scaling
- **Smart Feature Selection** - Combined RF + XGBoost
- **Polynomial Engineering** - Captures non-linear relationships
- **Automated Tuning** - Optuna hyperparameter optimization
- **Edge Deployment** - Simple rule extraction

## 📊 Performance Metrics

| Target | R² Score | RMSE | Features Used |
|--------|----------|------|---------------|
| Continuous | 0.9686 | 0.0424 | 119 (polynomial) |
| Rule-Based | 1.0000 | 0.0000 | 4 (original) |


## 📄 License

MIT License - feel free to use for learning and research.
---

⭐ **Star this repo** if you find it useful!
