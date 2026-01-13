#!/usr/bin/env python3
"""
Part 2: Rule Discovery for Target02 Prediction
===============================================

This script discovers simple if-else rules for predicting target02
that can be implemented on an edge device without ML libraries.

Usage:
    python part2_rule_discovery.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'problem_1/dataset_1.csv'
TARGET_PATH = 'problem_1/target_1.csv'
OUTPUT_DIR = 'outputs'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_data():
    """Load dataset and target files."""
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    target_df = pd.read_csv(TARGET_PATH)
    
    X = df.values
    y = target_df['target02'].values
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
    
    return df, X, y


def split_data(X, y):
    """Split data into training and test sets (80/20)."""
    print(f"\n{'=' * 60}")
    print("STEP 2: Train/Test Split (80/20)")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def analyze_correlations(X_train, y_train, feature_names):
    """Analyze feature correlations with target."""
    print(f"\n{'=' * 60}")
    print("STEP 3: Feature Correlation Analysis")
    print("=" * 60)
    
    train_df = pd.DataFrame(X_train, columns=feature_names)
    correlations = train_df.corrwith(pd.Series(y_train))
    top_corr = correlations.abs().sort_values(ascending=False).head(15)
    
    # Plot correlation bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['forestgreen' if correlations[f] > 0 else 'crimson' for f in top_corr.index]
    ax.barh(range(len(top_corr)), [correlations[f] for f in top_corr.index], color=colors)
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr.index)
    ax.set_xlabel('Correlation with target02')
    ax.set_title('Top 15 Features by Correlation with Target02')
    ax.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part2_correlation.png'), bbox_inches='tight')
    plt.close()
    
    print("\nTop 5 correlated features:")
    for feat in top_corr.index[:5]:
        print(f"  {feat}: {correlations[feat]:+.4f}")
    
    print(f"\n[SAVED] {OUTPUT_DIR}/part2_correlation.png")
    
    return correlations


def train_decision_tree(X_train, y_train, feature_names):
    """Train decision tree to identify important features and splits."""
    print(f"\n{'=' * 60}")
    print("STEP 4: Decision Tree Analysis")
    print("=" * 60)
    
    tree = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    
    # Feature importances
    importances = pd.Series(tree.feature_importances_, index=feature_names)
    top_features = importances[importances > 0].sort_values(ascending=False)
    
    print("\nImportant features from Decision Tree:")
    for feat, imp in top_features.items():
        print(f"  {feat}: {imp:.4f}")
    
    # Plot decision tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree, feature_names=list(feature_names), filled=True, 
              rounded=True, fontsize=9, ax=ax, proportion=True)
    ax.set_title('Decision Tree for Target02 Prediction (depth=3)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part2_decision_tree.png'), bbox_inches='tight')
    plt.close()
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(8, 4))
    top_features.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance from Decision Tree')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part2_feature_importance.png'), bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVED] {OUTPUT_DIR}/part2_decision_tree.png")
    print(f"[SAVED] {OUTPUT_DIR}/part2_feature_importance.png")
    
    # Print tree structure
    print("\nTree Structure:")
    print(export_text(tree, feature_names=list(feature_names)))
    
    return tree, top_features

#ChatGPT
def visualize_regions(X_train, y_train, idx_121):
    """Visualize the 4 regions based on feat_121 thresholds."""
    print(f"\n{'=' * 60}")
    print("STEP 5: Visualizing Feature Regions")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # feat_121 vs target02 with thresholds
    ax = axes[0]
    scatter = ax.scatter(X_train[:, idx_121], y_train, c=y_train, 
                         cmap='RdYlGn', alpha=0.5, s=10)
    ax.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='Threshold 0.2')
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Threshold 0.5')
    ax.axvline(x=0.7, color='blue', linestyle='--', linewidth=2, label='Threshold 0.7')
    ax.set_xlabel('feat_121')
    ax.set_ylabel('target02')
    ax.set_title('feat_121 vs target02 with Decision Thresholds')
    ax.legend()
    
    # Colored by region
    ax = axes[1]
    regions = [
        ('Region 1: f121 < 0.2', X_train[:, idx_121] < 0.2, 'red'),
        ('Region 2: 0.2 ≤ f121 < 0.5', (X_train[:, idx_121] >= 0.2) & (X_train[:, idx_121] < 0.5), 'orange'),
        ('Region 3: 0.5 ≤ f121 < 0.7', (X_train[:, idx_121] >= 0.5) & (X_train[:, idx_121] < 0.7), 'green'),
        ('Region 4: f121 ≥ 0.7', X_train[:, idx_121] >= 0.7, 'blue')
    ]
    for name, mask, color in regions:
        ax.scatter(X_train[mask, idx_121], y_train[mask], c=color, alpha=0.5, s=10, label=name)
    ax.set_xlabel('feat_121')
    ax.set_ylabel('target02')
    ax.set_title('4 Regions Based on feat_121 Thresholds')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part2_feat121_regions.png'), bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {OUTPUT_DIR}/part2_feat121_regions.png")

#ChatGPT
def discover_formulas(X_train, y_train, idx_121, idx_225, idx_259, idx_195):
    """Discover linear formulas for each region."""
    print(f"\n{'=' * 60}")
    print("STEP 6: Discovering Formulas for Each Region")
    print("=" * 60)
    
    regions = [
        ("feat_121 < 0.2", X_train[:, idx_121] < 0.2),
        ("0.2 <= feat_121 < 0.5", (X_train[:, idx_121] >= 0.2) & (X_train[:, idx_121] < 0.5)),
        ("0.5 <= feat_121 < 0.7", (X_train[:, idx_121] >= 0.5) & (X_train[:, idx_121] < 0.7)),
        ("feat_121 >= 0.7", X_train[:, idx_121] >= 0.7)
    ]
    
    formulas = []
    
    for name, mask in regions:
        X_region = X_train[mask][:, [idx_225, idx_259, idx_195]]
        y_region = y_train[mask]
        
        lr = LinearRegression()
        lr.fit(X_region, y_region)
        r2 = r2_score(y_region, lr.predict(X_region))
        
        c225, c259, c195 = lr.coef_
        formulas.append((c225, c259, c195))
        
        print(f"\n{name}:")
        print(f"  Samples: {len(y_region)}")
        print(f"  R² = {r2:.6f}")
        print(f"  Formula: {c225:.2f}*f225 + {c259:.2f}*f259 + {c195:.2f}*f195")
    
    # Plot coefficients
    fig, ax = plt.subplots(figsize=(10, 5))
    region_names = ['f121 < 0.2', '0.2 ≤ f121 < 0.5', '0.5 ≤ f121 < 0.7', 'f121 ≥ 0.7']
    x = np.arange(len(region_names))
    width = 0.25
    
    ax.bar(x - width, [f[0] for f in formulas], width, label='feat_225', color='steelblue')
    ax.bar(x, [f[1] for f in formulas], width, label='feat_259', color='coral')
    ax.bar(x + width, [f[2] for f in formulas], width, label='feat_195', color='seagreen')
    
    ax.set_xlabel('Region')
    ax.set_ylabel('Coefficient')
    ax.set_title('Formula Coefficients by Region')
    ax.set_xticks(x)
    ax.set_xticklabels(region_names)
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part2_coefficients.png'), bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVED] {OUTPUT_DIR}/part2_coefficients.png")
    
    return formulas


def predict_target02(X, idx_121, idx_225, idx_259, idx_195):
    """Apply the discovered conditional formulas."""
    predictions = np.zeros(len(X))
    
    # Region 1: feat_121 < 0.2
    mask1 = X[:, idx_121] < 0.2
    predictions[mask1] = 1.75*X[mask1, idx_225] - 1.85*X[mask1, idx_259] - 0.75*X[mask1, idx_195]
    
    # Region 2: 0.2 <= feat_121 < 0.5
    mask2 = (X[:, idx_121] >= 0.2) & (X[:, idx_121] < 0.5)
    predictions[mask2] = -0.65*X[mask2, idx_225] + 1.55*X[mask2, idx_259] + 0.55*X[mask2, idx_195]
    
    # Region 3: 0.5 <= feat_121 < 0.7
    mask3 = (X[:, idx_121] >= 0.5) & (X[:, idx_121] < 0.7)
    predictions[mask3] = 0.55*X[mask3, idx_225] + 1.25*X[mask3, idx_259] - 1.65*X[mask3, idx_195]
    
    # Region 4: feat_121 >= 0.7
    mask4 = X[:, idx_121] >= 0.7
    predictions[mask4] = 0.75*X[mask4, idx_225] - 0.55*X[mask4, idx_259] + 1.55*X[mask4, idx_195]
    
    return predictions


def validate_model(X_train, X_test, y_train, y_test, idx_121, idx_225, idx_259, idx_195):
    """Validate the discovered rules on training and test sets."""
    print(f"\n{'=' * 60}")
    print("STEP 7: Model Validation")
    print("=" * 60)
    
    # Training set
    y_train_pred = predict_target02(X_train, idx_121, idx_225, idx_259, idx_195)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    print(f"\nTraining Set:")
    print(f"  R² = {train_r2:.6f}")
    print(f"  RMSE = {train_rmse:.6f}")
    
    # Test set
    y_test_pred = predict_target02(X_test, idx_121, idx_225, idx_259, idx_195)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\nTest Set:")
    print(f"  R² = {test_r2:.6f}")
    print(f"  RMSE = {test_rmse:.6f}")
    
    # Plot predicted vs actual
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_test_pred, alpha=0.5, s=10, c='steelblue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual target02')
    ax.set_ylabel('Predicted target02')
    ax.set_title(f'Predicted vs Actual (Test Set)\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part2_pred_vs_actual.png'), bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVED] {OUTPUT_DIR}/part2_pred_vs_actual.png")
    
    return train_r2, test_r2


def print_summary():
    """Print final summary of discovered rules."""
    print(f"\n{'=' * 60}")
    print("SUMMARY: Discovered Rules for Target02")
    print("=" * 60)
    
    print("""
┌─────────────────────────┬─────────────────────────────────────────────────┐
│ Condition               │ Formula                                         │
├─────────────────────────┼─────────────────────────────────────────────────┤
│ feat_121 < 0.2          │ 1.75*feat_225 - 1.85*feat_259 - 0.75*feat_195   │
│ 0.2 ≤ feat_121 < 0.5    │ -0.65*feat_225 + 1.55*feat_259 + 0.55*feat_195  │
│ 0.5 ≤ feat_121 < 0.7    │ 0.55*feat_225 + 1.25*feat_259 - 1.65*feat_195   │
│ feat_121 ≥ 0.7          │ 0.75*feat_225 - 0.55*feat_259 + 1.55*feat_195   │
└─────────────────────────┴─────────────────────────────────────────────────┘

Features Used:
  - Condition feature: feat_121 (index 121)
  - Calculation features: feat_225, feat_259, feat_195 (indices 225, 259, 195)
""")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("PART 2: RULE DISCOVERY FOR TARGET02 PREDICTION")
    print("=" * 60)
    
    # Feature indices (discovered through analysis)
    IDX_121 = 121  # Primary condition feature
    IDX_225 = 225  # Calculation feature
    IDX_259 = 259  # Calculation feature
    IDX_195 = 195  # Calculation feature
    
    # Step 1: Load data
    df, X, y = load_data()
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 3: Correlation analysis
    correlations = analyze_correlations(X_train, y_train, df.columns)
    
    # Step 4: Decision tree analysis
    tree, top_features = train_decision_tree(X_train, y_train, df.columns)
    
    # Step 5: Visualize regions
    visualize_regions(X_train, y_train, IDX_121)
    
    # Step 6: Discover formulas
    formulas = discover_formulas(X_train, y_train, IDX_121, IDX_225, IDX_259, IDX_195)
    
    # Step 7: Validate model
    train_r2, test_r2 = validate_model(
        X_train, X_test, y_train, y_test, 
        IDX_121, IDX_225, IDX_259, IDX_195
    )
    
    # Print summary
    print_summary()
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
