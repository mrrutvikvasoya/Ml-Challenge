#!/usr/bin/env python3
"""
ML Project - Exploratory Data Analysis
========================================

This script performs comprehensive EDA on the ML challenge dataset:
- Data quality checks (missing values, duplicates, data types)
- Variance analysis (zero/low variance features)
- Outlier detection (Z-score method)
- Feature correlations (feature-feature and feature-target)
- Feature importance (RandomForest)
- Consensus features identification

All results are saved to outputs/results/*.json.
All visualizations are saved to outputs/plots/*.png

Usage:
    python eda_analysis.py

"""

import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_loader import DataLoader
from src.eda_analyzer import EDAAnalyzer
from src.visualization import Visualizer


def main():
    """Run the complete EDA pipeline."""
    
    print("="*70)
    print("ML CHALLENGE - EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    loader = DataLoader(data_dir="problem_1")
    dataset_train, target_train, dataset_eval = loader.load_all()
    
    # Print data info
    print("\n Data Information:")
    info = loader.get_data_info()
    print(f"   Training features shape: {info['train_shape']}")
    print(f"   Target shape: {info['target_shape']}")
    print(f"   Evaluation shape: {info['eval_shape']}")
    print(f"   Number of features: {info['num_features']}")
    print(f"   Target columns: {info['target_columns']}")
    
    # Validate data quality
    print("\n Data Quality Check:")
    validation = loader.validate_data()
    print(f"   Missing values (train): {validation['train_missing_total']}")
    print(f"   Missing values (eval): {validation['eval_missing_total']}")
    print(f"   Duplicates (train): {validation['train_duplicates']}")
    print(f"   Infinite values (train): {validation['train_infinite']}")
    
    # =========================================================================
    # 2. DATA INSPECTION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: DATA INSPECTION")
    print("="*70)
    
    print("\n Training Data Preview:")
    print(dataset_train.head())
    
    print("\n Target Data Preview:")
    print(target_train.head())
    
    print("\n Data Types:")
    print(dataset_train.dtypes.value_counts())
    
    print("\n Target Statistics:")
    print(target_train.describe())
    
    print("\nUnique Values per Feature (first 10):")
    print(dataset_train.nunique().head(10))
    
    print("\n Range Comparison (Train vs Eval) - first 5 features:")
    for col in dataset_train.columns[:5]:
        train_min, train_max = dataset_train[col].min(), dataset_train[col].max()
        eval_min, eval_max = dataset_eval[col].min(), dataset_eval[col].max()
        print(f"   {col}: Train [{train_min:.2f}, {train_max:.2f}] | Eval [{eval_min:.2f}, {eval_max:.2f}]")
    
    # =========================================================================
    # 3. EDA ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: EDA ANALYSIS")
    print("="*70)
    
    analyzer = EDAAnalyzer(
        dataset_train=dataset_train,
        target_train=target_train,
        dataset_eval=dataset_eval,
        results_dir="outputs/results"
    )
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    # Print summary
    analyzer.print_summary()
    
    # Save results to JSON
    analyzer.save_results("eda_results.json")
    
    # =========================================================================
    # 4. GENERATE VISUALIZATIONS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*70)
    
    visualizer = Visualizer(plots_dir="outputs/plots")
    saved_plots = visualizer.generate_all_plots(
        dataset_train=dataset_train,
        target_train=target_train,
        eda_results=results
    )
    
    # =========================================================================
    # 5. SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("EDA COMPLETE - SUMMARY")
    print("="*70)
    
    print("\n OUTPUT FILES:")
    print(f"   Results: outputs/results/eda_results.json")
    print(f"   Plots: {len(saved_plots)} images in outputs/plots/")
    
    print("\n KEY FINDINGS:")
    
    # Outliers
    outliers = results.get('outliers', {})
    print(f"\n   OUTLIERS:")
    print(f"   - {outliers.get('features_with_outliers', 'N/A')} features contain outliers")
    print(f"   - {outliers.get('rows_with_outliers_percentage', 'N/A')}% of rows have at least one outlier")
    
    # Correlations
    correlations = results.get('correlations', {})
    print(f"\n   HIGHLY CORRELATED PAIRS:")
    print(f"   - {correlations.get('highly_correlated_pairs_count', 'N/A')} pairs with |r| > 0.4")
    
    # Feature importance
    importance = results.get('feature_importance', {})
    print(f"\n   TOP FEATURES BY RANDOM FOREST:")
    if importance:
        print(f"   - target01: {importance['target01_importance']['top_15_features'][:5]}")
        print(f"   - target02: {importance['target02_importance']['top_15_features'][:5]}")
    
    # Consensus features
    consensus = results.get('consensus_features', {})
    print(f"\n   CONSENSUS FEATURES (Correlation + RF):")
    if consensus:
        print(f"   - target01: {consensus['target01_consensus']['count']} features")
        print(f"     {consensus['target01_consensus']['features']}")
        print(f"   - target02: {consensus['target02_consensus']['count']} features")
        print(f"     {consensus['target02_consensus']['features']}")
    
    print("\n" + "="*70)
    print("EDA ANALYSIS COMPLETE!")
    print("="*70)
    print()
    
    return results


if __name__ == "__main__":
    results = main()