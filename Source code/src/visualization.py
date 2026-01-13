"""
Visualization Module
All plotting functions for EDA analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, zscore
from pathlib import Path
from typing import List, Optional, Dict
import random


class Visualizer:
    """Generate and save EDA visualizations."""
    
    def __init__(self, plots_dir: str = "outputs/plots", figsize_default: tuple = (14, 6)):
        """
        Initialize Visualizer.
        
        Args:
            plots_dir: Directory to save plots
            figsize_default: Default figure size
        """
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.figsize_default = figsize_default
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def _save_plot(self, filename: str, dpi: int = 150):
        """Save current plot to file."""
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✓ Saved: {filepath}")
        return str(filepath)
    
    def plot_target_distributions(self, target_train: pd.DataFrame) -> List[str]:
        """
        Plot distribution analysis for both target variables.
        
        Args:
            target_train: DataFrame with target01 and target02 columns
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        # Simple histograms for both targets
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(target_train['target01'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_title('target01 Distribution')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(target_train['target02'], bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_title('target02 Distribution')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        saved_files.append(self._save_plot("01_target_distributions.png"))
        
        # Detailed analysis for target01
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(target_train['target01'], bins=50, edgecolor='black', color='steelblue')
        axes[0].set_title('target01 - Histogram')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        
        axes[1].boxplot(target_train['target01'])
        axes[1].set_title('target01 - Box Plot')
        axes[1].set_ylabel('Value')
        
        probplot(target_train['target01'], dist="norm", plot=axes[2])
        axes[2].set_title('target01 - Q-Q Plot')
        
        plt.tight_layout()
        saved_files.append(self._save_plot("02_target01_detailed.png"))
        
        # Detailed analysis for target02
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(target_train['target02'], bins=50, edgecolor='black', color='coral')
        axes[0].set_title('target02 - Histogram')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        
        axes[1].boxplot(target_train['target02'])
        axes[1].set_title('target02 - Box Plot')
        axes[1].set_ylabel('Value')
        
        probplot(target_train['target02'], dist="norm", plot=axes[2])
        axes[2].set_title('target02 - Q-Q Plot')
        
        plt.tight_layout()
        saved_files.append(self._save_plot("03_target02_detailed.png"))
        
        return saved_files
    
    def plot_variance_analysis(self, dataset_train: pd.DataFrame, 
                               low_variance_threshold: float = 0.01) -> List[str]:
        """
        Plot variance analysis visualizations.
        
        Args:
            dataset_train: Training features DataFrame
            low_variance_threshold: Threshold for low variance
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        feature_variance = dataset_train.var()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of variance
        axes[0].hist(feature_variance, bins=50, edgecolor='black', color='mediumpurple', alpha=0.7)
        axes[0].set_title('Distribution of Feature Variance')
        axes[0].set_xlabel('Variance')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(low_variance_threshold, color='red', linestyle='--', 
                       label=f'Low threshold ({low_variance_threshold})')
        axes[0].legend()
        
        # Bar plot of lowest variance features
        lowest_variance = feature_variance.nsmallest(20)
        axes[1].barh(range(len(lowest_variance)), lowest_variance.values, color='mediumpurple', alpha=0.7)
        axes[1].set_yticks(range(len(lowest_variance)))
        axes[1].set_yticklabels(lowest_variance.index)
        axes[1].set_xlabel('Variance')
        axes[1].set_title('20 Features with Lowest Variance')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        saved_files.append(self._save_plot("04_variance_analysis.png"))
        
        return saved_files
    
    def plot_random_feature_distributions(self, dataset_train: pd.DataFrame, 
                                          n_features: int = 10, 
                                          random_seed: int = 42) -> List[str]:
        """
        Plot distributions of randomly selected features.
        
        Args:
            dataset_train: Training features DataFrame
            n_features: Number of random features to plot
            random_seed: Random seed for reproducibility
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        all_features = dataset_train.columns.tolist()
        random_features = random.sample(all_features, n_features)
        
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        axes = axes.flatten()
        
        for idx, feature in enumerate(random_features):
            axes[idx].hist(dataset_train[feature], bins=50, edgecolor='black', 
                          alpha=0.7, color='steelblue')
            axes[idx].set_title(f'{feature}')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Distribution of 10 Random Features', fontsize=14, y=1.00)
        plt.tight_layout()
        saved_files.append(self._save_plot("05_random_features_distribution.png"))
        
        return saved_files
    
    def plot_outlier_analysis(self, dataset_train: pd.DataFrame, 
                              outlier_results: Dict,
                              z_score_threshold: float = 3.0) -> List[str]:
        """
        Plot outlier detection visualizations.
        
        Args:
            dataset_train: Training features DataFrame
            outlier_results: Results from EDAAnalyzer.detect_outliers()
            z_score_threshold: Z-score threshold used
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        outlier_counts = pd.Series(outlier_results['outlier_counts_per_feature'])
        
        # Main outlier overview
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        top20_outliers = outlier_counts.head(20)
        axes[0].barh(range(len(top20_outliers)), top20_outliers.values, color='crimson', alpha=0.6)
        axes[0].set_yticks(range(len(top20_outliers)))
        axes[0].set_yticklabels(top20_outliers.index)
        axes[0].set_xlabel('Number of Outliers')
        axes[0].set_title(f'Top 20 Features with Most Outliers\n(Z-score > {z_score_threshold})')
        axes[0].invert_yaxis()
        
        axes[1].hist(outlier_counts[outlier_counts > 0], bins=50, edgecolor='black', 
                    color='crimson', alpha=0.6)
        axes[1].set_xlabel('Number of Outliers')
        axes[1].set_ylabel('Number of Features')
        axes[1].set_title('Distribution of Outlier Counts Across Features')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        saved_files.append(self._save_plot("06_outlier_overview.png"))
        
        # Box plots for top outlier features
        top_outlier_features = list(outlier_results['top_20_outlier_features'].keys())[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_outlier_features):
            axes[idx].boxplot(dataset_train[feature].dropna(), vert=True)
            count = outlier_results['top_20_outlier_features'][feature]
            pct = count / len(dataset_train) * 100
            axes[idx].set_title(f'{feature}\n({count} outliers, {pct:.1f}%)')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Box Plots: Top 6 Features with Most Outliers', fontsize=14, y=1.00)
        plt.tight_layout()
        saved_files.append(self._save_plot("07_outlier_boxplots.png"))
        
        # Z-score distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_outlier_features):
            feature_z_scores = np.abs(zscore(dataset_train[feature]))
            axes[idx].hist(feature_z_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx].axvline(x=z_score_threshold, color='red', linestyle='--', 
                             linewidth=2, label=f'Threshold ({z_score_threshold})')
            axes[idx].set_xlabel('|Z-Score|')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{feature}\nMax Z-score: {feature_z_scores.max():.2f}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Z-Score Distributions for Top Outlier Features', fontsize=14, y=1.00)
        plt.tight_layout()
        saved_files.append(self._save_plot("08_zscore_distributions.png"))
        
        return saved_files
    
    def plot_correlation_heatmap(self, dataset_train: pd.DataFrame, 
                                 sample_size: int = 30) -> List[str]:
        """
        Plot correlation heatmap for a sample of features.
        
        Args:
            dataset_train: Training features DataFrame
            sample_size: Number of features to include
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        sample_corr = dataset_train.iloc[:, :sample_size].corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(sample_corr, cmap='coolwarm', center=0, 
                   annot=False, fmt='.2f', square=True, 
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Heatmap (First {sample_size} Features)')
        plt.tight_layout()
        saved_files.append(self._save_plot("09_correlation_heatmap.png"))
        
        return saved_files
    
    def plot_feature_target_correlations(self, correlation_results: Dict) -> List[str]:
        """
        Plot feature-target correlation bar charts.
        
        Args:
            correlation_results: Results from EDAAnalyzer.analyze_correlations()
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # target01 correlations (use top_50 dict, but only plot top 20)
        top50_target01 = correlation_results['target01_correlations']['top_50']
        top20_target01 = dict(list(top50_target01.items())[:20])
        features_01 = list(top20_target01.keys())
        values_01 = list(top20_target01.values())
        colors_01 = ['red' if v < 0 else 'green' for v in values_01]
        
        axes[0].barh(range(len(features_01)), values_01, color=colors_01, alpha=0.6)
        axes[0].set_yticks(range(len(features_01)))
        axes[0].set_yticklabels(features_01)
        axes[0].set_xlabel('Correlation Coefficient')
        axes[0].set_title('Top 20 Features Correlated with target01')
        axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        axes[0].invert_yaxis()
        
        # target02 correlations (use top_50 dict, but only plot top 20)
        top50_target02 = correlation_results['target02_correlations']['top_50']
        top20_target02 = dict(list(top50_target02.items())[:20])
        features_02 = list(top20_target02.keys())
        values_02 = list(top20_target02.values())
        colors_02 = ['red' if v < 0 else 'green' for v in values_02]
        
        axes[1].barh(range(len(features_02)), values_02, color=colors_02, alpha=0.6)
        axes[1].set_yticks(range(len(features_02)))
        axes[1].set_yticklabels(features_02)
        axes[1].set_xlabel('Correlation Coefficient')
        axes[1].set_title('Top 20 Features Correlated with target02')
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        saved_files.append(self._save_plot("10_feature_target_correlations.png"))
        
        return saved_files
    
    def plot_scatter_top_features(self, dataset_train: pd.DataFrame, 
                                  target_train: pd.DataFrame,
                                  correlation_results: Dict) -> List[str]:
        """
        Plot scatter plots of top correlated features vs targets.
        
        Args:
            dataset_train: Training features DataFrame
            target_train: Training targets DataFrame
            correlation_results: Results from EDAAnalyzer.analyze_correlations()
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        # Scatter plots for target01
        top6_target01 = correlation_results['target01_correlations']['top_20_features'][:6]
        top50_t01 = correlation_results['target01_correlations']['top_50']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top6_target01):
            axes[idx].scatter(dataset_train[feature], target_train['target01'], 
                             alpha=0.3, s=10, color='blue')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('target01')
            corr_val = top50_t01.get(feature, 0)
            axes[idx].set_title(f'{feature} vs target01\n(corr: {corr_val:.4f})')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        saved_files.append(self._save_plot("11_scatter_top_features_target01.png"))
        
        # Scatter plots for target02
        top6_target02 = correlation_results['target02_correlations']['top_20_features'][:6]
        top50_t02 = correlation_results['target02_correlations']['top_50']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top6_target02):
            axes[idx].scatter(dataset_train[feature], target_train['target02'], 
                             alpha=0.3, s=10, color='red')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('target02')
            corr_val = top50_t02.get(feature, 0)
            axes[idx].set_title(f'{feature} vs target02\n(corr: {corr_val:.4f})')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        saved_files.append(self._save_plot("12_scatter_top_features_target02.png"))
        
        return saved_files
    
    def plot_top_features_heatmap(self, dataset_train: pd.DataFrame,
                                  target_train: pd.DataFrame,
                                  correlation_results: Dict) -> List[str]:
        """
        Plot heatmap of top features + targets.
        
        Args:
            dataset_train: Training features DataFrame
            target_train: Training targets DataFrame
            correlation_results: Results from EDAAnalyzer.analyze_correlations()
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        top15_01 = correlation_results['target01_correlations']['top_20_features'][:15]
        top15_02 = correlation_results['target02_correlations']['top_20_features'][:15]
        top_features_all = list(set(top15_01 + top15_02))
        
        data_combined = pd.concat([dataset_train, target_train], axis=1)
        corr_matrix = data_combined[top_features_all + ['target01', 'target02']].corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   annot=False, fmt='.2f', square=True, 
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap: Top Features + Targets')
        plt.tight_layout()
        saved_files.append(self._save_plot("13_top_features_targets_heatmap.png"))
        
        return saved_files
    
    def plot_feature_distributions_kde(self, dataset_train: pd.DataFrame,
                                       target_train: pd.DataFrame,
                                       correlation_results: Dict) -> List[str]:
        """
        Plot histogram + KDE for top correlated features.
        
        Args:
            dataset_train: Training features DataFrame
            target_train: Training targets DataFrame
            correlation_results: Results from EDAAnalyzer.analyze_correlations()
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        # Top 6 for target01
        top6_target01 = correlation_results['target01_correlations']['top_20_features'][:6]
        top50_t01 = correlation_results['target01_correlations']['top_50']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top6_target01):
            axes[idx].hist(dataset_train[feature], bins=50, density=True, 
                          alpha=0.6, color='steelblue', edgecolor='black', label='Histogram')
            dataset_train[feature].plot(kind='kde', ax=axes[idx], 
                                        color='red', linewidth=2, label='KDE')
            corr_val = top50_t01.get(feature, 0)
            axes[idx].set_title(f'{feature}\n(corr with target01: {corr_val:.3f})')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Density')
            axes[idx].legend(loc='upper right', fontsize=8)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Top 6 Features for target01: Distribution Analysis', fontsize=14, y=1.00)
        plt.tight_layout()
        saved_files.append(self._save_plot("14_feature_kde_target01.png"))
        
        # Top 6 for target02
        top6_target02 = correlation_results['target02_correlations']['top_20_features'][:6]
        top50_t02 = correlation_results['target02_correlations']['top_50']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top6_target02):
            axes[idx].hist(dataset_train[feature], bins=50, density=True, 
                          alpha=0.6, color='coral', edgecolor='black', label='Histogram')
            dataset_train[feature].plot(kind='kde', ax=axes[idx], 
                                        color='darkred', linewidth=2, label='KDE')
            corr_val = top50_t02.get(feature, 0)
            axes[idx].set_title(f'{feature}\n(corr with target02: {corr_val:.3f})')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Density')
            axes[idx].legend(loc='upper right', fontsize=8)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Top 6 Features for target02: Distribution Analysis', fontsize=14, y=1.00)
        plt.tight_layout()
        saved_files.append(self._save_plot("15_feature_kde_target02.png"))
        
        return saved_files
    
    def plot_feature_importance_comparison(self, correlation_results: Dict,
                                           importance_results: Dict) -> List[str]:
        """
        Plot comparison of correlation vs RandomForest importance.
        
        Args:
            correlation_results: Results from EDAAnalyzer.analyze_correlations()
            importance_results: Results from EDAAnalyzer.analyze_feature_importance()
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # target01 - Correlation
        top50_corr_01 = correlation_results['target01_correlations']['top_50']
        top10_corr_01 = dict(list(top50_corr_01.items())[:10])
        features_01 = list(top10_corr_01.keys())[:10]
        values_01 = [abs(top10_corr_01[f]) for f in features_01]
        
        axes[0, 0].barh(range(len(features_01)), values_01, color='steelblue', alpha=0.7)
        axes[0, 0].set_yticks(range(len(features_01)))
        axes[0, 0].set_yticklabels(features_01)
        axes[0, 0].set_xlabel('Absolute Correlation')
        axes[0, 0].set_title('target01: Top 10 by Correlation')
        axes[0, 0].invert_yaxis()
        
        # target01 - RandomForest
        rf_imp_01 = importance_results['target01_importance']['top_50']
        rf_features_01 = list(rf_imp_01.keys())[:10]
        rf_values_01 = [rf_imp_01[f] for f in rf_features_01]
        
        axes[0, 1].barh(range(len(rf_features_01)), rf_values_01, color='forestgreen', alpha=0.7)
        axes[0, 1].set_yticks(range(len(rf_features_01)))
        axes[0, 1].set_yticklabels(rf_features_01)
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('target01: Top 10 by RandomForest')
        axes[0, 1].invert_yaxis()
        
        # target02 - Correlation
        top50_corr_02 = correlation_results['target02_correlations']['top_50']
        top10_corr_02 = dict(list(top50_corr_02.items())[:10])
        features_02 = list(top10_corr_02.keys())[:10]
        values_02 = [abs(top10_corr_02[f]) for f in features_02]
        
        axes[1, 0].barh(range(len(features_02)), values_02, color='coral', alpha=0.7)
        axes[1, 0].set_yticks(range(len(features_02)))
        axes[1, 0].set_yticklabels(features_02)
        axes[1, 0].set_xlabel('Absolute Correlation')
        axes[1, 0].set_title('target02: Top 10 by Correlation')
        axes[1, 0].invert_yaxis()
        
        # target02 - RandomForest
        rf_imp_02 = importance_results['target02_importance']['top_50']
        rf_features_02 = list(rf_imp_02.keys())[:10]
        rf_values_02 = [rf_imp_02[f] for f in rf_features_02]
        
        axes[1, 1].barh(range(len(rf_features_02)), rf_values_02, color='darkred', alpha=0.7)
        axes[1, 1].set_yticks(range(len(rf_features_02)))
        axes[1, 1].set_yticklabels(rf_features_02)
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('target02: Top 10 by RandomForest')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        saved_files.append(self._save_plot("16_feature_importance_comparison.png"))
        
        return saved_files
    
    def generate_all_plots(self, dataset_train: pd.DataFrame, 
                          target_train: pd.DataFrame,
                          eda_results: Dict) -> List[str]:
        """
        Generate all EDA visualization plots.
        
        Args:
            dataset_train: Training features DataFrame
            target_train: Training targets DataFrame
            eda_results: Results dictionary from EDAAnalyzer
            
        Returns:
            List of all saved file paths
        """
        all_files = []
        
        print("\n" + "="*60)
        print("GENERATING ALL EDA VISUALIZATIONS")
        print("="*60)
        
        print("\n[1/10] Target distributions...")
        all_files.extend(self.plot_target_distributions(target_train))
        
        print("\n[2/10] Variance analysis...")
        all_files.extend(self.plot_variance_analysis(dataset_train))
        
        print("\n[3/10] Random feature distributions...")
        all_files.extend(self.plot_random_feature_distributions(dataset_train))
        
        print("\n[4/10] Outlier analysis...")
        if 'outliers' in eda_results:
            all_files.extend(self.plot_outlier_analysis(dataset_train, eda_results['outliers']))
        
        print("\n[5/10] Correlation heatmap...")
        all_files.extend(self.plot_correlation_heatmap(dataset_train))
        
        print("\n[6/10] Feature-target correlations...")
        if 'correlations' in eda_results:
            all_files.extend(self.plot_feature_target_correlations(eda_results['correlations']))
        
        print("\n[7/10] Scatter plots...")
        if 'correlations' in eda_results:
            all_files.extend(self.plot_scatter_top_features(dataset_train, target_train, 
                                                            eda_results['correlations']))
        
        print("\n[8/10] Top features heatmap...")
        if 'correlations' in eda_results:
            all_files.extend(self.plot_top_features_heatmap(dataset_train, target_train, 
                                                            eda_results['correlations']))
        
        print("\n[9/10] Feature distributions (KDE)...")
        if 'correlations' in eda_results:
            all_files.extend(self.plot_feature_distributions_kde(dataset_train, target_train, 
                                                                  eda_results['correlations']))
        
        print("\n[10/10] Feature importance comparison...")
        if 'correlations' in eda_results and 'feature_importance' in eda_results:
            all_files.extend(self.plot_feature_importance_comparison(
                eda_results['correlations'], eda_results['feature_importance']))
        
        print(f"\n✓ Generated {len(all_files)} plots in {self.plots_dir}")
        return all_files
