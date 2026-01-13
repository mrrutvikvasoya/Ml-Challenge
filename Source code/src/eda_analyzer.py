"""
EDA Analyzer Module
Comprehensive exploratory data analysis for ML challenge.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class EDAAnalyzer:
    """Perform comprehensive EDA on ML challenge datasets."""
    
    def __init__(self, dataset_train: pd.DataFrame, target_train: pd.DataFrame, 
                 dataset_eval: pd.DataFrame, results_dir: str = "outputs/results"):
        """
        Initialize EDA Analyzer.
        
        Args:
            dataset_train: Training features DataFrame
            target_train: Training targets DataFrame
            dataset_eval: Evaluation features DataFrame
            results_dir: Directory to save analysis results
        """
        self.dataset_train = dataset_train
        self.target_train = target_train
        self.dataset_eval = dataset_eval
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Store analysis results
        self.results: Dict = {}
        
    def analyze_variance(self, low_variance_threshold: float = 0.01) -> Dict:
        """
        Analyze feature variance to identify constant or low-variance features.
        
        Args:
            low_variance_threshold: Threshold below which variance is considered low
            
        Returns:
            Dictionary with variance analysis results
        """
        feature_variance = self.dataset_train.var()
        
        zero_variance_features = feature_variance[feature_variance == 0].index.tolist()
        low_variance_features = feature_variance[
            (feature_variance > 0) & (feature_variance < low_variance_threshold)
        ].index.tolist()
        
        variance_results = {
            "zero_variance_features": zero_variance_features,
            "zero_variance_count": len(zero_variance_features),
            "low_variance_features": low_variance_features,
            "low_variance_count": len(low_variance_features),
            "low_variance_threshold": low_variance_threshold,
            "variance_stats": {
                "min": float(feature_variance.min()),
                "max": float(feature_variance.max()),
                "mean": float(feature_variance.mean()),
                "median": float(feature_variance.median()),
            },
            "lowest_20_variance": feature_variance.nsmallest(20).to_dict(),
        }
        
        self.results["variance"] = variance_results
        return variance_results
    
    def detect_outliers(self, z_score_threshold: float = 3.0) -> Dict:
        """
        Detect outliers using Z-score method.
        
        Args:
            z_score_threshold: Z-score threshold for outlier detection
            
        Returns:
            Dictionary with outlier detection results
        """
        z_scores = np.abs(stats.zscore(self.dataset_train))
        
        # Count outliers per feature
        outlier_counts = (z_scores > z_score_threshold).sum(axis=0)
        outlier_counts = pd.Series(outlier_counts, index=self.dataset_train.columns)
        outlier_counts_sorted = outlier_counts.sort_values(ascending=False)
        
        # Rows with at least one outlier
        rows_with_outliers = (z_scores > z_score_threshold).any(axis=1).sum()
        
        # Total outlier data points
        total_outliers = int((z_scores > z_score_threshold).sum())
        total_data_points = self.dataset_train.shape[0] * self.dataset_train.shape[1]
        
        outlier_results = {
            "z_score_threshold": z_score_threshold,
            "total_features": len(self.dataset_train.columns),
            "features_with_outliers": int((outlier_counts > 0).sum()),
            "features_without_outliers": int((outlier_counts == 0).sum()),
            "total_outlier_points": total_outliers,
            "total_data_points": total_data_points,
            "outlier_percentage": round(total_outliers / total_data_points * 100, 4),
            "rows_with_outliers": int(rows_with_outliers),
            "rows_with_outliers_percentage": round(rows_with_outliers / len(self.dataset_train) * 100, 2),
            "clean_rows": len(self.dataset_train) - int(rows_with_outliers),
            "outlier_counts_per_feature": outlier_counts_sorted.to_dict(),
            "top_20_outlier_features": outlier_counts_sorted.head(20).to_dict(),
        }
        
        self.results["outliers"] = outlier_results
        return outlier_results
    
    def analyze_correlations(self, high_correlation_threshold: float = 0.4) -> Dict:
        """
        Analyze feature correlations and feature-target relationships.
        
        Args:
            high_correlation_threshold: Threshold for highly correlated feature pairs
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Feature-feature correlation matrix
        feature_corr = self.dataset_train.corr()
        
        # Find highly correlated pairs
        highly_correlated_pairs = []
        for i in range(len(feature_corr.columns)):
            for j in range(i + 1, len(feature_corr.columns)):
                corr_value = feature_corr.iloc[i, j]
                if abs(corr_value) > high_correlation_threshold:
                    highly_correlated_pairs.append({
                        "feature1": feature_corr.columns[i],
                        "feature2": feature_corr.columns[j],
                        "correlation": round(float(corr_value), 4)
                    })
        
        # Sort by absolute correlation
        highly_correlated_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        # Feature-target correlations
        data_combined = pd.concat([self.dataset_train, self.target_train], axis=1)
        target_correlations = data_combined.corr()[["target01", "target02"]].drop(["target01", "target02"])
        
        # Sort by absolute correlation for each target
        target01_sorted = target_correlations["target01"].abs().sort_values(ascending=False)
        target02_sorted = target_correlations["target02"].abs().sort_values(ascending=False)
        
        correlation_results = {
            "high_correlation_threshold": high_correlation_threshold,
            "highly_correlated_pairs_count": len(highly_correlated_pairs),
            "highly_correlated_pairs": highly_correlated_pairs[:50],  # Top 50 pairs
            "target01_correlations": {
                "top_50": {k: round(float(target_correlations.loc[k, "target01"]), 4) 
                          for k in target01_sorted.head(50).index},
                "top_50_features": target01_sorted.head(50).index.tolist(),
                "top_20_features": target01_sorted.head(20).index.tolist(),
            },
            "target02_correlations": {
                "top_50": {k: round(float(target_correlations.loc[k, "target02"]), 4) 
                          for k in target02_sorted.head(50).index},
                "top_50_features": target02_sorted.head(50).index.tolist(),
                "top_20_features": target02_sorted.head(20).index.tolist(),
            },
        }
        
        # Store full correlations for later use
        self._target_correlations = target_correlations
        self._target01_sorted = target01_sorted
        self._target02_sorted = target02_sorted
        
        self.results["correlations"] = correlation_results
        return correlation_results
    
    def analyze_feature_importance(self, n_estimators: int = 100, max_depth: int = 10,
                                   random_state: int = 42) -> Dict:
        """
        Analyze feature importance using RandomForest for both targets.
        
        Args:
            n_estimators: Number of trees in RandomForest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with feature importance results
        """
        print("Training RandomForest for target01...")
        rf1 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                    random_state=random_state, n_jobs=-1)
        rf1.fit(self.dataset_train, self.target_train["target01"])
        rf_importance_01 = pd.Series(
            rf1.feature_importances_, index=self.dataset_train.columns
        ).sort_values(ascending=False)
        
        print("Training RandomForest for target02...")
        rf2 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                    random_state=random_state, n_jobs=-1)
        rf2.fit(self.dataset_train, self.target_train["target02"])
        rf_importance_02 = pd.Series(
            rf2.feature_importances_, index=self.dataset_train.columns
        ).sort_values(ascending=False)
        
        importance_results = {
            "model_params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
            },
            "target01_importance": {
                "top_50": rf_importance_01.head(50).to_dict(),
                "top_50_features": rf_importance_01.head(50).index.tolist(),
                "top_15_features": rf_importance_01.head(15).index.tolist(),
            },
            "target02_importance": {
                "top_50": rf_importance_02.head(50).to_dict(),
                "top_50_features": rf_importance_02.head(50).index.tolist(),
                "top_15_features": rf_importance_02.head(15).index.tolist(),
            },
        }
        
        # Store for consensus calculation
        self._rf_importance_01 = rf_importance_01
        self._rf_importance_02 = rf_importance_02
        
        self.results["feature_importance"] = importance_results
        return importance_results
    
    def get_consensus_features(self, top_n: int = 50) -> Dict:
        """
        Get consensus (intersection) and union features from correlation and RF importance.
        
        Args:
            top_n: Number of top features to consider from each method
            
        Returns:
            Dictionary with consensus and union features for both targets
        """
        if not hasattr(self, "_target_correlations") or not hasattr(self, "_rf_importance_01"):
            raise ValueError("Run analyze_correlations() and analyze_feature_importance() first.")
        
        # For target01
        corr_top_01 = set(self._target01_sorted.head(top_n).index)
        rf_top_01 = set(self._rf_importance_01.head(top_n).index)
        consensus_01 = list(corr_top_01 & rf_top_01)  # Intersection
        union_01 = list(corr_top_01 | rf_top_01)       # Union
        
        # For target02
        corr_top_02 = set(self._target02_sorted.head(top_n).index)
        rf_top_02 = set(self._rf_importance_02.head(top_n).index)
        consensus_02 = list(corr_top_02 & rf_top_02)  # Intersection
        union_02 = list(corr_top_02 | rf_top_02)       # Union
        
        consensus_results = {
            "top_n_considered": top_n,
            "target01_consensus": {
                "features": consensus_01,
                "count": len(consensus_01),
            },
            "target01_union": {
                "features": union_01,
                "count": len(union_01),
            },
            "target02_consensus": {
                "features": consensus_02,
                "count": len(consensus_02),
            },
            "target02_union": {
                "features": union_02,
                "count": len(union_02),
            },
        }
        
        self.results["consensus_features"] = consensus_results
        return consensus_results
    
    def get_important_features_summary(self) -> Dict:
        """
        Get a comprehensive summary of important features for model training.
        
        Returns:
            Dictionary with all important features organized by method
        """
        summary = {
            "target01": {
                "consensus": self.results.get("consensus_features", {}).get("target01_consensus", {}).get("features", []),
                "top_15_correlation": self.results.get("correlations", {}).get("target01_correlations", {}).get("top_20_features", [])[:15],
                "top_15_rf": self.results.get("feature_importance", {}).get("target01_importance", {}).get("top_15_features", []),
            },
            "target02": {
                "consensus": self.results.get("consensus_features", {}).get("target02_consensus", {}).get("features", []),
                "top_15_correlation": self.results.get("correlations", {}).get("target02_correlations", {}).get("top_20_features", [])[:15],
                "top_15_rf": self.results.get("feature_importance", {}).get("target02_importance", {}).get("top_15_features", []),
            },
        }
        
        self.results["important_features_summary"] = summary
        return summary
    
    def run_full_analysis(self) -> Dict:
        """
        Run all EDA analyses in sequence.
        
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "="*60)
        print("RUNNING FULL EDA ANALYSIS")
        print("="*60)
        
        print("\n[1/5] Analyzing variance...")
        self.analyze_variance()
        
        print("[2/5] Detecting outliers...")
        self.detect_outliers()
        
        print("[3/5] Analyzing correlations...")
        self.analyze_correlations()
        
        print("[4/5] Analyzing feature importance (this may take a minute)...")
        self.analyze_feature_importance()
        
        print("[5/5] Computing consensus features...")
        self.get_consensus_features()
        self.get_important_features_summary()
        
        print("\n✓ Full analysis complete!")
        return self.results
    
    def save_results(self, filename: str = "eda_results.json") -> str:
        """
        Save all analysis results to a JSON file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        output_path = self.results_dir / filename
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {output_path}")
        return str(output_path)
    
    def load_results(self, filename: str = "eda_results.json") -> Dict:
        """
        Load analysis results from a JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dictionary with loaded results
        """
        input_path = self.results_dir / filename
        
        with open(input_path, "r") as f:
            self.results = json.load(f)
        
        print(f"✓ Results loaded from: {input_path}")
        return self.results
    
    def print_summary(self):
        """Print a formatted summary of all analysis results."""
        print("\n" + "="*70)
        print("EDA ANALYSIS SUMMARY")
        print("="*70)
        
        if "variance" in self.results:
            v = self.results["variance"]
            print(f"\n📊 VARIANCE ANALYSIS:")
            print(f"   Zero variance features: {v['zero_variance_count']}")
            print(f"   Low variance features: {v['low_variance_count']}")
        
        if "outliers" in self.results:
            o = self.results["outliers"]
            print(f"\n🔍 OUTLIER DETECTION (Z-score > {o['z_score_threshold']}):")
            print(f"   Features with outliers: {o['features_with_outliers']}")
            print(f"   Total outlier points: {o['total_outlier_points']:,} ({o['outlier_percentage']:.2f}%)")
            print(f"   Rows with outliers: {o['rows_with_outliers']:,} ({o['rows_with_outliers_percentage']:.1f}%)")
            print(f"   Top 5 outlier features:")
            for i, (feat, count) in enumerate(list(o['top_20_outlier_features'].items())[:5], 1):
                print(f"      {i}. {feat}: {count}")
        
        if "correlations" in self.results:
            c = self.results["correlations"]
            print(f"\n🔗 CORRELATION ANALYSIS:")
            print(f"   Highly correlated pairs (|r| > {c['high_correlation_threshold']}): {c['highly_correlated_pairs_count']}")
            print(f"   Top 5 features for target01:", list(c['target01_correlations']['top_20_features'][:5]))
            print(f"   Top 5 features for target02:", list(c['target02_correlations']['top_20_features'][:5]))
        
        if "feature_importance" in self.results:
            fi = self.results["feature_importance"]
            print(f"\n🌲 RANDOM FOREST IMPORTANCE:")
            print(f"   Top 5 for target01:", fi['target01_importance']['top_15_features'][:5])
            print(f"   Top 5 for target02:", fi['target02_importance']['top_15_features'][:5])
        
        if "consensus_features" in self.results:
            cf = self.results["consensus_features"]
            print(f"\n✅ CONSENSUS FEATURES (top {cf['top_n_considered']} from both methods):")
            print(f"   target01: {cf['target01_consensus']['count']} features")
            print(f"      {cf['target01_consensus']['features']}")
            print(f"   target02: {cf['target02_consensus']['count']} features")
            print(f"      {cf['target02_consensus']['features']}")
        
        print("\n" + "="*70)
