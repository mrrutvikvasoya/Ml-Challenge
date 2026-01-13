"""
Data Loader Module
Handles loading and basic validation of ML challenge datasets.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Load and manage ML challenge datasets."""
    
    def __init__(self, data_dir: str = "problem_1"):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir: Path to directory containing the CSV files
        """
        self.data_dir = Path(data_dir)
        self.dataset_train: Optional[pd.DataFrame] = None
        self.target_train: Optional[pd.DataFrame] = None
        self.dataset_eval: Optional[pd.DataFrame] = None
        
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets: training features, targets, and evaluation set.
        
        Returns:
            Tuple of (dataset_train, target_train, dataset_eval)
        """
        self.dataset_train = pd.read_csv(self.data_dir / "dataset_1.csv")
        self.target_train = pd.read_csv(self.data_dir / "target_1.csv")
        self.dataset_eval = pd.read_csv(self.data_dir / "EVAL_1.csv")
        
        print(f"✓ Loaded training features: {self.dataset_train.shape}")
        print(f"✓ Loaded training targets: {self.target_train.shape}")
        print(f"✓ Loaded evaluation set: {self.dataset_eval.shape}")
        
        return self.dataset_train, self.target_train, self.dataset_eval
    
    def get_data_info(self) -> dict:
        """
        Get basic information about loaded datasets.
        
        Returns:
            Dictionary containing data shapes, types, and basic stats
        """
        if self.dataset_train is None:
            raise ValueError("Data not loaded. Call load_all() first.")
            
        info = {
            "train_shape": self.dataset_train.shape,
            "target_shape": self.target_train.shape,
            "eval_shape": self.dataset_eval.shape,
            "train_dtypes": self.dataset_train.dtypes.value_counts().to_dict(),
            "target_columns": self.target_train.columns.tolist(),
            "num_features": len(self.dataset_train.columns),
        }
        return info
    
    def validate_data(self) -> dict:
        """
        Validate data quality: check for missing values, duplicates, infinites.
        
        Returns:
            Dictionary containing validation results
        """
        if self.dataset_train is None:
            raise ValueError("Data not loaded. Call load_all() first.")
            
        import numpy as np
        
        validation = {
            "train_missing_total": int(self.dataset_train.isnull().sum().sum()),
            "target_missing": self.target_train.isnull().sum().to_dict(),
            "eval_missing_total": int(self.dataset_eval.isnull().sum().sum()),
            "train_duplicates": int(self.dataset_train.duplicated().sum()),
            "target_duplicates": int(self.target_train.duplicated().sum()),
            "eval_duplicates": int(self.dataset_eval.duplicated().sum()),
            "train_infinite": int(np.isinf(self.dataset_train.select_dtypes(include=[np.number])).sum().sum()),
            "eval_infinite": int(np.isinf(self.dataset_eval.select_dtypes(include=[np.number])).sum().sum()),
        }
        return validation
