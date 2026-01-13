"""
ML Pipeline Configuration
=========================
Central configuration file containing all constants, paths, and hyperparameters.
Modify these values to customize the pipeline behavior.
"""

# Reproducibility
RANDOM_STATE = 42

# Feature Selection
TOP_N = 10  # Number of top features to select

# Model Evaluation
OVERFIT_THRESHOLD = 0.05  # Gap threshold for overfitting detection

# Hyperparameter Tuning
N_OPTUNA_TRIALS = 150  # Number of Optuna optimization trials

# Data Paths
DATA_PATH = 'problem_1/dataset_1.csv'
TARGET_PATH = 'problem_1/target_1.csv'
TARGET_COLUMN = 'target01'

# Output Paths
OUTPUT_DIR = 'outputs'
