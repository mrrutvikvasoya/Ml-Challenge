# Part 2: Rule Discovery for Target02 Prediction

## Task Overview

The goal is to discover simple if-else rules for predicting `target02` that can be implemented on an edge device **without ML libraries**. The rules must use only basic comparisons and numerical functions.

### Requirements
- No ML routines on edge device
- Simple conditions and calculations only
- Implement in the provided `framework.py` template
- Submit as `framework_<ID>.py`

---

## Dataset Information

| File | Description |
|------|-------------|
| `problem_1/dataset_1.csv` | 10,000 samples × 273 features (`feat_0` to `feat_272`) |
| `problem_1/target_1.csv` | Target values: `target02` |

**Target02 Statistics:**
- Range: [-2.35, 2.22]
- Mean: 0.40
- Std: 0.78

---

## Methodology

### Step 1: Data Split (80/20)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training: 8000 samples
# Test: 2000 samples
```

### Step 2: Feature Correlation Analysis
Calculated Pearson correlation between each feature and `target02` on training data.

**Top correlated features:**
| Feature | Correlation |
|---------|-------------|
| `feat_121` | +0.4221 |
| `feat_225` | +0.1712 |
| `feat_122` | +0.0869 |
| `feat_66` | +0.0619 |
| `feat_259` | +0.0500 |

### Step 3: Decision Tree Analysis
Trained a shallow Decision Tree (depth=3) to identify important split features.

**Feature Importance from Tree:**
| Feature | Importance |
|---------|------------|
| `feat_121` | 0.5188 |
| `feat_259` | 0.2756 |
| `feat_195` | 0.1138 |
| `feat_225` | 0.0917 |

**Key Insight:** The tree primarily splits on `feat_121` with thresholds at 0.2 and 0.7.

### Step 4: Region-Based Formula Discovery
Based on tree analysis, we identified 4 regions defined by `feat_121` thresholds. For each region, we fit a Linear Regression on `feat_225`, `feat_259`, `feat_195`.

**Discovery:** Each region achieved **R² = 1.0**, indicating an exact linear formula exists!

---

## Discovered Rules

### Summary Table

| Condition | Formula | R² |
|-----------|---------|-----|
| `feat_121 < 0.2` | `1.75*f225 - 1.85*f259 - 0.75*f195` | 1.0 |
| `0.2 ≤ feat_121 < 0.5` | `-0.65*f225 + 1.55*f259 + 0.55*f195` | 1.0 |
| `0.5 ≤ feat_121 < 0.7` | `0.55*f225 + 1.25*f259 - 1.65*f195` | 1.0 |
| `feat_121 ≥ 0.7` | `0.75*f225 - 0.55*f259 + 1.55*f195` | 1.0 |

### Features Used
- **Condition Feature:** `feat_121` (index 121)
- **Calculation Features:** `feat_225`, `feat_259`, `feat_195` (indices 225, 259, 195)

### Thresholds
- 0.2 (splits region 1 from 2)
- 0.5 (splits region 2 from 3)
- 0.7 (splits region 3 from 4)

---

## Validation Results

| Dataset | R² | RMSE |
|---------|-----|------|
| Training (8000) | 1.000000 | 0.000000 |
| Test (2000) | 1.000000 | 0.000000 |

**Perfect prediction achieved!**

---

## Framework Implementation

The rules are implemented in `framework_1.py`:

```python
def main(args):
    IDX_121 = 121
    IDX_225 = 225
    IDX_259 = 259
    IDX_195 = 195
    
    # Region 1: feat_121 < 0.2
    condition1 = (IDX_121, "<", 0.2)
    def calc1(arr):
        return 1.75 * arr[IDX_225] - 1.85 * arr[IDX_259] - 0.75 * arr[IDX_195]
    
    # Region 2: 0.2 <= feat_121 < 0.5
    condition2 = (IDX_121, "<", 0.5)
    def calc2(arr):
        return -0.65 * arr[IDX_225] + 1.55 * arr[IDX_259] + 0.55 * arr[IDX_195]
    
    # Region 3: 0.5 <= feat_121 < 0.7
    condition3 = (IDX_121, "<", 0.7)
    def calc3(arr):
        return 0.55 * arr[IDX_225] + 1.25 * arr[IDX_259] - 1.65 * arr[IDX_195]
    
    # Region 4: feat_121 >= 0.7 (catch-all)
    condition4 = None
    def calc4(arr):
        return 0.75 * arr[IDX_225] - 0.55 * arr[IDX_259] + 1.55 * arr[IDX_195]
    
    pair_list = [
        (condition1, calc1),
        (condition2, calc2),
        (condition3, calc3),
        (condition4, calc4),
    ]
    
    data_array = pd.read_csv(args.eval_file_path).values
    return framework(pair_list, data_array)
```

**Important:** Order matters! Conditions are evaluated sequentially.

---

## Files

| File | Description |
|------|-------------|
| `part2_rule_discovery.py` | Complete analysis script with visualizations |
| `framework_1.py` | Implementation for submission |
| `outputs/part2_*.png` | Generated visualizations |

### Generated Plots
1. `part2_correlation.png` - Feature correlation bar chart
2. `part2_decision_tree.png` - Decision tree visualization
3. `part2_feature_importance.png` - Feature importance
4. `part2_feat121_regions.png` - 4 regions visualization
5. `part2_coefficients.png` - Formula coefficients by region
6. `part2_pred_vs_actual.png` - Predicted vs actual

---

## How to Run

```bash
# Run analysis script
source venv/bin/activate
python part2_rule_discovery.py

# Test framework implementation
python framework_1.py --eval_file_path problem_1/dataset_1.csv
```

---

## Key Findings

1. **target02 is deterministic** - Perfect R²=1.0 achieved
2. **Only 4 features needed** - feat_121, feat_225, feat_259, feat_195
3. **Simple piecewise linear** - 4 linear formulas based on feat_121 ranges
4. **No intercept needed** - All formulas pass through origin