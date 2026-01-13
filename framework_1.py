import argparse
import numpy as np
import pandas as pd
import operator 

def framework(pairs, arr):
    """
    Args:
       - pairs:  a list of (cond, calc) tuples. calc() must be an executable
       - arr: a numpy array with the features in order feat_1, feat_2, ...
    
    Executes the first calc() whose cond returns True.
    Returns None if no condition matches.
    """
    targets = []

    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
        
    return targets


def cond_eval(condition, arr):
    """evaluate a condition
        - condition: must be a tupe of (int, string, float). The second entry must be a string from the list below, describing the operator. Third entry of the tuple must be a float). If condition is None, it is always evaluated to true.
        - arr: array on which the condition is evaluated

    The python operator package is used. Second entry in condition must be one of those:
       ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    """
    ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition is None:
        return True
    
    op = ops[condition[1]]
    return op(arr[condition[0]], condition[2])


# TODO implement the missing parts of this function. You find an example below, main_example(args).
def main(args):
    # Feature indices
    IDX_121 = 121
    IDX_225 = 225
    IDX_259 = 259
    IDX_195 = 195
    
    # Condition 1: feat_121 < 0.2
    condition1 = (IDX_121, "<", 0.2)
    def calc1(arr):
        return 1.75 * arr[IDX_225] - 1.85 * arr[IDX_259] - 0.75 * arr[IDX_195]
    
    # Condition 2: feat_121 < 0.5 (implies >= 0.2 due to order)
    condition2 = (IDX_121, "<", 0.5)
    def calc2(arr):
        return -0.65 * arr[IDX_225] + 1.55 * arr[IDX_259] + 0.55 * arr[IDX_195]
    
    # Condition 3: feat_121 < 0.7 (implies >= 0.5 due to order)
    condition3 = (IDX_121, "<", 0.7)
    def calc3(arr):
        return 0.55 * arr[IDX_225] + 1.25 * arr[IDX_259] - 1.65 * arr[IDX_195]
    
    # Condition 4: else (feat_121 >= 0.7)
    condition4 = None
    def calc4(arr):
        return 0.75 * arr[IDX_225] - 0.55 * arr[IDX_259] + 1.55 * arr[IDX_195]
    
    pair_list = [(condition1, calc1), (condition2, calc2), (condition3, calc3), (condition4, calc4)]
    
    data_array = pd.read_csv(args.eval_file_path).values
    
    return framework(pair_list, data_array)

    
def main_example(args):

    # Example: 
    test_arr = np.ones((10,10))

    def calc1(arr):
        """square first array column"""
        return arr[0]**2

    def calc2(arr):
        """add columns 3 and 4"""
        return arr[2] + arr[3]

    condition1 = (0,">=", 0.5)
    condition2 = (8, "==", 0.0)

    predict_targets = framework([(condition1, calc1), (condition2, calc2)], test_arr)
    print (predict_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Task 2")
    parser.add_argument("--eval_file_path", required=True, help="Path to EVAL_<ID>.csv")
    args = parser.parse_args()

    target02 = main(args)