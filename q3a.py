import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from scipy.stats import entropy

# Table 2 data preparation
data_q3 = {
    'Record': list(range(1, 15)),
    'Vi': [5, 42, 43, 60, 60, 67, 67, 99, 115, 185, 203, 215, 271, 486],
    'Im': [7, 8, 2, 8, 12, 1, 6, 2, 16, 16, 0, 6, 13, 7],
    'Class': ['T', 'F', 'F', 'T', 'T', 'T', 'T', 'F', 'F', 'T', 'F', 'F', 'F', 'F']
}
df_q3 = pd.DataFrame(data_q3)
df_q3.to_csv('crowdfunding_data.csv', index=False)
print("crowdfunding_data.csv created.")

# Question 3 a) 代码
def gini(counts):
    total = sum(counts)
    if total == 0:
        return 0
    return 1 - sum([(c / total)**2 for c in counts])

def gini_split(D, D1, D2):
    return (len(D1) / len(D)) * gini(D1) + (len(D2) / len(D)) * gini(D2)

# 计算 Vi <= 83 的 GINI
D_full = df_q3['Class']
# Vi <= 83: R1 to R7 (5T, 2F)
D1_counts = [5, 2] # [T count, F count]
# Vi > 83: R8 to R14 (1T, 6F)
D2_counts = [1, 6] 
G_split_83 = gini_split(D_full, D1_counts, D2_counts)

print("\n--- Question 3 a) Decision Stump (GINI Index) ---")
print(f"Initial GINI(D): {gini([6, 8]):.4f}")
print(f"GINI Index for Vi <= 83: {G_split_83:.4f}")

# 结论
print("\nBest Decision Stump:")
print("Root Node Split: Vi <= 83")
print(f" - YES Branch (Vi <= 83): 5T, 2F -> Predict T")
print(f" - NO Branch (Vi > 83): 1T, 6F -> Predict F")