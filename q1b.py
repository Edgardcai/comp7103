import pandas as pd
import numpy as np

# Table 1 数据：Zone4 (X) 和 Zone5 (Y) 的六个记录
data = {
    'Record': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
    'Zone4 (X)': [124, 126, 109, 111, 110, 111],
    'Zone5 (Y)': [124, 125, 129, 129, 130, 127]
}

df = pd.DataFrame(data)

# 提取 X 和 Y
X = df['Zone4 (X)']
Y = df['Zone5 (Y)']
N = len(X)

# --- 1. 计算均值 (Mean) ---
mean_X = X.mean()
mean_Y = Y.mean()

# --- 2. 计算中心化数据 (Centered Data) ---
# X_centered = (X_k - mean_X)
X_centered = X - mean_X
# Y_centered = (Y_k - mean_Y)
Y_centered = Y - mean_Y

# --- 3. 计算分子 (Numerator) ---
# 分子 = Sigma((X_k - mean_X) * (Y_k - mean_Y))
numerator = (X_centered * Y_centered).sum()

# --- 4. 计算分母的平方和项 (Squared Sums for Denominator) ---
# Sum_X_sq = Sigma((X_k - mean_X)^2)
sum_X_sq = (X_centered**2).sum()
# Sum_Y_sq = Sigma((Y_k - mean_Y)^2)
sum_Y_sq = (Y_centered**2).sum()

# --- 5. 计算分母 (Denominator) ---
# 分母 = sqrt(Sum_X_sq) * sqrt(Sum_Y_sq)
denominator = np.sqrt(sum_X_sq) * np.sqrt(sum_Y_sq)

# --- 6. 计算 Pearson's Correlation (r) ---
if denominator == 0:
    r = 0.0
else:
    r = numerator / denominator

# --- 7. 输出详细的中间结果和最终结果 ---
print("--- Pearson's Correlation (Zone4 vs Zone5) Detailed Calculation ---")
print(f"Number of Records (N): {N}")
print(f"Mean of Zone4 (X_bar): {mean_X:.4f}")
print(f"Mean of Zone5 (Y_bar): {mean_Y:.4f}")

print("\n--- Intermediate Results ---")
print(f"1. Numerator (Sum of Cross-Products): {numerator:.4f}")
print(f"2. Denominator Squared Sum for X (Sum(X_k - X_bar)^2): {sum_X_sq:.4f}")
print(f"3. Denominator Squared Sum for Y (Sum(Y_k - Y_bar)^2): {sum_Y_sq:.4f}")
print(f"4. Denominator (sqrt(Sum_X_sq) * sqrt(Sum_Y_sq)): {denominator:.4f}")

print("\n--- Final Result ---")
print(f"Pearson's Correlation Coefficient (r): {r:.6f}")