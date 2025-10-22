import math
import pandas as pd

# 原始数据
data = pd.DataFrame({
    'Record': range(1, 15),
    'Vi': [5, 42, 43, 60, 60, 67, 67, 99, 115, 185, 203, 215, 271, 486],
    'Im': [7, 8, 2, 8, 12, 1, 6, 2, 16, 16, 0, 6, 13, 7],
    'Class': ['T','F','F','T','T','T','T','F','F','T','F','F','F','F']
})

# 离散化函数（题目给定规则）
def discretize_vi(v):
    if v < 64: return 'Short'
    elif v < 194: return 'Mid'
    else: return 'Long'

def discretize_im(v):
    if v < 7.5: return 'Low'
    else: return 'High'

data['Vi_range'] = data['Vi'].apply(discretize_vi)
data['Im_range'] = data['Im'].apply(discretize_im)

def entropy(subset):
    n = len(subset)
    if n == 0: return 0.0
    p_t = sum(subset['Class'] == 'T') / n
    p_f = 1 - p_t
    e = 0.0
    if p_t > 0:
        e -= p_t * math.log2(p_t)
    if p_f > 0:
        e -= p_f * math.log2(p_f)
    return e

# 根节点熵
H_root = entropy(data)

# 辅助：给定一个划分（两个集合的成员条件），计算加权后熵与增益
def gain_for_partition(cond_left):
    left = data[cond_left]
    right = data[~cond_left]
    H_left = entropy(left)
    H_right = entropy(right)
    H_split = (len(left)/len(data))*H_left + (len(right)/len(data))*H_right
    return H_root - H_split, H_left, H_right, len(left), len(right)

# 划分 A: {Short} vs {Mid+Long}
cond_A = data['Vi_range'] == 'Short'
gainA, HLA, HRA, nL, nR = gain_for_partition(cond_A)

# 划分 B: {Short+Mid} vs {Long}
cond_B = data['Vi_range'].isin(['Short','Mid'])
gainB, HLB, HRB, nLB, nRB = gain_for_partition(cond_B)

# 划分 C: {Short,Long} vs {Mid}  (非相邻)
cond_C = data['Vi_range'].isin(['Short','Long'])
gainC, HLC, HRC, nLC, nRC = gain_for_partition(cond_C)

print(f"H(root) = {H_root:.9f}")
print("Partition A: {Short} vs {Mid+Long} -> gain =", round(gainA,9), f" (left n={nL}, H_left={HLA:.6f}; right n={nR}, H_right={HRA:.6f})")
print("Partition B: {Short+Mid} vs {Long} -> gain =", round(gainB,9), f" (left n={nLB}, H_left={HLB:.6f}; right n={nRB}, H_right={HRB:.6f})")
print("Partition C: {Short,Long} vs {Mid} -> gain =", round(gainC,9), f" (left n={nLC}, H_left={HLC:.6f}; right n={nRC}, H_right={HRC:.6f})")

# 根据预剪枝阈值 0.1 选择
threshold = 0.1
best_gain = max(gainA, gainB, gainC)
if best_gain < threshold:
    print("Pre-pruning triggered at root: no split (majority class predicted).")
else:
    if gainB == best_gain:
        print("Select partition B: {Short+Mid} vs {Long}")
        # 显示子节点的多数类
        left = data[data['Vi_range'].isin(['Short','Mid'])]
        right = data[data['Vi_range']=='Long']
        print("Left node (Short+Mid): n=",len(left), " T=", sum(left['Class']=='T'), " F=", sum(left['Class']=='F'), " -> majority:", 'T' if sum(left['Class']=='T')>sum(left['Class']=='F') else 'F')
        print("Right node (Long): n=",len(right), " T=", sum(right['Class']=='T'), " F=", sum(right['Class']=='F'), " -> majority:", 'T' if sum(right['Class']=='T')>sum(right['Class']=='F') else 'F')
