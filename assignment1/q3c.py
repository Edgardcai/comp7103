import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 根据表2创建数据集
data = {
    'Vi': [5, 42, 43, 60, 60, 67, 67, 99, 115, 185, 203, 215, 271, 486],
    'Im': [7, 8, 2, 8, 12, 1, 6, 2, 16, 16, 0, 6, 13, 7],
    'Class': ['T', 'F', 'F', 'T', 'T', 'T', 'T', 'F', 'F', 'T', 'F', 'F', 'F', 'F']
}
df = pd.DataFrame(data)

# 实际的类别标签
y_actual = df['Class']

# 应用图1中的决策树规则：Vi > 10 * Im
# 如果 Vi > 10 * Im，预测为'F'，否则预测为'T'
y_predicted = np.where(df['Vi'] > 10 * df['Im'], 'F', 'T')

# 定义标签顺序和正类，以便scikit-learn正确计算
labels_order = ['T', 'F']
positive_class = 'T'

# 构建混淆矩阵
cm = confusion_matrix(y_actual, y_predicted, labels=labels_order)

print("--- 混淆矩阵 ---")
print(f"{'': <10} {'预测为: T'} {'预测为: F'}")
print(f"{'实际为: T': <10} {cm[0,0]:^12} {cm[0,1]:^12}")
print(f"{'实际为: F': <10} {cm[1,0]:^12} {cm[1,1]:^12}")
print("-" * 45)

# 计算精确率、召回率和F-measure
precision = precision_score(y_actual, y_predicted, labels=labels_order, pos_label=positive_class)
recall = recall_score(y_actual, y_predicted, labels=labels_order, pos_label=positive_class)
f_measure = f1_score(y_actual, y_predicted, labels=labels_order, pos_label=positive_class)

print(f"\n--- 评估指标 (关于类别 '{positive_class}') ---")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F-measure: {f_measure:.4f}")