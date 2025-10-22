import pandas as pd
import numpy as np
# 导入 numpy 用于数学运算，不再需要 sklearn 的 cosine_similarity

# Table 1 data (only the consumption zones and month for clarity)
data = {
    'Month': [10, 11, 12, 1, 2, 3], # 10-Mar
    'Zone1': [79, 78, 78, 67, 68, 67],
    'Zone2': [108, 159, 104, 103, 100, 95],
    'Zone3': [164, 156, 154, 155, 156, 142],
    'Zone4': [124, 126, 109, 111, 110, 111],
    'Zone5': [124, 125, 129, 129, 130, 127]
}

df_q1 = pd.DataFrame(data)
df_q1.to_csv('electricity_consumption.csv', index=False)
print("electricity_consumption.csv created.")
print(df_q1)


# Extract the consumption data (excluding Month)
# .values 将其转换为 numpy 数组
consumption_data = df_q1[['Zone1', 'Zone2', 'Zone3', 'Zone4', 'Zone5']].values

similarity_results = []
months = df_q1['Month'].values

print("\n--- Cosine Similarity Detailed Calculation ---")

for i in range(len(consumption_data) - 1):
    # d1 和 d2 是 numpy 向量 (1-D array)
    d1 = consumption_data[i]
    d2 = consumption_data[i+1]

    # 1. 计算点积 (Dot Product): d1 • d2
    dot_product = np.dot(d1, d2)
    
    # 2. 计算范数 (Magnitude/L2 Norm): ||d1|| 和 ||d2||
    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)
    
    # 3. 计算范数乘积 (Norm Product)
    norm_product = norm_d1 * norm_d2
    
    # 4. 计算余弦相似度
    # 检查分母是否接近零，以防除零错误
    if norm_product == 0:
        similarity = 0.0
    else:
        similarity = dot_product / norm_product

    month1 = months[i]
    month2 = months[i+1]

    # 打印中间结果
    print(f"\nComparing Month {month1} vs {month2}:")
    print(f"  Vector d1 (Month {month1}): {d1}")
    print(f"  Vector d2 (Month {month2}): {d2}")
    print(f"  Dot Product (d1 • d2): {dot_product}")
    print(f"  Norm ||d1||: {norm_d1:.4f} (||d1||^2 = {np.dot(d1, d1)})")
    print(f"  Norm ||d2||: {norm_d2:.4f} (||d2||^2 = {np.dot(d2, d2)})")
    print(f"  Norm Product (||d1|| ||d2||): {norm_product:.4f}")
    print(f"  Cosine Similarity: {similarity:.6f}")

    similarity_results.append({
        'Months': f'{month1} vs {month2}',
        'Dot Product': dot_product,
        'Norm_d1': norm_d1,
        'Norm_d2': norm_d2,
        'Norm_Product': norm_product,
        'Similarity': similarity
    })

df_sim = pd.DataFrame(similarity_results)
max_sim_row = df_sim.loc[df_sim['Similarity'].idxmax()]

print("\n--- Summary: Cosine Similarity Calculations ---")
# 打印包含所有中间结果的表格
print(df_sim[['Months', 'Dot Product', 'Norm_d1', 'Norm_d2', 'Norm_Product', 'Similarity']].round(6))
print(f"\nMost Similar Consecutive Months: {max_sim_row['Months']} with Similarity = {max_sim_row['Similarity']:.6f}")

# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# # Table 1 data (only the consumption zones and month for clarity)
# data = {
#     'Month': [10, 11, 12, 1, 2, 3], # 10-Mar
#     'Zone1': [79, 78, 78, 67, 68, 67],
#     'Zone2': [108, 159, 104, 103, 100, 95],
#     'Zone3': [164, 156, 154, 155, 156, 142],
#     'Zone4': [124, 126, 109, 111, 110, 111],
#     'Zone5': [124, 125, 129, 129, 130, 127]
# }

# df_q1 = pd.DataFrame(data)
# df_q1.to_csv('electricity_consumption.csv', index=False)
# print("electricity_consumption.csv created.")
# print(df_q1)


# # Extract the consumption data (excluding Month)
# consumption_data = df_q1[['Zone1', 'Zone2', 'Zone3', 'Zone4', 'Zone5']].values

# similarity_results = []
# months = df_q1['Month'].values

# for i in range(len(consumption_data) - 1):
#     d1 = consumption_data[i].reshape(1, -1)
#     d2 = consumption_data[i+1].reshape(1, -1)

#     # Calculate cosine similarity
#     similarity = cosine_similarity(d1, d2)[0][0]

#     month1 = months[i]
#     month2 = months[i+1]

#     similarity_results.append({
#         'Months': f'{month1} vs {month2}',
#         'Similarity': similarity
#     })

# df_sim = pd.DataFrame(similarity_results)
# max_sim_row = df_sim.loc[df_sim['Similarity'].idxmax()]

# print("\nCosine Similarity Calculations:")
# print(df_sim)
# print(f"\nMost Similar Consecutive Months: {max_sim_row['Months']} with Similarity = {max_sim_row['Similarity']:.6f}")