import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 读取10个 Excel 文件的 'Pre' 列数据并计算平均值
pre_data = []
for i in range(10):
    file_path = f"D:/ML/CV/{i}.xlsx"
    df = pd.read_excel(file_path)
    pre_column = df['Pre']
    pre_data.append(pre_column)

# 计算 'Pre' 列每行数据的平均值
pre_avg = np.mean(np.array(pre_data), axis=0)

# 读取 'Test.xlsx' 文件的数据
test_df = pd.read_excel("D:/ML/CV/Test.xlsx")

# 合并 'Lg-Cs'、'Class' 和计算的 'Pre' 列平均值
merged_df = pd.concat([test_df[['logCs', 'Class']], pd.Series(pre_avg, name='Pre')], axis=1)
merged_df.to_excel(r"D:\ML\CV\TCs.xlsx")

merged_df = pd.read_excel("D:\ML\CV\TCs.xlsx")
# 根据 'Class' 列分组
grouped = merged_df.groupby('Class')

# 初始化结果存储字典
results = {}

# 对每个分类进行线性回归和评估指标计算
for group_name, group_data in grouped:
    X = group_data['logCs'].values.reshape(-1, 1)
    y = group_data['Pre'].values

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 预测
    y_pred = model.predict(X)

    # 计算评估指标
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # 将结果存储到字典中
    results[group_name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}

# 打印结果
for class_name, metrics in results.items():
    print(f"Class: {class_name}")
    print(f"R2: {metrics['R2']}, RMSE: {metrics['RMSE']}, MAE: {metrics['MAE']}")

for class_name, metrics in results.items():
    print(f"Class: {class_name}")
    print(metrics['R2'])
    print(metrics['RMSE'])
    print(metrics['MAE'])