from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

# 读取测试数据
# 选择是否手动输入参数或从 Excel 文件中读取测试数据
use_manual_input = True  # 如果要手动输入参数，请将其设置为 True，如果要从 Excel 文件中读取测试数据，请将其设置为 False

# 如果使用手动输入参数，则手动输入测试数据
if use_manual_input:
    # 手动输入测试数据参数
    test_data_params = {
        'pH': [7.0, 7.5, 8.0],
        'CEC': [20.0, 25.0, 30.0],
        'OC': [1.5, 2.0, 2.5],
        'Clay': [30.0, 35.0, 40.0],
        'Sand': [40.0, 45.0, 50.0],
        'I': [0.1, 0.2, 0.3],
        'Ratio': [2.0, 2.5, 3.0],
        'logKow': [1.5, 2.0, 2.5],
        'pKa1': [4.0, 4.5, 5.0],
        'Ce': [0.01, 0.02, 0.03]
    }
    # 创建测试数据DataFrame
    test_data = pd.DataFrame(test_data_params)
else:
    # 从Excel文件中读取测试数据
    test_data = pd.read_excel("path_to_your_excel_file.xlsx")  # 请将"path_to_your_excel_file.xlsx"替换为你的Excel文件路径


# 定义函数用于划分数据集
def split_excel(excel, sheet_name1, sheet_name2, sheet_name3, random_state):
    # 读取Excel文件
    xls = pd.ExcelFile(excel)

    # 读取 'OTC'、'TC' 和 'CTC' sheet 中的数据
    data_otc = pd.read_excel(xls, sheet_name=sheet_name1)
    data_tc = pd.read_excel(xls, sheet_name=sheet_name2)
    data_ctc = pd.read_excel(xls, sheet_name=sheet_name3)

    # 分层划分数据集
    def split_data(data):
        groups = data.groupby(['Type'])
        grouped_data = [group for _, group in groups]
        train_data, test_data = train_test_split(grouped_data, test_size=0.15, random_state=random_state)
        return pd.concat(train_data), pd.concat(test_data)

    # 划分 'OTC'、'TC' 和 'CTC' 数据集
    train_data_otc_df, test_data_otc_df = split_data(data_otc)
    train_data_tc_df, test_data_tc_df = split_data(data_tc)
    train_data_ctc_df, test_data_ctc_df = split_data(data_ctc)

    # 合并 'OTC'、'TC' 和 'CTC' 训练数据和测试数据
    merged_train_data = pd.concat([train_data_tc_df, train_data_otc_df, train_data_ctc_df], axis=0)
    merged_test_data = pd.concat([test_data_tc_df, test_data_otc_df, test_data_ctc_df], axis=0)

    # 定义输入特征和输出值的列名
    features = ['pH', 'CEC', 'OC', 'Clay', 'Sand', 'I', 'Ratio', 'logKow', 'pKa1', 'Ce']
    output = 'logCs'

    X_train_data = merged_train_data[features]
    y_train_data = merged_train_data[output]
    X_test_data = test_data[features]

    # 将数据进行标准化处理
    scaler = StandardScaler()
    X_train_data_scaled = scaler.fit_transform(X_train_data)
    X_test_scaled = scaler.transform(X_test_data)

    # 创建DataFrame
    train_data_scaled = pd.DataFrame(
        data=np.concatenate((X_train_data_scaled, y_train_data.values.reshape(-1, 1)), axis=1),
        columns=features + [output])

    # 重置合并训练数据的索引
    merged_train_data.reset_index(drop=True, inplace=True)
    merged_test_data.reset_index(drop=True, inplace=True)
    # 将 'Type' 和 'Class' 列添加到训练数据中
    train_data_scaled['Type'] = merged_train_data['Type']
    train_data_scaled['Class'] = merged_train_data['Class']
    # 将DataFrame保存为Excel文件
    train_data_scaled.to_excel(r'D:\ML\TC\train_data.xlsx', index=False)

    return train_data_scaled, X_test_scaled


features = ['pH', 'CEC', 'OC', 'Clay', 'Sand', 'I', 'Ratio', 'logKow', 'pKa1', 'Ce']
output = 'logCs'

# 定义分层交叉验证对象
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 选择模型类型
model_type = 'TCs'  # 可根据需要更改为 'SAs' 或 'FQs'

# 定义模型参数
model_params = {
    'TCs': {'n_estimators': 194, 'max_depth': 3, 'learning_rate': 0.2, 'min_child_weight': 4, 'subsample': 1,
            'colsample_bytree': 0.9, 'gamma': 0},
    'SAs': {'n_estimators': 171, 'max_depth': 10, 'learning_rate': 0.2, 'min_child_weight': 10, 'subsample': 0.5,
            'colsample_bytree': 1.0, 'gamma': 0.0},
    'FQs': {'n_estimators': 184, 'max_depth': 4, 'learning_rate': 0.2, 'min_child_weight': 1, 'subsample': 0.5,
            'colsample_bytree': 0.7, 'gamma': 0.0}
}

# 根据选择的模型类型设置模型参数和数据集
if model_type == 'TCs':
    train_data_split = split_excel("D:\专利\软著\抗生素吸附软著\模型数据集.xlsx", 'OTC', 'TC', 'CTC', 210)
elif model_type == 'SAs':
    train_data_split = split_excel("D:\专利\软著\抗生素吸附软著\模型数据集.xlsx", 'SDZ', 'SCP', 'SMT', 490)
elif model_type == 'FQs':
    train_data_split = split_excel("D:\专利\软著\抗生素吸附软著\模型数据集.xlsx", 'ENR', 'CIP', 'NOR', 28)

train_data_scaled = train_data_split[0]
X_test_scaled = train_data_split[1]

groups = train_data_scaled.groupby(['Type'])
grouped_data = [group for _, group in groups]


models = []
predictions = []

for fold, (train_index, val_index) in enumerate(kf.split(grouped_data)):
    train_set = [grouped_data[i] for i in train_index]
    val_set = [grouped_data[i] for i in val_index]
    train_df = pd.concat(train_set)
    val_df = pd.concat(val_set)
    X_train = train_df[features]
    y_train = train_df[output]
    X_val = val_df[features]
    y_val = val_df[output]

    # 训练模型
    model = XGBRegressor(**model_params[model_type])
    reg_model = model.fit(X_train.values, y_train.values)
    models.append(reg_model)

    # 进行预测
    y_pre = reg_model.predict(X_test_scaled)
    predictions.append(y_pre)

# 计算预测值的平均值
pre_avg = np.mean(np.array(predictions), axis=0)
print(pre_avg)

