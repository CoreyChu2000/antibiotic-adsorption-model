import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def split_excel(excel, sheet_name1, sheet_name2,sheet_name3,sheet_name4, sheet_name5,sheet_name6,sheet_name7, sheet_name8,sheet_name9, random_state):
    # 读取Excel文件
    xls = pd.ExcelFile(excel)

    # 读取 'OTC' sheet 和 'TC' sheet 数据
    data_1 = pd.read_excel(xls, sheet_name=sheet_name1)
    data_2 = pd.read_excel(xls, sheet_name=sheet_name2)
    data_3 = pd.read_excel(xls, sheet_name=sheet_name3)
    data_4 = pd.read_excel(xls, sheet_name=sheet_name4)
    data_5 = pd.read_excel(xls, sheet_name=sheet_name5)
    data_6 = pd.read_excel(xls, sheet_name=sheet_name6)
    data_7 = pd.read_excel(xls, sheet_name=sheet_name7)
    data_8 = pd.read_excel(xls, sheet_name=sheet_name8)
    data_9 = pd.read_excel(xls, sheet_name=sheet_name9)


    # 分层划分数据集
    def split_data(data):
        groups = data.groupby(['Type'])
        grouped_data = [group for _, group in groups]
        train_data, test_data = train_test_split(grouped_data, test_size=0.01, random_state=random_state)
        return pd.concat(train_data), pd.concat(test_data)

    train_data_1, test_data_1 = split_data(data_1)
    train_data_2, test_data_2 = split_data(data_2)
    train_data_3, test_data_3 = split_data(data_3)
    train_data_4, test_data_4 = split_data(data_4)
    train_data_5, test_data_5 = split_data(data_5)
    train_data_6, test_data_6 = split_data(data_6)
    train_data_7, test_data_7 = split_data(data_7)
    train_data_8, test_data_8 = split_data(data_8)
    train_data_9, test_data_9 = split_data(data_9)

    # 合并 'OTC' 和 'TC' 训练数据和测试数据
    merged_train_data = pd.concat([train_data_1, train_data_2, train_data_3,train_data_4, train_data_5, train_data_6, train_data_7, train_data_8, train_data_9], axis=0)
    merged_test_data = pd.concat([test_data_1, test_data_2, test_data_3,test_data_4, test_data_5, test_data_6, train_data_7, train_data_8, train_data_9], axis=0)

    # 定义输入特征和输出值的列名
    features = ['pH(soil)','CEC', 'OC', 'Clay', 'pH(solution)','I', 'Ratio', 'Silt','Sand','Log Kow','Ce','pKa1','pKa2','T']#['pH(soil)','CEC', 'OC', 'Clay', 'pH(solution)','I', 'Ratio', 'Silt','Sand','Log Kow','Ce','pKa1','pKa2','T']
    output = 'Lg-Cs'

    X_train_data = merged_train_data[features]
    y_train_data = merged_train_data[output]
    X_test_data = merged_test_data[features]
    y_test_data = merged_test_data[output]
    # Apply Min-Max scaling to training and test data
    scaler = StandardScaler()
    X_train_data_scaled = scaler.fit_transform(X_train_data)
    X_test_scaled = scaler.transform(X_test_data)
    #print(X_train_data_scaled.shape)
    #print(y_train_data.shape)
    # 创建DataFrame
    train_data_scaled = pd.DataFrame(data=np.concatenate((X_train_data_scaled, y_train_data.values.reshape(-1, 1)), axis=1), columns=features + [output])
    test_data_scaled = pd.DataFrame(data=np.concatenate((X_test_scaled, y_test_data.values.reshape(-1, 1)), axis=1), columns=features + [output])

    # Reset the index of merged_train_data
    merged_train_data.reset_index(drop=True, inplace=True)
    merged_test_data.reset_index(drop=True, inplace=True)

    # Combine 'Type' and 'A' columns with features and output in the train_data_scaled DataFrame
    train_data_scaled['Type'] = merged_train_data['Type']
    train_data_scaled['Class'] = merged_train_data['Class']
    test_data_scaled['Class'] = merged_test_data['Class']
    #test_data_scaled['A'] = merged_test_data['A']


    # 将DataFrame保存为Excel文件
    #train_data_scaled.to_excel(r'D:\ML\TC\train_data.xlsx', index=False)
    test_data_scaled.to_excel(r"D:\ML\Combined\Combined Model TEST.xlsx", index=False)
    #merged_test_data.to_excel(r"D:\ML\SA\SA - test.xlsx", index=False)
    return train_data_scaled, X_test_scaled, y_test_data, merged_test_data,test_data_scaled

