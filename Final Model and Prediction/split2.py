import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def split_excel(excel, sheet_name1, sheet_name2,sheet_name3, random_state):
    # 读取Excel文件
    xls = pd.ExcelFile(excel)

    # 读取 'OTC' sheet 和 'TC' sheet 数据
    data_otc = pd.read_excel(xls, sheet_name=sheet_name1)
    data_tc = pd.read_excel(xls, sheet_name=sheet_name2)
    data_ctc = pd.read_excel(xls, sheet_name=sheet_name3)


    # 分层划分数据集
    def split_data(data):
        groups = data.groupby(['Type'])
        grouped_data = [group for _, group in groups]
        train_data, test_data = train_test_split(grouped_data, test_size=0.15, random_state=random_state)
        return pd.concat(train_data), pd.concat(test_data)


    train_data_otc_df, test_data_otc_df = split_data(data_otc)
    train_data_tc_df, test_data_tc_df = split_data(data_tc)
    train_data_ctc_df, test_data_ctc_df = split_data(data_ctc)

    # 合并 'OTC' 和 'TC' 训练数据和测试数据
    merged_train_data = pd.concat([train_data_tc_df, train_data_otc_df, train_data_ctc_df], axis=0)
    merged_test_data = pd.concat([test_data_tc_df, test_data_otc_df, test_data_ctc_df], axis=0)

    # 定义输入特征和输出值的列名
    features = ['pH', 'CEC', 'OC', 'Clay', 'Sand', 'I', 'Ratio', 'logKow', 'pKa1',
                'Ce']  # ,'pKa','LogP'
    # ['pH(soil)','CEC', 'OC', 'clay', 'sand','pH(solution)','I','T', 'ratio', 'logKow','pKa1','pKa2','Ce']
    output = 'logCs'

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
    test_data_scaled.to_excel(r"D:\ML\Combined\Categorial Model TEST.xlsx", index=False)
    #merged_test_data.to_excel(r"D:\ML\SA\SA - test.xlsx", index=False)
    return train_data_scaled, X_test_scaled, y_test_data, merged_test_data,test_data_scaled
