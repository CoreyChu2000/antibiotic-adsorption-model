import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import streamlit as st

# Define function to split dataset
features = ['pH', 'CEC', 'OC', 'Clay', 'Sand', 'I', 'Ratio', 'logKow', 'pKa1', 'Ce']
output = 'logCs'


def split_excel(excel, sheet_name1, sheet_name2, sheet_name3, random_state):
    xls = pd.ExcelFile(excel)
    data_otc = pd.read_excel(xls, sheet_name=sheet_name1)
    data_tc = pd.read_excel(xls, sheet_name=sheet_name2)
    data_ctc = pd.read_excel(xls, sheet_name=sheet_name3)

    def split_data(data):
        groups = data.groupby(['Type'])
        grouped_data = [group for _, group in groups]
        train_data, test_data = train_test_split(grouped_data, test_size=0.15, random_state=random_state)
        return pd.concat(train_data), pd.concat(test_data)

    train_data_otc_df, test_data_otc_df = split_data(data_otc)
    train_data_tc_df, test_data_tc_df = split_data(data_tc)
    train_data_ctc_df, test_data_ctc_df = split_data(data_ctc)

    merged_train_data = pd.concat([train_data_tc_df, train_data_otc_df, train_data_ctc_df], axis=0)
    merged_test_data = pd.concat([test_data_tc_df, test_data_otc_df, test_data_ctc_df], axis=0)

    X_train_data = merged_train_data[features]
    y_train_data = merged_train_data[output]

    scaler = StandardScaler()
    X_train_data_scaled = scaler.fit_transform(X_train_data)

    train_data_scaled = pd.DataFrame(
        data=np.concatenate((X_train_data_scaled, y_train_data.values.reshape(-1, 1)), axis=1),
        columns=features + [output])

    merged_train_data.reset_index(drop=True, inplace=True)
    merged_test_data.reset_index(drop=True, inplace=True)

    train_data_scaled['Type'] = merged_train_data['Type']
    train_data_scaled['Class'] = merged_train_data['Class']

    train_data_scaled.to_excel(r'D:\ML\TC\train_data.xlsx', index=False)

    return train_data_scaled,  scaler


# Define model prediction function
def XGBoost_model_predict(input_data, model_params):
    models = []
    predictions = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(grouped_data)):
        train_set = [grouped_data[i] for i in train_index]
        val_set = [grouped_data[i] for i in val_index]
        train_df = pd.concat(train_set)
        X_train = train_df[features]
        y_train = train_df[output]

        model = XGBRegressor(**model_params)
        reg_model = model.fit(X_train.values, y_train.values)
        models.append(reg_model)

        y_pre = reg_model.predict(input_data)
        predictions.append(y_pre)
    pre_avg = np.mean(np.array(predictions), axis=0)
    return pre_avg


# Define function to predict antibiotic adsorption
def predict_antibiotic_adsorption(pH, CEC, OC, Clay, Sand, I, Ratio, logKow, pKa1, Ce):
    input_features = [pH, CEC, OC, Clay, Sand, I, Ratio, logKow, pKa1, Ce]
    final_features = [np.array(input_features)]
    final_features_scaled = train_data_split[1].transform(final_features)
    predictions = XGBoost_model_predict(final_features_scaled)
    return predictions

# Set up model parameters and dataset based on the chosen model type
model_params = {
    'TCs': {'n_estimators': 194, 'max_depth': 3, 'learning_rate': 0.2, 'min_child_weight': 4, 'subsample': 1,
            'colsample_bytree': 0.9, 'gamma': 0},
    'SAs': {'n_estimators': 171, 'max_depth': 10, 'learning_rate': 0.2, 'min_child_weight': 10, 'subsample': 0.5,
            'colsample_bytree': 1.0, 'gamma': 0.0},
    'FQs': {'n_estimators': 184, 'max_depth': 4, 'learning_rate': 0.2, 'min_child_weight': 1, 'subsample': 0.5,
            'colsample_bytree': 0.7, 'gamma': 0.0}
}

model_type = 'TCs'

if model_type == 'TCs':
    train_data_split = split_excel("D:\专利\软著\抗生素吸附软著\模型数据集.xlsx", 'OTC', 'TC', 'CTC', 210)
elif model_type == 'SAs':
    train_data_split = split_excel("D:\专利\软著\抗生素吸附软著\模型数据集.xlsx", 'SDZ', 'SCP', 'SMT', 490)
elif model_type == 'FQs':
    train_data_split = split_excel("D:\专利\软著\抗生素吸附软著\模型数据集.xlsx", 'ENR', 'CIP', 'NOR', 28)

train_data_scaled = train_data_split[0]
groups = train_data_scaled.groupby(['Type'])
grouped_data = [group for _, group in groups]


def predict_antibiotic_adsorption(model_type, pH, CEC, OC, Clay, Sand, I, Ratio, logKow, pKa1, Ce):
    input_features = [pH, CEC, OC, Clay, Sand, I, Ratio, logKow, pKa1, Ce]
    final_features = [np.array(input_features)]
    final_features_scaled = train_data_split[1].transform(final_features)
    predictions = XGBoost_model_predict(final_features_scaled, model_params[model_type])
    return predictions

def predict_test_data(model_type, test_data):
    test_data_scaled = train_data_split[1].transform(test_data[features])
    predictions = XGBoost_model_predict(test_data_scaled, model_params[model_type])
    return predictions


def main():
    st.title("抗生素吸附模型")

    st.write("---")
    html_temp1 = """
        <div style="background-color:#001f3f;padding:5px;border-radius:5px">
        <h2 style="color:white;text-align:center;">选择模型</h2>
        </div>
        """
    st.markdown(html_temp1, unsafe_allow_html=True)
    model_type = st.selectbox('', ["四环素类 (TCs)", "磺胺类 (SAs)", "氟喹诺酮类 (FQs)"])
    if model_type == "四环素类 (TCs)":
        model_type = "TCs"
    elif model_type == "磺胺类 (SAs)":
        model_type = "SAs"
    elif model_type == "氟喹诺酮类 (FQs)":
        model_type = "FQs"

    st.write("---")

    # Manual Input Section
    st.write("手动输入预测")
    pH = st.text_input("pH值", "")
    CEC = st.text_input("阳离子交换量(CEC, cmol/kg)", "")
    OC = st.text_input("有机碳含量(OC, g/m3)", "")
    Clay = st.text_input("黏粒含量(Clay, %)", "")
    Sand = st.text_input("沙粒含量(Sand, %)", "")
    I = st.text_input("离子强度(I, mol/kg)", "")
    Ratio = st.text_input("土壤与溶液比值(Ratio, g/ml)", "")
    logKow = st.text_input("辛醇水分配系数对数(logKow)", "")
    pKa1 = st.text_input("酸解离常数(pKa1)", "")
    Ce = st.text_input("水中平衡浓度(Ce, mg/L)", "")

    if st.button("手动预测"):
        result = predict_antibiotic_adsorption(model_type, float(pH), float(CEC), float(OC), float(Clay), float(Sand),
                                               float(I), float(Ratio), float(logKow), float(pKa1), float(Ce))
        st.success('预测结果土壤吸附量（logCs, mg/kg）：{}'.format(result))

    st.write("---")

    # Batch Input Section
    st.write("批量输入预测")
    uploaded_file = st.file_uploader("上传测试数据文件", type=["xlsx"])
    if uploaded_file is not None:
        test_data = pd.read_excel(uploaded_file)
        if st.button("批量预测"):
            test_predictions = predict_test_data(model_type,test_data)
            test_data['logCs'] = test_predictions
            test_data['Cs'] = np.exp(test_predictions)
            st.write("输入测试数据和预测结果：")
            st.write(test_data)

    st.write("---")
    st.write("关于")
    st.write("系统内定的抗生素理化性质如下表：")
    st.write(pd.DataFrame({
        '抗生素': ['土霉素', '四环素', '金霉素', '磺胺甲嘧啶', '磺胺嘧啶', '磺胺氯哒嗪', '恩诺沙星', '环丙沙星',
                   '诺氟沙星'],
        'pKa1': [3.3, 3.2, 3.3, 2.07, 2.1, 1.87, 6.27, 6.09, 6.3],
        'logKow': [-0.9, -1.30, -0.62, 0.89, -0.09, 0.31, 1.1, 0.28, -1.03]
    }))



if __name__=='__main__':
    main()

