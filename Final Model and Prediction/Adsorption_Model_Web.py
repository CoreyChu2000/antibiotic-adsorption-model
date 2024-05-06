from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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

    features = ['pH', 'CEC', 'OC', 'Clay', 'Sand', 'I', 'Ratio', 'logKow', 'pKa1', 'Ce']
    output = 'logCs'

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
def XGBoost_model_predict(input_data):
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

        model = XGBRegressor(**model_params[model_type])
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
kf = KFold(n_splits=10, shuffle=True, random_state=42)

if model_type == 'TCs':
    train_data_split = split_excel("D:\专利\软著\抗生素吸附软著\模型数据集.xlsx", 'OTC', 'TC', 'CTC', 210)
elif model_type == 'SAs':
    train_data_split = split_excel("D:\专利\软著\抗生素吸附软著\模型数据集.xlsx", 'SDZ', 'SCP', 'SMT', 490)
elif model_type == 'FQs':
    train_data_split = split_excel("D:\专利\软著\抗生素吸附软著\模型数据集.xlsx", 'ENR', 'CIP', 'NOR', 28)

train_data_scaled = train_data_split[0]
groups = train_data_scaled.groupby(['Type'])
grouped_data = [group for _, group in groups]

# Make a prediction
result = predict_antibiotic_adsorption(7.5, 25, 2, 35, 45, 0.2, 2.5, 2.0, 4.5, 0.02)
print("result:", result)

import streamlit as st



def predict_antibiotic_adsorption(pH, CEC, OC, Clay, Sand, I, Ratio, logKow, pKa1, Ce):
    input_features = [pH, CEC, OC, Clay, Sand, I, Ratio, logKow, pKa1, Ce]
    final_features = [np.array(input_features)]
    final_features_scaled = train_data_split[1].transform(final_features)
    predictions = XGBoost_model_predict(final_features_scaled)
    return predictions

result=predict_antibiotic_adsorption(7.5,25,2,35,45,0.2,2.5,2.0,4.5,0.02)
print("reslut", result)
def main():
    st.title("抗生素吸附模型")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">预测抗生素在不同土壤环境下的吸附</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
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
    result = ""
    if st.button("Predict"):
        result=predict_antibiotic_adsorption(pH, CEC, OC, Clay, Sand, I, Ratio, logKow, pKa1, Ce)
        st.success('预测结果土壤吸附量（logCs, mg/kg）：{}'.format(result))

if __name__=='__main__':
    main()

