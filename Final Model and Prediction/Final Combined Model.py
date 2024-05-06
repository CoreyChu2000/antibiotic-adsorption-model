import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import Split as spt
import matplotlib.pyplot as plt
import shap
def evaluate_performance(model, x, y):
    """
    评估模型性能并返回R2分数和RMSE
    :param model: 模型对象
    :param x: 输入特征
    :param y: 真实标签
    :return: R2分数和RMSE
    """
    y_pred = model.predict(x)
    score = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return score, rmse

features = ['pH', 'CEC', 'OC', 'Clay', 'Sand', 'I', 'Ratio','logKow', 'pKa1',
                'Ce']  # ,'pKa','LogP'
    # ['pH(soil)','CEC', 'OC', 'clay', 'sand','pH(solution)','I','T', 'ratio', 'logKow','pKa1','pKa2','Ce']
output = 'logCs'

# Define the model

model = GradientBoostingRegressor(n_estimators=146,
                                  max_depth=9,
                                  learning_rate= 0.13204082495641492,
                                  min_samples_leaf= 4,
                                  min_samples_split= 2,
                                  max_features= 4,
                                  max_leaf_nodes= 30,
                                  subsample= 0.8941877492861185)

#(n_estimators=120,max_depth=5,learning_rate=0.11,min_samples_leaf=3,min_samples_split=5,max_features=6,max_leaf_nodes=24,subsample=0.9)
#model = SVR(C=1.6196705997522656,epsilon=0.01,kernel='rbf')
model = XGBRegressor(n_estimators=185,max_depth= 6,learning_rate= 0.2,min_child_weight=4,subsample=1.0,colsample_bytree=1.0,gamma=0.0)
# Define stratified cross-validation object based on column 'A'
kf = KFold(n_splits=10, shuffle=True, random_state=52)
# 读取Excel文件
# Split the dataset with the current random_state
train_data_split = spt.split_excel("D:\ML\Combined\Combined_Models.xlsx",
                                   'SDZ', 'SCP', 'SMT',
                                      'OTC', 'TC', 'CTC',
                                       'ENR', 'CIP', 'NOR',
                                   35)
train_data_scaled = train_data_split[0]
#train_data_scaled = pd.read_excel(r"D:\ML\SA\SA-Smogn1.xlsx")
X_test_scaled = train_data_split[1]
y_test_data = train_data_split[2]
test_data = train_data_split[4]
groups = train_data_scaled.groupby(['Type'])
grouped_data = []
for _, group in groups:
    grouped_data.append(group)
# Define lists to store each fold's scores
train_scores = []
validation_scores = []
train_rmse_scores = []
validation_rmse_scores = []
models = []
test_scores = []
test_rmse_scores = []
for fold, (train_index, val_index) in enumerate(kf.split(grouped_data)):
    train_set = [grouped_data[i] for i in train_index]
    val_set = [grouped_data[i] for i in val_index]
    train_df = pd.concat(train_set)
    val_df = pd.concat(val_set)
    X_train = train_df[features]
    y_train = train_df[output]
    X_val = val_df[features]
    y_val = val_df[output]
    reg_model = model.fit(X_train.values, y_train.values)
    models.append(reg_model)
    train_score, train_rmse = evaluate_performance(reg_model, X_train.values, y_train.values)
    validation_score, validation_rmse = evaluate_performance(reg_model, X_val.values, y_val.values)
    train_scores.append(train_score)
    validation_scores.append(validation_score)
    train_rmse_scores.append(train_rmse)
    validation_rmse_scores.append(validation_rmse)
    # Evaluate test performance for this fold
    test_score, test_rmse = evaluate_performance(reg_model, X_test_scaled, y_test_data)
    y_pre = reg_model.predict(X_test_scaled)
    df = pd.DataFrame(y_pre)
    df.to_excel(f"D:/ML/CV/{fold}.xlsx", engine='xlsxwriter',header=['Pre'], index=False)
    test_data.to_excel(r"D:/ML/CV/Test.xlsx", engine='xlsxwriter')

    test_scores.append(test_score)
    test_rmse_scores.append(test_rmse)
    print("Fold:", fold)
    print("Train R2 score:", train_score)
    print("Train RMSE score:", train_rmse)
    print("Validation R2 score:", validation_score)
    print("Validation RMSE score:", validation_rmse)
    print("Test R2 score:", test_score)
    print("Test RMSE score:", test_rmse)

# Calculate the average test score for this random_state
mean_val_score = sum(validation_scores) / len(validation_scores)
mean_test_score = sum(test_scores) / len(test_scores)
print('mean_validation_score', mean_val_score)
print('mean_test_score', mean_test_score)

# Find the index of the model with the highest validation R2 score
best_model_index = validation_scores.index(max(validation_scores))

# Use the best model to make predictions
best_model = models[best_model_index]
y_pred = best_model.predict(X_test_scaled)


# Calculate R2 score and RMSE for the best model's predictions
best_model_score, best_model_rmse = evaluate_performance(best_model, X_test_scaled, y_test_data)



# Print the performance of the best model
print("Best Model R2 score:", best_model_score)
print("Best Model RMSE:", best_model_rmse)
# Calculate the absolute errors (residuals)
errors = np.abs(y_test_data - y_pred)

