import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import Split3 as spt3
import split2 as spt2
import Split as spt
import matplotlib.pyplot as plt
import shap
from matplotlib import rcParams
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

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

features = ['pH', 'CEC', 'OC', 'Clay', 'Sand', 'I', 'Ratio', 'logKow', 'pKa1',
            'Ce']  # ,'pKa','LogP'
# ['pH(soil)','CEC', 'OC', 'clay', 'sand','pH(solution)','I','T', 'ratio', 'logKow','pKa1','pKa2','Ce']
output = 'logCs'
# Define the model

model = GradientBoostingRegressor(n_estimators=158,
                                          max_depth=9,
                                          learning_rate=0.19,
                                          min_samples_leaf=7,
                                          min_samples_split=9,
                                          max_features=6,
                                          max_leaf_nodes=39,
                                          subsample=0.6)
#(n_estimators=120,max_depth=5,learning_rate=0.11,min_samples_leaf=3,min_samples_split=5,max_features=6,max_leaf_nodes=24,subsample=0.9)
model = XGBRegressor(n_estimators=185,max_depth= 6,learning_rate= 0.2,min_child_weight=4,subsample=1.0,colsample_bytree=1.0,gamma=0.0)
#model = RandomForestRegressor(n_estimators=125,max_depth=10, min_samples_split=7, min_samples_leaf=1, max_features=7)
#model = SVR(C=1.6,epsilon=0.01,kernel='rbf')
# Define stratified cross-validation object based on column 'A'
kf = KFold(n_splits=10, shuffle=True, random_state=42)
# 读取Excel文件

# Split the dataset with the current random_state
train_data_split = spt.split_excel("D:\ML\Combined\Combined_Models.xlsx",

                                       'OTC', 'TC', 'CTC',
'SDZ', 'SCP', 'SMT',
                                       'ENR', 'CIP', 'NOR', 35)
train_data_scaled = train_data_split[0]
test_data = train_data_split[4]
#train_data_scaled = pd.read_excel(r"D:\ML\SA\SA-Smogn1.xlsx")
# Split the data based on LogP categories
print(train_data_scaled.shape)
print(test_data.shape)
"""
merged_train_data = pd.read_excel("D:\ML\Combined\DATA\Combined_Model_Data.xlsx", sheet_name='Train')
merged_test_data = pd.read_excel("D:\ML\Combined\DATA\Combined_Model_Data.xlsx", sheet_name='Test')
# 定义输入特征和输出值的列名
features = ['pH(soil)','CEC', 'OC', 'clay', 'sand','pH(solution)','I','ratio', 'logKow','pKa1','Ce']#,'pKa','LogP'
#['pH(soil)','CEC', 'OC', 'clay', 'sand','pH(solution)','I','T', 'ratio', 'logKow','pKa1','pKa2','Ce']
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
test_data = test_data_scaled
"""
X_test_scaled = test_data[features]
y_test_data = test_data[output]
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
test_mae_scores = []
for fold, (train_index, val_index) in enumerate(kf.split(grouped_data)):
    train_set = [grouped_data[i] for i in train_index]
    val_set = [grouped_data[i] for i in val_index]
    train_df = pd.concat(train_set)
    val_df = pd.concat(val_set)
    X_train = train_df[features]
    y_train = train_df[output]
    X_val = val_df[features]
    y_val = val_df[output]
    reg_model = model.fit(X_train, y_train)
    models.append(reg_model)
    train_score, train_rmse = evaluate_performance(reg_model, X_train, y_train)
    validation_score, validation_rmse = evaluate_performance(reg_model, X_val, y_val)
    train_scores.append(train_score)
    validation_scores.append(validation_score)
    train_rmse_scores.append(train_rmse)
    validation_rmse_scores.append(validation_rmse)
    # Evaluate test performance for this fold
    test_score, test_rmse = evaluate_performance(reg_model, X_test_scaled, y_test_data)
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
# Calculate the average test score for all LogP categories
mean_val_score = sum(validation_scores) / len(validation_scores)
mean_test_score = sum(test_scores) / len(test_scores)
mean_test_rmse = sum(test_rmse_scores) / len(test_rmse_scores)
print('mean_validation_score', mean_val_score)
print('mean_test_score', mean_test_score)
print('mean_test_rmse', mean_test_rmse)

# Find the index of the model with the highest validation R2 score
best_model_index = validation_scores.index(min(validation_scores))

# Use the best model to make predictions
best_model = models[best_model_index]
y_pred = best_model.predict(X_test_scaled)

best_model_score, best_model_rmse = evaluate_performance(best_model, X_test_scaled, y_test_data)

# Print the performance of the best model
print("Best Model R2 score:", best_model_score)
print("Best Model RMSE:", best_model_rmse)
# Calculate the absolute errors (residuals)
errors = np.abs(y_test_data - y_pred)


# Create a figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot the histogram of the original data
ax1.hist(train_data_scaled['logCs'], bins=50, color='blue', alpha=0.7, label='Original')
ax1.set_title('Original Data Distribution')
ax1.set_xlabel('logCs')
ax1.set_ylabel('Frequency')
ax1.legend()

# Plot the error plot
ax2.scatter(y_test_data, errors, color='red', alpha=0.7)
ax2.set_title('Error Plot')
ax2.set_xlabel('True logCs')
ax2.set_ylabel('Absolute Error')
ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax2.grid(True)

# Combine the two subplots
plt.tight_layout()
plt.show()

# Create a SHAP explainer for the best model
explainer = shap.Explainer(best_model)

# Calculate SHAP values for the test data
shap_values = explainer.shap_values(X_train)

# Summarize the feature importance
shap.summary_plot(shap_values, X_train, feature_names=features, show=False)

# Show the SHAP summary plot
plt.title("SHAP Feature Importance")
plt.show()

# Get feature importances from the best GBDT model
best_model = models[best_model_index]

# Create a SHAP explainer for the best model
explainer = shap.Explainer(best_model)

# Calculate SHAP values for the test data
shap_values = explainer.shap_values(X_train)

# Calculate mean absolute Shapley values
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

# Create a DataFrame to store feature names and mean absolute Shapley values
shap_summary_df = pd.DataFrame({'Feature': features, 'MeanAbsoluteShapley': mean_abs_shap_values})

# Sort the features by mean absolute Shapley values in descending order
shap_summary_df = shap_summary_df.sort_values(by='MeanAbsoluteShapley', ascending=False)

# Print the feature importances and mean absolute Shapley values
print("Mean Absolute Shapley Values:")
print(shap_summary_df)

# Plot the mean absolute Shapley values
plt.figure(figsize=(10, 6))
plt.barh(shap_summary_df['Feature'], shap_summary_df['MeanAbsoluteShapley'], color='green')
plt.xlabel('Mean Absolute Shapley Value')
plt.title('Mean Absolute Shapley Values for Best Model')
plt.gca().invert_yaxis()  # Invert the y-axis to show the most important features at the top
plt.show()

# Get feature importances from the best GBDT model
gbdt_feature_importances = best_model.feature_importances_

# Calculate the total importance
total_importance = np.sum(gbdt_feature_importances)

# Create a DataFrame to store feature names, their importances, and percentages
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': gbdt_feature_importances})
feature_importance_df['Percentage'] = feature_importance_df['Importance'] / total_importance * 100

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances and percentages
print("Feature Importance:")
print(feature_importance_df)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='blue')
plt.xlabel('Feature Importance')
plt.title('GBDT Model Feature Importance')
plt.gca().invert_yaxis()  # Invert the y-axis to show the most important features at the top
plt.show()

# Find the index of the model with the highest validation R2 score
best_model_index = validation_scores.index(max(validation_scores))
""""
# Use the best model to make predictions
best_model = models[best_model_index]
y_pred = best_model.predict(X_test_scaled)
y_test_data = train_data_split[2]

df = pd.DataFrame(y_pred)
with pd.ExcelWriter(r"D:\ML\Combined\Combined_Model_outlier.xlsx", engine='xlsxwriter') as writer:
    df.to_excel(writer,sheet_name='Pre')
    y_test_data.to_excel(writer,sheet_name='True')
"""
