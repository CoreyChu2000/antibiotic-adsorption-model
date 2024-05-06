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


features = ['CEC', 'OC', 'Clay', 'Sand', 'pH', 'I', 'Ratio', 'Ce']  # ,'pKa','LogP'
# ['pH(soil)','CEC', 'OC', 'clay', 'sand','pH(solution)','I','T', 'ratio', 'logKow','pKa1','pKa2','Ce']
output = 'logCs'

# Define the model
model = GradientBoostingRegressor(n_estimators=158,
                                          max_depth=9,
                                          learning_rate=0.19160346145805401,
                                          min_samples_leaf=7,
                                          min_samples_split=9,
                                          max_features=6,
                                          max_leaf_nodes=39,
                                          subsample=0.621205101216639,random_state=42)
#(n_estimators=120,max_depth=5,learning_rate=0.11,min_samples_leaf=3,min_samples_split=5,max_features=6,max_leaf_nodes=24,subsample=0.9)
#model = RandomForestRegressor(n_estimators=75,max_depth= 10,min_samples_split= 9,min_samples_leaf= 3,max_features= 9)
model = XGBRegressor(n_estimators= 185,max_depth=3,learning_rate= 0.2,min_child_weight= 6,subsample=1.0,colsample_bytree= 1.0,gamma= 0.0)
# Define stratified cross-validation object based on column 'A'
kf = KFold(n_splits=10, shuffle=True, random_state=42)
# 读取Excel文件
# Split the dataset with the current random_state
train_data_split = spt.split_excel("D:\ML\Combined\Combined_Models.xlsx",'NOR', random_state=28)#"D:\ML\Combined\SMOGN\Combined_Model_TC_Type.xlsx",'Sheet1',
# "D:\ML\Combined\Combined_Model.xlsx", sheet_name='TC'
train_data_scaled = train_data_split[0]
#train_data_scaled = pd.read_excel(r"D:\ML\SA\SA-Smogn1.xlsx")
X_test_scaled = train_data_split[1]
y_test_data = train_data_split[2]
print(train_data_scaled.shape)
print(X_test_scaled.shape)
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
best_model_index = 2
best_model = models[best_model_index]
y_pred = best_model.predict(X_test_scaled)

df = pd.DataFrame(y_pred)
with pd.ExcelWriter(r"D:\ML\Combined\Combined_Model_outlier.xlsx", engine='xlsxwriter') as writer:
    df.to_excel(writer,sheet_name='Pre')
    y_test_data.to_excel(writer,sheet_name='True')
# Calculate R2 score and RMSE for the best model's predictions
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
ax1.set_xlabel('Lg-Cs')
ax1.set_ylabel('Frequency')
ax1.legend()

# Plot the error plot
ax2.scatter(y_test_data, errors, color='red', alpha=0.7)
ax2.set_title('Error Plot')
ax2.set_xlabel('True Lg-Cs')
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
gbdt_feature_importances = best_model.feature_importances_

# Create a DataFrame to store feature names and their importances
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': gbdt_feature_importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='blue')
plt.xlabel('Feature Importance')
plt.title('GBDT Model Feature Importance')
plt.gca().invert_yaxis()  # Invert the y-axis to show the most important features at the top
plt.show()