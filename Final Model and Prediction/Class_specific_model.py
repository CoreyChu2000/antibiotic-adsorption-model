import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import Split as spt
import split2 as spt2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def evaluate_performance(model, x, y):
    """
    评估模型性能并返回R2分数、RMSE和MAE
    :param model: 模型对象
    :param x: 输入特征
    :param y: 真实标签
    :return: R2分数、RMSE和MAE
    """
    y_pred = model.predict(x)
    score = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return score, rmse, mae


features = ['pH', 'CEC', 'OC', 'Clay', 'Sand', 'I', 'Ratio','logKow', 'pKa1','Ce']  # ,'pKa','LogP'
    # ['pH(soil)','CEC', 'OC', 'clay', 'sand','pH(solution)','I','T', 'ratio', 'logKow','pKa1','pKa2','Ce']
output = 'logCs'
# Define the model


model = GradientBoostingRegressor(n_estimators=146,
                                          max_depth=6,
                                          learning_rate=0.16,
                                          min_samples_leaf=6,
                                          min_samples_split=11,
                                          max_features=10,
                                          max_leaf_nodes=29,
                                          subsample=0.63)
#model = GradientBoostingRegressor(n_estimators= 200,max_depth= 10,learning_rate= 0.2,min_samples_leaf= 10,min_samples_split= 20,max_features= 11,max_leaf_nodes= 50,subsample=0.5)
#model = XGBRegressor(n_estimators= 194,max_depth=3,learning_rate= 0.2,min_child_weight= 4,subsample=1,colsample_bytree= 0.9,gamma= 0)#TCs 210
#model = XGBRegressor(n_estimators= 171,max_depth=10,learning_rate= 0.2,min_child_weight= 10,subsample=0.5,colsample_bytree= 1.0,gamma= 0.0)#SAs 490
model = XGBRegressor(n_estimators= 184,max_depth=4,learning_rate= 0.2,min_child_weight= 1,subsample=0.5,colsample_bytree= 0.7,gamma= 0.0)#FQs 28
#model = XGBRegressor(n_estimators= 191,max_depth=6,learning_rate= 0.08196920749027042,min_child_weight= 2,subsample=0.8863919195588791,colsample_bytree= 0.8527442540575791,gamma= 0.0)#FQs 28
#model = XGBRegressor(n_estimators=185,max_depth= 6,learning_rate= 0.2,min_child_weight=4,subsample=1.0,colsample_bytree=1.0,gamma=0.0)

#model = RandomForestRegressor(n_estimators=125,max_depth=10, min_samples_split=7, min_samples_leaf=1, max_features=7)
#model = SVR(C=1.6196705997522656,epsilon=0.01,kernel='rbf')
# Define stratified cross-validation object based on column 'A'
kf = KFold(n_splits=10, shuffle=True, random_state=42)
# 读取Excel文件

# Split the dataset with the current random_state
train_data_split = spt2.split_excel("D:\ML\Combined\Combined_Models_logKow.xlsx",
                                       #'SDZ', 'SCP', 'SMT',
                                      #'OTC', 'TC', 'CTC',
                                       'ENR', 'CIP', 'NOR',
                                       28)
train_data_scaled = train_data_split[0]
test_data = train_data_split[4]
train_data_scaled.to_excel(r"D:\ML\Combined\Categorial Model VAL.xlsx")
# Split the data based on LogP categories

category_scores = {}

unique_category_values = test_data['Class'].unique()
for category_value in unique_category_values:
    category_data = test_data[test_data['Class'] == category_value]

    X_test_scaled = category_data[features]
    y_test_data = category_data[output]

    groups = train_data_scaled.groupby(['Type'])
    grouped_data = []
    for _, group in groups:
        grouped_data.append(group)
    # Define lists to store each fold's scores
    train_scores = []
    validation_scores = []
    category_validation_scores = {}
    train_rmse_scores = []
    validation_rmse_scores = []
    train_mae_scores = []
    validation_mae_scores = []
    models = []
    test_scores = []
    test_rmse_scores = []
    test_mae_scores = []
    for fold, (train_index, val_index) in enumerate(kf.split(grouped_data)):
        train_set = [grouped_data[i] for i in train_index]
        val_set = [grouped_data[i] for i in val_index]

        train_df = pd.concat(train_set)

        # Separating validation data based on 'Class'
        val_data_by_class = {}
        for data in val_set:
            val_class = data['Class'].iloc[0]  # Assuming 'Class' is the categorical column
            if val_class not in val_data_by_class:
                val_data_by_class[val_class] = []
            val_data_by_class[val_class].append(data)

        # Iterate through each category in validation data
        for val_class, val_data_list in val_data_by_class.items():
            val_df = pd.concat(val_data_list)
            X_train = train_df[features]
            y_train = train_df[output]
            X_val = val_df[features]
            y_val = val_df[output]

            reg_model = model.fit(X_train, y_train)
            models.append(reg_model)
            train_score, train_rmse, train_mae = evaluate_performance(reg_model, X_train, y_train)
            validation_score, validation_rmse, validation_mae = evaluate_performance(reg_model, X_val, y_val)


            if val_class not in category_validation_scores:
                category_validation_scores[val_class] = {
                    'validation_scores': [],
                    'validation_rmse_scores': [],
                    'validation_mae_scores': []
                }
            category_validation_scores[val_class]['validation_scores'].append(validation_score)
            category_validation_scores[val_class]['validation_rmse_scores'].append(validation_rmse)
            category_validation_scores[val_class]['validation_mae_scores'].append(validation_mae)

            train_scores.append(train_score)
            validation_scores.append(validation_score)
            train_rmse_scores.append(train_rmse)
            validation_rmse_scores.append(validation_rmse)
            train_mae_scores.append(train_mae)
            validation_mae_scores.append(validation_mae)

            print("Category:", val_class)
            print("Fold:", fold)
            print("Train R2 score:", train_score)
            print("Train RMSE score:", train_rmse)
            print("Train MAE score:", train_mae)
            print("Validation R2 score:", validation_score)
            print("Validation RMSE score:", validation_rmse)
            print("Validation MAE score:", validation_mae)
        # Evaluate test performance for this fold
        test_score, test_rmse,test_mae = evaluate_performance(reg_model, X_test_scaled, y_test_data)

        test_scores.append(test_score)
        test_rmse_scores.append(test_rmse)
        test_mae_scores.append(test_mae)
        print("LogP Value:", category_value)
        print("Fold:", fold)
        #print("Train R2 score:", train_score)
        #print("Train RMSE score:", train_rmse)
        #print("Train MAE score:", train_mae)
        #print("Validation R2 score:", validation_score)
        #print("Validation RMSE score:", validation_rmse)
        #print("Validation MAE score:",validation_mae)
        print("Test R2 score:", test_score)
        print("Test RMSE score:", test_rmse)
        print("Test MAE score:", test_mae)


    # Calculate the average test score for this random_state
    # Calculate the average test score for all LogP categories

    mean_train_score = sum(train_scores) / len(train_scores)
    #mean_val_score = sum(validation_scores) / len(validation_scores)
    mean_test_score = sum(test_scores) / len(test_scores)
    mean_train_rmse = sum(train_rmse_scores) / len(train_rmse_scores)
    #mean_val_rmse = sum(validation_rmse_scores) / len(validation_rmse_scores)
    mean_test_rmse = sum(test_rmse_scores) / len(test_rmse_scores)
    mean_train_mae = sum(train_mae_scores) / len(train_mae_scores)
    #mean_val_mae = sum(validation_mae_scores) / len(validation_mae_scores)
    mean_test_mae = sum(test_mae_scores) / len(test_mae_scores)



    category_scores[category_value] = {

        'mean_train_score': mean_train_score,
        #'mean_validation_score': mean_val_score,  # Add mean validation score
        'mean_test_score': mean_test_score,
        'mean_train_rmse': mean_train_rmse,
        'mean_test_rmse': mean_test_rmse,
        'mean_train_mae': mean_test_mae,
        'mean_test_mae': mean_test_mae
    }

for category_value, scores in category_scores.items():
    print(f"Category: {category_value}")
    print(f"Mean Train R2 Score: {scores['mean_train_score']}")
    print(f"Mean Validation R2 Score: {np.mean(category_validation_scores[category_value]['validation_scores'])}")
    print(f"Mean Test R2 Score: {scores['mean_test_score']}")
    print(f"Mean Train RMSE: {scores['mean_train_rmse']}")
    print(f"Mean Validation RMSE: {np.mean(category_validation_scores[category_value]['validation_rmse_scores'])}")
    print(f"Mean Test RMSE: {scores['mean_test_rmse']}")
    print(f"Mean Train MAE: {scores['mean_train_mae']}")
    print(f"Mean Validation MAE: {np.mean(category_validation_scores[category_value]['validation_mae_scores'])}")
    print(f"Mean Test MAE: {scores['mean_test_mae']}")
    print("-------------------------------------")
for category_value, scores in category_scores.items():
    print(category_value)
    print(scores['mean_train_score'])
    print(np.mean(category_validation_scores[category_value]['validation_scores']))
    print(scores['mean_test_score'])
    print(scores['mean_train_rmse'])
    print(np.mean(category_validation_scores[category_value]['validation_rmse_scores']))
    print(scores['mean_test_rmse'])
    print(scores['mean_train_mae'])
    print(np.mean(category_validation_scores[category_value]['validation_mae_scores']))
    print(scores['mean_test_mae'])
    print("-------------------------------------")