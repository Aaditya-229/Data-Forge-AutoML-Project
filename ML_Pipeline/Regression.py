import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def reg_model(input_val, target_col):
    input_col = pd.get_dummies(input_val,dtype = int)
    original_columns = input_val.columns.tolist()
    feature_names = input_col.columns.tolist()
    x_train, X_test, y_train, Y_test = train_test_split(input_col, target_col,test_size = 0.2, random_state =42)
    for col in x_train.columns:
        x_train[col] = x_train[col].fillna(x_train[col].mean())
    y_train = y_train.fillna(y_train.mean())
    L_reg = LinearRegression()
    R_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    G_reg = GradientBoostingRegressor(random_state=42)
    L_reg.fit(x_train,y_train)
    R_reg.fit(x_train,y_train)
    G_reg.fit(x_train,y_train)
    L_pred = L_reg.predict(X_test)
    R_pred = R_reg.predict(X_test)
    G_pred = G_reg.predict(X_test)
    score_L = round(r2_score(Y_test, L_pred),3)
    score_R = round(r2_score(Y_test, R_pred),3)
    score_G = round(r2_score(Y_test, G_pred),3)
    print(f"Linear regression's accuracy : {score_L}\nRandom forest regressor's accuracy: {score_R}\nGradient boost regressor's accuracy: {score_G}.")
    models = {
        "1": ("Linear Regression", L_reg),
        "2": ("Random Forest regressor", R_reg),
        "3": ("Gradient boosting regressor", G_reg)
    }

    choice = ""
    while choice not in models:
        choice = input("\nWhich model do you want to save and deploy? Enter 1, 2, or 3: ").strip()
        if choice not in models:
            print("Please enter a valid choice: 1, 2, or 3.\n")

    model_name, best_model  = models[choice]
    print(f"\nDataforge will save {model_name} as your deployment model..")
    print("-" * 55)
    return best_model, model_name, original_columns, feature_names
