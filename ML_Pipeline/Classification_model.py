import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def cls_model_train(X_train, X_test, Y_train, Y_test, scaler):

    log_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    log_model.fit(X_train, Y_train)
    log_pred = log_model.predict(X_test)

    rf_model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, Y_train)
    rf_pred = rf_model.predict(X_test)

    lgbm_model = LGBMClassifier(n_estimators=500, random_state=42, class_weight="balanced", verbose=-1)
    lgbm_model.fit(X_train, Y_train)
    lgbm_pred = lgbm_model.predict(X_test)

    print("Dataforge has trained three models on the given dataset and here are the results..\n")

    print("=" * 55)
    print(f"  [1] Logistic Regression  |  Accuracy: {round(accuracy_score(Y_test, log_pred) * 100, 2)}%")
    print("=" * 55)
    print(classification_report(Y_test, log_pred))

    print("=" * 55)
    print(f"  [2] Random Forest        |  Accuracy: {round(accuracy_score(Y_test, rf_pred) * 100, 2)}%")
    print("=" * 55)
    print(classification_report(Y_test, rf_pred))

    print("=" * 55)
    print(f"  [3] LightGBM             |  Accuracy: {round(accuracy_score(Y_test, lgbm_pred) * 100, 2)}%")
    print("=" * 55)
    print(classification_report(Y_test, lgbm_pred))

    models = {
        "1": ("Logistic Regression", log_model),
        "2": ("Random Forest", rf_model),
        "3": ("LightGBM", lgbm_model)
    }

    choice = ""
    while choice not in models:
        choice = input("\nWhich model do you want to save and deploy? Enter 1, 2, or 3: ").strip()
        if choice not in models:
            print("Please enter a valid choice: 1, 2, or 3.")

    model_name, best_model = models[choice]
    print(f"\nDataforge will save {model_name} as your deployment model..")
    print("-" * 55)
    return best_model, model_name
