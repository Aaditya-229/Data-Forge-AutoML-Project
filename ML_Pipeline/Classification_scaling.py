import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def Cls_model(input_val, target_col):
    #class imbalance module
    
    cls_bal = target_col.value_counts().to_dict()
    original_columns = input_val.columns.tolist()
    final_df = pd.get_dummies(input_val, dtype = int)
    feature_names = final_df.columns.tolist() 
    print("\nAfter the convertion of categorical data into numeric, the dataset looks like :\n", final_df.head())
    #Splitting
    
    X_train,X_test,Y_train,Y_test = train_test_split(final_df, target_col, test_size = 0.2, random_state = 42)
    #filling NA
    
    for col in X_train.columns:
        X_train[col] = X_train[col].fillna(X_train[col].mean())
        X_test[col] = X_test[col].fillna(X_train[col].mean())
    Y_train = Y_train.fillna(Y_train.mode()[0])
    Y_test = Y_test.fillna(Y_train.mode()[0])
    #Oversampling

    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_test = le.transform(Y_test)
    
    populate = ""
    while populate not in [ "y","n" ]:
        populate = input(f"\nDo you want to use SMOTE to populate the dataset..?\n The class balance is {cls_bal} : (Y/N)").lower()
        if populate == "y":
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            X_train, Y_train = smote.fit_resample(X_train, Y_train)
            print("Dataforge has successfully applied oversampling using smote..")
        elif populate == "n":
            print("Let's move on then..")
        else:
            print("Enter a proper input, either 'Y' or 'N'.")
    print("Dataforge next moves on to Scaling..")
    #scaling
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)
    print("Dataforge has successfully scaled the data using standard scaler...")
    print("-" * 55)
    return X_train_scaled, X_test_scaled, Y_train, Y_test, scale, feature_names , original_columns
