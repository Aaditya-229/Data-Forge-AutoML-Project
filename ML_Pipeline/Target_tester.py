# TARGET TESTING AND MODEL SUITABILITY MODULE..!
import pandas as pd
import numpy as np

def target_tester(target):
    model = ""
    while model not in ["r", "c"]:
        model = input(
            "Enter the type of ML algorithm you want to perform "
            "(Regression / Classification) : (R / C): ").lower()
        if model not in ["r", "c"]:
            print("Please enter a valid input (R or C):")
            
    target_dtype = target.dtype
    unique_num = target.nunique()
    unique_ratio = unique_num / len(target)

    is_numeric = np.issubdtype(target_dtype, np.number)
    is_object = target_dtype == "object"

    if model == "r":
        if is_numeric and unique_num > 10 and unique_ratio > 0.1:
            print("DataForge will now proceed to process the data for ML processes")
        else:
            confirmation = ""
            while confirmation not in ["y", "n"]:
                confirmation = input(
                    "⚠️ Target may be better suited for Classification. FYI...(T/F) or ('1'/'0') is a classification problem."
                    "Do you want to switch? (Y/N): "
                ).lower()
                if confirmation not in ["y","n"]:
                    print("Enter a valid input either 'Y' or 'N'.")
            if confirmation == "y":
                model = "c"
                print("The model has been changed to classification")
            else:
                print("DataForge will now proceed to process the data for ML processes")

    if model == "c":
        if (is_object) or (is_numeric and unique_num <= 20 and unique_ratio < 0.05):
            print("DataForge will now proceed to process the data for ML processes")
        else:
            confirmation = ""
            while confirmation not in ["y", "n"]:
                confirmation = input(
                    "⚠️ Target may be better suited for Regression. "
                    "Do you want to switch? (Y/N): "
                ).lower()
                if confirmation not in ["y","n"]:
                    print("Enter a valid input either 'Y' or 'N'.")
            if confirmation == "y":
                model = "r"
                print("The model has been changed to regression")
            else:
                print("DataForge will now proceed to process the data for ML processes")
    from sklearn.preprocessing import LabelEncoder

    le = None
    if target.dtype == "object" or target.dtype == "bool":
        le = LabelEncoder()
        target = pd.Series(le.fit_transform(target), index=target.index)
        print(f"✅ Target column encoded. Classes found: {list(le.classes_)}")
    print("-" * 55)
    return model, le
