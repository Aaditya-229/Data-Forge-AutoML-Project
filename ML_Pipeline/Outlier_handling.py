#OUTLIER HANDLING..!
import pandas as pd

def handling_outliers(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    print("The Outliers in the dataset have been successfully dealt with by DataForge( if there was any..! )\n")
    print("\n" + "=" * 55)
    return df
