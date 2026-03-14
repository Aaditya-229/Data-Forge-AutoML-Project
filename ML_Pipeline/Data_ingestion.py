# DATA IMPORT MODULE..!
import pandas as pd
import numpy as np

def data_ingestion(): 
    data = None
    print("WELCOME TO DATAFORGE..!!!\n")
    while data is None:
        try:
            file_path = input ("Give the path to the dataset that you want to be processed (enclose it within quotes):\n").strip(" \" ")
            
            if file_path.endswith(".csv") is True:
                data = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx") is True:
                data = pd.read_excel(file_path)
            else:
                print("The file needs to be in .csv or .xlsx format\n")
                continue
                
        except FileNotFoundError:
            print("❌ File not found. Please check the path.\n")
        except PermissionError:
            print("❌ Permission denied while accessing the file.\n")
        except AttributeError:
            print("Kindly Enter the correct file path again..!\n")
        else:
            print(f"\n\nThe Dataset has been successfully uploaded, here's a short preview..!\n\n\n{data.head()}")
            print("\n" + "=" * 55)
    return data
