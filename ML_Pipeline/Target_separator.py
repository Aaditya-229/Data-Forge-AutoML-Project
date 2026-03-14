#INPUT TARGET SEPERATION...!
import pandas as pd

def input_output_separator(df):
    target = None
    while target not in df.columns.to_list():
        target = input ("Now for the next part, Correctly type the output column's name :\n").strip("\"")
        if target not in df.columns.to_list(): 
            print("The input you have given doesnt seem to match any attributes in the dataset, kindly enter it again..!\n")
        
    input_data = df.columns.to_list()
    input_data.remove(target)
    input_data = df[input_data]
    target = df[target]
    print("DataForge has successfully seperated the dataset into input and target data..!\n")
    print("-"* 50)
    return input_data, target
