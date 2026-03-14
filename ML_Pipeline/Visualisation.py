import matplotlib.pyplot as plt
import seaborn as sns

def visual_Module(data):
    print("A Histogram of the given dataset is given below..\n")
    data.hist(bins=10, figsize=(10, 10))
    plt.show()
    print("\n" + "=" * 55)
    
    print("A Heatmap of the given dataset is given below..\n")
    corr = data.select_dtypes(include=[int,float]).corr()
    plt.figure(figsize=(8,7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    
    plt.title("Correlation Heatmap")
    plt.show()
    print("\n" + "=" * 55)
