import pandas as pd
import seaborn as sns

# to load dataset from seaborn 
diamonds = sns.load_dataset("diamonds")

# create a pandas dataFrame
dataF = pd.DataFrame(diamonds)

# save the DataFrame directly as a Parquet file
dataF.to_csv("diamonds2.csv", index=False)

