import pandas as pd

df = pd.read_csv("output.dat", delim_whitespace=True)
print(df.head())