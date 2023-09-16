import pandas as pd

df = pd.read_csv("../original.csv")
print(df[df.duplicated].shape)
print(df.describe())

df = df.drop_duplicates(keep="first")
df.to_csv("../processed.csv", index=False)
print(df.describe())
