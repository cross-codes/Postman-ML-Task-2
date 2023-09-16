import pandas as pd

df = pd.read_csv("../processed.csv")

df["DRK_YN"] = df["DRK_YN"].replace({"Y": 1, "N": 0})
df["sex"] = df["sex"].replace({"Male": 1, "Female": 2})
df.to_csv("../processed.csv", index=False)
