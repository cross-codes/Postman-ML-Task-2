import pandas as pd

df = pd.read_csv("../processed.csv")
df["bmi"] = round(df["weight"] / ((df["height"] / 100) ** 2), 2)
df["AST/ALT"] = round(df["SGOT_AST"] / df["SGOT_ALT"], 2)

columns_order = [
    "sex",
    "age",
    "height",
    "weight",
    "bmi",
    "waistline",
    "sight_left",
    "sight_right",
    "hear_left",
    "hear_right",
    "SBP",
    "DBP",
    "BLDS",
    "tot_chole",
    "HDL_chole",
    "LDL_chole",
    "triglyceride",
    "hemoglobin",
    "urine_protein",
    "serum_creatinine",
    "SGOT_AST",
    "SGOT_ALT",
    "AST/ALT",
    "gamma_GTP",
    "SMK_stat_type_cd",
    "DRK_YN",
]
df = df[columns_order]
df.to_csv("../processed.csv", index=False)
