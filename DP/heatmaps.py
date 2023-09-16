import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plot

df = pd.read_csv("../processed.csv")

columns_to_include = [
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

df_subset = df[columns_to_include]

plot.figure(figsize=(12, 10))

heatmap = sb.heatmap(
    df_subset.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={"size": 8},
)

heatmap.set_xticklabels(
    heatmap.get_xticklabels(),
    rotation=45,
    horizontalalignment="right",
)

heatmap.set_aspect("equal")
plot.savefig("./img/heatmap.png", bbox_inches="tight")
