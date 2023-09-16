import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sb


def generate_histplot(column_name):
    df = pd.read_csv("../processed.csv")
    plot.figure(figsize=(12, 6))
    sb.histplot(
        data=df, x=column_name, palette="pastel", hue="DRK_YN", multiple="stack"
    )
    plot.savefig("./img/sb_" + column_name.replace("/", "") + ".png")
    return 0


columns = [
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
    "gamma_GTP",
    "SMK_stat_type_cd",
]

for column in columns:
    generate_histplot(column)
