import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../processed.csv")
df_original = df.copy()
print(df.describe())


column_names = [
    "age",
    "height",
    "weight",
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

columns_to_be_removed = [
    "sight_left",
    "sight_right",
    "waistline",
    "SBP",
    "DBP",
    "BLDS",
    "tot_chole",
    "triglyceride",
    "serum_creatinine",
    "SGOT_AST",
    "SGOT_ALT",
]


def generate_box_plot(column_name):
    plt.figure(figsize=(10, 6))
    plt.boxplot(df[column_name], vert=False)
    plt.title("Box Plot for " + column_name)
    plt.xlabel(column_name)
    plt.savefig("./img/box_plot_" + column_name + ".png")
    return 0


def remove_outliers(dataframe, column_name, k=1.5):
    Q1 = dataframe[column_name].quantile(0.25)
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1
    LB = Q1 - k * IQR
    UB = Q3 + k * IQR
    dataframe[column] = dataframe[column].clip(lower=LB, upper=UB)
    return dataframe


for column in column_names:
    generate_box_plot(column)

for column in columns_to_be_removed:
    df = remove_outliers(df_original, column)

df.to_csv("../processed.csv", index=False)
print(df.describe())
