import pandas as pd
import matplotlib.pyplot as plot

df = pd.read_csv("../processed.csv")
plot.hist(df[df["DRK_YN"] == 0]["hemoglobin"], alpha=0.5, label="Non-Drinker")
plot.hist(df[df["DRK_YN"] == 1]["hemoglobin"], alpha=0.5, label="Drinker")
plot.xlabel("Hemoglobin")
plot.ylabel("Frequency")
plot.title("Histogram of Hemoglobin by Drinking Status")
plot.legend()
plot.savefig("./img/hemo_bar_graph.png")

plot.clf()
plot.hist(df[df["DRK_YN"] == 0]["gamma_GTP"], alpha=0.5, label="Non-Drinker")
plot.hist(df[df["DRK_YN"] == 1]["gamma_GTP"], alpha=0.5, label="Drinker")
plot.xlabel("gamma_GTP")
plot.ylabel("Frequency")
plot.title("Histogram of gamma_GTP by Drinking status")
plot.legend()
plot.savefig("./img/gamma_bar_graph.png")

plot.clf()
