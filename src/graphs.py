import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("resultados_agrupamento.csv")
sns.set_theme(style="whitegrid")

# ============================================================
# 🔷 GRÁFICO 1 — TEMPO MÉDIO POR MÉTRICA x ALGORITMO (SEM ZOOM)
# ============================================================

pivot = df.groupby(["Algorithm","Metric"])["Time_mean"].mean().unstack()
data_long = pivot.reset_index().melt(id_vars="Algorithm", var_name="Metric", value_name="Tempo")

plt.figure(figsize=(12,6))
sns.barplot(data=data_long,
            x="Algorithm", y="Tempo", hue="Metric",
            palette="Set2", width=0.68, edgecolor="black", linewidth=1.2)

# Define uma escala levemente aumentada para destacar diferenças
min_val = data_long["Tempo"].min()
max_val = data_long["Tempo"].max()
plt.ylim(min_val * 0.98, max_val * 1.03)

# 🔹 Exibe os valores sobre as barras
for i, bar in enumerate(plt.gca().patches):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + (max_val*0.005),
             f"{height:.5f}s", ha='center', fontsize=9)

plt.title("Tempo Médio de Execução — Sem Zoom", fontsize=16, weight="bold")
plt.ylabel("Tempo (segundos)", fontsize=13)
plt.xlabel("Algoritmo", fontsize=13)
plt.grid(axis='y', alpha=0.30)
plt.legend(title="Métrica")
plt.tight_layout()

plt.savefig("tempo_sem_zoom.png", dpi=320)
plt.show()

# ============================================================
# 🔶 GRÁFICO 2 — RAIO MÉDIO x DELTA NO 2-APROX (REFINEMENT)
# ============================================================
# ref = df[df["Algorithm"]=="Refinement"]

# plt.figure(figsize=(12,6))
# sns.lineplot(data=ref, x="Delta_Ratio", y="Radius_mean", hue="Metric", 
#              marker="o", linewidth=2.2, markersize=9, palette="Set1")

# plt.title("Impacto do Delta no Raio Final — Algoritmo Refinement", fontsize=15, weight="bold")
# plt.xlabel("Delta (%)", fontsize=13)
# plt.ylabel("Raio Médio", fontsize=13)

# plt.grid(alpha=0.25)
# plt.legend(title="Métrica", fontsize=10)
# plt.tight_layout()
# plt.savefig("grafico_refinement_delta_melhorado.png", dpi=300)
# plt.show()
