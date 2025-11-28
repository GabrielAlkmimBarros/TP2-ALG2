import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ================== CARREGA DADOS ==================
df = pd.read_csv("RESULTADOS_FULL.csv")   # altere caso o nome seja outro
sns.set_theme(style="whitegrid", font_scale=1.15)

# ================== AGREGA RESULTADOS ==================
summary = df.groupby(["Dataset","Algorithm","Metric","Delta_Ratio","k"]).agg(
    Time_mean      = ("Time_s","mean"),
    Radius_mean    = ("Max_Radius","mean"),
    Silhouette_mean= ("Silhouette","mean"),
    AdjRand_mean   = ("Adj_Rand","mean"),
).reset_index()


# ================== MELHOR CONFIGURAÇÃO K-CENTERS POR DATASET ==================
kc = summary[summary["Algorithm"]!="KMeans"]
best_kc = kc.loc[kc.groupby("Dataset")["Silhouette_mean"].idxmax()]
best_kc = best_kc.set_index("Dataset")[["Silhouette_mean","AdjRand_mean","Time_mean"]]
best_kc.columns = ["KC_Silhouette","KC_ARI","KC_Tempo"]


# ================== MÉDIA DO K-MEANS ==================
km = summary[summary["Algorithm"]=="KMeans"].groupby("Dataset").agg(
    KM_Silhouette=("Silhouette_mean","mean"),
    KM_ARI       =("AdjRand_mean","mean"),
    KM_Tempo     =("Time_mean","mean")
)


# junta em uma tabela comparativa
base = best_kc.join(km)


# =====================================================================
# 🔷 1) K-MEANS x MELHOR K-CENTERS — SILHUETA
# =====================================================================
base[["KC_Silhouette","KM_Silhouette"]].plot(kind="bar", figsize=(14,5), rot=40)
plt.title("K-Means x Melhor K-Centers — Silhueta")
plt.ylabel("Silhueta média")
plt.legend(["Melhor K-Centers","K-Means"])
plt.tight_layout()
plt.savefig("A_compare_silhouette.png",dpi=300)
plt.show()


# =====================================================================
# 🔷 2) K-MEANS x MELHOR K-CENTERS — RAND AJUSTADO
# =====================================================================
base[["KC_ARI","KM_ARI"]].plot(kind="bar", figsize=(14,5), rot=40,
                              color=["#1f77b4","#ff7f0e"])
plt.title("K-Means x Melhor K-Centers — Rand Ajustado (ARI)")
plt.ylabel("ARI médio")
plt.tight_layout()
plt.savefig("B_compare_ARI.png",dpi=300)
plt.show()


# =====================================================================
# 🔷 3) K-MEANS x MELHOR K-CENTERS — TEMPO
# =====================================================================
base[["KC_Tempo","KM_Tempo"]].plot(kind="bar", figsize=(14,5), rot=40,
                                  color=["#2ca02c","#d62728"])
plt.title("Tempo — K-Means vs Melhor K-Centers")
plt.ylabel("Tempo médio (s)")
plt.tight_layout()
plt.savefig("C_compare_tempo.png",dpi=300)
plt.show()


# =====================================================================
# 🔶 4) Comparação interna apenas entre K-Centers
# =====================================================================
kc_only = summary[summary["Algorithm"]!="KMeans"]

plt.figure(figsize=(11,5))
sns.barplot(kc_only, x="Algorithm", hue="Metric", y="Silhouette_mean")
plt.title("K-Centers — Silhueta por Métrica")
plt.ylabel("Silhueta média")
plt.tight_layout()
plt.savefig("D_kcenters_silhouette.png",dpi=300)
plt.show()


plt.figure(figsize=(11,5))
sns.barplot(kc_only, x="Algorithm", hue="Metric", y="AdjRand_mean")
plt.title("K-Centers — ARI por Métrica")
plt.tight_layout()
plt.savefig("E_kcenters_ARI.png",dpi=300)
plt.show()
