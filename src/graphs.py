import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid",font_scale=1.15)


def load_results(file):
    df = pd.read_csv(file)


    df["Algorithm"] = df["Algorithm"].replace({
        "KMeans_SKL":"KMeans", "kmeans":"KMeans",
        "MaxMin":"KCenters-MaxMin",
        "Refinement":"KCenters-Refinement"
    })

    df["Delta_Ratio"] = df["Delta_Ratio"].replace("N/A",np.nan).astype(float)

    return df



def gerar_tabelas(df):

    resumo_geral = df.groupby(["Algorithm","Metric"]).agg(
        Silhouette_mean=('Silhouette','mean'),
        Silhouette_std=('Silhouette','std'),
        AdjRand_mean=('Adj_Rand','mean'),
        AdjRand_std=('Adj_Rand','std'),
        Time_mean=('Time_s','mean'),
        Time_std=('Time_s','std'),
        Radius_mean=('Max_Radius','mean'),
        Radius_std=('Max_Radius','std')
    ).round(4)

    print("\n====================== TABELA GERAL ======================")
    print(resumo_geral.to_markdown())

    # === TABELA Δ SOMENTE PARA REFINEMENT ===
    ref = df[df["Algorithm"]=="KCenters-Refinement"]

    variacao_delta = ref.groupby(["Delta_Ratio","Metric"]).agg(
        Silhouette_mean=('Silhouette','mean'),
        Radius_mean=('Max_Radius','mean'),
        Time_mean=('Time_s','mean')
    ).round(4)

    print("\n================== VARIAÇÃO DE Δ (REFINEMENT) ==================")
    print(variacao_delta.to_markdown())

    # === MELHOR RESULTADO POR DATASET ===
    best = df.loc[df.groupby("Dataset")["Silhouette"].idxmax(),
                  ["Dataset","Algorithm","Metric","Silhouette"]].round(4)

    print("\n================ VENCEDORES POR DATASET ==================")
    print(best.to_markdown(index=False))




def gerar_graficos(df):


    plt.figure(figsize=(13,5))
    sns.barplot(df,x="Algorithm",y="Silhouette",hue="Metric")
    plt.title("Silhouette — Algoritmo × Métrica")
    plt.savefig("01_silhouette_alg_met.png",dpi=300); plt.close()

    plt.figure(figsize=(13,5))
    sns.barplot(df,x="Algorithm",y="Adj_Rand",hue="Metric")
    plt.title("ARI — Algoritmo × Métrica")
    plt.savefig("02_ARI_alg_met.png",dpi=300); plt.close()

    plt.figure(figsize=(13,5))
    sns.barplot(df,x="Algorithm",y="Time_s",hue="Metric")
    plt.title("Tempo — Algoritmo × Métrica")
    plt.ylabel("Segundos"); plt.savefig("03_tempo_alg_met.png",dpi=300); plt.close()


    df_ref = df[df["Algorithm"]=="KCenters-Refinement"].dropna(subset=["Delta_Ratio"])
    plt.figure(figsize=(12,5))
    sns.lineplot(df_ref,x="Delta_Ratio",y="Silhouette",hue="Metric",marker="o")
    plt.title("Impacto do Δ — KCenters Refinement")
    plt.savefig("04_refinement_delta.png",dpi=300); plt.close()


    kc = df[df["Algorithm"].str.contains("KCenters")].groupby("Dataset")["Silhouette"].max()
    km = df[df["Algorithm"]=="KMeans"].groupby("Dataset")["Silhouette"].mean()

    comp = pd.DataFrame({"KCenters Melhor":kc,"KMeans":km})
    comp.plot(kind="bar",figsize=(15,5),rot=40)
    plt.title("Comparação Final — KCenters (melhor) × KMeans")
    plt.ylabel("Silhouette Média")
    plt.savefig("05_kmeans_vs_kcenters.png",dpi=300); plt.close()


    plt.figure(figsize=(12,5))
    sns.barplot(df,x="Metric",y="Silhouette",palette="mako")
    plt.title("Ranking de Métricas — Média Global Silhouette")
    plt.savefig("06_metric_rank.png",dpi=300); plt.close()



def resumo_automatico(df):
    print("\n=============== RESUMO ANALÍTICO ===============")
    print(f"Melhor algoritmo no geral: {df.groupby('Algorithm')['Silhouette'].mean().idxmax()}")
    print(f"Algoritmo mais rápido: {df.groupby('Algorithm')['Time_s'].mean().idxmin()}")
    print(f"Métrica com maior média global: {df.groupby('Metric')['Silhouette'].mean().idxmax()}")
    print(f"KMeans comparação pronta — veja: 05_kmeans_vs_kcenters.png")



file = "RESULTADOS_FULL.csv"
df = load_results(file)

gerar_tabelas(df)
gerar_graficos(df)
resumo_automatico(df)

print("\nConcluído. Todas as tabelas + gráficos gerados.")
