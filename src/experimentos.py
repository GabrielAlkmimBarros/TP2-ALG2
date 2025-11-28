import numpy as np
import time
import pandas as pd
from typing import Callable, Any, Dict

# === Importação dos seus módulos (mantidos do original) ===
from geracao_dados import generate_sklearn_datasets, generate_multivariate_normal_datasets, load_real_datasets
from kcenters import k_centers_maxmin, k_centers_refinement, _get_max_radius, _get_distance_matrix
from distancias import minkowski, mahalanobis

from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import inv, LinAlgError


# =====================  PARÂMETROS EXPERIMENTO  =====================
N_RUNS = 15   # Execuções por configuração
DELTAS = [0.01, 0.05, 0.10, 0.15, 0.25]


# =====================  FUNÇÃO DE EXECUÇÃO  =====================
def run_experiment(
    X, y_true, k, metric_fn, algorithm_name,
    D_matrix=None, delta=None, **metric_params
) -> Dict[str,Any]:

    start = time.perf_counter()
    labels = None
    centers = None
    max_radius = np.nan

    try:
        # ---------------- K-MEANS ----------------
        if algorithm_name == "KMeans":
            model = SklearnKMeans(n_clusters=k, n_init='auto').fit(X)
            labels = model.labels_

            dist = euclidean_distances(X, model.cluster_centers_)
            max_radius = np.max(np.min(dist,axis=1))

        # ---------------- MAX-MIN ----------------
        elif algorithm_name == "MaxMin":
            centers = k_centers_maxmin(X, k, metric_fn, D_matrix=D_matrix, **metric_params)

        # ---------------- REFINEMENT ----------------
        elif algorithm_name == "Refinement":
            centers = k_centers_refinement(X, k, metric_fn, delta, D_matrix=D_matrix, **metric_params)

        else:
            raise ValueError("Algoritmo desconhecido")

    except Exception as e:
        return {"time":time.perf_counter()-start, "silhouette":np.nan, "adj_rand":np.nan,
                "max_radius":np.nan, "centers":None}

    end = time.perf_counter()


    # Se for K-Centers → definir rótulos e raio
    if algorithm_name in ["Refinement","MaxMin"] and centers is not None:
        dist = D_matrix[:, centers]
        labels = np.argmin(dist,axis=1)
        max_radius = _get_max_radius(D_matrix, centers)

    # métricas (se cluster >1)
    if labels is not None and len(set(labels))>1:
        sil = silhouette_score(X, labels)
        ari = adjusted_rand_score(y_true, labels)
    else:
        sil = np.nan; ari=np.nan

    return {
        "time":end-start,
        "silhouette":sil,
        "adj_rand":ari,
        "max_radius":max_radius,
        "centers":centers
    }



# =====================  LOOP PRINCIPAL — CORRIGIDO  =====================
def main_experiment_loop():

    datasets=[]
    datasets += generate_sklearn_datasets()           # 30 bases sintéticas
    datasets += generate_multivariate_normal_datasets() # +10 bases sintéticas
    datasets += load_real_datasets()                   # +11 bases reais UCI

    print(f"\n🔵 Total de bases geradas = {len(datasets)}\n")

    # Métricas testadas nos K-Centers
    METRIC_TESTS = [
        ("Minkowski-L1", minkowski, {"p":1.0}),
        ("Minkowski-L2", minkowski, {"p":2.0}),
        ("Minkowski-L3", minkowski, {"p":3.0}),
        ("Mahalanobis", mahalanobis, {}),
    ]

    results=[]


    #========================================================
    #  🔥 PARA CADA DATASET → roda KMeans primeiro
    #========================================================
    for X,y_true,k,name in datasets:

        print(f"\n===========================\n📌 BASE: {name} | k={k}\n===========================")

        # --------- KMEANS (Agora sim computado corretamente!) ---------
        print(f"➡ Rodando K-Means {N_RUNS}x\n")
        for _ in range(N_RUNS):
            r = run_experiment(X,y_true,k,None,"KMeans")
            results.append([name,"KMeans","Euclidiana","N/A",r["time"],r["max_radius"],r["silhouette"],r["adj_rand"],k])


        # --------- TESTE TODAS MÉTRICAS PARA K-CENTERS ---------
        for metric_name,metric_fn,metric_params in METRIC_TESTS:

            print(f"\n🔶 Métrica: {metric_name}")

            if metric_name=="Mahalanobis":      # cov única → rápido
                cov=np.cov(X,rowvar=False)
                try: inv_cov=inv(cov)
                except: inv_cov=np.linalg.pinv(cov)
                metric_params={"inv_cov":inv_cov}

            # matriz de distância 1x só
            D_matrix=_get_distance_matrix(X,metric_fn,**metric_params)

            # ======== MAX-MIN ========
            print("  ▪ MaxMin rodando...")
            for _ in range(N_RUNS):
                r=run_experiment(X,y_true,k,metric_fn,"MaxMin",D_matrix=D_matrix,**metric_params)
                results.append([name,"MaxMin",metric_name,"N/A",r["time"],r["max_radius"],r["silhouette"],r["adj_rand"],k])

            # ======== REFINEMENT ========
            base_width=_get_max_radius(D_matrix,[])

            for frac in DELTAS:
                delta=base_width*frac
                print(f"  ▪ Refinement Δ={int(frac*100)}% ...")
                for _ in range(N_RUNS):
                    r = run_experiment(X,y_true,k,metric_fn,"Refinement",D_matrix=D_matrix,delta=delta,**metric_params)
                    results.append([name,"Refinement",metric_name,frac,r["time"],r["max_radius"],r["silhouette"],r["adj_rand"],k])


    #========================================================
    # SALVAR RESULTADOS
    #========================================================
    df=pd.DataFrame(results,columns=["Dataset","Algorithm","Metric","Delta_Ratio",
                                     "Time_s","Max_Radius","Silhouette","Adj_Rand","k"])

    df.to_csv("RESULTADOS_FULL.csv",index=False)
    print("\n✅ EXPERIMENTO FINALIZADO — RESULTADOS EM:  RESULTADOS_FULL.csv")




if __name__=="__main__":
    main_experiment_loop()
