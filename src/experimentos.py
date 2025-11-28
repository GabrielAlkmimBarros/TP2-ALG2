import numpy as np
import time
import pandas as pd
from typing import List, Callable, Any, Tuple, Dict

# Importa as funções de geração de dados
from geracao_dados import generate_sklearn_datasets, generate_multivariate_normal_datasets

# Importa as funções de algoritmo, auxiliares e métrica
# Atenção: Certifique-se de que distancias.py e k_centers.py estão no mesmo diretório 
# ou no PYTHONPATH.
from kcenters import k_centers_maxmin, k_centers_refinement, _get_max_radius, _get_distance_matrix
from distancias import minkowski, mahalanobis 
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import inv, LinAlgError # Importa para cálculo de covariância inversa

# Constantes para o loop
N_RUNS = 15 # 15 execuções por teste

def run_experiment(
    X: np.ndarray, 
    y_true: np.ndarray, 
    k: int, 
    metric_fn: Callable, 
    algorithm_name: str, 
    D_matrix: np.ndarray = None, 
    delta: float = None, 
    **metric_params: Any
) -> Dict[str, Any]:
    """Executa um único teste e coleta métricas."""
    
    # 1. Medição de tempo e execução
    start_time = time.perf_counter()
    centers = None
    labels = None
    max_radius = np.nan
    
    # Junta os parâmetros da métrica e a D_matrix para passar como kwargs
    # Isso resolve o problema de argumento posicional
    all_kwargs = metric_params.copy()
    if D_matrix is not None:
        all_kwargs['D_matrix'] = D_matrix
    
    try:
        if algorithm_name == "MaxMin":

            centers = k_centers_maxmin(X, k, metric_fn, **all_kwargs)
        
        elif algorithm_name == "Refinement":
            if delta is None: raise ValueError("Refinement requires 'delta' parameter.")
            centers = k_centers_refinement(X, k, metric_fn, delta, **all_kwargs)

        elif algorithm_name == "SklearnKMeans":
            # K-Means do Scikit-Learn
            kmeans = SklearnKMeans(n_clusters=k, random_state=None, n_init='auto').fit(X)
            labels = kmeans.labels_
            
            # Cálculo do Raio Máximo do K-Means (usando Euclidiana)
            distances_to_centers = euclidean_distances(X, kmeans.cluster_centers_)
            min_distances_to_centers = np.min(distances_to_centers, axis=1)
            max_radius = np.max(min_distances_to_centers)
            centers = None # Não armazena índices de centros
            
        else:
            raise ValueError(f"Algoritmo desconhecido: {algorithm_name}")

    except Exception as e:
        metric_name = metric_fn.__name__ if metric_fn else 'Euclidiana (KMeans)'
        print(f"Erro na execução do algoritmo {algorithm_name} com {metric_name}: {e}")
        end_time = time.perf_counter()
        return {
            "time": end_time - start_time,
            "max_radius": np.nan,
            "silhouette": np.nan,
            "adj_rand": np.nan,
            "centers": None
        }

    end_time = time.perf_counter()
    

    if algorithm_name in ["MaxMin", "Refinement"] and centers is not None:
        if D_matrix is None:
             # Isso deve ocorrer apenas se a Mahalanobis tiver sido pulada
             if metric_fn is mahalanobis:
                print("Recalculando D_matrix com a métrica padrão (Euclidiana) para atribuição de rótulos.")
                D_matrix = _get_distance_matrix(X, minkowski, p=2.0)
             else:
                raise RuntimeError("D_matrix é None, necessário para K-Centers.")

        distances_to_centers = D_matrix[:, centers]
        labels = np.argmin(distances_to_centers, axis=1)
        
        # Raio máximo (Quality Measure: The Max-Min objective value)
        max_radius = _get_max_radius(D_matrix, centers)
    
    # 3. Coleta de Métricas (Requer rótulos válidos)
    if labels is not None and len(np.unique(labels)) > 1:
        try:
            # Silhouette: Usa métrica Euclidiana (padrão) para consistência
            silhouette = silhouette_score(X, labels, metric='euclidean') 
            
            # Rand Ajustado: Requer rótulos verdadeiros
            adj_rand = adjusted_rand_score(y_true, labels)

        except Exception as e:
            print(f"Erro no cálculo de métricas para {algorithm_name}: {e}")
            silhouette = np.nan
            adj_rand = np.nan
    else:
        # Se os rótulos forem inválidos (ex: todos no mesmo cluster)
        silhouette = np.nan
        adj_rand = np.nan


    return {
        "time": end_time - start_time,
        "max_radius": max_radius,
        "silhouette": silhouette,
        "adj_rand": adj_rand,
        "centers": centers
    }

def main_experiment_loop():
    # 1. Preparação dos dados
    all_datasets = []
    print("Gerando bases Scikit-Learn (30)...")
    all_datasets.extend(generate_sklearn_datasets())
    print("Gerando bases Multivariadas (10)...")
    all_datasets.extend(generate_multivariate_normal_datasets())
    print(f"Total de bases geradas: {len(all_datasets)}")

    # Lista de todas as combinações de métricas para testar
    METRIC_COMBINATIONS_TPL = [
        ("Minkowski-L1", minkowski, {"p": 1.0}),   # Manhattan
        ("Minkowski-L2", minkowski, {"p": 2.0}),   # Euclidiana
        ("Minkowski-L3", minkowski, {"p": 3.0}),   # Outro p
        ("Mahalanobis", mahalanobis, {}),
    ]
    
    # DataFrame para armazenar todos os resultados
    results_df = pd.DataFrame(columns=[
        "Dataset", "Algorithm", "Metric", "Delta_Ratio", "Run", 
        "Time_s", "Max_Radius", "Silhouette", "Adj_Rand", "k"
    ])
    record_count = 0

    for X, y_true, k, data_name in all_datasets:
        
        for metric_name, metric_fn, base_metric_params in METRIC_COMBINATIONS_TPL:
            print(f"\n--- Base: {data_name} | Métrica: {metric_name} | k={k} ---")
            
            # Copia os parâmetros base
            metric_params = base_metric_params.copy()

            # --- CORREÇÃO: Tratamento específico para Mahalanobis (inv_cov) ---
            if metric_fn == mahalanobis:
                try:
                    # Calcula a matriz de covariância
                    cov_matrix = np.cov(X, rowvar=False) 
                    
                    # Tenta calcular a inversa. Usa pseudo-inversa se for singular.
                    try:
                        inv_cov = inv(cov_matrix)
                    except LinAlgError:
                        print("ATENÇÃO: Matriz de covariância singular. Usando pseudo-inversa.")
                        inv_cov = np.linalg.pinv(cov_matrix)

                    # Adiciona a matriz inversa aos parâmetros da métrica
                    metric_params['inv_cov'] = inv_cov
                    
                except Exception as e:
                    print(f"ERRO CRÍTICO no cálculo da inv_cov para Mahalanobis: {e}. Pulando Mahalanobis para {data_name}.")
                    continue # Pula esta combinação se o cálculo falhar


            # --- CÁLCULO ÚNICO DA MATRIZ DE DISTÂNCIA ---
            D_matrix = _get_distance_matrix(X, metric_fn, **metric_params)
            
            # Verificação se D_matrix é válida após o cálculo
            if np.isnan(D_matrix).any() or D_matrix.sum() == 0 and D_matrix.shape[0] > 0:
                print(f"ATENÇÃO: D_matrix inválida/vazia para {data_name} com {metric_name}. Pulando esta combinação.")
                continue

            # --- 1. K-CENTERS MAX-MIN (15 Execuções) ---
            print(f"Executando MaxMin ({N_RUNS}x)")
            for run in range(N_RUNS):
                # O D_matrix é passado como argumento nomeado D_matrix=D_matrix
                results = run_experiment(X, y_true, k, metric_fn, "MaxMin", D_matrix=D_matrix, **metric_params)
                
                results_df.loc[record_count] = [
                    data_name, "MaxMin", metric_name, np.nan, run + 1, 
                    results["time"], results["max_radius"], results["silhouette"], results["adj_rand"], k
                ]
                record_count += 1


            # --- 2. K-CENTERS REFINAMENTO (5x15 Execuções) ---
            initial_radius = _get_max_radius(D_matrix, [])
            initial_width = initial_radius 
            
            # Valores de delta: 1%, 5%, 10%, 15%, 25% da largura inicial
            delta_fractions = [0.01, 0.05, 0.10, 0.15, 0.25]
            
            for fraction in delta_fractions:
                delta = initial_width * fraction
                print(f"Executando Refinamento (delta={fraction*100:.0f}%, {N_RUNS}x)")

                for run in range(N_RUNS):
                    # O D_matrix é passado como argumento nomeado D_matrix=D_matrix
                    results = run_experiment(X, y_true, k, metric_fn, "Refinement", D_matrix=D_matrix, delta=delta, **metric_params)
                    
                    results_df.loc[record_count] = [
                        data_name, "Refinement", metric_name, fraction, run + 1, 
                        results["time"], results["max_radius"], results["silhouette"], results["adj_rand"], k
                    ]
                    record_count += 1

        # --- 3. K-MEANS SCIKIT-LEARN (15 Execuções) ---
        # Executado apenas uma vez por Dataset
        print(f"Executando K-Means ({N_RUNS}x)")
        for run in range(N_RUNS):
             # O K-Means não depende da métrica
             results = run_experiment(X, y_true, k, None, "SklearnKMeans")
             
             results_df.loc[record_count] = [
                data_name, "KMeans_SKL", "Euclidiana", np.nan, run + 1, 
                results["time"], results["max_radius"], results["silhouette"], results["adj_rand"], k
             ]
             record_count += 1
                 
    # --- 4. Agregação e Exportação ---
    print("\n--- Agregando Resultados ---")
    
    # Define a coluna de grupo para agregação
    group_cols = ["Dataset", "Algorithm", "Metric", "Delta_Ratio", "k"]
    
    # Agrega os dados por média e desvio padrão
    summary_df = results_df.groupby(group_cols).agg(
        Time_mean=('Time_s', 'mean'),
        Time_std=('Time_s', 'std'),
        Radius_mean=('Max_Radius', 'mean'),
        Radius_std=('Max_Radius', 'std'),
        Silhouette_mean=('Silhouette', 'mean'),
        Silhouette_std=('Silhouette', 'std'),
        AdjRand_mean=('Adj_Rand', 'mean'),
        AdjRand_std=('Adj_Rand', 'std'),
    ).reset_index()

    # Formata a coluna Delta_Ratio para ser mais legível no relatório
    summary_df['Delta_Ratio'] = summary_df['Delta_Ratio'].fillna('N/A').apply(
        lambda x: f'{x*100:.0f}%' if isinstance(x, (float, np.float64)) else x
    )
    
    output_filename = "resultados_agrupamento.csv"
    summary_df.to_csv(output_filename, index=False)
    print(f"\n--- Sucesso! Resultados agregados salvos em {output_filename} ---")


if __name__ == '__main__':
    main_experiment_loop()