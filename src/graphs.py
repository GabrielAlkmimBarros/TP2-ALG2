import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.datasets import make_circles, make_blobs
from typing import List, Dict, Any, Tuple

# --- 1. SIMULAÇÃO DOS ALGORITMOS 2-APROXIMADOS (Você deve substituir por suas implementações) ---
# Como não temos suas implementações MaxMin e Refinement, vamos simular o comportamento.

def calculate_max_radius(X: np.ndarray, centers: np.ndarray) -> float:
    """Calcula o Max Radius (o raio da solução, que é o máximo min-distância)."""
    if centers.shape[0] == 0:
        return np.inf
    
    # Calcular a distância de cada ponto para cada centro
    # Usando broadcasting: (N, 2) - (k, 2) -> (N, k, 2)
    distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    
    # Encontrar a distância mínima de cada ponto para qualquer centro
    min_distances = np.min(distances, axis=1)
    
    # O raio máximo é o máximo dessas distâncias mínimas
    max_radius = np.max(min_distances)
    return max_radius

def maxmin_clustering(X: np.ndarray, k: int, metric: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simulação do algoritmo MaxMin.
    Retorna os rótulos, centros e tempo de execução simulado.
    """
    start_time = time.time()
    
    # Simulação: escolher k centros aleatórios (substitua pela lógica MaxMin real)
    indices = np.random.choice(X.shape[0], k, replace=False)
    centers = X[indices]
    
    # Rotular os pontos (atribuição por vizinho mais próximo)
    distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    
    end_time = time.time()
    
    max_radius = calculate_max_radius(X, centers)

    return labels, centers, end_time - start_time, max_radius

def refinement_clustering(X: np.ndarray, k: int, metric: str, delta_ratio: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simulação do algoritmo Refinement.
    Retorna os rótulos, centros e tempo de execução simulado.
    O delta_ratio afeta a qualidade e o tempo (simuladamente).
    """
    start_time = time.time()
    
    # Simulação: a largura do intervalo (delta_ratio) afeta a qualidade.
    # Usamos o K-Means como base e adicionamos um "ruído" controlado.
    
    kmeans = KMeans(n_clusters=k, random_state=None, n_init='auto')
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # O tempo de execução é simulado para ser maior para um refinamento mais fino (menor delta_ratio)
    simulated_time = 0.05 + (1 - delta_ratio) * 0.1 * np.random.rand()
    end_time = start_time + simulated_time
    
    # Simular que um refinamento menos restritivo (maior delta_ratio) tem uma solução pior (maior raio)
    max_radius = calculate_max_radius(X, centers) * (1 + (delta_ratio / 0.25) * 0.2 * np.random.rand())

    return labels, centers, end_time - start_time, max_radius

# --- 2. GERAÇÃO DE DADOS DE TESTE ---

def generate_datasets(n_samples: int = 500) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Gera diferentes conjuntos de dados para teste."""
    
    # SKL_NoisyCircles: Difícil para K-Means (não-esférico)
    X_circles, y_circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=42)
    
    # SKL_Blobs: Fácil para K-Means (esférico)
    X_blobs, y_blobs = make_blobs(n_samples=n_samples, random_state=42)
    
    # SKL_EllipticalBlobs: Difícil para distâncias esféricas como Euclidiana
    # Criamos blobs e os transformamos para criar um formato elíptico.
    X_elliptical, y_elliptical = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5], random_state=42)
    
    return {
        "SKL_NoisyCircles": (X_circles, y_circles),
        "SKL_Blobs": (X_blobs, y_blobs),
        "SKL_EllipticalBlobs": (X_elliptical, y_elliptical)
    }

# --- 3. FUNÇÃO PRINCIPAL DE EXECUÇÃO DE EXPERIMENTOS ---

def run_experiment(datasets: Dict[str, Tuple[np.ndarray, np.ndarray]], n_runs: int = 15, k: int = 2) -> pd.DataFrame:
    """Executa todos os experimentos e coleta os resultados."""
    results: List[Dict[str, Any]] = []
    
    # Parâmetros de variação para o Refinement (largura final de 1% a 25% da largura inicial)
    # Escolha linear de 5 valores
    delta_ratios = np.linspace(0.01, 0.25, 5).round(2) 
    
    for dataset_name, (X, y_true) in datasets.items():
        # --- K-MEANS (Baseline) ---
        print(f"Executando K-Means em {dataset_name}...")
        for run in range(n_runs):
            start_time = time.time()
            # O k-means++ é o padrão e geralmente faz um bom trabalho.
            # n_init='auto' garante múltiplas inicializações para melhor resultado.
            kmeans = KMeans(n_clusters=k, random_state=None, n_init='auto')
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
            end_time = time.time()
            
            # Cálculo das Métricas
            time_s = end_time - start_time
            max_radius = calculate_max_radius(X, centers)
            silhouette = silhouette_score(X, labels)
            adj_rand = adjusted_rand_score(y_true, labels)
            
            results.append({
                "Dataset": dataset_name,
                "Algorithm": "KMeans",
                "Metric": "Euclidiana", # K-Means padrão usa distância Euclidiana
                "Delta_Ratio": np.nan,
                "Time_s": time_s,
                "Max_Radius": max_radius,
                "Silhouette": silhouette,
                "Adj_Rand": adj_rand,
                "k": k
            })

        # --- MAXMIN (Algoritmo 2-Aproximado) ---
        print(f"Executando MaxMin em {dataset_name}...")
        for run in range(n_runs):
            # Assumimos Minkowski-L1 (Manhattan) para diversidade, mas ajuste conforme sua implementação.
            labels, centers, time_s, max_radius = maxmin_clustering(X, k, "Minkowski-L1") 
            
            # Cálculo das Métricas
            silhouette = silhouette_score(X, labels)
            adj_rand = adjusted_rand_score(y_true, labels)
            
            results.append({
                "Dataset": dataset_name,
                "Algorithm": "MaxMin",
                "Metric": "Minkowski-L1",
                "Delta_Ratio": np.nan,
                "Time_s": time_s,
                "Max_Radius": max_radius,
                "Silhouette": silhouette,
                "Adj_Rand": adj_rand,
                "k": k
            })

        # --- REFINEMENT (Algoritmo com variação de Delta_Ratio) ---
        print(f"Executando Refinement em {dataset_name} com variação de Delta_Ratio...")
        for delta_ratio in delta_ratios:
            for run in range(n_runs):
                # Assumimos Minkowski-L1, ajuste conforme sua implementação.
                labels, centers, time_s, max_radius = refinement_clustering(X, k, "Minkowski-L1", delta_ratio)
                
                # Cálculo das Métricas
                silhouette = silhouette_score(X, labels)
                adj_rand = adjusted_rand_score(y_true, labels)
                
                results.append({
                    "Dataset": dataset_name,
                    "Algorithm": "Refinement",
                    "Metric": "Minkowski-L1",
                    "Delta_Ratio": delta_ratio,
                    "Time_s": time_s,
                    "Max_Radius": max_radius,
                    "Silhouette": silhouette,
                    "Adj_Rand": adj_rand,
                    "k": k
                })
                
    return pd.DataFrame(results)

# --- 4. EXECUÇÃO E AGREGAÇÃO DE RESULTADOS ---

# 1. Gerar os dados
datasets = generate_datasets()

# 2. Executar os experimentos
raw_results_df = run_experiment(datasets, n_runs=15, k=2)

# 3. Agregação e Análise
# Agrupar por Dataset, Algorithm, Metric e Delta_Ratio (onde aplicável)
group_cols = ["Dataset", "Algorithm", "Metric", "Delta_Ratio", "k"]

# Calcular Média e Desvio-Padrão para as métricas numéricas
aggregated_results = raw_results_df.groupby(group_cols).agg(
    # Média
    Media_Time_s=('Time_s', 'mean'),
    Media_Max_Radius=('Max_Radius', 'mean'),
    Media_Silhouette=('Silhouette', 'mean'),
    Media_Adj_Rand=('Adj_Rand', 'mean'),
    # Desvio-Padrão
    DP_Time_s=('Time_s', 'std'),
    DP_Max_Radius=('Max_Radius', 'std'),
    DP_Silhouette=('Silhouette', 'std'),
    DP_Adj_Rand=('Adj_Rand', 'std')
).reset_index()

# Exibir a Tabela Final Agregada
print("\n" + "="*80)
print("TABELA DE RESULTADOS AGREGADOS (MÉDIA e DESVIO-PADRÃO)")
print("="*80)
print(aggregated_results.to_string())
