import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any

# --- Configurações ---
MIN_SAMPLES = 700 
N_VARIATIONS = 5  # 5 conjuntos por tipo Scikit-Learn

def generate_sklearn_datasets() -> List[Tuple[np.ndarray, np.ndarray, int, str]]:
    """
    Gera 30 conjuntos de dados (5 variações para 6 tipos) usando funções do Scikit-Learn.
    
    Retorna: Lista de (X, y_true, k, nome_tipo).
    """
    datasets_list = []
    
    DATASET_CONFIGS = [
        {"name": "NoisyCircles", "func": make_circles, "k": 2},
        {"name": "NoisyMoons", "func": make_moons, "k": 2},
        {"name": "VariedBlobs", "func": make_blobs, "k": 3},
        {"name": "AnisotropicBlobs", "func": make_blobs, "k": 3, "transform": [[0.6, -0.6], [-0.4, 0.8]]},
        {"name": "StandardBlobs", "func": make_blobs, "k": 3},
        {"name": "NoStructure", "func": np.random.rand, "k": 3},
    ]

    for config in DATASET_CONFIGS:
        base_name = config["name"]
        k = config["k"]
        
        for i in range(N_VARIATIONS):
            random_seed = i + 10 
            
            if base_name == "NoStructure":
                # Gera dados uniformes aleatórios entre 0 e 1 (sem estrutura)
                X = config["func"](MIN_SAMPLES, 2)
                y = np.zeros(MIN_SAMPLES) # Rótulos arbitrários, não usados para k, apenas para ARI base
            
            elif base_name == "NoisyCircles":
                X, y = config["func"](n_samples=MIN_SAMPLES, factor=0.5, noise=0.05 + i*0.02, random_state=random_seed)
            
            elif base_name == "NoisyMoons":
                X, y = config["func"](n_samples=MIN_SAMPLES, noise=0.05 + i*0.02, random_state=random_seed)
            
            elif base_name == "StandardBlobs":
                # Varia o desvio padrão de todos os clusters juntos
                X, y = config["func"](n_samples=MIN_SAMPLES, centers=k, cluster_std=1.0 + i*0.2, random_state=random_seed)

            elif base_name == "VariedBlobs":
                # Varia o desvio padrão de cada cluster individualmente
                stds = [1.0, 2.5 - i*0.2, 0.5 + i*0.1]
                X, y = config["func"](n_samples=MIN_SAMPLES, centers=k, cluster_std=stds, random_state=random_seed)
                
            elif base_name == "AnisotropicBlobs":
                # Gera blobs padrão e aplica transformação linear
                X_base, y = config["func"](n_samples=MIN_SAMPLES, random_state=random_seed)
                # Varia a transformação levemente (aumenta o alongamento)
                transformation = np.array(config["transform"]) + i * 0.1 
                X = np.dot(X_base, transformation)
            
            # Padroniza os dados (Média 0, Desvio Padrão 1)
            X = StandardScaler().fit_transform(X)
            
            dataset_name = f"SKL_{base_name}_{i+1}"
            datasets_list.append((X, y, k, dataset_name))

    return datasets_list

def generate_multivariate_normal_datasets() -> List[Tuple[np.ndarray, np.ndarray, int, str]]:
    """
    Gera 10 conjuntos de dados usando Distribuição Normal Multivariada (2D).
    """
    datasets_list = []
    
    # Parâmetros fixos
    k = 4 # Exemplo de 4 centros
    
    # Cenários de forma e sobreposição (Nome, Médias, Covariâncias)
    SCENARIOS = [
        # 1. Circular e Baixa Sobreposição
        ("Circular_Low", 
         [[-6, 6], [6, 6], [-6, -6], [6, -6]], 
         [np.diag([1.0, 1.0])] * k),
         
        # 2. Circular e Média Sobreposição
        ("Circular_Medium", 
         [[-3, 3], [3, 3], [-3, -3], [3, -3]], 
         [np.diag([2.0, 2.0])] * k),
         
        # 3. Elíptico Forte (X >> Y)
        ("Elliptical_Strong", 
         [[-4, 4], [4, 4], [-4, -4], [4, -4]], 
         [np.diag([6.0, 1.0])] * k),
         
        # 4. Elíptico Rotacionado (com correlação)
        ("Elliptical_Rotated", 
         [[-4, 0], [0, 4], [4, 0], [0, -4]], 
         [np.array([[2.0, 1.5], [1.5, 2.0]])] * k),

        # 5. Alta Sobreposição
        ("HighOverlap", 
         [[0, 1], [0, -1], [1, 0], [-1, 0]], 
         [np.diag([2.5, 2.5])] * k),
    ]

    for i in range(10): 
        # Alterna entre os cenários
        scenario_index = i % len(SCENARIOS)
        name, means, covs = SCENARIOS[scenario_index]
        
        # Gera o dobro de amostras para o teste de sobreposição (i//5 controla isso)
        n_per_cluster = (MIN_SAMPLES // k) + (i//5 * 100)
        X_all = []
        y_all = []
        
        for j in range(k):
            # Adiciona ruído leve ao centro (para variar as 10 execuções)
            current_mean = means[j] + np.random.normal(0, 0.1, 2)
            # Amostra pontos para o cluster j
            X_cluster = np.random.multivariate_normal(current_mean, covs[j], size=n_per_cluster)
            X_all.append(X_cluster)
            y_all.append(np.full(n_per_cluster, j))
        
        X = np.concatenate(X_all)
        y = np.concatenate(y_all)
        
        # Padroniza os dados
        X = StandardScaler().fit_transform(X)
        
        datasets_list.append((X, y, k, f"MultiVar_{name}_{i+1}"))
        
    return datasets_list