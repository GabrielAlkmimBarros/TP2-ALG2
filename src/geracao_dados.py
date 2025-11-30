import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any


MIN_SAMPLES = 700 
N_VARIATIONS = 5  # 5 conjuntos por tipo Scikit-Learn

# Caminho base para os dados reais 
REAL_DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "real_data", "real_data")

def generate_sklearn_datasets() -> List[Tuple[np.ndarray, np.ndarray, int, str]]:

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
                
                X = config["func"](MIN_SAMPLES, 2)
                y = np.zeros(MIN_SAMPLES) 
            
            elif base_name == "NoisyCircles":
                X, y = config["func"](n_samples=MIN_SAMPLES, factor=0.5, noise=0.05 + i*0.02, random_state=random_seed)
            
            elif base_name == "NoisyMoons":
                X, y = config["func"](n_samples=MIN_SAMPLES, noise=0.05 + i*0.02, random_state=random_seed)
            
            elif base_name == "StandardBlobs":
                
                X, y = config["func"](n_samples=MIN_SAMPLES, centers=k, cluster_std=1.0 + i*0.2, random_state=random_seed)

            elif base_name == "VariedBlobs":
                
                stds = [1.0, 2.5 - i*0.2, 0.5 + i*0.1]
                X, y = config["func"](n_samples=MIN_SAMPLES, centers=k, cluster_std=stds, random_state=random_seed)
                
            elif base_name == "AnisotropicBlobs":
                
                X_base, y = config["func"](n_samples=MIN_SAMPLES, random_state=random_seed)
               
                transformation = np.array(config["transform"]) + i * 0.1 
                X = np.dot(X_base, transformation)
            
           
            X = StandardScaler().fit_transform(X)
            
            dataset_name = f"SKL_{base_name}_{i+1}"
            datasets_list.append((X, y, k, dataset_name))

    return datasets_list

def generate_multivariate_normal_datasets() -> List[Tuple[np.ndarray, np.ndarray, int, str]]:

    datasets_list = []
    
 
    k = 4 # Exemplo de 4 centros
    
  
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
        
        scenario_index = i % len(SCENARIOS)
        name, means, covs = SCENARIOS[scenario_index]
        
        
        n_per_cluster = (MIN_SAMPLES // k) + (i//5 * 100)
        X_all = []
        y_all = []
        
        for j in range(k):
            
            current_mean = means[j] + np.random.normal(0, 0.1, 2)
            
            X_cluster = np.random.multivariate_normal(current_mean, covs[j], size=n_per_cluster)
            X_all.append(X_cluster)
            y_all.append(np.full(n_per_cluster, j))
        
        X = np.concatenate(X_all)
        y = np.concatenate(y_all)
        
        
        X = StandardScaler().fit_transform(X)
        
        datasets_list.append((X, y, k, f"MultiVar_{name}_{i+1}"))
        
    return datasets_list


def load_real_datasets() -> List[Tuple[np.ndarray, np.ndarray, int, str]]:

    datasets_list = []
    
    # Configuração dos datasets: (pasta, arquivo, coluna_alvo, k, nome)
    REAL_DATASETS_CONFIG = [
        {
            "path": "banknote+authentication",
            "file": "banknote.csv",
            "target": "class",
            "k": 2,
            "name": "UCI_Banknote"
        },
        {
            "path": "optical+recognition+of+handwritten+digits",
            "file": "optdigits.csv",
            "target": "digit",
            "k": 10,
            "name": "UCI_OptDigits"
        },
        {
            "path": "wine+quality",
            "file": "winequality_red.csv",
            "target": "quality",
            "k": 6,
            "name": "UCI_WineRed"
        },
        {
            "path": "wine+quality",
            "file": "winequality_white.csv",
            "target": "quality",
            "k": 7,
            "name": "UCI_WineWhite"
        },
        {
            "path": "taiwanese+bankruptcy+prediction",
            "file": "bankruptcy.csv",
            "target": "bankrupt",
            "k": 2,
            "name": "UCI_Bankruptcy"
        },
        {
            "path": "secom",
            "file": "secom.csv",
            "target": "label",
            "k": 2,
            "name": "UCI_SECOM"
        },
        {
            "path": "drug+consumption+quantified",
            "file": "drug_consumption.csv",
            "target": "Cannabis",
            "k": 7,
            "name": "UCI_DrugConsumption"
        },
        {
            "path": "myocardial+infarction+complications",
            "file": "mi.csv",
            "target": "target",
            "k": 8,
            "name": "UCI_MyocardialInfarction"
        },
        {
            "path": "estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition",
            "file": "obesity.csv",
            "target": "NObeyesdad",
            "k": 7,
            "name": "UCI_Obesity"
        },
        {
            "path": "cardiotocography",
            "file": "cardiotocography.csv",
            "target": "NSP",
            "k": 3,
            "name": "UCI_Cardiotocography"
        },
        {
            "path": "beed_+bangalore+eeg+epilepsy+dataset",
            "file": "BEED_Data.csv",
            "target": "y",
            "k": 4,
            "name": "UCI_BEED_EEG"
        },
    ]
    
    for config in REAL_DATASETS_CONFIG:
        try:
            filepath = os.path.join(REAL_DATA_BASE_PATH, config["path"], config["file"])
            

            df = pd.read_csv(filepath)
            
            target_col = config["target"]
            
            if target_col not in df.columns:
                possible_targets = [col for col in df.columns if target_col.lower() in col.lower()]
                if possible_targets:
                    target_col = possible_targets[0]
                else:
                    print(f"Coluna '{config['target']}' não encontrada em {config['name']}. Colunas: {df.columns.tolist()}")
                    continue
            

            y = df[target_col].values
            X = df.drop(columns=[target_col])
            
        
            X = X.select_dtypes(include=[np.number])
            
 
            X = X.fillna(X.mean())
            
   
            X = X.values.astype(float)
            
            # Codifica labels se forem strings
            if y.dtype == object:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
            

            X = StandardScaler().fit_transform(X)
            
            k = config["k"]
            name = config["name"]
            
            datasets_list.append((X, y, k, name))
            print(f"Carregado: {name} | Shape: {X.shape} | k={k}")
            
        except Exception as e:
            print(f"Erro ao carregar {config['name']}: {e}")
            continue
    
    return datasets_list