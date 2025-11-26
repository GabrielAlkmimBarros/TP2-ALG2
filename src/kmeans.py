
import numpy as np
from typing import Literal


class KMeans:
    """
    Implementação do algoritmo K-Means do zero.
    
    Parâmetros:
    -----------
    n_clusters : int
        Número de clusters (k).
    max_iter : int
        Número máximo de iterações.
    tol : float
        Tolerância para convergência (diferença mínima entre centroides).
    random_state : int ou None
        Seed para reprodutibilidade.
    distance_metric : str
        Métrica de distância: 'euclidean', 'manhattan' ou 'minkowski'.
    p : float
        Parâmetro p para distância de Minkowski (p=2 é euclidiana, p=1 é manhattan).
    init : str
        Método de inicialização: 'random' ou 'kmeans++'.
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = None,
        distance_metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: float = 2.0,
        init: Literal['random', 'kmeans++'] = 'kmeans++'
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.distance_metric = distance_metric
        self.p = p
        self.init = init
        
        # Atributos que serão preenchidos após o fit
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None  # Soma das distâncias ao quadrado
        self.n_iter_ = 0
        
    def _set_random_state(self):
        """Define a seed para reprodutibilidade."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def _compute_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calcula a distância entre cada ponto e cada centroide.
        
        Retorna:
        --------
        distances : np.ndarray de shape (n_samples, n_clusters)
        """
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        
        for i, centroid in enumerate(centroids):
            if self.distance_metric == 'euclidean':
                # Distância Euclidiana: sqrt(sum((x - c)^2))
                distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            
            elif self.distance_metric == 'manhattan':
                # Distância Manhattan: sum(|x - c|)
                distances[:, i] = np.sum(np.abs(X - centroid), axis=1)
            
            elif self.distance_metric == 'minkowski':
                # Distância de Minkowski: (sum(|x - c|^p))^(1/p)
                distances[:, i] = np.power(
                    np.sum(np.power(np.abs(X - centroid), self.p), axis=1),
                    1 / self.p
                )
            else:
                raise ValueError(f"Métrica de distância '{self.distance_metric}' não suportada.")
        
        return distances
    
    def _init_centroids_random(self, X: np.ndarray) -> np.ndarray:
        """
        Inicialização aleatória: seleciona k pontos aleatórios como centroides.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _init_centroids_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """
        Inicialização K-Means++: escolhe centroides de forma espalhada.
        
        1. Escolhe o primeiro centroide aleatoriamente.
        2. Para cada centroide seguinte, escolhe o ponto com maior probabilidade
           proporcional à distância ao centroide mais próximo.
        """
        n_samples = X.shape[0]
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        # Primeiro centroide: aleatório
        first_idx = np.random.randint(0, n_samples)
        centroids[0] = X[first_idx]
        
        # Centroides seguintes
        for k in range(1, self.n_clusters):
            # Calcula distância de cada ponto ao centroide mais próximo
            distances = self._compute_distance(X, centroids[:k])
            min_distances = np.min(distances, axis=1)
            
            # Probabilidade proporcional ao quadrado da distância
            probabilities = min_distances ** 2
            probabilities /= probabilities.sum()
            
            # Escolhe o próximo centroide
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids[k] = X[next_idx]
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Atribui cada ponto ao cluster do centroide mais próximo.
        """
        distances = self._compute_distance(X, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Atualiza os centroides calculando a média dos pontos de cada cluster.
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # Se o cluster ficou vazio, reinicializa com um ponto aleatório
                new_centroids[k] = X[np.random.randint(0, X.shape[0])]
        
        return new_centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Calcula a inércia (soma das distâncias ao quadrado ao centroide).
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Executa o algoritmo K-Means.
        
        Parâmetros:
        -----------
        X : np.ndarray de shape (n_samples, n_features)
            Dados de entrada.
            
        Retorna:
        --------
        self : KMeans
            Instância ajustada.
        """
        self._set_random_state()
        X = np.array(X, dtype=np.float64)
        
        # Inicialização dos centroides
        if self.init == 'random':
            self.centroids_ = self._init_centroids_random(X)
        elif self.init == 'kmeans++':
            self.centroids_ = self._init_centroids_kmeans_plus_plus(X)
        else:
            raise ValueError(f"Método de inicialização '{self.init}' não suportado.")
        
        # Loop principal do K-Means
        for iteration in range(self.max_iter):
            # Passo 1: Atribuir pontos aos clusters
            self.labels_ = self._assign_clusters(X, self.centroids_)
            
            # Passo 2: Atualizar centroides
            new_centroids = self._update_centroids(X, self.labels_)
            
            # Verificar convergência
            centroid_shift = np.sqrt(np.sum((new_centroids - self.centroids_) ** 2))
            self.centroids_ = new_centroids
            self.n_iter_ = iteration + 1
            
            if centroid_shift < self.tol:
                break
        
        # Calcula inércia final
        self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz o cluster para novos dados.
        
        Parâmetros:
        -----------
        X : np.ndarray de shape (n_samples, n_features)
            Dados de entrada.
            
        Retorna:
        --------
        labels : np.ndarray de shape (n_samples,)
            Índice do cluster para cada amostra.
        """
        if self.centroids_ is None:
            raise ValueError("O modelo não foi treinado. Execute fit() primeiro.")
        
        X = np.array(X, dtype=np.float64)
        return self._assign_clusters(X, self.centroids_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Executa fit e predict em uma única chamada.
        """
        self.fit(X)
        return self.labels_


# =============================================================================
# Teste rápido
# =============================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    # Gera dados sintéticos
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    
    # Testa nosso K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, init='kmeans++')
    labels = kmeans.fit_predict(X)
    
    print(f"Número de iterações: {kmeans.n_iter_}")
    print(f"Inércia: {kmeans.inertia_:.2f}")
    
    # Visualização
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50)
    plt.title("Dados Originais")
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(kmeans.centroids_[:, 0], kmeans.centroids_[:, 1], 
                c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    plt.title("K-Means (nossa implementação)")
    
    plt.tight_layout()
    plt.show()
