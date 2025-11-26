import numpy as np
from minkowski import minkowski

def _get_distance_matrix(X, distance_fn, **kwargs):
    """Calcula a matriz de distância entre todos os pares de pontos em X."""
    X = np.array(X, dtype=float)
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    
    # A função de distância deve ser chamada para cada par
    # Reutilizando a lógica da sua implementação 'minkowski_matrix'
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_fn(X[i], X[j], **kwargs)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix

def _dist_to_centers(D_matrix, center_indices):
    """
    Calcula, para cada ponto, a menor distância ao conjunto de centros.
    
    """
    if not center_indices:
        # Se não há centros, a distância é infinita (para garantir que o primeiro centro seja selecionado)
        return np.full(D_matrix.shape[0], np.inf)


    return np.min(D_matrix[:, center_indices], axis=1)

def _get_max_radius(D_matrix, center_indices):
    """Calcula o raio máximo r(C) para um conjunto de centros C."""
    if not center_indices:
        return np.max(D_matrix)
    
    min_dists = _dist_to_centers(D_matrix, center_indices)
    return np.max(min_dists)

def verify_radius(D_matrix, k, r):
    """
    Verifica se é possível cobrir todos os pontos com k ou menos centros, 
    garantindo que o raio máximo seja <= 2r.
    """
    n = D_matrix.shape[0]
    S_prime = set(range(n))  # Conjunto de pontos não cobertos (índices)
    C = []                   # Conjunto de centros (índices)
    

    while S_prime:
        # Selecione arbitrariamente um ponto s no conjunto não coberto (o primeiro)
        s = next(iter(S_prime))
        
        # Coloque s em C
        C.append(s)
        
        # Remova de S' todos os pontos que estiverem a uma distância máxima de 2r de s
        # Encontra os índices dos pontos em S' que estão a dist <= 2r de s
        points_to_remove = {j for j in S_prime if D_matrix[s, j] <= 2 * r}
        S_prime -= points_to_remove
        
        # Se o número de centros exceder k, não há solução para este r
        if len(C) > k:
            return None
            
    return C # Retorna os centros (ou None se |C| > k)

def k_centers_refinement(X, k, distance_fn, delta, **kwargs):
    """
    Algoritmo 2-aproximado para k-centros usando busca binária no raio ótimo.
    """
    D_matrix = _get_distance_matrix(X, distance_fn, **kwargs)
    
    # 1. intervalo inicial [low, high]
    low = 0.0
    high = np.max(D_matrix) # rmax = max dist(si, sj)
    
    # melhor solução válida
    best_C = None
    
    # 2. Refinar o intervalo até que a largura seja <= delta
    while high - low > delta:
        r = (low + high) / 2
        

        C = verify_radius(D_matrix, k, r)
        
        if C is not None:
            # |C| <= k. Tente um raio menor.
            high = r
            best_C = C
        else:
            # Solução não encontrada  O raio ótimo deve ser > r.
            low = r
            

    return best_C

def k_centers_maxmin(X, k, distance_fn, **kwargs):
    """
    Algoritmo 2-aproximado guloso para k-centros (Max-Min Distance).
    """
    n = X.shape[0]
    
    # Se k >= n, retorne todos os pontos
    if k >= n:
        return list(range(n))
        
    D_matrix = _get_distance_matrix(X, distance_fn, **kwargs)
    

    C = [0] 
    

    while len(C) < k:
        # Encontra as menores distâncias de cada ponto para os centros em C
    
        min_dists = _dist_to_centers(D_matrix, C)
        
        # Selecione s que maximize dist(s, C)
        # O próximo centro é o ponto com a maior distância mínima
        next_center_index = np.argmax(min_dists)
        

        C.append(next_center_index)
        
    return C