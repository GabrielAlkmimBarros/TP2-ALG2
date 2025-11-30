import numpy as np


from distancias import minkowski 

def _get_distance_matrix(X, distance_fn, **kwargs):
    """Calcula a matriz de distância entre todos os pares de pontos em X."""
    X = np.array(X, dtype=float)
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    

    for i in range(n):
        for j in range(i + 1, n):
           
            dist = distance_fn(X[i], X[j], **kwargs)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix

def _dist_to_centers(D_matrix, center_indices):
    """ Calcula, para cada ponto, a menor distância ao conjunto de centros."""
    if not center_indices:
        # Se não há centros, a distância é infinita
        return np.full(D_matrix.shape[0], np.inf)


    return np.min(D_matrix[:, center_indices], axis=1)

def _get_max_radius(D_matrix, center_indices):

    if not center_indices:

        return np.max(D_matrix)
    
    min_dists = _dist_to_centers(D_matrix, center_indices)
    return np.max(min_dists)

def verify_radius(D_matrix, k, r):

    n = D_matrix.shape[0]
    S_prime = set(range(n))  
    C = []                   # Conjunto de centros (índices)
    
    while S_prime:
        s = next(iter(S_prime))
        C.append(s)
        
    
        points_to_remove = {j for j in S_prime if D_matrix[s, j] <= 2 * r}
        S_prime -= points_to_remove
        
  
        if len(C) > k:
            return None
            
    return C 

def k_centers_refinement(X, k, distance_fn, delta, D_matrix=None, **kwargs):

    if D_matrix is None:
        D_matrix = _get_distance_matrix(X, distance_fn, **kwargs)
    

    low = 0.0
    high = _get_max_radius(D_matrix, []) # rmax = max dist(si, sj)
    

    best_C = None
    

    while high - low > delta:
        r = (low + high) / 2
        
        C = verify_radius(D_matrix, k, r)
        
        if C is not None:

            high = r
            best_C = C
        else:

            low = r
            

    return best_C

def k_centers_maxmin(X, k, distance_fn, D_matrix=None, **kwargs):

    n = X.shape[0]
    

    if k >= n:
        return list(range(n))

    if D_matrix is None:
        D_matrix = _get_distance_matrix(X, distance_fn, **kwargs)


    C = [0] 
    
    while len(C) < k:

        min_dists = _dist_to_centers(D_matrix, C)
        

        next_center_index = np.argmax(min_dists)
        
        C.append(next_center_index)
        
    return C