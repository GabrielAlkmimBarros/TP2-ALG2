import numpy as np
from numpy.linalg import inv

def minkowski(x,y,p = 2):

    x=np.array(x, dtype = float)
    y=np.array(y, dtype = float)

    diff = x - y

    abs_diff = np.abs(diff)
    powered = abs_diff ** p
    soma = np.sum(powered)

    dist = soma ** (1/p)

    return dist

def minkowski_matrix(X, p = 2):
    X = np.array(X, dtype= float)
    n = X.shape[0]

    dist_matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1,n):
            dist = minkowski(X[i], X[j], p)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix

def mahalanobis(x, y, inv_cov):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    diff = x-y
    temp = np.dot(np.dot(diff.T, inv_cov),diff)

    dist = np.sqrt(temp)

    return dist


def mahalanobis_matrix(X):
    X = np.array(X, dtype=float)
    n = X.shape[0]

    cov = np.cov(X, rowvar=False)
    inv_cov = inv(cov)

    dist_matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1, n):
            dist = mahalanobis(X[i], X[j], inv_cov)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix, inv_cov


x = np.array([1,2,3])
y = np.array([4,6,9])

print("Manhattan (p=1):", minkowski(x,y,1))
print("Euclidiana (p=2):", minkowski(x,y,2))
print("Outro p=3:", minkowski(x,y,3))

X = np.array([
    [1,2,3],
    [4,6,9],
    [2,1,0]
])

print(minkowski_matrix(X, p=2)) #euclidiano




X = np.array([
    [1,2],
    [2,3],
    [3,6]
])

dist_matrix, inv_cov = mahalanobis_matrix(X)

print("\nMatriz Mahalanobis:\n", dist_matrix)

print("\nDistância direta:", mahalanobis(X[0], X[2], inv_cov))
