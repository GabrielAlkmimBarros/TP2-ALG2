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


def mahalanobis(x, y, inv_cov):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    diff = x-y
    temp = np.dot(np.dot(diff.T, inv_cov),diff)

    dist = np.sqrt(temp)

    return dist











