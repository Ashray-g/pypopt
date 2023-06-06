import numpy as np
import scipy as sp
from math import *
import matplotlib.pyplot as plt

#Example function: f(x) = x³ − x

def hessian_modification_multiple_I(hessian):
    beta = 1e-3
    rows = len(hessian)
    min_diag = min(np.diag(hessian))
    tau = max(0, min_diag)

    if(min_diag > 0):
        tau = 0
    else:
        tau = -min_diag + beta

    while(True):
        try:
            np.linalg.cholesky(hessian + np.eye(rows) * tau)
            break
        except np.linalg.LinAlgError:
            tau = max(10 * tau, beta)

    hessian = hessian + np.eye(rows) * tau

    return hessian

hessian = [[-1, 0, 0], [0, -2, 0], [0, 0, 0]]

a = hessian_modification_multiple_I(hessian)

print(a)
