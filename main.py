import numpy as np
import scipy as sp
from math import *
import matplotlib.pyplot as plt
import sympy as sm

#Example function: f(x) = x³ − x

def hessian_modification_multiple_I(hessian):
    beta = 1e-3
    rows = len(hessian)
    min_diag = min(np.diag(hessian))
    tau = None

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

def hessian_f(f, variables, sub):
    hessian = np.matrix(sm.hessian(f, variables).evalf(subs=sub)).astype('float64')
    return hessian

def gradient_f(f, variables, sub):
    grad = []
    for i in range(len(variables)):
        grad.append([f.diff(variables[i]).evalf(subs=sub)])
    return grad


x = sm.symbols('x')
f = x**3 - x

hessian = hessian_f(f, [x], {x:-0.1})

a = hessian_modification_multiple_I(hessian)


print(a)
