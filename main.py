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

def pk(f, variables, sub):
    h = hessian_f(f, variables, sub)
    modified_h = hessian_modification_multiple_I(h)
    h_inv = np.linalg.inv(modified_h)

    g = gradient_f(f, variables, sub)

    return (-h_inv * g).astype('float64')

def armijo_condition(cur, new):
    return cur > new;

c1 = 1e-4
c2 = 0.9
line_search_tau = 0.5

x, y = sm.symbols('x y')
f = f = (4 - 2.1 * (x**2) + (x**4)/3) * (x**2) + x * y + (-4 + 4 * (y ** 2)) * (y ** 2)

x0 = np.matrix([[0.01], [0.01]])

xs = []
ys = []

xs.append(-1)
ys.append(f.evalf(subs={x:x0.item(0, 0), y:x0.item(1, 0)}))

for k in range(0, 50):
    sub = {x:x0.item(0, 0), y:x0.item(1, 0)}
    direction = pk(f, [x, y], sub)
    alpha = 1.0
    ct = 0
    while(True):
        ct+=1

        x_n = x0 + np.matrix(direction) * alpha

        cur = f.evalf(subs={x:x0.item(0, 0), y:x0.item(1, 0)})
        new = f.evalf(subs={x:x_n.item(0, 0), y:x_n.item(1, 0)})

        if(armijo_condition(cur, new)):
            break
        else:
            alpha *= line_search_tau
        if(ct == 30):
            break
            k = 100

    if (k > 50):
        break

    x0 += np.matrix(direction) * alpha

    xs.append(k)
    ys.append(f.evalf(subs={x:x0.item(0, 0), y:x0.item(1, 0)}))

    print(x0)

plt.plot(xs, ys)
plt.show()
