"""
Exercise 6.2.3, norminal problem
The purpose is to test scipy.optimize.minimize
"""

import numpy as np
from scipy.optimize import minimize


def obj(x0, *arg):
    abar = arg[0]
    cs = arg[1]
    ds = arg[2]
    
    tmp = cs * np.power(1 + x0 / ds, abar) - cs
    y = -np.sum(tmp)
    return y


def constr(ps, B):
    return {
        'type': 'ineq',
        'fun': lambda x: B - np.sum(ps * x)
    }

def bounds(n):
    bnds = [(0, None) for _ in range(n)]
    return tuple(bnds)


def solveNominalProb(n, abar, cs, ds, ps, B):
    x0 = B / n * np.ones(n)
    res = minimize(
        obj, x0, args=(abar, cs, ds), bounds=bounds(n),
        method="trust-constr", constraints=constr(ps, B),
        options={'disp':True}
    )

    print(res.message)
    if res.success:
        print(res.x)
        print("objective", np.round(-res.fun, 3))
    else:
        print("Fail to solve nominal problem")


def obj2(x0, *arg):
    """
    x = (x, v, w, t, lbd)
        x   - n
        v   - n
        w   - n
        lbd - 1
    """
    abar = arg[0]
    cs = arg[1]
    ds = arg[2]
    gma = arg[3]
    n = arg[4]

    x = x0[:n]
    v = x0[n:2*n]
    w = x0[2*n:3*n]
    lbd = x0[-1]

    y = 1 + x / ds
    constrfunc = cs - v / np.log(y) * np.log(-v / cs / np.log(y)) + v / np.log(y)
    objfunc = np.sum(abar * v) + np.sum(w) + gma * lbd + np.sum(constrfunc)
    return objfunc


def constr2(n, abar, ps, B):
    """
    x = (x, v, w, t, lbd)
    """
    return (
        {'type': 'ineq', 'fun': lambda x: x[-1] + 0.25 * abar * x[n:2*n] + x[2*n : 3*n]},
        {'type': 'ineq', 'fun': lambda x: B - np.sum(ps * x[:n])}
    )


def bounds2(n):
    x = [(0, None) for _ in range(n)]
    v = [(None, 0) for _ in range(n)]
    w = [(0, None) for _ in range(n)]
    lbd = [(0, None)]
    return tuple(x + v + w + lbd)


def solveBudgetedSet(n, abar, cs, ds, ps, B, gma):
    x0 = np.zeros(3*n+1)
    x0[:n] = B / n
    x0[n:2*n] = -1
    x0[2*n:3*n] = 0.25
    res = minimize(
        obj2, x0, args=(abar, cs, ds, gma, n), bounds=bounds2(n),
        method="trust-constr", constraints=constr2(n, abar, ps, B),
        options={'disp':True, 'maxiter': 10000}
    )

    print(res.message)
    if res.success:
        print(np.round(res.x[:n],4))
        print("objective", np.round(-res.fun, 3))
    else:
        print("Fail to solve RO problem with budgeted uncertainty set")


def verify():
    n = 4
    abar = np.array([0.2, 0.1875, 0.1625, 0.15])
    cs = 30 * np.ones(n)
    ds = 1000 * np.ones(n)
    ps = 0.1 * np.array([1.1, 0.85, 0.9, 0.8])
    B = 1
    rho = -1
    ahat = 0.25 * abar
    Gamma = 1

    # solveNominalProb(n, abar, cs, ds, ps, B)
    solveBudgetedSet(n, abar, cs, ds, ps, B, Gamma)


def scaledProb(n):
    abar = np.random.uniform(0.15, 2.0, size=n)
    cs = 30 * np.ones(n)
    ds = 1000 * np.ones(n)
    ps = 0.1 * np.random.uniform(0.8, 1.15, size=n)
    B = 1
    Gamma = 1
    solveNominalProb(n, abar, cs, ds, ps, B)
    solveBudgetedSet(n, abar, cs, ds, ps, B, Gamma)


if __name__ == "__main__":
    # scaledProb(4)
    verify()
