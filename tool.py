"""
    Calculate the SUP norm(\nabla_z f(x, z), 2)
"""


import numpy as np
from scipy.optimize import minimize


def maxGrad(abar, cs, ds, ps, B):
    n = len(abar)
    obj = lambda x: - np.linalg.norm(
        cs * np.log(1 + x / ds) * np.power(1 + x / ds, abar),
        2
    )
    bounds = ([
        (0, None) for _ in range(n)
    ])
    constr = (
        {'type': 'ineq', 'fun': lambda x: B - np.sum(ps * x)}
    )
    x0 = B / n * np.ones(n)
    res = minimize(
        obj, x0, bounds=bounds, constraints=constr, method='trust-constr'
    )

    return - res.fun


if __name__ == "__main__":
    n = 4
    abar = np.array([0.2, 0.1875, 0.1625, 0.15])
    cs = 30 * np.ones(n)
    ds = 1000 * np.ones(n)
    ps = 0.1 * np.array([1.1, 0.85, 0.9, 0.8])
    B = 1

    maxg = maxGrad(abar, cs, ds, ps, B)
    print(maxg)