"""
dual-subgradient epi-approximate robust optimization
"""

import numpy as np

from oracle import oracle
from uncertainty import BudgetUncertaintySet


def dualSubgradient(eps, U, abar, cs, ds, ps, B):
    """
        An epsilon-approximate robust optimization solver 
        based on dual-subgradient method.

        See Ben-Tal et al., 2015, oracle-based robust optimization via online learning

        Inputs:
            eps ~ accuracy parameter
            U   ~ uncertainty set
    """

    # get a uncertainty vector realization
    z = U.get()

    # call oracle to solve for x
    ahat = abar * (1.0 - 0.25 * z)
    fval, x = oracle(eps, ahat, cs, ds, ps, B)
    # calculate \nabla_z f(x, z)
    g = cs * np.log(1 + x / ds) * np.power(1 + x / ds, ahat)
    
    # for stopping
    g0 = g
    etol = 1e-6
    normg0 = np.linalg.norm(g0, 2)
    # max iter number
    # todo: should be determined properly
    itermax = 1e2

    # iterate count
    iter = 1
    
    # save x_t
    xs = [x]

    while True:
        # update z
        zprime = z + 1/np.sqrt(iter) * g
        
        # take project
        zprime = U.project(zprime)
        
        # how much z has moved
        cri = np.linalg.norm(zprime - z, 2)
        print(cri, fval)
        if iter > itermax or cri < etol:
            break

        z = zprime
        # new ahat
        ahat = abar * (1. - 0.25 * z)

        # solve for new x
        fval, x = oracle(eps, ahat, cs, ds, ps, B)
        xs.append(x)
        
        # update grad
        g = cs * np.log(1 + x / ds) * np.power(1 + x / ds, ahat)

        # update iter
        iter = iter + 1

    # return the average value
    print("number of iterations:", iter)
    return fval, np.mean(np.array(xs), axis=0)


if __name__ == "__main__":
    n = 4
    abar = np.array([0.2, 0.1875, 0.1625, 0.15])
    cs = 30 * np.ones(n)
    ds = 1000 * np.ones(n)
    ps = 0.1 * np.array([1.1, 0.85, 0.9, 0.8])
    B = 1
    gamma = 1
    eps = 1e-6

    U = BudgetUncertaintySet(n, gamma, half=True)

    fval, x = dualSubgradient(eps, U, abar, cs, ds, ps, B)
    print(fval, np.round(x, 4))
