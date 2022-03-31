"""
dual-subgradient epi-approximate robust optimization
"""

import numpy as np

from oracle import oracle
from uncertainty import BudgetUncertaintySet, CertaintySet


def dualSubgradient(eps, U, abar, cs, ds, ps, B, earlystop=True, disp=True):
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

    # max iter number
    itermax = 1e3
    # iterate count
    iter = 0
    # save x_t
    xs = [x]
    # output
    fmean = 0
    xmean = np.zeros(len(abar))

    while True:
        # update ahat
        ahat = abar * (1 - 0.25 * z)
        # call oracle
        fval, x = oracle(eps, ahat, cs, ds, ps, B)   
        # calulate \nabla_z f(x, z)
        g = cs * np.log(1 + x / ds) * np.power(1 + x / ds, ahat)
        # new z
        zprime = z + 1/np.sqrt(iter) * g 
        # update z
        z = U.project(zprime)
        # update iter
        iter = iter + 1
        
        # iterative average     
        fmeanprime = (iter - 1.0) / iter * fmean + fval / iter
        xmeanprime = (iter - 1.0) / iter * xmean + x / iter
        if disp: print("Objective value change: {:.6e}".format(fmeanprime - fmean))
        if iter > itermax: break
        # early stopping check
        if earlystop and np.fabs(fmeanprime - fmean) < eps: break
        # update fmean
        fmean = fmeanprime
        xmean = xmeanprime
        
    # return the average value
    print("number of iterations:", iter)
    return fmean, xmean


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

    fval, x = dualSubgradient(eps, U, abar, cs, ds, ps, B, disp=False)
    print(fval, np.round(x, 4))
