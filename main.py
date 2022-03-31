"""
    run enumerical study
"""


import numpy as np

from uncertainty import BudgetUncertaintySet
from dualsubgradient import dualSubgradient


def case1():
    """verification case, Exercise 6.2"""
    n = 4
    abar = np.array([0.2, 0.1875, 0.1625, 0.15])
    cs = 30 * np.ones(n)
    ds = 1000 * np.ones(n)
    ps = 0.1 * np.array([1.1, 0.85, 0.9, 0.8])
    B = 1
    gamma = 1
    eps = 1e-6
    U = BudgetUncertaintySet(n, gamma, half=True)
    # solve
    fval, x = dualSubgradient(eps, U, abar, cs, ds, ps, B, disp=False)
    print("The robust with budget solution predicts a profit of {:.3f}".format(fval))
    
def case2(n):
    abar = np.random.uniform(0.15, 0.2, n)
    cs = 30 * np.ones(n)
    ds = 1000 * np.ones(n)
    ps = 0.1 * np.random.uniform(0.8, 1.1, n)
    B = 1
    gamma = 1
    eps = 1e-6
    U = BudgetUncertaintySet(n, gamma, half=True)
    fval, x = dualSubgradient(eps, U, abar, cs, ds, ps, B, disp=True)
    print("The robust with budget solution predicts a profit of {:.3f}".format(fval))


if __name__ == "__main__":
    # case1()
    # case2(4)
    case2(10)