"""
    An epsilon-approximate optimization oracle 
    for the original optimization problem, with 
    a given set of uncertainty.
"""

import numpy as np
from scipy.optimize import minimize


def objective(x0, *args):
    """
        Objective function of the optimization problem. scipy.optimize.minimize
        require such a function to take x0 and several parameters, wrapped in 
        tuple arg.

            f(x) = sum_i c_i (1 + x_i / d_i)^a_i - c_i
        where
            a_i = abar_i (1 - z_i)

        Inputs:
            x0    ~ current iterate
            args  ~ parameters to define the objective function
                    args[0] ~ ahat_i
                    args[1] ~ c_i
                    args[2] ~ d_i
                    args[3] ~ eps
        Output:
            y     ~ real value
    """

    ahat = args[0]
    cs = args[1]
    ds = args[2]
    eps = args[3]

    tmp = cs * np.power(1 + x0 / ds, ahat) - cs
    return - np.sum(tmp) - eps

def constraints(ps, B):
    """
        generate constraints. Scipy.optimize.minimize requires a tuple of constraints,
        each of which is a dict with at least two keys, 'type' and 'fun'. 'type' can be
        either 'eq' or 'ineq'; 'fun' is a callable. Particularly, 'ineq' is non-negative.

        Inputs:
            ps  ~ vector of investment
            B   ~ total budget
        Outputs:
                ~ a tuple of constraints            
    """
    return (
        # sum_i p_i x_i <= B
        {'type': 'ineq', 'fun': lambda x: B - np.sum(ps * x)},
    )


def bounds(n):
    """
        generate lower and upper bounds for decision variables. 
        Scipy.optimize.minimize requires a tuple of (lb, ub). None indicates unbounded

            x_i >= 0, t is free

        Input:
            n ~ dimension of decision variable
        Output:
              ~ a tuple of bounds
    """

    bnds = [(0, None) for _ in range(n)]
    # bnds.append((None, None))
    return tuple(bnds)


def oracle(eps, ahat, cs, ds, ps, B):
    """
        An epsilon-approximate optimization oracle for the original 
        optimization problem, with a given set of uncertainty.

            min_x objective(x, *args)
            s.t.  constraints()
                  bounds()

        Inputs:
            z     ~ realized uncertainty value
            eps   ~ epsilon-approximate
            the rest inputs defines the optimization problem
        Output:
            x     ~ an epsilon-approximate solution
    """

    n = len(ahat)

    # select a initial iterate
    x0 = B / n * np.ones(n)

    # call solver
    res = minimize(
        objective, x0, args=(ahat, cs, ds, eps),
        method='trust-constr', bounds=bounds(n),
        constraints=constraints(ps, B),
        options={'disp': False}
    )

    if res.success:
        fval = -res.fun
        x = res.x
        return fval, x
    else:
        print(res.message)
        print('Fail to solve the oracle')
        return None, None


def test():
    n = 4
    abar = np.array([0.2, 0.1875, 0.1625, 0.15])
    cs = 30 * np.ones(n)
    ds = 1000 * np.ones(n)
    ps = 0.1 * np.array([1.1, 0.85, 0.9, 0.8])
    B = 1
    eps = 1e-4

    abar = abar * (1 - np.array([0.0539, 0.7472, 0.0259, 0.1730]))
    fval, x = oracle(eps, abar, cs, ds, ps, B)
    print(fval, x)


if __name__ == "__main__":
    test()