"""
Exercise 6.2.3, norminal problem
The purpose is to test scipy.optimize.minimize
"""

import numpy as np
from scipy.optimize import minimize
from tool import build_problem


class adCampaignProblem():
    r"""
    Abstract class object for Ad Campaign problem.

    Parameters
    ----------
    abar : Sequence
        Parameter for calculating :math:`h(x)` function.
    cs : Sequence
        Parameter for calculating :math:`h(x)` function.
    ds : Sequence
        Parameter for calculating :math:`h(x)` function.
    ps : Sequence
        Daily price per ad insertion.
    B : float
        Total daily budget for ads.

    Methods
    -------
    ``objct_fnc()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``constraints()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``bound()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``solve_prob()``: Function to build and solve problem with ``scipy.optimize.minimize``.
    """

    def __init__(self, abar, cs, ds, ps, B):
        self.abar = abar
        self.n = abar.size
        self.cs = cs
        self.ds = ds
        self.ps = ps
        self.B = B

    def objct_fnc(self, x):
        pass

    def constraints(self):
        pass

    def bound(self):
        pass

    def solve_prob(self):
        pass


class nominalProblem(adCampaignProblem):
    r"""
    Class object for solving Ad Campaign nominal problem.

    Parameters
    ----------
    abar : Sequence
        Parameter for calculating :math:`h(x)` function.
    cs : Sequence
        Parameter for calculating :math:`h(x)` function.
    ds : Sequence
        Parameter for calculating :math:`h(x)` function.
    ps : Sequence
        Daily price per ad insertion.
    B : float
        Total daily budget for ads.

    Methods
    -------
    ``objct_fnc()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``constraints()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``bound()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``solve_prob()``: Function to build and solve problem with ``scipy.optimize.minimize``.
    """

    def __init__(self, abar, cs, ds, ps, B, xfixed=[], **kwargs):
        self.abar = abar
        self.n = len(abar)
        self.cs = cs
        self.ds = ds
        self.ps = ps
        self.B = B
        self.xfixed = np.array(xfixed)
        self._verbose = bool(kwargs.get('verbose', True))

    def objct_fnc(self, x):
        """
        Callable used to create objective function for the problem.

        Parameters
        ----------
        x : Sequence
            The variables passed on by ``scipy.optimize.minimize``.

        Returns
        -------
        y : expression
            The objective function expression to be evaluated by ``scipy.optimize.minimize``.
        """
        temp = ((self.cs * np.power((1 + (x / self.ds)), self.abar)) - self.cs)
        y = - np.sum(temp)

        return y

    def constraints(self):
        """
        Callable used to create constraints for the problem.

        Returns
        -------
        cts : tuple
            A tuple with the problem constraints to be evaluated by ``scipy.optimize.minimize``.
        """
        cts = [
            {'type': 'ineq',
             'fun': (lambda x: self.B - np.sum((self.ps * x)))},
            ]
        if ((len(self.xfixed) > 0) and (len(self.xfixed) == self.n)):
            cts += [
                {'type': 'eq',
                 'fun': (lambda x: self.xfixed - x)}
                ]

        return tuple(cts)

    def bounds(self):
        """
        Callable used to create bounds on variables for the problem.

        Returns
        -------
        bnds : tuple
            A tuple with the problem's variables bounds.
        """
        bnds = tuple([(0.0, None) for _ in range(self.n)])

        return bnds

    def solve_prob(self):
        """
        Function builds and solves the optimization problem using ``scipy.optimize.minimize``.

        Returns
        -------
        float
            The objective function value.
        Sequence
            The solution vector.
        """
        x0 = (self.B / (self.n * np.ones(self.n)))

        res = minimize(
            fun=self.objct_fnc,
            x0=x0,
            args=(),
            method='trust-constr',
            bounds=self.bounds(),
            constraints=self.constraints(),
            options={'disp': self._verbose}
            )
        if (res.success):
            print("Objective function value for nominal problem is:", np.round(-res.fun, 3))
            print("Solution vector for nominal problem is:", res.x)

        return -res.fun, res.x


class budgetedProblem(adCampaignProblem):
    r"""
    Class object for solving Ad Campaign robust optimization problem with budgeted uncertainty set.

    Parameters
    ----------
    abar : Sequence
        Parameter for calculating :math:`h(x)` function.
    cs : Sequence
        Parameter for calculating :math:`h(x)` function.
    ds : Sequence
        Parameter for calculating :math:`h(x)` function.
    ps : Sequence
        Daily price per ad insertion.
    B : float
        Total daily budget for ads.
    Gamma : float
        The :math:`\Gamma` parameter for the budgeted uncertainty set.
    **kwargs : any
        Additional parameters

    Methods
    -------
    ``objct_fnc()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``constraints()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``bound()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``solve_prob()``: Function to build and solve problem with ``scipy.optimize.minimize``.
    """

    def __init__(self, abar, cs, ds, ps, B, Gamma, xfixed=[], **kwargs):
        self.abar = abar
        self.n = len(abar)
        self.cs = cs
        self.ds = ds
        self.ps = ps
        self.B = B
        self.Gamma = Gamma
        self.xfixed = np.array(xfixed)
        self._verbose = bool(kwargs.get('verbose', True))

    def objct_fnc(self, var):
        """
        Callable used to create objective function for the problem.

        Parameters
        ----------
        x : Sequence
            The variables passed on by ``scipy.optimize.minimize``.

        Returns
        -------
        y : expression
            The objective function expression to be evaluated by ``scipy.optimize.minimize``.
        """
        x = var[:self.n]
        v = var[self.n:(2 * self.n)]
        w = var[(2 * self.n):(3 * self.n)]
        lbd = var[-1]

        y = (1 + (x / self.ds))

        constrfunc = (self.cs - ((v / np.log(y)) * np.log(- v / self.cs / np.log(y))) + (v / np.log(y)))
        objfunc = (np.sum(self.abar * v) + np.sum(w) + (self.Gamma * lbd) + np.sum(constrfunc))

        return objfunc

    def constraints(self):
        """
        Callable used to create constraints for the problem.

        Returns
        -------
        cts : tuple
            A tuple with the problem constraints to be evaluated by ``scipy.optimize.minimize``.
        """
        cts = [
            {'type': 'ineq',
             'fun': lambda x: (x[-1] + (0.25 * self.abar * x[self.n:(2 * self.n)]) + x[(2 * self.n):(3 * self.n)])},
            {'type': 'ineq',
             'fun': lambda x: (self.B - np.sum(self.ps * x[:self.n]))}
            ]

        return tuple(cts)

    def bounds(self):
        """
        Callable used to create bounds on variables for the problem.

        Returns
        -------
        bnds : tuple
            A tuple with the problem's variables bounds.
        """
        bnds = [(0, None) for _ in range(self.n)]  # x
        bnds += [(None, 0) for _ in range(self.n)]  # v
        bnds += [(0, None) for _ in range(self.n)]  # w
        bnds += [(0, None)]  # lambda

        return tuple(bnds)

    def solve_prob(self, **kwargs):
        """
        Function builds and solves the optimization problem using ``scipy.optimize.minimize``.

        Parameters
        ----------
        **kwargs : any
            Additional parameters used in the model construction. E.g., ``'maxiter'`` for the maximum number of func. calls.

        Returns
        -------
        float
            The objective function value.
        Sequence
            The solution vector.
        """
        x0 = np.zeros(((3 * self.n) + 1), dtype=float)
        x0[:self.n] = (self.B / self.n)  # Initializing `x` vars
        x0[self.n:(2 * self.n)] = -1  # Initializing `v` vars
        x0[(2 * self.n):(3 * self.n)] = 0.25  # Initializing `w` vars

        res = minimize(
            fun=self.objct_fnc,
            x0=x0,
            args=(),
            bounds=self.bounds(),
            method='trust-constr',
            constraints=self.constraints(),
            options={'disp': self._verbose,
                     'maxiter': int(kwargs.get('maxiter', 10000))}
            )
        if (res.success):
            print("Objective function value for budgeted problem is:", np.round(-res.fun, 4))
            print("Solution vector for budgeted problem is:", np.round(res.x[:self.n], 5))

        return np.round(-res.fun, 4), np.round(res.x[:self.n], 5)


def original_prob():
    (abar, size, cs, ds, ps) = build_problem(n=4, option='original')
    B = 1.0
    Gamma = 1.0

    ahat = 0.25 * abar

    nom_fval, nom_x = nominalProblem(abar, cs, ds, ps, B, verbose=False).solve_prob()
    box_fval, box_x = nominalProblem((abar - ahat), cs, ds, ps, B, verbose=False).solve_prob()
    bud_fval, bud_x = budgetedProblem(abar, cs, ds, ps, B, Gamma, verbose=False).solve_prob()

    # solveNominalProb(size, abar, cs, ds, ps, B)
    # solveBudgetedSet(size, abar, cs, ds, ps, B, Gamma)


def scaled_prob(n):
    (abar, size, cs, ds, ps) = build_problem(n=n, option='scaled')
    B = 1.0
    Gamma = 1.0

    ahat = 0.25 * abar

    nom_fval, nom_x = nominalProblem(abar, cs, ds, ps, B, verbose=False).solve_prob()
    box_fval, box_x = nominalProblem((abar - ahat), cs, ds, ps, B, verbose=False).solve_prob()
    bud_fval, bud_x = budgetedProblem(abar, cs, ds, ps, B, Gamma, verbose=False).solve_prob()
    
    # solveNominalProb(n, abar, cs, ds, ps, B)
    # solveBudgetedSet(n, abar, cs, ds, ps, B, Gamma)


if __name__ == "__main__":
    original_prob()
    # scaled_prob(4)
