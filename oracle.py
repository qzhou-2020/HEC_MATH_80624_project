"""
An epsilon-approximate optimization Oracle 
for the original optimization problem, with 
a given set of uncertainty.
"""

import numpy as np
from scipy.optimize import minimize


class Oracle():
    r"""
    Creates an Oracle object that gererates and :math:`\epsilon`-approximate for the Ad Campaign problem.

    Parameters
    ----------
    eps : float
        The :math:`\epsilon` approximation tolerance.
    abar : Sequence
        List of parameters.
    cs : Sequence
        List of parameters.
    ds : Sequence
        List of parameters.
    ps : Sequence
        List of prices.
    B : float
        The total available budget.

    Methods
    -------
    ``objective_fnc()``

    ``constraints()``

    ``bounds()``

    ``solve()``
    """

    def __init__(self, eps, abar, cs, ds, ps, B):
        self.eps = eps
        self.abar = abar
        self.n = len(abar)
        self.cs = cs
        self.ds = ds
        self.ps = ps
        self.B = B

    def objective_fnc(self, x):
        r"""
        Objective function of the optimization problem.

        The scipy.optimize.minimize module requires a function to take :math:`x0` and several parameters, wrapped in
        tuple ``arg`` and use to construct the objective function

            .. math::
                f(x) = \sum_{i} \Big[ \big(c_{i} (1 + x_{i} / d_{i})^{a_{i}} \big) - c_{i} \Big]

        where :math:`a_{i} = \bar{a}_{i} (1 - z_{i})`.

        Parameters
        ----------
        x : Sequence
            The sequence of variables present in the objective function.

        Returns
        -------
        y : float
            The evaluation of the expression :math:`f(x)` at :math:`t^{th}` iteration.
        """
        tmp = ((self.cs * np.power((1 + (x / self.ds)), self.abar)) - self.cs)
        y = (- np.sum(tmp) - self.eps)

        return y

    def constraints(self):
        r"""
        Function generate constraints.

        Scipy.optimize.minimize module requires a tuple of constraints, each of which is a dict with
        at least two keys, ``'type'`` and ``'fun'``.

        * ``'type'`` can be either ``'eq'`` or ``'ineq'``. Particularly, ``'ineq'`` is non-negative, e.g., :math:`0 \leq b - Ax`

        * ``'fun'`` is a callable.

        Returns
        -------
        ct : tuple
            A tuple with the constraint mappings.
        """
        ct = ({'type': 'ineq',
              'fun': (lambda x: self.B - np.sum((self.ps * x)))},
              )
        return ct

    def bounds(self):
        r"""
        Function generates lower and upper bounds for decision variables.

        Scipy.optimize.minimize module requires a tuple of ``(lb, ub)`` for variables. ``None`` indicates unbounded, e.g.,

            * :math:`x_{i} \geq 0` => (0, None)
            * :math:`t \in \mathbb{R}` => (None, None)

        Returns
        -------
        bnds : tuple
            Tuple of tuples ``(lb, ub)`` with the lower and upper bounds for the :math:`n` variables.
        """
        bnds = tuple([(0.0, None) for _ in range(self.n)])

        return bnds

    def solve(self, display=False):
        r"""
        Function calls for Oracle solution.

        Generates an :math:`\epsilon`-approximate optimization oracle for the original optimization problem,
        with a given set of uncertainty.

        .. math::
            \min_{x} \  objective(x, *args)

            s.t.\ \ \  constraints()

                  bounds()

        Parameters
        ----------
        display : bool, optional
            Whether or not to display optimization information. The default is ``False``.

        Returns
        -------
        float
            The :math:`\epsilon`-approximate function value.
        Sequence
            The :math:`\epsilon`-approximate solution :math:`x^{t}`.

        """
        x0 = (self.B / (self.n * np.ones(self.n)))  # Select a initial iterate

        res = minimize(fun=self.objective_fnc,
                       x0=x0,
                       args=(),
                       method='trust-constr',
                       bounds=self.bounds(),
                       constraints=self.constraints(),
                       options={'disp': display}
                       )
        if (res.success):
            fval = - res.fun
            x = res.x

            return fval, x
        else:
            print(res.message)
            print('Fail to solve the oracle')

            return None, None
