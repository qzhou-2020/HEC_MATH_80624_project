r"""
Implementation of Nonlinear Program

Ad Campaign Problem (Exercise 6.2 from Quantitative Risk Management Using Robust Optimization Lecture Notes__).
The purpose of this code is to test implement the problem and test it with ``scipy.optimize.minimize``.

__ https://tintin.hec.ca/pages/erick.delage/MATH80624_LectureNotes.pdf
"""


import numpy as np
from scipy.optimize import minimize
from tools import build_problem
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


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
        """Abstract method for calculating objective function"""
        pass

    def constraints(self):
        """Abstract method for defining constraints"""
        pass

    def bound(self):
        """Abstract method for defining variable bounds"""
        pass

    def solve_prob(self):
        """Abstract method for driving problem solution"""
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

    ``bounds()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

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
        obj : expression
            The objective function expression to be evaluated by ``scipy.optimize.minimize``.
        """
        temp = ((self.cs * np.power((1 + (x / self.ds)), self.abar)) - self.cs)
        obj = - np.sum(temp)

        return obj

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
            {'type': 'ineq',
             'fun': (lambda x: x[:self.n])},
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

    def solve_prob(self, **kwargs):
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
            tol=kwargs.get('tol', None),
            options={'disp': self._verbose,
                     'maxiter': int(kwargs.get('maxiter', 10000))}
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
    kwargs : any
        Additional parameters

    Methods
    -------
    ``objct_fnc()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``constraints()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``bounds()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

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
        r"""
        Callable used to create objective function for the problem.

        The :math:`g_{*}(x, v)` is defined as

        .. math::
            \sum_{i} \bigg( c_{i} -
            \frac{v_{i}}{\ln{y_{i}}}
                           \cdot \ln{\Big( \frac{-v_{i} / c_{i}}{\ln{y_{i}}} \Big)} + \frac{v_{i}}{\ln{y_{i}}} \bigg)

        where :math:`y_{i} = 1 + \frac{x_{i}}{d_{i}}`. Also, for the Budgeted uncertainty set, where

        .. math::
            \mathcal{Z} := \Big\{ z \in \mathbb{R}^{n} \ \Big\vert \ 0 \leq z \leq 1,\ \sum_{i}z_{i} \leq \Gamma,
                                 \ a_{i} = \bar{a}_{i}(1 - 0.25 z_{i}) \Big\}

        we have

        .. math::
            \delta^{*}(v|\mathcal{U}) = \min \bar{a}^{T}v + \sum_{i} w_{i} + \Gamma \lambda.

        Parameters
        ----------
        var : Sequence
            The variables passed on by ``scipy.optimize.minimize``.

        Returns
        -------
        obj : expression
            The objective function expression to be evaluated by ``scipy.optimize.minimize``.
        """
        x = var[:self.n]
        v = var[self.n:(2 * self.n)]
        w = var[(2 * self.n):(3 * self.n)]
        lbd = var[-1]

        y = (1 + (x / self.ds))

        g_xv = np.sum((self.cs - ((v / np.log(y)) * np.log((- (v / self.cs) / np.log(y)))) + (v / np.log(y))))

        d_vz = (np.sum(self.abar * v) + np.sum(w) + (self.Gamma * lbd))

        obj = (d_vz + g_xv)

        return obj

    def constraints(self):
        r"""
        Callable used to create constraints for the problem. For the presente model, we have the constraints

        .. math::
            \begin{cases}
                B - \sum_{i} p_{i} x_{i} \geq 0,\\
                \lambda + 0.25\cdot\mathbf{diag}(\bar{a})v + w \geq 0,\\
                x \geq 0,\\
                - v \geq 0,\\
                w \geq 0,\\
                \lambda \geq 0
            \end{cases}

        Returns
        -------
        cts : tuple
            A tuple with the problem constraints to be evaluated by ``scipy.optimize.minimize``.
        """
        cts = [
            {'type': 'ineq',
             'fun': lambda x: (self.B - np.sum(self.ps * x[:self.n]))},
            {'type': 'ineq',
             'fun': lambda x: (x[-1] + (0.25 * self.abar * x[self.n:(2 * self.n)]) + x[(2 * self.n):(3 * self.n)])},
            {'type': 'ineq',
             'fun': lambda x: x[:self.n]},
            {'type': 'ineq',
             'fun': lambda x: (- x[self.n:(2 * self.n)])},
            {'type': 'ineq',
             'fun': lambda x: x[(2 * self.n):(3 * self.n)]},
            {'type': 'ineq',
             'fun': lambda x: x[-1]},
            ]
        if ((len(self.xfixed) > 0) and (len(self.xfixed) == self.n)):
            cts += [
                {'type': 'eq',
                 'fun': (lambda x: self.xfixed - x)}
                ]

        return tuple(cts)

    def bounds(self):
        r"""
        Callable used to create bounds on variables for the problem.

        Returns
        -------
        bnds : tuple
            A tuple with the problem's variables bounds.
        """
        bnds = [(0, None) for _ in range(self.n)]  # :math:`x`
        bnds += [(None, 0) for _ in range(self.n)]  # :math:`v`
        bnds += [(0, None) for _ in range(self.n)]  # :math:`w`
        bnds += [(0, None)]  # :math:`\lambda`

        return tuple(bnds)

    def solve_prob(self, **kwargs):
        r"""
        Function builds and solves the optimization problem using ``scipy.optimize.minimize``.

        For the Budgeted uncertainty set, we have the optimization model

        .. math::
            &\underset{x, y, t, \lambda, v, w}{\text{minimize}} \ \ - t

            \text{S.t.:} \ \ \ \ &t + \bar{a}^{T}v + \sum_{i}w_{i} + \Gamma\lambda
            + \sum_{i} \bigg( c_{i} - \frac{v_{i}}{\ln{y_{i}}} \ln{\Big( \frac{-v_{i} / c_{i}}{\ln{y_{i}}} \Big) }
                                                                        + \frac{v_{i}}{\ln{y_{i}}} \bigg) \leq 0

            &y_{i} = 1 + \frac{x_{i}}{d_{i}}

            &\sum_{i} p_{i} x_{i} \leq B

            &\lambda \geq -0.25 \mathbf{diag}(\bar{a})v - w

            &x_{i}, w_{i}, \lambda \geq 0

            &v \leq 0

        Parameters
        ----------
        kwargs : any
            Additional parameters used in the model construction. E.g., ``'maxiter'`` for the maximum number of func. calls.

        Returns
        -------
        float
            The objective function value.
        Sequence
            The solution vector.
        """
        x0 = np.zeros(((3 * self.n) + 1), dtype=float)
        x0[:self.n] = (self.B / self.n)  # Initializing :math:`x` vars
        x0[self.n:(2 * self.n)] = -1  # Initializing :math:`v` vars
        x0[(2 * self.n):(3 * self.n)] = 0.25  # Initializing :math:`w` vars

        res = minimize(
            fun=self.objct_fnc,
            x0=x0,
            args=(),
            bounds=self.bounds(),
            method='trust-constr',
            constraints=self.constraints(),
            tol=kwargs.get('tol', None),
            options={'disp': self._verbose,
                     'maxiter': int(kwargs.get('maxiter', 10000))}
            )
        if (res.success):
            print("Objective function value for budgeted problem is:", np.round(-res.fun, 5))
            print("Solution vector for budgeted problem is:", np.round(res.x[:self.n], 5))

        return np.round(-res.fun, 5), np.round(res.x[:self.n], 5)


class ellipsoidalProblem(adCampaignProblem):
    r"""
    Class object for solving Ad Campaign robust optimization problem with ellipsoidal uncertainty set.

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
    rho : float
        The :math:`\rho` parameter for the ellipsoidal uncertainty set.
    kwargs : any
        Additional parameters

    Methods
    -------
    ``objct_fnc()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``constraints()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``bounds()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``solve_prob()``: Function to build and solve problem with ``scipy.optimize.minimize``.
    """

    def __init__(self, abar, cs, ds, ps, B, rho, xfixed=[], **kwargs):
        self.abar = abar
        self.n = len(abar)
        self.cs = cs
        self.ds = ds
        self.ps = ps
        self.B = B
        self.rho = rho
        self.xfixed = np.array(xfixed)
        self._verbose = bool(kwargs.get('verbose', True))

    def objct_fnc(self, var):
        r"""
        Callable used to create objective function for the problem.

        The :math:`g_{*}(x, v)` is defined as

        .. math::
            \sum_{i} \bigg( c_{i} -
            \frac{v_{i}}{\ln{y_{i}}}
                           \cdot \ln{\Big( \frac{-v_{i} / c_{i}}{\ln{y_{i}}} \Big)} + \frac{v_{i}}{\ln{y_{i}}} \bigg)

        where :math:`y_{i} = 1 + \frac{x_{i}}{d_{i}}`. Also, for the Ellipsoidal uncertainty set, where

        .. math::
            \mathcal{Z} := \Big\{ z \in \mathbb{R}^{n} \ \Big\vert \ ||z||_{2} \leq \rho,
                                 \ a_{i} = \bar{a}_{i}(1 - 0.25 z_{i}) \Big\}

        we have

        .. math::
            \delta^{*}(v\ |\ \mathcal{Z}) := \rho \cdot ||\mathbf{v}||_{2}

        and

        .. math::
            \delta^{*}(v\ |\ \mathcal{U}) &:= \bar{a}^{T}v + \delta^{*}(-0.25\cdot\textbf{diag}(\bar{a})v\ |\ \mathcal{Z}).

            &= \bar{a}^{T}v + \rho||-0.25\cdot\textbf{diag}(\bar{a})v||_{2}

        Parameters
        ----------
        var : Sequence
            The variables passed on by ``scipy.optimize.minimize``.

        Returns
        -------
        obj : expression
            The objective function expression to be evaluated by ``scipy.optimize.minimize``.
        """
        x = var[:self.n]
        v = var[self.n:(2 * self.n)]

        y = (1 + (x / self.ds))

        g_xv = np.sum((self.cs - ((v / np.log(y)) * np.log((- (v / self.cs) / np.log(y)))) + (v / np.log(y))))

        d_vz = (np.sum(self.abar * v) + (self.rho * np.linalg.norm((- (0.25 * self.abar * v)), 2)))

        obj = (d_vz + g_xv)

        return obj

    def constraints(self):
        r"""
        Callable used to create constraints for the problem. For the presente model, we have the constraints

        .. math::
            \begin{cases}
                B - \sum_{i} p_{i} x_{i} \geq 0,\\
                %s - ||-0.25\cdot\textbf{diag}(\bar{a})v||_{2} \geq 0,\\
                \ x \geq 0,\\
                \ -v \geq 0,\\
                \lambda \geq 0
            \end{cases}

        Returns
        -------
        cts : tuple
            A tuple with the problem constraints to be evaluated by ``scipy.optimize.minimize``.
        """
        cts = [
            {'type': 'ineq',
             'fun': lambda x: (self.B - np.sum(self.ps * x[:self.n]))},
            # {'type': 'ineq',
            #  'fun': lambda x: (x[-1] - np.linalg.norm((- (0.25 * self.abar * x[self.n:(2 * self.n)])), 2))},
            {'type': 'ineq',
             'fun': lambda x: x[:self.n]},
            {'type': 'ineq',
             'fun': lambda x: (- x[self.n:(2 * self.n)])},
            ]
        if ((len(self.xfixed) > 0) and (len(self.xfixed) == self.n)):
            cts += [
                {'type': 'eq',
                 'fun': (lambda x: self.xfixed - x)}
                ]

        return tuple(cts)

    def bounds(self):
        r"""
        Callable used to create bounds on variables for the problem.

        Returns
        -------
        bnds : tuple
            A tuple with the problem's variables bounds.
        """
        bnds = [(0, None) for _ in range(self.n)]  # :math:`x`
        bnds += [(None, 0) for _ in range(self.n)]  # :math:`v`

        return tuple(bnds)

    def solve_prob(self, **kwargs):
        r"""
        Function builds and solves the optimization problem using ``scipy.optimize.minimize``.

        For the Ellpsoidal uncertainty set, we have the optimization model

        .. math::
            &\underset{x, y, t, \lambda, v}{\text{minimize}} \ \ - t

            \text{S.t.:} \ \ \ \ &t + \bar{a}^{T}v + \rho ||-0.25\cdot\textbf{diag}(\bar{a})v||_{2}
            + \sum_{i} \bigg( c_{i} - \frac{v_{i}}{\ln{y_{i}}} \ln{\Big( \frac{-v_{i} / c_{i}}{\ln{y_{i}}} \Big)}
                                                                        + \frac{v_{i}}{\ln{y_{i}}} \bigg) \leq 0

            &y_{i} = 1 + \frac{x_{i}}{d_{i}}

            &\sum_{i} p_{i} x_{i} \leq B

            %&s = ||-0.25\cdot\textbf{diag}(\bar{a})v||_{2}
            &x \geq 0

            &v \leq 0

            &\lambda \geq 0

        Parameters
        ----------
        kwargs : any
            Additional parameters used in the model construction. E.g., ``'maxiter'`` for the maximum number of func. calls.

        Returns
        -------
        float
            The objective function value.
        Sequence
            The solution vector.
        """
        x0 = np.zeros(((2 * self.n)), dtype=float)
        x0[:self.n] = (self.B / self.n)  # Initializing `x` vars
        x0[self.n:(2 * self.n)] = -1  # Initializing `v` vars

        res = minimize(
            fun=self.objct_fnc,
            x0=x0,
            args=(),
            bounds=self.bounds(),
            method='trust-constr',
            constraints=self.constraints(),
            tol=kwargs.get('tol', None),
            options={'disp': self._verbose,
                     'maxiter': int(kwargs.get('maxiter', 10000))}
            )
        if (res.success):
            print("Objective function value for ellipsoidal set problem is:", np.round(-res.fun, 5))
            print("Solution vector for ellipsoidal set problem is:", np.round(res.x[:self.n], 5))

        return np.round(-res.fun, 5), np.round(res.x[:self.n], 5)


class customizedSetProblem(adCampaignProblem):
    r"""
    Class object for solving Ad Campaign robust optimization problem with a customized uncertainty set.

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
    rho : float
        The :math:`\rho` parameter for the customized uncertainty set.
    kwargs : any
        Additional parameters

    Methods
    -------
    ``objct_fnc()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``constraints()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``bounds()``: Required function for building and solving problem with ``scipy.optimize.minimize``.

    ``solve_prob()``: Function to build and solve problem with ``scipy.optimize.minimize``.
    """

    def __init__(self, abar, cs, ds, ps, B, rho, xfixed=[], **kwargs):
        self.abar = abar
        self.n = len(abar)
        self.cs = cs
        self.ds = ds
        self.ps = ps
        self.B = B
        self.rho = rho
        self.xfixed = np.array(xfixed)
        self._verbose = bool(kwargs.get('verbose', True))

    def objct_fnc(self, var):
        r"""
        Callable used to create objective function for the problem.

        The :math:`g_{*}(x, v)` is defined as

        .. math::
            \sum_{i} \bigg(  c_{i} -
            \frac{v_{i}}{\ln{y_{i}}}
                           \cdot \ln{\Big( \frac{-v_{i} / c_{i}}{\ln{y_{i}}} \Big)} + \frac{v_{i}}{\ln{y_{i}}} \bigg)

        where :math:`y_{i} = 1 + \frac{x_{i}}{d_{i}}`. Also, for the customized uncertainty set, where

        .. math::
            \mathcal{Z} := \Big\{ z \in \mathbb{R}^{n} \ \Big\vert \ z \geq 0,\ \sum_{i} z_{i} = 1,
                             \ \sum_{i} z_{i} \ln(z_{i}) \leq \rho,\ a_{i} = \bar{a}_{i}(1 - 0.25 z_{i}) \Big\}

        we have

        .. math::
            \delta^{*}(v \ | \ \mathcal{Z}) := \lambda + \sum_{i} \mu \exp \Big( \frac{w_{2i}}{\mu} - 1 \Big) + \rho\mu

        and

        .. math::
            \delta^{*}(v \ |\ \mathcal{U}) &:= \bar{a}^{T}v + \delta^{*}(-0.25 \text{diag}(\bar{a})v \ \vert\ \mathcal{Z})

            &:= \min \bar{a}^{T}v + \lambda + \sum_{i} \mu \exp \Big( \frac{w_{2i}}{\mu} - 1 \Big) + \rho\mu

        with :math:`\lambda,\mu \in \mathbb{R}`, :math:`\lambda \geq -0.25 \text{diag}(\bar{a})v - w_{2}` and
        :math:`\mu \geq 0`.

        Parameters
        ----------
        var : Sequence
            The variables passed on by ``scipy.optimize.minimize``.

        Returns
        -------
        obj : expression
            The objective function expression to be evaluated by ``scipy.optimize.minimize``.
        """
        x = var[:self.n]
        v = var[self.n:(2 * self.n)]
        w2 = var[(2 * self.n):(3 * self.n)]
        lbd = var[-2]
        mu = var[-1]

        y = (1.0 + (x / self.ds))

        g_xv = np.sum((self.cs - ((v / np.log(y)) * np.log((- (v / self.cs) / np.log(y)))) + (v / np.log(y))))

        d_vz = (np.sum(self.abar * v) + lbd + np.sum((mu * np.exp(((w2 / mu) - 1)))) + (self.rho * mu))

        obj = (d_vz + g_xv)

        return obj

    def constraints(self):
        r"""
        Callable used to create constraints for the problem. For the presente model, we have

        .. math::
            \begin{cases}
                B - \sum_{i} p_{i} x_{i} \geq 0,\\
                %s - \lambda - \sum_{i} \mu \exp\Big(\frac{w_{2i}}{\mu - 1}\Big) - \rho\mu \geq 0,\\
                \lambda + 0.25 \cdot \mathbf{diag}(\bar{a})v + w \geq 0,\\
                \ x \geq 0,\\
                \mu \geq 0,\\
                \ - v \geq 0,
            \end{cases}

        Returns
        -------
        cts : tuple
            A tuple with the problem constraints to be evaluated by ``scipy.optimize.minimize``.
        """
        cts = [
            {'type': 'ineq',
             'fun': lambda x: (self.B - np.sum(self.ps * x[:self.n]))},
            {'type': 'ineq',
             'fun': lambda x: (x[-2] + (0.25 * self.abar * x[self.n:(2 * self.n)]) + x[(2 * self.n):(3 * self.n)])},
            {'type': 'ineq',
             'fun': lambda x: x[:self.n]},
            {'type': 'ineq',
             'fun': lambda x: x[-1]},
            {'type': 'ineq',
             'fun': lambda x: (- x[self.n:(2 * self.n)])}
            ]
        if ((len(self.xfixed) > 0) and (len(self.xfixed) == self.n)):
            cts += [
                {'type': 'eq',
                 'fun': lambda x: (self.xfixed - x)}
                ]

        return tuple(cts)

    def bounds(self):
        r"""
        Callable used to create bounds on variables for the problem.

        Returns
        -------
        bnds : tuple
            A tuple with the problem's variables bounds.
        """
        bnds = [(0.0, None) for _ in range(self.n)]  # :math:`x`
        bnds += [(None, 0.0) for _ in range(self.n)]  # :math:`v`
        bnds += [(None, None) for _ in range(self.n)]  # :math:`w`
        bnds += [(None, None)]  # :math:`\lambda`
        bnds += [(0.0, None)]  # :math:`\mu`

        return tuple(bnds)

    def solve_prob(self, **kwargs):
        r"""
        Function builds and solves the optimization problem using ``scipy.optimize.minimize``.

        For the Customized Uncertainty set, we have the optimization model

        .. math::
            &\underset{x, y, t, s, \lambda, \mu, v, w}{\text{minimize}} \ \ - t

            \text{S.t.:} \ \ \ \ &t + \bar{a}^{T}v + s
            + \sum_{i} \bigg( c_{i} - \frac{v_{i}}{\ln{y_{i}}} \ln{\Big( \frac{-v_{i} / c_{i}}{\ln{y_{i}}} \Big)}
                                                                        + \frac{v_{i}}{\ln{y_{i}}} \bigg) \leq 0

            &y_{i} = 1 + \frac{x_{i}}{d_{i}}

            &\sum_{i} p_{i} x_{i} \leq B

            &s = \lambda + \sum_{i} \mu e^{\left( \frac{w_{i}}{\mu} -1 \right)} + \rho\mu

            &\lambda \geq -0.25 \mathbf{diag}(\bar{a})v - w

            &x,\mu \geq 0

            &v \leq 0

        Parameters
        ----------
        kwargs : any
            Additional parameters used in the model construction. E.g., ``'maxiter'`` for the maximum number of func. calls.

        Returns
        -------
        float
            The objective function value.
        Sequence
            The solution vector.
        """
        x0 = np.zeros(((3 * self.n) + 2), dtype=float)
        x0[:self.n] = (self.B / self.n)  # Initializing :math:`x` vars
        x0[self.n:(2 * self.n)] = -1  # Initializing :math:`v` vars
        x0[-1] = 0.25  # Initializing :math:`\mu` var

        res = minimize(
            fun=self.objct_fnc,
            x0=x0,
            args=(),
            bounds=self.bounds(),
            method='trust-constr',
            constraints=self.constraints(),
            tol=kwargs.get('tol', None),
            options={'disp': self._verbose,
                     'maxiter': int(kwargs.get('maxiter', 10000))}
            )
        if (res.success):
            print("Objective function value for customized set problem is:", np.round(-res.fun, 5))
            print("Solution vector for customized set problem is:", np.round(res.x[:self.n], 5))

        return np.round(-res.fun, 5), np.round(res.x[:self.n], 5)


def solve_rc_reformulation_ch6(**kwargs):
    r"""
    Function solves the Ad Campaign problem of `Exercise 6.2`__ under different uncertainty sets.

    __ https://tintin.hec.ca/pages/erick.delage/MATH80624_LectureNotes.pdf

    Parameters
    ----------
    **kwargs : any
        The parameters used to build and solve the problem.
            prob : str, optional
                Which problem to solve amongst the options ``'nominal'``, ``'box'``, ``'budget'``, ``'ellipsoidal'``
                and ``'customized'``. The default is ``'budget'``.
            n : int
                The number of websites to consider in the problem.
            option : str
                The type of the problem. Either ``'original'`` or ``'scaled'``.
            a_params : tuple, optional
                Tuple with ``(low, high)`` for sampling interval. The default is ``(0.15000, 0.20000)``.
            c_param : float, optional
                Parameter :math:`c_{i}`. The default is ``30.0``.
            d_param : float, optional
                Parameter :math:`d_{i}`. The default is ``1000.0``.
            p_params : tuple, optional
                Tuple with ``(scaling, low, high)`` for scaling price and sampling interval.
                The default is ``(0.1, 0.800, 1.100)``.
            B : float
                The daily budget amount for the ads.
            seed : int, optional
                The seed used in the ``numpy.random.Generator`` object for reproducibility. The default is ``None``.
            Gamma : float
                The :math:`\Gamma` parameter for the Budgeted uncertainty set.
            rho : float
                The :math:`\rho` parameter for both the Ellipsoidal and Customized uncertainty sets.
            xfixed : Sequence
                The values for the solution vector :math:`\mathbf{x}` if the values are to be fixed.
            verbose : bool
                Flag to whether or not show ``scipy.optimize.minimize`` information.
            tol : float
                The tolerance value for stopping solver.
            maxiter : int
                The maximum number of solver iterations.

    Returns
    -------
    results : tuple
        Tuple with the function value and solution vector for the uncertainty set chosen.
    """
    (abar, size, cs, ds, ps) = build_problem(n=int(kwargs.get('n', 4)),
                                             a_params=kwargs.get('a_params', (0.15000, 0.20000)),
                                             c_param=kwargs.get('c_param', 30.0),
                                             d_param=kwargs.get('d_param', 1000.0),
                                             p_params=kwargs.get('p_params', (0.1, 0.800, 1.100)),
                                             option=kwargs.get('option', 'original').lower(),
                                             seed=kwargs.get('seed', None))
    ahat = (0.25 * abar)
    problem = kwargs.get('prob', 'budget').lower()

    if (problem == 'nominal'):
        fval, x = nominalProblem(abar=abar,
                                 cs=cs,
                                 ds=ds,
                                 ps=ps,
                                 B=kwargs.get('B', 1.0),
                                 xfixed=kwargs.get('xfixed', []),
                                 verbose=kwargs.get('verbose', True)).solve_prob(tol=kwargs.get('tol', None),
                                                                                 maxiter=int(kwargs.get('maxiter', 10000)))
    elif (problem == 'box'):
        fval, x = nominalProblem(abar=(abar - ahat),
                                 cs=cs,
                                 ds=ds,
                                 ps=ps,
                                 B=kwargs.get('B', 1.0),
                                 xfixed=kwargs.get('xfixed', []),
                                 verbose=kwargs.get('verbose', True)).solve_prob(tol=kwargs.get('tol', None),
                                                                                 maxiter=int(kwargs.get('maxiter', 10000)))
    elif (problem == 'ellipsoidal'):
        fval, x = ellipsoidalProblem(abar=abar,
                                     cs=cs,
                                     ds=ds,
                                     ps=ps,
                                     B=kwargs.get('B', 1.0),
                                     rho=kwargs.get('rho', 1.0),
                                     xfixed=kwargs.get('xfixed', []),
                                     verbose=kwargs.get('verbose', True)).solve_prob(tol=kwargs.get('tol', None),
                                                                                     maxiter=int(kwargs.get('maxiter',
                                                                                                            10000)))
    elif (problem == 'customized'):
        fval, x = customizedSetProblem(abar=abar,
                                       cs=cs,
                                       ds=ds,
                                       ps=ps,
                                       B=kwargs.get('B', 1.0),
                                       rho=kwargs.get('rho', 1.0),
                                       xfixed=kwargs.get('xfixed', []),
                                       verbose=kwargs.get('verbose', True)).solve_prob(tol=kwargs.get('tol', None),
                                                                                       maxiter=int(kwargs.get('maxiter',
                                                                                                              10000)))
    else:
        fval, x = budgetedProblem(abar=abar,
                                  cs=cs,
                                  ds=ds,
                                  ps=ps,
                                  B=kwargs.get('B', 1.0),
                                  Gamma=kwargs.get('Gamma', 1.0),
                                  xfixed=kwargs.get('xfixed', []),
                                  verbose=kwargs.get('verbose', True)).solve_prob(tol=kwargs.get('tol', None),
                                                                                  maxiter=int(kwargs.get('maxiter', 10000)))

    # solveNominalProb(size, abar, cs, ds, ps, B)
    # solveBudgetedSet(size, abar, cs, ds, ps, B, Gamma)

    return (fval, x)


# if __name__ == "__main__":
#     solve_exercise_prob(n=4, option='original')
#     solve_exercise_prob(n=4, option='scaled')
