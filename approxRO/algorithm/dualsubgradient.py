r"""
Code file contains the Dual-Subgradient Algorithm for :math:`epsilon`-approximate robust optimization.
"""

import numpy as np

from approxRO.algorithm.oracle import Oracle
from approxRO.algorithm.uncertainty import BudgetUncertaintySet, CertaintySet, EllipsoidalUncertaintySet
from approxRO.tools.tool import maxGrad, gradient


def dualSubgradient(eps, U, abar, cs, ds, ps, B, earlystop=True, itermax=1e3, display=True):
    r"""
    Function implements an :math:`\epsilon`-approximate robust optimization solver based on dual-subgradient method.

    See `Ben-Tal et al. (2015) Oracle-Based Robust Optimization via Online Learning`__

    The method calculates :math:`\bar{x}` and :math:`\bar{f}(x)` for the Ad Campaign problem, where a given number
    :math:`x_{i}` of ads is placed in a website :math:`i` for a given price :math:`p_{i}`. The consumer convertion
    is given by the function :math:`h(x)`, such that,

    .. math::
        h(x_{i}) = c_{i} \cdot \Big(1 + \frac{x_{i}}{d_{i}} \Big)^{a_{i}} - c_{i}

    where :math:`a_{i} = \bar{a}_{i} \cdot (1.0 - 0.25 \cdot z_{i})`. The problem is then to maximize the consumer
    conversion given an ad daily budget :math:`B`

    .. math::
        \underset{x}{\text{maximize}} &\ \ \ \ \ h(x)

        \text{S.t.:} \ \ \ \ &h(x_{i}) = c_{i} \cdot y_{i}^{a_{i}} - c_{i}, && \forall{i}

        &y_{i} = 1 + \frac{x_{i}}{d_{i}}, && \forall{i}

        &\sum_{i} p_{i} x_{i} \leq B, &&

        &x_{i} \geq 0, && \forall{i}

    __ http://pubsonline.informs.org/doi/10.1287/opre.2015.1374

    Parameters
    ----------
    eps : float
        The :math:`\epsilon`-approximate tolerance.
    U : object
        Object of class type ``UncertaintySetBase``.
    abar : Sequence
        Parameter whose value is uncertain.
    cs : Sequence
        Parameter used to calculate :math:`h(x)` function.
    ds : Sequence
        Parameter used to calculate :math:`h(x)` function.
    ps : Sequence
        Prices for each daily ad insertion on website :math:`i`.
    B : float
        The total daily budget for the ad campaign.
    earlystop : bool, optional
        Whether or not to allow stopping before maximum number of iterations. The default is ``True``.
    itermax : int, optional
        The maximum number of iterations the algorithm can run. The default is ``1000``.
    display : bool, optional
        Whether or not to display the algorithm progress at each iteration. The default is ``True``.

    Returns
    -------
    fmean : float
        The mean function value.
    xmean : Sequence
        The mean solution vector.
    """
    z = U.get()  # Get a uncertainty vector realization

    # Call oracle to solve for ``x``
    ahat = (abar * (1.000 - (0.250 * z)))
    fval, x = Oracle(eps, ahat, cs, ds, ps, B).solve()

    g = gradient(ahat, cs, ds, x)  # Calculate :math:`\nabla_{z} f(x, z)`

    iter_count = 1  # Iteration counter

    xs = [x]  # Save :math:`x_{t}`

    # output
    fmean = 0
    xmean = np.zeros(len(abar))

    if (display):
        print(" Iter |    fval     |    fval_change ")

    while True:
        ahat = abar * (1 - 0.25 * z)  # Update :math:`\hat{a}`
        fval, x = Oracle(eps, ahat, cs, ds, ps, B).solve()  # Call Oracle
        g = gradient(ahat, cs, ds, x)  # Calculate :math:`\nabla_{z} f(x, z)`
        z_prime = (z + ((1 / np.sqrt(iter_count)) * g))  # New :math:`z`
        z = U.project(z_prime)  # Update :math:`z`

        # Iteration average
        fmeanprime = ((((iter_count - 1.0) / iter_count) * fmean) + (fval / iter_count))
        xmeanprime = ((((iter_count - 1.0) / iter_count) * xmean) + (x / iter_count))

        if (display):
            print(f"{iter_count:4d}    {fmeanprime:.4e}    {fmeanprime-fmean:.6e}")

        if (iter_count > itermax):  # Stopping after the maximum number of iterations has been reached
            break

        if (earlystop and (np.fabs(fmeanprime - fmean) < eps)):  # Early stopping check
            break

        # Update fmean
        fmean = fmeanprime
        xmean = xmeanprime

        iter_count += 1  # Increasing iteration counter

    if (display):
        print("Number of iterations:", iter_count)

    return fmean, xmean  # Return the mean values
