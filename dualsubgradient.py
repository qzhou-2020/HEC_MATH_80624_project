r"""
Code file contains the Dual-Subgradient Algorithm for :math:`epsilon`-approximate robust optimization.
"""

import numpy as np

from oracle import Oracle
from uncertainty import BudgetUncertaintySet, CertaintySet
from tool import maxGrad, gradient


def dualSubgradient(eps, U, abar, cs, ds, ps, B, earlystop=True, itermax=1e3, display=True):
    r"""
    Function implements an :math:`\epsilon`-approximate robust optimization solver based on dual-subgradient method.

    See `Ben-Tal et al. (2015) Oracle-Based Robust Optimization via Online Learning`__

    The consumer convertion is given by function :math:`h(x)`, such that,

    .. math::
        h(x_{i}) = c_{i} \cdot \Big(1 + \frac{x_{i}}{d_{i}} \Big)^{\bar{a}_{i}} - c_{i}

    where :math:`\bar{a}_{i} = a_{i} \cdot (1.0 - 0.25 \cdot z_{i})`.

    __ http://pubsonline.informs.org/doi/10.1287/opre.2015.1374

    Parameters
    ----------
    eps : float
        The :math:`\epsilon`-approximate tolerance.
    U : object
        Object of class type ``UncertaintySet``.
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

    g = gradient(ahat, cs, ds, x)  # Calulate :math:`\nabla_z f(x, z)`

    iter_count = 1  # Iteration counter

    xs = [x]  # Save :math:`x_{t}`

    # output
    fmean = 0
    xmean = np.zeros(len(abar))

    if (display):
        print(" Iter |    fval     |    fval_change ")

    while True:
        ahat = abar * (1 - 0.25 * z)  # Updata ``\hat{a}``
        fval, x = Oracle(eps, ahat, cs, ds, ps, B).solve()  # Call Oracle
        g = gradient(ahat, cs, ds, x)  # Calulate :math:`\nabla_z f(x, z)`
        z_prime = (z + ((1 / np.sqrt(iter_count)) * g))  # New ``z``
        z = U.project(z_prime)  # Update ``z``

        # Iterative average
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
