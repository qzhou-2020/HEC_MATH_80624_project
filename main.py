r"""
The main function driver to run numerical studies.
"""

import numpy as np
from uncertainty import BudgetUncertaintySet, CertaintySet, EllipsoidalUncertaintySet, CustomizedUncertaintySet
from tool import build_problem
from dualsubgradient import dualSubgradient
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def case_study_1(B=1.0, Gamma=1.0, eps=1e-6, display=False):
    r"""
    Verification case, using data from Exercise 6.2 in `Delage, E. (2021)`__

    __ https://tintin.hec.ca/pages/erick.delage/MATH80624_LectureNotes.pdf

    Parameters
    ----------
    B : float, optional
        The daily budget for the ads. The default is ``1.0``.
    Gamma : float, optional
        DESCRIPTION. The default is ``1.0``.
    eps : float, optional
        The :math:`\epsilon` parameter for the :math:`\epsilon`-approximate RO algorithm. The default is ``1e-6``.
    display : bool, optional
        Flag to whether or not show algorithm progress information. The default is ``False``.

    Returns
    -------
    tuple
        The value function and the solution vector.
    """
    (abar, size, cs, ds, ps) = build_problem(n=4, option='original')

    U = BudgetUncertaintySet(size=size, Gamma=Gamma, half=True)

    fval, x = dualSubgradient(eps=eps,
                              U=U,
                              abar=abar,
                              cs=cs,
                              ds=ds,
                              ps=ps,
                              B=B,
                              display=display)  # Solving with Dual-Subgradient algorithm

    print("The robust with budget solution predicts a profit of {:.4f}".format(fval))
    print("The solution is:", x)

    return (fval, x)


def build_case_study(size=10, **kwargs):
    r"""
    Function for creating any case study configuration.

    Parameters
    ----------
    size : int, optional
        The size of vectors. The default is ``10``.
    **kwargs : any
        Additional parameter information to build problem, uncertainty set and algorithm.
            a_params : tuple, optional
                Tuple with ``(low, high)`` for sampling interval. The default is ``(0.15000, 0.20000)``.
            c_param : float, optional
                Parameter :math:`c_{i}`. The default is ``30.0``.
            d_param : float, optional
                Parameter :math:`d_{i}`. The default is ``1000.0``.
            p_params : tuple, optional
                Tuple with ``(scaling, low, high)`` for scaling price and sampling interval.
                The default is ``(0.1, 0.800, 1.100)``.
            B : float, optional
                The daily ad budget :math:`B` for the problem. The default is ``1.0``.
            seed : int, optional
                The seed used in the ``numpy.random.Generator`` object for reproducibility. The default is ``None``.
            option : str, optional
                Whether to build original (``'original'``) or scaled (``'scaled'``) problem. The default is ``'scaled'``.
            eps : float, optional
                The :math:`\epsilon`-approximation for the Dual-Subgradient algorithm. The default is ``1e-6``.
            U : str, optional
                Which uncertainty set to use. The options are ``'budget'``, ``'ellipsoidal'``, ``'customized'`` and
                ``'certainty'``. The default is ``'budget'``.
            Gamma : float, optional
                Parameter :math:`\Gamma` for Budget uncertainty set. The default is ``1.0``.
            half : bool, optional
                Whether or not the Budget uncertainty set is in the interval ``[0, 1]``. The default is ``True``.
            n_store : int, optional
                Number of feasible vectors :math:`z` to generate and store. The default is ``10``.
            rho : float, optional
                Parameter :math:`\rho` for Ellipsoidal and Customized uncertainty sets. The default is ``1.0``.
            sigma : numpy.ndarray, optional
                The matrix :math:`\Sigma` for the Ellipsoidal uncertainty set. The default is ``None``, meaning an identity
                matrix of size :math:`n \times n` (:math:`n=` ``size``) is created.
            earlystop : bool, optional
                Whether or not to allow algorithm to stop before ``itermax`` is reached. The default is ``True``.
            itermax : int, optional
                The maximum number of iterations of the algorithm. The default is ``1e3``.
            display : bool, optional
                Whether or not to show the information at each iteration of the algorithm. The default is ``True``.

    Returns
    -------
    tuple
        The value function and the solution vector.
    """
    (abar, _, cs, ds, ps) = build_problem(n=size,
                                          a_params=kwargs.get('a_params', (0.15000, 0.20000)),
                                          c_param=kwargs.get('c_param', 30.0),
                                          d_param=kwargs.get('d_param', 1000.0),
                                          p_params=kwargs.get('p_params', (0.1, 0.800, 1.100)),
                                          option=kwargs.get('option', 'scaled').lower(),
                                          seed=kwargs.get('seed', None))

    uncertainty_set = kwargs.get('U', 'budget').lower()

    if (uncertainty_set == 'ellipsoidal'):
        U_set = EllipsoidalUncertaintySet(size=size,
                                          rho=kwargs.get('rho', 1.0),
                                          sigma=kwargs.get('sigma', None),
                                          n_store=kwargs.get('n_store', 10))
    elif (uncertainty_set == 'customized'):
        U_set = CustomizedUncertaintySet(size=size,
                                         rho=kwargs.get('rho', 1.0),
                                         n_store=kwargs.get('n_store', 10))
    elif (uncertainty_set == 'certainty'):
        U_set = CertaintySet(size=size)
    else:
        U_set = BudgetUncertaintySet(size=size,
                                     Gamma=kwargs.get('Gamma', 1.0),
                                     half=kwargs.get('half', True),
                                     n_store=kwargs.get('n_store', 10))

    fval, x = dualSubgradient(eps=kwargs.get('eps', 1e-6),
                              U=U_set,
                              abar=abar,
                              cs=cs,
                              ds=ds,
                              ps=ps,
                              B=kwargs.get('B', 1.0),
                              earlystop=kwargs.get('earlystop', True),
                              itermax=kwargs.get('itermax', 1e3),
                              display=kwargs.get('display', True))  # Solving with Dual-Subgradient algorithm

    print(("The robust optimization problem with " + uncertainty_set + " solution predicts a profit of {:.4f}").format(fval))
    print("The solution is:", x)

    return (fval, x)


if __name__ == "__main__":
    case_study_1()
    # build_case_study(4, option='scaled')
    build_case_study(size=10, option='scaled')
