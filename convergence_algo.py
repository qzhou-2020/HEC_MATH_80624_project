r"""
Script to check Dual-Subgradient convergence.

Call algorithm with different maximum number of iterations and check its convergence.
"""


import numpy as np
from uncertainty import BudgetUncertaintySet, CertaintySet, EllipsoidalUncertaintySet, CustomizedUncertaintySet
from tool import build_problem
from dualsubgradient import dualSubgradient
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def dualSubgradient_convergence(it, eps, U_set, abar, cs, ds, ps, B):
    r"""
    Function to call Dual-Subgradient algorithm for a fixed number of iterations and test convergence.

    Parameters
    ----------
    it : ``int``
        Number of iterations.
    eps : ``float``
        Algorithm accuracy.
    U_set : ``uncertainty.UncertaintySetBase``
        Uncertainty set object.
    abar : ``numpy.ndarray``
        Parameters :math:`a_{i}` to calculate client conversion function :math:`h(x_{i})`.
    cs : ``numpy.ndarray``
        Parameters :math:`c_{i}` to calculate client conversion function :math:`h(x_{i})`.
    ds : ``numpy.ndarray``
        Parameters :math:`d_{i}` to calculate client conversion function :math:`h(x_{i})`.
    ps : ``numpy.ndarray``
        The daily prices :math:`p_{i}` per ad in website :math:`i`.
    B : ``float``
        The daily budget for ads.

    Returns
    -------
    ``list``
        List with ``[iteration, function value, solution vector]``.
    """
    (fval, x) = dualSubgradient(eps=eps,
                                U=U_set,
                                abar=abar,
                                cs=cs,
                                ds=ds,
                                ps=ps,
                                B=B,
                                earlystop=False,
                                itermax=it,
                                display=False)
    return [it, fval, x]


if __name__ == "__main__":
    (abar, size, cs, ds, ps) = build_problem(n=4,
                                             a_params=(0.15000, 0.20000),
                                             c_param=30.0,
                                             d_param=1000.0,
                                             p_params=(0.1, 0.800, 1.100),
                                             option='original',
                                             seed=None)
    U_set = BudgetUncertaintySet(size=size,
                                 Gamma=1.0,
                                 half=True,
                                 n_store=10)

    seq = np.array(([1] + [i for i in range(10, 501, 10)] + [i for i in range(525, 1001, 25)]), dtype=int)

    converg = Parallel(n_jobs=-1,
                       backend='multiprocessing',
                       verbose=25)(delayed(dualSubgradient_convergence)(it=it,
                                                                        eps=1e-6,
                                                                        U_set=U_set,
                                                                        abar=abar,
                                                                        cs=cs,
                                                                        ds=ds,
                                                                        ps=ps,
                                                                        B=1.0) for it in seq)
    converg = np.array(converg)
