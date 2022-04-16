r"""
Script to check Dual-Subgradient convergence.

Call algorithm with different maximum number of iterations and check its convergence.
"""


import numpy as np
from approxRO.algorithm.uncertainty import BudgetUncertaintySet, CertaintySet, EllipsoidalUncertaintySet, CustomizedUncertaintySet
from approxRO.algorithm.dualsubgradient import dualSubgradient
from approxRO.tools.tool import build_problem
from joblib import Parallel, delayed
import time
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


def dualSubgradient_runtime(eps, U_set, size, option, B, seed=None, earlystop=True, itermax=1e3, display=False, get_x=False):
    """
    Function to call Dual-Subgradient algorithm for a given instance and calculate runtime.

    Parameters
    ----------
    eps : ``float``
        Algorithm accuracy.
    U_set : ``uncertainty.UncertaintySetBase``
        Uncertainty set object.
    size : ``int``
        The problem instance size.
    option : ``str``
        The option to generate data (either 'original' or 'scaled').
    B : ``float``
        The daily budget for ads.
    seed : Union[int, None], optional
        The seed for the default random number generator for reproducibility. The default is ``None``.
    earlystop : ``bool``, optional
        Whether or not to allow stopping before maximum number of iterations. The default is ``True``.
    itermax : ``int``, optional
        The maximum number of iterations the algorithm can run. The default is ``1e3``.
    display : ``bool``, optional
        Whether or not to display the algorithm progress at each iteration. The default is ``True``.
    get_x : ``bool``, optional
        Whether or not to get solution vectors for each iteration test. The default is ``True``.

    Returns
    -------
    ``numpy.ndarray``
        Sequence with objective value function (and solution vector if ``get_x=True``), number iterations and runtime.
    """
    (abar, _, cs, ds, ps) = build_problem(n=size,
                                          a_params=(0.15000, 0.20000),
                                          c_param=30.0,
                                          d_param=1000.0,
                                          p_params=(0.1, 0.800, 1.100),
                                          option=option,
                                          seed=seed)
    t_start = time.time()
    (fval, x, iters) = dualSubgradient(eps=eps,
                                       U=U_set,
                                       abar=abar,
                                       cs=cs,
                                       ds=ds,
                                       ps=ps,
                                       B=B,
                                       earlystop=earlystop,
                                       itermax=itermax,
                                       display=display,
                                       get_iters=True)
    elapsed_time = (time.time() - t_start)

    if (get_x):
        return np.array([fval, x, iters, elapsed_time])
    else:
        return np.array([fval, iters, elapsed_time], dtype=float)


def test_convergence(eps, U_set, size, option, B, seq, seed=None, get_x=True, n_jobs=-1, backend='multiprocessing',
                     verbose=25):
    r"""
    Driver function to call method ``dualSubgradient_convergence()`` using parallelization.

    Parameters
    ----------
    eps : ``float``
        Algorithm accuracy.
    U_set : ``uncertainty.UncertaintySetBase``
        Uncertainty set object.
    size : ``int``
        The problem instance size.
    option : ``str``
        The option to generate data (either 'original' or 'scaled').
    B : ``float``
        The daily budget for ads.
    seq : numpy.ndarray
        Sequence with maximum number of iterations to test.
    seed : Union[int, None], optional
        The seed for the default random number generator for reproducibility. The default is ``None``.
    get_x : ``bool``, optional
        Whether or not to get solution vectors for each iteration test. The default is ``True``.
    n_jobs : ``int``, optional
        The maximum number of concurrently running jobs (``-1`` means all the available). The default is ``-1``.
    backend : ``str``, optional
        parallelization backend implementation. The default is ``'multiprocessing'``.
    verbose : ``int``, optional
        The verbosity level. The default is ``25``.

    Returns
    -------
    ``numpy.ndarray``
        Sequence with number of iterations and objective value function (and solution vector if ``get_x=True``).
    """
    if (seed is None):
        seed = int(np.random.randint(low=((2 ** (31 - 1)) + 1), high=((2 ** 31) - 1), size=1))
    (abar, _, cs, ds, ps) = build_problem(n=size,
                                          a_params=(0.15000, 0.20000),
                                          c_param=30.0,
                                          d_param=1000.0,
                                          p_params=(0.1, 0.800, 1.100),
                                          option=option,
                                          seed=seed)
    converg = Parallel(n_jobs=n_jobs,
                       backend=backend,
                       verbose=verbose)(delayed(dualSubgradient_convergence)(it=it,
                                                                             eps=eps,
                                                                             U_set=U_set,
                                                                             abar=abar,
                                                                             cs=cs,
                                                                             ds=ds,
                                                                             ps=ps,
                                                                             B=B) for it in seq)

    if (get_x):
        return np.array(converg)
    else:
        converg = np.array(converg)
        return np.array(converg[:, :2], dtype=float)


def test_runtime(n_tests, eps, U_set, size, option, B, seed=None, earlystop=True, itermax=1e3, display=False, get_x=True,
                 n_jobs=-1, backend='multiprocessing', verbose=25):
    r"""
    Driver function to call method ``dualSubgradient_runtime()`` using parallelization.

    Parameters
    ----------
    n_tests : ``int``
        Number of times to run the algorithm.
    eps : ``float``
        Algorithm accuracy.
    U_set : ``uncertainty.UncertaintySetBase``
        Uncertainty set object.
    size : ``int``
        The problem instance size.
    option : ``str``
        The option to generate data (either 'original' or 'scaled').
    B : ``float``
        The daily budget for ads.
    seed : Union[list, None], optional
        The list of seeds for the default random number generator for reproducibility. If ``None`` is passed, a sequence
        of potential large prime numbers is generate for seed values. The default is ``None``.
    earlystop : ``bool``, optional
        Whether or not to allow stopping before maximum number of iterations. The default is ``True``.
    itermax : ``int``, optional
        The maximum number of iterations the algorithm can run. The default is ``1e3``.
    display : ``bool``, optional
        Whether or not to display the algorithm progress at each iteration. The default is ``True``.
    get_x : ``bool``, optional
        Whether or not to get solution vectors for each iteration test. The default is ``True``.
    n_jobs : ``int``, optional
        The maximum number of concurrently running jobs (``-1`` means all the available). The default is ``-1``.
    backend :str, optional
        parallelization backend implementation. The default is ``'multiprocessing'``.
    verbose : ``int``, optional
        The verbosity level. The default is ``25``.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if (seed is None):
        seed = np.random.randint(low=((2 ** (31 - 1)) + 1),
                                 high=((2 ** 31) - 1),
                                 size=n_tests)
    elif (len(seed) < n_tests):
        seed = np.array((list(seed) + np.random.randint(low=((2 ** (31 - 1)) + 1),
                                                        high=((2 ** 31) - 1),
                                                        size=(n_tests - len(seed))).tolist()), dtype=int)
    else:
        seed = np.array(seed, dtype=int)
    runtime = Parallel(n_jobs=n_jobs,
                       backend=backend,
                       verbose=verbose)(delayed(dualSubgradient_runtime)(eps=eps,
                                                                         U_set=U_set,
                                                                         size=size,
                                                                         option=option,
                                                                         B=B,
                                                                         seed=seed[i],
                                                                         earlystop=earlystop,
                                                                         itermax=itermax,
                                                                         display=display,
                                                                         get_x=get_x) for i in range(n_tests))

    return np.array(runtime)
