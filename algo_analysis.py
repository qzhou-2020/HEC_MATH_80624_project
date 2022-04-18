r"""
Script with functions to perform tests on the Dual-Subgradient algorithm.

    ``case_study_verify()``
        Verification case, using data from Exercise 6.2 in `Delage, E. (2021)`__

        __ https://tintin.hec.ca/pages/erick.delage/MATH80624_LectureNotes.pdf

    ``solve_case_study()``
        Builds a problem instance and solves it with the Dual-Subgradient algorithm

    ``dualSubgradient_convergence()``
        Call algorithm with different maximum number of iterations and check its convergence.

    ``test_convergence()``
        Driver function to call method ``dualSubgradient_convergence()`` using parallelization.

    ``dualSubgradient_runtime()``
        Calls algorithm for a given instance and calculates runtime.

    ``test_runtime()``
        Driver function to call method ``dualSubgradient_runtime()`` using parallelization.

    ``solution_robustness()``
        Calculates the Ad Campaign problem solution value for fixed solution and uncertainty.

    ``test_robustness()``
        Driver function to call method ``solution_robustness()`` using parallelization.
"""


import numpy as np
from uncertainty import BudgetUncertaintySet, CertaintySet, EllipsoidalUncertaintySet, CustomizedUncertaintySet
from dualsubgradient import dualSubgradient
from tools import build_problem, generate_seed_sequence
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def case_study_verify(B=1.0, Gamma=1.0, eps=1e-6, display=False):
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


def solve_case_study(size=10, **kwargs):
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


def test_convergence(eps, size, B, seq, get_x=True, **kwargs):
    r"""
    Driver function to call method ``dualSubgradient_convergence()`` using parallelization.

    Parameters
    ----------
    eps : ``float``
        Algorithm accuracy.
    size : ``int``
        The problem instance size.
    B : ``float``
        The daily budget for ads.
    seq : ``numpy.ndarray``
        Sequence with maximum number of iterations to test.
    get_x : ``bool``, optional
        Whether or not to get solution vectors for each iteration test. The default is ``True``.
    **kwargs : ``Any``
        Several different parameters for the analysis.
            a_params : ``tuple``, optional
                Tuple with ``(low, high)`` for sampling interval. The default is ``(0.15000, 0.20000)``.
            c_param : ``float``, optional
                Parameter :math:`c_{i}`. The default is ``30.0``.
            d_param : ``float``, optional
                Parameter :math:`d_{i}`. The default is ``1000.0``.
            p_params : ``tuple``, optional
                Tuple with ``(scaling, low, high)`` for scaling price and sampling interval.
                The default is ``(0.1, 0.800, 1.100)``.
            seed : ``Union[int, None]``, optional
                The seed for the default random number generator for reproducibility. The default is ``None``.
            n_bits : ``int``, 16
                The number :math:`b` of bits for the calculating :math:`2^{b}`. The default is ``16``.
            option : ``str``, optional
                Whether to build original (``'original'``) or scaled (``'scaled'``) problem. The default is ``'scaled'``.
            U : ``str``, optional
                Which uncertainty set to use. The options are ``'budget'``, ``'ellipsoidal'``, ``'customized'`` and
                ``'certainty'``. The default is ``'budget'``.
            Gamma : ``float``, optional
                Parameter :math:`\Gamma` for Budget uncertainty set. The default is ``1.0``.
            half : ``bool``, optional
                Whether or not the Budget uncertainty set is in the interval ``[0, 1]``. The default is ``True``.
            n_store : ``int``, optional
                Number of feasible vectors :math:`z` to generate and store. The default is ``10``.
            rho : ``float``, optional
                Parameter :math:`\rho` for Ellipsoidal and Customized uncertainty sets. The default is ``1.0``.
            sigma : ``numpy.ndarray``, optional
                The matrix :math:`\Sigma` for the Ellipsoidal uncertainty set. The default is ``None``, meaning an identity
                matrix of size :math:`n \times n` (:math:`n=` ``size``) is created.
            decimals : ``int``, optional
                The number of decimals for the values in the random vector :math:`z\in\mathcal{Z}`. The default is ``9``.
            dist : ``str``, optional
                The type of distribution to generate the random vectors :math:`z`. Either ``'normal'`` or ``'uniform'`` for
                Budgeted Uncertainty set, and ``'uniform1'`` or ``'uniform2'`` for Customized Set. The default is
                ``'uniform'`` for Budgeted and ``'uniform2'`` for Customized.
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
    seed = kwargs.get('seed', None)
    if (seed is None):
        seed = generate_seed_sequence(n_bits=kwargs.get('n_bits', 16), init_seed=None, n_seq=1)

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
                                         n_store=kwargs.get('n_store', 10),
                                         decimals=kwargs.get('decimals', 9),
                                         dist=kwargs.get('dist', 'uniform2'))
    elif (uncertainty_set == 'certainty'):
        U_set = CertaintySet(size=size)
    else:
        U_set = BudgetUncertaintySet(size=size,
                                     Gamma=kwargs.get('Gamma', 1.0),
                                     half=kwargs.get('half', True),
                                     n_store=kwargs.get('n_store', 10),
                                     decimals=kwargs.get('decimals', 9),
                                     dist=kwargs.get('dist', 'uniform'))

    converg = Parallel(n_jobs=kwargs.get('n_jobs', -1),
                       backend=kwargs.get('backend', 'multiprocessing'),
                       verbose=kwargs.get('verbose', 25))(delayed(dualSubgradient_convergence)(it=it,
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


def test_runtime(n_tests, eps, size, option, B, seed=None, get_x=True, **kwargs):
    r"""
    Driver function to call method ``dualSubgradient_runtime()`` using parallelization.

    Parameters
    ----------
    n_tests : ``int``
        Number of times to run the algorithm.
    eps : ``float``
        Algorithm accuracy.
    size : ``int``
        The problem instance size.
    option : ``str``
        The option to generate data (either 'original' or 'scaled').
    B : ``float``
        The daily budget for ads.
    seed : Union[list, None], optional
        The list of seeds for the default random number generator for reproducibility. If ``None`` is passed, a sequence
        of potential large prime numbers is generate for seed values. The default is ``None``.
    get_x : ``bool``, optional
        Whether or not to get solution vectors for each iteration test. The default is ``True``.
    **kwargs : ``Any``
        Several different parameters for the analysis.
            n_bits : ``int``, 16
                The number :math:`b` of bits for the calculating :math:`2^{b}`. The default is ``16``.
            U : ``str``, optional
                Which uncertainty set to use. The options are ``'budget'``, ``'ellipsoidal'``, ``'customized'`` and
                ``'certainty'``. The default is ``'budget'``.
            Gamma : ``float``, optional
                Parameter :math:`\Gamma` for Budget uncertainty set. The default is ``1.0``.
            half : ``bool``, optional
                Whether or not the Budget uncertainty set is in the interval ``[0, 1]``. The default is ``True``.
            n_store : ``int``, optional
                Number of feasible vectors :math:`z` to generate and store. The default is ``10``.
            rho : ``float``, optional
                Parameter :math:`\rho` for Ellipsoidal and Customized uncertainty sets. The default is ``1.0``.
            sigma : ``numpy.ndarray``, optional
                The matrix :math:`\Sigma` for the Ellipsoidal uncertainty set. The default is ``None``, meaning an identity
                matrix of size :math:`n \times n` (:math:`n=` ``size``) is created.
            decimals : ``int``, optional
                The number of decimals for the values in the random vector :math:`z\in\mathcal{Z}`. The default is ``9``.
            dist : ``str``, optional
                The type of distribution to generate the random vectors :math:`z`. Either ``'normal'`` or ``'uniform'`` for
                Budgeted Uncertainty set, and ``'uniform1'`` or ``'uniform2'`` for Customized Set. The default is
                ``'uniform'`` for Budgeted and ``'uniform2'`` for Customized.
            earlystop : ``bool``, optional
                Whether or not to allow algorithm to stop before ``itermax`` is reached. The default is ``True``.
            itermax : ``int``, optional
                The maximum number of iterations of the algorithm. The default is ``1e3``.
            display : ``bool``, optional
                Whether or not to show the information at each iteration of the algorithm. The default is ``False``.
            n_jobs : ``int``, optional
                The maximum number of concurrently running jobs (``-1`` means all the available). The default is ``-1``.
            backend : ``str``, optional
                parallelization backend implementation. The default is ``'multiprocessing'``.
            verbose : ``int``, optional
                The verbosity level. The default is ``25``.

    Returns
    -------
    ``numpy.ndarray``
        Sequence with objective value function (and solution vector if ``get_x=True``), number of iterations and runtime.
    """
    if (seed is None):
        seed = generate_seed_sequence(n_bits=kwargs.get('n_bits', 16), init_seed=None, n_seq=n_tests)
    elif (len(seed) < n_tests):
        seed = np.array((list(seed)
                         + generate_seed_sequence(n_bits=16,
                                                  init_seed=None,
                                                  n_seq=(n_tests - len(seed))).tolist()), dtype=int)
    else:
        seed = np.array(seed, dtype=int)

    uncertainty_set = kwargs.get('U', 'budget').lower()

    if (uncertainty_set == 'ellipsoidal'):
        U_set = EllipsoidalUncertaintySet(size=size,
                                          rho=kwargs.get('rho', 1.0),
                                          sigma=kwargs.get('sigma', None),
                                          n_store=kwargs.get('n_store', 10))
    elif (uncertainty_set == 'customized'):
        U_set = CustomizedUncertaintySet(size=size,
                                         rho=kwargs.get('rho', 1.0),
                                         n_store=kwargs.get('n_store', 10),
                                         decimals=kwargs.get('decimals', 9),
                                         dist=kwargs.get('dist', 'uniform2'))
    elif (uncertainty_set == 'certainty'):
        U_set = CertaintySet(size=size)
    else:
        U_set = BudgetUncertaintySet(size=size,
                                     Gamma=kwargs.get('Gamma', 1.0),
                                     half=kwargs.get('half', True),
                                     n_store=kwargs.get('n_store', 10),
                                     decimals=kwargs.get('decimals', 9),
                                     dist=kwargs.get('dist', 'uniform'))

    runtime = Parallel(n_jobs=kwargs.get('n_jobs', -1),
                       backend=kwargs.get('backend', 'multiprocessing'),
                       verbose=kwargs.get('verbose', 25))(delayed(dualSubgradient_runtime)(eps=eps,
                                                                                           U_set=U_set,
                                                                                           size=size,
                                                                                           option=option,
                                                                                           B=B,
                                                                                           seed=seed[i],
                                                                                           earlystop=kwargs.get('earlystop',
                                                                                                                True),
                                                                                           itermax=kwargs.get('itermax',
                                                                                                              1e3),
                                                                                           display=kwargs.get('display',
                                                                                                              False),
                                                                                           get_x=get_x) for i in range(n_tests))

    return np.array(runtime)


def solution_robustness(abar, cs, ds, ps, B, eps, xs, zs):
    r"""
    Function calculates the Ad Campaign problem solution value for fixed solution and uncertainty.

    Given a solution vector :math:`` and an uncertain vector :math:``, the function calculates the ojjective
    function value for the Ad Campaign problem and whether or not the solution vector is feasible.

    Parameters
    ----------
    abar : ``numpy.ndarray``
        The vector with parameters :math:`\bar{a}_{i}`.
    cs : ``numpy.ndarray``
        The vector with parameters :math:`c_{i}`.
    ds : ``numpy.ndarray``
        The vector with parameters :math:`d_{i}`.
    ps : ``numpy.ndarray``
        The vector :math:`p` with prices for ad placements.
    B : ``float``
        The total daily budget for ads.
    eps : ``float``
        The target accuracy for the constraints.
    xs : ``numpy.ndarray``
        The solution vector :math:`x`.
    zs : ``numpy.ndarray``
        The vector with uncertain parameters :math:`z_{i}`.

    Returns
    -------
    ``list``
        A list ``[fval, is_feasible]`` with the objective function value and a ``float`` of whether (``1.0``) or
        not (``0.0``) the solution vector passed is feasible.

    Methods
    -------
    ``isfeasible_x(ps, B, xs)``
        Test if solution vector :math:`x` is feasible, i.e., :math:`x \geq 0` and :math:`p^{T}x \leq B`.
    ``get_ys(ds, xs)``
        Calculates :math:`y_{i} = \left( 1 + \frac{x_{i}}{d_{i}} \right)`, :math:`\forall{i \in I}`.
    ``get_ahat(abar, zs)``
        Calculates :math:`a_{i}(z_{i}) = \bar{a}_{1} \cdot (1 - 0.25 \cdot z_{i})`, :math:`\forall{i \in I}`.
    ``get_hs(ahat, cs, ys)``
        Calculates :math:`h_{i}(x_{i}) = (c_{i} \cdot y_{i}^{a_{i}}) - c_{i}`, :math:`\forall{i \in I}`.
    """
    def isfeasible_x(ps, B, eps, xs):
        r"""
        Helper function to test if solution vector :math:`x` is feasible.

        Parameters
        ----------
        ps : ``numpy.ndarray``
            The vector :math:`p` with prices for ad placements.
        B : ``numpy.ndarray``
            The total daily budget for ads.
        eps : ``float``
            The target accuracy for the constraints.
        xs : ``numpy.ndarray``
            The solution vector :math:`x` to be tested.

        Returns
        -------
        x_feasible : ``bool``
            Whether or not solution vector :math:`x` is feasible.
        """
        x_feasible = True
        if ((((ps @ xs) - B) > eps) or xs.any(where=(xs < 0.0))):
            x_feasible = False

        return float(x_feasible)

    def get_ys(ds, xs):
        r"""
        Helper function calculates :math:`y_{i} = \left( 1 + \frac{x_{i}}{d_{i}} \right)`, :math:`\forall{i \in I}`.

        Parameters
        ----------
        ds : ``numpy.ndarray``
            The vector with parameters :math:`d_{i}`.
        xs : ``numpy.ndarray``
            The solution vector :math:`x`.

        Returns
        -------
        ys : ``numpy.ndarray``
            The vector :math:`y`.
        """
        ys = (np.ones(xs.size, dtype=float) + (xs / ds))

        return ys

    def get_hs(ahat, cs, ys):
        r"""
        Helper function calculates :math:`h_{i}(x_{i}) = (c_{i} \cdot y_{i}^{a_{i}}) - c_{i}`, :math:`\forall{i \in I}`.

        Parameters
        ----------
        ahat : ``numpy.ndarray``
            The vector with parameters :math:`a_{i} = \bar{a}_{i}(1 - 0.25 \cdot z_{i})`.
        cs : ``numpy.ndarray``
            The vector with parameters :math:`c_{i}`.
        ys : ``numpy.ndarray``
            The vector :math:`y` with values defined as :math:`y_{i} = \left( 1 + \frac{x_{i}}{d_{i}} \right)`.

        Returns
        -------
        hs : TYPE
            DESCRIPTION.
        """
        hs = (np.multiply(cs, np.power(ys, ahat)) - cs)

        return hs

    def get_ahat(abar, zs):
        r"""
        Helper function calculates :math:`a_{i}(z_{i}) = \bar{a}_{1} \cdot (1 - 0.25 \cdot z_{i})`, :math:`\forall{i \in I}`.

        Parameters
        ----------
        abar : ``numpy.ndarray``
            The vector with parameters :math:`\bar{a}_{i}`.
        zs : ``numpy.ndarray``
            The vector with uncertain parameters :math:`z_{i}`.

        Returns
        -------
        ahat : ``numpy.ndarray``
            The vector with parameters :math:`a_{i}`.
        """
        zbar = (np.ones(zs.size, dtype=float) - (0.25 * zs))
        ahat = np.multiply(abar, zbar)

        return ahat

    x_feasible = isfeasible_x(ps, B, eps, xs)
    ys = get_ys(ds, xs)
    ahat = get_ahat(abar, zs)
    hs = get_hs(ahat, cs, ys)
    obj_val = np.sum(hs)

    return [obj_val, x_feasible]


def test_robustness(xs, n_tests, option, B, seed, eps=0.0, **kwargs):
    r"""
    Driver function to call method ``solution_robustness()`` using parallelization.

    Parameters
    ----------
    xs : ``numpy.ndarray``
        The solution vector :math:`x` to be tested.
    n_tests : ``int``
        Number of times to run the algorithm.
    option : ``str``
        The option to generate data (either 'original' or 'scaled').
    B : ``float``
        The daily budget for ads.
    seed : ``int``
        The seed for the default random number generator for reproducibility.
    eps : ``float``, optional
        The target accuracy for the solution from algorithm. The default value is ``0.0``.
    **kwargs : ``Any``
        Several different parameters for the analysis.
            a_params : ``tuple``, optional
                Tuple with ``(low, high)`` for sampling interval. The default is ``(0.15000, 0.20000)``.
            c_param : ``float``, optional
                Parameter :math:`c_{i}`. The default is ``30.0``.
            d_param : ``float``, optional
                Parameter :math:`d_{i}`. The default is ``1000.0``.
            p_params : ``tuple``, optional
                Tuple with ``(scaling, low, high)`` for scaling price and sampling interval.
                The default is ``(0.1, 0.800, 1.100)``.
            U : ``str``, optional
                Which uncertainty set to use. The options are ``'budget'``, ``'ellipsoidal'``, ``'customized'`` and
                ``'certainty'``. The default is ``'budget'``.
            Gamma : ``float``, optional
                Parameter :math:`\Gamma` for Budget uncertainty set. The default is ``1.0``.
            half : ``bool``, optional
                Whether or not the Budget uncertainty set is in the interval ``[0, 1]``. The default is ``True``.
            n_store : ``int``, optional
                Number of feasible vectors :math:`z` to generate and store. The default is ``10``.
            rho : ``float``, optional
                Parameter :math:`\rho` for Ellipsoidal and Customized uncertainty sets. The default is ``1.0``.
            sigma : ``numpy.ndarray``, optional
                The matrix :math:`\Sigma` for the Ellipsoidal uncertainty set. The default is ``None``, meaning an identity
                matrix of size :math:`n \times n` (:math:`n=` ``size``) is created.
            decimals : ``int``, optional
                The number of decimals for the values in the random vector :math:`z\in\mathcal{Z}`. The default is ``9``.
            dist : ``str``, optional
                The type of distribution to generate the random vectors :math:`z`. Either ``'normal'`` or ``'uniform'`` for
                Budgeted Uncertainty set, and ``'uniform1'`` or ``'uniform2'`` for Customized Set. The default is
                ``'uniform'`` for Budgeted and ``'uniform2'`` for Customized.
            n_jobs : ``int``, optional
                The maximum number of concurrently running jobs (``-1`` means all the available). The default is ``-1``.
            backend : ``str``, optional
                parallelization backend implementation. The default is ``'multiprocessing'``.
            verbose : ``int``, optional
                The verbosity level. The default is ``25``.

    Raises
    ------
    ValueError
        An integer ``seed`` value different than ``None`` must be passed. If ``None`` is passed, a ``ValueError`` is thrown.

    Returns
    -------
    ``numpy.ndarray``
        A matrix with number of rows equal to ``n_tests``. The irst column displays the objective function value for the
        problem with the solution vector ``xs`` passed and a random vector :math:`z \in \mathcal{Z}`. The second columns
        show either ``1.0``, if the vector ``xs`` respects all constraints
    """
    if (seed is None):
        raise ValueError("To run this test, the same seed used to build instace for soltuion vector must be passed...")

    (abar, size, cs, ds, ps) = build_problem(n=xs.size,
                                             a_params=kwargs.get('a_params', (0.15000, 0.20000)),
                                             c_param=kwargs.get('c_param', 30.0),
                                             d_param=kwargs.get('d_param', 1000.0),
                                             p_params=kwargs.get('p_params', (0.1, 0.800, 1.100)),
                                             option=option,
                                             seed=seed)

    uncertainty_set = kwargs.get('U', 'budget').lower()

    if (uncertainty_set == 'ellipsoidal'):
        U_set = EllipsoidalUncertaintySet(size=size,
                                          rho=kwargs.get('rho', 1.0),
                                          sigma=kwargs.get('sigma', None),
                                          n_store=kwargs.get('n_store', 10))
    elif (uncertainty_set == 'customized'):
        U_set = CustomizedUncertaintySet(size=size,
                                         rho=kwargs.get('rho', 1.0),
                                         n_store=kwargs.get('n_store', 10),
                                         decimals=kwargs.get('decimals', 9),
                                         dist=kwargs.get('dist', 'uniform2'))
    elif (uncertainty_set == 'certainty'):
        U_set = CertaintySet(size=size)
    else:
        U_set = BudgetUncertaintySet(size=size,
                                     Gamma=kwargs.get('Gamma', 1.0),
                                     half=kwargs.get('half', True),
                                     n_store=kwargs.get('n_store', 10),
                                     decimals=kwargs.get('decimals', 9),
                                     dist=kwargs.get('dist', 'uniform'))

    robustness = Parallel(n_jobs=kwargs.get('n_jobs', -1),
                          backend=kwargs.get('backend', 'multiprocessing'),
                          verbose=kwargs.get('verbose', 25))(delayed(solution_robustness)(abar=abar,
                                                                                          cs=cs,
                                                                                          ds=ds,
                                                                                          ps=ps,
                                                                                          B=B,
                                                                                          eps=eps,
                                                                                          xs=xs,
                                                                                          zs=U_set.get()) for i in range(n_tests))    

    return np.array(robustness, dtype=float)
