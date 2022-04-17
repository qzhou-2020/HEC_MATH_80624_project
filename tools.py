r"""
Helper functions to the implementation

    ``maxGrad()``
        Calculates :math:`\sup ||\nabla_{z} f(x, z)||_{2}`

    ``gradient()``
        Calculates :math:`||\nabla_{z} f(x, z)||_{2}`

    ``build_problem()``
        Builds a problem instance.

    ``generate_seed_sequence()``
        Generates a sequence of large numbers to be used as random number generator seeds.
"""


import numpy as np
from scipy.optimize import minimize


def maxGrad(abar, cs, ds, ps, B):
    r"""
    Function calculates the gradient  :math:`\sup||\nabla_{z} f(x, z)||_{2}`

    Parameters
    ----------
    abar : Sequence
        Parameter from problem.
    cs : Sequence
        Parameter from problem.
    ds : Sequence
        Parameter from problem.
    ps : Sequence
        Parameter from problem.
    B : float
        Parameter from problem.

    Returns
    -------
    float
        The gradient.
    """
    n = len(abar)
    obj = lambda x: - np.linalg.norm(
        (cs * np.log((1 + (x / ds))) * np.power((1 + (x / ds)), abar)), 2.0
        )
    bounds = tuple([(0, None) for _ in range(n)])
    constr = (
        {'type': 'ineq',
         'fun': lambda x: (B - np.sum((ps * x)))}
    )
    x0 = ((B / n) * np.ones(n, dtype=float))
    res = minimize(
        fun=obj,
        x0=x0,
        bounds=bounds,
        constraints=constr,
        method='trust-constr'
    )

    return - res.fun


def gradient(abar, cs, ds, xs):
    r"""
    Function calculates the gradient as

    .. math::
        \nabla_{z} f(x, z) = c_{i} \cdot \log(1 + x_{i} / d_{i}) \cdot \big(1 + x_{i} / d_{i} \big)^{a_{i}}

    Parameters
    ----------
    abar : Sequence
        Parameter from problem.
    cs : Sequence
        Parameter from problem.
    ds : Sequence
        Parameter from problem.
    xs : Sequence
        The solution vector.

    Returns
    -------
    grad : Sequence
        Vector of gradients.
    """
    grad = (cs * np.log((1 + (xs / ds))) * np.power((1 + (xs / ds)), abar))

    return grad


def build_problem(n=10, a_params=(0.15000, 0.20000), c_param=30.0, d_param=1000.0, p_params=(0.1, 0.800, 1.100),
                  option='original', seed=None):
    r"""
    Function generates the size if the data for the problem.

    Parameters
    ----------
    n : int, optional
        The size of problem's vectors. The default is ``10``.
    a_params : tuple, optional
        The [min, max] values of interval to generate values for vector ``a``. The default is ``(0.150, 0.200)``.
    c_param : float, optional
        The multiplier parameter :math:`c_{i}` in function :math:`h(x_{i})` (we assume the same for all values).
        The default is ``30.0``.
    d_param : float, optional
        The denominator parameter :math:`d_{i}` in function :math:`h(x_{i})` (we assume the same for all values).
        The default is ``1000.0``.
    p_params : tuple, optional
        The unit scaling and [min, max] values of interval to generate price values. The default is ``(0.10, 0.80, 1.10)``.
    option : str, optional
        The option to generate data (either ``'original'`` or ``'scaled'``). The default is ``'original'``.
    seed : Union[int, None], optional
        The seed for the default random number generator for reproducibility. The default is ``None``.

    Returns
    -------
    a_bar : Sequence
        Parameter for calculating :math:`h(x)` function.
    size : int
        The size of vectors.
    cs : Sequence
        Parameter for calculating :math:`h(x)` function.
    ds : Sequence
        Parameter for calculating :math:`h(x)` function.
    ps : Sequence
        Daily price per ad insertion.
    """
    a_bar = np.array([0.20000, 0.18750, 0.16250, 0.15000], dtype=float)
    size = len(a_bar)
    cs = (30.0 * np.ones(size, dtype=float))
    ds = (1000.0 * np.ones(size, dtype=float))
    ps = (0.1 * np.array([1.100, 0.850, 0.900, 0.800], dtype=float))

    if (option != 'original'):
        size = n
        rng = np.random.default_rng()
        if (seed is not None):
            rng = np.random.default_rng(seed)
        a_bar = rng.uniform(a_params[0], a_params[1], size=size)
        cs = (c_param * np.ones(size, dtype=float))
        ds = (d_param * np.ones(size, dtype=float))
        ps = (p_params[0] * rng.uniform(p_params[1], p_params[2], size=size))

    return (a_bar, size, cs, ds, ps)


def generate_seed_sequence(n_bits=16, init_seed=82561, n_seq=1):
    r"""
    Function generates a sequence of ``n_seq`` candidate prime numbers to be seeds for random number generators.

    Parameters
    ----------
    n_bits : ``int``, optional
        The number of bits :math:`b` for :math:`2^{b}`. The default is ``16``.
    init_seed : ``Union[int, None]``, optional
        The initial seed to generate seeds. ``None`` instantiates RNG based on NumPy. The default is ``82561``.
    n_seq : ``int``, optional
        The number of seeds to generate. The default is ``1``.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    rng = np.random.default_rng()
    if (init_seed is not None):
        rng = np.random.default_rng(init_seed)
    if (n_seq == 1):
        seed = int(rng.integers(low=((2 ** (n_bits - 1)) + 1), high=((2 ** n_bits) - 1), size=n_seq))
        if ((seed % 2) == 0):
            seed += 1

        return seed
    else:
        seeds = rng.integers(low=((2 ** (n_bits - 1)) + 1), high=((2 ** n_bits) - 1), size=n_seq)
        even = ((seeds % 2) == 0)
        seeds[even] += 1

        return seeds
