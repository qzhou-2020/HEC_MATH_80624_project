"""
Define the objects for the different uncertainty sets available.
"""


import numpy as np
from scipy.optimize import minimize


class UncertaintySetBase(object):
    r"""
    Abstract base class for a uncertainty set objects

    Methods
    -------
    ``get()``: Returns an uncertainty vector realization.

    ``diam()``: Returns the diameter of the uncertainty set.

    ``project()``: Returns the uncertainty set projection.
    """

    def __init__(self, **kwargs):
        pass

    def get(self):
        """
        Get a realized uncertainty vector

        Raises
        ------
        ``NotImplemented``
        """
        raise NotImplementedError("Function not yet implemented")

    def diam(self):
        """
        Get the diameter of the uncertainty set.

        Raises
        ------
        ``NotImplemented``
        """
        raise NotImplementedError("Function not yet implemented")

    def project(self, z):
        """
        Calculate the projection of :math:`z` onto the set.

        Raises
        ------
        ``NotImplemented``
        """
        raise NotImplementedError("Function not yet implemented")


class CertaintySet(UncertaintySetBase):
    r"""
    Creates a set with no uncertainty.

    Parameters
    ----------
    size : int
        The size of the vector realizations.
    **kwargs : any
        Additional parameters.

    Methods
    -------
    ``get()``: Returns an uncertainty vector realization.

    ``diam()``: Returns the diameter of the uncertainty set.

    ``project()``: Returns the uncertainty set projection.
    """

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.n = size

    def get(self):
        """Returns a vector of zeros."""
        return np.zeros(self.n)

    def diam(self):
        """Returns zero, since set has no diameter."""
        return 0

    def project(self, z):
        """Returns a vector of zeros."""
        return np.zeros(self.n)


class BudgetUncertaintySet(UncertaintySetBase):
    r"""
    Creates a budgeted uncertainty set object.

    Budgeted Uncertainty set defined as

    .. math::
        \Big\{z \in \mathbb{R}^{n} : -1 \leq z_{i} \leq 1, \sum_{i}|z_{i}| \leq \Gamma \Big\}

    If budgeted uncertainty set is ``'half'``, then

    .. math::
        \Big\{z \in \mathbb{R}^{n}_{+} : 0 \leq z_{i} \leq 1, \sum_{i}|z_{i}| \leq \Gamma \Big\}

    where :math:`n` and :math:`\Gamma` are two input parameters.

    Parameters
    ----------
    size : ``int``
        The size of vectors :math:`z`.
    Gamma : ``float``
        The uncertainty parameter.
    half : ``bool``, optional
        Whether or not to consider only the positive half. The default is ``False``.
    n_store : ``int``
        Number of feasible vectors to generate. The default is ``10``.
    **kwargs : any
        Other additional parameters.
            decimals : ``int``, optional
                The number of decimals for the values in the random vector :math:`z\in\mathcal{Z}`. The default is ``9``.
            dist : ``str``, optional
                The type of distribution to generate the random vectors :math:`z`. Either ``'normal'`` or ``'uniform'``.
                The default is ``'uniform'``.

    Methods
    -------
    ``get()``: Returns an uncertainty vector realization.

    ``diam()``: Returns the diameter of the uncertainty set.

    ``project()``: Returns the uncertainty set projection.

    ``generate()``: Generates a sequence of feasible vectors :math:`z`.

    ``feasible()``: Tests if the generated vector is feasible.
    """

    def __init__(self, size, Gamma, half=False, n_store=10, **kwargs):
        self.n = size
        self.Gamma = Gamma
        self.half = half
        self._n_store = n_store
        self.i = 0  # Save a list of feasible realization
        self._dist = kwargs.get('dist', 'uniform')
        self.store = self.generate(n_store, decimals=int(kwargs.get('decimals', 9)))
        self.diameter = self._diam()  # Diameter of the set

    def get(self):
        r"""
        Returns a feasible realization :math:`z\in\mathcal{Z}` from the stored sequence.

        Returns
        -------
        z : ``Sequence``
            Feasible realization :math:`z\in\mathcal{Z}`.
        """
        z = self.store[self.i]  # Fetch a realization i
        self.i += 1  # Update index to move forward

        if (self.i == len(self.store)):  # If all feasible realization have been used, generate new realizations
            self.i = 0  # Reset index
            self.store = self.generate(self._n_store)

        return z

    def diam(self):
        """Returns the set's diameter."""
        return self.diameter

    def _diam(self):
        r"""
        Calculates the diameter of the uncertainty set.

        The diameter is calculated as the :math:`l2`-norm of the set, i.e.,

        .. math::
            diam = \sup_{u, v \in \mathcal{U}} ||u - v||_{2}
        """
        # n = self.n
        # x.shape = (2n,), x = [u, v]
        obj = lambda x: - np.linalg.norm(x[:self.n] - x[self.n:], 2)
        constr = (
            {   # Budgeted constraint for u
                'type': 'ineq',
                'fun': lambda x: self.Gamma - np.linalg.norm(x[:self.n], 1)
            },
            {   # Budgeted constraint for v
                'type': 'ineq',
                'fun': lambda x: self.Gamma - np.linalg.norm(x[self.n:], 1)
            },)
        # Lower and upper bounds
        if (self.half):
            bounds = tuple([(0.0, 1.0) for _ in range(2 * self.n)])
        else:
            bounds = tuple([(-1.0, 1.0) for _ in range(2 * self.n)])
        # get initial iterate
        # ! initial matters a lot
        # ! set x0 = 0 does NOT yield a correct solution
        x0 = np.zeros(2 * self.n)
        x0[:self.n] = self.get()
        x0[self.n:] = self.get()
        # call solver
        res = minimize(
            fun=obj,
            x0=x0,
            bounds=bounds,
            constraints=constr
        )

        return -res.fun

    def project(self, z0):
        r"""
        Function finds the projection of :math:`z_{0}` onto the set.

        The projection is defined as

        .. math::
            \underset{y \in \mathcal{U}}{\text{argmin}} \  ||y - z||_{2}

        Parameters
        ----------
        z0 : ``Sequence``
            The vector :math:`z_{0}` to calculate the projection.

        Return
        ------
        The vector projection.
        """
        obj = lambda x: np.linalg.norm(x - z0, 2)
        constr = (
            {'type': 'ineq',
             'fun': lambda x: self.Gamma - np.linalg.norm(x, 1)
             })
        if (self.half):
            bounds = tuple([(0.0, 1.0) for _ in range(self.n)])
        else:
            bounds = tuple([(-1.0, 1.0) for _ in range(self.n)])
        res = minimize(  # Call solver, 'SLSQP'
            fun=obj,
            x0=np.zeros(self.n),
            bounds=bounds,
            constraints=constr
        )

        return res.x

    def generate(self, n, decimals=9):
        r"""
        Generates a random vector :math:`z` with feasible values.

        Parameters
        ----------
        n : ``int``
            The number of feasible vectors :math:`z`.

        Returns
        -------
        store : ``Sequence``
            A list with :math:`n` feasible vectors :math:`z \in \mathcal{U}`.
        """
        count = 0  # Keeps track of number of feasible z generated
        store = []
        while True:
            if (self._dist == 'normal'):
                # if (self.half):
                #     z = np.random.normal(0.50, (1.0 / 6.0), self.n)
                # else:
                #     z = np.random.normal(0.00, (1.0 / 3.0), self.n)
                z = np.random.normal(0.00, 0.34, self.n)
            else:
                z = self._generate_random(decimals=decimals)

            if (self.feasible(z)):  # Store `z` only if it is feasible
                store.append(z)
                count = count + 1

            if count >= n:
                # print("BudgetUncertaintySet:generate: success")
                break

        return np.array(store, dtype=float)

    def _generate_random(self, decimals=9):
        r"""
        Function generates a random vector :math:`z\in\mathcal{Z}` with :math:`n` values.

        The random values are sampled from a uniform distribution.

        Parameters
        ----------
        decimals : ``int``, optional
            The number of decimals for the values in the random vector :math:`z\in\mathcal{Z}`. The default is ``9``.

        Returns
        -------
        z : ``Sequence``
            A random vector :math:`z\in\mathcal{Z}`.
        """
        z = []

        for i in range(self.n):
            upper = float(np.min([1.0, (self.Gamma - np.sum(np.abs(z)))]))
            if (self.half):
                z += [float(np.random.uniform(low=0.0, high=upper, size=1))]
            else:
                z += [float(np.random.uniform(low=(- upper), high=upper, size=1))]
        z = np.round(np.array(z, dtype=float), decimals=decimals)

        for i in range(np.random.randint(1, 11)):
            np.random.shuffle(z)

        return z

    def feasible(self, z):
        r"""
        Function checks if a given vector :math:`z` is inside the uncertainty set :math:`\mathcal{Z}`.

        Parameters
        ----------
        z : ``Sequence``
            Vector to be tested.

        Return
        ------
        is_feasible : ``bool``
            ``True`` if :math:`z \in \mathcal{Z}`, ``False`` otherwise.
        """
        is_feasible = True

        if (self.half):  # Check bounds condition
            if (np.any(z < 0) or np.any(z > 1)):
                is_feasible = False
        else:
            if (np.any(z < -1) or np.any(z > 1)):
                is_feasible = False

        if (np.linalg.norm(z, ord=1) > self.Gamma):  # Check budgeted constraint
            is_feasible = False

        return is_feasible


class EllipsoidalUncertaintySet(UncertaintySetBase):
    r"""
    Creates an ellipsoidal uncertainty set object.

    Ellipsoidal Uncertainty set defined as

    .. math::
        \Big\{z \in \mathbb{R}^{n} : z^{T} \Sigma z \leq \rho^{2} \Big\}

    where :math:`\Sigma \in \mathbb{R}^{n \times n}` and :math:`\rho \in \mathbb{R}` are two input parameters.

    Parameters
    ----------
    size : int
        The size of vectors :math:`z`.
    rho : float
        The uncertainty parameter.
    sigma : Sequence
        The matrix :math:`\Sigma`.
    n_store : int
        Number of feasible vectors to generate. The default is ``10``.
    **kwargs : any
        Other additional parameters.

    Methods
    -------
    ``get()``: Returns an uncertainty vector realization.

    ``diam()``: Returns the diameter of the uncertainty set.

    ``project()``: Returns the uncertainty set projection.

    ``generate()``: Generates a sequence of feasible vectors :math:`z`.

    ``feasible()``: Tests if the generated vector is feasible.

    ``_check_sigma_valid()``: If the sigma matrix is given, this function tests if it is positive definite.
    """

    def __init__(self, size, rho, sigma=None, n_store=10, **kwargs):
        # super().__init__(**kwargs)
        self.n = size
        self.rho = rho
        if (sigma is not None):
            if (not self._check_sigma_valid(sigma, 1e-8, 2)):  # Check if sigma is a valid symmerticall matrix
                raise ValueError("The matrix sigma is not valid!")
            self.sigma = sigma
            self.L = np.linalg.cholesky(sigma)
        else:
            self.sigma = np.eye(size)
            self.L = np.eye(size)
        self._n_store = n_store
        self.i = 0
        self.store = self.generate(self._n_store)
        self.diameter = self._diam()

    def get(self):
        """
        Returns a feasible realization from the stored sequence.

        Returns
        -------
        z : Sequence
            Feasible realization.
        """
        z = self.store[self.i]  # Fetch a realization i
        self.i += 1  # Update index to move forward

        if (self.i == len(self.store)):  # If all feasible realization have been used, generate new realizations
            self.i = 0  # Reset index
            self.store = self.generate(self._n_store)

        return z

    def diam(self):
        """Returns the set's diameter."""
        return self.diameter

    def _diam(self):
        r"""
        Calculates the diameter of the uncertainty set.

        The diameter is calculated as the :math:`l2`-norm of the set, i.e.,

        .. math::
            diam = \sup_{u, v \in \mathcal{U}} ||u - v||_{2}
        """
        pass

    def project(self, z0):
        r"""
        Function finds the projection of :math:`z_{0}` onto the set.

        The projection is defined as :math:`\underset{y \in \mathcal{U}}{argmin} \  ||y - z||_{2}`

        Parameters
        ----------
        z0 : Sequence
            The vector :math:`z_{0}` to calculate the projection.

        Return
        ------
        The vector projection.
        """
        w = ((self.L @ z0) / self.rho)  # Transform and scale to unit circle
        norm_w = np.linalg.norm(w, 2)
        if (norm_w <= 1):  # z0 is inside the set

            return z0
        else:  # Find the projection point if z0 is outside
            w = ((w / norm_w) * self.rho)
            z = np.linalg.solve(self.L, w)

            return z

    def generate(self, n):
        r"""
        Generates a random vector :math:`z` with feasible.

        Parameters
        ----------
        n : int
            The number of feasible vectors :math:`z`.

        Returns
        -------
        store : Sequence
            A list with :math:`n` feasible vectors :math:`z \in \mathcal{U}`.
        """
        count = 0
        store = []
        while True:
            w = np.random.uniform(size=self.n)  # Random sample in unit hyper-square
            norm_w = np.linalg.norm(w, 2)
            if (norm_w > 1):
                continue
            z = np.linalg.solve(self.L, (w * self.rho))
            store.append(z)
            count = count + 1
            if (count >= n):
                break
        return np.array(store, dtype=float)

    def feasible(self, z):
        r"""
        Function checks if a given vector :math:`z` is inside the uncertainty set :math:`\mathcal{U}`.

        Parameters
        ----------
        z : Sequence
            Vector to be tested.

        Return
        ------
        is_feasible : bool
            ``True`` if :math:`z \in \mathcal{U}`, ``False`` otherwise.
        """
        is_feasible = False
        tmp = ((z.T @ self.sigma) @ z)

        if (tmp <= (self.rho ** 2)):
            is_feasible = True

        return is_feasible

    def _check_sigma_valid(self, mtrx, tolerance=1e-8, check=1):
        r"""
        Function checks if a given matrix :math:`\Sigma` is symmetric or positive definite.

        Parameters
        ----------
        mtrx : Sequence
            The matrix to be analysed.
        tolerance : float, optional
            Maximum tolerance used to overcome rounding issues. The default is ``1e-8``.
        check : int, optional
            Whether to check for symmetry (``1``) or for positive definiteness (``2``). The default is ``1``.

        Returns
        -------
        is_valid : bool
            Whether or not the matrix is symmetric/positive definite.
        """
        mtrx_T = mtrx.T
        is_valid = False

        if (check == 1):
            if (np.all(np.abs(mtrx - mtrx_T) < tolerance)):
                is_valid = True
        elif (check == 2):
            if (np.all(np.abs(mtrx - mtrx_T) < tolerance) and np.all(np.linalg.eigvals(mtrx) > 0.0)):
                is_valid = True

        return is_valid


class CustomizedUncertaintySet(UncertaintySetBase):
    r"""
    Creates a Customized uncertainty set object.

    Customized Uncertainty set defined as

    .. math::
        \Big\{z \in \mathbb{R}^{n} \ \vert \ z \geq 0,\ \sum_{i}z_{i} = 1,\ \sum_{i}z_{i}\ln{(z_{i})} \leq \rho \Big\}

    where :math:`\rho \in \mathbb{R}` is an input parameter.

    Parameters
    ----------
    size : int
        The size of vectors :math:`z`.
    rho : float
        The uncertainty parameter.
    n_store : int
        Number of feasible vectors to generate. The default is ``10``.
    **kwargs : any
        Other additional parameters.
            decimals : ``int``, optional
                The number of decimals for the values in the random vector :math:`z\in\mathcal{Z}`. The default is ``9``.
            dist : ``str``, optional
                The type of distribution to generate the random vectors :math:`z`. Either ``'uniform1'`` or ``'uniform2'``.
                The default is ``'uniform2'``.

    Methods
    -------
    ``get()``: Returns an uncertainty vector realization.

    ``diam()``: Returns the diameter of the uncertainty set.

    ``project()``: Returns the uncertainty set projection.

    ``generate()``: Generates a sequence of feasible vectors :math:`z`.

    ``feasible()``: Tests if the generated vector is feasible.
    """

    def __init__(self, size, rho, n_store=10, **kwargs):
        self.n = size
        self.rho = rho
        self._n_store = n_store
        self.i = 0
        self._dist = kwargs.get('dist', 'uniform2')
        self.store = self.generate(self._n_store, decimals=int(kwargs.get('decimals', 9)))
        self.diameter = self._diam()

    def get(self):
        r"""
        Returns a feasible realization of vector :math:`z` from the stored sequence.

        Returns
        -------
        z : ``Sequence``
            Feasible realization.
        """
        z = self.store[self.i]  # Fetch a realization i
        self.i += 1  # Update index to move forward

        if (self.i == len(self.store)):  # If all feasible realization have been used, generate new realizations
            self.i = 0  # Reset index
            self.store = self.generate(self._n_store)

        return z

    def diam(self):
        """Returns the set's diameter."""
        return self.diameter

    def _diam(self):
        r"""
        Calculates the diameter of the uncertainty set.

        The diameter is calculated as the :math:`l2`-norm of the set, i.e.,

        .. math::
            diam = \sup_{u, v \in \mathcal{U}} ||u - v||_{2}.

        For the ``scipy.optimize.minimize`` function, we then have:

        .. math::
            diam = \underset{u, v \in \mathcal{U}}{\text{minimize}}\ \ \ (- ||u - v||_{2}).
        """
        def objct_fnc(var):
            r"""
            Callable for the ``scipy.optimize.minimize`` function.

            Parameters
            ----------
            var : ``Sequence``
                The sequence with the problem variables.

            Returns
            -------
            objct : ``float``
                The diameter of the of the set.
            """
            objct = (- np.linalg.norm((var[:self.n] - var[self.n:]), 2))
            return objct

        def constraint(n, rho):
            r"""
            Callable for the ``scipy.optimize.minimize`` function.

            Parameters
            ----------
            n : ``int``
                Size of vector.
            rho : ``float``
                Parameter :math:`\rho`.

            Returns
            -------
            ``tuple``
                Tuple with constraints.
            """
            # Customized set constraint for u
            constr = [{'type': 'ineq',  # :math:`u_{i} \geq 0`
                       'fun': lambda x: x[:n]},
                      {'type': 'eq',  # :math:`\sum_{i} u_{i} = 1`
                       'fun': lambda x: (1.0 - np.sum(x[:n]))},
                      {'type': 'ineq',  # :math:`\sum_{i} u_{i} \ln(u_{i}) \leq \rho`
                       'fun': lambda x: (rho - np.sum((x[:n] * np.log(x[:n], where=(x[:n] > 0.0)))))},
                      ]
            # Customized set constraint for v
            constr += [{'type': 'ineq',  # :math:`v_{i} \geq 0`
                       'fun': lambda x: x[n:]},
                       {'type': 'eq',  # :math:`\sum_{i} v_{i} = 1`
                       'fun': lambda x: (1.0 - np.sum(x[n:]))},
                       {'type': 'ineq',  # :math:`\sum_{i} v_{i} \ln(v_{i}) \leq \rho`
                       'fun': lambda x: (rho - np.sum((x[n:] * np.log(x[n:], where=(x[n:] > 0.0)))))},
                       ]
            return tuple(constr)
        bounds = [(0.0, 1.0) for _ in range(2 * self.n)]  # Lower and upper bounds for :`math:`u,v \in \mathcal{U}`
        # get initial iterate
        # ! initial matters a lot
        # ! set x0 = 0 does NOT yield a correct solution
        x0 = np.zeros(2 * self.n)
        x0[:self.n] = self.get()
        x0[self.n:] = self.get()

        res = minimize(
            fun=objct_fnc,
            x0=x0,
            bounds=tuple(bounds),
            constraints=constraint(self.n, self.rho)
        )

        return -res.fun

    def project(self, z0):
        r"""
        Function finds the projection of :math:`z_{0}` onto the set.

        The projection is defined as :math:`\underset{y \in \mathcal{U}}{\text{argmin}} \  ||y - z||_{2}`

        Parameters
        ----------
        z0 : ``Sequence``
            The vector :math:`z_{0}` to calculate the projection.

        Return
        ------
        The vector projection.
        """
        def objct_fnc(x0, *args):
            """
            Callable for the ``scipy.optimize.minimize`` function.

            Parameters
            ----------
            x0 : ``Sequence``
                The sequence with the problem variables.
            *args : ``Sequence``
                The vector :math:`z_{0}` to project.

            Returns
            -------
            objct : ``float``
                The :math:`l2`-norm of the of the subtraction between :math:`z` and :math:`z_{0}`.
            """
            z_0 = args[0]
            objct = np.linalg.norm((x0 - z_0), 2)
            return objct
        constr = [{'type': 'ineq',  # :math:`z_{i} \geq 0`
                   'fun': lambda x: x[:self.n]},
                  {'type': 'eq',  # :math:`\sum_{i} z_{i} = 1`
                   'fun': lambda x: (np.sum(x[:self.n]) - 1.0)},
                  {'type': 'ineq',  # :math:`\sum_{i} z_{i} \ln(z_{i}) \leq \rho`
                   'fun': lambda x: (self.rho - np.sum((x[:self.n] * np.log(x[:self.n], where=(x[:self.n] > 0.0)))))},
                  ]
        bounds = [(0.0, 1.0) for _ in range(self.n)]

        res = minimize(  # Call solver, 'SLSQP'
            fun=objct_fnc,
            x0=np.ones(self.n),
            args=(z0),
            bounds=tuple(bounds),
            constraints=tuple(constr)
        )

        return res.x

    def generate(self, n, decimals=9):
        r"""
        Generates a random vector :math:`z` with feasible values.

        Parameters
        ----------
        n : ``int``
            The number of feasible vectors :math:`z`.

        Returns
        -------
        store : ``Sequence``
            A list with :math:`n` feasible vectors :math:`z \in \mathcal{U}`.
        """
        count = 0
        store = []
        while True:
            if (self._dist == 'uniform1'):
                z = np.random.uniform(low=0.0, high=1.0, size=self.n)  # Random sample
                z = (z / np.sum(z))
            else:
                z = self._generate_random(decimals=decimals)

            if (self.feasible(z)):  # Store :math:`z` only if it is feasible
                store.append(z)
                count = count + 1

            if (count >= n):
                break

        return np.array(store, dtype=float)

    def _generate_random(self, decimals=9):
        r"""
        Function generates a random vector :math:`z\in\mathcal{Z}` with :math:`n` values.

        The random values are sampled from a uniform distribution.

        Parameters
        ----------
        decimals : ``int``, optional
            The number of decimals for the values in the random vector :math:`z\in\mathcal{Z}`. The default is ``9``.

        Returns
        -------
        z : ``Sequence``
            A random vector :math:`z\in\mathcal{Z}`.
        """
        z = []

        for i in range((self.n - 1)):
            upper = float(np.min([1.0, (1.0 - np.sum(np.abs(z)))]))
            z += [float(np.random.uniform(low=0.0, high=upper, size=1))]
        z += [(1.0 - np.sum(np.abs(z)))]
        z = np.round(np.array(z, dtype=float), decimals=decimals)

        for i in range(np.random.randint(1, 11)):
            np.random.shuffle(z)

        return z

    def feasible(self, z):
        r"""
        Function checks if a given vector :math:`z` is inside the uncertainty set :math:`\mathcal{Z}`.

        Parameters
        ----------
        z : ``Sequence``
            Vector to be tested.

        Return
        ------
        is_feasible : ``bool``
            ``True`` if :math:`z \in \mathcal{Z}`, ``False`` otherwise.
        """
        is_feasible = True

        if (np.any(z < 0.0) or np.any(z > 1.0)):  # Check bounds on :math:`z`
            is_feasible = False

        if ((np.round(np.sum(z), 15) > 1.0) or (np.round(np.sum(z), 15) < 1.0)):  # Check sum of :math:`z_{i}` equals to 1
            is_feasible = False

        if (np.sum((z * np.log(z, where=(z > 0.0)))) > self.rho):  # Check sum of :math:`z_{i}\ln(z_{i})` <= to :math:`\rho`
            is_feasible = False

        return is_feasible
