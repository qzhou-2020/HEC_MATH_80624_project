"""
    define several useful uncertainty set
"""

import numpy as np
from scipy.optimize import minimize


class UncertaintySetBase(object):

    """abstract base class for a uncertainty set object"""

    def __init__(self, **kwargs):
        pass

    def get(self):
        """get a realized uncertainty vector"""
        raise NotImplemented

    def diam(self):
        """get the diameter of the set"""
        raise NotImplemented

    def project(self, z):
        """return the projection of z onto the set"""
        raise NotImplemented


class BudgetUncertaintySet(UncertaintySetBase):
    """
    Budgeted Uncertainty set is 

        {z \in R^n : -1 <= z_i <= 1, norm(z, 1) <= gamma}

    if budgeted uncertainty set is 'half', then

        {z \in R^n: 0 <= z_i <= 1, norm(z, 1 <= gamma)}

    where n and gamma are two input parameters
    """

    def __init__(self, size, gamma, half=False, **kwargs):
        self.n = size
        self.gamma = gamma
        self.half = half
        # save a list of feasible realization
        self.i = 0
        self.store = self.generate(10)
        # diameter of the set
        self.diameter = self._diam()

    def get(self):
        # fetch a realization
        z = self.store[self.i]
        # index move forward
        self.i = self.i + 1
        # re-generate
        if self.i == len(self.store):
            self.i = 0
            self.store = self.generate(10)
        return z

    def diam(self):
        return self.diameter 

    def _diam(self):
        """
            return the diameter of the uncertainty set

                diam = sup_{u, v \in Set} norm(u-v, 2)
        """

        n = self.n
        # x.shape = (2n,), x = [u, v]
        obj = lambda x: -np.linalg.norm(x[:n] - x[n:], 2)
        constr = (
            # budgeted constraint for u
            {'type': 'ineq', 'fun': lambda x: self.gamma - np.linalg.norm(x[:n], 1)},
            # budgeted constraint for v
            {'type': 'ineq', 'fun': lambda x: self.gamma - np.linalg.norm(x[n:], 1)}
        )
        # lower and upper bounds
        if self.half:
            bounds = tuple([ (0, 1) for _ in range(2*n)])
        else:
            bounds = tuple([ (-1, 1) for _ in range(2*n)])
        # get initial iterate
        # ! initial matters a lot
        # ! set x0 = 0 does NOT yield a correct solution
        x0 = np.zeros(2*n)
        x0[:n] = self.get()
        x0[n:] = self.get()
        # call solver
        res = minimize(
            obj, x0, bounds=bounds, constraints=constr
        )
        return -res.fun

    def project(self, z0):
        """
            find the project of z0 onto the set

                argmin_{y \in Set} norm(y-z, 2)
        """

        obj = lambda x: np.linalg.norm(x - z0, 2)
        constr = (
            {'type': 'ineq', 'fun': lambda x: self.gamma - np.linalg.norm(x, 1)}
        )
        if self.half:
            bounds = tuple([(0, 1) for _ in range(self.n)])
        else:
            bounds = tuple([(-1, 1) for _ in range(self.n)])
        # call solver, 'SLSQP'
        res = minimize(
            obj, np.zeros(self.n), bounds=bounds, constraints=constr
        )
        return res.x

    def generate(self, n):
        """randomly generate a list of feasible z """
        count = 0
        store = []
        while True:
            z = np.random.normal(0, 0.35, self.n)
            # store z only if it is feasible
            if self.feasbile(z):
                store.append(z)
                count = count + 1
            if count >= n:
                print("BudgetUncertaintySet:generate: success")
                break
        return store
    
    def feasbile(self, z):
        """Check if a given vector z is inside the uncertainty set"""
        # check bounds condition
        if self.half:
            if np.any(z < 0) or np.any(z > 1): return False
        else:
            if np.any(z < -1) or np.any(z > 1): return False
        # check budgeted constraint
        if np.linalg.norm(z, ord=1) > self.gamma: return False
        # True if all conditions are met
        return True


class CertaintySet(UncertaintySetBase):

    """no uncertainty"""

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.n = size

    def get(self):
        return np.zeros(self.n)

    def diam(self):
        return 0

    def project(self, z):
        return np.zeros(self.n)


class EllipsoidalUncertaintySet(UncertaintySetBase):
    """
    Ellipsoidal uncertainty set is:

        {z \in R^n: z^T \Sigma z <= \rho^2}
    
        where \Sigma \in R^{n \times n} and \rho \in R are two parameters.
    """

    def __init__(self, size, rho, sigma=None, **kwargs):
        super().__init__(**kwargs)
        self.n = size
        self.rho = rho
        if sigma:
            # todo: check if sigma is a valid symmerticall matrix
            self.sigma = sigma
            self.L = np.linalg.cholesky(sigma)
        else:
            self.sigma = np.eye(size)
            self.L = np.eye(size)
        self.i = 0
        self.store = self.generate(10)
        self.diameter = self._diam()

    def get(self):
        z = self.store[self.i]
        self.i = self.i + 1
        if self.i == len(self.store):
            self.i = 0
            self.store = self.generate(10)
        return z

    def diam(self):
        return self.diameter

    def _diam(self):
        pass

    def project(self, z0):
        # transform and scale to unit circle
        w = self.L @ z0 / self.rho
        normw = np.linalg.norm(w, 2)
        if normw <= 1:
            # z0 is inside the set
            return z0
        else:
            # find the projection point if z0 is outside
            w = w / normw * self.rho
            z = np.linalg.solve(self.L, w)
            return z

    def generate(self, n):
        count = 0
        store = []
        while True:
            # random sample in unit hyper-sqaure
            w = np.random.uniform(size=self.n)
            normw = np.linalg.norm(w, 2)
            if normw > 1:
                continue
            z = np.linalg.solve(self.L, w*self.rho)
            store.append(z)
            count = count + 1
            if count >= n:
                break
        return store

    def feasible(self, z):
        tmp = z.T @ self.sigma @ z
        if tmp <= self.rho**2:
            return True
        else:
            return False

    
def test_budgeted():
    """unit test: different methods in BudgetedUncertaintySet"""
    n = 10
    Z2 = BudgetUncertaintySet(n, 2, half=True)

    # check realizations
    for _ in range(50):
        print(Z2.get())

    # check diameter
    print("diameter:", Z2.diam())

    # z0 is outside uncertainty set
    z0 = np.ones(n)
    z = Z2.project(z0)
    print("z is feasible:", Z2.feasbile(z))
    print(z, np.linalg.norm(z - z0, 2))

    # z0 is inside
    z0 = np.zeros(n)
    z = Z2.project(z0)
    print("z is feasible:", Z2.feasbile(z))
    print(z, np.linalg.norm(z-z0, 2))

    # z0 is inside
    z0 = Z2.get()
    z = Z2.project(z0)
    print("z is feasible:", Z2.feasbile(z))
    print(z, np.linalg.norm(z-z0, 2))


def test_ellipsoidal():
    n = 2
    Z = EllipsoidalUncertaintySet(n, 1)
    for _ in range(10):
        z = Z.get()
        TF = Z.feasible(z)
        diam = np.linalg.norm(z,2)
        print(f"{z} is inside the set: {TF},  diameter: {diam}")

    z0 = Z.get()
    z = Z.project(z0)
    print(f"z0: {z0}, projected z: {z}")

    z0 = np.array([2, 3])
    z = Z.project(z0)
    print(f"z0: {z0}, projected z: {z}")
    print(f"|z0|: {np.linalg.norm(z0, 2)}, |z|: {np.linalg.norm(z,2)}")


if __name__ == "__main__":
    test_budgeted()
    test_ellipsoidal()