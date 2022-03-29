"""
    define several useful uncertainty set
"""

import numpy as np
from scipy.optimize import minimize


class UncertaintySetBase(object):

    def __init__(self, **kwargs):
        pass

    def get(self):
        raise NotImplemented

    def diam(self):
        raise NotImplemented

    def project(self, z):
        raise NotImplemented


class BudgetUncertaintySet(UncertaintySetBase):

    """
    Budgeted Uncertainty set is 

        {z \in R^n : -1 <= z_i <= 1, norm(z, 1) <= gamma}

    if budgeted uncertainty set is 'half'
    """
    def __init__(self, size, gamma, half=False, **kwargs):
        self.n = size
        self.gamma = gamma
        self.half = half
        self.i = 0
        self.store = self.generate(100)
        self.diameter = self._diam()

    def get(self):
        """
            get a feasible realized uncertainty vector
        """
        z = self.store[self.i]
        self.i = self.i + 1
        
        # re-generate
        if self.i == len(self.store):
            self.i = 0
            self.store = self.generate(100)

        return z

    def diam(self):
        return self.diameter 

    def _diam(self):
        """
            return the diameter of the uncertainty set

                diam = sup_{u, v \in Set} norm(u-v, 2)

            # ! the code doesn't work properly
        """
        n = self.n
        obj = lambda x: -np.linalg.norm(x[:n] - x[n:], 2)
        constr = (
            {'type': 'ineq', 'fun': lambda x: self.gamma - np.linalg.norm(x[:n], 1)},
            {'type': 'ineq', 'fun': lambda x: self.gamma - np.linalg.norm(x[n:], 1)}
        )
        if self.half:
            bounds = tuple([ (0, 1) for _ in range(2*n)])
        else:
            bounds = tuple([ (-1, 1) for _ in range(2*n)])

        res = minimize(
            obj, np.zeros(2*n), bounds=bounds, constraints=constr
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
        """
            randomly generate a list of feasible z
        """
        count = 0
        store = []
        while True:
            z = np.random.normal(0, 0.35, self.n)
            if self.feasbile(z):
                store.append(z)
                count = count + 1
            if count >= n:
                break
        return store
    
    def feasbile(self, z):
        """
            Check if a given vector z is inside the uncertainty set
        """
        if self.half:
            if np.any(z < 0) or np.any(z > 1): return False
        else:
            if np.any(z < -1) or np.any(z > 1): return False

        if np.linalg.norm(z, ord=1) > self.gamma: return False

        return True


def test_budgeted():
    Z2 = BudgetUncertaintySet(2, 1, half=True)
    for _ in range(10):
        print(Z2.get())

    print()

    print("diameter:", Z2.diam())

    print()

    z0 = np.array([1,1])
    z = Z2.project(z0)
    print("z is feasible:", Z2.feasbile(z))
    print(z, np.linalg.norm(z,1), np.linalg.norm(z - z0, 2))



if __name__ == "__main__":
    test_budgeted()