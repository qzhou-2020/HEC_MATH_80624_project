r"""
The main function driver to run numerical studies.
"""

import numpy as np
from uncertainty import BudgetUncertaintySet, CertaintySet, EllipsoidalUncertaintySet
from tool import build_problem
from dualsubgradient import dualSubgradient
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def case_study_1():
    """
    Verification case, using data from Exercise 6.2 in `Delage, E. (2021)`__

    __ https://zonecours2.hec.ca/access/content/group/MATH80624A.H2022/Dossier%20caché/LectureNotes_v7.pdf
    """
    (abar, size, cs, ds, ps) = build_problem(n=4, option='original')
    B = 1.0
    Gamma = 1.0
    eps = 1e-6
    U = BudgetUncertaintySet(size, Gamma, half=True)

    fval, x = dualSubgradient(eps, U, abar, cs, ds, ps, B, display=False)  # Solving with Dual-Subgradient algorithm

    print("The robust with budget solution predicts a profit of {:.3f}".format(fval))


def case_study_2(size=10, **kwargs):
    """
    Function for the second case study.

    Parameters
    ----------
    size : int, optional
        The size of vectors. The default is ``10``.
    **kwargs : any
        Additional parameter information to build problem (see function ``build_problem()``)

    Returns
    -------
    None.

    """
    (abar, _, cs, ds, ps) = build_problem(n=size,
                                          a_params=kwargs.get('a_params', (0.15000, 0.20000)),
                                          c_param=kwargs.get('c_param', 30.0),
                                          d_param=kwargs.get('d_param', 1000.0),
                                          p_params=kwargs.get('p_params', (0.1, 0.800, 1.100)),
                                          option='scaled')
    B = 1.0
    Gamma = 1.0
    eps = 1e-6

    U = BudgetUncertaintySet(size, Gamma, half=True)

    fval, x = dualSubgradient(eps, U, abar, cs, ds, ps, B, display=True)  # Solving with Dual-Subgradient algorithm

    print("The robust with budget solution predicts a profit of {:.3f}".format(fval))


if __name__ == "__main__":
    # case_study_1()
    # case_study_2(4)
    case_study_2(10)
