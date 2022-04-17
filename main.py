r"""
The main function driver to run numerical studies.
"""


import numpy as np
from algo_analysis import case_study_verify, solve_case_study, test_convergence, test_runtime, test_robustness
from demo_6_2 import solve_rc_reformulation_ch6
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == "__main__":

    tests = {'rc_reform': False,
             'verify_ad': False,
             'scaled_ad': False,
             'converges': False,
             'run_time': True,
             'solution': False,
             }

    params_general = {'eps': 1e-6,
                      'Gamma': 1.0,
                      'half': True,
                      'rho': 1.0,
                      'n_store': 10,
                      'B': 1.0,
                      'seed': 12107,
                      'earlystop': True,
                      'itermax': 1e3,
                      'display': False,
                      'n_jobs': -1,
                      'backend': 'multiprocessing',
                      'verbose': 25,
                      }

    if (tests['rc_reform']):
        (fval, x) = solve_rc_reformulation_ch6(n=4,
                                               option='original',
                                               prob='budget',
                                               Gamma=params_general['Gamma'],
                                               rho=params_general['rho'],
                                               verbose=params_general['display'],
                                               seed=params_general['seed'],
                                               maxiter=10000,
                                               )  # tol=1e-12

    if (tests['verify_ad']):
        (fval, x) = case_study_verify(B=params_general['B'],
                                      Gamma=params_general['Gamma'],
                                      eps=params_general['eps'],
                                      display=params_general['display']
                                      )

    if (tests['scaled_ad']):
        (fval, x) = solve_case_study(size=5,
                                     B=params_general['B'],
                                     seed=params_general['seed'],
                                     option='scaled',
                                     eps=params_general['eps'],
                                     U='budget',
                                     Gamma=params_general['Gamma'],
                                     half=params_general['half'],
                                     rho=params_general['rho'],
                                     n_store=params_general['n_store'],
                                     earlystop=params_general['earlystop'],
                                     itermax=params_general['itermax'],
                                     display=params_general['display']
                                     )

    if (tests['converges']):
        params_converge_test = {'U_set': 'budget',
                                'size': 4,
                                'option': 'original',
                                'seq': np.array(([1] + [i for i in range(10, 501, 10)]
                                                 + [i for i in range(525, 1001, 25)]), dtype=int),
                                'seq_small': np.array(([1] + [i for i in range(10, 201, 10)]), dtype=int),
                                }
        converge = test_convergence(eps=params_general['eps'],
                                    U=params_converge_test['U_set'],
                                    Gamma=params_general['Gamma'],
                                    half=params_general['half'],
                                    rho=params_general['rho'],
                                    n_store=params_general['n_store'],
                                    size=params_converge_test['size'],
                                    option=params_converge_test['option'],
                                    B=params_general['B'],
                                    seq=params_converge_test['seq'],
                                    # seq=params_converge_test['seq_small'],
                                    seed=None,
                                    get_x=False,
                                    n_jobs=params_general['n_jobs'],
                                    backend=params_general['backend'],
                                    verbose=params_general['verbose']
                                    )

    if (tests['run_time']):
        params_runtime_test = {'n_tests': 50,
                               'instances': np.array([4, 5, 6, 8, 10, 20], dtype=int),
                               'U_set': 'customized',
                               'option': 'scaled',
                               }
        runtime = {}
        for size in params_runtime_test['instances']:
            runtime[size] = test_runtime(n_tests=params_runtime_test['n_tests'],
                                         eps=params_general['eps'],
                                         U=params_runtime_test['U_set'],
                                         Gamma=params_general['Gamma'],
                                         half=params_general['half'],
                                         rho=params_general['rho'],
                                         n_store=params_general['n_store'],
                                         size=size,
                                         option=params_runtime_test['option'],
                                         B=params_general['B'],
                                         seed=None,
                                         earlystop=True,
                                         itermax=1e3,
                                         display=False,
                                         get_x=False,
                                         n_jobs=params_general['n_jobs'],
                                         backend=params_general['backend'],
                                         verbose=params_general['verbose']
                                         )

    if (tests['solution']):
        params_solution_test = {'n_tests': 1000,
                                'eps': 0.0,
                                'U_set': 'budget',
                                'option': 'original',
                                'dist': 'uniform',
                                'n_jobs': -1,
                                'verbose': 0,
                                }
        solution_perform = test_robustness(xs=x,
                                           n_tests=params_solution_test['n_tests'],
                                           option=params_solution_test['option'],
                                           B=params_general['B'],
                                           seed=params_general['seed'],
                                           eps=params_solution_test['eps'],
                                           U=params_solution_test['U_set'],
                                           Gamma=params_general['Gamma'],
                                           half=params_general['half'],
                                           rho=params_general['rho'],
                                           n_store=params_general['n_store'],
                                           dist=params_solution_test['dist'],
                                           n_jobs=params_solution_test['n_jobs'],
                                           backend=params_general['backend'],
                                           verbose=params_solution_test['verbose']
                                           )
        solution_perform = {'solution': x,
                            'mean value': solution_perform[:, 0].mean(axis=0),
                            'std': solution_perform[:, 0].std(axis=0),
                            'min value': solution_perform[:, 0].min(axis=0),
                            'max value': solution_perform[:, 0].max(axis=0),
                            'feasible percent': solution_perform[:, 1].mean(axis=0),
                            }
