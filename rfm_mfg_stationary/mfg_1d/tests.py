import numpy as np

from rfm_mfg_stationary.mfg_1d.policy_iter_1d import policy_iteration
from rfm_mfg_stationary.mfg_1d.utils.utils_1d import set_seed, plot_RFM_1d, plot_by_iter


def test():
    set_seed(100)
    M_arr = [2, 4, 8]
    J_arr = [10, 20, 40, 80]
    Q_arr = [20, 40, 80, 160]
    assert len(J_arr) == len(Q_arr)

    n_iters = 30

    system_residuals = np.zeros((len(M_arr), len(J_arr), n_iters+1))

    for i in range(len(M_arr)):
        for j in range(len(J_arr)):
            M = M_arr[i]
            J = J_arr[j]
            Q = Q_arr[j]

            _, _, system_residuals[i, j, :] = policy_iteration(M, J, M, J, Q, Q, n_iters)
            plot_by_iter(system_residuals[i, j, :], f"MFG residual with M={M}, J={J}, Q={Q}", 'Residual error')


test()
