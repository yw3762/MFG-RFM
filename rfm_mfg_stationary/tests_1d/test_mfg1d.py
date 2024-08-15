import matplotlib.pyplot as plt

from rfm_mfg_stationary.mfg_1d.policy_iter_mfg1d import solve_1d_stationary_mfg
from utils.utils_1d import plot_RFM_1d, set_seed


def test():
    set_seed(100)
    M_p_hjb, J_n_hjb = 4, 100
    M_p_fp, J_n_fp = 4, 100
    Q_hjb, Q_fp = 100, 100

    models_fp, w_fp, models_hjb, w_hjb = solve_1d_stationary_mfg(4, 100, 4, 100, 100, 100, n_iters=20)

    plot_RFM_1d(models_fp, w_fp, "m")
    plot_RFM_1d(models_hjb, w_hjb, "u")

test()
