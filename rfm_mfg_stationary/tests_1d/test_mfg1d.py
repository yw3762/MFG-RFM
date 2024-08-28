from rfm_mfg_stationary.mfg_1d.policy_iter_mfg1d import solve_1d_stationary_mfg
from utils.utils_1d import plot_RFM_1d, set_seed


def test():
    set_seed(100)
    M_p_hjb, J_n_hjb = 4, 100
    M_p_fp, J_n_fp = 4, 100
    Q_hjb, Q_fp = 200, 200
    n_iters = 10

    m, u = solve_1d_stationary_mfg(M_p_hjb, J_n_hjb, M_p_fp, J_n_fp, Q_hjb, Q_fp, n_iters,
                                   intermediate_plot=False, random_init_q=False)

    plot_RFM_1d(m, "final m")
    plot_RFM_1d(u, "final u")


test()
