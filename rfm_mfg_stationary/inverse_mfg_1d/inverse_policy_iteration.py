import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq, pinv
from torch import sin, cos, pi, exp

from rfm_mfg_stationary.mfg_1d.policy_iter_1d import solve_FP_forward, policy_iteration, solve_HJB_forward
from rfm_mfg_stationary.mfg_1d.utils.utils_1d import init_rfm, RFM_function_factory, second_diff_RFM_function, \
    plot_RFM_1d, lagrangian_1d, plot_RFM_vs_true
from rfm_mfg_stationary.inverse_mfg_1d.utils_inv_1d import RF_valuation, concatenate_collocs, \
    func_valuation, build_RFM_matrix


def inverse_HJB(models_u, models_b, collocs_b_cat, m_func, q_func, observed_u, Fl_pinv, H_pinv, U_l, Mu, Mb, Ju, Jb, Qb,
                eps):
    """
    Solve the inverse problem for HJB, the observed data g(x) is the observed solution for u(x), which we evaluate
    on finite grid.

    :param f_nj: RFM functions for u,
    :param collocs_b: collocs points for b,
    :param m: solved m(x)
    :param observed_u: observed data for u as an RFM function, but we only assume knowledge over some finite points
    :param M_p: number of partitions
    :param J_n: number of RF functions
    :param Q: number of collocation points
    :return:
    """

    q_vals = q_func(collocs_b_cat)

    W_b = (m_func(collocs_b_cat) ** 2 + lagrangian_1d(collocs_b_cat, q_vals, None)).view(-1, 1).detach().numpy()

    M = build_RFM_matrix(models_u, q_vals, collocs_b_cat, Mb, Qb, Ju, eps)
    tilde_theta_nj = torch.tensor((H_pinv @ (M @ Fl_pinv @ U_l - W_b)).T)
    theta_nj, recovered_lambda = tilde_theta_nj[0,:-1].reshape((Mb, Jb)), tilde_theta_nj[0, -1]

    recovered_b = RFM_function_factory(models_b, theta_nj)
    return recovered_b, recovered_lambda


def PI_stationary(u_true, Mu, Mm, Mb, Ju, Jm, Jb, Il, Qu, Qm, Qb, n_iters=20, eps=0.3, tau=1e-6, plot=False):
    """
    Solves the inverse problem with final-time derivative information (i.e. case ii in Ren et al.)

    :param u_true: observed data for u as an RFM function
    :param Mu:
    :param Mm:
    :param Mb:
    :param Ju:
    :param Jm:
    :param Jb:
    :param Il:
    :param Qu:
    :param Qm:
    :param Qb:
    :param n_iters:
    :param eps:
    :param tau:
    :param plot:
    :return:
    """
    # fix datatype
    torch.set_default_dtype(torch.float64)

    # Initialize RFMs for FP and HJB with zero weights
    models_m, collocs_m = init_rfm(Mm, Jm, Qm)
    models_u, collocs_u = init_rfm(Mu, Ju, Qu)
    models_b, collocs_b = init_rfm(Mb, Jb, Qb)
    w_hjb = torch.zeros((Mu, Ju))
    w_fp = torch.zeros((Mm, Jm))
    x_l = torch.linspace(0, 1, steps=Il + 1)[:-1].view(-1, 1)

    # Pre-compute F_l, H_b, W_l, U_l matrices (as tensors)
    collocs_b_cat = concatenate_collocs(collocs_b)
    F_l = RF_valuation(models_u, x_l)
    H_b = RF_valuation(models_b, collocs_b_cat)
    H_b = torch.cat([RF_valuation(models_b, collocs_b_cat), -torch.ones((H_b.shape[0], 1))], dim=1)
    Fl_pinv = pinv(F_l.detach().numpy())  # F_l^\dagger
    H_pinv = pinv(H_b.detach().numpy())  # \tilde{H}_b^\dagger
    U_l = func_valuation(u_true, x_l).detach().numpy()  # U_l

    # historical solutions and policies
    historical_m = [RFM_function_factory(models_m, w_fp)]
    historical_u = [RFM_function_factory(models_u, w_hjb)]
    historical_b = [None]
    historical_lam = [None]
    historical_q = [None]
    historical_q[0], dq = second_diff_RFM_function(historical_u[0])

    # main loop
    for k in range(1, n_iters + 1):
        # Step (1): Solve FP equation.
        historical_m.append(solve_FP_forward(models_m, collocs_m, historical_q[k - 1], dq, Mm, Jm, Qm))
        if plot:
            plot_RFM_1d(historical_m[k], "m^(" + str(k) + ")")

        # Step (2): Solve Inverse HJB equation for b and lambda
        recovered_b, recovered_lambda = inverse_HJB(models_u, models_b, collocs_b_cat, historical_m[k],
                                                    historical_q[k - 1], u_true, Fl_pinv, H_pinv, U_l, Mu, Mb, Ju, Jb,
                                                    Qb, eps)
        historical_b.append(recovered_b)
        historical_lam.append(recovered_lambda)

        # Step (3): Obtain u from recovered b, and update policy from the newly obtained u.
        curr_u, curr_lam = solve_HJB_forward(models_u, collocs_u, historical_m[k], historical_q[k-1], Mu, Ju, Qu,
                                             lagrangian_1d, b=recovered_b, eps=eps)
        historical_u.append(curr_u)

        new_q, dq = second_diff_RFM_function(historical_u[k])
        historical_q.append(new_q)

    return historical_b[-1], historical_lam[-1]


def test():
    Mu, Mm, Mb = 4, 4, 5
    Ju, Jm, Jb = 20, 20, 30
    Il = 400
    Qu, Qm, Qb = 40, 40, 60

    def true_b(x):
        return 0.1 * (sin(2 * pi * x - sin(4 * pi * x)) + exp(cos(2 * pi * x)))

    m_true, u_true, _ = policy_iteration(Mu, Ju, Mm, Jm, Qu, Qm, b=true_b, n_iters=4)
    recovered_b, recovered_lam = PI_stationary(u_true, Mu, Mm, Mb, Ju, Jm, Jb, Il, Qu, Qm, Qb)

    # Plot the recovered b against true b
    plot_RFM_vs_true(recovered_b, true_b, recovered_lam,"b")


test()
