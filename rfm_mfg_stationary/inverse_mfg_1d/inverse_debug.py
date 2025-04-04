import torch
import numpy as np
from scipy.linalg import pinv, inv
from torch import sin, cos, pi, exp

from rfm_mfg_stationary.mfg_1d.policy_iter_1d import solve_FP_forward, policy_iteration
from rfm_mfg_stationary.mfg_1d.utils.utils_1d import init_rfm, RFM_function_factory, second_diff_RFM_function, \
    plot_RFM_1d, lagrangian_1d, plot_RFM_vs_true, set_seed
from rfm_mfg_stationary.inverse_mfg_1d.utils_inv_1d import RF_valuation, concatenate_collocs, \
    func_valuation, build_RFM_matrix, build_RFM_matrix_normalize


def inverse_HJB(models_u, models_b, collocs_u_cat, collocs_b_cat, m_func, q_func, F_l, Fb_avg, H_u, H_b,
                U_l, x_l, F_u, Mb, Ju, Jb, Qu, Qb,
                eps, regularization=False):
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
    q_b = q_func(collocs_b_cat)
    q_u = q_func(collocs_u_cat)
    q_x = q_func(x_l)

    W_b = (m_func(collocs_b_cat) ** 2 + lagrangian_1d(collocs_b_cat, q_b, None)).view(-1, 1).detach().numpy()
    W_u = (m_func(collocs_u_cat) ** 2 + lagrangian_1d(collocs_u_cat, q_u, None)).view(-1, 1).detach().numpy()

    #W_hat = np.vstack((W_u, [[0]]))
    W_hat = np.vstack((W_b, [[0]]))
    M_hat = build_RFM_matrix_normalize(models_u, q_b, collocs_b_cat, Mb, Qb, Ju, eps)
    M_hat_pinv = pinv(M_hat, atol=1e-4, rtol=1e-4)
    H = build_RFM_matrix(models_b, q_b, collocs_b_cat, Mb, Qb, Jb, eps)
    H_tilde = np.hstack((H, -1 * np.ones((H.shape[0], 1))))
    H_hat = np.vstack((H_tilde, np.zeros((1, H_tilde.shape[1]))))
    theta_hat = pinv(F_l @ M_hat_pinv @ H_hat) @ (U_l - F_l @ M_hat_pinv @ W_hat)

    theta_nj, recovered_lambda = theta_hat[:-1, 0].reshape((Mb, Jb)), theta_hat[-1, 0]

    Mu = len(models_u)
    w_nj = (M_hat_pinv @ H_hat @ theta_hat + M_hat_pinv @ W_hat).reshape((Mu, Ju))

    recovered_b = RFM_function_factory(models_b, torch.from_numpy(theta_nj))
    recovered_u = RFM_function_factory(models_u, torch.from_numpy(w_nj))

    return recovered_b, recovered_lambda, recovered_u


def inverse_PI_stationary(u_true, Mu, Mm, Mb, Ju, Jm, Jb, Il, Qu, Qm, Qb, n_iters=20, eps=0.3, tau=1e-6, plot=False):
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
    collocs_u_cat = concatenate_collocs(collocs_u)
    collocs_b_cat = concatenate_collocs(collocs_b)
    F_l = RF_valuation(models_u, x_l).detach().numpy()
    F_u = RF_valuation(models_u, collocs_u_cat).detach().numpy()
    F_b = RF_valuation(models_u, collocs_b_cat).detach().numpy()
    Fb_avg = np.mean(F_b, axis=0)
    H_b = RF_valuation(models_b, collocs_b_cat).detach().numpy()
    H_b = np.concatenate([H_b, -np.ones((H_b.shape[0], 1))], axis=1)
    H_u = RF_valuation(models_b, collocs_u_cat).detach().numpy()
    H_u = np.concatenate([H_u, -np.ones((H_u.shape[0], 1))], axis=1)
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
        # See if the u integrates to 0.
        recovered_b, recovered_lambda, recovered_u = inverse_HJB(models_u, models_b, collocs_u_cat, collocs_b_cat,
                                                                 historical_m[k], historical_q[k - 1], F_l, Fb_avg, H_u,
                                                                 H_b, U_l, x_l, F_u, Mb, Ju, Jb, Qu, Qb, eps,
                                                                 regularization=False)
        historical_b.append(recovered_b)
        historical_lam.append(recovered_lambda)
        plot_RFM_vs_true(recovered_u, u_true, recovered_lambda, "u")

        # Step (3): Obtain u from recovered b, and update policy from the newly obtained u.
        # TODO: Pass in u into eq (9) and (10), check u has conserved mass

        new_q, dq = second_diff_RFM_function(recovered_u)
        historical_q.append(new_q)

    # TODO: Check recovered b gives the same solution to the MFG, re-create the original data

    return historical_b[-1], historical_lam[-1]


def test():
    set_seed(100)
    Mu, Mm, Mb = 4, 4, 5
    Ju, Jm, Jb = 20, 20, 30
    Il = 400
    Qu, Qm, Qb = 40, 40, 60

    def true_b(x):
        return 0.1 * (sin(2 * pi * x - sin(4 * pi * x)) + exp(cos(2 * pi * x)))

    m_true, u_true, _ = policy_iteration(Mu, Ju, Mm, Jm, Qu, Qm, b=true_b, n_iters=20)
    recovered_b, recovered_lam = inverse_PI_stationary(u_true, Mu, Mm, Mb, Ju, Jm, Jb, Il, Qu, Qm, Qb)

    # Plot the recovered b against true b
    plot_RFM_vs_true(recovered_b, true_b, recovered_lam, "b")


test()
