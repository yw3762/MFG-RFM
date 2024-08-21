import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from utils.utils_1d import init_rfm, calculate_error_fp, calculate_error_hjb, plot_RFM_1d
from rfm_mfg_stationary.mfg_1d.rfm_FP import solve_fokker_planck_1d
from rfm_mfg_stationary.mfg_1d.rfm_HJB import solve_hjb_1d


def solve_1d_stationary_mfg(M_p_hjb, J_n_hjb, M_p_fp, J_n_fp, Q_hjb, Q_fp, n_iters=20, eps=0.3, tau=1e-8,
                            intermetidate_plot=False, random_init_q=False):
    """
    Solve the mfg_1d stationary mean-field game with policy iteration, in each iteration, the two PDE systems (FP, HJB)
    are numerically solved using RFM method.

    :param M_p: number of partitions
    :param J_n: number of RF basis functions in a partition
    :param Q: number of collocation points inside a partition
    :param eps: diffusion constant, i.e. the constant before Lagrangian in MFG system
    :param tau: convergence tolerance constant
    :return: optimal policy q
    """

    # fix datatype
    torch.set_default_dtype(torch.float64)

    # Initialize solutions for FP and HJB with zero weights
    models_fp, collocs_fp = init_rfm(M_p_fp, J_n_fp, Q_fp)
    models_hjb, collocs_hjb = init_rfm(M_p_hjb, J_n_hjb, Q_hjb)
    w_hjb = None
    if random_init_q:
        w_hjb = np.random.uniform(-1, 1, (M_p_hjb, J_n_hjb))
    else:
        w_hjb = np.zeros((M_p_hjb, J_n_hjb))

    w_fp = np.zeros((M_p_fp, J_n_fp))


    errors_fp = []
    errors_hjb = []
    plot_RFM_1d(models_hjb, w_hjb, "u")

    for _ in range(n_iters):
        print("Iteration {}".format(_ + 1))
        start_time = time.time()
        w_fp = solve_fokker_planck_1d(models_fp, collocs_fp, models_hjb, w_hjb, M_p_fp, J_n_fp, Q_fp)
        finish_fp = time.time()
        print(f"FP took: {finish_fp-start_time:.6f} seconds")
        errors_fp.append(calculate_error_fp(models_fp, w_fp, models_hjb, w_hjb, plot=intermetidate_plot))
        plot_RFM_1d(models_fp, w_fp, "m")

        start_time = time.time()
        old_w_hjb = w_hjb
        w_hjb = solve_hjb_1d(models_hjb, w_hjb, collocs_hjb, models_fp, w_fp, M_p_hjb, J_n_hjb, Q_hjb)
        finish_hjb = time.time()
        print(f"HJB took: {finish_hjb - start_time:.6f} seconds")
        errors_hjb.append(calculate_error_hjb(models_fp, w_fp, models_hjb, w_hjb, old_w_hjb, plot=True))

        plot_RFM_1d(models_hjb, w_hjb, "u")

    # plot cumulative error per iteration
    cumulative_errors_fp = []
    cumulative_errors_hjb = []
    for i in range(n_iters):
        cumulative_errors_fp.append(errors_fp[i].sum().item())
        cumulative_errors_hjb.append(errors_hjb[i].sum().item())
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_iters), cumulative_errors_fp, label='FP-Error')
    plt.plot(range(n_iters), cumulative_errors_hjb, label='HJB-Error')
    plt.xlabel('iterations')
    plt.ylabel('Error')
    plt.title('Cumulative Error of HJB and FP')
    plt.legend()
    plt.show()

    return models_fp, w_fp, models_hjb, w_hjb
