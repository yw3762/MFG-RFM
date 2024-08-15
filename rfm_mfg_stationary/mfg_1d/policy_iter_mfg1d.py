import torch
import numpy as np

from utils.utils_1d import init_rfm, calculate_error_fp, calculate_error_hjb
from rfm_mfg_stationary.mfg_1d.rfm_FP import solve_fokker_planck_1d
from rfm_mfg_stationary.mfg_1d.rfm_HJB import solve_hjb_1d


def solve_1d_stationary_mfg(M_p_hjb, J_n_hjb, M_p_fp, J_n_fp, Q_hjb, Q_fp, n_iters=20, eps=0.3, tau=1e-8):
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
    w_hjb = np.zeros((M_p_hjb, J_n_hjb))
    w_fp = np.zeros((M_p_fp, J_n_fp))

    errors_fp = []
    errors_hjb = []

    for _ in range(n_iters):
        print("Iteration {}".format(_ + 1))
        w_fp = solve_fokker_planck_1d(models_fp, collocs_fp, models_hjb, w_hjb, M_p_fp, J_n_fp, Q_fp)
        error_fp = calculate_error_fp(models_fp, w_fp, models_hjb, w_hjb)

        w_hjb = solve_hjb_1d(models_hjb, w_hjb, collocs_hjb, models_fp, w_fp, M_p_hjb, J_n_hjb, Q_hjb)
        error_hjb = calculate_error_hjb(models_fp, w_fp, models_hjb, w_hjb)

    return models_fp, w_fp, models_hjb, w_hjb
