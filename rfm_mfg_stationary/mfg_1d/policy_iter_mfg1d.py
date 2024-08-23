import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Callable, Any

from classes.error_tracker import ErrorTracker1D
from utils.types import create_error_array, ErrorArray
from utils.utils_1d import init_rfm, calculate_error_fp, calculate_error_hjb, plot_RFM_1d, constraint_test, \
    RFM_function_factory, update_l1_err_test, plot_errors
from rfm_mfg_stationary.mfg_1d.rfm_FP import solve_fokker_planck_1d
from rfm_mfg_stationary.mfg_1d.rfm_HJB import solve_hjb_1d


def solve_1d_stationary_mfg(M_p_hjb, J_n_hjb, M_p_fp, J_n_fp, Q_hjb, Q_fp, n_iters=20, eps=0.3, tau=1e-6,
                            intermediate_plot=False, random_init_q=False):
    """
    Solve the mfg_1d stationary mean-field game with policy iteration, in each iteration, the two PDE systems (FP, HJB)
    are numerically solved using RFM method.

    :param M_p_hjb: number of partitions for HJB
    :param J_n_hjb: number of RF basis functions in a partition for HJB
    :param M_p_fp: number of partitions for FP
    :param J_n_fp: number of RF basis functions in a partition for FP
    :param Q_hjb: number of collocation points inside a partition for HJB
    :param Q_fp: number of collocation points inside a partition for FP
    :param n_iters: number of iterations
    :param eps: diffusion constant, i.e. the constant before Lagrangian in MFG system
    :param tau: convergence tolerance constant
    :param intermediate_plot: whether to plot intermediate plot
    :param random_init_q: whether to randomly initialize the policy in the beginning
    :return: solved m and u
    """

    # fix datatype
    torch.set_default_dtype(torch.float64)

    # Initialize solutions for FP and HJB with zero weights
    models_fp, collocs_fp = init_rfm(M_p_fp, J_n_fp, Q_fp)
    models_hjb, collocs_hjb = init_rfm(M_p_hjb, J_n_hjb, Q_hjb)
    w_hjb = torch.zeros((M_p_hjb, J_n_hjb))
    if random_init_q:
        w_hjb = w_hjb.uniform_(-1, 1)

    w_fp = torch.zeros((M_p_fp, J_n_fp))

    # Record all intermediate functions
    historical_m: npt.NDArray[Callable[[Any], Any]] = np.empty(n_iters+1, dtype=object)
    historical_u: npt.NDArray[Callable[[Any], Any]] = np.empty(n_iters+1, dtype=object)

    historical_m[0] = RFM_function_factory(models_fp, w_fp)
    historical_u[0] = RFM_function_factory(models_fp, w_hjb)

    # Record all intermediate error data
    errors_fp: ErrorArray = create_error_array(n_iters + 1)
    errors_hjb: ErrorArray = create_error_array(n_iters + 1)

    plot_RFM_1d(historical_u[0], "u^0")

    actual_n_iters = n_iters

    for _ in range(1, n_iters+1):
        # Solve Fokker-Planck equation in policy iteration method
        print("Iteration {}".format(_))
        start_time = time.time()
        w_fp = solve_fokker_planck_1d(models_fp, collocs_fp, models_hjb, w_hjb, M_p_fp, J_n_fp, Q_fp)
        finish_fp = time.time()
        print(f"FP took: {finish_fp - start_time:.6f} seconds")

        # Record solution and error for Fokker-Planck equation
        historical_m[_] = RFM_function_factory(models_fp, w_fp)
        errors_fp[_] = ErrorTracker1D(
            calculate_error_fp(historical_m[_], historical_u[_-1], plot=intermediate_plot),
            *constraint_test(historical_m[_]),  # * for tuple unpacking in constructor call
            update_l1_err_test(historical_m[_-1], historical_m[_])
        )
        plot_RFM_1d(historical_m[_], "m^" + str(_))

        # Solve HJB Equation in Policy iteration method
        start_time = time.time()
        w_hjb = solve_hjb_1d(models_hjb, w_hjb, collocs_hjb, models_fp, w_fp, M_p_hjb, J_n_hjb, Q_hjb)
        finish_hjb = time.time()
        print(f"HJB took: {finish_hjb - start_time:.6f} seconds")

        # Record solution and error for HJB equation
        historical_u[_] = RFM_function_factory(models_hjb, w_hjb)
        errors_hjb[_] = ErrorTracker1D(
            calculate_error_hjb(historical_u[_], historical_u[_-1], historical_m[_], plot=intermediate_plot),
            *constraint_test(historical_u[_]),  # * for tuple unpacking in constructor call
            update_l1_err_test(historical_u[_-1], historical_u[_])
        )
        plot_RFM_1d(historical_u[_], "u^" + str(_))

        # Plotting u_1 to true u_1 together
        if _ == 1:
            n_pts = 2000
            pts = torch.linspace(0, 1, n_pts, dtype=torch.float64).reshape(-1, 1)
            u1_true = - 1 / (12 * eps) + pts / (2 * eps) + np.sin(2 * np.pi * pts) / (4 * eps * np.pi ** 2) + np.cos(
                4 * np.pi * pts) / (16 * eps * np.pi ** 2) - pts ** 2 / (2 * eps)

            plt.figure(figsize=(10, 6))
            plt.plot(pts, u1_true, label='true u1')
            plt.plot(pts, (historical_u[_])(pts).detach().numpy(), label='numerical u1')
            plt.xlabel('x')
            plt.ylabel('u^1(x)')
            plt.title('Numerical u^1 vs true u^1')
            plt.legend()
            plt.show()

        # loop termination condition once accuracy is small
        if errors_fp[_].errors['l1-error'] < tau and errors_hjb[_].errors['l1-error'] < tau:
            actual_n_iters = _
            break

    plot_errors(errors_fp, actual_n_iters, label="FP")
    plot_errors(errors_hjb, actual_n_iters, label="HJB")

    return historical_m[-1], historical_u[-1]
