
from rfm_FP import solve_fokker_planck_1d

def solve_1d_stationary_mfg(M_p, J_n, Q, n_iters=20, tau=1e-8):
    """
    Solve the 1d stationary mean-field game with policy iteration, in each iteration, the PDE is numerically solved
    using RFM method

    :param M_p: number of partitions
    :param J_n: number of RF basis functions in a partition
    :param Q: number of collocation points inside a partition
    :param eps: diffusion constant, i.e. the constant before Lagrangian in MFG system
    :param tau: convergence tolerance constant
    :return:
    """

    # Initialize policy
    q = []

    for _ in range(n_iters):
        models_fp, collocation_pts_fp, w_fp = solve_fokker_planck_1d(M_p, J_n, Q, q)
        # TODO: Implement the following algorithm:
        #  1. Generate numerical solution for FP on the grid
        #  2. Pass the solution on the grid to the coupling term F(m) = m^2
        #  3. Solve HJB equation with given coupling term
        #  4. Choose optimal policy q w.r.t numerical solution to FP and HJB

