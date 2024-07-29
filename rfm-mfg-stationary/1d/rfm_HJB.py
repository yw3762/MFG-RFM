import numpy as np
import torch
from scipy.linalg import lstsq, pinv

from utils.utils_1d import init_rfm, lagrangian_1d, evaluate_RFM_1d


def solve_hjb_1d(models_fp, w_fp, M_p, J_n, Q, q, eps=0.3, tau=1e-8, plot=False, moore=False):
    """
    This function solves the HJB PDE in second step of policy iteration algorithm for ergodic 1d MFG

    The equation is:
    $-\varepsilon\frac{d^2m^{(k)}}{dx^2}-\frac{dm^{(k)}q^{(k)}}{dx}=0$ on $\mathbb{T}^1 = [0,1]$
    subject to the condition:
        1. (Probability density) $\int m(x)dx = 1$
        2. (Non-negativity) $m \geq 0$
        3. (Periodicity) $m(0) = m(1)$

    We identify the 1d-torus with [0,1] with identified endpoints, and write u instead of m for consistency.

    :param models_fp: the RFM solution models for Fokker-Planck PDE
    :param w_fp: the weights for Fokker-Planck RFM model
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in a partition
    :param Q: number of collocation points inside a partition
    :param q: the policy in MFG system
    :param dq: the numerical derivative of policy q
    :param eps: diffusion constant, i.e. the constant before Lagrangian in MFG system
    :param tau: convergence tolerance constant
    :param plot: whether to plot the solution or oot
    :param moore: whether to use Moore-Penrose inverse or not
    :return: ...
    """
    models_hjb, collocation_pts = init_rfm(M_p, J_n, Q)

    A, f = get_lstsq_system_HJB(models_hjb, collocation_pts, models_fp, w_fp, M_p, J_n, Q, eps, q)

    # Solve
    if moore:
        A_inv = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(A_inv, f)
    else:
        w = lstsq(A, f)[0]

    return models_hjb, collocation_pts, w


def get_lstsq_system_HJB(models, points, models_fp, w_fp, M_p, J_n, Q, eps, q, lam):
    """
    Calculate the matrix A and vector f in linear least square 'Au=f' associated with the Fokker-Planck PDE
    :param models: A list of local RFM models, one for each partition. Think of each model as a map R -> R^{J_n}
    :param points: Each element in this variable is a list of collocation points for a partition
    :param models_fp: the RFM solution models for Fokker-Planck PDE
    :param w_fp: the weights for Fokker-Planck RFM model
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in each partition, each RF basis function is a RFM_Rep object
    :param Q: number of collocation points inside a partition
    :param eps: diffusion constant, i.e. the constant before Lagrangian in MFG system
    :param q: the policy in MFG system
    :param dq: the numerical derivative of policy q
    :param lam: the erdogic constant \lambda
    :return: matrix A and vector f for the linear least square system
    """

    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])
    A_constraints = np.zeros([2, M_p * J_n])
    f = np.zeros([M_p * Q + 2, 1])

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m
            out = models[m](points[k])
            values = out.detach().numpy()

            # Compute first and second order derivative du/dx and d^2u/dx^2
            grads = []
            grads_2 = []

            q_du = []
            Lq = []
            for i in range(J_n):
                # Compute gradient of i-th basis function
                g_1 = torch.autograd.grad(outputs=out[:, i], inputs=points[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=True, retain_graph=True)[0]
                # Remove dims of size 1, unrequire gradients, then convert to np.array
                grads.append(g_1.squeeze().detach().numpy())

                # Compute second order gradient for i-th basis function
                g_2 = torch.autograd.grad(outputs=g_1[:, 0], inputs=points[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=False, retain_graph=True)[0]
                grads_2.append(g_2.squeeze().detach().numpy())

                q_du.append(grads[i] * q[k])

            grads = np.array(grads).T
            grads_2 = np.array(grads_2).T
            q_du = np.array(q_du).T
            Lq = lagrangian_1d(points[k], q[k])

            Lu = - eps * grads_2 + q_du - Lq

            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lu[:Q, :] - lam

            # Periodicity constraint, evaluate on boundary
            if k == 0:
                A_constraints[0, m * J_n: (m + 1) * J_n] = values[0, :]
            elif k == M_p - 1:
                A_constraints[0, m * J_n: (m + 1) * J_n] -= values[Q, :]

            # Normalization constraint:
            for i in range(Q):
                A_constraints[1, m * J_n: (m + 1) * J_n] += values[i, :]

        # The f-side of discretized Lu=f system
        m = evaluate_RFM_1d(models_fp, w_fp, points)
        f[k * Q:(k + 1) * Q, :] = m ** 2  # The coupling term is F(m) = m^2

    A = np.concatenate((A_pde, A_constraints), axis=0)
    f[-1] = 0  # Normalize to 0

    return A, f
