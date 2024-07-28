import numpy as np
import torch
from scipy.linalg import lstsq, pinv

from utils.utils_1d import get_differential_1d, init_rfm


def solve_fokker_planck_1d(M_p, J_n, Q, q, dq, eps=0.3, tau=1e-8, plot=False, moore=False):
    """
    This function solves the Fokker-Planck PDE in first step of policy iteration algorithm for ergodic 1d MFG

    The equation is:
    $-\varepsilon\frac{d^2m^{(k)}}{dx^2}-\frac{dm^{(k)}q^{(k)}}{dx}=0$ on $\mathbb{T}^1 = [0,1]$
    subject to the condition:
        1. (Probability density) $\int m(x)dx = 1$
        2. (Non-negativity) $m \geq 0$
        3. (Periodicity) $m(0) = m(1)$

    We identify the 1d-torus with [0,1] with identified endpoints, and write u instead of m for consistency.
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
    models, collocation_pts = init_rfm(M_p, J_n, Q)

    dq = get_differential_1d(q, collocation_pts)

    A, f = get_lstsq_system_fp(models, collocation_pts, M_p, J_n, Q, eps, q, dq)

    # Solve
    if moore:
        inv_coeff_mat = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(inv_coeff_mat, f)
    else:
        w = lstsq(A, f)[0]

    return models, collocation_pts, w


def get_lstsq_system_fp(models, points, M_p, J_n, Q, eps, q, dq):
    """
    Calculate the matrix A and vector f in linear least square 'Au=f' associated with the Fokker-Planck PDE
    :param models: A list of local RFM models, one for each partition. Think of each model as a map R -> R^{J_n}
    :param points: Each element in this variable is a list of collocation points for a partition
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in each partition, each RF basis function is a RFM_Rep object
    :param Q: number of collocation points inside a partition
    :param eps: diffusion constant, i.e. the constant before Lagrangian in MFG system
    :param q: the policy in MFG system
    :param dq: the numerical derivative of policy q
    :return: matrix A and vector f for the linear least square system
    """

    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])
    A_boundary = np.zeros([2, M_p * J_n])
    f = np.zeros([M_p * Q + 2, 1])

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m
            out = models[m](points[k])
            values = out.detach().numpy()

            # Compute first and second order derivative dm/dx and d^2m/dx^2
            grads = []
            grads_2 = []

            div = []
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

                # In d=1, div(u*q) = d(u*q)/dx = du/dx * q + u * dq/dx
                # TODO: Check if the dimension matches
                div.append(grads[i] * q[k] + values[:, i] * dq[k])

            grads = np.array(grads).T
            grads_2 = np.array(grads_2).T
            div = np.array(div).T

            # Impose PDE condition: Lu = -eps * du^2/dx^2 - div(u * q)
            Lu = - eps * grads_2 - div

            # TODO: Think of a way to impose the three conditions
            #  1. (Probability density) $\int m(x)dx = 1$
            #  2. (Non-negativity) $m \geq 0$
            #  3. (Periodicity) $m(0) = m(1)$

            # TODO: Specify A_pde
            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lu[:Q,:]

            # TODO: Specify constraints (none in FP)

    A = np.concatenate((A_pde, A_boundary), axis=0)
    return A, f
