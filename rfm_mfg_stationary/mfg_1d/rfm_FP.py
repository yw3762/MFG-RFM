import numpy as np
import torch
from scipy.linalg import lstsq, pinv

from utils.utils_1d import second_derivative_RFM_1d


def solve_fokker_planck_1d(models, collocs, models_u, w_u, M_p, J_n, Q, eps=0.3, tau=1e-8, plot=False, moore=False):
    """
    This function solves the Fokker-Planck PDE in first step of policy iteration algorithm for ergodic mfg_1d MFG

    The equation is:
    $-\varepsilon\frac{d^2m^{(k)}}{dx^2}-\frac{dm^{(k)}q^{(k)}}{dx}=0$ on $\mathbb{T}^1 = [0,1]$
    subject to the condition:
        1. (Probability density) $\int m(x)dx = 1$
        2. (Non-negativity) $m \geq 0$
        3. (Periodicity) $m(0) = m(1)$

    We identify the mfg_1d-torus with [0,1] with identified endpoints, and write u instead of m for consistency.
    :param models:
    :param collocs:
    :param models_u:
    :param w_u:
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
    q, dq = second_derivative_RFM_1d(models_u, w_u, collocs)

    A, f = get_lstsq_system_fp(models, collocs, models_u, w_u, M_p, J_n, Q, q, dq, eps)

    # Solve
    if moore:
        A_inv = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(A_inv, f)
    else:
        w = lstsq(A, f)[0]

    w = w.reshape((M_p, J_n))
    return w


def get_lstsq_system_fp(models, points, models_u, w_u, M_p, J_n, Q, q, dq, eps):
    """
    Calculate the matrix A and vector f in linear least square 'Au=f' associated with the Fokker-Planck PDE
    :param models: A list of local RFM models, one for each partition. Think of each model as a map R -> R^{J_n}
    :param points: Each element in this variable is a list of collocation points for a partition
    :param models_u:
    :param w_u:
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

    # TODO: For the moment, we assume non-negativity constraint in RFM also follows from normalization constraint,
    #   We should check if this is true afterward.
    # NOTE: It seems to be true
    A_constraints = np.zeros([2, M_p * J_n])  # one for boundary, one for normalization -> 2 in total
    f = np.zeros([M_p * Q + 2, 1])

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m
            out = models[m](points[k])
            values = out.detach().numpy()

            # Compute first and second order derivative dm/dx and d^2m/dx^2
            grads = []
            grads_2 = []

            # Compute divergence term div(q m)
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
                div.append(grads[i] * q[k] + values[:, i] * dq[k])

            grads = np.array(grads).T
            grads_2 = np.array(grads_2).T
            div = np.array(div).T

            # Impose PDE condition: Lu = -eps * du^2/dx^2 - div(u * q)
            Lu = - eps * grads_2 - div

            # Specifying A_pde
            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lu[:Q, :]

            # Periodicity constraint, evaluate on boundary
            if k == 0:
                A_constraints[0, m * J_n: (m + 1) * J_n] = values[0, :]
            elif k == M_p - 1:
                A_constraints[0, m * J_n: (m + 1) * J_n] -= values[Q, :]

            # Normalization constraint:
            for i in range(Q):
                A_constraints[1, m * J_n: (m + 1) * J_n] += values[i, :]

    A = np.concatenate((A_pde, A_constraints), axis=0)
    f[-1] = 1 * M_p * Q  # normalize to 1

    return A, f
