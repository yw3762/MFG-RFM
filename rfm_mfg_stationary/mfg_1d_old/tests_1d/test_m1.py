import numpy as np
import time
import torch
from scipy.linalg import lstsq, pinv

from rfm_mfg_stationary.mfg_1d_old.utils.utils_1d import second_derivative_RFM_1d, set_seed, init_rfm, plot_RFM_1d


def solve_fokker_planck_1d_simplified(models, collocs, models_u, w_u, M_p, J_n, Q, eps=0.3, tau=1e-8, plot=False, moore=False):
    """
    This function solves the Fokker-Planck PDE in first step of policy iteration algorithm for ergodic mfg_1d_old MFG

    The equation is:
    $-\varepsilon\frac{d^2m^{(k)}}{dx^2}-\frac{dm^{(k)}q^{(k)}}{dx}=0$ on $\mathbb{T}^1 = [0,1]$
    subject to the condition:
        1. (Probability density) $\int m(x)dx = 1$
        2. (Non-negativity) $m \geq 0$
        3. (Periodicity) $m(0) = m(1)$

    We identify the mfg_1d_old-torus with [0,1] with identified endpoints, and write u instead of m for consistency.
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
    start_time = time.time()
    q, dq = second_derivative_RFM_1d(models_u, w_u, collocs)
    end_time = time.time()
    print(f"second_diff took: {end_time-start_time:.6f} seconds")

    start_time = time.time()
    A, f = get_lstsq_system_fp(models, collocs, models_u, w_u, M_p, J_n, Q, q, dq, eps)
    end_time = time.time()
    print(f"get_lstsq_system took: {end_time - start_time:.6f} seconds")

    # Solve
    start_time = time.time()
    if moore:
        A_inv = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(A_inv, f)
    else:
        w = lstsq(A, f)[0]
    end_time = time.time()
    print(f"Solve system took: {end_time - start_time:.6f} seconds")

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
            # values[i,j] = f_{mj}(points[k,i]), where f_{mj} is feature function
            values = out.detach().numpy()  # shape: (Q+1, J_n),

            # Compute first and second order derivative dm/dx and d^2m/dx^2
            grads_2 = []

            # Compute divergence term div(q m)
            div = []

            for i in range(J_n):
                # Compute gradient of i-th basis function
                g_1 = torch.autograd.grad(outputs=out[:, i], inputs=points[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=True, retain_graph=True)[0]

                # Compute second order gradient for i-th basis function
                g_2 = torch.autograd.grad(outputs=g_1[:, 0], inputs=points[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          retain_graph=True)[0]
                grads_2.append(g_2.squeeze().detach().numpy())

                # In d=1, div(m*q) = d(m*q)/dx = m' * q + m * q'
                div.append((g_1.squeeze() * q[k] + out[:, i] * dq[k]).detach().numpy())

            grads_2 = np.array(grads_2).T  # grads[j,i] = f''_{mi}(points[k, j])
            # div = np.array(div).T  # div[j,i] = div(f_{mi}q)(points[k, j])

            # Impose PDE condition: Lm = -eps * dm^2/dx^2 - div(m * q)
            Lm = - eps * grads_2 # - div  # Lm[j,i] = Lm(points[k, j]) with i-th factor of m

            # Specifying A_pde
            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lm[:Q, :]

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


def test_m1():
    set_seed(1000)
    M_p_hjb, J_n_hjb = 4, 100
    M_p_fp, J_n_fp = 4, 100
    Q_hjb, Q_fp = 100, 100

    models_fp, collocs_fp = init_rfm(M_p_fp, J_n_fp, Q_fp)
    models_hjb, collocs_hjb = init_rfm(M_p_hjb, J_n_hjb, Q_hjb)
    w_hjb = np.zeros((M_p_hjb, J_n_hjb))
    w_fp = np.zeros((M_p_fp, J_n_fp))

    w_fp = solve_fokker_planck_1d_simplified(models_fp, collocs_fp, models_hjb, w_hjb, M_p_fp, J_n_fp, Q_fp, plot=True)

    plot_RFM_1d(models_fp, w_fp, "m")


test_m1()
