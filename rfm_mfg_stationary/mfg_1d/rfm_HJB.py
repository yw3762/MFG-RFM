import numpy as np
import torch
from scipy.linalg import lstsq, pinv

from utils.utils_1d import lagrangian_1d, evaluate_RFM_1d, differentiate_RFM_1d


def solve_hjb_1d(models_hjb, w_hjb, collocs, m_func, q_func, M_p, J_n, Q, eps=0.3, moore=False):
    """
    This function solves the HJB PDE in second step of policy iteration algorithm for ergodic mfg_1d MFG

    The equation is:
    $-\epsilon\frac{d^2m^{(k)}}{dx^2}-\frac{dm^{(k)}q^{(k)}}{dx}=0$ on $\mathbb{T}^1 = [0,1]$
    subject to the condition:
        1. (Probability density) $\int m(x)dx = 1$
        2. (Non-negativity) $m \geq 0$
        3. (Periodicity) $m(0) = m(1)$

    We identify the mfg_1d-torus with [0,1] with identified endpoints, and write u instead of m for consistency.

    :param models_hjb: RFM model for HJB PDE
    :param w_hjb: weights for HJB RFM from previous iteration
    :param collocs: collocation points for HJB PDE
    :param models_fp: the RFM solution models for Fokker-Planck PDE
    :param w_fp: the weights for Fokker-Planck RFM model
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in a partition
    :param Q: number of collocation points inside a partition
    :param eps: diffusion constant, i.e. the constant before Lagrangian in MFG system
    :param moore: whether to use Moore-Penrose inverse or not
    :return: ...
    """
    q = [q_func(collocs[i]).view(-1) for i in range(len(collocs))]
    A, f = get_lstsq_system_HJB(models_hjb, collocs, m_func, M_p, J_n, Q, q, eps)

    # Solve
    if moore:
        A_inv = pinv(A)  # moore-penrose inverse, shape: (n_units, n_colloc+2)
        w = np.matmul(A_inv, f)
    else:
        w = lstsq(A, f)[0]

    w = w.reshape((M_p, J_n))
    return torch.tensor(w)


def get_lstsq_system_HJB(models_hjb, points, m_func, M_p, J_n, Q, q, eps, lam=0):
    """
    Calculate the matrix A and vector f in linear least square 'Au=f' associated with the Fokker-Planck PDE

    :param models_hjb: A list of local RFM models, one for each partition. Think of each model as a map R -> R^{J_n}
    :param points: Each element in this variable is a list of collocation points for a partition
    :param m_func: Numerical solution to FP as a function
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in each partition, each RF basis function is an RFM_Rep object
    :param Q: number of collocation points inside a partition
    :param eps: diffusion constant, i.e. the constant before Lagrangian in MFG system
    :param q: the policy in MFG system
    :param lam: the ergodic constant \lambda
    :return: matrix A and vector f for the linear least square system
    """

    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])
    A_constraints = np.zeros([2, M_p * J_n])
    f = np.zeros([M_p * Q + 2, 1])

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m (HJB) and M_m (FP)
            out_hjb = models_hjb[m](points[k])
            values_hjb = out_hjb.detach().numpy()

            # Compute first and second order derivative du/dx and d^2u/dx^2 for HJB
            grads_2_hjb = []
            q_du = [] # Compute the (q * Du) term

            for i in range(J_n):
                # Compute gradient of i-th basis function
                g_1 = torch.autograd.grad(outputs=out_hjb[:, i], inputs=points[k],
                                          grad_outputs=torch.ones_like(out_hjb[:, i]),
                                          create_graph=True, retain_graph=True)[0]

                # Compute second order gradient for i-th basis function
                g_2 = torch.autograd.grad(outputs=g_1[:, 0], inputs=points[k],
                                          grad_outputs=torch.ones_like(out_hjb[:, i]),
                                          retain_graph=True)[0]
                grads_2_hjb.append(g_2.squeeze().detach().numpy())

                q_du.append((g_1.squeeze() * q[k]).detach().numpy())

            grads_2_hjb = np.array(grads_2_hjb).T  # grads[j,i] = f''_{mi}(points[k, j])
            q_du = np.array(q_du).T  # q_du[j, i] = f'_{mi}(points[k, j]) * q(points[k, j])

            # Impose PDE condition: Lu = -eps * du^2/dx^2 - div(u * q)
            # Lu[j, i] =  - eps * f'_{mi}(points[k, j]) + f'_{mi}(points[k, j]) * q(points[k, j])
            Lu = - eps * grads_2_hjb + q_du  # shape=(Q+1, J_n)

            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lu[:Q, :] + lam

            # Periodicity constraint, evaluate on boundary
            if k == 0:
                A_constraints[0, m * J_n: (m + 1) * J_n] = values_hjb[0, :]
            elif k == M_p - 1:
                A_constraints[0, m * J_n: (m + 1) * J_n] -= values_hjb[Q, :]

            # Normalization constraint:
            for i in range(Q):
                A_constraints[1, m * J_n: (m + 1) * J_n] += values_hjb[i, :]

        # The f-side of discretized Lu=f system
        Lq = torch.cat(lagrangian_1d(points[k], q[k]))
        Fm = m_func(points[k]).view(-1) ** 2
        # Fm = evaluate_RFM_1d(models_fp, w_fp, points[k]) ** 2  # The coupling term is F(m) = m^2
        summed = Fm + Lq
        trimmed = summed[:Q].detach().numpy().reshape(-1, 1)
        f[k * Q:(k + 1) * Q, :] = trimmed

    A = np.concatenate((A_pde, A_constraints), axis=0)
    f[-1] = 0  # Normalize to 0

    return A, f
