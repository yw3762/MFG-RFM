import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import lstsq, pinv

from utils.utils_1d import plot_RFM_1d, set_seed, RFM_function_factory, init_rfm


def solve_fp_1d(models, collocs, models_u, w_u, M_p, J_n, Q, eps=0.3, tau=1e-8, plot=False, moore=False):
    u = RFM_function_factory(models_u, w_u)

    A, f = get_lstsq_system_fp(models, collocs, M_p, J_n, Q, u, eps)
    if moore:
        A_inv = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(A_inv, f)
    else:
        w = lstsq(A, f)[0]
    w = w.reshape((M_p, J_n))
    return w


def get_lstsq_system_fp(models, points, M_p, J_n, Q, u, eps):
    A_pde = np.zeros([M_p * Q, M_p * J_n])
    A_constraints = np.zeros([2, M_p * J_n])
    f = np.zeros([M_p * Q + 2, 1])

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m (HJB) and M_m (FP)
            out = models[m](points[k])
            values = out.detach().numpy()

            grads_2 = []
            div = []

            for i in range(J_n):
                g_1 = torch.autograd.grad(outputs=out[:, i], inputs=points[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=True, retain_graph=True)[0]

                g_2 = torch.autograd.grad(outputs=g_1[:, 0], inputs=points[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=False, retain_graph=True)[0]
                grads_2.append(g_2.squeeze().detach().numpy())

                div.append(g_1.squeeze().detach().numpy() * q[k] + values[:, i] * dq[k])

            grads_2 = np.array(grads_2).T  # grads[j,i] = f''_{mi}(points[k, j])
            div = np.array(div).T  # div[j,i] = div(f_{mi}q)(points[k, j])

            Lm = - eps * grads_2 - div

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


def solve_hjb_1d(models_hjb, w_hjb, collocs, models_fp, w_fp, M_p, J_n, Q, eps=0.3, tau=1e-8, plot=False, moore=False):
    m = RFM_function_factory(models_fp, w_fp)
    old_u = RFM_function_factory(models_hjb, w_hjb)

    A, f = get_lstsq_system_hjb(models_hjb, collocs, M_p, J_n, Q, old_u, m, eps)
    if moore:
        A_inv = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(A_inv, f)
    else:
        w = lstsq(A, f)[0]
    w = w.reshape((M_p, J_n))
    return w


def get_lstsq_system_hjb(models_hjb, collocs, M_p, J_n, Q, old_u, m, eps):
    pass

def solve_1d_stationary_mfg_new(M_p_hjb, J_n_hjb, M_p_fp, J_n_fp, Q_hjb, Q_fp, n_iters=20, eps=0.3, tau=1e-8,
                            intermetidate_plot=False):
    torch.set_default_dtype(torch.float64)

    # Initialize solutions for FP and HJB with zero weights
    models_fp, collocs_fp = init_rfm(M_p_fp, J_n_fp, Q_fp)
    models_hjb, collocs_hjb = init_rfm(M_p_hjb, J_n_hjb, Q_hjb)
    w_hjb = np.zeros((M_p_hjb, J_n_hjb))
    w_fp = np.zeros((M_p_fp, J_n_fp))

    for _ in range(n_iters):
        w_fp = solve_fp_1d(models_fp, collocs_fp, models_hjb, w_hjb, M_p_fp, J_n_fp, Q_fp)
        w_hjb = solve_hjb_1d(models_hjb, w_hjb, collocs_hjb, models_fp, w_fp, M_p_hjb, J_n_hjb, Q_hjb)

    return models_fp, w_fp, w_hjb, w_hjb


def main():
    set_seed(100)
    M_p_hjb, J_n_hjb = 8, 100
    M_p_fp, J_n_fp = 8, 100
    Q_hjb, Q_fp = 100, 100
    n_iters = 20

    models_fp, w_fp, models_hjb, w_hjb = solve_1d_stationary_mfg_new(M_p_hjb, J_n_hjb, M_p_fp, J_n_fp, Q_hjb, Q_fp, n_iters,
                                                                 intermetidate_plot=False)
    m = RFM_function_factory(models_fp, w_fp)
    u = RFM_function_factory(models_hjb, w_hjb)
    plot_RFM_1d(models_fp, w_fp, "m")
    plot_RFM_1d(models_hjb, w_hjb, "u")


main()
