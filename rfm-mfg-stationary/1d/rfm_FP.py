import numpy as np
import torch
from scipy.linalg import lstsq, pinv

from utils.utils_1d import RFM_rep, weights_init, init_local_RFM1d
from utils.config import INTERVAL_LENGTH

# fix random seed
torch.set_default_dtype(torch.float64)

def solve_fokker_planck_1d(M_p, J_n, Q, q, eps=0.3, tau=1e-8, plot=False, moore=False):
    """
    This function solves the Fokker-Planck PDE in first step of policy iteration algorithm for Ergodic 1d MFG

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
    :param eps: diffusion constant, i.e. the constant before Lagrangian in MFG system
    :param tau: convergence tolerance constant
    :param plot: whether to plot the solution or oot
    :param moore: whether to use Moore-Penrose inverse or not
    :return: ...
    """
    models, collocation_pts = init_rfm(M_p, J_n, Q)

    #TODO: specify how q should be passed
    A, f = get_lstsq_system_fp(models, collocation_pts, M_p, J_n, Q, eps, q)

    # Solve
    if moore:
        inv_coeff_mat = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(inv_coeff_mat, f)
    else:
        w = lstsq(A,f)[0]

    return models, collocation_pts, w


def V(x):
    """
    Analytical bounded potential function

    The associated Hamiltonian is H(x, p) = 1/2 |p|^2 - V(x), where p is the variable for Dx
    """
    return np.sin(2. * np.pi * x) + np.cos(4. * np.pi * x)


def init_rfm(M_p, J_n, Q):
    """
    Define the RFM model on each partition and their collocation points.
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in a partition
    :param Q: number of collocation points inside a partition
    :return: 1. a list of local NNs, one for each partition
             2. lists of collocation points for every partition
    """
    models = []
    points = []
    for k in range(M_p):
        # Define RFM model in each partition, in 1d, partition is just an interval [x_min, x_max]
        x_min = INTERVAL_LENGTH / M_p * k
        x_max = INTERVAL_LENGTH / M_p * (k + 1)
        models.append(init_local_RFM1d(J_n, x_min, x_max))

        # Within each partition, get the boundary points (1d) as a column vector
        points.append(torch.tensor(np.linspace(x_min, x_max, Q + 1), requires_grad=True).reshape([-1, 1]))
    return models, points


def get_lstsq_system_fp(models, points, M_p, J_n, Q, eps, q):
    """
    Calculate the matrix A and vector f in linear least square 'Au=f' associated with the Fokker-Planck PDE
    :param models: A list of local RFM models, one for each partition
    :param points: Each element in this variable is a list of collocation points for a partition
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in each partition, each RF basis function is a RFM_Rep object
    :param Q: number of collocation points inside a partition
    :return: matrix A and vector f for the linear least square system
    """
    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])
    A_boundary = np.zeros([2, M_p * J_n])
    f = np.zeros([M_p*Q + 2, 1])

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction_points of partition-k on RFM of U_m
            out = models[m](points[k])
            values = out.detach().numpy()

            # first and second order derivative dm/dx and d^2m/dx^2
            grads = []
            grads_2 = []
            for i in range(J_n):
                # Compute gradient of i-th basis function
                g_1 = torch.autograd.grad(outputs=out[:, i], inputs=points[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=True, retain_graph=True)[0]
                # remove dims of size 1, unrequire gradients, convert to np.array
                grads.append(g_1.squeeze().detach().numpy())

                # Compute second order gradient for i-th basis function
                g_2 = torch.autograd.grad(outputs=g_1[:, 0], inputs=points[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=False, retain_graph=True)[0]
                grads_2.append(g_2.squeeze().detach().numpy())
            grads = np.array(grads).T
            grads_2 = np.array(grads_2).T

            # TODO: Implement Lu and think about input format of policy q, note
            #  Lu should be -eps * grads_2 - div(grad * q), and in dimension 1,
            #  the div is just d/dx
            Lu = - eps * grads_2

            # TODO: Think of a way to impose the three conditions
            #  1. (Probability density) $\int m(x)dx = 1$
            #  2. (Non-negativity) $m \geq 0$
            #  3. (Periodicity) $m(0) = m(1)$

            # TODO: Specify A_pde and A_boundary

    A = np.concatenate((A_pde, A_boundary), axis=0)
    return A, f