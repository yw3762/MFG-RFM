import scipy.linalg
import torch
import pandas as pd
import scipy.sparse.linalg as spla
from torch import sin, cos, pi, exp
import torch.nn as nn
import numpy as np
import sympy as sym
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import random
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
from scipy.linalg import pinv, solve, lstsq


INTERVAL_LENGTH = 1.0  # we consider the domain to be the unit torus.


class RFM_rep(nn.Module):
    def __init__(self, in_features, J_n, x_max, x_min):
        super(RFM_rep, self).__init__()
        self.in_features = in_features  # num input features
        self.hidden_features = J_n  # width of hidden layer
        self.J_n = J_n  # J_n is the number of local RF functions
        self.x_min = x_min  # x_{nj} - r_{nj}
        self.x_max = x_max  # x_{nj} + r_{nj}
        self.a = 2. / (x_max - x_min)  # this is the 1/r_{nj}
        self.x_0 = (x_max + x_min) / 2  # center of partition.
        self.gap = INTERVAL_LENGTH  * (x_max - x_min) / 4.

        # Hidden layer is a simple linear FC layer passed to kernel function Tanh.
        self.hidden_layer = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features, bias=True),
            nn.Tanh()  # batch apply the above layer choice of kernel function
        )

    def forward(self, x):
        """
        The input x will be pass through the network in the following ways:
        1. Perform a change of variable x -> tilde{x}, i.e. y in the code
        2. Pass through the hidden layer (i.e. Linear layer + tanh), i.e. we obtain J_n RF functions \phi_nj(x)
        3. Glue the solution at x using partition of unity, i.e. we obtain
        """

        # preprocess the input x so that when the partition [x_min, x_max] overlaps the boundary points 0, or
        # INTERVAL_LENGTH, we map the points from other ends so that it's covered by the "identified PoU"
        if self.x_min == 0:
            x = torch.where(x >= 1 - self.gap, x - 1, x)
        elif self.x_max == INTERVAL_LENGTH:
            x = torch.where(x <= self.gap , x + 1, x)

        # y is the change of variable in normalized coordinate \tilde{x}
        tilde_x = self.a * (x - self.x_0)

        # pass the normalized variable into hidden-layer
        phi_nj = self.hidden_layer(tilde_x)  # phi_nj[i, j] = \phi_{n,j}(tilde_x[i])

        # location indicator
        d1 = (tilde_x >= -5 / 4) & (tilde_x < -3 / 4)
        d2 = (tilde_x >= -3 / 4) & (tilde_x < 3 / 4)
        d3 = (tilde_x >= 3 / 4) & (tilde_x < 5 / 4)

        # y_i are the PoU w.r.t each location of x
        sin_term = torch.sin(2 * np.pi * tilde_x)
        y1 = phi_nj * (1 + sin_term) / 2
        y2 = phi_nj
        y3 = phi_nj * (1 - sin_term) / 2

        values = d1 * y1 + d2 * y2 + d3 * y3
        return values


def weights_init(m):
    """
    Initialize weights for the given nn.Module
    :param m: some nn.Module
    :return: None
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a=-1, b=1)
        nn.init.uniform_(m.bias, a=-1, b=1)


def init_local_RFM1d(J_n, x_min, x_max):
    """
    Initialize the random feature functions inside a partition [x_min, x_max]
    :param J_n: number of random feature functions
    :param x_min: left end of partition
    :param x_max: right end of partition
    :return: An RF model on a partition
    """
    model = RFM_rep(in_features=1, J_n=J_n, x_min=x_min, x_max=x_max)

    # Randomly initialize parameter (uniform[-1,1]), in double precision
    model = model.apply(weights_init)
    model = model.double()

    # Freeze the randomly initialized parameters
    for param in model.parameters():
        param.requires_grad = False
    return model


def init_rfm(M_p, J_n, Q):
    """
    Define the RFM model on each partition and their collocation points.

    :param M_p: number of partitions
    :param J_n: number of RF basis functions in a partition
    :param Q: number of collocation points inside a partition
    :return: models: a list of local NNs, one for each partition
             points: a list of M_p tensors, each tensor contains collocation points, with shape (Q+1, 1).
    """
    models = []
    points = []
    for k in range(M_p):
        # Define RFM model in each partition, in mfg_1d_old, partition is just an interval [x_min, x_max]
        x_min = INTERVAL_LENGTH / M_p * k
        x_max = INTERVAL_LENGTH / M_p * (k + 1)
        models.append(init_local_RFM1d(J_n, x_min, x_max))

        # Within each partition, get the collocation points (mfg_1d_old) as a column vector
        points.append(torch.tensor(np.linspace(x_min, x_max, Q + 1), requires_grad=True).reshape([-1, 1]))
    return models, points


def RFM_function_factory(models: List[Callable[[torch.Tensor], torch.Tensor]], w: torch.Tensor) -> Callable[
    [torch.Tensor], torch.Tensor]:
    """
    Factory function to create an RFM function from given models and weights

    Args:
        models (List[Callable[[torch.Tensor], torch.Tensor]]): List of callable models.
        w (torch.Tensor): Tensor of weights with shape (number_of_models, output_dim).

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: A function that computes the weighted sum of model outputs.
    """
    def rfm_function(x):
        return torch.sum(
            torch.stack([
                model(x) * w[i, :].clone().detach().to(torch.float64)
                for i, model in enumerate(models)
            ]),
            dim=(0, 2)
        ).view(-1)
    return rfm_function


def diff_RFM_function(f: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    def df(x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)  # Ensure x requires grad
        y = f(x)
        y.backward(torch.ones_like(y))
        return x.grad.view(-1)
    return df


def second_diff_RFM_function(f: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    df = diff_RFM_function(f)

    def d2f(x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)  # Ensure x requires grad
        y = f(x).squeeze()
        grad_y = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        grad2_y = torch.autograd.grad(grad_y, x, grad_outputs=torch.ones_like(grad_y), create_graph=True)[0]
        return grad2_y.view(-1)

    return df, d2f


def plot_by_iter(value, title, ylabel):
    iterations = np.arange(1, len(value) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, value, label=title, marker='o')
    plt.scatter(iterations[-1], value[-1], color='red')
    plt.xlabel('iterations')
    plt.xticks(iterations)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.text(iterations[-1], value[-1], f'({np.round(value[-1], 10)})', fontsize=10, ha='left', va='bottom')
    plt.show()


def v_cacace(x):
    """
    The given bounded potential function as in Cacace et al's example.
    """
    if isinstance(x, torch.Tensor):
        return sin(2. * pi * x) + cos(4. * pi * x)
    else:
        return np.sin(2. * np.pi * x) + np.cos(4. * np.pi * x)


def b_ren(x):
    if isinstance(x, torch.Tensor):
        return 0.1*sin(2. * pi * x - sin(4. * pi * x)) + exp(cos(2. * pi * x))
    else:
        return 0.1*np.sin(2. * np.pi * x - np.sin(4. * np.pi * x)) + np.exp(np.cos(2. * pi * x))


def hamiltonian_1d(x, p, setting="Cacace"):
    """
    Return the Hamiltonian 1/2 * |Du| ** 2 - V(x) for HJB on (x, p)

    Note in mfg_1d_old, |Du|**2 = (du/dx)**2
    :param x: spatial variable, each element in this variable is a list of collocation points for a partition
    :param p: velocity variable, same format as x, values are derivative Du at each point in x
    :return: value of Hamiltonian on (x, p), same shape as x
    """
    assert len(x) == len(p)

    if setting == "Cacace":
        v = v_cacace
    elif setting == "Ren":
        v = None
    elif setting == "Yang":
        v = None # TODO: implement
    else:
        raise ValueError

    if isinstance(p, torch.Tensor):
        if p.requires_grad:
            p.detach()
        return p ** 2 / 2 - (v(x).view(-1) if v else 0)
    else:
        result = []
        for i in range(len(x)):
            result.append(p[i] ** 2 / 2 - (v(x[i]) if v else 0))
        return result


def lagrangian_1d(x, q, setting="Cacace"):
    assert len(x) == len(q)

    if setting == "Cacace":
        v = v_cacace
    elif setting == "Ren":
        v = None
    elif setting == "Yang":
        v = None  # TODO: implement
    else:
        raise ValueError

    # if isinstance(q, torch.Tensor):
    #     if q.requires_grad:
    #         q.detach()
    #     vx = v(x).view(-1) if v else 0
    #     return q ** 2 / 2 + vx
    # else:
    #     result = []
    #     for i in range(len(x)):
    #         vx = v(x[i]) if v else 0
    #         if isinstance(vx, torch.Tensor) or isinstance(vx, np.ndarray):
    #             bx = vx.view(-1) if v else 0
    #         result.append(q[i] ** 2 / 2 + vx)
    #     return np.array(result)
    if isinstance(q, torch.Tensor):
        if q.requires_grad:
            q.detach()
        return q ** 2 / 2 + (v(x).view(-1) if v else 0)
    else:
        result = []
        for i in range(len(x)):
            result.append(q[i] ** 2 / 2 + (v(x[i]) if v else 0))
        return result


def fd_laplacian(vals, h):
    rolled_left = torch.roll(vals, -1)
    rolled_right = torch.roll(vals, 1)
    diff = rolled_left - 2 * vals + rolled_right
    return diff / (h ** 2)


def fd_derivative(vals, h):
    return (torch.roll(vals, -1) - torch.roll(vals, 1)) / (2 * h)


def plot_RFM_1d(f, label, n_pts=1000, test_values=None, interval_length=INTERVAL_LENGTH):
    pts = torch.tensor(np.linspace(0, interval_length, n_pts), dtype=torch.float64, requires_grad=False).reshape(
        [-1, 1])
    fx = f(pts)
    plt.figure()
    plt.plot(pts, fx, label=label, color='darkblue', linestyle='--')
    if test_values is not None:
        plt.plot(pts, test_values[:-1], color='red')
    plt.legend()
    plt.show()


def get_fd_residuals(prev_q, curr_m, curr_u, curr_q, curr_lam, eps, plot=False):
    n_pts = 1001
    pts = torch.linspace(0, 1, n_pts)[:-1].reshape(-1, 1)
    h = (pts[1] - pts[0]).item()

    # Calculate FD residual for FP
    m_vals = curr_m(pts).view(-1)
    curr_q_vals = curr_q(pts).view(-1)
    prev_q_vals = prev_q(pts).view(-1)
    LapM = fd_laplacian(m_vals, h)
    prev_div_mq = fd_derivative(m_vals * prev_q_vals, h)
    fp_fd_residual = -eps * LapM - prev_div_mq

    # Calculate FD residual for HJB
    u_vals = curr_u(pts)
    eLapU = eps * fd_laplacian(u_vals, h)
    prevQ_Du = prev_q_vals * fd_derivative(u_vals, h)
    prevLq = lagrangian_1d(pts.view(-1), prev_q_vals)
    Fm = m_vals ** 2
    hjb_fd_residual = - eLapU + prevQ_Du - prevLq - Fm + curr_lam

    # Calculate system residual
    hjb_system_fd_residual = -eps * fd_laplacian(u_vals, h) + hamiltonian_1d(pts, fd_derivative(u_vals, h)) + curr_lam - m_vals ** 2
    fp_system_fd_residual = -eps * fd_laplacian(m_vals, h) - fd_derivative(m_vals * curr_q_vals, h)
    system_fd_residual = torch.abs(hjb_system_fd_residual) + torch.abs(fp_system_fd_residual)

    # System Residual with autograd
    d_u, lap_u = second_diff_RFM_function(curr_u)
    d_m, lap_m = second_diff_RFM_function(curr_m)

    d_u_ = d_u(pts).view(-1)
    lap_u_ = lap_u(pts).view(-1)
    d_m_ = d_m(pts).view(-1)
    lap_m_ = lap_m(pts).view(-1)

    hjb_system_residual = - eps * lap_u_ + hamiltonian_1d(pts, d_u_) + curr_lam - m_vals ** 2
    fp_system_residual = -eps * lap_m_ - m_vals * lap_u_ - d_m_ * d_u_
    system_residual = torch.abs(hjb_system_residual) + torch.abs(fp_system_residual)

    # Plot residual
    if plot:
        plot_residuals(fp_fd_residual, fp_system_fd_residual, pts, "Fokker Planck")
        plot_residuals(hjb_fd_residual, hjb_system_fd_residual, pts, "HJB")

    return torch.sum(torch.abs(fp_fd_residual)) / n_pts, torch.sum(torch.abs(hjb_fd_residual)) / n_pts, torch.sum(
        torch.abs(system_fd_residual)) / n_pts, torch.sum(
        torch.abs(system_residual)) / n_pts


def plot_residuals(pde_residual, system_residual, pts, label):
    plt.plot(pts.view(-1).numpy(), pde_residual.numpy(), label="Single PDE FD residual for " + label)
    plt.plot(pts.view(-1).numpy(), system_residual.numpy(), label="System FD residual for " + label, linestyle='--')
    plt.xlabel('pts')
    plt.ylabel('FD error')
    plt.legend()
    plt.show()


def constrained_lsq(A, b, w, c, regularization=1e-3):
    """
    Solve the least squares problem Ax = b under linear constraint w^T x = c with KKT.
    """
    # Compute A^T A and A^T b
    AtA = A.T @ A + regularization * np.eye(A.shape[1])
    Atb = A.T @ b

    # Build the KKT matrix:
    # [A^T A   w]
    # [w^T     0]
    KKT_matrix = np.block([
        [AtA, w.reshape(-1, 1)],
        [w.reshape(1, -1), np.zeros((1, 1))]
    ])

    # Build the right-hand side
    rhs = np.append(Atb.flatten(), c)

    # Solve the KKT system
    sol = solve(KKT_matrix, rhs)
    x = sol[:-1]  # the solution vector
    # lambda_val = sol[-1]  # the Lagrange multiplier
    return x

def constrained_lsq_fp(A, b, w, c, regularization=1e-3):
    """
    Solve the least squares problem Ax = b under linear constraint w^T x = c with KKT. Parameters are np.array.
    """
    # Compute A^T A and A^T b
    AtA = A.T @ A
    Atb = A.T @ b

    # Build the KKT matrix:
    # [A^T A   w]
    # [w^T     0]
    KKT_matrix = np.block([
        [AtA, w.reshape(-1, 1)],
        [w.reshape(1, -1), np.zeros((1, 1))]
    ]) + regularization * np.eye(A.shape[1]+1)

    # Build the right-hand side
    rhs = np.append(Atb.flatten(), c)

    # Solve the KKT system
    sol = solve(KKT_matrix, rhs)
    x = sol[:-1]  # the solution vector
    # lambda_val = sol[-1]  # the Lagrange multiplier
    return x

def constrained_lsq_pinv(A, b, w, c):
    """
    Solve the least squares problem Ax = b under linear constraint w^T x = c, with pseudoinverse.
    """
    # Compute the pseudoinverse of A
    A_pinv = np.linalg.pinv(A, rcond=1e-6)
    # Unconstrained least squares solution
    x0 = A_pinv @ b
    # Compute the projection matrix onto the nullspace of A (if needed)
    P = np.eye(A.shape[1]) - A_pinv @ A

    # Compute the correction factor to satisfy w^T x = c
    denominator = w.T @ P @ w
    if np.abs(denominator) < 1e-10:
        raise ValueError("The constraint vector w is nearly in the range of A^T.")
    correction = (c - w.T @ x0) / denominator

    # Final solution
    x = x0 + P @ w.reshape(-1, 1) * correction
    return x


def solve_FP_forward(models, collocs, q_func, dq_func, M_p, J_n, Q, eps=0.3):
    q = [q_func(collocs[i]).detach() for i in range(M_p)]
    dq = [dq_func(collocs[i]).detach() for i in range(M_p)]

    # Compute lstsq system
    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])

    # Choosing rescaling parameters
    c = 100
    divisor_interiors = np.zeros(M_p * Q)

    # We assume non-negativity constraint in RFM also follows from normalization constraints
    A_constraints = np.zeros([1, M_p * J_n])
    f = np.zeros([M_p * Q, 1])

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m
            out = models[m](collocs[k])
            # values[i,j] = f_{mj}(points[k,i]), where f_{mj} is feature function
            values = out.detach().numpy()  # shape: (Q+1, J_n),

            # Compute second order derivative d^2m/dx^2, also the div term
            grads_2, div = [], []

            for i in range(J_n):
                # Compute gradient of i-th basis function
                g_1 = torch.autograd.grad(outputs=out[:, i], inputs=collocs[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=True, retain_graph=True)[0]

                # Compute second order gradient for i-th basis function
                g_2 = torch.autograd.grad(outputs=g_1[:, 0], inputs=collocs[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          retain_graph=True)[0]
                grads_2.append(g_2.squeeze().detach().numpy())

                # In d=1, div(m*q) = d(m*q)/dx = m' * q + m * q'
                div.append((g_1.squeeze() * q[k] + out[:, i] * dq[k]).detach().numpy())
            grads_2 = np.array(grads_2).T  # grads_2[j,i] = D^2(\phi_{mi}(points[k, j]) * \psi_m(points[k, j]))
            div = np.array(div).T  # div[j,i] = div(\phi_{mi}*\psi_m * q)(points[k, j]), 0<=j<=Q, 0<=i<J_n

            # Impose PDE condition: Lm = -eps * dm^2/dx^2 - div(m * q)
            Lm = - eps * grads_2 - div  # Lm[j,i] = L(\phi_{mi}\psi_m)(points[k, j]) 0<=j<=Q, 0<=i<J_n

            # Calculate rescaling divisor
            max_row = np.max(np.abs(Lm), axis=1)

            indices = np.arange(k * Q, (k + 1) * Q + 1) % len(divisor_interiors)
            divisor_interiors[indices] = np.maximum(divisor_interiors[indices], max_row)

            # Specifying A_pde
            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lm[:Q, :]

            # Normalization constraint:
            for i in range(Q):
                A_constraints[0, m * J_n: (m + 1) * J_n] += values[i, :]

    # Rescaling parameters
    lambda_interiors = c / divisor_interiors.reshape((-1, 1))
    A = A_pde * lambda_interiors
    f = f * lambda_interiors

    # Solve constrained lstsq system
    w = constrained_lsq_pinv(A, f, A_constraints.reshape(-1), M_p * Q) # with constraint: mass = 1
    w_pinv = constrained_lsq_pinv(A, f, A_constraints.reshape(-1), M_p * Q)  # with constraint: mass = 1
    w = torch.tensor(w.reshape((M_p, J_n)))

    solution = RFM_function_factory(models, w)

    return solution


def solve_HJB_forward(models, collocs, m_func, q_func, M_p, J_n, Q, L_func, b=None, eps=0.3):
    q = [q_func(collocs[i]).detach() for i in range(M_p)]

    # Compute lstsq system
    # place-holder variables for A
    A_pde = np.zeros([M_p * Q, M_p * J_n])

    # We assume non-negativity constraint in RFM also follows from normalization constraint
    A_constraints = np.zeros([1, M_p * J_n])
    f = np.zeros([M_p * Q, 1])

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m (HJB) and M_m (FP)
            out = models[m](collocs[k])
            values_hjb = out.detach().numpy()

            # Compute first and second order derivative du/dx and d^2u/dx^2 for HJB
            grads_2 = []
            q_du = []  # Compute the (q * Du) term

            for i in range(J_n):
                # Compute gradient of i-th basis function
                g_1 = torch.autograd.grad(outputs=out[:, i], inputs=collocs[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=True, retain_graph=True)[0]

                # Compute second order gradient for i-th basis function
                g_2 = torch.autograd.grad(outputs=g_1[:, 0], inputs=collocs[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          retain_graph=True)[0]
                grads_2.append(g_2.squeeze().detach().numpy())

                q_du.append((g_1.squeeze() * q[k]).detach().numpy())

            grads_2 = np.array(grads_2).T  # grads[j,i] = f''_{mi}(points[k, j])
            q_du = np.array(q_du).T  # q_du[j, i] = f'_{mi}(points[k, j]) * q(points[k, j])

            # Impose PDE condition: Lu = -eps * du^2/dx^2 - div(u * q)
            # Lu[j, i] =  - eps * f'_{mi}(points[k, j]) + f'_{mi}(points[k, j]) * q(points[k, j])
            Lu = - eps * grads_2 + q_du  # shape=(Q+1, J_n)

            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lu[:Q, :]

            # Normalization constraint:
            for i in range(Q):
                A_constraints[0, m * J_n: (m + 1) * J_n] += values_hjb[i, :]

        # The f-side of discretized Lu=f system
        if b:
            Lq = L_func(collocs[k], q[k], b)
        else:
            Lq = L_func(collocs[k], q[k])
        Fm = m_func(collocs[k]).view(-1) ** 2  # The coupling term is F(m) = m^2
        summed = Fm + Lq
        trimmed = summed[:Q].detach().numpy().reshape(-1, 1)
        f[k * Q:(k + 1) * Q, :] = trimmed

    A = A_pde #* lambda_interiors
    f = f #* lambda_interiors

    # Solve lstsq system
    dim = A.shape[0]
    proj = np.eye(dim) - np.ones((dim, dim)) / dim

    w = constrained_lsq(proj @ A, proj @ f, A_constraints.reshape(-1), 0)
    lam = np.mean(f - A @ w)
    w = torch.tensor(w.reshape((M_p, J_n)))

    solution = RFM_function_factory(models, w)
    return solution, lam


def solve_HJB_classical(N, epsilon, q_func, b_func, m_func, L_func):
    """
    Solve -εΔu + q·∇u + λ = b + F(m) + L(q) on [0,1] with periodic BC and ∫u=0.
    Returns grid x, solution u, and ergodic constant λ.
    :param N:
    :param epsilon:
    :param q_func:
    :param b_func:
    :param m_func:
    :param L_func:
    :return:
    """
    x = np.linspace(0, 1, N, endpoint=False)
    x_tensor = torch.tensor(x).reshape(-1, 1)
    h = x[1] - x[0]


    q = q_func(x_tensor).view(-1)
    m = m_func(x_tensor).view(-1)

    # Compute RHS
    Fm = m.pow(2).numpy()
    Lq_tensor = L_func(x_tensor, q, b_func)
    if isinstance(Lq_tensor, list):
        Lq_tensor = torch.cat(Lq_tensor, dim=0)
    Lq = Lq_tensor.view(-1).detach().numpy()
    rhs = Fm + Lq

    # Discrete HJB operator
    I = np.eye(N)
    lap_op = (np.roll(I, -1, axis=1) + np.roll(I, 1, axis=1) - 2 * I) / h**2
    grad = (np.roll(I, 1, axis=1) - np.roll(I, -1, axis=1)) / (2 * h)
    pde_op = - epsilon * lap_op + np.diag(q) * grad

    # Need to solve: A u + 1 λ = rhs, s.t. 1^T u = 0, where A is the discrete PDE operator
    # Equivalent to solving:
    # [A   1] [u] = [b]
    # [1^T 0] [λ]   [0]
    ones = np.ones((N, 1))
    pde_augmented = np.block([
        [pde_op, ones],
        [ones.T, np.zeros((1, 1))]
    ])

    u, lam = np.split(lstsq(pde_augmented, np.append(rhs, 0))[0], [N])
    lam = lam.item()

    return u, lam, x


def fd_operators(h, n_points):
    # Building the discrete Laplacian for periodic domain
    ones = np.ones(n_points)
    L = np.diag(-2 * ones) + np.diag(ones[:-1], 1) + np.diag(ones[:-1], -1)
    L[0, -1] = L[-1, 0] = 1
    L = L / h ** 2

    # Building the discrete derivative operator for periodic domain
    D_L = np.diag(ones) + np.diag(-ones[:-1], -1)
    D_R = np.diag(ones[:-1], 1) - np.diag(ones)
    D_L[0, -1] = -1
    D_R[-1, 0] = 1
    D_L = D_L / h
    D_R = D_R / h
    return L, D_L, D_R


def policy_iteration_fd(n_points=1000, n_iters=20, eps=0.3, R=0.5, plot=False, direct_solve=True, A_Q_iterate=False, use_kernel=False):
    def plot_fd_system(x, m_vals, u_vals):
        x_periodic = np.append(x, 1.0)
        m_periodic = np.append(m_vals, m_vals[0])
        u_periodic = np.append(u_vals, u_vals[0])

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x_periodic, m_periodic, label='m(x)')
        plt.xlabel('x')
        plt.ylabel('m(x)')
        plt.title('Fokker-Planck Solution')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x_periodic, u_periodic, label='u(x)', color='orange')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('HJB Solution')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_fd_residual(m_residuals, u_residuals, system_residuals):
        iterations = np.arange(1, len(m_residuals) + 1)
        plt.figure(figsize=(12, 12))

        plt.subplot(3, 1, 1)
        plt.plot(iterations, m_residuals, label='m residuals', marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('FD Residuals for m over Iterations')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(iterations, u_residuals, label='u residuals', marker='s', color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('FD Residuals for u over Iterations')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(iterations, system_residuals, label='system residuals', marker='^', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('FD Residuals for system over Iterations')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    historical_m, historical_u, historical_lam = [None], [None], [None]
    historical_q_L = [np.zeros(n_points)]
    historical_q_R = [np.zeros(n_points)]
    m_residuals, u_residuals, system_residuals = [], [], []

    x = np.linspace(0, 1, n_points, endpoint=False)
    h = x[1] - x[0]

    # Building the discrete Laplacian for periodic domain
    ones = np.ones(n_points)
    L, D_L, D_R = fd_operators(h, n_points)

    # Policy Iteration Algorithm
    for _ in range(n_iters):
        Q_plus = np.maximum(historical_q_L[-1], 0)
        Q_minus = np.minimum(historical_q_R[-1], 0)
        # Step 1: Solve FP
        A_Q_T = - eps * L + np.diag(Q_plus) @ D_L + np.diag(Q_minus) @ D_R
        A_Q = A_Q_T.T

        # Check the A_Q_T (and hence A_Q) is correctly defined
        for i in range(n_points):
            diff_mid = A_Q_T[i, i] - 2 * eps / h**2 - Q_plus[i] / h + Q_minus[i] / h
            if abs(diff_mid) > 1e-8:
                print("middle entry differs", diff_mid)
            diff_left = A_Q_T[i, (n_points + i-1) % n_points] + eps / h**2 + Q_plus[i] / h
            if abs(diff_left) > 1e-8:
                print("middle entry differs", diff_left)
            diff_right = A_Q_T[i, (n_points + i+1) % n_points] + eps / h**2 - Q_minus[i] / h
            if abs(diff_right) > 1e-8:
                print("middle entry differs", diff_right)
        w_fp = np.ones(n_points) * h
        m_vals_no_iter = constrained_lsq_fp(A_Q, np.zeros(n_points), w_fp, 1)
        print("total mass of M without iteration", np.average(np.abs(m_vals_no_iter)))

        # Finding m_vals iteratively until convergence
        mu = 1
        W = np.ones_like(x)  # initial choice of W
        M = pinv(mu * np.eye(n_points) + A_Q)
        W_next = mu * M @ W
        while np.linalg.norm(W - W_next) > 1e-8:
            W = W_next
            W_next = mu * M @ W
        m_vals = W_next
        print("total mass of M with iterative computation", np.average(np.abs(m_vals)))

        historical_m.append(m_vals)

        # (Computing residual for FP)
        residual_m = -eps * L @ m_vals - D_L @ (m_vals * historical_q_L[-1]) - D_R @ (m_vals * historical_q_R[-1])
        m_residuals.append(np.sum(np.abs(residual_m)) * h)
        print("residual of m", m_residuals[-1])

        # Step 2: Solve HJB
        rhs_hjb = (Q_plus ** 2 + Q_minus ** 2) / 2 + v_cacace(x) + m_vals **2
        A_Q_T_pinv = pinv(A_Q_T)

        # Compute the spanning vector of ker(A(Q))
        w = scipy.linalg.null_space(A_Q_T.T).flatten()
        lam_ker = np.dot(w, rhs_hjb) / np.dot(w, ones)

        P = A_Q_T @ A_Q_T_pinv
        diff = (rhs_hjb - P @ rhs_hjb)
        lam = diff[0] # FIXME: Using first entry could be problematic because it may not a multiple of \mathds{1}.
        lam_avg = np.average(diff)
        # print("A_Q_T 1 = ", np.abs(A_Q_T @ np.ones(n_points)))
        lam = lam_ker
        u_vals = A_Q_T_pinv @ (rhs_hjb - lam)
        print("total mass of U", np.average(u_vals))

        historical_u.append(u_vals)
        historical_lam.append(lam)

        # (Computing residual for HJB)
        V = v_cacace(x)
        residual_u = (-eps * L @ u_vals + Q_plus * (D_L @ u_vals)  + Q_minus * (D_R @ u_vals) + lam * np.ones(n_points)
                      - (Q_plus ** 2 + Q_minus ** 2) / 2 - V - m_vals**2)
        u_residuals.append(np.sum(np.abs(residual_u)))
        print("residual for U is", u_residuals[-1])

        # Step 3: Update the policy
        q_L_new = D_L @ u_vals
        q_R_new = D_R @ u_vals
        DU_norm = np.sum(np.abs(q_L_new)) + np.sum(np.abs(q_R_new))
        if DU_norm > R:
            q_L_new = q_L_new * R / DU_norm
            q_R_new = q_R_new * R / DU_norm
        historical_q_L.append(q_L_new)
        historical_q_R.append(q_R_new)

        # Testing system residual
        # H_Du = np.sum((D_L @ u_vals) ** 2) + np.sum((D_R @ u_vals) ** 2)
        # residual_hjb_sys = -eps * L @ u_vals + H_Du + ones * lam - V - m_vals**2 # F(x) = x**2
        # residual_fp_sys = -eps * L @ m_vals - D @ (m_vals * historical_q[-1])
        # residual_sys = np.sum(np.abs(residual_hjb_sys)) + np.sum(np.abs(residual_fp_sys))
        # system_residuals.append(residual_sys)
        system_residuals.append(0)

        # Plot intermediate solutions
        if plot:
            plot_fd_system(x, m_vals, u_vals)

    # Plotting residuals
    if plot:
        plot_fd_residual(m_residuals, u_residuals, system_residuals)

    return historical_m[-1], historical_u[-1], historical_lam[-1]


def policy_iteration(Mu, Ju, Mm, Jm, Qu, Qm, b=None, n_iters=20, eps=0.3, tau=1e-6, plot=False):
    # fix datatype
    torch.set_default_dtype(torch.float64)
    n_test = 400

    # Initialize RFMs for FP and HJB with zero weights
    models_fp, collocs_fp = init_rfm(Mm, Jm, Qm)
    models_hjb, collocs_hjb = init_rfm(Mu, Ju, Qu)
    w_hjb = torch.zeros((Mu, Ju))
    w_fp = torch.zeros((Mm, Jm))

    # historical solutions and policies
    historical_m = [RFM_function_factory(models_fp, w_fp)]
    historical_u = [RFM_function_factory(models_hjb, w_hjb)]
    historical_lam = [0]
    historical_q = [None]
    historical_q[0], dq = second_diff_RFM_function(historical_u[0])

    # Calculate FD residuals
    fd_residual_m = np.zeros(n_iters + 1)
    fd_residual_u = np.zeros(n_iters + 1)
    fd_system_residual = np.zeros(n_iters + 1)
    system_residual = np.zeros(n_iters + 1)

    # main loop
    for k in range(1, n_iters + 1):
        # Step (1): Solve FP equation
        print("solving FP", k)
        historical_m.append(solve_FP_forward(models_fp, collocs_fp, historical_q[k - 1], dq, Mm, Jm, Qm))
        if plot:
            plot_RFM_1d(historical_m[k], "m^(" + str(k) + ")")

        # Step (2): Solve HJB and the ergodic cost
        print("solving HJB", k)
        curr_u, curr_lam = solve_HJB_forward(models_hjb, collocs_hjb, historical_m[k], historical_q[k - 1], Mu,
                                             Ju, Qu, lagrangian_1d, b)
        historical_u.append(curr_u)
        historical_lam.append(curr_lam)
        test_u, test_lam, points = solve_HJB_classical(n_test, eps, historical_q[k - 1], b, historical_m[k], lagrangian_1d)
        test_u = np.concatenate((test_u, np.array([test_u[0]])))
        print("RFM_lam =", curr_lam, "FD_lam =", test_lam)
        if plot:
            plot_RFM_1d(curr_u, "u^(" + str(k) + ")", n_pts=n_test, test_values=test_u)

        # Testing refining mesh for FD solution:
        # if plot:
        #     for n_pts_ in [800]:
        #         test_u_, test_lam_, points_, FD_residue_ = solve_HJB_classical(n_pts_, eps, historical_q[k - 1], b,
        #                                                                    historical_m[k], lagrangian_1d)
        #         test_u_ = np.concatenate((test_u_, np.array([test_u_[0]])))
        #         plot_RFM_1d(curr_u, "u^(" + str(k) + ") vs FD solution with " + str(n_pts_) + " points", n_pts=n_pts_, test_values=test_u_)

        # Step (3): Update the policy
        new_q, dq = second_diff_RFM_function(historical_u[k])
        historical_q.append(new_q)

        # Compute residuals
        fd_residual_m[k], fd_residual_u[k], fd_system_residual[k], system_residual[k] = get_fd_residuals(historical_q[k - 1],
                                                                                     historical_m[k], historical_u[k],
                                                                                     historical_q[k], curr_lam, eps)

    if plot:
        plot_by_iter(fd_residual_m, 'FD Residual error of Fokker-Planck', 'Residual error')
        plot_by_iter(fd_residual_u, 'FD Residual error of HJB', 'Residual error')
        plot_by_iter(fd_system_residual, 'FD Residual error of System', 'Residual error')
        plot_by_iter(system_residual, 'Residual error of System', 'Residual error')

    print("Final residual of Fokker-Planck is", fd_residual_m[-1])
    print("Final residual of HJB is", fd_residual_u[-1])

    return historical_m[-1], historical_u[-1], historical_lam[-1], fd_system_residual


def test():
    # set_seed(100)
    Mu, Mm, Mb = 8, 4, 5
    Ju, Jm, Jb = 40, 20, 30
    Il = 400
    Qu, Qm, Qb = 80, 40, 60
    eps = 0.3

    def test_hjb_mms(n_points=1000):
        def plot_u_vs_true(x, u_vals, true_u_vals, title="Comparison of u and true u"):
            h = x[1] - x[0]
            plt.figure(figsize=(10, 5))
            plt.plot(x, u_vals, label='u_vals (Numerical)', linestyle='--', color='blue')
            plt.plot(x, true_u_vals, label='true_u_vals (Exact)', linestyle='-', color='red')
            # Compute L1, L2 residual between numerical and true solutions
            l1_res = np.sum(np.abs(u_vals - true_u_vals)) * h
            l2_res = np.sqrt(np.sum((u_vals - true_u_vals) ** 2) * h)
            # Annotate L1 and L2 on the graph
            textstr = f"L1 error = {l1_res:.4e}\nL2 error = {l2_res:.4e}"
            plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            plt.xlabel('x')
            plt.ylabel('Function value')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        def generate_test_u():
            # u1(x) = sin(2πx)
            u1 = lambda x: np.sin(2 * np.pi * x)
            du1 = lambda x: 2 * np.pi * np.cos(2 * np.pi * x)
            lap_u1 = lambda x: - (2 * np.pi) ** 2 * np.sin(2 * np.pi * x)

            # u2(x) = sin(2πx) + 0.5 * cos(4πx)
            u2 = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
            du2 = lambda x: 2 * np.pi * np.cos(2 * np.pi * x) - 2 * np.pi * np.sin(4 * np.pi * x)
            lap_u2 = lambda x: - (2 * np.pi) ** 2 * np.sin(2 * np.pi * x) - (4 * np.pi) ** 2 * 0.5 * np.cos(
                4 * np.pi * x)

            # u3(x) = sin(2πx) * cos(4πx)
            u3 = lambda x: np.sin(2 * np.pi * x) * np.cos(4 * np.pi * x)
            du3 = lambda x: (2 * np.pi * np.cos(2 * np.pi * x) * np.cos(4 * np.pi * x)
                             - 4 * np.pi * np.sin(2 * np.pi * x) * np.sin(4 * np.pi * x))
            lap_u3 = lambda x: (
                    - (2 * np.pi) ** 2 * np.sin(2 * np.pi * x) * np.cos(4 * np.pi * x)
                    - (4 * np.pi) ** 2 * np.sin(2 * np.pi * x) * np.cos(4 * np.pi * x)
                    - 2 * 2 * np.pi * 4 * np.pi * np.cos(2 * np.pi * x) * np.sin(4 * np.pi * x)
            )
            u = [u1, u2, u3]
            du = [du1, du2, du3]
            lap_u = [lap_u1, lap_u2, lap_u3]
            return u, du, lap_u

        u, du, lap_u = generate_test_u()
        lam_true = 1
        prev_u = lambda x: np.sin(2 * np.pi * x) / (2 * np.pi)
        q = lambda x: np.cos(2 * np.pi * x)
        fm = [(lambda x, i=i:
               - eps * lap_u[i](x) + q(x) * du[i](x) + lam_true - v_cacace(x) - q(x) ** 2 / 2)
            for i in range(len(u))
        ]
        x = np.linspace(0, 1, n_points, endpoint=False)
        h = x[1] - x[0]

        # Building the discrete Laplacian for periodic domain
        ones = np.ones(n_points)
        L, D_L, D_R = fd_operators(h, n_points)

        for i in range(len(u)):
            fm_vals = (fm[i])(x)
            prev_u_vals = prev_u(x)
            q_L = D_L @ prev_u_vals
            q_R = D_R @ prev_u_vals
            Q_plus = np.maximum(q_L, 0)
            Q_minus = np.minimum(q_R, 0)
            A_Q_T = - eps * L + np.diag(Q_plus) @ D_L + np.diag(Q_minus) @ D_R

            rhs_hjb = (Q_plus ** 2 + Q_minus ** 2) / 2 + v_cacace(x) + fm_vals
            A_Q_T_pinv = pinv(A_Q_T)

            # Compute the spanning vector of ker(A(Q))
            w = scipy.linalg.null_space(A_Q_T.T).flatten()
            lam_ker = np.dot(w, rhs_hjb) / np.dot(w, ones)

            P = A_Q_T @ A_Q_T_pinv
            diff = (rhs_hjb - P @ rhs_hjb)
            lam = diff[0]  # FIXME: Using first entry could be problematic because it may not a multiple of \mathds{1}.
            lam_avg = np.average(diff)
            # print("A_Q_T 1 = ", np.abs(A_Q_T @ np.ones(n_points)))
            print("recovered lambda (should be 1) are: ker =", lam_ker, "diff[0] =", lam, "lam_avg =", lam_avg)
            lam = lam_ker
            u_vals = A_Q_T_pinv @ (rhs_hjb - lam)

            true_u_vals = u[i](x)
            plot_u_vs_true(x, u_vals, true_u_vals)


    def test_fp_mms(n_points=1000):
        def plot_m_vs_true(x, m_vals, true_m_vals, title="Comparison of m and true m"):
            h = x[1] - x[0]
            plt.figure(figsize=(10, 5))
            plt.plot(x, m_vals, label='m_vals (Numerical)', linestyle='--', color='blue')
            plt.plot(x, true_m_vals, label='true_m_vals (Exact)', linestyle='-', color='red')
            # Compute L1, L2 residual between numerical and true solutions
            l1_res = np.sum(np.abs(m_vals - true_m_vals)) * h
            l2_res = np.sqrt(np.sum((m_vals - true_m_vals)**2) * h)
            # Annotate L1 and L2 on the graph
            textstr = f"L1 error = {l1_res:.4e}\nL2 error = {l2_res:.4e}"
            plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            plt.xlabel('x')
            plt.ylabel('Function value')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        def generate_test_m():
            # Symbolic function definitions
            x_sym = sym.symbols('x')
            m_syms = [
                1 + 0.5 * sym.cos(2 * sym.pi * x_sym),
                1 + 0.4 * sym.cos(2 * sym.pi * x_sym) + 0.1 * sym.cos(4 * sym.pi * x_sym),
                1 + 0.5 * sym.sin(2 * sym.pi * x_sym) ** 2 - 0.25
            ]
            dm_syms = [sym.simplify(sym.diff(m, x_sym)) for m in m_syms]
            lap_m_syms = [sym.simplify(sym.diff(dm, x_sym)) for dm in dm_syms]
            # Use explicit 0.3 for epsilon
            q_syms = [sym.simplify((-0.3 * dm_syms[i]) / m_syms[i]) for i in range(len(m_syms))]

            # Compute average of log(m_i) over [0,1] for each symbolic m_i (explicit closed-form)
            mean_log = [
                sym.log((2 + sym.sqrt(3)) / 4),
                sym.log((9 + sym.sqrt(105) + 2 * sym.sqrt(sym.sqrt(105) - 7) + sym.sqrt(14 * (sym.sqrt(105) + 7))) / 40),
                sym.log((4 + sym.sqrt(15)) / 8)
            ]

            # Build u_syms symbolically using mean_log
            u_syms = [
                -0.3 * (sym.log(m_syms[i]) - mean_log[i])
                for i in range(len(m_syms))
            ]

            # Convert symbolic expressions to numpy lambda functions
            m = [sym.lambdify(x_sym, expr, modules='numpy') for expr in m_syms]
            dm = [sym.lambdify(x_sym, expr, modules='numpy') for expr in dm_syms]
            lap_m = [sym.lambdify(x_sym, expr, modules='numpy') for expr in lap_m_syms]
            q = [sym.lambdify(x_sym, expr, modules='numpy') for expr in q_syms]
            u = [sym.lambdify(x_sym, expr, modules='numpy') for expr in u_syms]

            # Numerical checks for u: after u is constructed as list of functions
            xs = np.linspace(0, 1, 100000, endpoint=False)
            for i, uf in enumerate(u):
                # Check mean of u is zero
                integral_val = np.trapezoid(uf(xs), xs)
                if np.abs(integral_val) > 1e-5:
                    raise ValueError(f"Failure at i={i}: u[{i}] integrates to {integral_val}, which exceeds tolerance")
                # Check derivative matches q
                u_diff = np.gradient(uf(xs), xs[1] - xs[0])
                q_vals = q[i](xs)
                max_diff = np.max(np.abs(u_diff - q_vals))
                if max_diff > 1e-4:
                    raise ValueError(f"Failure at i={i}: u[{i}]' differs from q[{i}] by max difference {max_diff}")
            return m, dm, lap_m, q, u

        m, dm, lap_m, q, u = generate_test_m()

        x = np.linspace(0, 1, n_points, endpoint=False)
        h = x[1] - x[0]

        # Building the discrete Laplacian for periodic domain
        ones = np.ones(n_points)
        L, D_L, D_R = fd_operators(h, n_points)

        for i in range(len(m)):
            u_vals = u[i](x)
            q_L = D_L @ u_vals
            q_R = D_R @ u_vals
            Q_plus = np.maximum(q_L, 0)
            Q_minus = np.minimum(q_R, 0)

            # Solve FP
            A_Q_T = - eps * L + np.diag(Q_plus) @ D_L + np.diag(Q_minus) @ D_R
            A_Q = A_Q_T.T

            w_fp = np.ones(n_points) * h
            m_vals_no_iter = constrained_lsq_fp(A_Q, np.zeros(n_points), w_fp, 1)
            print("total mass of M without iteration", np.average(np.abs(m_vals_no_iter)))

            # Finding m_vals iteratively until convergence
            mu = 1
            W = np.ones_like(x)  # initial choice of W
            M = pinv(mu * np.eye(n_points) + A_Q)
            W_next = mu * M @ W
            while np.linalg.norm(W - W_next) > 1e-8:
                W = W_next
                W_next = mu * M @ W
            m_vals = W_next
            print("total mass of M with iterative computation", np.average(np.abs(m_vals)))

            true_m_vals = m[i](x)
            plot_m_vs_true(x, m_vals, true_m_vals)


    # test_hjb_mms()
    # test_fp_mms()

    m_fd, u_fd, lam_fd = policy_iteration_fd(1000, n_iters=20, R=2000, plot=True)

    def true_b(x):
        return 0.1 * (sin(2 * pi * x - sin(4 * pi * x)) + exp(cos(2 * pi * x)))


    # m_true, u_true, lam_true, _ = policy_iteration(Mu, Ju, Mm, Jm, Qu, Qm, b=true_b, n_iters=20, plot=True)

    # print("lam_true:", lam_true)
    # recovered_b, recovered_lam = inverse_PI_stationary(u_tcrue, Mu, Mm, Mb, Ju, Jm, Jb, Il, Qu, Qm, Qb)

    # Plot the recovered b against true b
    # plot_RFM_vs_true(recovered_b, true_b, recovered_lam, "b")


if __name__ == '__main__':
    test()