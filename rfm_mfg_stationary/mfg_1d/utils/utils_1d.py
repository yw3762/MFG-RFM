import torch
from torch import sin, cos, pi
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

from rfm_mfg_stationary.mfg_1d.utils.config import INTERVAL_LENGTH
from rfm_mfg_stationary.mfg_1d.models.RFM_rep import RFM_rep


def set_seed(seed):
    """
    Set random seed for reproducibility.
    :param seed: seed to set
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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


def init_rfm(M_p, J_n, Q, debug=False):
    """
    Define the RFM model on each partition and their collocation points.

    :param M_p: number of partitions
    :param J_n: number of RF basis functions in a partition
    :param Q: number of collocation points inside a partition
    :param debug: debug flag
    :return: models: a list of local NNs, one for each partition
             points: a list of M_p tensors, each tensor contains collocation points, with shape (Q+1, 1).
    """
    models = []
    points = []
    for k in range(M_p):
        # Define RFM model in each partition, in mfg_1d_old, partition is just an interval [x_min, x_max]
        x_min = INTERVAL_LENGTH / M_p * k
        x_max = INTERVAL_LENGTH / M_p * (k + 1)
        models.append(init_local_RFM1d(J_n, x_min, x_max, debug))

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

def hamiltonian_1d(x, p):
    """
    Return the Hamiltonian 1/2 * |Du| ** 2 - V(x) for HJB on (x, p)

    Note in mfg_1d_old, |Du|**2 = (du/dx)**2
    :param x: spatial variable, each element in this variable is a list of collocation points for a partition
    :param p: velocity variable, same format as x, values are derivative Du at each point in x
    :param v: bounded potential function
    :return: value of Hamiltonian on (x, p), same shape as x
    """
    assert len(x) == len(p)
    def v(x):
        return sin(2. * pi * x) + cos(4. * pi * x)

    if isinstance(p, torch.Tensor):
        if p.requires_grad:
            p.detach()

        return p ** 2 / 2 - v(x).view(-1)
    else:
        result = []
        for i in range(len(x)):
            result.append(p[i] ** 2 / 2 - v(x[i]))
        return result


def lagrangian_1d(x, q):
    assert len(x) == len(q)
    def v(x):
        return sin(2. * pi * x) + cos(4. * pi * x)

    if isinstance(q, torch.Tensor):
        if q.requires_grad:
            q.detach()
        return q ** 2 / 2 + v(x).view(-1)
    else:
        result = []
        for i in range(len(x)):
            result.append(q[i] ** 2 / 2 + v(x[i]))
        return result


def fd_laplacian(vals, h):
    rolled_left = torch.roll(vals, -1)
    rolled_right = torch.roll(vals, 1)
    diff = rolled_left - 2 * vals + rolled_right
    return diff / (h ** 2)
    # return (-torch.roll(vals, -2) + 16 * torch.roll(vals, -1) - 30 * vals + 16 * torch.roll(vals, 1)
    #         -torch.roll(vals, 2)) / (12 * h**2)


def fd_derivative(vals, h):
    return (torch.roll(vals, -1) - torch.roll(vals, 1)) / (2 * h)


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


def test_differentials(f_func, df_func, df2_func):
    n_pts = 1000
    pts = torch.linspace(0, 1, n_pts + 1).reshape([-1, 1])
    h = (pts[1] - pts[0])[0]

    f = f_func(pts).view(-1)
    df_fd = fd_derivative(f, h)
    df2_fd = fd_laplacian(f, h)

    df = df_func(pts).view(-1)
    df2 = df2_func(pts).view(-1)

    plt.plot(pts, df_fd, label="FD Q")
    plt.plot(pts, df, label="Q", linestyle='--')
    plt.legend()
    plt.title('Comparison of Q and FD Q')
    plt.show()

    plt.plot(pts[1:-1], df_fd[1:-1], label="FD Q")
    plt.plot(pts, df, label="Q", linestyle='--')
    plt.legend()
    plt.title('Comparison of Q and FD Q (No endpoints)')
    plt.show()

    plt.plot(pts, df2_fd, label="FD dQ")
    plt.plot(pts.detach().numpy(), df2.detach().numpy(), label="dQ", linestyle='--')
    plt.legend()
    plt.title('Comparison of dQ and FD dQ')
    plt.show()

    plt.plot(pts[1:-1], df2_fd[1:-1], label="FD dQ")
    plt.plot(pts.detach().numpy(), df2.detach().numpy(), label="dQ", linestyle='--')
    plt.legend()
    plt.title('Comparison of dQ and FD dQ (No endpoints)')
    plt.show()


def plot_RFM_1d(f, label, n_pts=1000, interval_length=INTERVAL_LENGTH):
    pts = torch.tensor(np.linspace(0, interval_length, n_pts), dtype=torch.float64, requires_grad=False).reshape(
        [-1, 1])
    fx = f(pts)
    plt.figure()
    plt.plot(pts, fx, label=label, color='darkblue', linestyle='--')
    plt.legend()
    plt.show()


def get_fd_residuals(prev_q, curr_m, curr_u, curr_q, curr_lam, eps):
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

    # Plot residuals
    plot_residuals(fp_fd_residual, fp_system_fd_residual, pts, "Fokker Planck")
    plot_residuals(hjb_fd_residual, hjb_system_fd_residual, pts, "HJB")

    return torch.sum(torch.abs(fp_fd_residual)) / n_pts, torch.sum(torch.abs(hjb_fd_residual)) / n_pts, torch.sum(
        torch.abs(system_fd_residual)) / n_pts


def plot_residuals(pde_residual, system_residual, pts, label):
    plt.plot(pts.view(-1).numpy(), pde_residual.numpy(), label="Single PDE FD residual for " + label)
    plt.plot(pts.view(-1).numpy(), system_residual.numpy(), label="System FD residual for " + label, linestyle='--')
    plt.xlabel('pts')
    plt.ylabel('FD error')
    plt.legend()
    plt.show()