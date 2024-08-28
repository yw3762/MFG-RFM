import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from typing import List, Callable

from utils.types import ErrorArray
from utils.config import INTERVAL_LENGTH


def set_seed(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True


def weights_init(m):
    """
    Randomly initialize parameters in the Conv2d or Linear layer.
    :param m:  the given layer
    :return:   None
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a=-1, b=1)
        nn.init.uniform_(m.bias, a=-1, b=1)


def weights_init_debug(m):
    """
    Randomly initialize parameters in the Conv2d or Linear layer.
    :param m:  the given layer
    :return:   None
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


def init_local_RFM1d(J_n, x_min, x_max, debug=False):
    """
    Initialize RFM network on a mfg_1d domain, i.e. the interval [x_min, x_max].
    :param J_n:
    :param x_min:
    :param x_max:
    :return:
    """
    model = RFM_rep(in_features=1, J_n=J_n, x_min=x_min, x_max=x_max)

    # Randomly initialize parameter (uniform[-1,1]), in double precision
    if debug:
        model = model.apply(weights_init_debug)
    else:
        model = model.apply(weights_init)
    model = model.double()

    # Freeze the randomly initialized parameters
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_differential_1d(f, points, method="center"):
    """
    Numerically compute us of f(x)

    :param f: function values evaluated at each point, same shape as points. f must be periodic in the domain
    :param points: Partitions of domain, each partition contains a list of collocation points uniformly selected.
                   Each partition have boundary coinciding at a point, and domain is periodic.
    :param method: Method for getting the numerical differential.
    :return:
    """

    k = len(points)  # number of partitions
    n = len(points[0])  # number of collocation points

    df = np.zeros((k, n))

    # Check input points have coinciding boundary
    for i in range(len(points) - 1):
        assert points[i][-1] == points[i + 1][1]

    if method == "center":
        # f'(x) = (f(x+h) - f(x-h))/2h
        divisor = 2 * points[1][0] - 2 * points[0][0]  # 2h

        for i in range(k):
            for j in range(1, n - 1):
                df[i][j] = (f[i][j + 1] - f[i][j - 1]) / divisor

        # Periodicity condition
        for i in range(k):
            df[i][-1] = (f[(i + 1) % k][1] - f[i][-2]) / divisor
            df[(i + 1) % k][0] = df[i][-1]
    else:
        raise NotImplementedError("Only center method is implemented")
    return df


class RFM_rep(nn.Module):
    def __init__(self, in_features, J_n, x_max, x_min):
        super(RFM_rep, self).__init__()
        self.in_features = in_features  # num input features
        self.hidden_features = J_n  # width of hidden layer
        self.J_n = J_n  # J_n is the number of local RF functions
        self.x_min = x_min  # x_{nj} - r_{nj}
        self.x_max = x_max  # x_{nj} + r_{nj}
        self.a = 2.0 / (x_max - x_min)  # this is the 1/r_{nj}
        self.x_0 = (x_max + x_min) / 2  # center of partition.

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
        d = (x - self.x_min) / (self.x_max - self.x_min)
        # indicator of location of x
        d0 = d <= -1 / 4
        d1 = (d <= 1 / 4) & (d > -1 / 4)
        d2 = (d <= 3 / 4) & (d > 1 / 4)
        d3 = (d <= 5 / 4) & (d > 3 / 4)
        d4 = d > 5 / 4

        # y is the change of variable in normalized coordinate \tilde{x}
        y = self.a * (x - self.x_0)

        # pass the normalized variable into hidden-layer
        # print('Before passing to hidden layer, y is', y)
        y = self.hidden_layer(y)
        # print('After passing to hidden layer, y is', y)

        # y_i are the PoU w.r.t each location of x
        y0 = 0
        y1 = y * (1 + torch.sin(2 * np.pi * d)) / 2
        y2 = y
        y3 = y * (1 - torch.sin(2 * np.pi * (d - 1))) / 2
        y4 = 0

        # check boundary cases, on boundaries, there is no need for sin() smoothing
        if self.x_min == 0:
            return d0 * y0 + (d1 + d2) * y2 + d3 * y3 + d4 * y4
        elif self.x_max == INTERVAL_LENGTH:
            return d0 * y0 + d1 * y1 + (d2 + d3) * y2 + d4 * y4
        else:
            return d0 * y0 + d1 * y1 + d2 * y2 + d3 * y3 + d4 * y4


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
        # Define RFM model in each partition, in mfg_1d, partition is just an interval [x_min, x_max]
        x_min = INTERVAL_LENGTH / M_p * k
        x_max = INTERVAL_LENGTH / M_p * (k + 1)
        models.append(init_local_RFM1d(J_n, x_min, x_max, debug))

        # Within each partition, get the collocation points (mfg_1d) as a column vector
        points.append(torch.tensor(np.linspace(x_min, x_max, Q + 1), requires_grad=True).reshape([-1, 1]))
    return models, points


def V(x):
    """
    Analytical bounded potential function

    The associated Hamiltonian is H(x, p) = 1/2 |p|^2 - V(x), where p is the variable for Dx
    """
    return np.sin(2. * np.pi * x) + np.cos(4. * np.pi * x)


def hamiltonian_1d(x, p, v=V):
    """
    Return the Hamiltonian 1/2 * |Du| ** 2 - V(x) for HJB on (x, p)

    Note in mfg_1d, |Du|**2 = (du/dx)**2
    :param x: spatial variable, each element in this variable is a list of collocation points for a partition
    :param p: velocity variable, same format as x, values are derivative Du at each point in x
    :param v: bounded potential function
    :return: value of Hamiltonian on (x, p), same shape as x
    """
    assert len(x) == len(p)

    result = []
    for i in range(len(x)):
        result.append(p[i] ** 2 / 2 - v(x[i]))

    return result


def lagrangian_1d(x, q, v=V):
    """
    Return the Lagrangian associated with Hamiltonian H(x,Du) = 1/2 * |Du| ** 2 - V(x) for HJB on (x, q).

    By simple calculation, we see our lagrangian L(x, q) = 1/2 * |q| ** 2 + V(x)

    Note in mfg_1d, |Du|**2 = (du/dx)**2
    :param x: spatial variable, each element in this variable is a list of collocation points for a partition
    :param q: dual variable
    :param v: bounded potential function
    :return: value of Hamiltonian on (x, p), same shape as x
    """
    assert len(x) == len(q)

    result = []
    for i in range(len(x)):
        # Check if q[i] is a tensor and requires gradients
        if isinstance(q[i], torch.Tensor):
            qi = q[i].detach() if q[i].requires_grad else q[i]
        else:
            qi = q[i]

        # Similarly, check for x[i]
        if isinstance(x[i], torch.Tensor):
            xi = x[i].detach() if x[i].requires_grad else x[i]
        else:
            xi = x[i]
        result.append(qi ** 2 / 2 + v(xi))

    return result


def evaluate_RFM_1d(models, w, points):
    """
    Given RFM models and a trained set of weights, evaluate the model on every point
    :param models: list of RFM models, for each partition
    :param w: a list trained weights, for each partition
    :param points: the points we want to evaluate the model on
    :return: evaluated numerical solution, same shape as points
    """
    numerical_values = []

    out_total = None
    for m in range(len(models)):
        out = models[m](points)
        values = out.detach().numpy()
        if out_total is None:
            out_total = values
        else:
            out_total = np.concatenate((out_total, values), axis=1)

    numerical_values = np.dot(np.array(out_total), w.reshape(-1, 1))
    return numerical_values


def differentiate_RFM_1d(models, w, points):
    """
    Given RFM models on each partition and a trained set of weights, evaluate the model's derivative on every point in
    each partition
    :param models: RFM models
    :param w: trained weights
    :param points: Collocation points for each partition
    :return: evaluated derivative
    """
    u = RFM_function_factory(models, w)
    derivatives = []
    for k in range(len(points)):
        u_x = u(points[k])
        derivative = torch.autograd.grad(u_x, points[k], grad_outputs=torch.ones_like(u_x))[
            0].squeeze()

        derivatives.append(derivative)
    return derivatives


def second_derivative_RFM_1d(models, w: torch.Tensor, points: List[torch.Tensor]):
    """
    Given RFM models on each partition and a trained set of weights, evaluate the model's first and second order
    derivative on every point in each partition
    :param models: RFM models
    :param w: trained weights
    :param points: Collocation points for each partition
    :return: evaluated first and second order derivatives, same shape as points
    """
    u = RFM_function_factory(models, w)
    derivatives = []
    second_derivatives = []
    for k in range(len(points)):
        u_x = u(points[k])
        derivative = torch.autograd.grad(u_x, points[k], grad_outputs=torch.ones_like(u_x), create_graph=True)[
            0].squeeze()
        second_derivative = \
        torch.autograd.grad(derivative, points[k], grad_outputs=torch.ones_like(derivative))[
            0].squeeze()
        derivatives.append(derivative)
        second_derivatives.append(second_derivative)

    return derivatives, second_derivatives


def plot_RFM_1d(f, label, n_pts=1000, interval_length=INTERVAL_LENGTH):
    """
    :param f:
    :param label:
    :param n_pts:
    :return:
    """
    pts = torch.tensor(np.linspace(0, interval_length, n_pts), dtype=torch.float64, requires_grad=False).reshape([-1, 1])
    fx = f(pts)
    plt.figure()
    plt.plot(pts, fx, label=label, color='darkblue', linestyle='--')
    plt.legend()
    plt.show()


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
        )

    return rfm_function


def calculate_error_fp(m, u, eps=0.3, plot=False):
    """
    Calculate error in Fokker-Planck equation solved w.r.t given HJB
    :param m: ...
    :param u: ...
    :param eps:
    :param plot:
    :return:
    """

    pts = torch.tensor(np.linspace(0, 1, 1000), dtype=torch.float64, requires_grad=True).reshape([-1, 1])
    u_values = u(pts)
    q = torch.autograd.grad(u_values, pts, grad_outputs=torch.ones_like(u_values), create_graph=True)[0].squeeze()
    mq = m(pts) * q
    div = torch.autograd.grad(mq, pts, grad_outputs=torch.ones_like(mq))[0].squeeze()

    m_values = m(pts)
    dm = torch.autograd.grad(m_values, pts, grad_outputs=torch.ones_like(m_values), create_graph=True)[0].squeeze()
    laplace = torch.autograd.grad(dm, pts, grad_outputs=torch.ones_like(dm))[0].squeeze()

    error = - eps * laplace - div

    if plot:
        # plot error
        pts_np = pts.detach().numpy()
        error_np = error.detach().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(pts_np, error_np, label='Error')
        plt.xlabel('x')
        plt.ylabel('Error')
        plt.title('Fokker-Planck Error')
        plt.legend()
        plt.show()

    return error


def calculate_error_hjb(u, old_u, m, eps=0.3, plot=False):
    """
    Calculate error in HJB equation solved w.r.t given Fokker-Planck
    :param u: ...
    :param old_u: ...
    :param m: ...
    :return:
    """

    pts = torch.tensor(np.linspace(0, 1, 1000), dtype=torch.float64, requires_grad=True).reshape([-1, 1])
    q_x = torch.autograd.grad(old_u(pts), pts, grad_outputs=torch.ones_like(old_u(pts)), create_graph=True)[0].view(-1)

    m_x = m(pts)

    Fm_x = m_x ** 2

    Lq = torch.stack(lagrangian_1d(pts, q_x)).view(-1)

    Du = torch.autograd.grad(u(pts).sum(), pts, create_graph=True)[0].view(-1)
    Laplace_u = torch.autograd.grad(Du.sum(), pts, create_graph=True)[0].view(-1)

    error = - eps * Laplace_u + q_x * Du - Lq - Fm_x

    if plot:
        # plot error
        pts_np = pts.detach().numpy()
        error_np = error.detach().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(pts_np, error_np, label='Error')
        plt.xlabel('x')
        plt.ylabel('Error')
        plt.title('HJB Error')
        plt.legend()
        plt.show()

    return error


def constraint_test(f):
    n_pts = 2000
    pts = torch.tensor(np.linspace(0, 1, n_pts), dtype=torch.float64, requires_grad=False).reshape([-1, 1])

    values = f(pts)
    integral = (values.sum() / n_pts).item()

    periodicity_err = (values[0] - values[-1]).item()

    return integral, periodicity_err


def update_l1_err_test(old_f, new_f):
    n_pts = 2000
    pts = torch.tensor(np.linspace(0, 1, n_pts), dtype=torch.float64, requires_grad=False).reshape([-1, 1])

    diffs = torch.abs(old_f(pts) - new_f(pts))
    l1_err = (diffs.sum() / n_pts).item()
    return l1_err


def l1_norm(f, g, n_pts=2000):
    x = torch.linspace(0, 1, n_pts)
    abs_diff = torch.abs(f(x) - g(x))
    return torch.trapz(abs_diff, x)


def l2_norm(f, g, n_pts=2000):
    x = torch.linspace(0, 1, n_pts)
    diff_squared = (f(x) - g(x)) **2
    return torch.sqrt(torch.trapz(diff_squared, x))


def sup_norm(f, g, n_pts=2000):
    x = torch.linspace(0, 1, n_pts)
    diff_squared = (f(x) - g(x)) **2
    return torch.sqrt(torch.trapz(diff_squared, x))


def plot_errors(error_arr: ErrorArray, last_idx, label):
    iterations = range(1, last_idx+1)

    # plot cumulative error per iteration
    # cumulative_errors = np.zeros(last_idx)
    # for i in range(last_idx):
    #     cumulative_errors[i] = error_arr[i].sum().item()
    # plt.figure(figsize=(10, 6))
    # plt.plot(iterations, cumulative_errors, label=label+'Error')
    # plt.xlabel('iterations')
    # plt.ylabel('Error')
    # plt.title('Cumulative Error of'+label)
    # plt.legend()
    # plt.show()

    # plot L1 convergence by iteration
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, [error_arr[i].errors['l1-error'] for i in iterations], label=label + 'L1-Convergence')
    plt.xlabel('iterations')
    plt.ylabel('L1 Error')
    plt.title('L1 convergence error of' + label)
    plt.legend()
    plt.show()