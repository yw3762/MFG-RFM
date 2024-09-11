import torch
from torch import sin, cos, pi
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from typing import Callable, List, Tuple

INTERVAL_LENGTH = 1.0


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


def set_seed(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True


def init_local_RFM1d(J_n, x_min, x_max, debug=False):
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.uniform_(m.weight, a=-1, b=1)
            nn.init.uniform_(m.bias, a=-1, b=1)

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
        # Define RFM model in each partition, in mfg_1d, partition is just an interval [x_min, x_max]
        x_min = INTERVAL_LENGTH / M_p * k
        x_max = INTERVAL_LENGTH / M_p * (k + 1)
        models.append(init_local_RFM1d(J_n, x_min, x_max, debug))

        # Within each partition, get the collocation points (mfg_1d) as a column vector
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
    return (torch.roll(vals, -1) - 2 * vals + torch.roll(vals, 1)) / (h ** 2)


def fd_derivative(vals, h):
    return (torch.roll(vals, -1) - torch.roll(vals, 1)) / (2 * h)


# def solve_FP_fd(models, collocs, q_func, M_p, J_n, Q, eps=0.3):
#     q = [q_func(collocs[i]) for i in range(M_p)]
#     A_pde_fd = np.zeros([M_p * Q, M_p * J_n])
#
#     # We assume non-negativity constraint in RFM also follows from normalization constraint
#     A_constraints = np.zeros([2, M_p * J_n])  # one for boundary, one for normalization -> 2 in total
#     f = np.zeros([M_p * Q + 2, 1])
#
#     h = collocs[0][1] - collocs[0][0]
#
#     # all_values[m * J_n + j, k * Q + i] = \phi_{m,j}(p_{k, i})
#     all_values = np.zeros((M_p * J_n, M_p * collocs))
#
#     for k in range(M_p):
#         for m in range(M_p):
#             all_values[m*J_n: (m+1)*J_n, k * Q: (k+1)*Q] = models[m](collocs[k]).detach().numpy
#
#     for m in range(M_p):
#         for j in range(J_n):
#             Phi_m_j_idx = m * J_n + j
#             Phi_m_j_idx
#
#
#     for k in range(M_p):
#         for m in range(M_p):
#             # Evaluate the colloction points of partition-k on RFM of U_m
#             out = models[m](collocs[k])
#             # values[i,j] = f_{mj}(points[k,i]), where f_{mj} is feature function
#             values = out.detach().numpy()  # shape: (Q+1, J_n)
#             # Periodicity constraint, evaluate on boundary
#             if k == 0:
#                 A_constraints[0, m * J_n: (m + 1) * J_n] = values[0, :]
#             elif k == M_p - 1:
#                 A_constraints[0, m * J_n: (m + 1) * J_n] -= values[Q, :]
#
#             # Normalization constraint:
#             for i in range(Q):
#                 A_constraints[1, m * J_n: (m + 1) * J_n] += values[i, :]
#     A_fd = np.concatenate((A_pde_fd, A_constraints), axis=0)
#     f[-1] = 1 * M_p * Q  # normalize to 1
#
#     # Solve lstsq system
#     w_fd = lstsq(A_fd, f)[0]
#     w_fd = torch.tensor(w_fd.reshape((M_p, J_n)))
#     solution_fd = RFM_function_factory(models, w_fd)
#
#     plot_RFM_1d(solution_fd, "m_fd")
#
#     return solution_fd


# def solve_HJB_fd(models, collocs, m_func, q_func, M_p, J_n, Q, eps=0.3, lam=0):
#     pass


def test_fp_r_1(x, eps):
    """
    Assume m = pi/2 sin(pi x), u = const, then q = 0, then residual r for FP is eps * pi^3/2 sin(pi x)
    """
    return eps * (torch.pi ** 3) * torch.sin(torch.pi * x) / 2


def test_fp_r_2(x, eps):
    """
    Assume m = pi/2 sin(pi x), u = x, then q = 1, dq = 0, then residual r for FP is eps * pi^3/2 sin(pi x) - pi^2 / 2 cos(pi x)
    """
    return eps * (torch.pi ** 3) * torch.sin(torch.pi * x) / 2 - (torch.pi ** 2) * torch.cos(torch.pi * x) / 2


def test_fp_r_3(x, eps):
    """
    Assume m = pi/2 sin(pi x), u = x^2, then q = 2x, dq = 2, then residual r for FP is eps * pi^3/2 sin(pi x) - pi^2 / 2 cos(pi x)
    """
    return eps * (torch.pi ** 3) * torch.sin(torch.pi * x) / 2 - torch.pi * torch.sin(
        torch.pi * x) - torch.pi ** 2 * x * torch.cos(torch.pi * x)


def test_fp_r_4(x, eps):
    """
    Assume m = pi/2 sin(pi x), u = cos(2 pi x), then q = -2pi sin(2pi x), dq = -4 pi**2 cos(2pi x),
    then residual r for FP is eps * pi^3/2 sin(pi x) + pi^3 (cos(pi x) sin(2pi x) + 2sin(pi x) cos(2pi x))
    """
    return eps * (pi ** 3) * sin(pi * x) / 2 + (pi ** 3) * (
                cos(pi * x) * sin(2 * pi * x) + 2 * sin(pi * x) * cos(2 * pi * x))


def solve_FP(models, collocs, q_func, dq_func, M_p, J_n, Q, eps=0.3, MMS_r=test_fp_r_4):
    q = [q_func(collocs[i]) for i in range(M_p)]
    dq = [dq_func(collocs[i]) for i in range(M_p)]

    # q_MMS = [torch.ones_like(q[i]) for i in range(M_p)]
    # dq_MMS = [torch.zeros_like(dq[i]) for i in range(M_p)]
    # q_MMS = [2*collocs[i].view(-1) for i in range(M_p)]
    # dq_MMS = [2*torch.ones_like(dq[i]) for i in range(M_p)]
    q_MMS = [- 2 * pi * (sin(2 * pi * collocs[i])).view(-1) for i in range(M_p)]
    dq_MMS = [- 4 * pi ** 2 * (cos(2 * pi * collocs[i])).view(-1) for i in range(M_p)]

    # Compute lstsq system
    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])
    A_pde_fd = np.zeros([M_p * Q, M_p * J_n])

    # We assume non-negativity constraint in RFM also follows from normalization constraint
    A_constraints = np.zeros([2, M_p * J_n])  # one for boundary, one for normalization -> 2 in total
    f = np.zeros([M_p * Q + 2, 1])

    h = collocs[0][1] - collocs[0][0]

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m
            out = models[m](collocs[k])
            # values[i,j] = f_{mj}(points[k,i]), where f_{mj} is feature function
            values = out.detach().numpy()  # shape: (Q+1, J_n),

            # Compute first and second order derivative dm/dx and d^2m/dx^2
            grads_2 = []

            # Compute divergence term div(q m)
            div = []
            div_MMS = []

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
                div_MMS.append((g_1.squeeze() * q_MMS[k] + out[:, i] * dq_MMS[k]).detach().numpy())

            grads_2 = np.array(grads_2).T  # grads_2[j,i] = f''_{mi}(points[k, j])
            div = np.array(div).T  # div[j,i] = div(f_{mi}q)(points[k, j])
            div_MMS = np.array(div_MMS).T

            # Impose PDE condition: Lm = -eps * dm^2/dx^2 - div(m * q)
            Lm = - eps * grads_2 - div
            Lm_MMS = - eps * grads_2 - div_MMS  # Lm[j,i] = Lm(points[k, j]) with i-th factor of m

            # Specifying A_pde
            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lm[:Q, :]
            # A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lm_MMS[:Q, :]

            # Periodicity constraint, evaluate on boundary
            if k == 0:
                A_constraints[0, m * J_n: (m + 1) * J_n] = values[0, :]
            elif k == M_p - 1:
                A_constraints[0, m * J_n: (m + 1) * J_n] -= values[Q, :]

            # Normalization constraint:
            for i in range(Q):
                A_constraints[1, m * J_n: (m + 1) * J_n] += values[i, :]

        # MMS RHS r
        # f[k * Q:(k + 1) * Q, :] = MMS_r(collocs[k], eps)[:Q].detach().numpy()
    A = np.concatenate((A_pde, A_constraints), axis=0)
    f[-1] = 1 * M_p * Q  # normalize to 1

    # Solve lstsq system
    w = lstsq(A, f)[0]
    w = torch.tensor(w.reshape((M_p, J_n)))

    solution = RFM_function_factory(models, w)

    plot_RFM_1d(solution, "m")

    # Anticipated MMS solution is m = pi/2 sin(pi x)
    x = np.linspace(0, 1, M_p * Q + 1)
    mx = np.pi / 2 * np.sin(np.pi * x)
    plt.plot(x, mx, "")
    plt.show()

    return solution


def compare_RFM_true(RFM_sol, true_sol):
    n_pts = 1000
    pts = torch.tensor(np.linspace(0, 1, n_pts), dtype=torch.float64, requires_grad=False).reshape([-1, 1])

    diffs = torch.abs(RFM_sol(pts).view(-1) - true_sol(pts).view(-1))
    l1_err = (diffs.sum() / n_pts).item()
    return l1_err


def test_hjb_r_1(x, eps):
    """
    Let u_1 = sin(2 pi x), so Du_1=2*pi *cos(2*pi * x), Laplacian_u_1 = -4*pi**2 * sin(2 * pi * x)
    Let m_1 = pi * sin(pi * x) / 2
    Let u_0 = cos(2pi x) hence q_1 = -2*pi*sin(2*pi*x),
    Our r should be -eps*Laplacian_u_1 + q_1 *Du_1 - L_q_1 - F(m_1(x))
    """
    return (eps * 4 * pi ** 2 * sin(2 * pi * x) - 4 * pi ** 2 * sin(2 * pi * x) * cos(2 * pi * x) - 2 * (pi ** 2) * (
                sin(2 * pi * x) ** 2)) - sin(2 * pi * x) - cos(4 * pi * x) - pi ** 2 * (sin(pi * x) ** 2) / 4


def test_hjb_r_2(x, eps):
    return eps * 4 * (pi ** 2) * sin(2 * pi * x) + 2* pi * cos(2 * pi * x) - 1/2 - sin(2*pi*x) - cos(4*pi*x)


def test_hjb_r_3(x, eps):
    return eps * 4 * (pi ** 2) * sin(2 * pi * x) + 2* pi * cos(2 * pi * x) - 1/2 - sin(2*pi*x) - cos(4*pi*x) - x**2


def test_hjb_r_4(x, eps):
    return eps * 4 * (pi ** 2) * sin(2 * pi * x) + 2*(x**2)* pi * cos(2 * pi * x) - x**4/2 - sin(2*pi*x) - cos(4*pi*x) - x**2


def test_hjb_r_5(x, eps):
    return (eps * 4 * pi ** 2 * sin(2 * pi * x) - 4 * pi ** 2 * sin(2 * pi * x) * cos(2 * pi * x) - 2 * (pi ** 2) * (
                sin(2 * pi * x) ** 2)) - sin(2 * pi * x) - cos(4 * pi * x)

def solve_HJB(models, collocs, m_func, q_func, M_p, J_n, Q, eps=0.3, lam=0, MMS_attempt=5):
    q = [q_func(collocs[i]) for i in range(M_p)]

    if MMS_attempt == 1:
        MMS_r = test_hjb_r_1
        q_MMS = [-2 * pi * sin(2 * pi * collocs[i]).view(-1) for i in range(M_p)]
    elif MMS_attempt == 2:
        MMS_r = test_hjb_r_2
        q_MMS = [torch.ones_like(collocs[i]).view(-1) for i in range(M_p)]
    elif MMS_attempt == 3:
        MMS_r = test_hjb_r_3
        q_MMS = [torch.ones_like(collocs[i]).view(-1) for i in range(M_p)]
    elif MMS_attempt == 4:
        MMS_r = test_hjb_r_4
        q_MMS = [(collocs[i] ** 2).view(-1) for i in range(M_p)]
    elif MMS_attempt == 5:
        MMS_r = test_hjb_r_5
        q_MMS = [-2 * pi * sin(2 * pi * collocs[i]).view(-1) for i in range(M_p)]

    # Compute lstsq system
    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])

    # We assume non-negativity constraint in RFM also follows from normalization constraint
    A_constraints = np.zeros([2, M_p * J_n])  # one for boundary, one for normalization -> 2 in total
    f = np.zeros([M_p * Q + 2, 1])

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m (HJB) and M_m (FP)
            out = models[m](collocs[k])
            values_hjb = out.detach().numpy()

            # Compute first and second order derivative du/dx and d^2u/dx^2 for HJB
            grads_2 = []
            q_du = []  # Compute the (q * Du) term
            q_du_MMS = []

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
                q_du_MMS.append((g_1.squeeze() * q_MMS[k]).detach().numpy())

            grads_2 = np.array(grads_2).T  # grads[j,i] = f''_{mi}(points[k, j])
            q_du = np.array(q_du).T  # q_du[j, i] = f'_{mi}(points[k, j]) * q(points[k, j])
            q_du_MMS = np.array(q_du_MMS).T

            # Impose PDE condition: Lu = -eps * du^2/dx^2 - div(u * q)
            # Lu[j, i] =  - eps * f'_{mi}(points[k, j]) + f'_{mi}(points[k, j]) * q(points[k, j])
            Lu = - eps * grads_2 + q_du  # shape=(Q+1, J_n)
            Lu_MMS = - eps * grads_2 + q_du_MMS

            # A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lu[:Q, :] + lam
            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lu_MMS[:Q, :] + lam

            # Periodicity constraint, evaluate on boundary
            if k == 0:
                A_constraints[0, m * J_n: (m + 1) * J_n] = values_hjb[0, :]
            elif k == M_p - 1:
                A_constraints[0, m * J_n: (m + 1) * J_n] -= values_hjb[Q, :]

            # Normalization constraint:
            for i in range(Q):
                A_constraints[1, m * J_n: (m + 1) * J_n] += values_hjb[i, :]

        # The f-side of discretized Lu=f system
        Lq = lagrangian_1d(collocs[k], q[k])
        Fm = m_func(collocs[k]).view(-1) ** 2  # The coupling term is F(m) = m^2
        summed = Fm + Lq
        trimmed = summed[:Q].detach().numpy().reshape(-1, 1)

        # The f-side of MMS
        Lq_MMS = lagrangian_1d(collocs[k], q_MMS[k])

        if MMS_attempt == 1:
            Fm_MMS = ((pi * sin(pi * collocs[k]) / 2) ** 2).view(-1)
        elif MMS_attempt == 2:
            Fm_MMS = torch.zeros_like(collocs[k]).view(-1)
        elif MMS_attempt == 3:
            Fm_MMS = (collocs[k] ** 2).view(-1)
        elif MMS_attempt == 4:
            Fm_MMS = (collocs[k] ** 2).view(-1)
        elif MMS_attempt == 5:
            Fm_MMS = torch.zeros_like(collocs[k]).view(-1)
        summed_MMS = Fm_MMS + Lq_MMS

        trimmed_MMS = summed_MMS[:Q].detach().numpy().reshape(-1, 1)
        # f[k * Q:(k + 1) * Q, :] = trimmed
        MMS_r_value = MMS_r(collocs[k], eps)[:Q].detach().numpy()
        f[k * Q:(k + 1) * Q, :] = trimmed_MMS + MMS_r_value

    A = np.concatenate((A_pde, A_constraints), axis=0)
    f[-1] = 0  # normalize to 0

    # Solve lstsq system
    w = lstsq(A, f)[0]
    w = torch.tensor(w.reshape((M_p, J_n)))

    solution = RFM_function_factory(models, w)
    plot_RFM_1d(solution, "u")


    # Anticipated MMS solution is u = sin(2 pi x),
    x = np.linspace(0, 1, M_p * Q + 1)
    ux = np.sin(2 * np.pi * x)
    rfm_x = solution(torch.tensor(x.reshape([-1, 1]))).view(-1).numpy()
    plt.plot(x, ux, label="true u", linestyle='-')
    plt.plot(x, rfm_x, label="RFM u", linestyle='--')
    plt.legend()
    plt.show()
    print("The L1 difference between true solution and RFM solution is", compare_RFM_true(solution, lambda x: sin(2*pi*x)))

    return solution


def policy_iteration(M_p_hjb, J_n_hjb, M_p_fp, J_n_fp, Q_hjb, Q_fp, n_iters=20, eps=0.3, tau=1e-6):
    # fix datatype
    torch.set_default_dtype(torch.float64)

    # Initialize RFMs for FP and HJB with zero weights
    models_fp, collocs_fp = init_rfm(M_p_fp, J_n_fp, Q_fp)
    models_hjb, collocs_hjb = init_rfm(M_p_hjb, J_n_hjb, Q_hjb)
    w_hjb = torch.zeros((M_p_hjb, J_n_hjb))
    w_fp = torch.zeros((M_p_fp, J_n_fp))

    # historical solutions and policies
    historical_m = [RFM_function_factory(models_fp, w_fp)]
    historical_u = [RFM_function_factory(models_hjb, w_hjb)]
    historical_q = [None]
    historical_q[0], dq = second_diff_RFM_function(historical_u[0])

    # residual
    residuals_m = []
    residuals_u = []
    fd_residual_m = []
    fd_residuals_u = []

    # main loop
    for k in range(1, n_iters + 1):
        print("Iteration {}".format(k))

        historical_m.append(solve_FP(models_fp, collocs_fp, historical_q[k - 1], dq, M_p_fp, J_n_fp, Q_fp))

        historical_u.append(
            solve_HJB(models_hjb, collocs_hjb, historical_m[k], historical_q[k - 1], M_p_hjb, J_n_hjb, Q_hjb))

        new_q, dq = second_diff_RFM_function(historical_u[k])
        historical_q.append(new_q)
    return historical_m[-1], historical_u[-1]


def plot_RFM_1d(f, label, n_pts=1000, interval_length=INTERVAL_LENGTH):
    pts = torch.tensor(np.linspace(0, interval_length, n_pts), dtype=torch.float64, requires_grad=False).reshape(
        [-1, 1])
    fx = f(pts)
    plt.figure()
    plt.plot(pts, fx, label=label, color='darkblue', linestyle='--')
    plt.legend()
    plt.show()


def test():
    set_seed(100)
    M_p_hjb = M_p_fp = 4
    J_n_hjb = J_n_fp = 50
    Q_hjb = Q_fp = 100
    n_iters = 8

    m, u = policy_iteration(M_p_hjb, J_n_hjb, M_p_fp, J_n_fp, Q_hjb, Q_fp, n_iters)

    plot_RFM_1d(m, "final m")
    plot_RFM_1d(u, "final u")


if __name__ == '__main__':
    test()
