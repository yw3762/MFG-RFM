import torch
from torch import sin, cos, pi
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from typing import Callable, List, Tuple

from rfm_mfg_stationary.mfg_1d_old.utils.utils_1d import plot_by_iter, hamiltonian_1d

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
            return d0 * y0 + d1 * y1 + (d2 + d3) * y2 + d4 * y4 # d3 * y2 instead of d3 * y3
        else:
            return d0 * y0 + d1 * y1 + d2 * y2 + d3 * y3 + d4 * y4


        # y is the change of variable in normalized coordinate \tilde{x}
        # tilde_x = self.a * (x - self.x_0)

        # pass the normalized variable into hidden-layer
        # phi_nj = self.hidden_layer(tilde_x)  # phi_nj[i, j] = \phi_{n,j}(tilde_x[i])

        # location indicator
        # d0 = tilde_x < -5 / 4
        # d1 = (tilde_x >= -5 / 4) & (tilde_x < -3 / 4)
        # d2 = (tilde_x >= -3 / 4) & (tilde_x < 3 / 4)
        # d3 = (tilde_x >= 3 / 4) & (tilde_x < 5 / 4)
        # d4 = tilde_x >= 5 / 4

        # y_i are the PoU w.r.t each location of x
        # y0 = 0
        # y1 = phi_nj * (1 + torch.sin(2 * np.pi * tilde_x)) / 2
        # y2 = phi_nj
        # y3 = phi_nj * (1 - torch.sin(2 * np.pi * tilde_x)) / 2
        # y4 = 0
        #
        # values = d1 * y1 + d2 * y2 + d3 * y3
        #
        # if self.x_min == 0:
        #     values +=
        # elif self.x_max == INTERVAL_LENGTH:
        #     values +=

        #
        # return values


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


def solve_FP_auto(models, collocs, q_func, dq_func, M_p, J_n, Q, eps=0.3):
    q = [q_func(collocs[i]).detach() for i in range(M_p)]
    dq = [dq_func(collocs[i]).detach() for i in range(M_p)]

    # Compute lstsq system
    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])

    # Choosing rescaling parameters
    c = 100
    divisor_interiors = np.zeros(M_p * Q)
    divisor_boundary = np.zeros(3)

    # We assume non-negativity constraint in RFM also follows from normalization constraints
    # One for boundary, one for normalization, new one for derivative on boundary -> 3 in total
    A_constraints = np.zeros([3, M_p * J_n])
    f = np.zeros([M_p * Q + 3, 1])

    h = collocs[0][1] - collocs[0][0]

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m
            out = models[m](collocs[k])
            # values[i,j] = f_{mj}(points[k,i]), where f_{mj} is feature function
            values = out.detach().numpy()  # shape: (Q+1, J_n),

            # Compute first and second order derivative dm/dx and d^2m/dx^2
            grads_1 = []
            grads_2 = []

            # Compute divergence term div(q m)
            div = []

            for i in range(J_n):
                # Compute gradient of i-th basis function
                g_1 = torch.autograd.grad(outputs=out[:, i], inputs=collocs[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=True, retain_graph=True)[0]
                grads_1.append(g_1.squeeze().detach().numpy())

                # Compute second order gradient for i-th basis function
                g_2 = torch.autograd.grad(outputs=g_1[:, 0], inputs=collocs[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          retain_graph=True)[0]
                grads_2.append(g_2.squeeze().detach().numpy())

                # In d=1, div(m*q) = d(m*q)/dx = m' * q + m * q'
                div.append((g_1.squeeze() * q[k] + out[:, i] * dq[k]).detach().numpy())

            grads_1 = np.array(grads_1).T  # grads_1[j,i] = D(\phi_{mi}(points[k, j]) * \psi_m(points[k, j]))
            grads_2 = np.array(grads_2).T  # grads_2[j,i] = D^2(\phi_{mi}(points[k, j]) * \psi_m(points[k, j]))
            div = np.array(div).T  # div[j,i] = div(\phi_{mi}*\psi_m * q)(points[k, j]), 0<=j<=Q, 0<=i<J_n

            # Impose PDE condition: Lm = -eps * dm^2/dx^2 - div(m * q)
            Lm = - eps * grads_2 - div # Lm[j,i] = L(\phi_{mi}\psi_m)(points[k, j]) 0<=j<=Q, 0<=i<J_n

            # Calculate rescaling divisor
            max_row = np.max(np.abs(Lm), axis=1)

            indices = np.arange(k*Q, (k+1)*Q+1) % len(divisor_interiors)
            divisor_interiors[indices] = np.maximum(divisor_interiors[indices], max_row)

            # Specifying A_pde
            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lm[:Q, :]

            # Periodicity constraint, evaluate on boundary
            if k == 0:
                A_constraints[0, m * J_n: (m + 1) * J_n] += values[0, :]
                divisor_boundary[0] = max(divisor_boundary[0], np.max(np.abs(values[0, :])))
            elif k == M_p - 1:
                A_constraints[0, m * J_n: (m + 1) * J_n] -= values[Q, :]
                divisor_boundary[0] = max(divisor_boundary[0], np.max(np.abs(values[Q, :])))

            # New C^1 condition on boundary
            if k == 0:
                A_constraints[1, m * J_n: (m + 1) * J_n] += grads_1[0, :]
                divisor_boundary[1] = max(divisor_boundary[1], np.max(np.abs(grads_1[0, :])))
            elif k == M_p - 1:
                A_constraints[1, m * J_n: (m + 1) * J_n] -= grads_1[Q, :]
                divisor_boundary[1] = max(divisor_boundary[1], np.max(np.abs(grads_1[Q, :])))

            # Normalization constraint:
            for i in range(Q):
                A_constraints[2, m * J_n: (m + 1) * J_n] += values[i, :]

    # Rescaling parameters
    lambda_interiors = c / divisor_interiors
    lambda_boundary = c / divisor_boundary

    lambda_boundary[0] = 1000
    lambda_boundary[1] = 1000
    lambda_boundary[2] = 10

    lambda_together = np.concatenate((lambda_interiors, lambda_boundary), axis=0).reshape((-1, 1))

    # A_pde = A_pde[1:, ]  # Take out boundary points w.r.t. interior PDE condition
    # lambda_interiors = lambda_interiors[1:]  # Take out boundary points w.r.t interior PDE condition
    # f = f[1:, ]

    # reset
    # lambda_together = np.ones_like(lambda_together)

    concatenated_A = np.concatenate((A_pde, A_constraints), axis=0)
    A = concatenated_A * lambda_together
    f[-1] = 1 * M_p * Q  # normalize to 1

    f = f * lambda_together

    # Solve lstsq system
    w = lstsq(A, f)[0]
    w = torch.tensor(w.reshape((M_p, J_n)))

    solution = RFM_function_factory(models, w)

    return solution


def solve_FP(models, collocs, q_func, dq_func, M_p, J_n, Q, eps=0.3):
    q = [q_func(collocs[i]).detach() for i in range(M_p)]
    dq = [dq_func(collocs[i]).detach() for i in range(M_p)]

    # Compute lstsq system
    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])
    A_MMS_pde = np.zeros([M_p * Q, M_p * J_n])

    # We assume non-negativity constraint in RFM also follows from normalization constraints
    # One for boundary, one for normalization, new one for derivative on boundary -> 3 in total
    A_constraints = np.zeros([3, M_p * J_n])
    # A_4 = np.zeros([M_p - 1, M_p * J_n])
    f = np.zeros([M_p * Q + 3, 1])

    h = collocs[0][1] - collocs[0][0]

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m
            out = models[m](collocs[k])
            # values[i,j] = f_{mj}(points[k,i]), where f_{mj} is feature function
            values = out.detach().numpy()  # shape: (Q+1, J_n),

            # Compute first and second order derivative dm/dx and d^2m/dx^2
            grads_1 = []
            grads_2 = []

            # Compute divergence term div(q m)
            div = []

            for i in range(J_n):
                # Compute gradient of i-th basis function
                g_1 = torch.autograd.grad(outputs=out[:, i], inputs=collocs[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          create_graph=True, retain_graph=True)[0]
                grads_1.append(g_1.squeeze().detach().numpy())

                # Compute second order gradient for i-th basis function
                g_2 = torch.autograd.grad(outputs=g_1[:, 0], inputs=collocs[k],
                                          grad_outputs=torch.ones_like(out[:, i]),
                                          retain_graph=True)[0]
                grads_2.append(g_2.squeeze().detach().numpy())

                # In d=1, div(m*q) = d(m*q)/dx = m' * q + m * q'
                div.append((g_1.squeeze() * q[k] + out[:, i] * dq[k]).detach().numpy())

            grads_1 = np.array(grads_1).T  # grads_1[j,i] = f'_{mi}(points[k, j])
            grads_2 = np.array(grads_2).T  # grads_2[j,i] = f''_{mi}(points[k, j])
            div = np.array(div).T  # div[j,i] = div(f_{mi}q)(points[k, j])

            # Impose PDE condition: Lm = -eps * dm^2/dx^2 - div(m * q)
            Lm = - eps * grads_2 - div  # Lm[j,i] = Lm(points[k, j]) with i-th factor of m

            # Specifying A_pde
            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lm[:Q, :]

            # Periodicity constraint, evaluate on boundary
            weight_c0 = 100
            if k == 0:
                A_constraints[0, m * J_n: (m + 1) * J_n] += weight_c0 * values[0, :]
            elif k == M_p - 1:
                A_constraints[0, m * J_n: (m + 1) * J_n] -= weight_c0 * values[Q, :]

            # New C^1 condition on boundary
            weight_c1 = 100
            if k == 0:
                A_constraints[2, m * J_n: (m + 1) * J_n] += weight_c1 * grads_1[0, :]
            elif k == M_p - 1:
                A_constraints[2, m * J_n: (m + 1) * J_n] -= weight_c1 * grads_1[Q, :]

            # Normalization constraint:
            for i in range(Q):
                A_constraints[1, m * J_n: (m + 1) * J_n] += 1 * values[i, :]

    A = np.concatenate((A_pde, A_constraints), axis=0)
    f[M_p * Q + 1] = 1 * M_p * Q  # normalize to 1

    A = A[1:, ]  # Take out boundary points w.r.t. interior PDE condition
    f = f[1:, ]

    # Solve lstsq system
    w = lstsq(A, f)[0]
    w = torch.tensor(w.reshape((M_p, J_n)))

    solution = RFM_function_factory(models, w)

    return solution


def compare_RFM_true(RFM_sol, true_sol):
    n_pts = 1000
    pts = torch.tensor(np.linspace(0, 1, n_pts), dtype=torch.float64, requires_grad=False).reshape([-1, 1])

    diffs = torch.abs(RFM_sol(pts).view(-1) - true_sol(pts).view(-1))
    l1_err = (diffs.sum() / n_pts).item()
    return l1_err


def solve_HJB(models, collocs, m_func, q_func, M_p, J_n, Q, eps=0.3, lam=0, MMS_attempt=5):
    q = [q_func(collocs[i]).detach() for i in range(M_p)]

    # Compute lstsq system
    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])

    # We assume non-negativity constraint in RFM also follows from normalization constraint
    A_constraints = np.zeros([3, M_p * J_n])  # 2 for boundary, one for normalization -> 3 in total
    f = np.zeros([M_p * Q + 3, 1])

    for k in range(M_p):
        # Lq = lagrangian_1d(collocs[k], q[k]).detach().numpy().reshape(-1, 1)
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m (HJB) and M_m (FP)
            out = models[m](collocs[k])
            values_hjb = out.detach().numpy()

            # Compute first and second order derivative du/dx and d^2u/dx^2 for HJB
            grads_1 = []
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
                grads_1.append(g_1.squeeze().detach().numpy())
                grads_2.append(g_2.squeeze().detach().numpy())

                q_du.append((g_1.squeeze() * q[k]).detach().numpy())

            grads_1 = np.array(grads_1).T
            grads_2 = np.array(grads_2).T  # grads[j,i] = f''_{mi}(points[k, j])
            q_du = np.array(q_du).T  # q_du[j, i] = f'_{mi}(points[k, j]) * q(points[k, j])

            # Impose PDE condition: Lu = -eps * du^2/dx^2 - div(u * q)
            # Lu[j, i] =  - eps * f'_{mi}(points[k, j]) + f'_{mi}(points[k, j]) * q(points[k, j])
            Lu = - eps * grads_2 + q_du  # shape=(Q+1, J_n)

            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lu[:Q, :] + lam

            # Periodicity constraint, evaluate on boundary
            weight_c0 = 400
            if k == 0:
                A_constraints[0, m * J_n: (m + 1) * J_n] += weight_c0 * values_hjb[0, :]  # Q * M_p
            elif k == M_p - 1:
                A_constraints[0, m * J_n: (m + 1) * J_n] -= weight_c0 * values_hjb[Q, :]

            # C^1 Periodicity constraint, evaluate on boundary
            weight_c1 = 400
            if k == 0:
                A_constraints[1, m * J_n: (m + 1) * J_n] += weight_c1 * grads_1[0, :]
            elif k == M_p - 1:
                A_constraints[1, m * J_n: (m + 1) * J_n] -= weight_c1 * grads_1[Q, :]

            # Normalization constraint:
            for i in range(Q):
                A_constraints[2, m * J_n: (m + 1) * J_n] += 1 * values_hjb[i, :]

        # The f-side of discretized Lu=f system
        Lq = lagrangian_1d(collocs[k], q[k])
        weight_Lq = 1
        weight_Fm = 1
        Fm = m_func(collocs[k]).view(-1) ** 2  # The coupling term is F(m) = m^2
        summed = weight_Fm * Fm + weight_Lq * Lq
        trimmed = summed[:Q].detach().numpy().reshape(-1, 1)
        f[k * Q:(k + 1) * Q, :] = trimmed

    A = np.concatenate((A_pde, A_constraints), axis=0)
    f[-1] = 0  # normalize to 0

    A = A[1:, ]  # Take out boundary points w.r.t. interior PDE condition
    f = f[1:, ]

    # Solve lstsq system
    w = lstsq(A, f)[0]
    w = torch.tensor(w.reshape((M_p, J_n)))

    solution = RFM_function_factory(models, w)

    return solution


def solve_HJB_auto(models, collocs, m_func, q_func, M_p, J_n, Q, eps=0.3, lam=0):
    q = [q_func(collocs[i]).detach() for i in range(M_p)]

    # Choosing rescaling parameters
    c = 100
    divisor_interiors = np.zeros(M_p * Q)
    divisor_boundary = np.zeros(3)

    # Compute lstsq system
    # place-holder variables for A, where f is 0 by definition
    A_pde = np.zeros([M_p * Q, M_p * J_n])

    # We assume non-negativity constraint in RFM also follows from normalization constraint
    A_constraints = np.zeros([3, M_p * J_n])  # 2 for boundary, one for normalization -> 3 in total
    f = np.zeros([M_p * Q + 3, 1])

    for k in range(M_p):
        for m in range(M_p):
            # Evaluate the colloction points of partition-k on RFM of U_m (HJB) and M_m (FP)
            out = models[m](collocs[k])
            values_hjb = out.detach().numpy()

            # Compute first and second order derivative du/dx and d^2u/dx^2 for HJB
            grads_1 = []
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
                grads_1.append(g_1.squeeze().detach().numpy())
                grads_2.append(g_2.squeeze().detach().numpy())

                q_du.append((g_1.squeeze() * q[k]).detach().numpy())

            grads_1 = np.array(grads_1).T
            grads_2 = np.array(grads_2).T  # grads[j,i] = f''_{mi}(points[k, j])
            q_du = np.array(q_du).T  # q_du[j, i] = f'_{mi}(points[k, j]) * q(points[k, j])

            # Impose PDE condition: Lu = -eps * du^2/dx^2 - div(u * q)
            # Lu[j, i] =  - eps * f'_{mi}(points[k, j]) + f'_{mi}(points[k, j]) * q(points[k, j])
            Lu = - eps * grads_2 + q_du # shape=(Q+1, J_n)

            # Calculate rescaling divisor
            max_row = np.max(np.abs(Lu), axis=1)

            indices = np.arange(k * Q, (k + 1) * Q + 1) % len(divisor_interiors)
            divisor_interiors[indices] = np.maximum(divisor_interiors[indices], max_row)

            A_pde[k * Q: (k + 1) * Q, m * J_n: (m + 1) * J_n] = Lu[:Q, :] + lam

            # Periodicity constraint, evaluate on boundary
            if k == 0:
                A_constraints[0, m * J_n: (m + 1) * J_n] += values_hjb[0, :]
            elif k == M_p - 1:
                A_constraints[0, m * J_n: (m + 1) * J_n] -= values_hjb[Q, :]

            # C^1 Periodicity constraint, evaluate on boundary
            if k == 0:
                A_constraints[1, m * J_n: (m + 1) * J_n] += grads_1[0, :]
            elif k == M_p - 1:
                A_constraints[1, m * J_n: (m + 1) * J_n] -= grads_1[Q, :]

            # Normalization constraint:
            for i in range(Q):
                A_constraints[2, m * J_n: (m + 1) * J_n] += values_hjb[i, :]

        # The f-side of discretized Lu=f system

        Lq = lagrangian_1d(collocs[k], q[k])
        Fm = m_func(collocs[k]).view(-1) ** 2  # The coupling term is F(m) = m^2
        summed = Fm + Lq
        trimmed = summed[:Q].detach().numpy().reshape(-1, 1)
        f[k * Q:(k + 1) * Q, :] = trimmed

    # Rescaling parameters
    lambda_interiors = c / divisor_interiors
    A_pde = A_pde * lambda_interiors

    lambda_boundary = c / divisor_boundary

    lambda_boundary[0] = 100
    lambda_boundary[1] = 100
    lambda_boundary[2] = 10


    lambda_together = np.concatenate((lambda_interiors, lambda_boundary), axis=0).reshape((-1, 1))

    # reset
    lambda_together = np.ones_like(lambda_together)

    concatenated_A = np.concatenate((A_pde, A_constraints), axis=0)
    A = concatenated_A * lambda_together

    f[-1] = 0  # normalize to 0
    f = f * lambda_together

    # Solve lstsq system
    w = lstsq(A, f)[0]
    w = torch.tensor(w.reshape((M_p, J_n)))

    solution = RFM_function_factory(models, w)

    return solution


def plot_residuals(pde_residual, system_residual, pts, label):
    plt.plot(pts.view(-1).numpy(), pde_residual.numpy(), label="Single PDE FD residual for " + label)
    plt.plot(pts.view(-1).numpy(), system_residual.numpy(), label="System FD residual for " + label, linestyle='--')
    plt.xlabel('pts')
    plt.ylabel('FD error')
    plt.legend()
    plt.show()


def get_fd_residuals(prev_q, curr_m, curr_u, curr_q, eps):

    # TODO: Check how we take these derivatives. It should be clear that
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
    hjb_fd_residual = - eLapU + prevQ_Du - prevLq - Fm

    # Calculate system residual
    hjb_system_fd_residual = -eps * fd_laplacian(u_vals, h) + hamiltonian_1d(pts, fd_derivative(u_vals, h)) - m_vals ** 2
    fp_system_fd_residual = -eps * fd_laplacian(m_vals, h) - fd_derivative(m_vals * curr_q_vals, h)
    system_fd_residual = torch.abs(hjb_system_fd_residual) + torch.abs(fp_system_fd_residual)

    # Plot residuals
    plot_residuals(fp_fd_residual, fp_system_fd_residual, pts, "Fokker Planck")
    plot_residuals(hjb_fd_residual, hjb_system_fd_residual, pts, "HJB")

    return torch.sum(torch.abs(fp_fd_residual)) / n_pts, torch.sum(torch.abs(hjb_fd_residual)) / n_pts, torch.sum(
        torch.abs(system_fd_residual)) / n_pts


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

    # Calculate FD residuals
    fd_residual_m = np.zeros(n_iters + 1)
    fd_residual_u = np.zeros(n_iters + 1)
    fd_residual_system = np.zeros(n_iters + 1)

    # main loop
    for k in range(1, n_iters + 1):
        print("Iteration {}".format(k))

        historical_m.append(solve_FP_auto(models_fp, collocs_fp, historical_q[k - 1], dq, M_p_fp, J_n_fp, Q_fp))
        plot_RFM_1d(historical_m[k], "m^(" + str(k) + ")")

        historical_u.append(
            solve_HJB(models_hjb, collocs_hjb, historical_m[k], historical_q[k - 1], M_p_hjb, J_n_hjb, Q_hjb))

        new_q, dq = second_diff_RFM_function(historical_u[k])
        historical_q.append(new_q)

        if k == 1:
            def plot_u1(u1, eps):
                # Plotting u_1 to true u_1 together
                n_pts = 1000
                pts = torch.linspace(0, 1, n_pts, dtype=torch.float64).reshape(-1, 1)
                u1_true = - 1 / (12 * eps) + pts / (2 * eps) + np.sin(2 * np.pi * pts) / (
                            4 * eps * np.pi ** 2) + np.cos(
                    4 * np.pi * pts) / (16 * eps * np.pi ** 2) - pts ** 2 / (2 * eps)

                # FD difference using true_u1
                h = (pts[1] - pts[0])[0]
                eLapU = eps * fd_laplacian(u1_true.view(-1), h)
                Lq = lagrangian_1d(pts.view(-1), torch.zeros_like(u1_true).view(-1))
                true_FD_residual = eLapU.view(-1) - Lq - torch.ones_like(Lq)

                plt.figure(figsize=(10, 6))
                plt.plot(pts, u1_true, label='true u1')
                plt.plot(pts, u1(pts).detach().numpy(), label='numerical u1')
                plt.xlabel('x')
                plt.ylabel('u^1(x)')
                plt.title('Numerical u^1 vs true u^1')
                plt.legend()
                plt.show()
            plot_u1(historical_u[k], eps)
        else:
            plot_RFM_1d(historical_u[k], "u^(" + str(k) + ")")



        # Test whether this computation of q and dq are accurate using finite difference method

        # Compute residuals
        fd_residual_m[k], fd_residual_u[k], fd_residual_system[k] = get_fd_residuals(historical_q[k - 1],
                                                                                     historical_m[k], historical_u[k],
                                                                                     historical_q[k], eps)

    plot_by_iter(fd_residual_m, 'FD Residual error of Fokker-Planck', 'Residual error')
    plot_by_iter(fd_residual_u, 'FD Residual error of HJB', 'Residual error')
    plot_by_iter(fd_residual_system, 'FD Residual error of System', 'Residual error')

    return historical_m[-1], historical_u[-1]


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


def test():
    set_seed(100)
    M_p_hjb = M_p_fp = 4
    J_n_hjb = J_n_fp = 50
    Q_fp = 200
    Q_hjb = 200
    n_iters = 10

    m, u = policy_iteration(M_p_hjb, J_n_hjb, M_p_fp, J_n_fp, Q_hjb, Q_fp, n_iters)

    plot_RFM_1d(m, "final m")
    plot_RFM_1d(u, "final u")


if __name__ == '__main__':
    test()
