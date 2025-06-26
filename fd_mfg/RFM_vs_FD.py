import scipy.linalg
import torch
import pandas as pd
import scipy.sparse.linalg as spla
from torch import sin, cos, pi, exp
import torch.nn as nn
from functorch import make_functional_with_buffers, vmap
from torch.func import functional_call, vmap, jacrev, jacfwd
import numpy as np
import sympy as sym
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import random
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Any, Union, Optional
from scipy.linalg import pinv, solve, lstsq


INTERVAL_LENGTH = 1.0  # we consider the domain to be the unit torus.


def v_cacace(x):
    """
    The given bounded potential function as in Cacace et al's example.
    """
    if isinstance(x, torch.Tensor):
        return sin(2. * pi * x) + cos(4. * pi * x)
    else:
        return np.sin(2. * np.pi * x) + np.cos(4. * np.pi * x)


def v_yang_1(x):
    """
    The given bounded potential function as in Yang et al's example in section 5.2,
    F(m) = m^4
    """
    if isinstance(x, torch.Tensor):
        return 2 * (sin(pi * x) + cos(5. * pi * x))
    else:
        return 2 * (np.sin(np.pi * x) + np.cos(5. * np.pi * x))


def v_yang_2(x):
    """
    The given bounded potential function as in Yang et al's example in section 5.3.1,
    F(m) = m^3
    """
    if isinstance(x, torch.Tensor):
        return .5 * (sin(2. * pi * x) + cos(4. * pi * x))
    else:
        return .5 * (np.sin(2. * np.pi * x) + np.cos(4. * np.pi * x))


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


def lagrangian_1d(x, q, v):
    assert len(x) == len(q)

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

class RFM_rep(nn.Module):
    def __init__(self, in_features, J_n, x_min, x_max):
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


class RFM_global(nn.Module):
    """
    The global RFM module for periodic 1D domain [left, right]
    """
    def __init__(self, M_n:int, J_n:int, Q:int, partitions:Optional[List[Tuple[float, float]]], in_features=1, left=0., right=1.):
        super(RFM_global, self).__init__()
        if partitions is None:
            edges = torch.linspace(left, right, steps=M_n + 1).tolist()
            # create uniform partition [(0, 1/M_n), (1/M_n, 2/M_n), ..., ((M_n−1)/M_n,1)]
            partitions = [(edges[i], edges[i + 1]) for i in range(M_n)]
        else:
            assert len(partitions) == M_n and partitions[0][0] == left and partitions[-1][1] == right

        meshes = [
            torch.linspace(x_min, x_max, steps=Q, dtype=torch.float64)
            for (x_min, x_max) in partitions
        ]
        self.collocs = torch.cat(meshes, dim=0)

        self.local_rfms = nn.ModuleList([
            RFM_rep(in_features, J_n, x_min, x_max) for (x_min, x_max) in partitions
        ])

        for rfm in self.local_rfms:
            rfm.apply(weights_init)  # apply uniform[-1,1] init
            rfm.double()  # in double precision
            for p in rfm.parameters():
                p.requires_grad = False  # freeze parameters

        self.features_vals = self.features(self.collocs)
        self.features_grad = self.features_diff(self.collocs)
        self.features_grad2 = self.features_diff2(self.collocs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [rfm(x) for rfm in self.local_rfms]
        return torch.cat(outs, dim=1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature evaluations for x.
        """
        if x.ndim == 1:
            x_in = x.unsqueeze(1)
        else:
            x_in = x
        return self.forward(x_in)  # [n, F]

    def features_diff(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute first derivative of features for x.
        """
        if x.ndim == 1:
            x_in = x.unsqueeze(1)
        else:
            x_in = x
        # Flatten to [n]
        x_flat = x_in.squeeze(-1)

        # Per-sample derivative function
        def _deriv_fn(x_scalar):
            # x_scalar: scalar tensor
            out = self.forward(x_scalar.unsqueeze(0).unsqueeze(-1))  # shape [1, F]
            return out.squeeze(0)  # shape [F]
        grad1 = vmap(jacrev(_deriv_fn))(x_flat)
        return grad1

    def features_diff2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute second derivative of features for x.
        """
        if x.ndim == 1:
            x_in = x.unsqueeze(1)
        else:
            x_in = x
            # flatten to [n]
        x_flat = x_in.squeeze(-1)

        # scalar-to-F-vector function
        def _f(x_scalar):
            # x_scalar is shape [], we make it [1,1] for forward()
            out = self.forward(x_scalar.unsqueeze(0).unsqueeze(-1))  # [1, F]
            return out.squeeze(0)  # [F]

        # Mixed-mode Hessian: first reverse, then forward
        hess_fn = jacfwd(jacrev(_f))  # each call is O(F)
        # vectorize over all n points
        grad2 = vmap(hess_fn)(x_flat)  # [n, F]
        return grad2

    def make_features_func(self, w: torch.Tensor):
        w_flat = w.detach().clone().to(torch.float64).view(-1)

        def f(x):
            y = self.features_vals.matmul(w_flat)
            return y.view(x.shape)

        return f

    def make_features_diff_func(self, w: torch.Tensor):
        w_flat = w.detach().clone().to(torch.float64).view(-1)

        def df(x):
            y1 = self.features_grad.matmul(w_flat)
            return y1.view(x.shape)

        return df

    def make_features_diff2_func(self, w: torch.Tensor):
        w_flat = w.detach().clone().to(torch.float64).view(-1)

        def d2f(x):
            y2 = self.features_grad2.matmul(w_flat)
            return y2.view(x.shape)

        return d2f

    def make_funcs(self, w: torch.Tensor) -> Tuple[Callable, Callable, Callable]:
        """
        Given weight vector w of shape [M_n*J_n], return callables (f, df, d2f)
        that accept x ([n] or [n,1]) and return same-shape outputs.
        """
        w_flat = w.detach().clone().to(torch.float64).view(-1)

        def f(x):
            y = self.features_vals.matmul(w_flat)
            return y.view(x.shape)

        def df(x):
            y1 = self.features_grad.matmul(w_flat)
            return y1.view(x.shape)

        def d2f(x):
            y2    = self.features_grad2.matmul(w_flat)
            return y2.view(x.shape)

        return f, df, d2f


def weights_init(m):
    """
    Initialize weights for the given nn.Module
    :param m: some nn.Module
    :return: None
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a=-1, b=1)
        nn.init.uniform_(m.bias, a=-1, b=1)


def RFM_global_factory(rfm_global: torch.nn.Module, w: Union[torch.Tensor, np.ndarray]) \
        -> Callable[[Union[torch.Tensor,np.ndarray]], Union[torch.Tensor,np.ndarray]]:
    if not torch.is_tensor(w):
        w_tensor = torch.tensor(w, dtype=torch.float64)
    else:
        w_tensor = w.detach().clone().to(torch.float64)
    w_tensor = w_tensor.view(-1)

    def RFM_global_function(x: Union[torch.Tensor, np.ndarray]):
        is_numpy = isinstance(x, np.ndarray)
        # bring x into torch with shape [batch, 1]
        if is_numpy:
            x_t = torch.from_numpy(x).to(torch.float64)
        else:
            x_t = x.clone().detach().to(torch.float64)
        orig_shape = x_t.shape
        # ensure a 2D column for RFM_global
        if x_t.ndim == 1:
            x_in = x_t.unsqueeze(1)  # [n] → [n,1]
        else:
            x_in = x_t

        # compute features: [batch, M_n*J_n]
        feats = rfm_global(x_in)

        # weighted sum → shape [batch]
        y = feats.matmul(w_tensor)  # [batch]

        # reshape back to original shape
        y = y.view(orig_shape)

        # convert back to numpy if needed
        return y.numpy() if is_numpy else y

    return RFM_global_function



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


def solve_HJB_forward(models, collocs, m_func, q_func, M_p, J_n, Q, L_func, V_func, F_func, eps=0.3):
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

        Lq = L_func(collocs[k], q[k], V_func)
        Fm = F_func(m_func(collocs[k]).view(-1))
        summed = Fm + Lq
        trimmed = summed[:Q].detach().numpy().reshape(-1, 1)
        f[k * Q:(k + 1) * Q, :] = trimmed

    A = A_pde
    f = f

    # Solve lstsq system
    dim = A.shape[0]
    proj = np.eye(dim) - np.ones((dim, dim)) / dim

    w = constrained_lsq(proj @ A, proj @ f, A_constraints.reshape(-1), 0)
    lam = np.mean(f - A @ w)
    w = torch.tensor(w.reshape((M_p, J_n)))

    solution = RFM_function_factory(models, w)
    return solution, lam


def fd_residuals_for_rfm(prev_q, curr_m, curr_u, curr_q, curr_lam, eps, plot=False):
    def fd_laplacian(vals, h):
        rolled_left = torch.roll(vals, -1)
        rolled_right = torch.roll(vals, 1)
        diff = rolled_left - 2 * vals + rolled_right
        return diff / (h ** 2)

    def fd_derivative(vals, h):
        return (torch.roll(vals, -1) - torch.roll(vals, 1)) / (2 * h)

    def plot_residuals(pde_residual, system_residual, pts, label):
        plt.plot(pts.view(-1).numpy(), pde_residual.numpy(), label="Single PDE FD residual for " + label)
        plt.plot(pts.view(-1).numpy(), system_residual.numpy(), label="System FD residual for " + label, linestyle='--')
        plt.xlabel('pts')
        plt.ylabel('FD error')
        plt.legend()
        plt.show()


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


def rfm_global(Mu, Ju, Mm, Jm, Qu, Qm, test_case="cacace", n_iters=20, eps=0.3, R=0.5, plot=False, final_plot=True):
    fp, hjb = RFM_global(Mm, Jm, Qm, None), RFM_global(Mu, Ju, Qu, None)
    w_hjb = torch.zeros(Mu * Ju)
    w_fp = torch.zeros(Mm * Jm)

    m, dm, d2m = fp.make_funcs(w_fp)
    u, du, d2u = hjb.make_funcs(w_hjb)

    for k in range(1, n_iters + 1):
        # Step 1: Solve FP
        q_val = du(fp.collocs).detach()
        dq_val = d2u(fp.collocs).detach()

        div = fp.features_grad * q_val + fp.features_vals * dq_val
        LM = -eps * fp.features_grad2 - div



        # Step 2: Solve HJB





def rfm_vs_fd(Mu, Ju, Mm, Jm, Qu, Qm, test_case="cacace", n_points=1000, n_iters=20, eps=0.3, R=0.5, plot=False, final_plot=True):
    """
    Compare RFM and FD within each iteration.
    :param test_case: Should be either "cacace" or "yang1" or "yang2"
    :param n_points: Number of grid points along each dimension in the domain
    :param n_iters: Number of iterations to perform
    :param eps: The viscosity constant in the MFG
    :param R: Radius to trigger a policy normalization
    :param plot: Whether to plot the intermediate result
    :param final_plot: Whether to save the final result
    :return:
    """
    torch.set_default_dtype(torch.float64)

    def plot_fd_and_rfm(x, m_vals, u_vals, m_rfm, u_rfm):
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

    def normalize_policy(q_L, q_R, R):
        q_L_new = q_L
        q_R_new = q_R
        DU_norm = np.sum(np.abs(q_L_new)) + np.sum(np.abs(q_R_new))
        if DU_norm > R:
            q_L_new = q_L_new * R / DU_norm
            q_R_new = q_R_new * R / DU_norm
        return q_L_new, q_R_new

    def build_A_Q(Q_plus, Q_minus, L, D_L, D_R, eps, h):
        A_Q_T = - eps * L + np.diag(Q_plus) @ D_L + np.diag(Q_minus) @ D_R
        # Check the A_Q_T (and hence A_Q) is correctly defined
        for i in range(n_points):
            diff_mid = A_Q_T[i, i] - 2 * eps / h**2 - Q_plus[i] / h + Q_minus[i] / h
            if abs(diff_mid) > 1e-10:
                print("middle entry differs", diff_mid)
            diff_left = A_Q_T[i, (n_points + i-1) % n_points] + eps / h**2 + Q_plus[i] / h
            if abs(diff_left) > 1e-10:
                print("middle entry differs", diff_left)
            diff_right = A_Q_T[i, (n_points + i+1) % n_points] + eps / h**2 - Q_minus[i] / h
            if abs(diff_right) > 1e-10:
                print("middle entry differs", diff_right)
        return A_Q_T.T

    def solve_m_fd(A_Q, mu=1000, threshold=1e-6, reg=0):
        """
        Solve the problem [µ I + A(Q)] M = µ M for M by iteratively solving
            [µ I + A(Q)] W^{s+1} = µ W^s
        until {W^s} converges to some M s.t. \int M = 1 and M >= 0
        :param A_Q: A(Q) in our problem, possibly singular
        :param mu: µ so that µ I + A(Q) is non-singular
        :param threshold: The convergence criterion for the iterative problem
        :return: resulting solution M
        """
        W = np.ones_like(x)  # initial choice of W
        N = pinv((mu + reg) * np.eye(n_points) + A_Q)
        W_next = mu * N @ W
        while np.linalg.norm(W - W_next) > threshold:
            W = W_next
            W_next = mu * N @ W
        return W_next

    def solve_u_fd(A_Q, Q_plus, Q_minus, V, F, method="direct_inverse"):
        rhs_hjb = (Q_plus ** 2 + Q_minus ** 2) / 2 + V + F(m_vals)
        if method == "direct_inverse":
            # Try using the direct inverse method
            M = np.block([
                [A_Q.T, ones.reshape((-1, 1))],
                [ones.reshape((1, -1)), np.zeros((1, 1))]
            ])
            U_lam = pinv(M) @ np.append(rhs_hjb, 0)
            u_vals = U_lam[:-1]
            lam = U_lam[-1]
        elif method == "kernel":
            # Compute the spanning vector of ker(A(Q))
            w = scipy.linalg.null_space(A_Q).flatten()
            lam = np.dot(w, rhs_hjb) / np.dot(w, ones)
            u_vals = pinv(A_Q.T) @ (rhs_hjb - lam)
        elif method == "diff":
            A_Q_T_pinv = pinv(A_Q.T)
            P = A_Q.T @ A_Q_T_pinv
            diff = (rhs_hjb - P @ rhs_hjb)
            lam_first = diff[0]
            lam_avg = np.average(diff)
            lam = lam_avg
            u_vals = A_Q_T_pinv @ (rhs_hjb - lam)
        else:
            raise ValueError("method must be one of 'direct_inverse', 'kernel', or 'diff'")
        # print("total mass of U", np.average(u_vals))
        return u_vals, lam

    # Initialize RFMs for FP and HJB with zero weights
    models_fp, collocs_fp = init_rfm(Mm, Jm, Qm)
    models_hjb, collocs_hjb = init_rfm(Mu, Ju, Qu)


    w_hjb = torch.zeros((Mu, Ju))
    w_fp = torch.zeros((Mm, Jm))
    historical_m_rfm, historical_u_rfm, historical_lam_rfm = [RFM_function_factory(models_fp, w_fp)], [RFM_function_factory(models_hjb, w_hjb)], [0]

    historical_q_rfm = [None]
    historical_q_rfm[0], dq = second_diff_RFM_function(historical_u_rfm[0])

    # Record FD solutions and initialize policy
    historical_m_fd, historical_u_fd, historical_lam_fd = [None], [None], [None]
    historical_q_L = [np.zeros(n_points)]
    historical_q_R = [np.zeros(n_points)]

    # Record residuals
    m_residuals_rfm, u_residuals_rfm, fd_system_residuals_rfm, system_residuals_rfm = np.zeros(n_iters + 1), np.zeros(n_iters + 1), np.zeros(n_iters + 1), np.zeros(n_iters + 1)
    m_residuals_fd, u_residuals_fd, system_residuals_fd = [], [], []

    x = np.linspace(0, 1, n_points, endpoint=False)
    h = x[1] - x[0]

    if test_case == "cacace":
        V = v_cacace(x)
        V_func = v_cacace
        F_func = lambda dat: dat ** 2
    elif test_case == "yang1":
        V = v_yang_1(x)
        V_func = v_yang_1
        F_func = lambda dat: dat ** 4
    elif test_case == "yang2":
        V = v_yang_2(x)
        V_func = v_yang_2
        F_func = lambda dat: dat ** 3
    else:
        raise ValueError("test_case must be one of 'cacace', 'yang1', or 'yang2'")

    # Building the discrete Laplacian for periodic domain
    ones = np.ones(n_points)
    L, D_L, D_R = fd_operators(h, n_points)

    # Policy Iteration Algorithm
    for k in range(1, n_iters + 1):
        # Step 1: Solve FP using FD method
        Q_plus = np.maximum(historical_q_L[-1], 0)
        Q_minus = np.minimum(historical_q_R[-1], 0)
        A_Q = build_A_Q(Q_plus, Q_minus, L, D_L, D_R, eps, h)
        m_vals = solve_m_fd(A_Q, mu=1, threshold=1e-8, reg=0)
        historical_m_fd.append(m_vals)

        # Step 1': Solve FP using RFM method
        m_rfm = solve_FP_forward(models_fp, collocs_fp, historical_q_rfm[k - 1], dq, Mm, Jm, Qm)
        historical_m_rfm.append(m_rfm)

        # Step 2: Solve HJB using FD method
        u_vals, lam = solve_u_fd(A_Q, Q_plus, Q_minus, V, F_func)
        historical_u_fd.append(u_vals)
        historical_lam_fd.append(lam)

        # Step 2': Solve HJB using RFM method
        u_rfm, lam_rfm = solve_HJB_forward(models_hjb, collocs_hjb, historical_m_rfm[k], historical_q_rfm[k - 1], Mu,
                                             Ju, Qu, lagrangian_1d, V_func, F_func)
        historical_u_rfm.append(u_rfm)
        historical_lam_rfm.append(lam_rfm)

        # Step 3: Update the FD policy
        Q_L_new, Q_R_new = normalize_policy(DLU, DRU, R)
        historical_q_L.append(Q_L_new)
        historical_q_R.append(Q_R_new)

        # Step 3': Update the RFM policy
        new_q, dq = second_diff_RFM_function(historical_u_rfm[k])
        historical_q_rfm.append(new_q)

        # (Computing residual for FP)
        residual_m = -eps * L @ m_vals - D_R @ (m_vals * Q_plus) - D_L @ (m_vals * Q_minus)
        m_residuals_fd.append(np.sum(np.abs(residual_m)) * h)
        # print(f"residual of M^{_} is", m_residuals_fd[-1])

        # (Computing residual for HJB)
        DLU, DRU = D_L @ u_vals, D_R @ u_vals
        residual_u = (-eps * L @ u_vals + Q_plus * DLU + Q_minus * DRU + lam * np.ones(n_points)
                      - (Q_plus ** 2 + Q_minus ** 2) / 2 - V - F_func(m_vals))
        u_residuals_fd.append(np.sum(np.abs(residual_u)))
        # print(f"residual for U^{_} is", u_residuals_fd[-1])

        # Computing system residuals
        # The HJB part
        DLU_plus = np.maximum(DLU, 0)
        DRU_minus = np.minimum(DRU, 0)
        residual_hjb_sys = (-eps * L @ u_vals + ((DLU_plus ** 2) + (DRU_minus ** 2)) / 2
                            + ones * lam - V - F_func(m_vals))

        # The FP part (correct)
        MDLU_plus = m_vals * np.maximum(DLU, 0)
        MDRU_minus = m_vals * np.minimum(DRU, 0)
        div_MDU_pm = D_R @ MDLU_plus + D_L @ MDRU_minus
        residual_fp_sys = -eps * L @ m_vals - div_MDU_pm
        residual_sys = (np.sum(np.abs(residual_hjb_sys)) + np.sum(np.abs(residual_fp_sys))) * h
        system_residuals_fd.append(residual_sys)

        # Compute residuals
        m_residuals_rfm[k], u_residuals_rfm[k], fd_system_residuals_rfm[k], system_residuals_rfm[k] = fd_residuals_for_rfm(
            historical_q_rfm[k - 1],
            historical_m_rfm[k], historical_u_rfm[k],
            historical_q_rfm[k], lam_rfm, eps)

        # Plot intermediate solutions
        if plot:
            plot_fd_and_rfm(x, m_vals, u_vals, m_rfm, u_rfm)

    # Plotting residuals
    if final_plot:
        plot_fd_and_rfm(x, historical_m_fd[-1], historical_u_fd[-1], historical_m_rfm[-1], historical_u_rfm[-1])
        plot_fd_residual(m_residuals_fd, u_residuals_fd, system_residuals_fd)

    print(system_residuals_fd)

    return historical_m_fd[-1], historical_u_fd[-1], historical_lam_fd[-1]


def test(MMS=False):
    eps = 0.3

    def test_hjb_mms_fd(n_points=1600):
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

            M = np.block([
                [A_Q_T, ones.reshape((-1, 1))],
                [ones.reshape((1, -1)), np.zeros((1, 1))]
            ])
            U_lam = pinv(M) @ np.append(rhs_hjb, 0)
            u_vals = U_lam[:-1]
            lam = U_lam[-1]
            print("recovered lambda (should be 1) is:", lam)

            true_u_vals = u[i](x)
            plot_u_vs_true(x, u_vals, true_u_vals)


    def test_fp_mms_fd(n_points=200):
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


    if MMS:
        test_hjb_mms_fd()
        test_fp_mms_fd()


    Mu, Mm, Mb = 8, 4, 5
    Ju, Jm, Jb = 40, 20, 30
    Il = 400
    Qu, Qm, Qb = 80, 40, 60

    # rfm_global(Mu, Ju, Mm, Jm, Qu, Qm)

    m_fd, u_fd, lam_fd = rfm_vs_fd(Mu, Ju, Mm, Jm, Qu, Qm, "cacace", 200, n_iters=20, R=2000, plot=False, final_plot=True)
    # print("Cacace lambda", lam_fd)

    # Check against Fig 2 in Yang.
    # m_fd, u_fd, lam_fd = rfm_vs_fd("yang1", 100, n_iters=30, eps=0.5, R=2000, plot=False, final_plot=True)
    # print(lam_fd)
    # print("Yang 5.2 lambda", lam_fd)

    # Check against Fig 3 in Yang.
    # m_fd, u_fd, lam_fd = rfm_vs_fd("yang2", 100, n_iters=100, R=2000, plot=False, final_plot=True)
    # print("Yang 5.3.1 lambda", lam_fd)


if __name__ == '__main__':
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
    set_seed(100)
    test()