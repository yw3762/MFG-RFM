import scipy.linalg
import torch
from torch import sin, cos, pi, exp
import torch.nn as nn
from functorch import vmap
from torch.func import vmap, jacrev, jacfwd
import numpy as np
import sympy as sym
import random
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional
from scipy.linalg import pinv, solve

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
    x = x0 + P @ w * correction
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
            y1 = self.features_diff(x).matmul(w_flat)
            return y1.view(x.shape)

        return df

    def make_features_diff2_func(self, w: torch.Tensor):
        w_flat = w.detach().clone().to(torch.float64).view(-1)

        def d2f(x):
            y2 = self.features_grad2.matmul(w_flat)
            return y2.view(x.shape)

        return d2f

    def get_feature_diffs(self, w: np.ndarray) -> Tuple[Callable, Callable, Callable]:
        """
        Given weight vector w of shape [M_n*J_n], return callables (f, df, d2f)
        that accept x ([n] or [n,1]) and return same-shape outputs.
        """
        w_flat = torch.from_numpy(w).to(torch.float64).view(-1)

        def f(x):
            y = self.features(x).matmul(w_flat)
            return y.view(x.shape)

        def df(x):
            y1 = self.features_diff(x).matmul(w_flat)
            return y1.view(x.shape)

        def d2f(x):
            y2    = self.features_diff2(x).matmul(w_flat)
            return y2.view(x.shape)

        return f, df, d2f

    def plot(self, w: np.ndarray, n_pts=400, y_labal="values", title="plot"):
        w_flat = torch.from_numpy(w).to(torch.float64).view(-1)
        x = torch.linspace(0, 1, n_pts, dtype=torch.float64)
        vals = ((self.features(x)).matmul(w_flat)).detach().numpy()
        plt.figure()
        plt.plot(x.detach().numpy(), vals)
        plt.xlabel('x')
        plt.ylabel(y_labal)
        plt.title(title)
        plt.show()


def weights_init(m):
    """
    Initialize weights for the given nn.Module
    :param m: some nn.Module
    :return: None
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a=-1, b=1)
        nn.init.uniform_(m.bias, a=-1, b=1)


def rfm_vs_fd(Mu, Ju, Mm, Jm, Qu, Qm, rescale=False, test_case="cacace", n_points=1000, n_iters=20, eps=0.3, R=0.5, plot=False, final_plot=True):
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

        # Convert x_periodic to a tensor for RFM inputs
        x_periodic_t = torch.from_numpy(x_periodic).to(torch.float64)
        m_rfm_val = m_rfm(x_periodic_t).detach().numpy()
        u_rfm_val = u_rfm(x_periodic_t).detach().numpy()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x_periodic, m_periodic, label='Reference m(x)', color='black', linestyle='-', linewidth=2)
        plt.plot(x_periodic, m_rfm_val, label='RFM m(x)', color='tab:blue', linestyle='--', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('m(x)')
        plt.title('Fokker-Planck Solution')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x_periodic, u_periodic, label='Reference u(x)', color='black', linestyle='-', linewidth=2)
        plt.plot(x_periodic, u_rfm_val, label='RFM u(x)', color='tab:red', linestyle='--', linewidth=2)
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
        rhs_hjb = (Q_plus ** 2 + Q_minus ** 2) / 2 + V + F(m_vals_fd)
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

    def get_test_funcs(test_case, x):
        if test_case == "cacace":
            V_func = v_cacace
            F_func = lambda dat: dat ** 2
        elif test_case == "yang1":
            V_func = v_yang_1
            F_func = lambda dat: dat ** 4
        elif test_case == "yang2":
            V_func = v_yang_2
            F_func = lambda dat: dat ** 3
        else:
            raise ValueError("test_case must be one of 'cacace', 'yang1', or 'yang2'")
        V = V_func(x)
        return V_func, F_func, V

    # Record FD solutions and initialize policy
    historical_m_fd, historical_u_fd, historical_lam_fd = [None], [None], [None]
    historical_q_L = [np.zeros(n_points)]
    historical_q_R = [np.zeros(n_points)]

    # Record residuals
    m_residuals_fd, u_residuals_fd, system_residuals_fd = [], [], []
    x = np.linspace(0, 1, n_points, endpoint=False)
    h = x[1] - x[0]
    V_func, F_func, V = get_test_funcs(test_case, x)

    # Building the discrete Laplacian for periodic domain
    ones = np.ones(n_points)
    L, D_L, D_R = fd_operators(h, n_points)

    # Setup for RFM method
    fp_rfm, hjb_rfm = RFM_global(Mm, Jm, Qm, None), RFM_global(Mu, Ju, Qu, None)
    w_hjb = np.zeros(Mu * Ju)
    w_fp = np.zeros(Mm * Jm)

    m_rfm, dm_rfm, d2m_rfm = fp_rfm.get_feature_diffs(w_fp)
    u_rfm, du_rfm, d2u_rfm = hjb_rfm.get_feature_diffs(w_hjb)
    M_normalization = fp_rfm.features_vals.sum(dim=0).detach().numpy()
    U_normalization = hjb_rfm.features_vals.sum(dim=0).detach().numpy()
    U_normalization_aug = np.append(U_normalization, 0)

    # Policy Iteration Algorithm
    for k in range(1, n_iters + 1):
        # Step 1: Solve FP using FD method
        Q_plus = np.maximum(historical_q_L[-1], 0)
        Q_minus = np.minimum(historical_q_R[-1], 0)
        A_Q = build_A_Q(Q_plus, Q_minus, L, D_L, D_R, eps, h)
        m_vals_fd = solve_m_fd(A_Q, mu=1, threshold=1e-8, reg=0)
        historical_m_fd.append(m_vals_fd)

        # (Computing residual for FP)
        residual_m_fd = -eps * L @ m_vals_fd - D_R @ (m_vals_fd * Q_plus) - D_L @ (m_vals_fd * Q_minus)
        m_residuals_fd.append(np.sum(np.abs(residual_m_fd)) * h)
        # print(f"residual of M^{_} is", m_residuals_fd[-1])

        # Step 1': Solve FP using RFM method
        q_val_rfm = du_rfm(fp_rfm.collocs).detach()
        dq_val_rfm = d2u_rfm(fp_rfm.collocs).detach()
        div_MQ_rfm = fp_rfm.features_grad.T * q_val_rfm + fp_rfm.features_vals.T * dq_val_rfm
        LM_rfm = (-eps * fp_rfm.features_grad2 - div_MQ_rfm.T).detach().numpy()
        rescale_fp = np.eye(Mm * Qm)
        if rescale: # FIXME: It seems rescaling causes more error
            c = 100
            rescale_fp = np.diag(np.sqrt(c / (np.abs(LM_rfm)).max(axis=1)))

        w_fp_new = constrained_lsq_pinv(rescale_fp @ LM_rfm, np.zeros([Mm * Qm]), M_normalization, Mm * Qm)
        m_rfm, dm_rfm, d2m_rfm = fp_rfm.get_feature_diffs(w_fp_new)
        # fp_rfm.plot(w_fp_new, y_labal="M", title=f"RFM solution of M^{k}")

        # -------------------------------------------

        # Step 2: Solve HJB using FD method
        u_vals_fd, lam_fd = solve_u_fd(A_Q, Q_plus, Q_minus, V, F_func)
        historical_u_fd.append(u_vals_fd)
        historical_lam_fd.append(lam_fd)

        # Step 2': Solve HJB using RFM method
        Q_DU_rfm = hjb_rfm.features_grad.T.detach() * q_val_rfm
        LU_rfm = (-eps * hjb_rfm.features_grad2 + Q_DU_rfm.T).detach().numpy()

        # TODO: Make this more efficient
        # The f_hjb_rfm should be correct
        Q_square_rfm = q_val_rfm ** 2 / 2
        f_hjb_rfm = (V_func(hjb_rfm.collocs) + F_func(m_rfm(hjb_rfm.collocs)) + Q_square_rfm).detach().numpy()
        rescale_hjb = np.eye(Mu * Qu)
        if rescale: # FIXME: It seems rescaling causes more error
            c = 100
            rescale_hjb = np.diag(np.sqrt(c / (np.abs(LU_rfm)).max(axis=1)))
        LU_rfm = rescale_hjb @ LU_rfm
        LU_rfm = np.hstack((LU_rfm, np.ones((LU_rfm.shape[0], 1))))
        f_hjb_rfm = rescale_hjb @ f_hjb_rfm

        w_hjb_new = constrained_lsq_pinv(LU_rfm, rescale_hjb @ f_hjb_rfm, U_normalization_aug, 0)
        lam_rfm = w_hjb_new[-1]
        w_hjb_new = w_hjb_new[:-1]
        # hjb_rfm.plot(w_hjb_new, y_labal="U", title=f"RFM solution of U^{k}")
        u_rfm, du_rfm, d2u_rfm = hjb_rfm.get_feature_diffs(w_hjb_new)


        # -------------------------------------------
        # Plot intermediate solutions
        if plot:
            plot_fd_and_rfm(x, m_vals_fd, u_vals_fd, m_rfm, u_rfm)

        # (Computing residual for HJB)
        DLU, DRU = D_L @ u_vals_fd, D_R @ u_vals_fd
        residual_u = (-eps * L @ u_vals_fd + Q_plus * DLU + Q_minus * DRU + lam_fd * np.ones(n_points)
                      - (Q_plus ** 2 + Q_minus ** 2) / 2 - V - F_func(m_vals_fd))
        u_residuals_fd.append(np.sum(np.abs(residual_u)))
        # print(f"residual for U^{_} is", u_residuals_fd[-1])

        # Step 3: Update the FD policy
        Q_L_new, Q_R_new = normalize_policy(DLU, DRU, R)
        historical_q_L.append(Q_L_new)
        historical_q_R.append(Q_R_new)

        # Computing system residuals
        # The HJB part
        DLU_plus = np.maximum(DLU, 0)
        DRU_minus = np.minimum(DRU, 0)
        residual_hjb_sys = (-eps * L @ u_vals_fd + ((DLU_plus ** 2) + (DRU_minus ** 2)) / 2
                            + ones * lam_fd - V - F_func(m_vals_fd))

        # The FP part (correct)
        MDLU_plus = m_vals_fd * np.maximum(DLU, 0)
        MDRU_minus = m_vals_fd * np.minimum(DRU, 0)
        div_MDU_pm = D_R @ MDLU_plus + D_L @ MDRU_minus
        residual_fp_sys = -eps * L @ m_vals_fd - div_MDU_pm
        residual_sys = (np.sum(np.abs(residual_hjb_sys)) + np.sum(np.abs(residual_fp_sys))) * h
        system_residuals_fd.append(residual_sys)

        # TODO: Compute residuals for RFM





    # Plotting residuals
    if final_plot:
        plot_fd_and_rfm(x, historical_m_fd[-1], historical_u_fd[-1], m_rfm, u_rfm)
        plot_fd_residual(m_residuals_fd, u_residuals_fd, system_residuals_fd)

    print(system_residuals_fd)

    return historical_m_fd[-1], historical_u_fd[-1], historical_lam_fd[-1]


def test_forward(MMS=False):
    eps = 0.3

    def test_hjb_mms(Mu, Ju, Qu, n_points=1600):
        def plot_u_vs_true(x, u_vals, u_rfm_vals, true_u_vals, title="Comparison of u and true u"):
            h = x[1] - x[0]
            plt.figure(figsize=(10, 5))
            plt.plot(x, u_vals, label='u_vals (FD)', linestyle='--', color='blue')
            plt.plot(x, u_rfm_vals, label='u_vals (RFM)', linestyle='-.', color='red')
            plt.plot(x, true_u_vals, label='true_u_vals (Exact)', linestyle='solid', color='black')
            # Compute L1, L2 residual between numerical and true solutions
            l1_res = np.sum(np.abs(u_vals - true_u_vals)) * h
            l2_res = np.sqrt(np.sum((u_vals - true_u_vals) ** 2) * h)
            l1_res_rfm = np.sum(np.abs(u_rfm_vals - true_u_vals)) * h
            l2_res_rfm = np.sqrt(np.sum((u_rfm_vals - true_u_vals) ** 2) * h)

            # Annotate L1 and L2 on the graph
            textstr = f"L1 error = {l1_res:.4e}\nL2 error = {l2_res:.4e}\nL1 error (RFM) = {l1_res_rfm:.4e}\nL2 error (RFM) = {l2_res_rfm:.4e}"
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

        # RFM setup
        hjb_rfm = RFM_global(Mu, Ju, Qu, None)
        U_normalization = hjb_rfm.features_vals.sum(dim=0).detach().numpy()
        U_normalization_aug = np.append(U_normalization, 0)
        V_func = v_cacace

        for i in range(len(u)):
            # FD method
            fm_vals = (fm[i])(x)
            prev_u_vals = prev_u(x)
            q_L = D_L @ prev_u_vals
            q_R = D_R @ prev_u_vals
            Q_plus = np.maximum(q_L, 0)
            Q_minus = np.minimum(q_R, 0)
            A_Q_T = - eps * L + np.diag(Q_plus) @ D_L + np.diag(Q_minus) @ D_R

            rhs_hjb = (Q_plus ** 2 + Q_minus ** 2) / 2 + V_func(x) + fm_vals

            M = np.block([
                [A_Q_T, ones.reshape((-1, 1))],
                [ones.reshape((1, -1)), np.zeros((1, 1))]
            ])
            U_lam = pinv(M) @ np.append(rhs_hjb, 0)
            u_vals = U_lam[:-1]
            lam = U_lam[-1]
            print("recovered lambda (should be 1) is:", lam)


            # RFM method
            rfm_collocs = hjb_rfm.collocs.detach().numpy()
            Q_rfm = q(rfm_collocs)

            # Testing fix 2, replacing DU_rfm with Q_rfm
            Q_DU_rfm = hjb_rfm.features_grad.T.detach().numpy() * Q_rfm
            LU_rfm = (-eps * hjb_rfm.features_grad2.detach().numpy() + Q_DU_rfm.T)

            Q_square_rfm = Q_rfm ** 2 / 2
            fm_rfm_vals = (fm[i])(rfm_collocs)

            f_hjb_rfm = V_func(rfm_collocs) + fm_rfm_vals + Q_square_rfm

            # Testing fix 1:
            LU_rfm = np.hstack((LU_rfm, np.ones((LU_rfm.shape[0], 1)))) # Adding a column of 1's to the system

            rfm_total = constrained_lsq(LU_rfm, f_hjb_rfm, U_normalization_aug, 0, regularization=1e-6)
            w_hjb_new = rfm_total[:-1]
            lam_rfm = rfm_total[-1]
            print("recovered RFM lambda (should be 1) is:", lam_rfm)
            # lam_rfm = np.mean(f_hjb_rfm - LU_rfm @ w_hjb_new)

            # Plot against true values
            true_u_vals = u[i](x)
            u_rfm, du_rfm, d2u_rfm = hjb_rfm.get_feature_diffs(w_hjb_new)
            x_tensor = torch.from_numpy(x).to(torch.float64).view(-1)
            u_rfm_vals = u_rfm(x_tensor).detach().numpy()
            plot_u_vs_true(x, u_vals, u_rfm_vals, true_u_vals)
            print(lam_rfm, lam)

    def test_fp_mms(Mm, Jm, Qm, n_points=200):
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

    Mu, Mm, Mb = 5, 5, 5
    Ju, Jm, Jb = 40, 40, 30
    Il = 400
    Qu, Qm, Qb = 80, 80, 60

    if MMS:
        test_hjb_mms(Mu, Ju, Qu)
        test_fp_mms(Mm, Jm, Qm)

    m_fd, u_fd, lam_fd = rfm_vs_fd(Mu, Ju, Mm, Jm, Qu, Qm, False, "cacace", 400, n_iters=20, R=2000, plot=True, final_plot=True)
    print("Cacace lambda", lam_fd)

    # Check against Fig 2 in Yang.
    m_fd, u_fd, lam_fd = rfm_vs_fd(Mu, Ju, Mm, Jm, Qu, Qm, False, "yang1", 400, n_iters=10, eps=0.5, R=2000, plot=True, final_plot=True)
    print("Yang 5.2 lambda", lam_fd)

    # Check against Fig 3 in Yang.
    m_fd, u_fd, lam_fd = rfm_vs_fd(Mu, Ju, Mm, Jm, Qu, Qm, False, "yang2", 100, n_iters=100, R=2000, plot=True, final_plot=True)
    print("Yang 5.3.1 lambda", lam_fd)


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

    test_forward(MMS=False)