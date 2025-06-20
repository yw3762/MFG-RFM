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


def policy_iteration_fd(n_points=1000, n_iters=20, eps=0.3, R=0.5, plot=False):
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

    def solve_m(A_Q, mu=1000, threshold=1e-6, reg=1e-3):
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

    def solve_u(A_Q, Q_plus, Q_minus, method="direct_inverse"):
        rhs_hjb = (Q_plus ** 2 + Q_minus ** 2) / 2 + v_cacace(x) + m_vals ** 2
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
        A_Q = build_A_Q(Q_plus, Q_minus, L, D_L, D_R, eps, h)
        m_vals = solve_m(A_Q, mu=1, threshold=1e-8, reg=0)
        historical_m.append(m_vals)

        # (Computing residual for FP)
        residual_m = -eps * L @ m_vals - D_R @ (m_vals * Q_plus) - D_L @ (m_vals * Q_minus)
        m_residuals.append(np.sum(np.abs(residual_m)) * h)
        print(f"residual of M^{_} is", m_residuals[-1])
        # print("total mass of M with iterative computation", np.average(np.abs(m_vals)))

        # Step 2: Solve HJB
        u_vals, lam = solve_u(A_Q, Q_plus, Q_minus)
        historical_u.append(u_vals)
        historical_lam.append(lam)

        # (Computing residual for HJB)
        V = v_cacace(x)
        DLU = D_L @ u_vals
        DRU = D_R @ u_vals
        residual_u = (-eps * L @ u_vals + Q_plus * DLU  + Q_minus * DRU + lam * np.ones(n_points)
                      - (Q_plus ** 2 + Q_minus ** 2) / 2 - V - m_vals**2)
        u_residuals.append(np.sum(np.abs(residual_u)))
        print(f"residual for U^{_} is", u_residuals[-1])

        # Step 3: Update the policy
        Q_L_new, Q_R_new = normalize_policy(DLU, DRU, R)
        historical_q_L.append(Q_L_new)
        historical_q_R.append(Q_R_new)

        # Testing system residual
        # The HJB part
        DLU_plus = np.maximum(DLU, 0)
        DRU_minus = np.minimum(DRU, 0)
        residual_hjb_sys = (-eps * L @ u_vals + ((DLU_plus ** 2) + (DRU_minus ** 2)) / 2
                            + ones * lam - V - m_vals**2) # F(x) = x**2

        # The FP part (correct)
        MDLU_plus = m_vals * np.maximum(DLU, 0)
        MDRU_minus = m_vals * np.minimum(DRU, 0)
        div_MDU_pm = D_R @ MDLU_plus + D_L @ MDRU_minus
        residual_fp_sys = -eps * L @ m_vals - div_MDU_pm
        residual_sys = (np.sum(np.abs(residual_hjb_sys)) + np.sum(np.abs(residual_fp_sys))) * h
        system_residuals.append(residual_sys)


        # Plot intermediate solutions
        if plot:
            plot_fd_system(x, m_vals, u_vals)

    # Plotting residuals
    if plot:
        plot_fd_residual(m_residuals, u_residuals, system_residuals)

    print(system_residuals)

    return historical_m[-1], historical_u[-1], historical_lam[-1]


def test():
    eps = 0.3

    def test_hjb_mms(n_points=1600):
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


    def test_fp_mms(n_points=200):
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

    m_fd, u_fd, lam_fd = policy_iteration_fd(200, n_iters=20, R=2000, plot=True)


if __name__ == '__main__':
    test()