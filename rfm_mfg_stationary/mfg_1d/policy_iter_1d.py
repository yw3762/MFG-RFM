import torch
import numpy as np
from scipy.linalg import lstsq
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from rfm_mfg_stationary.mfg_1d.utils.utils_1d import plot_by_iter, init_rfm, RFM_function_factory, \
    second_diff_RFM_function, plot_RFM_1d, lagrangian_1d, get_fd_residuals, constrained_lsq, constrained_lsq_pinv


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

    # Choosing rescaling parameters
    # c = 100
    # divisor_interiors = np.zeros(M_p * Q)

    # Compute lstsq system
    # place-holder variables for A, where f is 0 by definition
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

            # Calculate rescaling divisor
            max_row = np.max(np.abs(Lu), axis=1)

            # indices = np.arange(k * Q, (k + 1) * Q + 1) % len(divisor_interiors)
            # divisor_interiors[indices] = np.maximum(divisor_interiors[indices], max_row)

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

    # Rescaling parameters
    # lambda_interiors = c / divisor_interiors.reshape((-1, 1))
    # reset
    # lambda_interiors = np.ones_like(lambda_interiors)

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
    x = np.linspace(0, 1, N, endpoint=False)
    x_tensor = torch.tensor(x).reshape(-1, 1)
    h = x[1] - x[0]

    # First and second derivative matrices with C1 periodic boundary conditions
    e = np.ones(N)
    D1 = sp.diags([-e, e], [-1, 1], shape=(N, N)) / (2 * h)
    D2 = sp.diags([e, -2 * e, e], [-1, 0, 1], shape=(N, N)) / (h ** 2)
    D1 = D1.tolil()
    D2 = D2.tolil()
    D1[0, -1] = -1 / (2 * h)
    D1[-1, 0] = 1 / (2 * h)
    D2[0, -1] = 1 / (h ** 2)
    D2[-1, 0] = 1 / (h ** 2)
    D1 = D1.tocsr()
    D2 = D2.tocsr()

    # Evaluate functions
    q = q_func(x_tensor).numpy()
    m = m_func(x_tensor).numpy()
    Fm = m ** 2
    Lq = torch.cat(L_func(x_tensor, q, b_func), dim=0).numpy()

    # RHS vector
    rhs = Fm + Lq

    # Advection operator
    Adv = sp.diags(q) @ D1

    # Operator A and constraint row for \int u dx = 0
    A = -epsilon * D2 + Adv
    constraint = np.ones((1, N)) * h
    A_aug = sp.vstack([A, constraint])
    rhs_aug = np.concatenate([rhs, [0]])

    # Solve augmented system for u only
    u = spla.lsqr(A_aug, rhs_aug)[0]

    # Compute lambda from PDE expression
    Au = A @ u
    lambda_val = np.mean(Fm + Lq - Au)

    # Compute finite difference residual
    residual = A @ u - (rhs - lambda_val)
    res_norm = np.linalg.norm(residual)
    print(f"Finite difference residual (L2 norm): {res_norm:.2e}")

    return u, lambda_val, x


def policy_iteration(Mu, Ju, Mm, Jm, Qu, Qm, b=None, n_iters=20, eps=0.3, tau=1e-6, plot=False):
    # fix datatype
    torch.set_default_dtype(torch.float64)
    n_test = 200

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
        if plot:
            plot_RFM_1d(curr_u, "u^(" + str(k) + ")", n_pts=200, test_values=test_u)


        # Step (3): Update the policy
        new_q, dq = second_diff_RFM_function(historical_u[k])
        historical_q.append(new_q)

        # Compute residuals
        fd_residual_m[k], fd_residual_u[k], fd_system_residual[k] = get_fd_residuals(historical_q[k - 1],
                                                                                     historical_m[k], historical_u[k],
                                                                                     historical_q[k], curr_lam, eps)

    if plot:
        plot_by_iter(fd_residual_m, 'FD Residual error of Fokker-Planck', 'Residual error')
        plot_by_iter(fd_residual_u, 'FD Residual error of HJB', 'Residual error')
        plot_by_iter(fd_system_residual, 'FD Residual error of System', 'Residual error')

    print("Final residual of Fokker-Planck is", fd_residual_m[-1])
    print("Final residual of HJB is", fd_residual_u[-1])

    return historical_m[-1], historical_u[-1], historical_lam[-1], fd_system_residual
