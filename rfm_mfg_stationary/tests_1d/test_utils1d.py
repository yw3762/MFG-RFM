import unittest
import torch
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

from rfm_mfg_stationary.mfg_1d.utils.utils_1d import set_seed, init_rfm, plot_RFM_1d, second_derivative_RFM_1d
from rfm_helm1d.RFM_helm1d_PoU_psi_b_pytorch import pre_define, cal_matrix

class MyTest(unittest.TestCase):
    M_p = 5
    J_n = 50
    Q = 50
    def test_init(self):
        torch.set_default_dtype(torch.float64)
        models, points = init_rfm(self.M_p, self.J_n, self.Q, debug=True)
        assert len(models) == self.M_p
        assert len(points) == self.M_p
        for i in range(self.M_p):
            assert len(points[i]) == self.Q + 1
            if (i < self.M_p - 1):
                assert points[i][-1] == points[(i+1)][0]

        assert points[0][0] == 0
        assert points[-1][-1] == 1

    def test_derivative(self):
        torch.set_default_dtype(torch.float64)
        models, points = init_rfm(self.M_p, self.J_n, self.Q, debug=True)
        evaluation_points = [torch.linspace(0, 1, 501)]
        w = np.ones((self.M_p, self.J_n))
        q, dq = second_derivative_RFM_1d(models, w, points)
        x = torch.cat(points).detach().numpy()
        q = np.concatenate(q)
        dq = np.concatenate(dq)
        plt.figure()
        plt.plot(x, q, label="q", color='darkblue', linestyle='--')
        plt.plot(x, dq, label="dq", color='black', linestyle='-')
        plt.legend()
        plt.show()

    def test_1d_helm_derivative(self):
        models, points = pre_define(self.M_p, self.J_n, self.Q)

        # Solve PDE with RFM
        A, f = cal_matrix(models, points, self.M_p, self.J_n, self.Q)
        w = lstsq(A, f)[0]
        w = w.reshape((self.M_p, self.J_n))

        q, dq = second_derivative_RFM_1d(models, w, points)
        x_pts = torch.cat(points).detach().numpy()
        q = np.concatenate(q)
        dq = np.concatenate(dq)
        plt.figure()
        plt.title("Numerical derivative of $u(x)$ on the interval [0, 8]")
        plt.plot(x_pts, q, label="q", color='darkblue', linestyle='--')
        plt.legend()
        plt.show()

        # Analytical solution
        # Define the symbolic variable and function
        x = sp.symbols('x')
        u = sp.sin(3 * sp.pi * x + 3 * sp.pi / 20) * sp.cos(2 * sp.pi * x + sp.pi / 10) + 2

        # Compute the derivative symbolically
        du_dx = sp.diff(u, x)

        # Convert symbolic expressions to numerical functions
        f_u = sp.lambdify(x, u, 'numpy')
        f_du_dx = sp.lambdify(x, du_dx, 'numpy')

        # Define the interval [0, 1]
        x_values = np.linspace(0, 8, 400)
        u_values = f_u(x_values)
        du_dx_values = f_du_dx(x_values)

        # Plot derivative
        plt.figure()
        plt.title("Analytical derivative of $u(x)$ on the interval [0, 8]")
        plt.plot(x_values, du_dx_values, label="q", color='darkblue', linestyle='--')
        plt.legend()
        plt.show()

        # Plot analytical vs numerical solutions
        plot_RFM_1d(models, w, "Numerical Solution", interval_length=8.0)

        plt.plot(x_values, u_values, label="$u(x)$")
        plt.title("Analytical solution on the interval [0, 8]")
        plt.xlabel("x")
        plt.ylabel("$u(x)$")
        plt.legend()
        plt.show()

        # Compute second order analytical and numerical derivatives
        d2u_dx = sp.diff(du_dx, x)
        f_d2u_dx = sp.lambdify(x, d2u_dx, 'numpy')
        d2u_dx_values = f_d2u_dx(x_values)

        # Plot second order derivatives
        plt.title("Analytical second order derivative of $u(x)$ on the interval [0, 8]")
        plt.plot(x_values, d2u_dx_values, label="q", color='darkblue', linestyle='--')
        # plt.plot(x, dq, label="dq", color='black', linestyle='-')
        plt.legend()
        plt.show()

        plt.title("Numerical second order derivative of $u(x)$ on the interval [0, 8]")
        plt.plot(x_pts, dq, label="q", color='darkblue', linestyle='--')
        plt.legend()
        plt.show()




if __name__ == '__main__':
    set_seed(100)
    unittest.main()
