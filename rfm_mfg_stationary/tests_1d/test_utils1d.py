import unittest
import torch
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import lstsq,pinv

from utils.utils_1d import set_seed, init_rfm, evaluate_RFM_1d, plot_RFM_1d, second_derivative_RFM_1d
from rfm_helm1d.RFM_helm1d_PoU_psi_b_pytorch import pre_define, cal_matrix

class MyTest(unittest.TestCase):
    M_p = 5
    J_n = 10
    Q = 20
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
        #plot_RFM_1d(models, w, "test function")
        q, dq = second_derivative_RFM_1d(models, w, points)
        x = torch.cat(points).detach().numpy()
        q = np.concatenate(q)
        dq = np.concatenate(dq)
        plt.figure()
        plt.plot(x, q, label="q", color='darkblue', linestyle='--')
        plt.plot(x, dq, label="dq", color='black', linestyle='-')
        plt.legend()
        plt.show()
        plt.savefig("test_derivatives.png")

    def test_1d_helm_derivative(self):
        models, points = pre_define(self.M_p, self.J_n, self.Q)

        # matrix define (Aw=b)
        A, f = cal_matrix(models, points, self.M_p, self.J_n, self.Q)
        w = lstsq(A, f)[0]
        w = w.reshape((self.M_p, self.J_n))
        plot_RFM_1d(models, w, "RFM sol for 1dhelm")

        q, dq = second_derivative_RFM_1d(models, w, points)
        x = torch.cat(points).detach().numpy()
        q = np.concatenate(q)
        dq = np.concatenate(dq)
        plt.figure()
        plt.title("Numerical derivative of $u(x)$ on the interval [0, 1]")
        plt.plot(x, q, label="q", color='darkblue', linestyle='--')
        #plt.plot(x, dq, label="dq", color='black', linestyle='-')
        plt.legend()
        plt.show()
        plt.savefig("test_derivatives.png")

        # true solution
        # Define the symbolic variable and function
        x = sp.symbols('x')
        u = sp.sin(3 * sp.pi * x + 3 * sp.pi / 20) * sp.cos(2 * sp.pi * x + sp.pi / 10) + 2

        # Compute the derivative symbolically
        du_dx = sp.diff(u, x)

        # Convert symbolic expressions to numerical functions
        f_u = sp.lambdify(x, u, 'numpy')
        f_du_dx = sp.lambdify(x, du_dx, 'numpy')

        # Define the interval [0, 1]
        x_values = np.linspace(0, 1, 400)
        u_values = f_u(x_values)
        du_dx_values = f_du_dx(x_values)

        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot u(x)
        plt.subplot(2, 1, 1)
        plt.plot(x_values, u_values, label="$u(x)$")
        plt.title("Function $u(x)$ and its Derivative on the interval [0, 1]")
        plt.xlabel("x")
        plt.ylabel("$u(x)$")
        plt.legend()
        plt.grid(True)

        # Plot derivative
        plt.subplot(2, 1, 2)
        plt.plot(x_values, du_dx_values, label="Derivative of $u(x)$", color='orange')
        plt.xlabel("x")
        plt.ylabel("Derivative")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        plt.savefig("true_derivatives.png")


if __name__ == '__main__':
    set_seed(100)
    unittest.main()
