
"""
Settings:
d = 1,
\tau = 1e-8,
\eps = 0.3,
Bounded potential: V(x) = sin(2\pi x) + cos(4\pi x)
Hamiltonian: H(x, Du) = 1/2 |Du|^2 - V(x)
F(m)=m^2
"""
import numpy as np


def V(x):
    """
    Analytical bounded potential function
    :param x:
    :return:
    """
    return np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x)

def policy_iteration(n_iter=25):

    # initialize policy q
    for _ in range(n_iter):



