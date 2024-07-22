import numpy as np
import torch

from utils.utils_1d import RFM_rep, weights_init
from utils.config import INTERVAL_LENGTH

# fix random seed
torch.set_default_dtype(torch.float64)


"""
Settings:
d = 1,
\tau = 1e-8,
\eps = 0.3,
Bounded potential: V(x) = sin(2\pi x) + cos(4\pi x)
Hamiltonian: H(x, Du) = 1/2 |Du|^2 - V(x)
F(m)=m^2
"""



def V(x):
    """
    Analytical bounded potential function
    """
    return np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x)


def init_model(M_p,J_n,Q):
    """
    Define the local-networks and points in the corresponding regions
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in a partition
    :param Q: number of collocation points inside a partition
    :return: 1. a list of local NNs, one for each partition
             2. lists of collocation points for every partition
    """
    models = []
    points = []
    for k in range(M_p):
        # In a total of M_p partitions, we define local models to be RFM module previously defined
        x_min = INTERVAL_LENGTH / M_p * k  # interval l.b.
        x_max = INTERVAL_LENGTH / M_p * (k + 1)  # interval u.b.
        model = RFM_rep(in_features=1, J_n=J_n, x_min=x_min, x_max=x_max)  #
        model = model.apply(weights_init)  # For any linear layer in the model, initialize unif-random w/b in [-1,1]
        model = model.double()  # Make all param/buffers in double precision
        for param in model.parameters():
            param.requires_grad = False  # Freeze the layers
        models.append(model)

        # Within each partition, get the boundary pts (1d) as a column vector
        points.append(torch.tensor(np.linspace(x_min, x_max, Q + 1), requires_grad=True).reshape([-1, 1]))
    return (models, points)

