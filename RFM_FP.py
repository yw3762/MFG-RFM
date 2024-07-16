import numpy as np
import torch
import torch.nn as nn
import math
from scipy.linalg import lstsq,pinv
import matplotlib.pyplot as plt
import random

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




