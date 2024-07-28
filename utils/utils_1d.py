import numpy as np
import torch
import torch.nn as nn
import random

from config import INTERVAL_LENGTH
from utils.config import INTERVAL_LENGTH


def set_seed(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True


def weights_init(m):
    """
    Randomly initialize parameters in the Conv2d or Linear layer.
    :param m:  the given layer
    :return:   None
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a=-1, b=1)
        nn.init.uniform_(m.bias, a=-1, b=1)


def init_local_RFM1d(J_n, x_min, x_max):
    """
    Initialize RFM network on an 1d domain, i.e. the interval [x_min, x_max].
    :param J_n:
    :param x_min:
    :param x_max:
    :return:
    """
    model = RFM_rep(in_features=1, J_n=J_n, x_min=x_min, x_max=x_max)

    # Randomly initialize parameter (uniform[-1,1]), in double precision
    model = model.apply(weights_init)
    model = model.double()

    # Freeze the randomly initialized parameters
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_differential_1d(f, points, method="center"):
    """
    Numerically compute derivatives of f(x)

    :param f: function values evaluated at each point, same shape as points. f must be periodic in the domain
    :param points: Partitions of domain, each partition contains a list of collocation points uniformly selected.
                   Each partition have boundary coinciding at a point, and domain is periodic.
    :param method: Method for getting the numerical differential.
    :return:
    """

    k = len(points)  # number of partitions
    n = len(points[0])  # number of collocation points

    df = np.zeros((k, n))

    # Check input points have coinciding boundary
    for i in range(len(points)-1):
        assert points[i][-1] == points[i+1][1]

    if method == "center":
        # f'(x) = (f(x+h) - f(x-h))/2h
        divisor = 2*points[1][0] - 2*points[0][0] # 2h

        for i in range(k):
            for j in range(1, n-1):
                df[i][j] = (f[i][j+1] - f[i][j-1]) / divisor

        # Periodicity condition
        for i in range(k):
            df[i][-1] = (f[(i + 1) % k][1] - f[i][-2]) / divisor
            df[(i + 1) % k][0] = df[i][-1]
    else:
        raise NotImplementedError("Only center method is implemented")
    return df


class RFM_rep(nn.Module):
    def __init__(self, in_features, J_n, x_max, x_min):
        super(RFM_rep, self).__init__()
        self.in_features = in_features  # num input features
        self.hidden_features = J_n      # width of hidden layer
        self.J_n = J_n                  # J_n is the number of local RF functions
        self.x_min = x_min              # x_{nj} - r_{nj}
        self.x_max = x_max              # x_{nj} + r_{nj}
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
        1. Perform a change of variable x -> \tilde{x}, i.e. y in the code
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
        print('Before passing to hidden layer, y is', y)
        y = self.hidden_layer(y)
        print('After passing to hidden layer, y is', y)

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
            return d0 * y0 + d1 * y1 + (d2 + d3) * y2 + d4 * y4
        else:
            return d0 * y0 + d1 * y1 + d2 * y2 + d3 * y3 + d4 * y4


def init_rfm(M_p, J_n, Q):
    """
    Define the RFM model on each partition and their collocation points.
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in a partition
    :param Q: number of collocation points inside a partition
    :return: 1. a list of local NNs, one for each partition
             2. a list of M_p tensors, each tensor contains collocation points, with shape (Q+1, 1).
    """
    models = []
    points = []
    for k in range(M_p):
        # Define RFM model in each partition, in 1d, partition is just an interval [x_min, x_max]
        x_min = INTERVAL_LENGTH / M_p * k
        x_max = INTERVAL_LENGTH / M_p * (k + 1)
        models.append(init_local_RFM1d(J_n, x_min, x_max))

        # Within each partition, get the boundary points (1d) as a column vector
        points.append(torch.tensor(np.linspace(x_min, x_max, Q + 1), requires_grad=True).reshape([-1, 1]))
    return models, points


def V(x):
    """
    Analytical bounded potential function

    The associated Hamiltonian is H(x, p) = 1/2 |p|^2 - V(x), where p is the variable for Dx
    """
    return np.sin(2. * np.pi * x) + np.cos(4. * np.pi * x)


def hamiltonian_1d(x, p, V):
    """
    Return the Hamiltonian 1/2 * |Du| ** 2 - V(x) for HJB on (x, p)

    Note in 1d, |Du|**2 = (du/dx)**2
    :param x: spatial variable, each element in this variable is a list of collocation points for a partition
    :param p: velocity variable, same format as x, values are derivative Du at each point in x
    :param V: bounded potential function
    :return: value of Hamiltonian on (x, p), same shape as x
    """
    assert len(x) == len(p)

    result = []
    for i in range(len(x)):
        result.append(p[i]**2 / 2 - V(x[i]))

    return result


def lagrangian_1d(x, q, v=V):
    """
    Return the Lagrangian associated with Hamiltonian H(x,Du) = 1/2 * |Du| ** 2 - V(x) for HJB on (x, q).

    By simple calculation, we see our lagrangian L(x, q) = 1/2 * |q| ** 2 + V(x)

    Note in 1d, |Du|**2 = (du/dx)**2
    :param x: spatial variable, each element in this variable is a list of collocation points for a partition
    :param p: velocity variable, same format as x, values are derivative Du at each point in x
    :param v: bounded potential function
    :return: value of Hamiltonian on (x, p), same shape as x
    """
    assert len(x) == len(q)

    result = []
    for i in range(len(x)):
        result.append(q[i] ** 2 / 2 + v(x[i]))

    return result


def evaluate_RFM_1d(models, w, points):
    """
    Given RFM models on each partition and a trained set of weights, evaluate the model on every point in each partition
    :param models: RFM models
    :param w: trained weights
    :param points: Collocation points for each partition
    :return: evaluated numerical solution
    """
    numerical_values = []
    for k in range(len(points)):
        out_total = None
        for m in range(len(models)):
            out = models[m](points[k])
            values = out.detach().numpy()
            if out_total is None:
                out_total = values
            else:
                out_total = np.concatenate((out_total, values),axis=1)

        numerical_value = np.dot(np.array(out_total), w)
        numerical_values.extend(numerical_value)
    return numerical_values
