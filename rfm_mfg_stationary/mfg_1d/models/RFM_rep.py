import torch
import numpy as np
import torch.nn as nn

from rfm_mfg_stationary.mfg_1d.utils.config import INTERVAL_LENGTH

class RFM_rep(nn.Module):
    def __init__(self, in_features, J_n, x_max, x_min):
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
