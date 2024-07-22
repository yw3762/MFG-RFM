# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import math
from scipy.linalg import lstsq,pinv
import matplotlib.pyplot as plt
import random

# fix random seed
torch.set_default_dtype(torch.float64)
def set_seed(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True

# random initialization for parameters in FC layer
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a = -1, b = 1)
        nn.init.uniform_(m.bias, a = -1, b = 1)
        #nn.init.normal_(m.weight, mean=0, std=1)
        #nn.init.normal_(m.bias, mean=0, std=1)

# network definition
class RFM_rep(nn.Module):
    def __init__(self, in_features, J_n, x_max, x_min):
        super(RFM_rep, self).__init__()
        self.in_features = in_features # num input features
        self.hidden_features = J_n     # width of hidden layer
        self.J_n = J_n                 # J_n is the number of local RF functions
        self.x_min = x_min             # x_{nj} - r_{nj}
        self.x_max = x_max             # x_{nj} + r_{nj}
        self.a = 2.0/(x_max - x_min)   # this is the 1/r_{nj}
        self.x_0 = (x_max + x_min)/2   # center of partition.

        # Hidden layer one simple
        self.hidden_layer = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features, bias=True),
            nn.Tanh() # batch apply the above layer choice of kernel function
        )


    def forward(self,x):
        """
        The input x will be pass through the network in the following ways:
        1. Perform a change of variable x -> \tilde{x}, i.e. y in the code
        2. Pass through the hidden layer (i.e. Linear layer + tanh), i.e. we obtain J_n RF functions \phi_nj(x)
        3. Glue the solution at x using partition of unity, i.e. we obtain
        """
        d = (x - self.x_min) / (self.x_max - self.x_min)
        # indicator of location of x
        d0 = d <= -1/4
        d1 = (d <= 1/4) & (d > -1/4)
        d2 = (d <= 3/4) & (d > 1/4)
        d3 = (d <= 5/4) & (d > 3/4)
        d4 = d > 5/4

        # y is the change of variable in normalized coordinate \tilde{x}
        y = self.a * (x - self.x_0)

        # pass the normalized variable into hidden-layer
        print('Before passing to hidden layer, y is', y)
        y = self.hidden_layer(y)
        print('After passing to hidden layer, y is', y)

        # y_i are the PoU w.r.t each location of x
        y0 = 0
        y1 = y * (1 + torch.sin(2*np.pi*d)) / 2
        y2 = y
        y3 = y * (1 - torch.sin(2*np.pi*(d-1))) / 2
        y4 = 0

        # check boundary cases, on boundaries, there is no need for sin() smoothing
        if self.x_min == 0:
            return d0*y0+(d1 + d2)*y2+d3*y3+d4*y4
        elif self.x_max == interval_length:
            return d0*y0+d1*y1+(d2 + d3)*y2+d4*y4
        else:
            return d0*y0+d1*y1+d2*y2+d3*y3+d4*y4

# analytical solution parameters
AA = 1
aa = 2.0*np.pi
bb = 3.0*np.pi
interval_length = 8.
lamb = 4

def anal_u(x):
    return AA * np.sin(bb * (x + 0.05)) * np.cos(aa * (x + 0.05)) + 2.0

def anal_dudx_2nd(x):
    return -AA*(aa*aa+bb*bb)*np.sin(bb*(x+0.05))*np.cos(aa*(x+0.05))\
           -2.0*AA*aa*bb*np.cos(bb*(x+0.05))*np.sin(aa*(x+0.05))

def Lu_f(pointss, lambda_ = 4):
    r = []
    for x in pointss:
        f = anal_dudx_2nd(x) - lambda_*anal_u(x)
        r.append(f)
    return(np.array(r))


def pre_define(M_p,J_n,Q):
    """
    define the local-networks and points in the corresponding regions
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in a partition
    :param Q: number of collocation points inside a partition
    :return: 1. a list of local NNs, one for each partition
             2. a 2d list of points, each element in the outer list is a list of collocation points for a partition
    """
    models = []
    points = []
    for k in range(M_p):
        # In a total of M_p partitions, we define local models to be RFM module previously defined
        x_min = 8.0/M_p * k                 # interval l.b.
        x_max = 8.0/M_p * (k+1)             # interval u.b.
        model = RFM_rep(in_features = 1, J_n = J_n, x_min = x_min, x_max = x_max) #
        model = model.apply(weights_init)   # For any linear layer in the model, initialize unif-random w/b in [-1,1]
        model = model.double()              # Make all param/buffers in double precision
        for param in model.parameters():
            param.requires_grad = False     # Freeze the layers, i.e. fix parameters
        models.append(model)

        # Within each partition, get the boundary pts (1d) as a column vector
        points.append(torch.tensor(np.linspace(x_min, x_max, Q+1),requires_grad=True).reshape([-1,1]))
    return(models, points)


def cal_matrix(models,points,M_p,J_n,Q):
    """
    Calculate the matrix A,f in linear equations system 'Au=f'
    :param models: A list of local NNs, one for each partition
    :param points: Each element in this variable is a list of collocation points for a partition
    :param M_p: number of partitions
    :param J_n: number of RF basis functions in each partition, each RF basis function is a RFM_Rep object
    :param Q: number of collocation points inside a partition
    :return: matrix A and vector f for over-determined system
    """
    # matrix define (Aw=b)
    A_1 = np.zeros([M_p*Q,M_p*J_n])         # for PDE
    A_2 = np.zeros([2,M_p*J_n])             # for BC
    f = np.zeros([M_p*Q + 2, 1])            #
    
    for k in range(M_p): # iterate over all partitions
        # forward and grad
        for m in range(M_p): #
            # In m-th partition, evaluate each of the RFM feature function on the collocation points of k-th partition.
            out = models[m](points[k])
            values = out.detach().numpy()   # unrequire gradients, convert torch.tensor to np.array
            grads = []
            grads_2 = []
            for i in range(J_n): # within a partition, check for every basis function
                # Compute gradient of i-th basis function
                g_1 = torch.autograd.grad(outputs=out[:,i], inputs=points[k],
                                      grad_outputs=torch.ones_like(out[:,i]),
                                      create_graph = True, retain_graph = True)[0]
                # remove dims of size 1, unrequire gradients, convert to np.array
                grads.append(g_1.squeeze().detach().numpy())

                # Compute second order gradient for i-th basis function
                g_2 = torch.autograd.grad(outputs=g_1[:,0], inputs=points[k],
                                      grad_outputs=torch.ones_like(out[:,i]),
                                      create_graph = False, retain_graph = True)[0]
                grads_2.append(g_2.squeeze().detach().numpy())
            grads = np.array(grads).T
            grads_2 = np.array(grads_2).T

            Lu = grads_2 - lamb * values # PDE LHS on m-th partition, evaluated on the collocation points of k-th partition
            # Lu = f condition
            A_1[k*Q:(k + 1)*Q, m*J_n:(m + 1)*J_n] = Lu[:Q,:]
            # boundary condition
            if k == 0 and m==k:
                A_2[0, :J_n] = values[0,:]
            elif k == M_p - 1 and m==k:
                A_2[1, -J_n:] = values[-1,:]

        # get the true values for f's
        true_f = Lu_f(points[k].detach().numpy(), lamb).reshape([(Q + 1),1])
        f[k*Q:(k + 1)*Q,: ] = true_f[:Q]
    A = np.concatenate((A_1,A_2),axis=0)

    # initial conditions
    f[M_p*Q,:] = anal_u(0.)
    f[M_p*Q+1,:] = anal_u(8.)
    return(A,f)


# calculate the l^{inf}-norm and l^{2}-norm error for u,v,p
def test(models,M_p,J_n,Q,w,plot = False):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Q = int(1000/M_p)
    for k in range(M_p):
        points = torch.tensor(np.linspace(8.0/M_p * (k), 8.0/M_p * (k+1), test_Q+1),requires_grad=False).reshape([-1,1])
        out_total = None
        for m in range(M_p):
            out = models[m](points)
            values = out.detach().numpy()
            if out_total is None:
                out_total = values
            else:
                out_total = np.concatenate((out_total,values),axis=1)
        true_value = anal_u(points.numpy()).reshape([-1,1])
        numerical_value = np.dot(np.array(out_total), w)
        true_values.extend(true_value)
        numerical_values.extend(numerical_value)
        epsilon.extend(true_value - numerical_value)
    true_values = np.array(true_values)
    numerical_values = np.array(numerical_values)
    epsilon = np.array(epsilon)
    epsilon = np.maximum(epsilon, -epsilon)
    print('********************* ERROR *********************')
    print('M_p=%s,J_n=%s,Q=%s'%(M_p,J_n,Q))
    print('L_inf=',epsilon.max(),'L_2=',math.sqrt(8*sum(epsilon*epsilon)/len(epsilon)))
        
    x = [(interval_length/M_p)*i / test_Q  for i in range(M_p*(test_Q+1))]
    if plot == True:
        plt.figure()
        plt.plot(x, true_values, label = "exact solution", color='black')
        plt.plot(x, numerical_values, label = "numerical solution", color='darkblue', linestyle='--')
        plt.legend()
        plt.title('exact solution')
        #plt.savefig('./numerical_solution.pdf', dpi=100)
        
        plt.figure()
        plt.plot(x, epsilon, label = "absolute error", color='black')
        plt.legend()
        plt.title('RFM error, $\psi^2$, J_n=%s Q=%s'%(M_p*J_n,M_p*Q))
        #plt.savefig('./error_Ne=%sJ_n=%sQ=%s.pdf'%(M_p,J_n,Q), dpi=100)
    return(epsilon.max(),math.sqrt(8*sum(epsilon*epsilon)/len(epsilon)))


def main(M_p,J_n,Q,plot = False, moore = False):
    # prepare models and collocation pointss
    models, points = pre_define(M_p,J_n,Q)
    
    # matrix define (Aw=b)
    A, f = cal_matrix(models,points,M_p,J_n,Q)
    
    # solve
    if moore:
        inv_coeff_mat = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(inv_coeff_mat, f)
    else:
        w = lstsq(A,f)[0]
    
    # test
    return(test(models,M_p,J_n,Q,w,plot))



if __name__ == '__main__':
    set_seed(100)
    #M_p = 4 # the number of basis center points
    J_n = 50 # the number of basis functions per center points
    Q = 50 # the number of collocation pointss per basis functions support
    for M_p in [4,8,16]:
        main(M_p,J_n,Q,True,False)