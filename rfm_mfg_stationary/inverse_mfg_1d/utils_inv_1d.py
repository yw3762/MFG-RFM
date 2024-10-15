import torch
import numpy as np


def build_RFM_matrix(models_u, q_vals, collocs_b_cat, Mb, Qb, Ju, eps):
    Mu = len(models_u)
    M = np.zeros((Mb * Qb, Mu * Ju))

    for m in range(Mu):
        out = models_u[m](collocs_b_cat)

        grads_2 = []
        q_du = []

        for i in range(Ju):
            g_1 = torch.autograd.grad(outputs=out[:, i], inputs=collocs_b_cat, grad_outputs=torch.ones_like(out[:, i]),
                                      create_graph=True, retain_graph=True)[0]
            g_2 = torch.autograd.grad(outputs=g_1[:, 0], inputs=collocs_b_cat, grad_outputs=torch.ones_like(out[:, i]),
                                          retain_graph=True)[0]
            grads_2.append(g_2.squeeze().detach().numpy())
            q_du.append((g_1.squeeze() * q_vals).detach().numpy())

        grads_2 = np.array(grads_2).T
        q_du = np.array(q_du).T

        Lu = - eps * grads_2 + q_du
        M[:, m * Ju: (m + 1) * Ju] = Lu

    return M


def concatenate_collocs(collocs_l):
    return torch.cat([t[:-1] for t in collocs_l])


def RF_valuation(models, points):
    """
    :param models: RFM models
    :param points: some 1d torch.Tensor
    :return:
    """
    return torch.cat([model(points) for model in models], dim=1)


def func_valuation(func, points):
    return func(points).view(-1, 1)


def lagrangian_sep_1d(x, q, b):
    """
    The separable Lagrangian L(q(x)) + b(x)
    :param x: the points on which we need to evaluate the lagrangian
    :param q: evaluation of q on (x) as a 1d torch tensor
    :param b: the obstruction term b(x)
    :return:
    """
    assert len(x) == len(q)

    if isinstance(q, torch.Tensor):
        if q.requires_grad:
            q.detach()
        return q ** 2 / 2 + b(x).view(-1)
    else:
        result = []
        for i in range(len(x)):
            result.append(q[i] ** 2 / 2 + b(x[i]))
        return result
