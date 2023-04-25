import numpy as np
from quanty.matrix.norm import frobenius_norm


def loss_frobenius(m):
    """Return loss function based on Frobenius norm (true)"""
    m_inf = np.zeros_like(m)
    m_inf[-1, -1] = 1
    return frobenius_norm(m - m_inf)
