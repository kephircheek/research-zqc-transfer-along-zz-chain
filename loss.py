import numpy as np
from quanty.matrix.norm import frobenius_norm


def loss_frobenius(m):
    """Return loss function based on Frobenius norm (true)"""
    m_inf = np.zeros_like(m)
    m_inf[-1, -1] = 1
    return frobenius_norm(m - m_inf)


def trace_except_bottom_right(state):
    return np.sum(np.abs(np.diag(state)[:-1]))


def loss_frobenius_qip2022(m):
    """Return loss function based on Frobenius norm (not true)"""
    diff = m.copy()
    diff[-1, -1] -= 1
    return np.real(np.sqrt((diff.conj() @ diff).trace()))


def top_left_element(state):
    return np.abs(state[0, 0])


def min_diag_elem_in_ex2block_q3_exordered(s):
    """
    Return min element in diagonal of block corresponded to second
    excitation in ordered by excitation matrix of three qubits.
    """
    return np.min(np.diag(s[4:7, 4:7]))
