"""Integrazion Modul.

Funcs:
    - trap_tab (trapez mit tabelarischen Werten)
    - recht (rechteck)
    - simp (Simpson)
    - romb (Romberg Extrapolazion)
    - gauss (Gaussformeln für n = 1, 2, 3)
    - err_est (Fehlerabschätzung für summierte quadraturformeln)

@author: Martin Oswald
@license: MIT
@version: 1.0
"""
from typing import Callable
import numpy as np
import sympy as sp



def trap_tab(x: np.ndarray, y: np.ndarray) -> float:
    """Integratl mittels Trapez Verfahren mit tabelarischen Werten.

    Args:
        x (ndarray): x values
        y (ndarray): y values

    Returns:
        float: integral result. 
    """
    dx = x[1:] - x[:-1]
    dy = ( y[1:] + y[:-1] ) / 2
    return np.sum(dx * dy)



def trap(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Integral mittels Trapez Verfahren.

    Args:
        f (function): function f(x) -> y
        a (float): integration start x-val
        b (float): integraton end x-val
        n (int): number of segments. call h2n() to change step width to number of segments

    Returns:
        float: integral result.
    """
    inv = 1 if a < b else -1
    x = np.linspace(a, b, n+1)
    return inv * trap_tab(x, f(x))
trap = np.vectorize(trap, excluded=[0, 1, 2])



def recht(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Integral mittels Rechteck Verfahren.

    Args:
        f (function): function f(x) -> y
        a (float): integration start x-val
        b (float): integraton end x-val
        n (int): number of segments. call h2n() to change step width to number of segments

    Returns:
        float: integral result.
    """

    inv = 1 if a < b else -1
    x = np.linspace(a, b, n+1)
    h = x[1] - x[0]
    dx = x[:-1] + h/2
    return inv * np.sum(h * f(dx))



def simp(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Integral mittels Simpson Verfahren.

    Args:
        f (function): function f(x) -> y
        a (float): integration start x-val
        b (float): integraton end x-val
        n (int): number of segments. call h2n() to change step width to number of segments

    Returns:
        float: integral result.
    """
    return 1/3 * (trap(f,a,b,n) + 2*recht(f,a,b,n))



def romb(f: Callable[[float], float], a: float, b: float, m: int, print_matrix=False) -> float:
    """Integral mittels Romberg Extrapolazion.

    Args:
        f (function): function f(x) -> y
        a (float): integration start x-val
        b (float): integraton end x-val
        m (int): number romberg matrix rows
        print_matrix (bool, optional): print romberg matrix. Defaults to False

    Returns:
        float: integral result.
    """
    M = np.zeros((m+1, m+1), dtype=np.float64)
    # First Column
    M[:,0] = trap(f, a, b, np.power(2, range(0, m+1)))
    # Rest of columns recursively
    for i in range(1,m+1):
        p = np.power(4, i)
        M[:-i,i] = ( p * M[1:m+2-i,i-1] - M[:-i,i-1] ) / (p-1)
    if print_matrix: print(M)
    return M[0,-1]



def gauss(f: Callable[[float], float], a: float, b: float, n = 3):
    """Integration mittels Gaussformeln für n = 1, 2 und 3

    Args:
        f (function): function f(x) -> y
        a (float): integration start x-val
        b (float): integration end x-val
        n (_type_, optional): Gauss formel Nr. (1, 2, 3). Defaults to 3.

    Returns:
        _type_: _description_
    """
    assert n in [1, 2, 3], 'n needs to be 1, 2 or 3!'
    if n == 1:
        return (b-a) * f((b+a)/2)
    if n == 2:
        sqrt3 = np.sqrt(3)
        return (b-a)/2 * (f(-1/sqrt3 * (b-a)/2 + (b+a)/2) + f(1/sqrt3 * (b-a)/2 + (b+a)/2))
    if n == 3:
        sqrt06 = np.sqrt(0.6)
        return (b-a)/2 * (5/9 * f(-sqrt06 * (b-a)/2 + (b+a)/2) + 8/9 * f((b+a)/2) + 5/9 * f(sqrt06 * (b-a)/2 + (b+a)/2))



def err_est(f: sp.Expr, a: float, b: float, tol: float, type: str) -> float:
    """Fehlerabschätzung für summierte quadraturformeln

    Args:
        f (Sympy Expr.): Sympy function f(x)
        a (float): lower bound
        b (float): higher bound
        tol (float): error tolerance
        type (str): 'R': Rechteck, 'T': Trapez, 'S': Simpson

    Returns:
        float: Max step width. choose h smaller or equal to this. 
    """
    assert type in ['R', 'T', 'S'], "Type needs to be 'R', 'T' or 'S'"
    a, b = float(a), float(b)
    x = list(f.free_symbols)[0]
    fd = sp.diff(f, x, 4 if type == ('S' or 's') else 2)
    print(f'fd: {fd}')
    fdpos = float(sp.Float(sp.maximum(fd, x, sp.Interval(a, b))))
    fdneg = float(sp.Float(sp.maximum(-fd, x, sp.Interval(a, b))))
    fdmax = np.max([fdpos, fdneg])
    print(f'fd_max: {fdmax}')
    if type == 'R':
        return np.sqrt((tol * 24) / (fdmax * (b-a)))
    if type == 'T':
        return np.sqrt((tol * 12) / (fdmax * (b-a)))
    if type == 'S':
        return np.sqrt(np.sqrt((tol * 2880) / (fdmax * (b-a))))



####################################################################################################
# EXAMPLE INTEGRATION
####################################################################################################
if __name__ == '__main__':
    print('example integration')
    f = lambda x: 1/x
    a = 4
    b = 2
    n = 4
    print(romb(f, a, b, n))

    print('\nexample err_est')
    x = sp.symbols('x')
    f = sp.ln(x ** 2)
    a = 1
    b = 2
    tol = 10 ** -5
    print(err_est(f, a, b, tol, 'S'))