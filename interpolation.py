"""Interpolazion Modul.

Funcs:
    - lagrange
    - err_est (Fehlerabschätzung)
    - nat_spline (Natürliche Kubische Spline)

@author: Martin Oswald
@license: MIT
@version: 1.0
"""
from typing import Callable
import numpy as np
import sympy as sp
import math


def lagrange(x: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    """Lagrange interpolation. suiteble for small datasets <20

    Args:
        x (ndarray): x values
        y (ndarray): y values

    Returns:
        function: interpolated function f(x_eval) -> y
    """
    #assert
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    assert np.shape(x) == np.shape(y), 'x and y need to be the same size'

    def interpol(x_eval):
        li = np.ones_like(x, dtype=np.float64)
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j:
                    li[i] = li[i] * (x_eval - x[j])/(x[i] - x[j])
        return y@li    
    return np.vectorize(interpol)



def err_est(f: Callable[[float], float], f_ex: sp.Expr, x: np.ndarray):
    """Fehlerabschätzung für die Lagrange Interpolation.

        Damit die Fehlerabschätzung funktioniert muss f_ex genügend oft stetig differenzierbar sein!

    Args:
        f (function): interpolated function
        f_ex (sp.Expr): exact function
        x (np.ndarray): x values used for interpolation
    Returns:
        function: max error f(x_eval) -> y
    """
    x = np.array(x, dtype=np.float64)
    a = x.min()
    b = x.max()
    z = list(f.free_symbols)[0]
    fd = sp.diff(f, z, len(x))
    print(f'fdiff: {fd}')
    fdpos = float(sp.Float(sp.maximum(fd, z, sp.Interval(a, b))))
    fdneg = float(sp.Float(sp.maximum(-fd, z, sp.Interval(a, b))))
    fdmax = np.max([fdpos, fdneg])
    print(f'fdmax: {fdmax}')
    def err(xx: float):
        return np.abs(np.prod(np.full_like(x, xx) - x)) / math.factorial(len(x)) * fdmax
    return err




def nat_spline(x: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    """Natural cubic spline interpolation. suiteble for medium datasets <50


        - This is the natural cubic spline!
        - For 'not-a-knot', 'periodic', 'clamped' use scipy.interpolate.CubicSpline(x, y, axis=0, bc_type='not-a-knot', extrapolate=None)

    Args:
        x (ndarray): x values
        y (ndarray): y values

    Returns:
        function: interpolated function f(x_eval) -> y
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    #assert
    assert np.shape(x) == np.shape(y), 'x and y need to have the same shape'
    n = len(x)
    S = np.zeros((n-1, 4))
    #1  a_i = y_i
    S[:,0] = y[:-1]
    #2  h_i = x_i+1 - x_i
    h = np.diff(x)
    #3  c_0, c_n = 0
    c = np.zeros(n)
    #4
    if n >= 3:
        # Create matrix A using fill_diagonal
        A = np.zeros((n-2,n-2))
        np.fill_diagonal(A, 2*(h[:-1] + h[1:]))     # diagonal
        np.fill_diagonal(A[:-1, 1:], h[1:-1])       # upper diagonal
        np.fill_diagonal(A[1:, :-1], h[1:-1])       # lower diagonal
        # step 4 - solve linear systems for c 
        z = 3*((y[2:]-y[1:-1])/h[1:] - (y[1:-1]-y[:-2])/h[:-1])
        c[1:-1] = np.linalg.solve(A,z).reshape((n-2))
    S[:,2] = c[:-1]
    #5 b_i
    S[:,1] = (np.diff(y) / h) - (h / 3 * (2*c[:-1] + c[1:]))
    #6 d_i
    S[:,3] = 1/(3*h) * np.diff(c)
    # Print coeffs
    print(f'Spline Coefficients [ai, bi, ci, di]:\n{S}\n')
    # define evaluation function that calculates interpolated y
    def interpol(x_eval):
        assert np.logical_and(x_eval >= x.min(), x_eval <= x.max()), 'x_eval needs to be in x range'
        idx = np.flatnonzero(x_eval >= x[:-1])[-1]
        x_i = np.tile(x_eval - x[idx], 4)
        v = np.power(x_i, [0,1,2,3])
        return S[idx]@v
    return np.vectorize(interpol)



####################################################################################################
# EXAMPLE INTERPOLATE
####################################################################################################
if __name__ == '__main__':
    import plots
    x = np.array([8, 10, 12, 14], dtype=np.float64)
    y = np.array([11.2, 13.4, 15.3, 19.5], dtype=np.float64)
    f_interpol = nat_spline(x, y)
    plots.ausgleich_plot(f_interpol, x, y).show()
