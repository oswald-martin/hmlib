"""Ausgleichsrechnung Modul.

Funcs:
    - gauss-newton verfahren
    - lineare Ausgleichsrechnung
    - Fehlerfunktional

@author: Martin Oswald
@license: MIT
@version: 1.0
"""
from typing import Callable, overload
import sympy as sp
import numpy as np



def gauss_newton_ausg(f: Callable[[float, np.ndarray], float], x: np.ndarray, y: np.ndarray, lam0: np.ndarray, tol: float, max_iter: int, pmax=5, damping=False) -> Callable[[float], float]:
    """Gedämpftes Gauss-Newton verfahren Ausgleichsrechnung.

        - Use Sympy expressions in function f (e.g sp.exp(), sp.sin())!

    Args:
        f (function): function(x, p) where p is lambda ndarray
        x (ndarray): x datapoints
        y (ndarray): y datapoints
        lam0 (ndarray): initial p vector/guess
        tol (float): error tolarance
        max_iter (int): max number of iterations
        pmax (int, optional): max damping value 2^pmax. Defaults to 5.
        damping (bool, optional): enable damping. Defaults to False.

    Returns:
        function: fittet function f(x) -> y
    """
    # Sympy
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    lam0 = np.array(lam0, dtype=np.float64)
    p = sp.symbols('p0:%d' % len(lam0))
    g = sp.Matrix([y[k]-f(x[k], p) for k in range(len(x))])
    Dg = g.jacobian(p)
    g = sp.lambdify([p], g, 'numpy')
    Dg = sp.lambdify([p], Dg, 'numpy')
    # Numpy
    k = 0
    lam = np.copy(lam0)
    increment = tol+1
    err_func = np.linalg.norm(g(lam))**2
    # Gauss-Newton
    while increment > tol and k < max_iter:
        [Q, R] = np.linalg.qr(Dg(lam))
        delta = np.linalg.solve(R, -Q.T @ g(lam)).flatten()
        p = 0
        while damping == True:
            p_error_func = np.linalg.norm(g(lam + (delta / 2**p)))**2
            if p_error_func < err_func:
                break
            else:
                p = p+1
            if p > pmax:
                p = 0
                break
        if p != 0: print(f'damping with p={p}')
        lam = lam + delta / 2**p
        err_func = np.linalg.norm(g(lam))**2
        increment = np.linalg.norm(delta / 2**p)
        k = k+1
        print(f'Iteration:  {k}')
        print(f'lambda =    {lam}')
        print(f'Inkrement = {increment}')
        print(f'Fehlerfunktional = {err_func}')
        print('')
    # create func
    t = sp.symbols('t')
    F = f(t, lam)
    F = sp.lambdify([t], F, 'numpy')
    return F

def __get_normalen_A__(f, x, lam_nr):
    f_vec = np.vectorize(f, excluded=[1])
    lam_mat = np.eye(lam_nr)
    A = np.zeros((len(x), np.max([len(np.array([x[0]]).flatten()), lam_nr])))
    for i in range(lam_nr):
        A[:,i] = f_vec(x, lam_mat[i])
    return A

@overload
def linear_ausg(f: Callable[[np.ndarray, np.ndarray], float], x: np.ndarray, y: np.ndarray, lam_nr: int):
    ...
@overload
def linear_ausg(f: Callable[[float, np.ndarray], float], x: np.ndarray, y: np.ndarray, lam_nr: int):
    """Lineare Ausgleichsrechnung. Methode der kleinsten Quadrate.

    Args:
        f (function): function(x, p) where p is lambda ndarray
        x (ndarray): x datapoints
        y (ndarray): y datapoints
        lam_nr (int): number of lambdas in the function f

    Returns:
        function: fittet function f(x) -> y
    """
    ...
def linear_ausg(f, x, y, lam_nr):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    A = __get_normalen_A__(f, x, lam_nr)
    q, r = np.linalg.qr(A)
    lamb = np.linalg.solve(r, q.T @ y)
    print(f'lambda = {lamb}')
    def F(x): return f(x, lamb)
    return F



def fehlerfunktional(f: Callable[[float], float], x: np.ndarray, y: np.ndarray):
    """Rechnet das Fehlerfunktional E einer gefitteten Ausgleichsfunktion

    Args:
        f (function): Ausgleichsfunktion f(x) -> y
        x (ndarray): x values
        y (ndarray): y values

    Returns:
        float: sum of squares
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return np.power(np.linalg.norm(y - f(x), 2), 2)



####################################################################################################
# EXAMPLE AUSGLEICHSRECHNUNG
####################################################################################################
if __name__ == '__main__':
    from plots import ausgleich_plot
    f = lambda x, p: p[0] * sp.exp(p[1] * x)
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([3, 1, 0.5, 0.2, 0.05])
    lam0 = np.array([2, 2])
    tol = 1e-5
    max_iter = 30
    pmax = 5
    damping = True
    F = gauss_newton_ausg(f, x, y, lam0, tol, max_iter, pmax, damping)
    ausgleich_plot(F, x, y).show()

    x = [1, 2, 3, 4]
    y = [6, 6.8, 10, 10.5]
    f = lambda x, p: p[0]*x + p[1]
    F = linear_ausg(f, x, y, 2)
    ausgleich_plot(F, x, y).show()