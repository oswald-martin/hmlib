"""Newton Modul. Nullstellenbestimmung

Funcs:
    - newton

@author: Martin Oswald
@license: MIT
@version: 1.0
"""
import numpy as np
import sympy as sp


def newton(f: sp.Expr, sy: sp.Expr, x0: np.ndarray, tol: float, max_iter: int, pmax=10, damping=False, simplyfied=False):
    """Newton Verfahren zur Nullstellenbestimmung für Systeme

    Args:
        f (Sympy Expr.): Sympy Expression (e.g Matrix)
        sy (Sympy Expr.): Symbols used in f (e.g Matrix)
        x0 (ndarray): initial vector/guess
        tol (float): error tolerance from root of f
        max_iter (int): max number of iterations
        pmax (int, optional): max damping value 2^pmax. Defaults to 10
        damping (bool, optional): enable damping. Defaults to False
        simplyfied (bool, optional): uses simplyfied newton procedure. Defaults to False

    Returns:
        list: root of f
    """
    # Sympy
    sy = sp.Matrix([sy])
    f = sp.Matrix([f])
    df = f.jacobian(sy)
    f = sp.lambdify([sy], f, 'numpy')
    df = sp.lambdify([sy], df, 'numpy')

    # Numpy
    x0 = np.array([x0], dtype=np.float64).flatten()
    xn_min1 = np.full_like(x0, np.inf)
    xn = np.copy(x0)
    k = 1
    while np.linalg.norm(xn - xn_min1, 2) > tol and k <= max_iter:
        print(f'it:\t {k}')
        d = np.linalg.solve(df(x0) if simplyfied else df(xn) , -1 * f(xn)).flatten()
        # damping
        if damping:
            p = 0
            for i in range(pmax):
                if np.linalg.norm(xn + (d / (2 ** i)), 2) < np.linalg.norm(f(xn), 2):
                    p = i
                    break
            if p == 0:
                xn_min1 = xn
                xn = xn + d
            else:
                print(f'damping with p={p}')
                xn_min1 = xn
                xn = xn, xn + d / (2 ** p)
        else:
            xn_min1 = xn
            xn = xn + d
        print(f'x{k} =\t {xn}') 
        print(f'd{k} =\t {d}\n')
        k = k + 1
    return xn



####################################################################################################
# EXAMPLE NEWTON
####################################################################################################
if __name__ == '__main__':
    print('system example')
    x1, x2 = sp.symbols('x1, x2')
    f1 = 2*x1 + 4*x2
    f2 = 4*x1 + 8*x2**3
    f = sp.Matrix([f1, f2])
    x0 = np.array([4, 2])
    tol = 1e-5
    max_iter = 100
    print(newton(f, [x1, x2], x0, tol, max_iter))

    print('\nno system example')
    x = sp.symbols('x')
    f = sp.exp(x) - 5
    x0 = 4
    tol = 1e-5
    max_iter = 10
    print(newton(f, x, x0, tol, max_iter))