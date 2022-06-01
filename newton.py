import numpy as np
import sympy
import sympy as sp

def newton(f, x0, tol, max_iter, pmax=10, damping=True):
    """Newton Verfahren zur Nullstellenbestimmung

    Args:
        f (Sympy Expr.): Sympy Expression (e.g Matrix)
        x0 (npArray): initial vector/guess
        tol (float): error tolerance from root of f
        max_iter (int): max number of iterations
        pmax (int, optional): max damping value 2^pmax. Defaults to 10.
        damping (bool, optional): enable damping. Defaults to True.

    Returns:
        list: root of f containing names and values
    """
    # Sympy
    x = list(f.free_symbols)
    df = f.jacobian(sp.Matrix(x))
    f = sp.lambdify([x], f, 'numpy')
    df = sp.lambdify([x], df, 'numpy')

    # Numpy
    xn = np.copy(x0)
    k = 0
    while np.linalg.norm(f(xn), 2) > tol and k < max_iter:
        d = np.linalg.solve(df(xn), -1 * f(xn)).flatten()
        # damping
        if damping:
            p = 0
            for i in range(pmax):
                if np.linalg.norm(xn + (d / (2 ** i)), 2) < np.linalg.norm(f(xn), 2):
                    p = i
                    break
            if p == 0:
                xn = xn + d
            else:
                print(f'damping with p={p}\n')
                xn = xn + d / (2 ** p)
        else:
            xn = xn + d
    return list(zip(x, xn))


# EXAMPLE NEWTON
if __name__ == '__main__':
    x1, x2, x3 = sp.symbols('x1, x2, x3')
    X = sp.Matrix([x1, x2, x3])

    f1 = x1 + x2 ** 2 + x3 ** 2 - 13
    f2 = sp.ln(x2 / 4) + sp.exp(0.5 * x3 - 1) - 1
    f3 = (x2 - 3) ** 2 - x3 ** 3 + 7
    f = sp.Matrix([f1, f2, f3])

    x0 = np.array([1.5, 3, 2.5])
    tol = 1e-5
    max_iter = 10

    print(newton(f, x0, tol, max_iter))