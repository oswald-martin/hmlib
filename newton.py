import numpy as np
import sympy as sp


def newton(f: sp.Expr, x0: np.ndarray, tol: float, max_iter: int, pmax=10, damping=False, simplyfied=False):
    """Newton Verfahren zur Nullstellenbestimmung für Systeme

    Args:
        f (Sympy Expr.): Sympy Expression (e.g Matrix)
        x0 (ndarray): initial vector/guess
        tol (float): error tolerance from root of f
        max_iter (int): max number of iterations
        pmax (int, optional): max damping value 2^pmax. Defaults to 10
        damping (bool, optional): enable damping. Defaults to False
        simplyfied (bool, optional): uses simplyfied newton procedure. Defaults to False

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
    k = 1
    while np.linalg.norm(f(xn), 2) > tol and k <= max_iter:
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
                xn = xn + d
            else:
                print(f'damping with p={p}')
                xn = xn + d / (2 ** p)
        else:
            xn = xn + d
        print(f'x{k} =\t {xn}') 
        print(f'd{k} =\t {d}\n')
        k = k + 1
    return list(zip(x, xn))



####################################################################################################
# EXAMPLE NEWTON
####################################################################################################
if __name__ == '__main__':
    x1, x2, x3 = sp.symbols('x1, x2, x3')

    f1 = x1 + x2 ** 2 + x3 ** 2 - 13
    f2 = sp.ln(x2 / 4) + sp.exp(0.5 * x3 - 1) - 1
    f3 = (x2 - 3) ** 2 - x3 ** 3 + 7
    f = sp.Matrix([f1, f2, f3])

    x0 = np.array([1.5, 3, 2.5])
    tol = 1e-5
    max_iter = 10

    print(newton(f, x0, tol, max_iter, damping=False, simplyfied=True))