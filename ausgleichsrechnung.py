import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def ausgleich_plot(f, x, y, lo=None, hi=None, n=1000):
    """plottet x,y Werte sowie n datenpunkte der funktion f

    Edit return with title(), xlabel() etc.

    Args:
        f (funktion): function(x) -> y
        x (npArray): x values
        y (npArray): y values
        lo (int, optional): lower bound. Defaults to x.min
        hi (int, optional): higher bound. Defaults to x.max
        n (int, optional): datapoints between lo and hi. Defaults to 1000.

    Returns:
        plt: plt object. use plt.show() to show it.
    """
    lo = lo if lo != None else x.min()
    hi = hi if hi != None else x.max()
    plt.plot(x, y, 'o')
    xx = np.linspace(lo, hi, n)
    plt.plot(xx, f(xx))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ausgleichsrechnung')
    return plt



def gauss_newton(f, x, y, lam0, tol, max_iter, pmax=5, damping=True):
    """Gedämpftes Gauss-Newton verfahren Ausgleichsrechnung.

    Args:
        f (function): function(x, p) where p is npArray
        x (npArray): x datapoints
        y (npArray): y datapoints
        lam0 (npArray): initial p vector/guess
        tol (float): error tolarance
        max_iter (int): max number of iterations
        pmax (int, optional): max damping value 2^pmax. Defaults to 5.
        damping (bool, optional): enable damping. Defaults to True.

    Returns:
        function: fittet function f(x) -> y
    """
    # Sympy
    p = sp.symbols('p0:%d' % len(lam0))
    g = sp.Matrix([y[k]-f(x[k],p) for k in range(len(x))])
    Dg = g.jacobian(p)
    g = sp.lambdify([p], g, 'numpy')
    Dg = sp.lambdify([p], Dg, 'numpy')

    # Numpy
    k=0
    lam=np.copy(lam0)
    increment = tol+1
    err_func = np.linalg.norm(g(lam))**2
    
    # Gauss-Newton
    while increment > tol and k < max_iter:
        [Q,R] = np.linalg.qr(Dg(lam))
        delta = np.linalg.solve(R,-Q.T @ g(lam)).flatten()
        p=0
        while damping == True:
            p_error_func = np.linalg.norm(g(lam + (delta / 2**p)))**2
            if p_error_func<err_func :
                break
            else:
                p = p+1
            if p > pmax:
                p=0
                break           
        lam = lam + delta / 2**p
        err_func = np.linalg.norm(g(lam))**2
        increment =np.linalg.norm(delta / 2**p)
        k = k+1
        print(f'Iteration:  {k}')
        print(f'lambda =    {lam}')
        print(f'Inkrement = {increment}')
        print(f'Fehlerfunktional = {err_func}')
        print('')

    # create func
    t = sp.symbols('t')
    F = f(t,lam)
    F = sp.lambdify([t],F,'numpy')

    return F

# EXAMPLE GAUSS-NEWTON
if __name__ == '__main__':
    f = lambda x, p: (p[0] + p[1] * 10**(p[2]+p[3] * x)) / (1 + 10**(p[2]+p[3] * x)) 
    x=np.array([2, 2.5, 3, 3.5, 4, 4.5, 5], dtype=np.float64)
    y=np.array([159.57, 159.88, 159.89, 160.30, 160.84, 160.94, 161.56], dtype=np.float64)
    lam0 = np.array([100, 120, 3, -1],dtype=np.float64)
    tol = 1e-5
    max_iter = 30
    pmax = 5
    damping = True

    F = gauss_newton(f, x, y, lam0, tol, max_iter, pmax, damping)
    ausgleich_plot(F, x, y).show()