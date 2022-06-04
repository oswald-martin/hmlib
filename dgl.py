from typing import Callable, Tuple, overload
import numpy as np
import functools


def h2n(lo: float, hi: float, h: float) -> int:
    """Anzahl Abschnitte n aus Abschnittsbreite h rechnen

    Args:
        lo (float): lower bound
        hi (float): higher bound
        h (float): step width

    Returns:
        int: number of segments
    """
    lo, hi = (lo, hi) if lo < hi else (hi, lo)
    return int(np.ceil((hi - lo)/h))


def __rk_step__(yi, h, b, k):
    return yi + h * np.sum((b*k.T).T, axis=0)


def __get_k__(a, c, h, f, xi, yi):
    k = np.zeros(len(a) if isinstance(yi, (float, int)) else (len(a), len(yi)), dtype=np.float64) 
    for n in range(len(a)):
        k[n] = np.array([f(xi+c[n]*h, yi+h*np.sum((a[n]*k.T).T, axis=0))]).flatten()
    return k


def __tril_a__(a):
    n = int((np.sqrt(8*len(a)+1)-1)/2)
    res = np.zeros((n+1, n+1))
    ind = np.tril_indices(n)
    res[1:,:][ind] = a
    return res


@overload
def butcher(f: Callable[[float, float], float], lo: float, hi: float, n: int, y0: float, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Custom Butcher table für DGL erster Ordnung

    Args:
        f (function): y' = f(x, y)
        lo (float): lower bound
        hi (float): higher bound
        n (int): number of segments. call h2n() to change step width to number of segments
        y0 (float): start condition
        a (ndarray): butcher c column
        b (ndarray): butcher b row
        c (ndarray): c matrix in 1dArray or Matrix

    Returns:
        Tuple[ndarray, ndarray]: x, y values for numeric dgl solution
    """
    ...
@overload
def butcher(f: Callable[[float, np.ndarray], np.ndarray], lo: float, hi: float, n: int, y0: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Custom Butcher table für DGL Systeme

    Args:
        f (function): y' = f(x, y)
        lo (float): lower bound
        hi (float): higher bound
        n (int): number of segments. call h2n() to change step width to number of segments
        y0 (ndarray): start condition
        a (ndarray): butcher c column
        b (ndarray): butcher b row
        c (ndarray): c matrix in 1dArray or Matrix

    Returns:
        Tuple[ndarray, ndarray]: x, y values for numeric dgl solution
    """
    ...
def butcher(f, lo, hi, n, y0, a, b, c):
    if len(np.shape(a)) == 1:
        a = __tril_a__(a)
    h = np.divide(hi-lo, n)
    x = np.linspace(lo, hi, n+1, dtype=np.float64)
    def rk_reducer(y, x):
        k = __get_k__(a, c, h, f, x, y[-1])
        return np.append(y, [__rk_step__(y[-1], h, b, k)], axis=0)
    y = functools.reduce(rk_reducer, x[:-1], [y0])
    return(x, y if isinstance(y[0], float) else y[:,0])


@overload
def euler(f: Callable[[float, float], float], lo: float, hi: float, n: int, y0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Euler DGL Verfahren erster Ordnung

    Args:
        f (function): y' = f(x, y)
        lo (float): loewr bound
        hi (float): higher bound
        n (int): number of segments. call h2n() to change step width to number of segments
        y0 (float): start condition

    Returns:
        (ndarray, ndarray): x, y values for numeric dgl solution
    """
    ...
@overload
def euler(f: Callable[[float, np.ndarray], np.ndarray], lo: float, hi: float, n: int, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Euler DGL Verfahren für Systeme

    Args:
        f (function): y' = f(x, y)
        lo (float): loewr bound
        hi (float): higher bound
        n (int): number of segments. call h2n() to change step width to number of segments
        y0 (ndarray): start condition

    Returns:
        (ndarray, ndarray): x, y values for numeric dgl solution
    """
    ...
def euler(f, lo, hi, n, y0):
    a = []
    b = [1]
    c = [0]
    return butcher(f, lo, hi, n, y0, a, b, c)



@overload
def mittelpunkt(f: Callable[[float, float], float], lo: float, hi: float, n: int, y0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mittelpunkt DGL Verfahren erster Ordnung

    Args:
        f (function): y' = f(x, y)
        lo (float): loewr bound
        hi (float): higher bound
        n (int): number of segments. call h2n() to change step width to number of segments
        y0 (float): start condition

    Returns:
        (ndarray, ndarray): x, y values for numeric dgl solution
    """
    ...
@overload
def mittelpunkt(f: Callable[[float, np.ndarray], np.ndarray], lo: float, hi: float, n: int, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mittelpunkt DGL Verfahren für Systeme

    Args:
        f (function): y' = f(x, y)
        lo (float): loewr bound
        hi (float): higher bound
        n (int): number of segments. call h2n() to change step width to number of segments
        y0 (ndarray): start condition

    Returns:
        (ndarray, ndarray): x, y values for numeric dgl solution
    """
    ...
def mittelpunkt(f, lo, hi, n, y0):
    a = [0.5]
    b = [0, 1]
    c = [0, 0.5]
    return butcher(f, lo, hi, n, y0, a, b, c)



@overload
def heun(f: Callable[[float, float], float], lo: float, hi: float, n: int, y0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Heun bzw. ModEuler DGL Verfahren erster Ordnung

    Args:
        f (function): y' = f(x, y)
        lo (float): loewr bound
        hi (float): higher bound
        n (int): number of segments. call h2n() to change step width to number of segments
        y0 (float): start condition

    Returns:
        (ndarray, ndarray): x, y values for numeric dgl solution
    """
    ...
@overload
def heun(f: Callable[[float, np.ndarray], np.ndarray], lo: float, hi: float, n: int, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Heun bzw. ModEuler DGL Verfahren für Systeme

    Args:
        f (function): y' = f(x, y)
        lo (float): loewr bound
        hi (float): higher bound
        n (int): number of segments. call h2n() to change step width to number of segments
        y0 (ndarray): start condition

    Returns:
        (ndarray, ndarray): x, y values for numeric dgl solution
    """
    ...
def heun(f, lo, hi, n, y0):
    a = [1]
    b = [0.5, 0.5]
    c = [0, 1]
    return butcher(f, lo, hi, n, y0, a, b, c)



@overload
def rk4(f: Callable[[float, float], float], lo: float, hi: float, n: int, y0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Runge-Kutta-4 DGL Verfahren erster Ordnung

    Args:
        f (function): y' = f(x, y)
        lo (float): loewr bound
        hi (float): higher bound
        n (int): number of segments. call h2n() to change step width to number of segments
        y0 (float): start condition

    Returns:
        (ndarray, ndarray): x, y values for numeric dgl solution
    """
    ...
@overload
def rk4(f: Callable[[float, np.ndarray], np.ndarray], lo: float, hi: float, n: int, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Runge-Kutta-4 DGL Verfahren für Systeme

    Args:
        f (function): y' = f(x, y)
        lo (float): loewr bound
        hi (float): higher bound
        n (int): number of segments. call h2n() to change step width to number of segments
        y0 (ndarray): start condition

    Returns:
        (ndarray, ndarray): x, y values for numeric dgl solution
    """
    ...
def rk4(f, lo, hi, n, y0):
    a = [0.5, 0, 0.5, 0, 0, 1]
    b = [1/6, 1/3, 1/3, 1/6]
    c = [0, 0.5, 0.5, 1]
    return butcher(f, lo, hi, n, y0, a, b, c)



####################################################################################################
# EXAMPLE DGL SYSTEM AND SINGLE DGL
####################################################################################################
if __name__ == '__main__':
    f = lambda x, z: [z[1], z[2], 10*np.exp(-x) - 5*z[2] - 8*z[1] - 6*z[0]]
    z0 = [2, 0, 0]
    lo = 0
    hi = 1
    h = 0.2
    n = h2n(lo, hi, h)
    print('dgl systems example')
    print(euler(f, lo, hi, n, z0))

    f = lambda x, y: x**2 + 0.1 * y
    lo = -1.5
    hi = 1.5
    n = 5
    y0 = 0
    print('single dgl example')
    print(rk4(f, lo, hi, n, y0))