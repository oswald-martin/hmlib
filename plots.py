"""Plots Modul.

Funcs:
    - simple
    - plot_vector_field
    - ausgleich_plot (für Ausgleichsrechnungen)
    - wireframe
    - surface
    - contour

@author: Martin Oswald
@license: MIT
@version: 1.0
"""
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def simple(f: Callable[[float], float], lo: float, hi: float, n=1000, label=None):
    """plottet ein einfachen Plot der Funktion f

    Args:
        f (function): f(x, y)
        xmin (float): lower bound
        xmax (float): higher bound
        n (int, optional): datapoints between lo and hi. Defaults to 1000.
        label (str, optional): legend name of function f

    Returns:
        plt: plt object. use plt.show() to show it.
    """
    x = np.linspace(lo, hi, n)
    plt.plot(x, f(x), label='func' if label == None else label)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simple Plot')
    return plt


def plot_vector_field(f: Callable[[float], float], xmin: float, xmax: float, ymin: float, ymax: float, hx: float, hy: float):
    """plottet ein vektorfeld

    Args:
        f (function): y' = f(x, y)
        xmin (float): start x value
        xmax (float): end x value
        ymin (float): start y value
        ymax (float): end y value
        hx (float): x step width
        hy (float): y step width

    Returns:
        plt: plt object. use plt.show() to show it.
    """
    x = np.arange(xmin, xmax + hx,hx)
    y = np.arange(ymin, ymax+ hy,hy)
    [x,y] = np.meshgrid(x,y)
    dy = f(x,y)
    dx = np.ones(np.shape(dy))
    dz = np.sqrt(dx**2 + dy**2)
    dx = dx/dz
    dy = dy/dz
    plt.quiver(x,y,dx,dy, angles='xy', width=0.003 )
    plt.xlim(xmin - hx, xmax + hx)
    plt.ylim(ymin - hy, ymax + hy)
    return plt



def ausgleich_plot(f: Callable[[float], float], x: np.ndarray, y: np.ndarray, lo=None, hi=None, n=1000, label=None):
    """plottet x,y Werte sowie n datenpunkte der funktion f

    Edit return with title(), xlabel() etc.

    Args:
        f (funktion): f(x) -> y
        x (ndarray): x values
        y (ndarray): y values
        lo (int, optional): lower bound. Defaults to x.min
        hi (int, optional): higher bound. Defaults to x.max
        n (int, optional): datapoints between lo and hi. Defaults to 1000.
        label (str, optional): legend name of function f

    Returns:
        plt: plt object. use plt.show() to show it.
    """
    lo = lo if lo != None else np.min(x)
    hi = hi if hi != None else np.max(x)
    plt.plot(x, y, 'o')
    xx = np.linspace(lo, hi, n)
    plt.plot(xx, f(xx), label='func' if label == None else label)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ausgleichsrechnung')
    return plt

def wireframe(f: Callable[[float, float], float], x: np.ndarray, y: np.ndarray, cstride=5, rstride=5):
    """creates a 3d wireframe plot with basic axis names x,y,z.

    Edit return with title(), xlabel() etc.

    Args:
        f (function): f(x, y) -> z
        x (ndarray): x values
        y (ndarray): y values
        cstride (int, optional): column stride. Defaults to 5
        rstride (int, optional): row stride. Defaults to 5

    Returns:
        plt: plt object. use plt.show() to show it.
    """
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, f(X, Y), color='black', cstride=cstride, rstride=rstride)
    ax.set_title('wireframe')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return plt


def surface(f: Callable[[float, float], float], x: np.ndarray, y: np.ndarray):
    """creates a 3d meshgrid plot with basic axis names x,y,z.

    Edit return with title(), xlabel() etc.

    Args:
        f (function): f(x, y) -> z
        x (ndarray): x values
        y (ndarray): y values

    Returns:
        plt: plt object. use plt.show() to show it.
    """
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, f(X, Y), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title('Meshgrid')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return plt


def contour(f: Callable[[float, float], float], x: np.ndarray, y: np.ndarray):
    """creates a 2d contour plot with basic axis names x,y.

    Edit return with title(), xlabel() etc.

    Args:
        f (function): f(x, y) -> z
        x (ndarray): x values
        y (ndarray): y values

    Returns:
        plt: plt object. use plt.show() to show it.
    """
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, f(X, Y))
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return plt


####################################################################################################
# EXAPLE
####################################################################################################
if __name__ == '__main__':
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, np.pi/2, 100)
    def f(x, y): return (x ** 2) * np.sin(2 * y)/(9.81)
    wireframe(f, x, y).show()
    surface(f, x, y).show()
    contour(f, x, y).show()
