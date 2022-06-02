import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def wireframe(x, y, f, cstride=5, rstride=5):
    """creates a 3d wireframe plot with basic axis names x,y,z.

    Edit return with title(), xlabel() etc.

    Args:
        x (npArray): x values
        y (npArray): y values
        f (function): f(x, y) -> z
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


def surface(x, y, f):
    """creates a 3d meshgrid plot with basic axis names x,y,z.

    Edit return with title(), xlabel() etc.

    Args:
        x (npArray): x values
        y (npArray): y values
        f (function): f(x, y) -> z

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


def contour(x, y, f):
    """creates a 2d contour plot with basic axis names x,y.

    Edit return with title(), xlabel() etc.

    Args:
        x (npArray): x values
        y (npArray): y values
        f (function): f(x, y) -> z

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


# EXAPLE
if __name__ == '__main__':
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, np.pi/2, 100)
    def f(x, y): return (x ** 2) * np.sin(2 * y)/(9.81)
    wireframe(x, y, f).show()
    surface(x, y, f).show()
    contour(x, y, f).show()
