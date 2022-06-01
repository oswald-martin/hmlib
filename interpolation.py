import numpy as np


def lagrange(x, y):
    """Lagrange interpolation. suiteble for small datasets <20

    Args:
        x (npArray): x values
        y (npArray): y values

    Returns:
        function: interpolated function
    """
    assert np.shape(x) == np.shape(y), 'x and y need to be the same size'
    li = np.ones(np.shape(x), dtype=np.float64)
    def interpol(x_int):
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j:
                    li[i] = li[i] * (x_int - x[j])/(x[i] - x[j])
        return y@li
    return interpol

# EXAMPLE LAGRANGE
if __name__ == '__main__':
    x = np.array([0, 2_500, 5_000, 10_000])
    y = np.array([1_013, 747, 540, 226])
    x_int = np.array([3_750])
    print(lagrange_int(x, y)(3_750))
    '''
    [637.328125] stimmt
    '''
