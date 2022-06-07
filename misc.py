import numpy as np

def h2n(lo: float, hi: float, h: float) -> int:
    """Anzahl Abschnitte n aus Abschnittsbreite h rechnen

    Args:
        lo (float): lower bound
        hi (float): higher bound
        h (float): step width

    Returns:
        int: number of segments
    """
    a, b = (a, b) if a < b else (b, a)
    return int(np.ceil((b - a)/h))