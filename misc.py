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
    lo, hi = (lo, hi) if lo < hi else (hi, lo)
    return int(np.ceil((hi - lo)/h))