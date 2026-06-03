import numpy as np

def pt_source(f, t):
    """
    Retourne l'array de dimension N, retracant l'ondelette tronquée sur toute la durée N*dt
    :param f: float, fréquence d'oscillation
    :param N: int, discrétisation temporelle
    :param dt: float, infinitésimal temporel
    :return: np.ndarray(), ondelette tronquée totale
    """
    omega = 2 * np.pi * f
    a = [0, 1, -21/32, 63/768, -1/152]
    return sum([a[m] * np.sin(omega * 2**(m-1) * t) for m in range(1, len(a))])*(0 < t < 1/f)
