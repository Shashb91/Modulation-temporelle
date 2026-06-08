import numpy as np
import matplotlib.pyplot as plt

def pt_source(f, t):
    """
    Retourne l'array de dimension N, retracant l'ondelette tronquée sur toute la durée N*dt
    :param f: float, fréquence d'oscillation
    :param t: float, temps d'oscillation en seconde
    :return: np.ndarray(), ondelette tronquée totale
    """
    omega = 2 * np.pi * f
    a = [1, -21/32, 63/768, -1/512]
    return (a[0]*np.sin(omega*t) + a[1]*np.sin(omega*2*t) + a[2]*np.sin(omega*4*t) + a[3]*np.sin(omega*8*t))*(0 < t < 1/f)

# t = np.linspace(0, 0.1, 200)
# s = [pt_source(20, t[i]) for i in range(200)]
# plt.plot(t,s,'r-')
# plt.grid(True)
# plt.show()