import numpy as np
import matplotlib.pyplot as plt

def pt_source_1D(f, t):
    """
    Retourne le float de l'ondelette tronquée au temps t en 1D
    :param f: float, fréquence d'oscillation
    :param t: float, temps en seconde
    :return: float, correspondant à l'ondelette tronquée au temps t
    """
    omega = 2 * np.pi * f
    a = [1, -21/32, 63/768, -1/512]
    return (a[0]*np.sin(omega*t) + a[1]*np.sin(omega*2*t) + a[2]*np.sin(omega*4*t) + a[3]*np.sin(omega*8*t))*(0 < t < 1/f)

def pt_source_2D(f, t):
    """
    Retourne le float de l'ondelette tronquée au temps t en 2D
    :param f: float, fréquence d'oscillation
    :param t: float, temps en seconde
    :return:  float, correspondant à l'ondelette tronquée au temps t
    """
    omega = 2 * np.pi * f
    b = [1, -21/(2*32), 63/(4*768), -1/(8*512)]
    return (sum(b) - (b[0]*np.cos(omega*t) + b[1]*np.cos(omega*2*t) + b[2]*np.cos(omega*4*t) + b[3]*np.cos(omega*8*t)))*(0 < t < 1/f)/omega

def pt_source_pression(f, t):
    """
    Retourne le float de l'ondelette tronquée au temps t en 2D
    :param f: float, fréquence d'oscillation
    :param t: float, temps en seconde
    :return:  float, correspondant à l'ondelette tronquée au temps t
    """
    omega = 2 * np.pi * f
    b = [1, -21/(2*32), 63/(4*768), -1/(8*512)]
    return (sum(b) - (b[0]*np.cos(omega*t) + b[1]*np.cos(omega*2*t) + b[2]*np.cos(omega*4*t) + b[3]*np.cos(omega*8*t)))*(0 < t < 1/f)/omega

# t = np.linspace(0, 0.1, 200)
# s = [pt_source(20, t[i]) for i in range(200)]
# plt.plot(t,s,'r-')
# plt.grid(True)
# plt.show()