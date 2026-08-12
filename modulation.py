"""
============================================================
Ces fonctions permettent de réaliser différentes modulations
en masse volumique et en module d'Young. Elles renvoient alors
les 3 premières dérivées correspondant aux profils suivants :
- Sinusoidal
- Echelon
- Triangle

!!
- eps doit être petit devant 1
- alpha doit être compris entre 0 et 1
============================================================
"""

import numpy as np

def deriv_inv(a):
    """
    Calcule le développement (valeur + 3 premières dérivées temporelles) de 1/f à partir du développement a = [f, f', f'', f'''] de f.
    :param a: np.ndarray de taille 4, [f, f', f'', f''']
    :return: np.ndarray de taille 4, [1/f, (1/f)', (1/f)'', (1/f)''']
    """
    f, f1, f2, f3 = a[0], a[1], a[2], a[3]
    h = 1 / f
    h1 = -f1 / f ** 2
    h2 = 2 * f1 ** 2 / f ** 3 - f2 / f ** 2
    h3 = -6 * f1 ** 3 / f ** 4 + 6 * f1 * f2 / f ** 3 - f3 / f ** 2
    return np.array([h, h1, h2, h3])

def modulo(t,T):
    while T < t:
        t -= T
    return t

def rho_sinus(data, eps):
    def f(t):
        w = data.omega
        return  np.array([(1 + eps*np.sin(w*t + np.pi/2)), eps * w * np.cos(w*t + np.pi/2), - eps * w**2 * np.sin(w*t + np.pi/2), - eps * w**3 * np.cos(w*t + np.pi/2)])
    return f

def kappa_sinus(data, eps):
    def f(t):
        w = data.omega
        return  np.array([1 / (1 + eps * np.sin(w * t + np.pi / 2)),
                - eps * w * np.cos(w * t + np.pi / 2) / (1 + eps * np.sin(w * t + np.pi / 2)) ** 2,
                eps * w ** 2 * (np.sin(w * t + np.pi / 2) * (1 + eps * np.sin(w * t + np.pi / 2)) - 2 * eps * np.cos(w * t + np.pi / 2) ** 2) / (1 + eps * np.sin(w * t + np.pi / 2)) ** 3,
                - (eps * w ** 3 * np.cos(w * t + np.pi / 2) * (1 + eps * np.sin(w * t + np.pi / 2)) ** 2 + 6 * (1 + eps * np.sin(w * t + np.pi / 2)) * eps ** 2 * w ** 3 * np.cos(w * t + np.pi / 2) * np.sin(w * t + np.pi / 2) + 6 * eps * w ** 3 * np.cos(w * t + np.pi / 2)) / (1 + eps * np.sin(w * t + np.pi / 2)) ** 4])
    return f

def rho_echelon(data, eps):
    def f(t):
        T, alpha = (data.omega != 0) * (2*np.pi)/data.omega, data.param
        return  np.array([(1 + eps*(int(0 <= modulo(t,T) < alpha*T) - int(alpha*T <= modulo(t,T) < T))), 0,0,0])
    return f

def kappa_echelon(data, eps):
    def f(t):
        T, alpha = (data.omega != 0) * (2*np.pi)/data.omega, data.param
        return  np.array([1 / (1 + eps * (int(0 <= modulo(t, T) < alpha * T) - int(alpha * T <= modulo(t, T) < T))), 0, 0, 0])
    return f

def rho_triangle(data, eps):
    def f(t):
        T, alpha = (2 * np.pi) / data.omega, data.param
        tau = modulo(t, T)
        return  np.array([(1 + eps * (np.where(tau <= alpha * T, (2 * tau / (alpha * T)) - 1, (-2 / ((1 - alpha) * T)) * (tau - alpha * T) + 1))),
                (eps * (np.where(tau <= alpha * T, (2 / (alpha * T)) - 1, (-2 / ((1 - alpha) * T))))), 0, 0])
    return f

def kappa_triangle(data, eps):
    def f(t):
        T, alpha = (2*np.pi)/data.omega, data.param
        tau = modulo(t,T)
        return np.array([1 / (1 + eps * (np.where(tau <= alpha * T, (2 * tau / (alpha * T)) - 1, (-2 / ((1 - alpha) * T)) * (tau - alpha * T) + 1))),
                1 / (eps * (np.where(tau <= alpha * T, (2 / (alpha * T)) - 1, (-2 / ((1 - alpha) * T))))), 0, 0])
    return f

def rho_moy(data):
    def f(t):
        alpha = data.alpha
        return alpha*data.rho[0]*data.rho_mt[0](data, data.eps_r[0])(t) + (1-alpha)*data.rho[1]*data.rho_mt[1](data, data.eps_r[1])(t)
    return f

def kappa_moy(data):
    def f(t):
        alpha = data.alpha
        kappa_c0 = data.kappa[0]*data.kappa_mt[0](data, data.eps_kappa[0])(t)
        kappa_c1 = data.kappa[1]*data.kappa_mt[1](data, data.eps_kappa[1])(t)
        return deriv_inv(alpha * deriv_inv(kappa_c0) + (1 - alpha) * deriv_inv(kappa_c1))
    return f