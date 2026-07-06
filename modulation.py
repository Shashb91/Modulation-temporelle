"""
============================================================
Ces fonctions permettent de réaliser différentes modulations
en masse volumique et en module d'Young. Elles renvoient alors
les 3 premières dérivées correspondant aux profils suivants :
- Sinusoidal
- Echelon

!!
- eps doit être petit devant 1
- alpha doit être compris entre 0 et 1
============================================================
"""

import numpy as np

def modulo(t,T):
    while T < t:
        t -= T
    return t

def rho_sinus(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_r
        w = data.omega
        return [(1 - eps*np.sin(w*t + np.pi/2)), eps * w * np.cos(w*t + np.pi/2), eps * w**2 * np.sin(w*t + np.pi/2), eps * w**3 * np.cos(w*t + np.pi/2)]
    return f

def kappa_sinus(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_kappa
        w = data.omega
        return [1 / (1 + eps * np.sin(w * t + np.pi / 2)),
                - eps * w * np.cos(w * t + np.pi / 2) / (1 + eps * np.sin(w * t + np.pi / 2)) ** 2,
                eps * w ** 2 * (np.sin(w * t + np.pi / 2) * (1 + eps * np.sin(w * t + np.pi / 2)) - 2 * eps * np.cos(w * t + np.pi / 2) ** 2) / (1 + eps * np.sin(w * t + np.pi / 2)) ** 3,
                - (eps * w ** 3 * np.cos(w * t + np.pi / 2) * (1 + eps * np.sin(w * t + np.pi / 2)) ** 2 + 6 * (1 + eps * np.sin(w * t + np.pi / 2)) * eps ** 2 * w ** 3 * np.cos(w * t + np.pi / 2) * np.sin(w * t + np.pi / 2) + 6 * eps * w ** 3 * np.cos(w * t + np.pi / 2)) / (1 + eps * np.sin(w * t + np.pi / 2)) ** 4]
    return f

def rho_echelon(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_r
        T, alpha = (data.omega != 0) * (2*np.pi)/data.omega, data.param
        return [(1 + eps*(int(0 <= modulo(t,T) < alpha*T) - int(alpha*T <= modulo(t,T) < T))), 0,0,0]
    return f

def kappa_echelon(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_kappa
        T, alpha = (data.omega != 0) * (2*np.pi)/data.omega, data.param
        return [1 / (1 + eps * (int(0 <= modulo(t, T) < alpha * T) - int(alpha * T <= modulo(t, T) < T))), 0, 0, 0]
    return f

def rho_triangle(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_r
        T, alpha = (2 * np.pi) / data.omega, data.param
        tau = modulo(t, T)
        return [(1 + eps * (np.where(tau <= alpha * T, (2 * tau / (alpha * T)) - 1, (-2 / ((1 - alpha) * T)) * (tau - alpha * T) + 1))),
                (eps * (np.where(tau <= alpha * T, (2 / (alpha * T)) - 1, (-2 / ((1 - alpha) * T))))), 0, 0]
    return f

def kappa_triangle(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_kappa
        T, alpha = (2*np.pi)/data.omega, data.param
        tau = modulo(t,T)
        return [1 / (1 + eps * (np.where(tau <= alpha * T, (2 * tau / (alpha * T)) - 1, (-2 / ((1 - alpha) * T)) * (tau - alpha * T) + 1))),
                1 / (eps * (np.where(tau <= alpha * T, (2 / (alpha * T)) - 1, (-2 / ((1 - alpha) * T))))), 0, 0]
    return f