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

def rho_echelon_moy(data):
    def f(t):
        T, alpha = (data.omega != 0) * (2*np.pi)/data.omega, data.param
        r = data.alpha*data.rho[0] + (1-data.alpha)*data.rho[1]
        reps = data.alpha*data.rho[0]*data.eps_r[0] + (1-data.alpha)*data.rho[1]*data.eps_r[1]
        return np.array([r + reps*(int(0 <= modulo(t,T) < alpha*T) - int(alpha*T <= modulo(t,T) < T)),0,0,0])
    return f

def kappa_echelon(data, eps):
    def f(t):
        T, alpha = (data.omega != 0) * (2*np.pi)/data.omega, data.param
        return  np.array([1 / (1 + eps * (int(0 <= modulo(t, T) < alpha * T) - int(alpha * T <= modulo(t, T) < T))), 0, 0, 0])
    return f

def kappa_echelon_moy(data):
    def f(t):
        T, alpha = (data.omega != 0) * (2 * np.pi) / data.omega, data.param
        return np.array([(data.alpha/(data.kappa[0]*kappa_echelon(data, data.eps_kappa[0])(t)[0])+ (1-data.alpha)/(data.kappa[1]*kappa_echelon(data, data.eps_kappa[1])(t)[0]))**(-1), 0, 0, 0])
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