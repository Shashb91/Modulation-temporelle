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
        return [data.rho * (1 - eps*np.sin(w*t)), -data.rho * eps * w * np.cos(w*t), data.rho * eps * w**2 * np.sin(w*t), data.rho * eps * w**3 * np.cos(w*t)]
    return f

def E_sinus(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_E
        w = data.omega
        return [data.e * 1/(1 + eps*np.sin(w*t)),
                - data.e * eps * w *np.cos(w*t)/(1 + eps*np.sin(w*t))**2,
                data.e * eps * w**2 * (np.sin(w*t)*(1+eps*np.sin(w*t))-2*eps*np.cos(w*t)**2)/(1 + eps*np.sin(w*t))**3,
                - data.e * (eps*w**3*np.cos(w*t)*(1+eps*np.sin(w*t))**2+6*(1+eps*np.sin(w*t))*eps**2*w**3*np.cos(w*t)*np.sin(w*t)+6*eps*w**3*np.cos(w*t))/(1 + eps*np.sin(w*t))**4]
    return f

def rho_echelon(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_r
        T, alpha = (data.omega != 0) * (2*np.pi)/data.omega, data.alpha
        return [data.rho * (1 + eps*(int(0 <= modulo(t,T) < alpha*T) - int(alpha*T <= modulo(t,T) < T))), 0,0,0]
    return f

def E_echelon(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_E
        T, alpha = (data.omega != 0) * (2*np.pi)/data.omega, data.alpha
        return [data.e / (1 + eps*(int(0 <= modulo(t,T) < alpha*T) - int(alpha*T <= modulo(t,T) < T))), 0,0,0]
    return f

def rho_triangle(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_r
        T, alpha = (2 * np.pi) / data.omega, data.alpha
        tau = modulo(t, T)
        return [data.rho * (1 + eps * (np.where(tau <= alpha * T, (2 * tau / (alpha * T)) - 1, (-2 / ((1 - alpha) * T)) * (tau - alpha * T) + 1))),
                data.rho * (eps * (np.where(tau <= alpha * T, (2 / (alpha * T)) - 1, (-2 / ((1 - alpha) * T))))), 0, 0]
    return f

def E_triangle(data):
    def f(t):
        try : eps = data.eps
        except : eps = data.eps_E
        T, alpha = (2*np.pi)/data.omega, data.alpha
        tau = modulo(t,T)
        return [data.e / (1 + eps * (np.where(tau <= alpha * T, (2 * tau / (alpha * T)) - 1, (-2 / ((1 - alpha) * T)) * (tau - alpha * T) + 1))),
                data.e / (eps * (np.where(tau <= alpha * T, (2 / (alpha * T)) - 1, (-2 / ((1 - alpha) * T))))), 0, 0]
    return f