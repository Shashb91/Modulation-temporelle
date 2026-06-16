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
        eps, w = data.eps, data.omega
        return [data.rho * (1 - eps*np.sin(w*t)), -data.rho * eps * w * np.cos(w*t), data.rho * eps * w**2 * np.sin(w*t), data.rho * eps * w**3 * np.cos(w*t)]
    return f

def E_sinus(data):
    def f(t):
        eps, w = data.eps, data.omega
        return [data.e * 1/(1 + eps*np.sin(w*t)),
                - data.e * eps * w *np.cos(w*t)/(1 + eps*np.sin(w*t))**2,
                data.e * eps * w**2 * (2*eps + np.sin(w*t) - eps*np.sin(w*t)**2)/(1 + eps*np.sin(w*t))**3,
                data.e * eps * w**3 * (np.cos(w*t)*(1+2*eps**2) - 3 * eps * np.cos(w*t)*np.sin(w*t) + 2 * eps ** 2 * np.cos(w*t)*np.sin(w*t)**2)/(1 + eps*np.sin(w*t))**4]
    return f

def rho_echelon(data):
    def f(t):
        eps, T, alpha = data.eps, (2*np.pi)/data.omega, data.alpha
        return [data.rho * (1 + eps*(int(0 < modulo(t,T) < alpha*T) - int(alpha*T < modulo(t,T) < T))), 0,0,0]
    return f

def E_echelon(data):
    def f(t):
        eps, T, alpha = data.eps, (2*np.pi)/data.omega, data.alpha
        return [data.e * (1 + eps*(int(0 < modulo(t,T) < alpha*T) - int(alpha*T < modulo(t,T) < T))), 0,0,0]
    return f