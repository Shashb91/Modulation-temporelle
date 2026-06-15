import numpy as np
"""
============================================================
Ces fonctions permettent de réaliser différentes modulations
en masse volumique et en module d'Young. Elles renvoient alors
les 3 premières dérivées correspondant aux profils suivants :
- Sinusoidal
- Echelon

/!\ 
- epsilon doit être petit devant 1
- alpha doit être compris entre 0 et 1 
============================================================ 
"""


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
        return [1/data.e * (1 + eps*np.sin(w*t)), 1/data.e * eps * w * np.cos(w*t), - 1/data.e * eps * w**2 * np.sin(w*t), - 1/data.e * eps * w**3 * np.cos(w*t)]
    return f

def rho_echelon(data, alpha):
    def f(t):
        eps, T = data.eps, (2*np.pi)/data.omega
        return [data.rho * (1 + eps*(0 < modulo(t,T) < alpha*T)), 0,0,0]
    return f

def E_echelon(data, alpha):
    def f(t):
        eps, T = data.eps, (2*np.pi)/data.omega
        return [1/data.e * (1 + eps*(alpha*T < modulo(t,T) < 1)), 0,0,0]
    return f