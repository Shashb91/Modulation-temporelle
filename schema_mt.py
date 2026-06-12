import numpy as np

from donnee import *
from tqdm import trange
from math import factorial
ncols = 125                                                                       #largeur de la barre de chargement

def A_mt(data):
    """
    Retourne une fonction mathématique retournant une liste des 3 dérivées premières dans le cas modulé choisi !
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: f, fonction à évaluer
    """
    def f(t):
        rE, eps, w = data.rho*data.e, data.eps, data.omega
        elmt = np.array([[0,0],[1,0]])
        A = np.array([[0, 1],[1/rE * (1 + eps*np.sin(w*t))/(1-eps*np.sin(w*t)), 0]])
        B = 1/rE * (2*eps*w*np.cos(w*t))/(1-eps*np.sin(w*t))**2 * elmt
        C = 1/rE * (2*eps*w**2*(2*eps-np.sin(w*t)*(1+eps**2)+eps*np.sin(w*t)**2))/(1-eps*np.sin(w*t))**3 * elmt
        D = 1/rE * (-2*eps*w**2*np.cos(w*t)*(5*eps**2*np.sin(w*t)**2+4*eps*(1+eps**2)*np.sin(w*t)-(1+8*eps**2)))/(1-eps*np.sin(w*t))**4 * elmt
        return [A, B, C, D]
    return f

def ADER41D_mt(data):
    """
    Utilise le schéma d'ADER4 en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en quantité de mouvement et pression du problème 1D
    """
    data.U = np.zeros((data.N, data.M, 2))

    for n in trange(data.N-1, ncols = ncols):
        for i in range(2,data.M-2):
            rE, eps, w, t = data.rho/data.e, data.eps, data.omega, n*data.dt
            elmt = np.array([[0, 0], [1, 0]])
            g = [np.array([[0,1],[1/rE * (1 + eps*np.sin(w*t))/(1-eps*np.sin(w*t)), 0]]),
                 1/rE * (2*eps*w*np.cos(w*t))/(1-eps*np.sin(w*t))**2 * elmt,
                 1/rE * (2*eps*w**2*(2*eps-np.sin(w*t)*(1+eps**2)+eps*np.sin(w*t)**2))/(1-eps*np.sin(w*t))**3 * elmt,
                 1/rE * (-2*eps*w**2*np.cos(w*t)*(5*eps**2*np.sin(w*t)**2+4*eps*(1+eps**2)*np.sin(w*t)-(1+8*eps**2)))/(1-eps*np.sin(w*t))**4 * elmt]
            # g = A_mt(data)(n*data.dt)
            a = -g[0]
            b1, b2 = - g[1], g[0] @ g[0]
            c1, c2, c3 = - g[2], 3 * g[1] @ g[0], - g[0] @ g[0] @ g[0]
            d1, d2, d3, d4 = - g[3], 4 * g[2] @ g[0] + 3 * g[2] @ g[2], -6 * g[1] @ g[0] @ g[0], g[0] @ g[0] @ g[0] @ g[0]

            dxU = np.array([1/(12*data.dx) * (data.U[n, i-2,:] - 8*data.U[n,i-1,:] + 8*data.U[n,i+1,:] - data.U[n,i+2,:]),
                            1/(12*data.dx**2) * (-data.U[n, i-2,:] + 16*data.U[n,i-1,:] - 30*data.U[n,i,:] + 16*data.U[n,i+1,:] - data.U[n,i+2,:]),
                            1/(12*data.dx**3) * (-data.U[n, i-2,:] + 2*data.U[n,i-1,:] - 2*data.U[n,i+1,:] + data.U[n,i+2,:]),
                            1/(24*data.dx**4) * (-data.U[n, i-2,:] + 4*data.U[n,i-1,:] - 6*data.U[n,i,:] + 4*data.U[n,i+1,:] - data.U[n,i+2,:])])

            dtU = np.array([a @ dxU[0], b1 @ dxU[0] + b2 @ dxU[1], c1 @ dxU[0] + c2 @ dxU[1] + c3 @ dxU[2], d1 @ dxU[0] + d2 @ dxU[1] + d3 @ dxU[2] + d4 @ dxU[3]])
            s = (data.dt / data.dx) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * np.array([data.opt, not data.opt])
            data.U[n + 1, i, :] = data.U[n, i, :] + sum([data.dt**(j + 1)/factorial(j + 1)*dtU[j, :] for j in range(4)]) + s

    for n in range(data.N):
        for i in range(data.M):
            data.U[n, i, :] = data.U[n, i, :]/(data.rho*(1-data.eps*np.sin(data.omega*n*data.dt)))
    return data.U