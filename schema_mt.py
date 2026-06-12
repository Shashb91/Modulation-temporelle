import numpy as np

from donnee import *
from tqdm import trange
from math import factorial
ncols = 125                                                                       #largeur de la barre de chargement

def liA_mt(data):
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

def A_mt(data):
    """
    Retourne les coefficients de la décomposition des dérivées temporelles du vecteur U
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: li, fonction à évaluer
    """
    def li(t):
        g = liA_mt(data)(t)
        a = -g[0]
        b1, b2 = - g[1], g[0] @ g[0]
        c1, c2, c3 = - g[2], 3 * g[1] @ g[0], - g[0] @ g[0] @ g[0]
        d1, d2, d3, d4 = - g[3], 4*g[2] @ g[0] + 3*g[1] @ g[1], -6*g[1] @ g[0] @ g[0], g[0] @ g[0] @ g[0] @ g[0]
        zero = np.zeros((2,2))
        M = np.array([[a, zero, zero, zero],
                      [b1, b2, zero, zero],
                      [c1, c2, c3, zero],
                      [d1, d2, d3, d4]])
        return M
    return li

def ADER41D_mt(data):
    """
    Utilise le schéma d'ADER4 en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en quantité de mouvement et pression du problème 1D
    """
    data.U = np.zeros((data.N, data.M, 2))
    gamma = np.array([[1 / 12, 1 / 24, -1 / 12, -1 / 24],
                      [-2 / 3, -2 / 3, 1 / 6, 1 / 6],
                      [0, 5 / 4, 0, -1 / 4],
                      [2 / 3, -2 / 3, -1 / 6, 1 / 6],
                      [-1 / 12, 1 / 24, 1 / 12, -1 / 24]])

    for n in trange(data.N-1, ncols = ncols):
        for i in range(2,data.M-2):
            u_li = np.array([data.U[n,i-2,:],data.U[n,i-1,:], data.U[n,i,:], data.U[n,i+1,:],data.U[n,i+2,:]]).transpose()
            dxU = (np.array([(u_li @ gamma)[:, i] /data.dx**(i + 1) for i in range(4)])).reshape(4,2,1)
            dtU = np.einsum('ijab, jbk -> iak',A_mt(data)(n*data.dt),dxU).squeeze(-1)
            s = (data.dt / data.dx) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * np.array([data.opt, not data.opt])
            data.U[n + 1, i, :] = data.U[n, i, :] + sum([data.dt**(j + 1)/factorial(j + 1)*dtU[j, :] for j in range(4)]) + s

    for n in range(data.N):
        for i in range(data.M):
            data.U[n, i, :] = data.U[n, i, :]/(data.rho*(1-data.eps*np.sin(data.omega*n*data.dt)))
    return data.U