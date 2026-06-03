from donnee import *

def signe(a):
    return -1 * (a < 0) + 1 * (a >= 0)

def LaxWendroff1D(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, regroupe l'ensemble des données du problème avec la solution numérique
    """
    data.U = np.zeros((data.N, data.M, 2))
    A = np.array([[0, 1/data.rho], [data.rho * data.c**2, 0]])

    for n in range(0, data.N - 1):
        for i in range(1, data.M - 1):
            a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i+1, :] - data.U[n, i-1, :])
            a2 = (0.5 * (data.dt / data.dx)**2) * (A @ A) @ (data.U[n, i+1, :] + data.U[n, i-1, :] - 2 * data.U[n, i, :])
            data.U[n+1, i, :] = data.U[n, i, :] - a1 + a2 + data.dt/data.dx * data.S(data.f,n*data.dt) * (i == data.xs) * np.array([data.opt, not data.opt]).transpose()
    return data.U

def analytique1D(data):
    """
    Solution analytique pour le problème de propagation
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, regroupe l'ensemble des données du problème avec la solution analytique
    """
    data.U = np.zeros((data.N, data.M, 2))
    for n in range(1, data.N):
        for i in range(1, data.M-1):
            ind = n*data.dt - np.abs(i-data.xs)*data.dx/data.c
            a = 1/(2*data.c) * data.S(data.f, ind)
            b = signe(i - data.M//2) * data.rho/2 * data.S(data.f, ind)
            data.U[n, i, :] = np.array([a,b])
    return data.U