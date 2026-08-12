import numpy as np
from tqdm import trange
from time import sleep
ncols = 125                                                                       #largeur de la barre de chargement
from math import factorial
from modulation import *

#dérivées temporelles des matrices du probleme
def A1D_mt(data):
    def f(t):
        rho = data.rho*data.rho_mt(data, data.eps_r)(t)
        kappa = data.kappa*data.kappa_mt(data, data.eps_kappa)(t)
        A = np.array([[0, 1/rho[0]],
                      [kappa[0], 0]])
        B = np.array([[0,-rho[1]/rho[0]**2],
                      [kappa[1], 0]])
        C = np.array([[0,-(rho[2]*rho[0] - 2*rho[1]**2)/rho[0]**3],
                      [kappa[2], 0]])
        D = np.array([[0,-(rho[3]*rho[0]**2 - 6*rho[2]*rho[1]*rho[0] + 6*rho[1]**3)/rho[0]**4],
                      [kappa[3], 0]])
        return [A,B,C,D]
    return f

def A2D_mt(data):
    def f(t):
        rho = data.rho*data.rho_mt(data, data.eps_r)(t)
        kappa = data.kappa*data.kappa_mt(data, data.eps_kappa)(t)
        A = np.array([[0, 0, 1/rho[0]],
                      [0, 0, 0],
                      [kappa[0], 0, 0]])
        B = np.array([[0,0,-rho[1]/rho[0]**2],
                      [0, 0, 0],
                      [kappa[1], 0, 0]])
        C = np.array([[0, 0, (rho[2]*rho[0] - 2*rho[1]**2)/rho[0]**3],
                      [0, 0, 0],
                      [kappa[2], 0, 0]])
        D = np.array([[0, 0, -(rho[3]*rho[0]**2 - 6*rho[2]*rho[0]*rho[1] + 6*rho[1]**3)/rho[0]**4],
                      [0, 0, 0],
                      [kappa[3], 0, 0]])
        return [A,B,C,D]
    return f

def B2D_mt(data):
    def f(t):
        rho = data.rho*data.rho_mt(data, data.eps_r)(t)
        kappa = data.kappa*data.kappa_mt(data, data.eps_kappa)(t)
        A = np.array([[0, 0, 0],
                      [0, 0, 1/rho[0]],
                      [0, kappa[0], 0]])
        B = np.array([[0,0, 0],
                      [0, 0, -rho[1]/rho[0]**2],
                      [0, kappa[1], 0]])
        C = np.array([[0, 0, 0],
                      [0, 0, (rho[2]*rho[0] - 2*rho[1]**2)/rho[0]**3],
                      [0, kappa[2], 0]])
        D = np.array([[0, 0, 0],
                      [0, 0, -(rho[3]*rho[0]**2 - 6*rho[2]*rho[1]*rho[0] + 6*rho[1]**3)/rho[0]**4],
                      [0, kappa[3], 0]])
        return [A,B,C,D]
    return f

def c_mt(data):
    def f(t):
        rho = data.rho*data.rho_mt(data, data.eps_r)(t)
        kappa = data.kappa*data.kappa_mt(data, data.eps_kappa)(t)
        c = np.sqrt(kappa[0]/rho[0])
        c_ = 1/(2*c*rho[0]**2)*(kappa[1]*rho[0]-rho[1]*kappa[0])
        c__ = 1/(kappa[0]*rho[0])*((kappa[1]*rho[0]-rho[1]*kappa[1])*c_ - 2*c*((kappa[2]*rho[0] - rho[2]*kappa[0])*rho[0] - (kappa[1]*rho[0]-rho[1]*kappa[1])*rho[1])/rho[0])
        return [c, c_, c__]
    return f

"""
Modulation temporelle dans un problème de propagation 1D 
"""

def LaxWendroff1D_mt(data):
    """
    Utilise le schéma de LaxWendroff en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    sleep(0.01)
    print("\nLaxWendroff 1D Modulation Temporelle()")
    sleep(0.01)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.M, 2))

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data, data.eps_r)
        kappa = data.kappa_mt(data, data.eps_kappa)
        A, A_ = A1D_mt(data)(t)[0], A1D_mt(data)(t)[1]
        U_temp = data.U[n, :, :]
        U_temp_ = np.zeros(data.U[n, :, :].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0], [0, kappa(t)[1] / kappa(t)[0]]])
        d0 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)])
        U_temp[:] = U_temp * d0

        M1 = 0.5 / data.dx * (data.dt * A + data.dt**2 / 2 * A_)
        M2 = (0.5 * (data.dt / data.dx) ** 2) * (A @ A)
        diff = U_temp[2:, :] - U_temp[:-2, :]
        lap = U_temp[2:, :] + U_temp[:-2, :] - 2 * U_temp[1:-1, :]
        U_temp_[1:-1, :] = U_temp[1:-1, :] - (diff @ M1.T) + (lap @ M2.T)
        if 1 <= data.xs <= data.M - 2:
            U_temp_[data.xs, :] += (data.dt * data.rho / (data.dx * rho(t)[0])) * data.S(data.f, (n + 1) * data.dt) * np.array([data.opt, not data.opt])

        S_n = np.array([[-rho(t+data.dt)[1] / rho(t+data.dt)[0], 0], [0, kappa(t+data.dt)[1] / kappa(t+data.dt)[0]]])
        d1 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)])
        data.U[n + 1, :, :] = U_temp_ * d1

    rho = np.array([data.rho*data.rho_mt(data, data.eps_r)(data.dt * n)[0] for n in range(data.N)])
    kappa = np.array([data.kappa*data.kappa_mt(data, data.eps_kappa)(data.dt * n)[0] for n in range(data.N)])
    data.E = np.sum((0.5 * rho[:, None] * data.U[..., 0] ** 2 + data.U[..., 1]**2 / (2 * kappa[:, None])) * data.dx, axis=1)

def ADER41D_mt(data):
    """
    Utilise le schéma d'ADER4 en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    sleep(0.01)
    print("\nADER4 1D Modulation Temporelle()")
    sleep(0.01)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.M, 2))

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data, data.eps_r)
        kappa = data.kappa_mt(data, data.eps_kappa)
        g = A1D_mt(data)(t)
        U_temp = data.U[n, :, :]
        U_temp_ = np.zeros(data.U[n, :, :].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0], [0, kappa(t)[1] / kappa(t)[0]]])
        d0 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)])
        U_temp[:] = U_temp * d0

        def sh(p):
            return U_temp[2 + p:data.M - 2 + p, :]
        dxU = [1 / (12 * data.dx) * (sh(-2) - 8 * sh(-1) + 8 * sh(1) - sh(2)),
               1 / (12 * data.dx ** 2) * (-sh(-2) + 16 * sh(-1) - 30 * sh(0) + 16 * sh(1) - sh(2)),
               6 / (12 * data.dx ** 3) * (-sh(-2) + 2 * sh(-1) - 2 * sh(1) + sh(2)),
               - 1 / (data.dx ** 4) * (-sh(-2) + 4 * sh(-1) - 6 * sh(0) + 4 * sh(1) - sh(2))]

        a = -g[0]
        b1, b2 = - g[1], g[0] @ g[0]
        c1, c2, c3 = - g[2], 3 * g[1] @ g[0], - g[0] @ g[0] @ g[0]
        d1, d2, d3, d4 = - g[3], 4 * g[2] @ g[0] + 3 * g[2] @ g[2], -6 * g[1] @ g[0] @ g[0], g[0] @ g[0] @ g[0] @ g[0]
        dtU = [dxU[0] @ a.T,
               dxU[0] @ b1.T + dxU[1] @ b2.T,
               dxU[0] @ c1.T + dxU[1] @ c2.T + dxU[2] @ c3.T,
               dxU[0] @ d1.T + dxU[1] @ d2.T + dxU[2] @ d3.T + dxU[3] @ d4.T]
        U_temp_[2:-2, :] = U_temp[2:-2, :] + sum(data.dt ** (j + 1) / factorial(j + 1) * dtU[j] for j in range(4))
        if 2 <= data.xs <= data.M - 3:
            U_temp_[data.xs, :] += (data.dt * data.rho / (data.dx * rho(t)[0])) * data.S(data.f, (n + 1) * data.dt) * np.array([data.opt, not data.opt])

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, kappa(t + data.dt)[1] / kappa(t + data.dt)[0]]])
        d1e = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)])
        data.U[n + 1, :, :] = U_temp_ * d1e

    rho = np.array([data.rho*data.rho_mt(data, data.eps_r)(data.dt * n)[0] for n in range(data.N)])
    kappa = np.array([data.kappa*data.kappa_mt(data, data.eps_kappa)(data.dt * n)[0] for n in range(data.N)])
    data.E = np.sum((0.5 * rho[:, None] * data.U[..., 0] ** 2 + data.U[..., 1]**2 / (2*kappa[:, None])) * data.dx, axis=1)

"""
Résolution du problème de Cauchy 1D modulé en temps
"""

def LaxWendroff1D_cauchy_mt(data):
    """
    Utilise le schéma de LaxWendroff en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    sleep(0.01)
    print("\nLaxWendroff 1D Cauchy Modulation Temporelle()")
    sleep(0.01)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.M, 2))

    # init
    r = data.rho*data.rho_mt(data, data.eps_r)(0)[0]
    c = np.sqrt(data.kappa*data.kappa_mt(data, data.eps_kappa)(0)[0] / r)
    i = np.arange(data.M)
    arg = 1/data.f + data.tc[0] - data.dx * (i - 1) / c
    Si = np.array([data.S(data.f, a) for a in arg])
    data.U[0] = (1/c * Si)[:, None] * np.array([1, c*r])

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data, data.eps_r)
        kappa = data.kappa_mt(data, data.eps_kappa)
        A, A_ = A1D_mt(data)(t)[0], A1D_mt(data)(t)[1]
        U_temp = data.U[n, :, :]
        U_temp_ = np.zeros(data.U[n, :, :].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0], [0, kappa(t)[1] / kappa(t)[0]]])
        d0 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)])
        U_temp[:] = U_temp * d0

        M1 = 0.5 / data.dx * (data.dt * A + data.dt**2 / 2 * A_)
        M2 = (0.5 * (data.dt / data.dx) ** 2) * (A @ A)
        diff = U_temp[2:, :] - U_temp[:-2, :]
        lap = U_temp[2:, :] + U_temp[:-2, :] - 2 * U_temp[1:-1, :]
        U_temp_[1:-1, :] = U_temp[1:-1, :] - (diff @ M1.T) + (lap @ M2.T)

        S_n = np.array([[-rho(t+data.dt)[1] / rho(t+data.dt)[0], 0], [0, kappa(t+data.dt)[1] / kappa(t+data.dt)[0]]])
        d1 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)])
        data.U[n + 1, :, :] = U_temp_ * d1

    rho = np.array([data.rho*data.rho_mt(data, data.eps_r)(data.dt * n)[0] for n in range(data.N)])
    kappa = np.array([data.kappa*data.kappa_mt(data, data.eps_kappa)(data.dt * n)[0] for n in range(data.N)])
    data.E = np.sum((0.5 * rho[:, None] * data.U[..., 0] ** 2 + data.U[..., 1]**2 / (2*kappa[:, None])) * data.dx, axis=1)

def ADER41D_cauchy_mt(data):
    """
    Utilise le schéma d'ADER4 en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    sleep(0.01)
    print("\nADER4 1D Cauchy Modulation Temporelle()")
    sleep(0.01)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.M, 2))

    # init
    r = data.rho*data.rho_mt(data, data.eps_r)(0)[0]
    c = np.sqrt(data.kappa*data.kappa_mt(data, data.eps_kappa)(0)[0] / r)
    i = np.arange(data.M)
    arg = 1/data.f + data.tc[0] - data.dx * (i - 1) / c
    Si = np.array([data.S(data.f, a) for a in arg])
    data.U[0] = (1/c * Si)[:, None] * np.array([1, c*r])

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data, data.eps_r)
        kappa = data.kappa_mt(data, data.eps_kappa)
        g = A1D_mt(data)(t)
        U_temp = data.U[n, :, :]
        U_temp_ = np.zeros(data.U[n, :, :].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0], [0, kappa(t)[1] / kappa(t)[0]]])
        d0 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)])
        U_temp[:] = U_temp * d0

        def sh(p):
            return U_temp[2 + p:data.M - 2 + p, :]
        dxU = [1 / (12 * data.dx) * (sh(-2) - 8 * sh(-1) + 8 * sh(1) - sh(2)),
               1 / (12 * data.dx ** 2) * (-sh(-2) + 16 * sh(-1) - 30 * sh(0) + 16 * sh(1) - sh(2)),
               6 / (12 * data.dx ** 3) * (-sh(-2) + 2 * sh(-1) - 2 * sh(1) + sh(2)),
               - 1 / (data.dx ** 4) * (-sh(-2) + 4 * sh(-1) - 6 * sh(0) + 4 * sh(1) - sh(2))]

        a = -g[0]
        b1, b2 = - g[1], g[0] @ g[0]
        c1, c2, c3 = - g[2], 3 * g[1] @ g[0], - g[0] @ g[0] @ g[0]
        d1, d2, d3, d4 = - g[3], 4 * g[2] @ g[0] + 3 * g[2] @ g[2], -6 * g[1] @ g[0] @ g[0], g[0] @ g[0] @ g[0] @ g[0]
        dtU = [dxU[0] @ a.T,
               dxU[0] @ b1.T + dxU[1] @ b2.T,
               dxU[0] @ c1.T + dxU[1] @ c2.T + dxU[2] @ c3.T,
               dxU[0] @ d1.T + dxU[1] @ d2.T + dxU[2] @ d3.T + dxU[3] @ d4.T]
        U_temp_[2:-2, :] = U_temp[2:-2, :] + sum(data.dt ** (j + 1) / factorial(j + 1) * dtU[j] for j in range(4))

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, kappa(t + data.dt)[1] / kappa(t + data.dt)[0]]])
        d1e = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)])
        data.U[n + 1, :, :] = U_temp_ * d1e

    rho = np.array([data.rho*data.rho_mt(data, data.eps_r)(data.dt * n)[0] for n in range(data.N)])
    kappa = np.array([data.kappa*data.kappa_mt(data, data.eps_kappa)(data.dt * n)[0] for n in range(data.N)])
    data.E = np.sum((0.5 * rho[:, None] * data.U[..., 0] ** 2 + data.U[..., 1]**2 /(2*kappa[:, None])) * data.dx, axis=1)

"""
Résolution du problème de propagation 2D en milieu modulé en temps
"""

def LaxWendroff2D_mt(data):
    """
    Utilise le schéma de Lax Wendroff en 2D dans un mileu modulé en temps
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    sleep(0.01)
    print("\nLaxWendroff 2D Modulation Temporelle()")
    sleep(0.01)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.Mx, data.My, 3))

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data, data.eps_r)
        kappa = data.kappa_mt(data, data.eps_kappa)
        c = np.sqrt(data.kappa*kappa(t)[0]/(data.rho*rho(t)[0]))
        A, A_ = A2D_mt(data)(t)[0], A2D_mt(data)(t)[1]
        B, B_ = B2D_mt(data)(t)[0], B2D_mt(data)(t)[1]
        U_temp = data.U[n, ...]
        U_temp_ = np.zeros(data.U[n,...].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0,0],[0, -rho(t)[1] / rho(t)[0],0], [0,0, kappa(t)[1] / kappa(t)[0]]])
        d0 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)])
        U_temp[:] = U_temp * d0

        M1 = 0.5 / data.dx * (data.dt * A + data.dt ** 2 / 2 * A_)
        M1b = 0.5 / data.dy * (data.dt * B + data.dt ** 2 / 2 * B_)
        xp, xm = U_temp[2:, 1:-1, :], U_temp[:-2, 1:-1, :]
        yp, ym = U_temp[1:-1, 2:, :], U_temp[1:-1, :-2, :]
        c0 = U_temp[1:-1, 1:-1, :]
        a1 = (xp - xm) @ M1.T
        b1 = (yp - ym) @ M1b.T
        a2 = (0.5 * (data.dt * c / data.dx) ** 2) * (xp + xm - 2 * c0)
        b2 = (0.5 * (data.dt * c / data.dy) ** 2) * (yp + ym - 2 * c0)
        U_temp_[1:-1, 1:-1, :] = c0 - a1 - b1 + b2 + a2
        sval = data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * np.array([0, 0, 1]).transpose()
        for (pi, pj) in data.ps:
            if 1 <= pi <= data.Mx - 2 and 1 <= pj <= data.My - 2:
                U_temp_[pi, pj, :] += sval

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0, 0],[0, -rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, 0, kappa(t + data.dt)[1] / kappa(t + data.dt)[0]]])
        d1 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)])
        data.U[n + 1, ...] = U_temp_ * d1

    rho = [data.rho*data.rho_mt(data, data.eps_r)(data.dt * n)[0] for n in range(data.N)]
    kappa = [data.kappa*data.kappa_mt(data, data.eps_kappa)(data.dt * n)[0] for n in range(data.N)]
    data.E = [np.sum((0.5 * rho[n] * data.U[n, ..., 0] ** 2 + data.U[n,..., 1]**2 /(2*kappa[n]))*data.dx*data.dy, axis = (0,1)) for n in range(data.N)]

def ADER42D_mt(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D
    :param data: Donnee2D, regroupe l'ensemble des données du probleme 2D
    :return: np.ndarray(), solution en vitesse et pression du problème 2D
    """
    sleep(0.01)
    print("\nADER4 2D Modulation Temporelle()")
    sleep(0.01)
    data.U = np.zeros((data.N, data.Mx, data.My, 3))
    coeff = [1/(12*data.dx),1/(12*data.dx**2),1/(2*data.dx**3),1/(144*data.dx*data.dy**2),1/data.dx**4,1/(144*data.dx*data.dy),1/(144*(data.dx*data.dy)**2)]

    for n in trange(0, data.N - 1, ncols = ncols):
        t = n * data.dt
        rho = data.rho_mt(data, data.eps_r)
        kappa = data.kappa_mt(data, data.eps_kappa)
        A = A2D_mt(data)(t)
        B = B2D_mt(data)(t)
        c = c_mt(data)(t)
        U_temp = data.U[n, ...]
        U_temp_ = np.zeros(data.U[n, ...].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0, 0], [0, -rho(t)[1] / rho(t)[0], 0], [0, 0, kappa(t)[1] / kappa(t)[0]]])
        d0 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)])
        U_temp[:] = U_temp * d0

        def sh(pp, qq):
            return U_temp[2 + pp:data.Mx - 2 + pp, 2 + qq:data.My - 2 + qq, :]
        def ap(MM, d):
            return d @ MM.T
        dxU = [0,
               coeff[0] * (sh(-2,0) - 8*sh(-1,0) + 8*sh(1,0) - sh(2,0)),
               coeff[1] * (- sh(-2,0) + 16*sh(-1,0) - 30*sh(0,0) + 16*sh(1,0) - sh(2,0)),
               coeff[2] * (- sh(-2,0) + 2*sh(-1,0) - 2*sh(1,0) + sh(2,0)),
               coeff[4] * (sh(-2,0) - 4*sh(-1,0) + 6*sh(0,0) - 4*sh(1,0) + sh(2,0))]
        dyU = [0,
               coeff[0] * (sh(0,-2) - 8*sh(0,-1) + 8*sh(0,1) - sh(0,2)),
               coeff[1] * (- sh(0,-2) + 16*sh(0,-1) - 30*sh(0,0) + 16*sh(0,1) - sh(0,2)),
               coeff[2] * (- sh(0,-2) + 2*sh(0,-1) - 2*sh(0,1) + sh(0,2)),
               coeff[4] * (sh(0,-2) - 4*sh(0,-1) + 6*sh(0,0) - 4*sh(0,1) + sh(0,2))]
        dxyU = [coeff[5] * (sh(-2,-2) - 8*sh(-1,-2) + 8*sh(1,-2) - sh(2,-2)
                            - 8*sh(-2,-1) + 64*sh(-1,-1) - 64*sh(1,-1) + 8*sh(2,-1)
                            + 8*sh(-2,1) - 64*sh(-1,1) + 64*sh(1,1) - 8*sh(2,1)
                            - sh(-2,2) + 8*sh(-1,2) - 8*sh(1,2) + sh(2,2)),
                coeff[3] * (- sh(-2,-2) + 8*sh(-1,-2) - 8*sh(1,-2) + sh(2,-2)
                            + 16*sh(-2,-1) -128*sh(-1,-1) + 128*sh(1,-1) - 16*sh(2,-1)
                            - 30*sh(-2,0) + 240*sh(-1,0) - 240*sh(1,0) + 30*sh(2,0)
                            + 16*sh(-2,1) - 128*sh(-1,1) + 128*sh(1,1) - 16*sh(2,1)
                            - sh(-2,2) + 8*sh(-1,2) - 8*sh(1,2) + sh(2,2)),
                coeff[3] * (- sh(-2,-2) + 8*sh(-2,-1) - 8*sh(-2,1) + sh(-2,2)
                            + 16*sh(-1,-2) -128*sh(-1,-1) + 128*sh(-1,1) - 16*sh(-1,2)
                            - 30*sh(0,-2) + 240*sh(0,-1) - 240*sh(0,1) + 30*sh(0,2)
                            + 16*sh(1,-2) - 128*sh(1,-1) + 128*sh(1,1) - 16*sh(1,2)
                            - sh(2,-2) + 8*sh(2,-1) - 8*sh(2,1) + sh(2,2)),
                coeff[6] * (sh(-2,-2) -16*sh(-2,-1) +30*sh(-2,0) - 16*sh(-2,1) + sh(-2,2)
                            - 16*sh(-1,-2) +256*sh(-1,-1) -480*sh(-1,0) + 256*sh(-1,1) - 16*sh(-1,2)
                            + 30*sh(0,-2) -480*sh(0,-1) +900*sh(0,0) -480*sh(0,1) + 30*sh(0,2)
                            - 16*sh(1,-2) +256*sh(1,-1) -480*sh(1,0) + 256*sh(1,1) - 16*sh(1,2)
                            + sh(2,-2) -16*sh(2,-1) +30*sh(2,0) - 16*sh(2,1) + sh(2,2))]

        a1 = - data.dt * (ap(A[0], dxU[1]) + ap(B[0], dyU[1]))
        a2 = data.dt**2/2 * (- ap(A[1], dxU[1]) - ap(B[1], dyU[1]) + c[0]**2 * (dxU[2] + dyU[2]))
        a3 = data.dt**3/6 * (- c[0]**2 * (ap(A[0], dxU[3]) + ap(B[0], dyU[3]) + ap(A[0], dxyU[1]) + ap(B[0], dxyU[2]))
                             + 2*c[0]*c[1] * (dxU[2] + dyU[2]) - (ap(A[2], dxU[1]) + ap(B[2], dyU[1]))
                             + ap(A[1] @ A[0], dxU[2]) + ap(B[1] @ B[0], dyU[2]) + ap(A[1] @ B[0] + B[1] @ A[0], dxyU[0]))
        a4 = data.dt**4/24 * (- (ap(A[3], dxU[1]) + ap(B[3], dyU[1])) + 2 * (c[1]**2 + c[0] * c[2]) * (dxU[2] + dyU[2])
                              - 4 * c[0]*c[1]*(ap(A[0], dxU[3]) + ap(B[0], dyU[3]) + ap(B[0], dxyU[2]) + ap(A[0], dxyU[1]))
                              + 2 * (ap(A[2] @ A[0], dxU[2]) + ap(A[2] @ B[0] + B[2] @ A[0], dxyU[0]) + ap(B[2] @ B[0], dyU[2]))
                              + (ap(A[1] @ A[1], dxU[2]) + ap(B[1] @ B[1], dyU[2]) + ap(A[1] @ B[1] + B[1] @ A[1], dxyU[0]))
                              - (ap(A[1] @ A[0] @ A[0], dxU[3]) + ap(A[1] @ A[0] @ B[0] + B[1] @ A[0] @ A[0], dxyU[2])
                              + ap(B[1] @ B[0] @ B[0], dyU[3]) + ap(B[1] @ B[0] @ A[0] + A[1] @ B[0] @ B[0], dxyU[1]))
                              + c[0]**4*(dxU[4] + 2 * dxyU[3] + dyU[4]))

        U_temp_[2:-2, 2:-2, :] = U_temp[2:-2, 2:-2, :] + a1 + a2 + a3 + a4
        sval = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * np.array([0, 0, 1]).transpose()
        for (pi, pj) in data.ps:
            if 2 <= pi <= data.Mx - 3 and 2 <= pj <= data.My - 3:
                U_temp_[pi, pj, :] += sval

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0, 0],[0, -rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, 0, kappa(t + data.dt)[1] / kappa(t + data.dt)[0]]])
        d1 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)])
        data.U[n + 1, ...] = U_temp_ * d1

    rho = [data.rho*data.rho_mt(data, data.eps_r)(data.dt * n)[0] for n in range(data.N)]
    kappa = [data.kappa*data.kappa_mt(data, data.eps_kappa)(data.dt * n)[0] for n in range(data.N)]
    data.E = [np.sum((0.5 * rho[n] * data.U[n, ..., 0] ** 2 + data.U[n,..., 1]**2 /(2*kappa[n]))*data.dx*data.dy, axis = (0,1)) for n in range(data.N)]

"""
Résolution du problème de Cauchy 2D modulé en temps
"""

def LaxWendroff2D_cauchy_mt(data):
    """
    Utilise le schéma de Lax Wendroff en 2D dans un mileu modulé en temps
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    sleep(0.01)
    print("\nLax Wendroff 2D Cauchy Modulation Temporelle()")
    sleep(0.01)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.Mx + 2, data.My, 3))

    # init
    r = data.rho*data.rho_mt(data, data.eps_r)(0)[0]
    c = np.sqrt(data.kappa*data.kappa_mt(data, data.eps_kappa)(0)[0] / r)
    j = np.arange(data.My)
    Sj = np.array([data.S(data.f, 1/data.f + data.tc[0] - data.dy * (jj - 1) / c) for jj in j])
    data.U[0, :, :, :] = (1/c) * Sj[None, :, None] * np.array([0, 1, r * c])

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data, data.eps_r)
        kappa = data.kappa_mt(data, data.eps_kappa)
        c = np.sqrt(data.kappa*kappa(t)[0]/(data.rho*rho(t)[0]))
        A, A_ = A2D_mt(data)(t)[0], A2D_mt(data)(t)[1]
        B, B_ = B2D_mt(data)(t)[0], B2D_mt(data)(t)[1]
        U_temp = np.zeros((data.Mx + 2, data.My, 3))
        U_temp_ = np.zeros((data.Mx + 2, data.My, 3))

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0,0],[0, -rho(t)[1] / rho(t)[0],0], [0,0, kappa(t)[1] / kappa(t)[0]]])
        d0 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)])
        U_temp = data.U[n] * d0

        MA = data.dt * A + data.dt ** 2 / 2 * A_
        MB = data.dt * B + data.dt ** 2 / 2 * B_
        xp = np.roll(U_temp, -1, axis=0)[:, 1:-1, :]     # U[i+1, j] (periodique en x)
        xm = np.roll(U_temp, 1, axis=0)[:, 1:-1, :]      # U[i-1, j]
        c0 = U_temp[:, 1:-1, :]                            # U[i, j]
        yp = U_temp[:, 2:, :]                              # U[i, j+1]
        ym = U_temp[:, :-2, :]                             # U[i, j-1]
        a1 = (1 / (2 * data.dx)) * ((xp - xm) @ MA.T)
        a2 = (1 / (2 * data.dy)) * ((yp - ym) @ MB.T)
        b1 = (0.5 * (data.dt * c) ** 2) * (xp + xm - 2 * c0) / data.dx ** 2
        b2 = (0.5 * (data.dt * c) ** 2) * (yp + ym - 2 * c0) / data.dy ** 2
        U_temp_[:, 1:-1, :] = c0 - a1 - a2 + b1 + b2
        # bord gauche (indice 0) : b1, b2 utilisent data.c (et non c), comme dans l'original
        b1_0 = (0.5 * (data.dt * data.c) ** 2) * (xp[0] + xm[0] - 2 * c0[0]) / data.dx ** 2
        b2_0 = (0.5 * (data.dt * data.c) ** 2) * (yp[0] + ym[0] - 2 * c0[0]) / data.dy ** 2
        U_temp_[0, 1:-1, :] = c0[0] - a1[0] - a2[0] + b1_0 + b2_0

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0, 0],[0, -rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, 0, kappa(t + data.dt)[1] / kappa(t + data.dt)[0]]])
        d1 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)])
        data.U[n + 1, ...] = U_temp_ * d1

    data.U = data.U[:, 1:data.Mx + 1, :, :]
    rho = [data.rho*data.rho_mt(data, data.eps_r)(data.dt * n)[0] for n in range(data.N)]
    kappa = [data.kappa*data.kappa_mt(data, data.eps_kappa)(data.dt * n)[0] for n in range(data.N)]
    data.kappa = [np.sum((0.5 * rho[n] * data.U[n, ..., 0] ** 2 + data.U[n,..., 1]**2 /(2*kappa[n]))*data.dx*data.dy, axis = (0,1)) for n in range(data.N)]

def ADER42D_cauchy_mt(data):
    """
    Utilise le schéma de ADER4 en 2D dans un mileu modulé en temps
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    sleep(0.01)
    print("\nADER4 2D Cauchy Modulation Temporelle()")
    sleep(0.01)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.Mx + 4, data.My, 3))

    # init
    r = data.rho*data.rho_mt(data, data.eps_r)(0)[0]
    c = np.sqrt(data.kappa*data.kappa_mt(data, data.eps_kappa)(0)[0] / r)
    j = np.arange(data.My)
    Sj = np.array([data.S(data.f, 1/data.f + data.tc[0] - data.dy * (jj - 1) / c) for jj in j])
    data.U[0, :, :, :] = (1/c) * Sj[None, :, None] * np.array([0, 1, r * c])

    coeff = [1/(12*data.dx),1/(12*data.dx**2),1/(2*data.dx**3),1/(144*data.dx*data.dy**2),1/data.dx**4,1/(144*data.dx*data.dy),1/(144*(data.dx*data.dy)**2)]
    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data, data.eps_r)
        kappa = data.kappa_mt(data, data.eps_kappa)
        A = A2D_mt(data)(t)
        B = B2D_mt(data)(t)
        c = c_mt(data)(t)
        U_temp = np.zeros((data.Mx + 4, data.My, 3))
        U_temp_ = np.zeros((data.Mx + 4, data.My, 3))

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0,0],[0, -rho(t)[1] / rho(t)[0],0], [0,0, kappa(t)[1] / kappa(t)[0]]])
        d0 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)])
        U_temp = data.U[n] * d0

        def sh(pp, qq):
            return np.roll(U_temp, -pp, axis=0)[:, 2 + qq:data.My - 2 + qq, :]
        def ap(MM, d):
            return d @ MM.T
        dxU = [0,
               coeff[0] * (sh(-2,0) - 8*sh(-1,0) + 8*sh(1,0) - sh(2,0)),
               coeff[1] * (- sh(-2,0) + 16*sh(-1,0) - 30*sh(0,0) + 16*sh(1,0) - sh(2,0)),
               coeff[2] * (- sh(-2,0) + 2*sh(-1,0) - 2*sh(1,0) + sh(2,0)),
               coeff[4] * (sh(-2,0) - 4*sh(-1,0) + 6*sh(0,0) - 4*sh(1,0) + sh(2,0))]
        dyU = [0,
               coeff[0] * (sh(0,-2) - 8*sh(0,-1) + 8*sh(0,1) - sh(0,2)),
               coeff[1] * (- sh(0,-2) + 16*sh(0,-1) - 30*sh(0,0) + 16*sh(0,1) - sh(0,2)),
               coeff[2] * (- sh(0,-2) + 2*sh(0,-1) - 2*sh(0,1) + sh(0,2)),
               coeff[4] * (sh(0,-2) - 4*sh(0,-1) + 6*sh(0,0) - 4*sh(0,1) + sh(0,2))]
        dxyU = [coeff[5] * (sh(-2,-2) - 8*sh(-1,-2) + 8*sh(1,-2) - sh(2,-2)
                            - 8*sh(-2,-1) + 64*sh(-1,-1) - 64*sh(1,-1) + 8*sh(2,-1)
                            + 8*sh(-2,1) - 64*sh(-1,1) + 64*sh(1,1) - 8*sh(2,1)
                            - sh(-2,2) + 8*sh(-1,2) - 8*sh(1,2) + sh(2,2)),
                coeff[3] * (- sh(-2,-2) + 8*sh(-1,-2) - 8*sh(1,-2) + sh(2,-2)
                            + 16*sh(-2,-1) -128*sh(-1,-1) + 128*sh(1,-1) - 16*sh(2,-1)
                            - 30*sh(-2,0) + 240*sh(-1,0) - 240*sh(1,0) + 30*sh(2,0)
                            + 16*sh(-2,1) - 128*sh(-1,1) + 128*sh(1,1) - 16*sh(2,1)
                            - sh(-2,2) + 8*sh(-1,2) - 8*sh(1,2) + sh(2,2)),
                coeff[3] * (- sh(-2,-2) + 8*sh(-2,-1) - 8*sh(-2,1) + sh(-2,2)
                            + 16*sh(-1,-2) -128*sh(-1,-1) + 128*sh(-1,1) - 16*sh(-1,2)
                            - 30*sh(0,-2) + 240*sh(0,-1) - 240*sh(0,1) + 30*sh(0,2)
                            + 16*sh(1,-2) - 128*sh(1,-1) + 128*sh(1,1) - 16*sh(1,2)
                            - sh(2,-2) + 8*sh(2,-1) - 8*sh(2,1) + sh(2,2)),
                coeff[6] * (sh(-2,-2) -16*sh(-2,-1) +30*sh(-2,0) - 16*sh(-2,1) + sh(-2,2)
                            - 16*sh(-1,-2) +256*sh(-1,-1) -480*sh(-1,0) + 256*sh(-1,1) - 16*sh(-1,2)
                            + 30*sh(0,-2) -480*sh(0,-1) +900*sh(0,0) -480*sh(0,1) + 30*sh(0,2)
                            - 16*sh(1,-2) +256*sh(1,-1) -480*sh(1,0) + 256*sh(1,1) - 16*sh(1,2)
                            + sh(2,-2) -16*sh(2,-1) +30*sh(2,0) - 16*sh(2,1) + sh(2,2))]

        a1 = - data.dt * (ap(A[0], dxU[1]) + ap(B[0], dyU[1]))
        a2 = data.dt**2/2 * (- ap(A[1], dxU[1]) - ap(B[1], dyU[1]) + c[0]**2 * (dxU[2] + dyU[2]))
        a3 = data.dt**3/6 * (- c[0]**2 * (ap(A[0], dxU[3]) + ap(B[0], dyU[3]) + ap(A[0], dxyU[1]) + ap(B[0], dxyU[2]))
                             + 2*c[0]*c[1] * (dxU[2] + dyU[2]) - (ap(A[2], dxU[1]) + ap(B[2], dyU[1]))
                             + ap(A[1] @ A[0], dxU[2]) + ap(B[1] @ B[0], dyU[2]) + ap(A[1] @ B[0] + B[1] @ A[0], dxyU[0]))
        a4 = data.dt**4/24 * (- (ap(A[3], dxU[1]) + ap(B[3], dyU[1])) + 2 * (c[1]**2 + c[0] * c[2]) * (dxU[2] + dyU[2])
                              - 4 * c[0]*c[1]*(ap(A[0], dxU[3]) + ap(B[0], dyU[3]) + ap(B[0], dxyU[2]) + ap(A[0], dxyU[1]))
                              + 2 * (ap(A[2] @ A[0], dxU[2]) + ap(A[2] @ B[0] + B[2] @ A[0], dxyU[0]) + ap(B[2] @ B[0], dyU[2]))
                              + (ap(A[1] @ A[1], dxU[2]) + ap(B[1] @ B[1], dyU[2]) + ap(A[1] @ B[1] + B[1] @ A[1], dxyU[0]))
                              - (ap(A[1] @ A[0] @ A[0], dxU[3]) + ap(A[1] @ A[0] @ B[0] + B[1] @ A[0] @ A[0], dxyU[2])
                              + ap(B[1] @ B[0] @ B[0], dyU[3]) + ap(B[1] @ B[0] @ A[0] + A[1] @ B[0] @ B[0], dxyU[1]))
                              + c[0]**4*(dxU[4] + 2 * dxyU[3] + dyU[4]))

        U_temp_[:, 2:-2, :] = sh(0,0) + a1 + a2 + a3 + a4

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0, 0],[0, -rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, 0, kappa(t + data.dt)[1] / kappa(t + data.dt)[0]]])
        d1 = np.array([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)])
        data.U[n + 1, ...] = U_temp_ * d1

    data.U = data.U[:, 2:data.Mx + 2, :, :]
    rho = [data.rho*data.rho_mt(data, data.eps_r)(data.dt * n)[0] for n in range(data.N)]
    kappa = [data.kappa*data.kappa_mt(data, data.eps_kappa)(data.dt * n)[0] for n in range(data.N)]
    data.E = [np.sum((0.5 * rho[n] * data.U[n, ..., 0] ** 2 + data.U[n,..., 1]**2 /(2*kappa[n]))*data.dx*data.dy, axis = (0,1)) for n in range(data.N)]
