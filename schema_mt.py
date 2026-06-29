import numpy as np
from tqdm import trange
from time import sleep
ncols = 125                                                                       #largeur de la barre de chargement
from math import factorial
from modulation import *

#dérivées temporelles des matrices du probleme
def A1D_mt(data):
    def f(t):
        rho = data.rho_mt(data)(t)
        E = data.E_mt(data)(t)

        A = np.array([[0, 1/rho[0]],
                      [E[0], 0]])
        B = np.array([[0,-rho[1]/rho[0]**2],
                      [E[1], 0]])
        C = np.array([[0,-(rho[2]*rho[0] - 2*rho[1]**2)/rho[0]**3],
                      [E[2], 0]])
        D = np.array([[0,-(rho[3]*rho[0]**2 - 6*rho[2]*rho[1]*rho[0] + 6*rho[1]**3)/rho[0]**4],
                      [E[3], 0]])
        return [A,B,C,D]
    return f

def A2D_mt(data):
    def f(t):
        rho = data.rho_mt(data)(t)
        E = data.E_mt(data)(t)
        A = np.array([[0, 0, 1/rho[0]],
                      [0, 0, 0],
                      [E[0], 0, 0]])
        B = np.array([[0,0,-rho[1]/rho[0]**2],
                      [0, 0, 0],
                      [E[1], 0, 0]])
        C = np.array([[0, 0, (rho[2]*rho[0] - 2*rho[1]**2)/rho[0]**3],
                      [0, 0, 0],
                      [E[2], 0, 0]])
        D = np.array([[0, 0, -(rho[3]*rho[0]**2 - 6*rho[2]*rho[0]*rho[1] + 6*rho[1]**3)/rho[0]**4],
                      [0, 0, 0],
                      [E[3], 0, 0]])
        return [A,B,C,D]
    return f

def B2D_mt(data):
    def f(t):
        rho = data.rho_mt(data)(t)
        E = data.E_mt(data)(t)
        A = np.array([[0, 0, 0],
                      [0, 0, 1/rho[0]],
                      [0, E[0], 0]])
        B = np.array([[0,0, 0],
                      [0, 0, -rho[1]/rho[0]**2],
                      [0, E[1], 0]])
        C = np.array([[0, 0, 0],
                      [0, 0, (rho[2]*rho[0] - 2*rho[1]**2)/rho[0]**3],
                      [0, E[2], 0]])
        D = np.array([[0, 0, 0],
                      [0, 0, -(rho[3]*rho[0]**2 - 6*rho[2]*rho[1]*rho[0] + 6*rho[1]**3)/rho[0]**4],
                      [0, E[3], 0]])
        return [A,B,C,D]
    return f

def c_mt(data):
    def f(t):
        rho = data.rho_mt(data)(t)
        E = data.E_mt(data)(t)
        c = np.sqrt(E[0]/rho[0])
        c_ = 1/(2*c*rho[0]**2)*(E[1]*rho[0]-rho[1]*E[0])
        c__ = 1/(E[0]*rho[0])*((E[1]*rho[0]-rho[1]*E[1])*c_ - 2*c*((E[2]*rho[0] - rho[2]*E[0])*rho[0] - (E[1]*rho[0]-rho[1]*E[1])*rho[1])/rho[0])
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
    print("LaxWendroff 1D mt()")
    sleep(0.001)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.M, 2))

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data)
        E = data.E_mt(data)
        A, A_ = A1D_mt(data)(t)[0], A1D_mt(data)(t)[1]
        U_temp = data.U[n, :, :]
        U_temp_ = np.zeros(data.U[n, :, :].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0], [0, E(t)[1] / E(t)[0]]])
        for i in range(0, data.M):
            U_temp[i, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)]) @ data.U[n, i, :]

        for i in range(1, data.M - 1):
            s = (data.dt * data.rho/ (data.dx * rho(t)[0]) ) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * np.array([data.opt, not data.opt])
            a1 = 0.5 / data.dx * (data.dt * A + data.dt**2 /2 * A_) @ (U_temp[i + 1, :] - U_temp[i - 1, :])
            a2 = (0.5 * (data.dt / data.dx) ** 2) * (A @ A) @ (U_temp[i + 1, :] + U_temp[i - 1, :] - 2 * U_temp[i, :])
            U_temp_[i, :] = U_temp[i, :] - a1 + a2 + s

        S_n = np.array([[-rho(t+data.dt)[1] / rho(t+data.dt)[0], 0], [0, E(t+data.dt)[1] / E(t+data.dt)[0]]])
        for i in range(data.M):
            data.U[n + 1, i, :] = np.diag([np.exp(-S_n[0,0]*data.dt/2), np.exp(-S_n[1,1]*data.dt/2)]) @ U_temp_[i, :]

    rho = [data.rho_mt(data)(data.dt * n)[0] for n in range(data.N)]
    E = [data.E_mt(data)(data.dt * n)[0] for n in range(data.N)]
    data.E = np.array([sum([0.5 * rho[n] * data.U[n, i, 0] ** 2 + data.U[n, i, 1]**2 / E[n] for i in range(data.M)]) for n in range(0, data.N)])

def ADER41D_mt(data):
    """
    Utilise le schéma d'ADER4 en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    print("ADER4 1D mt()")
    sleep(0.001)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.M, 2))

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data)
        E = data.E_mt(data)
        g = A1D_mt(data)(t)
        U_temp = data.U[n, :, :]
        U_temp_ = np.zeros(data.U[n, :, :].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0], [0, E(t)[1] / E(t)[0]]])
        for i in range(0, data.M):
            U_temp[i, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)]) @ data.U[n, i, :]

        for i in range(2, data.M-2):
            s = (data.dt * data.rho / (data.dx * rho(t)[0]) ) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * np.array([data.opt, not data.opt])
            dxU = np.array([1 / (12 * data.dx) * (U_temp[i - 2, :] - 8 * U_temp[i - 1, :] + 8 * U_temp[i + 1, :] - U_temp[i + 2, :]),
                        1 / (12 * data.dx ** 2) * (-U_temp[i - 2, :] + 16 * U_temp[i - 1, :] - 30 * U_temp[i, :] + 16 * U_temp[i + 1, :] - U_temp[i + 2, :]),
                        6 / (12 * data.dx ** 3) * (-U_temp[i - 2, :] + 2 * U_temp[i - 1, :] - 2 * U_temp[i + 1, :] + U_temp[i + 2, :]),
                        - 1 / (data.dx ** 4) * (-U_temp[i - 2, :] + 4 * U_temp[i - 1, :] - 6 * U_temp[i, :] + 4 * U_temp[i + 1, :] - U_temp[i + 2, :])])

            a = -g[0]
            b1, b2 = - g[1], g[0] @ g[0]
            c1, c2, c3 = - g[2], 3 * g[1] @ g[0], - g[0] @ g[0] @ g[0]
            d1, d2, d3, d4 = - g[3], 4 * g[2] @ g[0] + 3 * g[2] @ g[2], -6 * g[1] @ g[0] @ g[0], g[0] @ g[0] @ g[0] @ g[0]
            dtU = np.array([a @ dxU[0], b1 @ dxU[0] + b2 @ dxU[1], c1 @ dxU[0] + c2 @ dxU[1] + c3 @ dxU[2], d1 @ dxU[0] + d2 @ dxU[1] + d3 @ dxU[2] + d4 @ dxU[3]])
            U_temp_[i,:] = U_temp[i, :] + sum([data.dt ** (j + 1) / factorial(j + 1) * dtU[j, :] for j in range(4)]) + s

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, E(t + data.dt)[1] / E(t + data.dt)[0]]])
        for i in range(data.M):
            data.U[n + 1, i, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)]) @ U_temp_[i, :]

    rho = [data.rho_mt(data)(data.dt * n)[0] for n in range(data.N)]
    E = [data.E_mt(data)(data.dt * n)[0] for n in range(data.N)]
    data.E = data.dx*np.array([sum([0.5 * rho[n] * data.U[n, i, 0] ** 2 + data.U[n, i, 1]**2 / E[n] for i in range(data.M)]) for n in range(0, data.N)])

"""
Résolution du problème de Cauchy 1D modulé en temps
"""

def LaxWendroff1D_cauchy_mt(data):
    """
    Utilise le schéma de LaxWendroff en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    print("LaxWendroff 1D Cauchy mt()")
    sleep(0.001)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.M, 2))

    # init
    for i in range(0, data.M):
        r = data.rho_mt(data)(0)[0]
        c = np.sqrt(data.E_mt(data)(0)[0]/r)
        data.U[0,i,:] = 1/c * data.S(data.f, 1/data.f +  data.tc[0] - data.dx*i/c) * np.array([1, c*r])

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data)
        E = data.E_mt(data)
        A, A_ = A1D_mt(data)(t)[0], A1D_mt(data)(t)[1]
        U_temp = data.U[n, :, :]
        U_temp_ = np.zeros(data.U[n, :, :].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0], [0, E(t)[1] / E(t)[0]]])
        for i in range(0, data.M):
            U_temp[i, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)]) @ data.U[n, i, :]

        for i in range(1, data.M - 1):
            a1 = 0.5 / data.dx * (data.dt * A + data.dt**2 /2 * A_) @ (U_temp[i + 1, :] - U_temp[i - 1, :])
            a2 = (0.5 * (data.dt / data.dx) ** 2) * (A @ A) @ (U_temp[i + 1, :] + U_temp[i - 1, :] - 2 * U_temp[i, :])
            U_temp_[i, :] = U_temp[i, :] - a1 + a2

        S_n = np.array([[-rho(t+data.dt)[1] / rho(t+data.dt)[0], 0], [0, E(t+data.dt)[1] / E(t+data.dt)[0]]])
        for i in range(data.M):
            data.U[n + 1, i, :] = np.diag([np.exp(-S_n[0,0]*data.dt/2), np.exp(-S_n[1,1]*data.dt/2)]) @ U_temp_[i, :]

    rho = [data.rho_mt(data)(data.dt * n)[0] for n in range(data.N)]
    E = [data.E_mt(data)(data.dt * n)[0] for n in range(data.N)]
    data.E = np.array([sum([0.5 * rho[n] * data.U[n, i, 0] ** 2 + data.U[n, i, 1]**2 / E[n] for i in range(data.M)]) for n in range(0, data.N)])

def ADER41D_cauchy_mt(data):
    """
    Utilise le schéma d'ADER4 en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    print("ADER4 1D Cauchy mt()")
    sleep(0.001)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.M, 2))

    # init
    for i in range(0, data.M):
        r = data.rho_mt(data)(0)[0]
        c = np.sqrt(data.E_mt(data)(0)[0]/r)
        data.U[0,i,:] = 1/c * data.S(data.f, 1/data.f +  data.tc[0] - data.dx*i/c) * np.array([1, c*r])

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data)
        E = data.E_mt(data)
        g = A1D_mt(data)(t)
        U_temp = data.U[n, :, :]
        U_temp_ = np.zeros(data.U[n, :, :].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0], [0, E(t)[1] / E(t)[0]]])
        for i in range(0, data.M):
            U_temp[i, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)]) @ data.U[n, i, :]

        for i in range(2, data.M-2):
            dxU = np.array([1 / (12 * data.dx) * (U_temp[i - 2, :] - 8 * U_temp[i - 1, :] + 8 * U_temp[i + 1, :] - U_temp[i + 2, :]),
                        1 / (12 * data.dx ** 2) * (-U_temp[i - 2, :] + 16 * U_temp[i - 1, :] - 30 * U_temp[i, :] + 16 * U_temp[i + 1, :] - U_temp[i + 2, :]),
                        6 / (12 * data.dx ** 3) * (-U_temp[i - 2, :] + 2 * U_temp[i - 1, :] - 2 * U_temp[i + 1, :] + U_temp[i + 2, :]),
                        - 1 / (data.dx ** 4) * (-U_temp[i - 2, :] + 4 * U_temp[i - 1, :] - 6 * U_temp[i, :] + 4 * U_temp[i + 1, :] - U_temp[i + 2, :])])

            a = -g[0]
            b1, b2 = - g[1], g[0] @ g[0]
            c1, c2, c3 = - g[2], 3 * g[1] @ g[0], - g[0] @ g[0] @ g[0]
            d1, d2, d3, d4 = - g[3], 4 * g[2] @ g[0] + 3 * g[2] @ g[2], -6 * g[1] @ g[0] @ g[0], g[0] @ g[0] @ g[0] @ g[0]
            dtU = np.array([a @ dxU[0], b1 @ dxU[0] + b2 @ dxU[1], c1 @ dxU[0] + c2 @ dxU[1] + c3 @ dxU[2], d1 @ dxU[0] + d2 @ dxU[1] + d3 @ dxU[2] + d4 @ dxU[3]])
            U_temp_[i,:] = U_temp[i, :] + sum([data.dt ** (j + 1) / factorial(j + 1) * dtU[j, :] for j in range(4)])

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, E(t + data.dt)[1] / E(t + data.dt)[0]]])
        for i in range(data.M):
            data.U[n + 1, i, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)]) @ U_temp_[i, :]

    rho = [data.rho_mt(data)(data.dt * n)[0] for n in range(data.N)]
    E = [data.E_mt(data)(data.dt * n)[0] for n in range(data.N)]
    data.E = data.dx*np.array([sum([0.5 * rho[n] * data.U[n, i, 0] ** 2 + data.U[n, i, 1]**2 / E[n] for i in range(data.M)]) for n in range(0, data.N)])

"""
Résolution du problème de propagation 2D en mil
"""

def LaxWendroff2D_mt(data):
    """
    Utilise le schéma de Lax Wendroff en 2D dans un mileu modulé en temps
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    print("LaxWendroff 2D mt()")
    sleep(0.001)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.Mx, data.My, 3))

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data)
        E = data.E_mt(data)
        c = np.sqrt(E(t)[0]/rho(t)[0])
        A, A_ = A2D_mt(data)(t)[0], A2D_mt(data)(t)[1]
        B, B_ = B2D_mt(data)(t)[0], B2D_mt(data)(t)[1]
        U_temp = data.U[n, ...]
        U_temp_ = np.zeros(data.U[n,...].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0,0],[0, -rho(t)[1] / rho(t)[0],0], [0,0, E(t)[1] / E(t)[0]]])
        for i in range(0, data.Mx):
            for j in range(0, data.My):
                U_temp[i, j, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)]) @ data.U[n, i, j, :]

        for i in range(1, data.Mx - 1):
            for j in range(1, data.My - 1):
                s = data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * ((i,j) in data.ps) * np.array([0, 0, 1]).transpose()
                a1 = 0.5 / data.dx * (data.dt * A + data.dt ** 2 / 2 * A_) @ (U_temp[i + 1, j, :] - U_temp[i - 1, j, :])
                b1 = 0.5 / data.dy * (data.dt * B + data.dt ** 2 / 2 * B_) @ (U_temp[i, j + 1, :] - U_temp[i, j - 1, :])

                a2 = (0.5 * (data.dt * c/ data.dx) ** 2) * (U_temp[i + 1, j, :] + U_temp[i - 1, j, :] - 2 * U_temp[i, j, :])
                b2 = (0.5 * (data.dt * c/ data.dy) ** 2) * (U_temp[i, j + 1, :] + U_temp[i, j - 1, :] - 2 * U_temp[i, j, :])

                U_temp_[i, j, :] = U_temp[i, j, :] - a1 - b1 + b2 + a2 + s

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0, 0],[0, -rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, 0, E(t + data.dt)[1] / E(t + data.dt)[0]]])
        for i in range(data.Mx):
            for j in range(data.My):
                data.U[n + 1, i, j, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)]) @ U_temp_[i, j, :]

        data.calcul_energie()

def ADER42D_mt(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D
    :param data: Donnee2D, regroupe l'ensemble des données du probleme 2D
    :return: np.ndarray(), solution en vitesse et pression du problème 2D
    """
    print("ADER4 2D mt()")
    sleep(0.001)
    data.U = np.zeros((data.N, data.Mx, data.My, 3))
    coeff = [1/(12*data.dx),1/(12*data.dx**2),1/(2*data.dx**3),1/(144*data.dx*data.dy**2),1/data.dx**4,1/(144*data.dx*data.dy),1/(144*(data.dx*data.dy)**2)]

    for n in trange(0, data.N - 1, ncols = ncols):
        t = n * data.dt
        rho = data.rho_mt(data)
        E = data.E_mt(data)
        A = A2D_mt(data)(t)
        B = B2D_mt(data)(t)
        c = c_mt(data)(t)
        U_temp = data.U[n, ...]
        U_temp_ = np.zeros(data.U[n, ...].shape)

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0, 0], [0, -rho(t)[1] / rho(t)[0], 0], [0, 0, E(t)[1] / E(t)[0]]])
        for i in range(0, data.Mx):
            for j in range(0, data.My):
                U_temp[i, j, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)]) @ data.U[n, i, j, :]

        for i in range(2, data.Mx - 2):
            for j in range(2, data.My - 2):
                dxU = [0,
                       coeff[0] * (U_temp[i-2,j,:] - 8*U_temp[i-1,j,:] + 8*U_temp[i+1,j,:] - U_temp[i+2,j,:]),
                       coeff[1] * (- U_temp[i-2,j,:] + 16*U_temp[i-1,j,:] - 30*U_temp[i,j,:] + 16*U_temp[i+1,j,:] - U_temp[i+2,j,:]),
                       coeff[2] * (- U_temp[i-2,j,:] + 2*U_temp[i-1,j,:] - 2*U_temp[i+1,j,:] + U_temp[i+2,j,:]),
                       coeff[4] * (U_temp[i-2,j,:] - 4*U_temp[i-1,j,:] + 6*U_temp[i,j,:] - 4*U_temp[i+1,j,:] + U_temp[i+2,j,:])]

                dyU = [0,
                       coeff[0] * (U_temp[i,j-2,:] - 8*U_temp[i,j-1,:] + 8*U_temp[i,j+1,:] - U_temp[i,j+2,:]),
                       coeff[1] * (- U_temp[i,j-2,:] + 16*U_temp[i,j-1,:] - 30*U_temp[i,j,:] + 16*U_temp[i,j+1,:] - U_temp[i,j+2,:]),
                       coeff[2] * (- U_temp[i,j-2,:] + 2*U_temp[i,j-1,:] - 2*U_temp[i,j+1,:] + U_temp[i,j+2,:]),
                       coeff[4] * (U_temp[i,j-2,:] - 4*U_temp[i,j-1,:] + 6*U_temp[i,j,:] - 4*U_temp[i,j+1,:] + U_temp[i,j+2,:])]

                dxyU = [coeff[5] * (U_temp[i-2, j-2,:] - 8*U_temp[i-1,j-2,:] + 8*U_temp[i+1,j-2,:] - U_temp[i+2,j-2,:]                                    #dxyU
                                    - 8*U_temp[i-2, j-1,:] + 64*U_temp[i-1,j-1,:] - 64*U_temp[i+1,j-1,:] + 8*U_temp[i+2,j-1,:]
                                    + 8*U_temp[i-2, j+1,:] - 64*U_temp[i-1,j+1,:] + 64*U_temp[i+1,j+1,:] - 8*U_temp[i+2,j+1,:]
                                    - U_temp[i-2, j+2,:] + 8*U_temp[i-1,j+2,:] - 8*U_temp[i+1,j+2,:] + U_temp[i+2,j+2,:]),
                        coeff[3] * (- U_temp[i-2,j-2,:] + 8*U_temp[i-1,j-2,:] - 8*U_temp[i+1,j-2,:] + U_temp[i+2,j-2,:]                                  #dxyyU
                                    + 16*U_temp[i-2,j-1,:] -128*U_temp[i-1,j-1,:] + 128*U_temp[i+1,j-1,:] - 16*U_temp[i+2,j-1,:]
                                    - 30*U_temp[i-2,j,:] + 240*U_temp[i-1,j,:] - 240*U_temp[i+1,j,:] + 30*U_temp[i+2,j,:]
                                    + 16*U_temp[i-2,j+1,:] - 128*U_temp[i-1,j+1,:] + 128*U_temp[i+1,j+1,:] - 16*U_temp[i+2,j+1,:]
                                    - U_temp[i-2,j+2,:] + 8*U_temp[i-1,j+2,:] - 8*U_temp[i+1,j+2,:] + U_temp[i+2,j+2,:]),
                        coeff[3] * (- U_temp[i-2,j-2,:] + 8*U_temp[i-2,j-1,:] - 8*U_temp[i-2,j+1,:] + U_temp[i-2,j+2,:]                                   #dyxxU
                                    + 16*U_temp[i-1,j-2,:] -128*U_temp[i-1,j-1,:] + 128*U_temp[i-1,j+1,:] - 16*U_temp[i-1,j+2,:]
                                    - 30*U_temp[i,j-2,:] + 240*U_temp[i,j-1,:] - 240*U_temp[i,j+1,:] + 30*U_temp[i,j+2,:]
                                    + 16*U_temp[i+1,j-2,:] - 128*U_temp[i+1,j-1,:] + 128*U_temp[i+1,j+1,:] - 16*U_temp[i+1,j+2,:]
                                    - U_temp[i+2,j-2,:] + 8*U_temp[i+2,j-1,:] - 8*U_temp[i+2,j+1,:] + U_temp[i+2,j+2,:]),
                        coeff[6] * (U_temp[i-2,j-2,:] -16*U_temp[i-2,j-1,:] +30 * U_temp[i-2,j,:] - 16*U_temp[i-2,j+1,:] + U_temp[i-2,j+2,:]              #dxxyyU
                                    - 16*U_temp[i-1,j-2,:] +256*U_temp[i-1,j-1,:]-480*U_temp[i-1,j,:] + 256*U_temp[i-1,j+1,:] - 16*U_temp[i-1,j+2,:]
                                    + 30*U_temp[i,j-2,:] -480*U_temp[i,j-1,:] +900*U_temp[i,j,:] -480*U_temp[i,j+1,:] + 30*U_temp[i,j+2,:]
                                    - 16*U_temp[i+1,j-2,:] +256*U_temp[i+1,j-1,:]-480*U_temp[i+1,j,:] + 256*U_temp[i+1,j+1,:] - 16*U_temp[i+1,j+2,:]
                                    + U_temp[i+2,j-2,:] -16*U_temp[i+2,j-1,:] +30*U_temp[i+2,j,:] - 16*U_temp[i+2,j+1,:] + U_temp[i+2,j+2,:])]

                a1 = - data.dt * (A[0] @ dxU[1] + B[0] @ dyU[1])
                a2 = data.dt**2/2 * (- A[1] @ dxU[1] - B[1] @ dyU[1] + c[0]**2 * (dxU[2] + dyU[2]))
                a3 = data.dt**3/6 * (- c[0]**2 * (A[0] @ dxU[3] + B[0] @ dyU[3] + A[0] @ dxyU[1] + B[0] @ dxyU[2])
                                     + 2*c[0]*c[1] * (dxU[2] + dyU[2]) - (A[2] @ dxU[1] + B[2] @ dyU[1])
                                     + A[1] @ A[0] @ dxU[2] + B[1] @ B[0] @ dyU[2] + (A[1] @ B[0] + B[1] @ A[0]) @ dxyU[0])
                a4 = data.dt**4/24 * (- (A[3] @ dxU[1] + B[3] @ dyU[1]) + 2 * (c[1]**2 + c[0] * c[2]) * (dxU[2] + dyU[2])
                                      - 4 * c[0]*c[1]*(A[0] @ dxU[3] + B[0] @ dyU[3] + B[0] @ dxyU[2] + A[0] @ dxyU[1])
                                      + 2 * (A[2] @ A[0] @ dxU[2] + (A[2] @ B[0] + B[2] @ A[0]) @ dxyU[0] + B[2] @ B[0] @ dyU[2])
                                      + (A[1] @ A[1] @ dxU[2] + B[1] @ B[1] @ dyU[2] + (A[1] @ B[1] + B[1] @ A[1]) @ dxyU[0])
                                      - (A[1] @ A[0] @ A[0] @ dxU[3] + (A[1] @ A[0] @ B[0] + B[1] @ A[0] @ A[0]) @ dxyU[2]
                                      + B[1] @ B[0] @ B[0] @ dyU[3] + (B[1] @ B[0] @ A[0] + A[1] @ B[0] @ B[0]) @ dxyU[1])
                                      + c[0]**4*(dxU[4] + 2 * dxyU[3] + dyU[4]))


                s = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * ((i,j) in data.ps) * np.array([0, 0, 1]).transpose()
                U_temp_[i, j, :] = U_temp[i, j, :] + a1 + a2 + a3 + a4 + s

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0, 0], [0, -rho(t + data.dt)[1] / rho(t + data.dt)[0], 0],[0, 0, E(t + data.dt)[1] / E(t + data.dt)[0]]])
        for i in range(data.Mx):
            for j in range(data.My):
                data.U[n + 1, i, j, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2),np.exp(-S_n[2, 2] * data.dt / 2)]) @ U_temp_[i, j, :]

    data.calcul_energie()

"""
Résolution du problème de Cauchy 2D modulé en temps
"""

def LaxWendroff2D_cauchy_mt(data):
    """
    Utilise le schéma de Lax Wendroff en 2D dans un mileu modulé en temps
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    print("Lax Wendroff 2D Cauchy mt()")
    sleep(0.001)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.Mx + 2, data.My, 3))

    # init
    r = data.rho_mt(data)(0)[0]
    c = np.sqrt(data.E_mt(data)(0)[0] / r)
    for i in range(0, data.Mx + 2):
        for j in range(0, data.My):
            data.U[0, i, j, :] = 1/c * data.S(data.f,1/data.f + data.dy/c + data.tc[0] - data.dy * j / c) * np.array([0, 1, r * c]).transpose()

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data)
        E = data.E_mt(data)
        c = np.sqrt(E(t)[0]/rho(t)[0])
        A, A_ = A2D_mt(data)(t)[0], A2D_mt(data)(t)[1]
        B, B_ = B2D_mt(data)(t)[0], B2D_mt(data)(t)[1]
        U_temp = np.zeros((data.Mx + 2, data.My, 3))
        U_temp_ = np.zeros((data.Mx + 2, data.My, 3))

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0,0],[0, -rho(t)[1] / rho(t)[0],0], [0,0, E(t)[1] / E(t)[0]]])
        for i in range(0, data.Mx + 2):
            for j in range(0, data.My):
                U_temp[i, j, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)]) @ data.U[n, i, j, :]

        for j in range(1, data.My - 1):
            # condition de periodicité gauche
            a1 = (1 / (2 * data.dx)) * (data.dt * A + data.dt ** 2 / 2 * A_) @ (U_temp[1, j, :] - U_temp[- 1, j, :])
            a2 = (1 / (2 * data.dy)) * (data.dt * B + data.dt ** 2 / 2 * B_) @ (U_temp[0, j + 1, :] - U_temp[0, j - 1, :])
            b1 = (0.5 * (data.dt * data.c) ** 2) * ((U_temp[1, j, :] + U_temp[- 1, j, :] - 2 * U_temp[0, j, :]) / data.dx ** 2)
            b2 = (0.5 * (data.dt * data.c) ** 2) * ((U_temp[0, j + 1, :] + U_temp[0, j - 1, :] - 2 * U_temp[0, j, :]) / data.dy ** 2)
            U_temp_[0, j, :] = U_temp[0, j, :] - a1 - a2 + b1 + b2

            # condition de periodicité droite
            a1 = (1 / (2 * data.dx)) * (data.dt * A + data.dt ** 2 / 2 * A_) @ (U_temp[0, j, :] - U_temp[-2, j, :])
            a2 = (1 / (2 * data.dy)) * (data.dt * B + data.dt ** 2 / 2 * B_) @ (U_temp[-1, j + 1, :] - U_temp[-1, j - 1, :])
            b1 = (0.5 * (data.dt * c) ** 2) * ((U_temp[0, j, :] + U_temp[-2, j, :] - 2 * U_temp[-1, j, :]) / data.dx ** 2)
            b2 = (0.5 * (data.dt * c) ** 2) * ((U_temp[-1, j + 1, :] + U_temp[-1, j - 1, :] - 2 * U_temp[-1, j, :]) / data.dy ** 2)
            U_temp_[-1, j, :] = U_temp[-1, j, :] - a1 - a2 + b1 + b2
            for i in range(1, data.Mx + 1):
                # Lax-Wendroff 2D
                a1 = (1 / (2 * data.dx)) * (data.dt * A + data.dt ** 2 / 2 * A_) @ (U_temp[i + 1, j, :] - U_temp[i - 1, j, :])
                a2 = (1 / (2 * data.dy)) * (data.dt * B + data.dt ** 2 / 2 * B_) @ (U_temp[i, j + 1, :] - U_temp[i, j - 1, :])
                b1 = (0.5 * (data.dt * c) ** 2) * ((U_temp[i + 1, j, :] + U_temp[i - 1, j, :] - 2 * U_temp[i, j, :]) / data.dx ** 2)
                b2 = (0.5 * (data.dt * c) ** 2) * ((U_temp[i, j + 1, :] + U_temp[i, j - 1, :] - 2 * U_temp[i, j, :]) / data.dy ** 2)
                U_temp_[i, j, :] = U_temp[i, j, :] - a1 - a2 + b1 + b2

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0, 0],[0, -rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, 0, E(t + data.dt)[1] / E(t + data.dt)[0]]])
        for i in range(0, data.Mx + 2):
            for j in range(0, data.My):
                data.U[n + 1, i, j, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)]) @ U_temp_[i, j, :]

    data.U = data.U[:, 1:data.Mx + 1, :, :]
    data.calcul_energie()

def ADER42D_cauchy_mt(data):
    """
    Utilise le schéma de ADER4 en 2D dans un mileu modulé en temps
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    print("ADER4 2D Cauchy mt()")
    sleep(0.001)
    data.CFL_maj()
    data.U = np.zeros((data.N, data.Mx + 4, data.My, 3))

    # init
    r = data.rho_mt(data)(0)[0]
    c = np.sqrt(data.E_mt(data)(0)[0] / r)
    for i in range(0, data.Mx + 4):
        for j in range(0, data.My):
            data.U[0, i, j, :] = 1/c * data.S(data.f,1/data.f + data.dy/c + data.tc[0] - data.dy * j / c) * np.array([0, 1, r * c]).transpose()

    for n in trange(data.N - 1, ncols=ncols):
        t = n * data.dt
        rho = data.rho_mt(data)
        E = data.E_mt(data)
        c = np.sqrt(E(t)[0]/rho(t)[0])
        A, A_ = A2D_mt(data)(t)[0], A2D_mt(data)(t)[1]
        B, B_ = B2D_mt(data)(t)[0], B2D_mt(data)(t)[1]
        U_temp = np.zeros((data.Mx + 2, data.My, 3))
        U_temp_ = np.zeros((data.Mx + 2, data.My, 3))

        S_n = np.array([[-rho(t)[1] / rho(t)[0], 0,0],[0, -rho(t)[1] / rho(t)[0],0], [0,0, E(t)[1] / E(t)[0]]])
        for i in range(0, data.Mx + 4):
            for j in range(0, data.My):
                U_temp[i, j, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)]) @ data.U[n, i, j, :]

        for j in range(1, data.My - 1):
            # condition de periodicité gauche
            dxU = [0,
                   coeff[0] * (U_temp[-2,j,:] - 8*U_temp[-1,j,:] + 8*U_temp[1,j,:] - U_temp[2,j,:]),
                   coeff[1] * (- U_temp[-2,j,:] + 16*U_temp[-1,j,:] - 30*U_temp[0,j,:] + 16*U_temp[1,j,:] - U_temp[2,j,:]),
                   coeff[2] * (- U_temp[-2,j,:] + 2*U_temp[-1,j,:] - 2*U_temp[1,j,:] + U_temp[2,j,:]),
                   coeff[4] * (U_temp[-2,j,:] - 4*U_temp[-1,j,:] + 6*U_temp[0,j,:] - 4*U_temp[1,j,:] + U_temp[2,j,:])]

            dyU = [0,
                   coeff[0] * (U_temp[0,j-2,:] - 8*U_temp[0,j-1,:] + 8*U_temp[0,j+1,:] - U_temp[0,j+2,:]),
                   coeff[1] * (- U_temp[0,j-2,:] + 16*U_temp[0,j-1,:] - 30*U_temp[0,j,:] + 16*U_temp[0,j+1,:] - U_temp[0,j+2,:]),
                   coeff[2] * (- U_temp[0,j-2,:] + 2*U_temp[0,j-1,:] - 2*U_temp[0,j+1,:] + U_temp[0,j+2,:]),
                   coeff[4] * (U_temp[0,j-2,:] - 4*U_temp[0,j-1,:] + 6*U_temp[0,j,:] - 4*U_temp[0,j+1,:] + U_temp[0,j+2,:])]

            dxyU = [coeff[5] * (U_temp[-2, j-2,:] - 8*U_temp[-1,j-2,:] + 8*U_temp[1,j-2,:] - U_temp[2,j-2,:]                                    #dxyU
                                - 8*U_temp[-2, j-1,:] + 64*U_temp[-1,j-1,:] - 64*U_temp[1,j-1,:] + 8*U_temp[2,j-1,:]
                                + 8*U_temp[-2, j+1,:] - 64*U_temp[-1,j+1,:] + 64*U_temp[1,j+1,:] - 8*U_temp[2,j+1,:]
                                - U_temp[-2, j+2,:] + 8*U_temp[-1,j+2,:] - 8*U_temp[1,j+2,:] + U_temp[2,j+2,:]),
                    coeff[3] * (- U_temp[-2,j-2,:] + 8*U_temp[-1,j-2,:] - 8*U_temp[1,j-2,:] + U_temp[2,j-2,:]                                  #dxyyU
                                + 16*U_temp[-2,j-1,:] -128*U_temp[-1,j-1,:] + 128*U_temp[1,j-1,:] - 16*U_temp[2,j-1,:]
                                - 30*U_temp[-2,j,:] + 240*U_temp[-1,j,:] - 240*U_temp[1,j,:] + 30*U_temp[2,j,:]
                                + 16*U_temp[-2,j+1,:] - 128*U_temp[-1,j+1,:] + 128*U_temp[1,j+1,:] - 16*U_temp[2,j+1,:]
                                - U_temp[-2,j+2,:] + 8*U_temp[-1,j+2,:] - 8*U_temp[1,j+2,:] + U_temp[2,j+2,:]),
                    coeff[3] * (- U_temp[-2,j-2,:] + 8*U_temp[-2,j-1,:] - 8*U_temp[-2,j+1,:] + U_temp[-2,j+2,:]                                   #dyxxU
                                + 16*U_temp[-1,j-2,:] -128*U_temp[-1,j-1,:] + 128*U_temp[-1,j+1,:] - 16*U_temp[-1,j+2,:]
                                - 30*U_temp[0,j-2,:] + 240*U_temp[0,j-1,:] - 240*U_temp[0,j+1,:] + 30*U_temp[0,j+2,:]
                                + 16*U_temp[1,j-2,:] - 128*U_temp[1,j-1,:] + 128*U_temp[1,j+1,:] - 16*U_temp[1,j+2,:]
                                - U_temp[2,j-2,:] + 8*U_temp[2,j-1,:] - 8*U_temp[2,j+1,:] + U_temp[2,j+2,:]),
                    coeff[6] * (U_temp[-2,j-2,:] -16*U_temp[-2,j-1,:] +30 * U_temp[-2,j,:] - 16*U_temp[-2,j+1,:] + U_temp[-2,j+2,:]              #dxxyyU
                                - 16*U_temp[-1,j-2,:] +256*U_temp[-1,j-1,:]-480*U_temp[-1,j,:] + 256*U_temp[-1,j+1,:] - 16*U_temp[-1,j+2,:]
                                + 30*U_temp[0,j-2,:] -480*U_temp[0,j-1,:] +900*U_temp[0,j,:] -480*U_temp[0,j+1,:] + 30*U_temp[0,j+2,:]
                                - 16*U_temp[1,j-2,:] +256*U_temp[1,j-1,:]-480*U_temp[1,j,:] + 256*U_temp[1,j+1,:] - 16*U_temp[1,j+2,:]
                                + U_temp[2,j-2,:] -16*U_temp[2,j-1,:] +30*U_temp[2,j,:] - 16*U_temp[2,j+1,:] + U_temp[2,j+2,:])]

            a1 = - data.dt * (A[0] @ dxU[1] + B[0] @ dyU[1])
            a2 = data.dt**2/2 * (- A[1] @ dxU[1] - B[1] @ dyU[1] + c[0]**2 * (dxU[2] + dyU[2]))
            a3 = data.dt**3/6 * (- c[0]**2 * (A[0] @ dxU[3] + B[0] @ dyU[3] + A[0] @ dxyU[1] + B[0] @ dxyU[2])
                                 + 2*c[0]*c[1] * (dxU[2] + dyU[2]) - (A[2] @ dxU[1] + B[2] @ dyU[1])
                                 + A[1] @ A[0] @ dxU[2] + B[1] @ B[0] @ dyU[2] + (A[1] @ B[0] + B[1] @ A[0]) @ dxyU[0])
            a4 = data.dt**4/24 * (- (A[3] @ dxU[1] + B[3] @ dyU[1]) + 2 * (c[1]**2 + c[0] * c[2]) * (dxU[2] + dyU[2])
                                  - 4 * c[0]*c[1]*(A[0] @ dxU[3] + B[0] @ dyU[3] + B[0] @ dxyU[2] + A[0] @ dxyU[1])
                                  + 2 * (A[2] @ A[0] @ dxU[2] + (A[2] @ B[0] + B[2] @ A[0]) @ dxyU[0] + B[2] @ B[0] @ dyU[2])
                                  + (A[1] @ A[1] @ dxU[2] + B[1] @ B[1] @ dyU[2] + (A[1] @ B[1] + B[1] @ A[1]) @ dxyU[0])
                                  - (A[1] @ A[0] @ A[0] @ dxU[3] + (A[1] @ A[0] @ B[0] + B[1] @ A[0] @ A[0]) @ dxyU[2]
                                  + B[1] @ B[0] @ B[0] @ dyU[3] + (B[1] @ B[0] @ A[0] + A[1] @ B[0] @ B[0]) @ dxyU[1])
                                  + c[0]**4*(dxU[4] + 2 * dxyU[3] + dyU[4]))

            U_temp_[0, j, :] = U_temp[0, j, :] + a1 + a2 + a3 + a4
            
            dxU = [0,
                       coeff[0] * (U_temp[-1,j,:] - 8*U_temp[0,j,:] + 8*U_temp[2,j,:] - U_temp[3,j,:]),
                       coeff[1] * (- U_temp[-1,j,:] + 16*U_temp[0,j,:] - 30*U_temp[1,j,:] + 16*U_temp[2,j,:] - U_temp[3,j,:]),
                       coeff[2] * (- U_temp[-1,j,:] + 2*U_temp[0,j,:] - 2*U_temp[2,j,:] + U_temp[3,j,:]),
                       coeff[4] * (U_temp[-1,j,:] - 4*U_temp[0,j,:] + 6*U_temp[1,j,:] - 4*U_temp[2,j,:] + U_temp[3,j,:])]

            dyU = [0,
                   coeff[0] * (U_temp[1,j-2,:] - 8*U_temp[1,j-1,:] + 8*U_temp[1,j+1,:] - U_temp[1,j+2,:]),
                   coeff[1] * (- U_temp[1,j-2,:] + 16*U_temp[1,j-1,:] - 30*U_temp[1,j,:] + 16*U_temp[1,j+1,:] - U_temp[1,j+2,:]),
                   coeff[2] * (- U_temp[1,j-2,:] + 2*U_temp[1,j-1,:] - 2*U_temp[1,j+1,:] + U_temp[1,j+2,:]),
                   coeff[4] * (U_temp[1,j-2,:] - 4*U_temp[1,j-1,:] + 6*U_temp[1,j,:] - 4*U_temp[1,j+1,:] + U_temp[1,j+2,:])]

            dxyU = [coeff[5] * (U_temp[-1, j-2,:] - 8*U_temp[0,j-2,:] + 8*U_temp[2,j-2,:] - U_temp[3,j-2,:]                                    #dxyU
                                - 8*U_temp[-1, j-1,:] + 64*U_temp[0,j-1,:] - 64*U_temp[2,j-1,:] + 8*U_temp[3,j-1,:]
                                + 8*U_temp[-1, j+1,:] - 64*U_temp[0,j+1,:] + 64*U_temp[2,j+1,:] - 8*U_temp[3,j+1,:]
                                - U_temp[-1, j+2,:] + 8*U_temp[0,j+2,:] - 8*U_temp[2,j+2,:] + U_temp[3,j+2,:]),
                    coeff[3] * (- U_temp[-1,j-2,:] + 8*U_temp[0,j-2,:] - 8*U_temp[2,j-2,:] + U_temp[3,j-2,:]                                  #dxyyU
                                + 16*U_temp[-1,j-1,:] -128*U_temp[0,j-1,:] + 128*U_temp[2,j-1,:] - 16*U_temp[3,j-1,:]
                                - 30*U_temp[-1,j,:] + 240*U_temp[0,j,:] - 240*U_temp[2,j,:] + 30*U_temp[3,j,:]
                                + 16*U_temp[-1,j+1,:] - 128*U_temp[0,j+1,:] + 128*U_temp[2,j+1,:] - 16*U_temp[3,j+1,:]
                                - U_temp[-1,j+2,:] + 8*U_temp[0,j+2,:] - 8*U_temp[2,j+2,:] + U_temp[3,j+2,:]),
                    coeff[3] * (- U_temp[-1,j-2,:] + 8*U_temp[-1,j-1,:] - 8*U_temp[-1,j+1,:] + U_temp[-1,j+2,:]                                   #dyxxU
                                + 16*U_temp[0,j-2,:] -128*U_temp[0,j-1,:] + 128*U_temp[0,j+1,:] - 16*U_temp[0,j+2,:]
                                - 30*U_temp[1,j-2,:] + 240*U_temp[1,j-1,:] - 240*U_temp[1,j+1,:] + 30*U_temp[1,j+2,:]
                                + 16*U_temp[2,j-2,:] - 128*U_temp[2,j-1,:] + 128*U_temp[2,j+1,:] - 16*U_temp[2,j+2,:]
                                - U_temp[3,j-2,:] + 8*U_temp[3,j-1,:] - 8*U_temp[3,j+1,:] + U_temp[3,j+2,:]),
                    coeff[6] * (U_temp[-1,j-2,:] -16*U_temp[-1,j-1,:] +30 * U_temp[-1,j,:] - 16*U_temp[-1,j+1,:] + U_temp[-1,j+2,:]              #dxxyyU
                                - 16*U_temp[0,j-2,:] +256*U_temp[0,j-1,:]-480*U_temp[0,j,:] + 256*U_temp[0,j+1,:] - 16*U_temp[0,j+2,:]
                                + 30*U_temp[1,j-2,:] -480*U_temp[1,j-1,:] +900*U_temp[1,j,:] -480*U_temp[1,j+1,:] + 30*U_temp[1,j+2,:]
                                - 16*U_temp[2,j-2,:] +256*U_temp[2,j-1,:]-480*U_temp[2,j,:] + 256*U_temp[2,j+1,:] - 16*U_temp[2,j+2,:]
                                + U_temp[3,j-2,:] -16*U_temp[3,j-1,:] +30*U_temp[3,j,:] - 16*U_temp[3,j+1,:] + U_temp[3,j+2,:])]

            a1 = - data.dt * (A[0] @ dxU[1] + B[0] @ dyU[1])
            a2 = data.dt**2/2 * (- A[1] @ dxU[1] - B[1] @ dyU[1] + c[0]**2 * (dxU[2] + dyU[2]))
            a3 = data.dt**3/6 * (- c[0]**2 * (A[0] @ dxU[3] + B[0] @ dyU[3] + A[0] @ dxyU[1] + B[0] @ dxyU[2])
                                 + 2*c[0]*c[1] * (dxU[2] + dyU[2]) - (A[2] @ dxU[1] + B[2] @ dyU[1])
                                 + A[1] @ A[0] @ dxU[2] + B[1] @ B[0] @ dyU[2] + (A[1] @ B[0] + B[1] @ A[0]) @ dxyU[0])
            a4 = data.dt**4/24 * (- (A[3] @ dxU[1] + B[3] @ dyU[1]) + 2 * (c[1]**2 + c[0] * c[2]) * (dxU[2] + dyU[2])
                                  - 4 * c[0]*c[1]*(A[0] @ dxU[3] + B[0] @ dyU[3] + B[0] @ dxyU[2] + A[0] @ dxyU[1])
                                  + 2 * (A[2] @ A[0] @ dxU[2] + (A[2] @ B[0] + B[2] @ A[0]) @ dxyU[0] + B[2] @ B[0] @ dyU[2])
                                  + (A[1] @ A[1] @ dxU[2] + B[1] @ B[1] @ dyU[2] + (A[1] @ B[1] + B[1] @ A[1]) @ dxyU[0])
                                  - (A[1] @ A[0] @ A[0] @ dxU[3] + (A[1] @ A[0] @ B[0] + B[1] @ A[0] @ A[0]) @ dxyU[2]
                                  + B[1] @ B[0] @ B[0] @ dyU[3] + (B[1] @ B[0] @ A[0] + A[1] @ B[0] @ B[0]) @ dxyU[1])
                                  + c[0]**4*(dxU[4] + 2 * dxyU[3] + dyU[4]))

            U_temp_[1, j, :] = U_temp[1, j, :] + a1 + a2 + a3 + a4
            
            # condition de periodicité droite
            dxU = [0,
                       coeff[0] * (U_temp[-4,j,:] - 8*U_temp[-3,j,:] + 8*U_temp[-1,j,:] - U_temp[0,j,:]),
                       coeff[1] * (- U_temp[-4,j,:] + 16*U_temp[-3,j,:] - 30*U_temp[-2,j,:] + 16*U_temp[-1,j,:] - U_temp[0,j,:]),
                       coeff[2] * (- U_temp[-4,j,:] + 2*U_temp[-3,j,:] - 2*U_temp[-1,j,:] + U_temp[0,j,:]),
                       coeff[4] * (U_temp[-4,j,:] - 4*U_temp[-3,j,:] + 6*U_temp[-2,j,:] - 4*U_temp[-1,j,:] + U_temp[0,j,:])]

            dyU = [0,
                   coeff[0] * (U_temp[-2,j-2,:] - 8*U_temp[-2,j-1,:] + 8*U_temp[-2,j+1,:] - U_temp[-2,j+2,:]),
                   coeff[1] * (- U_temp[-2,j-2,:] + 16*U_temp[-2,j-1,:] - 30*U_temp[-2,j,:] + 16*U_temp[-2,j+1,:] - U_temp[-2,j+2,:]),
                   coeff[2] * (- U_temp[-2,j-2,:] + 2*U_temp[-2,j-1,:] - 2*U_temp[-2,j+1,:] + U_temp[-2,j+2,:]),
                   coeff[4] * (U_temp[-2,j-2,:] - 4*U_temp[-2,j-1,:] + 6*U_temp[-2,j,:] - 4*U_temp[-2,j+1,:] + U_temp[-2,j+2,:])]

            dxyU = [coeff[5] * (U_temp[-4, j-2,:] - 8*U_temp[-3,j-2,:] + 8*U_temp[-1,j-2,:] - U_temp[0,j-2,:]                                    #dxyU
                                - 8*U_temp[-4, j-1,:] + 64*U_temp[-3,j-1,:] - 64*U_temp[-1,j-1,:] + 8*U_temp[0,j-1,:]
                                + 8*U_temp[-4, j+1,:] - 64*U_temp[-3,j+1,:] + 64*U_temp[-1,j+1,:] - 8*U_temp[0,j+1,:]
                                - U_temp[-4, j+2,:] + 8*U_temp[-3,j+2,:] - 8*U_temp[-1,j+2,:] + U_temp[0,j+2,:]),
                    coeff[3] * (- U_temp[-4,j-2,:] + 8*U_temp[-3,j-2,:] - 8*U_temp[-1,j-2,:] + U_temp[0,j-2,:]                                  #dxyyU
                                + 16*U_temp[-4,j-1,:] -128*U_temp[-3,j-1,:] + 128*U_temp[-1,j-1,:] - 16*U_temp[0,j-1,:]
                                - 30*U_temp[-4,j,:] + 240*U_temp[-3,j,:] - 240*U_temp[-1,j,:] + 30*U_temp[0,j,:]
                                + 16*U_temp[-4,j+1,:] - 128*U_temp[-3,j+1,:] + 128*U_temp[-1,j+1,:] - 16*U_temp[0,j+1,:]
                                - U_temp[-4,j+2,:] + 8*U_temp[-3,j+2,:] - 8*U_temp[-1,j+2,:] + U_temp[0,j+2,:]),
                    coeff[3] * (- U_temp[-4,j-2,:] + 8*U_temp[-4,j-1,:] - 8*U_temp[-4,j+1,:] + U_temp[-4,j+2,:]                                   #dyxxU
                                + 16*U_temp[-3,j-2,:] -128*U_temp[-3,j-1,:] + 128*U_temp[-3,j+1,:] - 16*U_temp[-3,j+2,:]
                                - 30*U_temp[-2,j-2,:] + 240*U_temp[-2,j-1,:] - 240*U_temp[-2,j+1,:] + 30*U_temp[-2,j+2,:]
                                + 16*U_temp[-1,j-2,:] - 128*U_temp[-1,j-1,:] + 128*U_temp[-1,j+1,:] - 16*U_temp[-1,j+2,:]
                                - U_temp[0,j-2,:] + 8*U_temp[0,j-1,:] - 8*U_temp[0,j+1,:] + U_temp[0,j+2,:]),
                    coeff[6] * (U_temp[-4,j-2,:] -16*U_temp[-4,j-1,:] +30 * U_temp[-4,j,:] - 16*U_temp[-4,j+1,:] + U_temp[-4,j+2,:]              #dxxyyU
                                - 16*U_temp[-3,j-2,:] +256*U_temp[-3,j-1,:]-480*U_temp[-3,j,:] + 256*U_temp[-3,j+1,:] - 16*U_temp[-3,j+2,:]
                                + 30*U_temp[-2,j-2,:] -480*U_temp[-2,j-1,:] +900*U_temp[-2,j,:] -480*U_temp[-2,j+1,:] + 30*U_temp[-2,j+2,:]
                                - 16*U_temp[-1,j-2,:] +256*U_temp[-1,j-1,:]-480*U_temp[-1,j,:] + 256*U_temp[-1,j+1,:] - 16*U_temp[-1,j+2,:]
                                + U_temp[0,j-2,:] -16*U_temp[0,j-1,:] +30*U_temp[0,j,:] - 16*U_temp[0,j+1,:] + U_temp[0,j+2,:])]

            a1 = - data.dt * (A[0] @ dxU[1] + B[0] @ dyU[1])
            a2 = data.dt**2/2 * (- A[1] @ dxU[1] - B[1] @ dyU[1] + c[0]**2 * (dxU[2] + dyU[2]))
            a3 = data.dt**3/6 * (- c[0]**2 * (A[0] @ dxU[3] + B[0] @ dyU[3] + A[0] @ dxyU[1] + B[0] @ dxyU[2])
                                 + 2*c[0]*c[1] * (dxU[2] + dyU[2]) - (A[2] @ dxU[1] + B[2] @ dyU[1])
                                 + A[1] @ A[0] @ dxU[2] + B[1] @ B[0] @ dyU[2] + (A[1] @ B[0] + B[1] @ A[0]) @ dxyU[0])
            a4 = data.dt**4/24 * (- (A[3] @ dxU[1] + B[3] @ dyU[1]) + 2 * (c[1]**2 + c[0] * c[2]) * (dxU[2] + dyU[2])
                                  - 4 * c[0]*c[1]*(A[0] @ dxU[3] + B[0] @ dyU[3] + B[0] @ dxyU[2] + A[0] @ dxyU[1])
                                  + 2 * (A[2] @ A[0] @ dxU[2] + (A[2] @ B[0] + B[2] @ A[0]) @ dxyU[0] + B[2] @ B[0] @ dyU[2])
                                  + (A[1] @ A[1] @ dxU[2] + B[1] @ B[1] @ dyU[2] + (A[1] @ B[1] + B[1] @ A[1]) @ dxyU[0])
                                  - (A[1] @ A[0] @ A[0] @ dxU[3] + (A[1] @ A[0] @ B[0] + B[1] @ A[0] @ A[0]) @ dxyU[2]
                                  + B[1] @ B[0] @ B[0] @ dyU[3] + (B[1] @ B[0] @ A[0] + A[1] @ B[0] @ B[0]) @ dxyU[1])
                                  + c[0]**4*(dxU[4] + 2 * dxyU[3] + dyU[4]))

            U_temp_[-2, j, :] = U_temp[-2, j, :] + a1 + a2 + a3 + a4
            
            dxU = [0,
                       coeff[0] * (U_temp[-3,j,:] - 8*U_temp[-2,j,:] + 8*U_temp[0,j,:] - U_temp[1,j,:]),
                       coeff[1] * (- U_temp[-3,j,:] + 16*U_temp[-2,j,:] - 30*U_temp[-1,j,:] + 16*U_temp[0,j,:] - U_temp[1,j,:]),
                       coeff[2] * (- U_temp[-3,j,:] + 2*U_temp[-2,j,:] - 2*U_temp[0,j,:] + U_temp[1,j,:]),
                       coeff[4] * (U_temp[-3,j,:] - 4*U_temp[-2,j,:] + 6*U_temp[-1,j,:] - 4*U_temp[0,j,:] + U_temp[1,j,:])]

            dyU = [0,
                   coeff[0] * (U_temp[-1,j-2,:] - 8*U_temp[-1,j-1,:] + 8*U_temp[-1,j+1,:] - U_temp[-1,j+2,:]),
                   coeff[1] * (- U_temp[-1,j-2,:] + 16*U_temp[-1,j-1,:] - 30*U_temp[-1,j,:] + 16*U_temp[-1,j+1,:] - U_temp[-1,j+2,:]),
                   coeff[2] * (- U_temp[-1,j-2,:] + 2*U_temp[-1,j-1,:] - 2*U_temp[-1,j+1,:] + U_temp[-1,j+2,:]),
                   coeff[4] * (U_temp[-1,j-2,:] - 4*U_temp[-1,j-1,:] + 6*U_temp[-1,j,:] - 4*U_temp[-1,j+1,:] + U_temp[-1,j+2,:])]

            dxyU = [coeff[5] * (U_temp[-3, j-2,:] - 8*U_temp[-2,j-2,:] + 8*U_temp[0,j-2,:] - U_temp[1,j-2,:]                                    #dxyU
                                - 8*U_temp[-3, j-1,:] + 64*U_temp[-2,j-1,:] - 64*U_temp[0,j-1,:] + 8*U_temp[1,j-1,:]
                                + 8*U_temp[-3, j+1,:] - 64*U_temp[-2,j+1,:] + 64*U_temp[0,j+1,:] - 8*U_temp[1,j+1,:]
                                - U_temp[-3, j+2,:] + 8*U_temp[-2,j+2,:] - 8*U_temp[0,j+2,:] + U_temp[1,j+2,:]),
                    coeff[3] * (- U_temp[-3,j-2,:] + 8*U_temp[-2,j-2,:] - 8*U_temp[0,j-2,:] + U_temp[1,j-2,:]                                  #dxyyU
                                + 16*U_temp[-3,j-1,:] -128*U_temp[-2,j-1,:] + 128*U_temp[0,j-1,:] - 16*U_temp[1,j-1,:]
                                - 30*U_temp[-3,j,:] + 240*U_temp[-2,j,:] - 240*U_temp[0,j,:] + 30*U_temp[1,j,:]
                                + 16*U_temp[-3,j+1,:] - 128*U_temp[-2,j+1,:] + 128*U_temp[0,j+1,:] - 16*U_temp[1,j+1,:]
                                - U_temp[-3,j+2,:] + 8*U_temp[-2,j+2,:] - 8*U_temp[0,j+2,:] + U_temp[1,j+2,:]),
                    coeff[3] * (- U_temp[-3,j-2,:] + 8*U_temp[-3,j-1,:] - 8*U_temp[-3,j+1,:] + U_temp[-3,j+2,:]                                   #dyxxU
                                + 16*U_temp[-2,j-2,:] -128*U_temp[-2,j-1,:] + 128*U_temp[-2,j+1,:] - 16*U_temp[-2,j+2,:]
                                - 30*U_temp[-1,j-2,:] + 240*U_temp[-1,j-1,:] - 240*U_temp[-1,j+1,:] + 30*U_temp[-1,j+2,:]
                                + 16*U_temp[0,j-2,:] - 128*U_temp[0,j-1,:] + 128*U_temp[0,j+1,:] - 16*U_temp[0,j+2,:]
                                - U_temp[1,j-2,:] + 8*U_temp[1,j-1,:] - 8*U_temp[1,j+1,:] + U_temp[1,j+2,:]),
                    coeff[6] * (U_temp[-3,j-2,:] -16*U_temp[-3,j-1,:] +30 * U_temp[-3,j,:] - 16*U_temp[-3,j+1,:] + U_temp[-3,j+2,:]              #dxxyyU
                                - 16*U_temp[-2,j-2,:] +256*U_temp[-2,j-1,:]-480*U_temp[-2,j,:] + 256*U_temp[-2,j+1,:] - 16*U_temp[-2,j+2,:]
                                + 30*U_temp[-1,j-2,:] -480*U_temp[-1,j-1,:] +900*U_temp[-1,j,:] -480*U_temp[-1,j+1,:] + 30*U_temp[-1,j+2,:]
                                - 16*U_temp[0,j-2,:] +256*U_temp[0,j-1,:]-480*U_temp[0,j,:] + 256*U_temp[0,j+1,:] - 16*U_temp[0,j+2,:]
                                + U_temp[1,j-2,:] -16*U_temp[1,j-1,:] +30*U_temp[1,j,:] - 16*U_temp[1,j+1,:] + U_temp[1,j+2,:])]

            a1 = - data.dt * (A[0] @ dxU[1] + B[0] @ dyU[1])
            a2 = data.dt**2/2 * (- A[1] @ dxU[1] - B[1] @ dyU[1] + c[0]**2 * (dxU[2] + dyU[2]))
            a3 = data.dt**3/6 * (- c[0]**2 * (A[0] @ dxU[3] + B[0] @ dyU[3] + A[0] @ dxyU[1] + B[0] @ dxyU[2])
                                 + 2*c[0]*c[1] * (dxU[2] + dyU[2]) - (A[2] @ dxU[1] + B[2] @ dyU[1])
                                 + A[1] @ A[0] @ dxU[2] + B[1] @ B[0] @ dyU[2] + (A[1] @ B[0] + B[1] @ A[0]) @ dxyU[0])
            a4 = data.dt**4/24 * (- (A[3] @ dxU[1] + B[3] @ dyU[1]) + 2 * (c[1]**2 + c[0] * c[2]) * (dxU[2] + dyU[2])
                                  - 4 * c[0]*c[1]*(A[0] @ dxU[3] + B[0] @ dyU[3] + B[0] @ dxyU[2] + A[0] @ dxyU[1])
                                  + 2 * (A[2] @ A[0] @ dxU[2] + (A[2] @ B[0] + B[2] @ A[0]) @ dxyU[0] + B[2] @ B[0] @ dyU[2])
                                  + (A[1] @ A[1] @ dxU[2] + B[1] @ B[1] @ dyU[2] + (A[1] @ B[1] + B[1] @ A[1]) @ dxyU[0])
                                  - (A[1] @ A[0] @ A[0] @ dxU[3] + (A[1] @ A[0] @ B[0] + B[1] @ A[0] @ A[0]) @ dxyU[2]
                                  + B[1] @ B[0] @ B[0] @ dyU[3] + (B[1] @ B[0] @ A[0] + A[1] @ B[0] @ B[0]) @ dxyU[1])
                                  + c[0]**4*(dxU[4] + 2 * dxyU[3] + dyU[4]))

            U_temp_[-1, j, :] = U_temp[-1, j, :] + a1 + a2 + a3 + a4

            for i in range(2, data.Mx + 2):
                # ADER4 2D
                dxU = [0,
                       coeff[0] * (U_temp[i-2,j,:] - 8*U_temp[i-1,j,:] + 8*U_temp[i+1,j,:] - U_temp[i+2,j,:]),
                       coeff[1] * (- U_temp[i-2,j,:] + 16*U_temp[i-1,j,:] - 30*U_temp[i,j,:] + 16*U_temp[i+1,j,:] - U_temp[i+2,j,:]),
                       coeff[2] * (- U_temp[i-2,j,:] + 2*U_temp[i-1,j,:] - 2*U_temp[i+1,j,:] + U_temp[i+2,j,:]),
                       coeff[4] * (U_temp[i-2,j,:] - 4*U_temp[i-1,j,:] + 6*U_temp[i,j,:] - 4*U_temp[i+1,j,:] + U_temp[i+2,j,:])]

                dyU = [0,
                       coeff[0] * (U_temp[i,j-2,:] - 8*U_temp[i,j-1,:] + 8*U_temp[i,j+1,:] - U_temp[i,j+2,:]),
                       coeff[1] * (- U_temp[i,j-2,:] + 16*U_temp[i,j-1,:] - 30*U_temp[i,j,:] + 16*U_temp[i,j+1,:] - U_temp[i,j+2,:]),
                       coeff[2] * (- U_temp[i,j-2,:] + 2*U_temp[i,j-1,:] - 2*U_temp[i,j+1,:] + U_temp[i,j+2,:]),
                       coeff[4] * (U_temp[i,j-2,:] - 4*U_temp[i,j-1,:] + 6*U_temp[i,j,:] - 4*U_temp[i,j+1,:] + U_temp[i,j+2,:])]

                dxyU = [coeff[5] * (U_temp[i-2, j-2,:] - 8*U_temp[i-1,j-2,:] + 8*U_temp[i+1,j-2,:] - U_temp[i+2,j-2,:]                                    #dxyU
                                    - 8*U_temp[i-2, j-1,:] + 64*U_temp[i-1,j-1,:] - 64*U_temp[i+1,j-1,:] + 8*U_temp[i+2,j-1,:]
                                    + 8*U_temp[i-2, j+1,:] - 64*U_temp[i-1,j+1,:] + 64*U_temp[i+1,j+1,:] - 8*U_temp[i+2,j+1,:]
                                    - U_temp[i-2, j+2,:] + 8*U_temp[i-1,j+2,:] - 8*U_temp[i+1,j+2,:] + U_temp[i+2,j+2,:]),
                        coeff[3] * (- U_temp[i-2,j-2,:] + 8*U_temp[i-1,j-2,:] - 8*U_temp[i+1,j-2,:] + U_temp[i+2,j-2,:]                                  #dxyyU
                                    + 16*U_temp[i-2,j-1,:] -128*U_temp[i-1,j-1,:] + 128*U_temp[i+1,j-1,:] - 16*U_temp[i+2,j-1,:]
                                    - 30*U_temp[i-2,j,:] + 240*U_temp[i-1,j,:] - 240*U_temp[i+1,j,:] + 30*U_temp[i+2,j,:]
                                    + 16*U_temp[i-2,j+1,:] - 128*U_temp[i-1,j+1,:] + 128*U_temp[i+1,j+1,:] - 16*U_temp[i+2,j+1,:]
                                    - U_temp[i-2,j+2,:] + 8*U_temp[i-1,j+2,:] - 8*U_temp[i+1,j+2,:] + U_temp[i+2,j+2,:]),
                        coeff[3] * (- U_temp[i-2,j-2,:] + 8*U_temp[i-2,j-1,:] - 8*U_temp[i-2,j+1,:] + U_temp[i-2,j+2,:]                                   #dyxxU
                                    + 16*U_temp[i-1,j-2,:] -128*U_temp[i-1,j-1,:] + 128*U_temp[i-1,j+1,:] - 16*U_temp[i-1,j+2,:]
                                    - 30*U_temp[i,j-2,:] + 240*U_temp[i,j-1,:] - 240*U_temp[i,j+1,:] + 30*U_temp[i,j+2,:]
                                    + 16*U_temp[i+1,j-2,:] - 128*U_temp[i+1,j-1,:] + 128*U_temp[i+1,j+1,:] - 16*U_temp[i+1,j+2,:]
                                    - U_temp[i+2,j-2,:] + 8*U_temp[i+2,j-1,:] - 8*U_temp[i+2,j+1,:] + U_temp[i+2,j+2,:]),
                        coeff[6] * (U_temp[i-2,j-2,:] -16*U_temp[i-2,j-1,:] +30 * U_temp[i-2,j,:] - 16*U_temp[i-2,j+1,:] + U_temp[i-2,j+2,:]              #dxxyyU
                                    - 16*U_temp[i-1,j-2,:] +256*U_temp[i-1,j-1,:]-480*U_temp[i-1,j,:] + 256*U_temp[i-1,j+1,:] - 16*U_temp[i-1,j+2,:]
                                    + 30*U_temp[i,j-2,:] -480*U_temp[i,j-1,:] +900*U_temp[i,j,:] -480*U_temp[i,j+1,:] + 30*U_temp[i,j+2,:]
                                    - 16*U_temp[i+1,j-2,:] +256*U_temp[i+1,j-1,:]-480*U_temp[i+1,j,:] + 256*U_temp[i+1,j+1,:] - 16*U_temp[i+1,j+2,:]
                                    + U_temp[i+2,j-2,:] -16*U_temp[i+2,j-1,:] +30*U_temp[i+2,j,:] - 16*U_temp[i+2,j+1,:] + U_temp[i+2,j+2,:])]

                a1 = - data.dt * (A[0] @ dxU[1] + B[0] @ dyU[1])
                a2 = data.dt**2/2 * (- A[1] @ dxU[1] - B[1] @ dyU[1] + c[0]**2 * (dxU[2] + dyU[2]))
                a3 = data.dt**3/6 * (- c[0]**2 * (A[0] @ dxU[3] + B[0] @ dyU[3] + A[0] @ dxyU[1] + B[0] @ dxyU[2])
                                     + 2*c[0]*c[1] * (dxU[2] + dyU[2]) - (A[2] @ dxU[1] + B[2] @ dyU[1])
                                     + A[1] @ A[0] @ dxU[2] + B[1] @ B[0] @ dyU[2] + (A[1] @ B[0] + B[1] @ A[0]) @ dxyU[0])
                a4 = data.dt**4/24 * (- (A[3] @ dxU[1] + B[3] @ dyU[1]) + 2 * (c[1]**2 + c[0] * c[2]) * (dxU[2] + dyU[2])
                                      - 4 * c[0]*c[1]*(A[0] @ dxU[3] + B[0] @ dyU[3] + B[0] @ dxyU[2] + A[0] @ dxyU[1])
                                      + 2 * (A[2] @ A[0] @ dxU[2] + (A[2] @ B[0] + B[2] @ A[0]) @ dxyU[0] + B[2] @ B[0] @ dyU[2])
                                      + (A[1] @ A[1] @ dxU[2] + B[1] @ B[1] @ dyU[2] + (A[1] @ B[1] + B[1] @ A[1]) @ dxyU[0])
                                      - (A[1] @ A[0] @ A[0] @ dxU[3] + (A[1] @ A[0] @ B[0] + B[1] @ A[0] @ A[0]) @ dxyU[2]
                                      + B[1] @ B[0] @ B[0] @ dyU[3] + (B[1] @ B[0] @ A[0] + A[1] @ B[0] @ B[0]) @ dxyU[1])
                                      + c[0]**4*(dxU[4] + 2 * dxyU[3] + dyU[4]))

                U_temp_[i, j, :] = U_temp[i, j, :] + a1 + a2 + a3 + a4

        S_n = np.array([[-rho(t + data.dt)[1] / rho(t + data.dt)[0], 0, 0],[0, -rho(t + data.dt)[1] / rho(t + data.dt)[0], 0], [0, 0, E(t + data.dt)[1] / E(t + data.dt)[0]]])
        for i in range(0, data.Mx + 4):
            for j in range(0, data.My):
                data.U[n + 1, i, j, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2), np.exp(-S_n[2, 2] * data.dt / 2)]) @ U_temp_[i, j, :]

    data.U = data.U[:, 2:data.Mx + 2, :, :]
    data.calcul_energie()
