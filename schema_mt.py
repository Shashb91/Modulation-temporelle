import numpy as np
from tqdm import trange
ncols = 125                                                                       #largeur de la barre de chargement
from math import factorial
from modulation import *

def A1D_mt(data):
    def f(t):
        rho = data.rho_mt(data)(t)
        E = data.E_mt(data)(t)

        A = np.array([[0, 1/rho[0]],
                      [E[0], 0]])
        B = np.array([[0,-rho[1]/rho[0]**2],
                      [E[1], 0]])
        C = np.array([[0,- (rho[2]*rho[0] - 2*rho[1])/rho[0]**3],
                      [E[2], 0]])
        D = np.array([[0,-(rho[3]*rho[2]**2 - 5*rho[2]*rho[0] + 6*rho[1])/rho[0]**4],
                      [E[3], 0]])
        return [A,B,C,D]
    return f

"""
Modulation temporelle dans un problème de propagation 1D 
"""

def LaxWendroff1D_mt(data):
    """
    Utilise le schéma d'ADER4 en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    print("LaxWendroff 1D mt()")
    data.CFL_maj()
    data.U = np.zeros((data.N, data.M, 2))

    for n in trange(data.N - 1, ncols=ncols):
        for i in range(2, data.M - 2):
            rE, t = data.rho / data.e, n * data.dt
            rho = data.rho_mt(data)
            E = data.E_mt(data)
            A, A_ = A1D_mt(data)(t)[0], A1D_mt(data)(t)[1]

            S_n = np.array([[-rho(t)[1] / rho(t)[0], 0], [0, E(t)[1] / E(t)[0]]])
            U_temp = data.U[n, :, :]
            U_temp[i, :] = np.diag([np.exp(-S_n[0, 0] * data.dt / 2), np.exp(-S_n[1, 1] * data.dt / 2)]) @ data.U[n, i, :]

            a1 = (1 / (2 * data.dx)) * ( data.dt * A + data.dt**2 /2 * A_) @ (data.U[n, i + 1, :] - data.U[n, i - 1, :])
            a2 = (0.5 * (data.dt / data.dx) ** 2) *(-1/ (A @ A) @ (data.U[n, i + 1, :] + data.U[n, i - 1, :] - 2 * data.U[n, i, :]))
            data.U[n + 1, i, :] = data.U[n, i, :] - a1 + a2 + data.dt / data.dx * data.S(data.f,(n + 1) * data.dt) * (i == data.xs) * np.array([1, 0]).transpose()

def ADER41D_mt(data):
    """
    Utilise le schéma d'ADER4 en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    print("ADER4 1D mt()")
    data.CFL_maj()
    data.U = np.zeros((data.N, data.M, 2))

    for n in trange(data.N-1, ncols = ncols):
        for i in range(2,data.M-2):
            rE, t = data.rho/data.e, n*data.dt
            rho = data.rho_mt(data)
            E = data.E_mt(data)
            g = A1D_mt(data)(t)

            S_n = np.array([[-rho(t)[1] / rho(t)[0], 0], [0, E(t)[1] / E(t)[0]]])
            U_temp = data.U[n, :, :]
            U_temp[i,:] = np.diag([np.exp(-S_n[0,0]*data.dt/2), np.exp(-S_n[1,1]*data.dt/2)]) @ data.U[n, i, :]

            dxU = np.array([1/(12*data.dx) * (U_temp[i-2,:] - 8*U_temp[i-1,:] + 8*U_temp[i+1,:] - U_temp[i+2,:]),
                            1/(12*data.dx**2) * (-U_temp[i-2,:] + 16*U_temp[i-1,:] - 30*U_temp[i,:] + 16*U_temp[i+1,:] - U_temp[i+2,:]),
                            6/(12*data.dx**3) * (-U_temp[i-2,:] + 2*U_temp[i-1,:] - 2*U_temp[i+1,:] + U_temp[i+2,:]),
                            - 1/(data.dx**4) * (-U_temp[i-2,:] + 4*U_temp[i-1,:] - 6*U_temp[i,:] + 4*U_temp[i+1,:] - U_temp[i+2,:])])

            a = -g[0]
            b1, b2 = - g[1], g[0] @ g[0]
            c1, c2, c3 = - g[2], 3 * g[1] @ g[0], - g[0] @ g[0] @ g[0]
            d1, d2, d3, d4 = - g[3], 4 * g[2] @ g[0] + 3 * g[2] @ g[2], -6 * g[1] @ g[0] @ g[0], g[0] @ g[0] @ g[0] @ g[0]
            dtU = np.array([a @ dxU[0], b1 @ dxU[0] + b2 @ dxU[1], c1 @ dxU[0] + c2 @ dxU[1] + c3 @ dxU[2], d1 @ dxU[0] + d2 @ dxU[1] + d3 @ dxU[2] + d4 @ dxU[3]])
            etape2 = U_temp[i, :] + sum([data.dt**(j + 1)/factorial(j + 1)*dtU[j, :] for j in range(4)])

            s = (data.dt * data.rho / (data.dx * rho(t)[0]) ) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * np.array([data.opt, not data.opt])
            S_n = np.array([[-rho(t+data.dt)[1] / rho(t+data.dt)[0], 0], [0, E(t+data.dt)[1] / E(t+data.dt)[0]]])
            data.U[n + 1, i, :] = np.diag([np.exp(-S_n[0,0]*data.dt/2), np.exp(-S_n[1,1]*data.dt/2)]) @ etape2 + s

"""
Modulation temporelle dans un problème de propagation 2D
"""

def ADER42D_mt(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D
    :param data: Donnee2D, regroupe l'ensemble des données du probleme 2D
    :return: np.ndarray(), solution en vitesse et pression du problème 2D
    """
    print("ADER4 2D mt()")
    data.U = np.zeros((data.N, data.Mx, data.My, 3))
    rc2 = data.rho*data.c**2
    A = np.array([[0,0, 1 / data.rho],
                  [0,0, 0],
                  [rc2, 0, 0]])
    B = np.array([[0,0,0],
                  [0,0,1/data.rho],
                  [0, rc2, 0]])
    b2 = - (data.c*data.dt)**2/24
    b3 = data.c**2*data.dt**3/6
    b4 = - (data.c*data.dt)**4/24

    c = [1/(12*data.dx),1/data.dx**2,1/(2*data.dx**3),1/(144*data.dx*data.dy**2),1/data.dx**4,2/(144*data.dx*data.dy)**2]
    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(2, data.Mx - 2):
            for j in range(2, data.My - 2):
                a1 = data.dt * (c[0] * A @ (data.U[n,i-2,j,:] - 8*data.U[n,i-1,j,:] + 8*data.U[n,i+1,j,:] - data.U[n,i+2,j,:]) +
                                c[0] * B @ (data.U[n,i,j-2,:] - 8*data.U[n,i,j-1,:] + 8*data.U[n,i,j+1,:] - data.U[n,i,j+2,:]))
                a2 = b2 * (c[1] * (- data.U[n,i-2,j,:] + 16*data.U[n,i-1,j,:] - 30*data.U[n,i,j,:] + 16*data.U[n,i+1,j,:] - data.U[n,i+2,j,:]) +
                           c[1] * (- data.U[n,i,j-2,:] + 16*data.U[n,i,j-1,:] - 30*data.U[n,i,j,:] + 16*data.U[n,i,j+1,:] - data.U[n,i,j+2,:]))
                a3 = b3 * (A @ (c[2] * (- data.U[n,i-2,j,:] + 2*data.U[n,i-1,j,:] - 2*data.U[n,i+1,j,:] + data.U[n,i+2,j,:]) +
                                c[3] * (- data.U[n,i-2,j-2,:] + 8*data.U[n,i-1,j-2,:] - 8*data.U[n,i+1,j-2,:] + data.U[n,i+2,j-2,:]+
                                        16*data.U[n,i-2,j-1,:] -128*data.U[n,i-1,j-1,:] + 128*data.U[n,i+1,j-1,:] - 16*data.U[n,i+2,j-1,:]
                                        -30*data.U[n,i-2,j,:] + 240*data.U[n,i-1,j,:] - 240*data.U[n,i+1,j,:] + 30*data.U[n,i+2,j,:]+
                                        16*data.U[n,i-2,j+1,:] - 128*data.U[n,i-1,j+1,:] + 128*data.U[n,i+1,j+1,:] - 16*data.U[n,i+2,j+1,:]
                                        - data.U[n,i-2,j+2,:] + 8*data.U[n,i-1,j+2,:] - 8*data.U[n,i+1,j+2,:] + data.U[n,i+2,j+2,:])) +
                           B @ (c[2] * (- data.U[n,i,j-2,:] + 2*data.U[n,i,j-1,:] - 2*data.U[n,i,j+1,:] + data.U[n,i,j+2,:]) +
                                c[3] * (- data.U[n,i-2,j-2,:] + 8*data.U[n,i-2,j-1,:] - 8*data.U[n,i-2,j+1,:] + data.U[n,i-2,j+2,:]
                                        + 16*data.U[n,i-1,j-2,:] -128*data.U[n,i-1,j-1,:] + 128*data.U[n,i-1,j+1,:] - 16*data.U[n,i-1,j+2,:]
                                        - 30*data.U[n,i,j-2,:] + 240*data.U[n,i,j-1,:] - 240*data.U[n,i,j+1,:] + 30*data.U[n,i,j+2,:]
                                        + 16*data.U[n,i+1,j-2,:] - 128*data.U[n,i+1,j-1,:] + 128*data.U[n,i+1,j+1,:] - 16*data.U[n,i+1,j+2,:]
                                        - data.U[n,i+2,j-2,:] + 8*data.U[n,i+2,j-1,:] - 8*data.U[n,i+2,j+1,:] + data.U[n,i+2,j+2,:])))
                a4 = b4 * (c[4] * (data.U[n,i-2,j,:] - 4*data.U[n,i-1,j,:] + 6*data.U[n,i,j,:] - 4*data.U[n,i+1,j,:] + data.U[n,i+2,j,:]) +
                           c[4] * (data.U[n,i,j-2,:] - 4*data.U[n,i,j-1,:] + 6*data.U[n,i,j,:] - 4*data.U[n,i,j+1,:] + data.U[n,i,j+2,:]) +
                           c[5] * (data.U[n,i-2,j-2,:] -16*data.U[n,i-2,j-1,:] +30 * data.U[n,i-2,j,:] - 16*data.U[n,i-2,j+1,:] + data.U[n,i-2,j+2,:]
                                   -16*data.U[n,i-1,j-2,:] +256*data.U[n,i-1,j-1,:]-480*data.U[n,i-1,j,:] + 256*data.U[n,i-1,j+1,:] - 16*data.U[n,i-1,j+2,:]
                                   +30*data.U[n,i,j-2,:] -480*data.U[n,i,j-1,:] +900*data.U[n,i,j,:] -480*data.U[n,i,j+1,:] + 30*data.U[n,i,j+2,:]
                                   -16*data.U[n,i+1,j-2,:] +256*data.U[n,i+1,j-1,:]-480*data.U[n,i+1,j,:] + 256*data.U[n,i+1,j+1,:] - 16*data.U[n,i+1,j+2,:]
                                   + data.U[n,i+2,j-2,:] -16*data.U[n,i+2,j-1,:] +30*data.U[n,i+2,j,:] - 16*data.U[n,i+2,j+1,:] + data.U[n,i+2,j+2,:]))
                s = ((data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * ((i,j) in data.ps) * np.array([0, 0, 1]).transpose()) * (not data.opt) +
                     (data.dt / (data.rho * np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * ((i,j) in data.ps) * np.array([1, 0, 0]).transpose()) * (data.opt))
                data.U[n + 1, i, j, :] = data.U[n,i,j,:] - a1 - a2 - a3 - a4 + s
