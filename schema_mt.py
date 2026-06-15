from tqdm import trange
ncols = 125                                                                       #largeur de la barre de chargement
from math import factorial
from modulation import *

def A_mt(data, **kwargs):
    def f(t):
        if "alpha" in kwargs.keys():
            alpha = kwargs["alpha"]
            rho = data.rho_mt(data,alpha)(t)
            E = data.E_mt(data,alpha)(t)
        else:
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

def ADER41D_mt(data, **kwargs):
    """
    Utilise le schéma d'ADER4 en 1D dans un mileu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en quantité de mouvement et pression du problème 1D
    """
    data.U = np.zeros((data.N, data.M, 2))

    for n in trange(data.N-1, ncols = ncols):
        for i in range(2,data.M-2):
            rE, eps, w, t = data.rho/data.e, data.eps, data.omega, n*data.dt
            if "alpha" in kwargs.keys():
                alpha = kwargs["alpha"]
                rho = data.rho_mt(data, alpha)(t)
                E = data.E_mt(data, alpha)(t)
                g = A_mt(data, alpha = alpha)(t)
            else:
                rho = data.rho_mt(data)(t)
                E = data.E_mt(data)(t)
                g = A_mt(data)(t)

            dxU = np.array([1/(12*data.dx) * (data.U[n, i-2,:] - 8*data.U[n,i-1,:] + 8*data.U[n,i+1,:] - data.U[n,i+2,:]),
                            1/(12*data.dx**2) * (-data.U[n, i-2,:] + 16*data.U[n,i-1,:] - 30*data.U[n,i,:] + 16*data.U[n,i+1,:] - data.U[n,i+2,:]),
                            1/(12*data.dx**3) * (-data.U[n, i-2,:] + 2*data.U[n,i-1,:] - 2*data.U[n,i+1,:] + data.U[n,i+2,:]),
                            1/(24*data.dx**4) * (-data.U[n, i-2,:] + 4*data.U[n,i-1,:] - 6*data.U[n,i,:] + 4*data.U[n,i+1,:] - data.U[n,i+2,:])])

            a = -g[0]
            b1, b2 = - g[1], g[0] @ g[0]
            c1, c2, c3 = - g[2], 3 * g[1] @ g[0], - g[0] @ g[0] @ g[0]
            d1, d2, d3, d4 = - g[3], 4 * g[2] @ g[0] + 3 * g[2] @ g[2], -6 * g[1] @ g[0] @ g[0], g[0] @ g[0] @ g[0] @ g[0]
            dtU = np.array([a @ dxU[0], b1 @ dxU[0] + b2 @ dxU[1], c1 @ dxU[0] + c2 @ dxU[1] + c3 @ dxU[2], d1 @ dxU[0] + d2 @ dxU[1] + d3 @ dxU[2] + d4 @ dxU[3]])

            s = (data.dt / data.dx) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * np.array([data.opt, not data.opt]) + np.array([[-rho[1]/rho[0], 0],[0, E[1]/E[0]]]) @ data.U[n+1, i, :]
            data.U[n + 1, i, :] = data.U[n, i, :] + sum([data.dt**(j + 1)/factorial(j + 1)*dtU[j, :] for j in range(4)]) + s

    return data.U