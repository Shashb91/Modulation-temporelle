import numpy as np
from time import sleep

from matplotlib import pyplot as plt
from tqdm import trange
ncols = 125
from donnee import Donnee2D

def cmax(data, opt = True):
    """
    Renvoie la vitesse de phase polaire maximale et trace le profil de vitesse
    :param data:
    :return: float: vitesse de phase maximale
    """
    (r0, r1) = data.rho
    r00 = r0*data.alpha + r1*(1-data.alpha)
    r11 = r0*r1/(data.alpha*r1 + r0*(1-data.alpha))
    e = 1/(data.alpha/data.e[0] + (1-data.alpha)/data.e[1])
    c0,c1 = np.sqrt(e/r00), np.sqrt(e/r11)
    theta = np.linspace(0, 2*np.pi, 300)
    Vp = np.sqrt(c0**2*np.cos(theta)**2 + c1**2*np.sin(theta)**2)
    x, y = np.cos(theta), np.sin(theta)
    if opt:
        plt.plot(Vp*y, Vp*x, c = 'blue', lw = 2, label = 'Vitesse polaire', ms = 0)
        plt.plot(c0*y, c0*x, c = 'red', lw = 2, label = f'$c_{0}$', ms = 0)
        plt.plot(c1*y, c1*x, c = 'green', lw = 2, label = f'$c_{1}$', ms = 0)
        plt.grid(True)
        plt.xlabel("Vitesse en x")
        plt.ylabel("Vitesse en y")
        plt.title(f"Vitesse polaire maximale")
        plt.legend()
        plt.axis('square')
        plt.show()

    return np.max(Vp)

def LaxWendroff_aniso(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 2D dans un milieu anisotrope
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    def G(i, j):
        sigma, R = 5*data.dx, 10*data.dx
        x, y = np.abs(data.ps[0][0] - i)*data.dx, np.abs(data.ps[0][1] - j)*data.dy
        return (1/(np.pi*sigma**2)*np.exp(-(x**2 + y**2)/sigma**2))*(0 <= x**2 + y**2 <= R**2)

    sleep(0.01)
    print("LaxWendroff Anisotrope()")
    sleep(0.01)
    c = cmax(data)
    data.CFL_aniso(c)
    data.U = np.zeros((data.N, data.Mx, data.My, 3))

    (r0, r1) = data.rho
    r00 = r0*data.alpha + r1*(1-data.alpha)
    r11 = r0*r1/(data.alpha*r1 + r0*(1-data.alpha))
    e = 1/(data.alpha/data.e[0] + (1-data.alpha)/data.e[1])
    A = np.array([[0, 0, 1/r11],
                  [0, 0, 0],
                  [e, 0, 0]])
    B = np.array([[0, 0, 0],
                  [0, 0, 1/r00],
                  [0, e, 0]])

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(1, data.Mx - 1):
            for j in range(1, data.My - 1):
                s = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * G(i ,j) * np.array([0, 0, 1]).transpose()
                a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i + 1, j, :] - data.U[n, i - 1, j, :])
                a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, i, j + 1, :] - data.U[n, i, j - 1, :])
                b1 = (0.5 * data.dt**2 * A @ A) @ ((data.U[n, i + 1, j, :] + data.U[n, i - 1, j, :] - 2 * data.U[n, i, j, :]) / data.dx ** 2)
                b2 = (0.5 * data.dt**2 * B @ B) @ ((data.U[n, i, j + 1, :] + data.U[n, i, j - 1, :] - 2 * data.U[n, i, j, :]) / data.dy ** 2)
                b3 = 1/(4*data.dx*data.dy) * (0.5 * data.dt**2 * (A @ B + B @ A)) @ (data.U[n,i + 1, j + 1] - data.U[n, i + 1, j - 1] - data.U[n, i - 1, j + 1] + data.U[n, i - 1, j - 1])
                data.U[n + 1, i, j, :] = data.U[n, i, j, :] - a1 - a2 + b1 + b2 + b3 + s


    ec = 0.5 * (r00 * data.U[..., 0] ** 2 + r11 * data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 /(2*e)
    data.E = np.sum(ec + ep, axis=(1, 2))

def ADER4_aniso(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D dans un milieu anisotrope
    :param data: Donnee2D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 2D
    """
    def G(i, j):
        sigma, R = 5*data.dx, 10*data.dx
        x, y = np.abs(data.ps[0][0] - i)*data.dx, np.abs(data.ps[0][1] - j)*data.dy
        return (1/(np.pi*sigma**2)*np.exp(-(x**2 + y**2)/sigma**2))*(0 <= x**2 + y**2 <= R**2)

    sleep(0.01)
    print("ADER4 Anisotrope()")
    sleep(0.01)
    c = cmax(data, False)
    data.CFL_aniso(c)
    data.U = np.zeros((data.N, data.Mx, data.My, 3))

    (r0, r1) = data.rho
    r00 = r0*data.alpha + r1*(1-data.alpha)
    r11 = r0*r1/(data.alpha*r1 + r0*(1-data.alpha))
    e = 1/(data.alpha/data.e[0] + (1-data.alpha)/data.e[1])
    A = np.array([[0, 0, 1/r11],
                  [0, 0, 0],
                  [e, 0, 0]])
    B = np.array([[0, 0, 0],
                  [0, 0, 1/r00],
                  [0, e, 0]])

    coeff = [1/(12*data.dx),1/(12*data.dx**2),1/(2*data.dx**3),1/(144*data.dx*data.dy**2),1/data.dx**4,1/(144*data.dx*data.dy),1/(144*(data.dx*data.dy)**2),1/(24*data.dx**3*data.dy)]
    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(2, data.Mx - 2):
            for j in range(2, data.My - 2):
                s = data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * G(i, j) * np.array([0, 0, 1]).transpose()
                dxU = [0,
                       coeff[0] * (data.U[n,i-2,j,:] - 8*data.U[n,i-1,j,:] + 8*data.U[n,i+1,j,:] - data.U[n,i+2,j,:]),
                       coeff[1] * (- data.U[n,i-2,j,:] + 16*data.U[n,i-1,j,:] - 30*data.U[n,i,j,:] + 16*data.U[n,i+1,j,:] - data.U[n,i+2,j,:]),
                       coeff[2] * (- data.U[n,i-2,j,:] + 2*data.U[n,i-1,j,:] - 2*data.U[n,i+1,j,:] + data.U[n,i+2,j,:]),
                       coeff[4] * (data.U[n,i-2,j,:] - 4*data.U[n,i-1,j,:] + 6*data.U[n,i,j,:] - 4*data.U[n,i+1,j,:] + data.U[n,i+2,j,:])]

                dyU = [0,
                       coeff[0] * (data.U[n,i,j-2,:] - 8*data.U[n,i,j-1,:] + 8*data.U[n,i,j+1,:] - data.U[n,i,j+2,:]),
                       coeff[1] * (- data.U[n,i,j-2,:] + 16*data.U[n,i,j-1,:] - 30*data.U[n,i,j,:] + 16*data.U[n,i,j+1,:] - data.U[n,i,j+2,:]),
                       coeff[2] * (- data.U[n,i,j-2,:] + 2*data.U[n,i,j-1,:] - 2*data.U[n,i,j+1,:] + data.U[n,i,j+2,:]),
                       coeff[4] * (data.U[n,i,j-2,:] - 4*data.U[n,i,j-1,:] + 6*data.U[n,i,j,:] - 4*data.U[n,i,j+1,:] + data.U[n,i,j+2,:])]

                dxyU = [coeff[5] * (data.U[n,i-2, j-2,:] - 8*data.U[n,i-1,j-2,:] + 8*data.U[n,i+1,j-2,:] - data.U[n,i+2,j-2,:]                                     #dxyU
                                    - 8*data.U[n,i-2, j-1,:] + 64*data.U[n,i-1,j-1,:] - 64*data.U[n,i+1,j-1,:] + 8*data.U[n,i+2,j-1,:]
                                    + 8*data.U[n,i-2, j+1,:] - 64*data.U[n,i-1,j+1,:] + 64*data.U[n,i+1,j+1,:] - 8*data.U[n,i+2,j+1,:]
                                    - data.U[n,i-2, j+2,:] + 8*data.U[n,i-1,j+2,:] - 8*data.U[n,i+1,j+2,:] + data.U[n,i+2,j+2,:]),
                        coeff[3] * (- data.U[n,i-2,j-2,:] + 8*data.U[n,i-1,j-2,:] - 8*data.U[n,i+1,j-2,:] + data.U[n,i+2,j-2,:]                                    #dxyyU
                                    + 16*data.U[n,i-2,j-1,:] -128*data.U[n,i-1,j-1,:] + 128*data.U[n,i+1,j-1,:] - 16*data.U[n,i+2,j-1,:]
                                    - 30*data.U[n,i-2,j,:] + 240*data.U[n,i-1,j,:] - 240*data.U[n,i+1,j,:] + 30*data.U[n,i+2,j,:]
                                    + 16*data.U[n,i-2,j+1,:] - 128*data.U[n,i-1,j+1,:] + 128*data.U[n,i+1,j+1,:] - 16*data.U[n,i+2,j+1,:]
                                    - data.U[n,i-2,j+2,:] + 8*data.U[n,i-1,j+2,:] - 8*data.U[n,i+1,j+2,:] + data.U[n,i+2,j+2,:]),
                        coeff[3] * (- data.U[n,i-2,j-2,:] + 8*data.U[n,i-2,j-1,:] - 8*data.U[n,i-2,j+1,:] + data.U[n,i-2,j+2,:]                                    #dyxxU
                                    + 16*data.U[n,i-1,j-2,:] -128*data.U[n,i-1,j-1,:] + 128*data.U[n,i-1,j+1,:] - 16*data.U[n,i-1,j+2,:]
                                    - 30*data.U[n,i,j-2,:] + 240*data.U[n,i,j-1,:] - 240*data.U[n,i,j+1,:] + 30*data.U[n,i,j+2,:]
                                    + 16*data.U[n,i+1,j-2,:] - 128*data.U[n,i+1,j-1,:] + 128*data.U[n,i+1,j+1,:] - 16*data.U[n,i+1,j+2,:]
                                    - data.U[n,i+2,j-2,:] + 8*data.U[n,i+2,j-1,:] - 8*data.U[n,i+2,j+1,:] + data.U[n,i+2,j+2,:]),
                        coeff[6] * (data.U[n,i-2,j-2,:] -16*data.U[n,i-2,j-1,:] +30 * data.U[n,i-2,j,:] - 16*data.U[n,i-2,j+1,:] + data.U[n,i-2,j+2,:]             #dxxyyU
                                    - 16*data.U[n,i-1,j-2,:] +256*data.U[n,i-1,j-1,:]-480*data.U[n,i-1,j,:] + 256*data.U[n,i-1,j+1,:] - 16*data.U[n,i-1,j+2,:]
                                    + 30*data.U[n,i,j-2,:] -480*data.U[n,i,j-1,:] +900*data.U[n,i,j,:] -480*data.U[n,i,j+1,:] + 30*data.U[n,i,j+2,:]
                                    - 16*data.U[n,i+1,j-2,:] +256*data.U[n,i+1,j-1,:]-480*data.U[n,i+1,j,:] + 256*data.U[n,i+1,j+1,:] - 16*data.U[n,i+1,j+2,:]
                                    + data.U[n,i+2,j-2,:] -16*data.U[n,i+2,j-1,:] +30*data.U[n,i+2,j,:] - 16*data.U[n,i+2,j+1,:] + data.U[n,i+2,j+2,:]),
                        coeff[7] * (-data.U[n,i-2,j-2,:] + 8*data.U[n,i-2,j-1,:] - 8*data.U[n,i-2,j+1,:] + data.U[n,i-2,j+2,:]                                             #dxxxyU
                                    + 2*data.U[n,i-1,j-2,:] - 16*data.U[n,i-1,j-1,:] + 16*data.U[n,i-1,j+1,:] - 2*data.U[n,i-1,j+2,:]
                                    - 2*data.U[n,i+1,j-2,:] + 16*data.U[n,i+1,j-1,:] - 16*data.U[n,i+1,j+1,:] + 2*data.U[n,i+1,j+2,:]
                                    + data.U[n,i+2,j-2,:] - 8*data.U[n,i+2,j-1,:] +8*data.U[n,i+2,j+1,:] - data.U[n,i+2,j+2,:]),
                        coeff[7] * (-data.U[n,i-2,j-2,:] + 2*data.U[n,i-2,j-1,:] - 2*data.U[n,i-2,j+1,:] + data.U[n,i-2,j+2,:]                                             #dyyyxU
                                    + 8*data.U[n,i-1,j-2,:] - 16*data.U[n,i-1,j-1,:] + 16*data.U[n,i-1,j+1,:] - 8*data.U[n,i-1,j+2,:]
                                    - 8*data.U[n,i+1,j-2,:] + 16*data.U[n,i+1,j-1,:] - 16*data.U[n,i+1,j+1,:] + 8*data.U[n,i+1,j+2,:]
                                    + data.U[n,i+2,j-2,:] - 2*data.U[n,i+2,j-1,:] +2*data.U[n,i+2,j+1,:] - data.U[n,i+2,j+2,:])
                        ]
                
                a1 = - data.dt* (A @ dxU[1] + B @ dyU[1]) 
                a2 = data.dt**2/2 * (A @ A @ dxU[2] + (A @ B + B @ A) @ dxyU[0] + B @ B @ dyU[2])
                a3 = - data.dt**3/6 * (A @ A @ A @ dxU[3] + B @ B @ B @ dyU[3] + (A @ A @ B + B @ A @ A) @ dxyU[2] + (B @ B @ A + A @ B @ B) @ dxyU[1])
                a4 = data.dt**4/24 * (A @ A @ A @ A @ dxU[4] + B @ B @ B @ B @ dyU[4] + (A @ A @ A @ B + B @ A @ A @ A) @ dxyU[4] + (A @ A @ B @ B + B @ B @ A @ A) @ dxyU[3] + (B @ B @ B @ A + A @ B @ B @ B) @ dxyU[5])
                
                data.U[n + 1, i, j, :] = data.U[n,i,j,:] + a1 + a2 + a3 + a4 + s

    ec = 0.5 * (r00 * data.U[..., 0] ** 2 + r11*data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (2*e)
    data.E = np.sum(ec + ep, axis=(1, 2))