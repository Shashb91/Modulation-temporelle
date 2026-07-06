import numpy as np
from time import sleep
from donnee import Donnee2D
from matplotlib import pyplot as plt
from tqdm import trange
ncols = 125

def A2D_mt(data):
    def f(t):
        rho = data.rho[0]*data.rho_mt[0](data)(t)
        kappa = data.kappa*data.kappa_mt(data)(t)
        A = np.array([[0, 0, 1 / rho[0]],
                      [0, 0, 0],
                      [kappa[0], 0, 0]])
        B = np.array([[0, 0, -rho[1] / rho[0] ** 2],
                      [0, 0, 0],
                      [kappa[1], 0, 0]])
        C = np.array([[0, 0, (rho[2] * rho[0] - 2 * rho[1] ** 2) / rho[0] ** 3],
                      [0, 0, 0],
                      [kappa[2], 0, 0]])
        D = np.array([[0, 0, -(rho[3] * rho[0] ** 2 - 6 * rho[2] * rho[0] * rho[1] + 6 * rho[1] ** 3) / rho[0] ** 4],
                      [0, 0, 0],
                      [kappa[3], 0, 0]])
        return [A, B, C, D]

    return f


def B2D_mt(data):
    def f(t):
        rho = data.rho[1]*data.rho_mt[1](data)(t)
        kappa = data.kappa*data.kappa_mt(data)(t)
        A = np.array([[0, 0, 0],
                      [0, 0, 1 / rho[0]],
                      [0, kappa[0], 0]])
        B = np.array([[0, 0, 0],
                      [0, 0, -rho[1] / rho[0] ** 2],
                      [0, kappa[1], 0]])
        C = np.array([[0, 0, 0],
                      [0, 0, (rho[2] * rho[0] - 2 * rho[1] ** 2) / rho[0] ** 3],
                      [0, kappa[2], 0]])
        D = np.array([[0, 0, 0],
                      [0, 0, -(rho[3] * rho[0] ** 2 - 6 * rho[2] * rho[1] * rho[0] + 6 * rho[1] ** 3) / rho[0] ** 4],
                      [0, kappa[3], 0]])
        return [A, B, C, D]
    return f


def cmax(r00, r11, kappa, opt=True):
    """
    Renvoie la vitesse de phase polaire maximale et trace le profil de vitesse
    :type r00: float, valeur moyenne de rho
    :param r11: float, l'inverse de la valeur moyenne de rho^-1
    :param kappa: float, l'inverse de la valeur moyenne de kappa^-1
    :param opt: bool, True, tracé de la vitesse polaire
    :return: float: vitesse de phase maximale
    """
    c0, c1 = np.sqrt(kappa / r00), np.sqrt(kappa / r11)
    theta = np.linspace(0, 2 * np.pi, 300)
    Vp = np.sqrt(c0 ** 2 * np.cos(theta) ** 2 + c1 ** 2 * np.sin(theta) ** 2)
    x, y = np.cos(theta), np.sin(theta)
    if opt:
        plt.plot(Vp * y, Vp * x, c='blue', lw=2, label='Vitesse polaire', ms=0)
        plt.plot(c0 * y, c0 * x, c='red', lw=2, label=f'$c_{0}$', ms=0)
        plt.plot(c1 * y, c1 * x, c='green', lw=2, label=f'$c_{1}$', ms=0)
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
        sigma, R = 5 * data.dx, 10 * data.dx
        x, y = np.abs(data.ps[0][0] - i) * data.dx, np.abs(data.ps[0][1] - j) * data.dy
        return (1 / (np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)) * (0 <= x ** 2 + y ** 2 <= R ** 2)

    print("\nLaxWendroff Anisotrope()")
    sleep(0.01)
    data.aniso()
    (r00, r11) = data.rho
    kappa = data.kappa
    c = cmax(r00, r11, kappa)
    data.CFL_aniso(c)
    data.U = np.zeros((data.N, data.Mx, data.My, 3))

    A = np.array([[0, 0, 1 / r11],
                  [0, 0, 0],
                  [kappa, 0, 0]])
    B = np.array([[0, 0, 0],
                  [0, 0, 1 / r00],
                  [0, kappa, 0]])

    for n in trange(0, data.N - 1, ncols=ncols):
        for i in range(1, data.Mx - 1):
            for j in range(1, data.My - 1):
                s = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * G(i, j) * np.array(
                    [0, 0, 1]).transpose()
                a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i + 1, j, :] - data.U[n, i - 1, j, :])
                a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, i, j + 1, :] - data.U[n, i, j - 1, :])
                b1 = (0.5 * data.dt ** 2 * A @ A) @ (
                            (data.U[n, i + 1, j, :] + data.U[n, i - 1, j, :] - 2 * data.U[n, i, j, :]) / data.dx ** 2)
                b2 = (0.5 * data.dt ** 2 * B @ B) @ (
                            (data.U[n, i, j + 1, :] + data.U[n, i, j - 1, :] - 2 * data.U[n, i, j, :]) / data.dy ** 2)
                b3 = 1 / (4 * data.dx * data.dy) * (0.5 * data.dt ** 2 * (A @ B + B @ A)) @ (
                            data.U[n, i + 1, j + 1] - data.U[n, i + 1, j - 1] - data.U[n, i - 1, j + 1] + data.U[
                        n, i - 1, j - 1])
                data.U[n + 1, i, j, :] = data.U[n, i, j, :] - a1 - a2 + b1 + b2 + b3 + s

    ec = 0.5 * (r00 * data.U[..., 0] ** 2 + r11 * data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (2 * kappa)
    data.E = np.sum((ec + ep) * data.dx * data.dy, axis=(1, 2))


def ADER4_aniso(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D dans un milieu anisotrope
    :param data: Donnee2D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 2D
    """

    def G(i, j):
        sigma, R = 5 * data.dx, 10 * data.dx
        x, y = np.abs(data.ps[0][0] - i) * data.dx, np.abs(data.ps[0][1] - j) * data.dy
        return (1 / (np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)) * (0 <= x ** 2 + y ** 2 <= R ** 2)

    sleep(0.01)
    print("\nADER4 Anisotrope()")
    sleep(0.01)
    data.aniso()
    (r00, r11) = data.rho
    kappa = data.kappa
    c = cmax(r00, r11, kappa, False)
    data.CFL_aniso(c)
    data.U = np.zeros((data.N, data.Mx, data.My, 3))

    A = np.array([[0, 0, 1 / r11],
                  [0, 0, 0],
                  [kappa, 0, 0]])
    B = np.array([[0, 0, 0],
                  [0, 0, 1 / r00],
                  [0, kappa, 0]])

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

    ec = 0.5 * (r00 * data.U[..., 0] ** 2 + r11 * data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (2 * kappa)
    data.E = np.sum((ec + ep) * data.dx * data.dy, axis=(1, 2))

def LaxWendroff_aniso_mt(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 2D dans un milieu anisotrope
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """

    def G(i, j):
        sigma, R = 5 * data.dx, 10 * data.dx
        x, y = np.abs(data.ps[0][0] - i) * data.dx, np.abs(data.ps[0][1] - j) * data.dy
        return (1 / (np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)) * (0 <= x ** 2 + y ** 2 <= R ** 2)

    print("\nLaxWendroff Anisotrope()")
    sleep(0.01)
    data.aniso()
    (r00, r11) = data.rho
    kappa = data.kappa
    c = cmax(r00, r11, kappa)
    data.CFL_aniso(c)
    data.U = np.zeros((data.N, data.Mx, data.My, 3))

    for n in trange(0, data.N - 1, ncols=ncols):
        t = data.dt * n
        rho1 = data.rho_mt[0](data)
        rho2 = data.rho_mt[1](data)
        kappa = data.kappa_mt(data)
        A, A_ = A2D_mt(data)(t)[0], A2D_mt(data)(t)[1]
        B, B_ = B2D_mt(data)(t)[0], B2D_mt(data)(t)[1]
        U_temp = data.U[n, ...]
        U_temp_ = np.zeros(data.U[n,...].shape)

        S = [-rho1(t)[1]/rho1(t)[0], -rho2(t)[1]/rho2(t)[0], kappa(t)[1]/kappa(t)[0]]
        for i in range(0, data.Mx):
            for j in range(0, data.My):
                U_temp[i, j, :] = np.diag([np.exp(-S[0] * data.dt / 2), np.exp(-S[1] * data.dt / 2), np.exp(-S[2] * data.dt / 2)]) @ data.U[n, i, j, :]

        for i in range(1, data.Mx - 1):
            for j in range(1, data.My - 1):
                s = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * G(i, j) * np.array([0, 0, 1]).transpose()
                a1 = (1 / (2 * data.dx)) * (data.dt* A + data.dt**2/2*A_) @ (U_temp[i + 1, j, :] - U_temp[i - 1, j, :])
                a2 = (1 / (2 * data.dy)) * (data.dt* B + data.dt**2/2*B_) @ (U_temp[i, j + 1, :] - U_temp[i, j - 1, :])
                b1 = (0.5 * data.dt ** 2 * A @ A) @ ((U_temp[i + 1, j, :] + U_temp[i - 1, j, :] - 2 * U_temp[i, j, :]) / data.dx ** 2)
                b2 = (0.5 * data.dt ** 2 * B @ B) @ ((U_temp[i, j + 1, :] + U_temp[i, j - 1, :] - 2 * U_temp[i, j, :]) / data.dy ** 2)
                b3 = 1 / (4 * data.dx * data.dy) * (0.5 * data.dt ** 2 * (A @ B + B @ A)) @ (U_temp[i + 1, j + 1] - U_temp[i + 1, j - 1] - U_temp[i - 1, j + 1] + U_temp[i - 1, j - 1])
                U_temp_[ i, j, :] = U_temp[i, j, :] - a1 - a2 + b1 + b2 + b3 + s

        S = [-rho1(t+data.dt)[1]/rho1(t+data.dt)[0], -rho2(t+data.dt)[1]/rho2(t+data.dt)[0], kappa(t+data.dt)[1]/kappa(t+data.dt)[0]]
        for i in range(0, data.Mx):
            for j in range(0, data.My):
                data.U[n + 1, i, j, :] = np.diag([np.exp(-S[0] * data.dt / 2), np.exp(-S[1] * data.dt / 2), np.exp(-S[2] * data.dt / 2)]) @ U_temp_[i, j, :]

    rho = [data.rho*data.rho_mt(data)(data.dt * n)[0] for n in range(data.N)]
    kappa = [data.kappa*data.kappa_mt(data)(data.dt * n)[0] for n in range(data.N)]
    data.E = [np.sum((0.5 * rho[n] * data.U[n, ..., 0] ** 2 + data.U[n,..., 1]**2 /(2*kappa[n]))*data.dx*data.dy, axis = (0,1)) for n in range(data.N)]