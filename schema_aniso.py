import numpy as np
from time import sleep
from modulation import rho_echelon, kappa_echelon, rho_moy, kappa_moy
from matplotlib.animation import FuncAnimation

from donnee import Donnee2D
from matplotlib import pyplot as plt
from tqdm import trange
ncols = 125
pas_sauvegarde = 25


def moyennes_aniso(data):
    """
    Calcule les valeurs homogénéisées du milieu stratifié à partir des paramètres des deux couches
    :param data: Donnee2D, avec data.rho et data.kappa les couples (couche 0, couche 1)
                 et data.alpha la proportion de la couche 0
    :return: tuple (r00, r11, kappa) des valeurs moyennes homogénéisées
    """
    (rho0, rho1) = data.rho
    (kappa0, kappa1) = data.kappa
    alpha = data.alpha
    r00 = alpha * rho0 + (1 - alpha) * rho1
    r11 = 1 / (alpha / rho0 + (1 - alpha) / rho1)
    kappa = 1 / (alpha / kappa0 + (1 - alpha) / kappa1)
    return r00, r11, kappa


def deriv_inv(a):
    """
    Calcule le développement (valeur + 3 premières dérivées temporelles) de 1/f à partir du développement a = [f, f', f'', f'''] de f.
    :param a: np.ndarray de taille 4, [f, f', f'', f''']
    :return: np.ndarray de taille 4, [1/f, (1/f)', (1/f)'', (1/f)''']
    """
    f, f1, f2, f3 = a[0], a[1], a[2], a[3]
    h = 1 / f
    h1 = -f1 / f ** 2
    h2 = 2 * f1 ** 2 / f ** 3 - f2 / f ** 2
    h3 = -6 * f1 ** 3 / f ** 4 + 6 * f1 * f2 / f ** 3 - f3 / f ** 2
    return np.array([h, h1, h2, h3])


def moyennes_aniso_mt(data, t):
    """
    Valeurs homogénéisées à l'instant t (ordre 0 uniquement), calculées à partir des paramètres des DEUX COUCHES modulés en temps, puis homogénéisés
    :return: tuple (r00(t), r11(t), kappa(t)) des valeurs moyennes à l'instant t
    """
    (rho0, rho1) = data.rho
    (kappa0, kappa1) = data.kappa
    alpha = data.alpha
    r0 = rho0 * data.rho_mt[0](data, data.eps_r[0])(t)[0]
    r1 = rho1 * data.rho_mt[1](data, data.eps_r[1])(t)[0]
    k0 = kappa0 * data.kappa_mt[0](data, data.eps_kappa[0])(t)[0]
    k1 = kappa1 * data.kappa_mt[1](data, data.eps_kappa[1])(t)[0]
    r00 = alpha * r0 + (1 - alpha) * r1
    r11 = 1 / (alpha / r0 + (1 - alpha) / r1)
    kappa = 1 / (alpha / k0 + (1 - alpha) / k1)
    return r00, r11, kappa


def A2D_aniso(data):
    (rho0, rho1) = data.rho
    (kappa0, kappa1) = data.kappa
    alpha = data.alpha
    def f(t):
        rho_c0 = rho0 * data.rho_mt[0](data, data.eps_r[0])(t)
        rho_c1 = rho1 * data.rho_mt[1](data, data.eps_r[1])(t)
        kappa_c0 = kappa0 * data.kappa_mt[0](data, data.eps_kappa[0])(t)
        kappa_c1 = kappa1 * data.kappa_mt[1](data, data.eps_kappa[1])(t)
        rho = deriv_inv(alpha * deriv_inv(rho_c0) + (1 - alpha) * deriv_inv(rho_c1))
        kappa = deriv_inv(alpha * deriv_inv(kappa_c0) + (1 - alpha) * deriv_inv(kappa_c1))
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


def B2D_aniso(data):
    (rho0, rho1) = data.rho
    (kappa0, kappa1) = data.kappa
    alpha = data.alpha
    def f(t):
        rho_c0 = rho0 * data.rho_mt[0](data, data.eps_r[0])(t)
        rho_c1 = rho1 * data.rho_mt[1](data, data.eps_r[1])(t)
        kappa_c0 = kappa0 * data.kappa_mt[0](data, data.eps_kappa[0])(t)
        kappa_c1 = kappa1 * data.kappa_mt[1](data, data.eps_kappa[1])(t)
        rho = alpha * rho_c0 + (1 - alpha) * rho_c1
        kappa = deriv_inv(alpha * deriv_inv(kappa_c0) + (1 - alpha) * deriv_inv(kappa_c1))
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

def A2D_ms(data):
    def f(t):
        rho = (data.rho[0]*data.rho_mt[0](data, data.eps_r[0])(t),data.rho[1]*data.rho_mt[1](data, data.eps_r[1])(t), rho_moy(data)(t))
        kappa = (data.kappa[0]*data.kappa_mt[0](data, data.eps_kappa[0])(t), data.kappa[1]*data.kappa_mt[0](data, data.eps_kappa[1])(t), kappa_moy(data)(t))
        retour = []
        for i in [0,1,2]:
            A = np.array([[0, 0, 1 / rho[i][0]],
                          [0, 0, 0],
                          [kappa[i][0], 0, 0]])
            B = np.array([[0, 0, -rho[i][1] / rho[0][0] ** 2],
                          [0, 0, 0],
                          [kappa[i][1], 0, 0]])
            C = np.array([[0, 0, (rho[i][2] * rho[i][0] - 2 * rho[i][1] ** 2) / rho[i][0] ** 3],
                          [0, 0, 0],
                          [kappa[i][2], 0, 0]])
            D = np.array([[0, 0, -(rho[i][3] * rho[i][0] ** 2 - 6 * rho[i][2] * rho[i][0] * rho[i][1] + 6 * rho[i][1] ** 3) / rho[i][0] ** 4],
                          [0, 0, 0],
                          [kappa[i][3], 0, 0]])
            retour.append([A, B, C, D])
        return retour
    return f


def B2D_ms(data):
    def f(t):
        rho = (data.rho[0]*data.rho_mt[0](data, data.eps_r[0])(t),data.rho[1]*data.rho_mt[1](data, data.eps_r[1])(t), rho_moy(data)(t))
        kappa = (data.kappa[0]*data.kappa_mt[0](data, data.eps_kappa[0])(t), data.kappa[1]*data.kappa_mt[0](data, data.eps_kappa[1])(t), kappa_moy(data)(t))
        retour = []
        for i in [0,1,2]:
            A = np.array([[0, 0, 0],
                          [0, 0, 1 / rho[i][0]],
                          [0,kappa[i][0], 0]])
            B = np.array([[0, 0, 0],
                          [0, 0, -rho[i][1] / rho[0][0] ** 2],
                          [0,kappa[i][1], 0]])
            C = np.array([[0, 0, 0],
                          [0, 0, (rho[i][2] * rho[i][0] - 2 * rho[i][1] ** 2) / rho[i][0] ** 3],
                          [0,kappa[i][2], 0]])
            D = np.array([[0, 0, 0],
                          [0, 0, -(rho[i][3] * rho[i][0] ** 2 - 6 * rho[i][2] * rho[i][0] * rho[i][1] + 6 * rho[i][1] ** 3) / rho[i][0] ** 4],
                          [0, kappa[i][3], 0]])
            retour.append([A, B, C, D])
        return retour
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
        plt.plot(c0 * y, c0 * x, c='red', lw=2, label=f'$c_{1}$', ms=0)
        plt.plot(c1 * y, c1 * x, c='green', lw=2, label=f'$c_{2}$', ms=0)
        plt.grid(True)
        plt.xlabel("Vitesse en x")
        plt.ylabel("Vitesse en y")
        plt.title(f"Vitesse polaire maximale")
        plt.legend()
        plt.axis('square')
        plt.show()

    return np.max(Vp)

def cmax_mt(data, opt = True):
    """
    Renvoie la vitesse de phase polaire maximale et trace le profil de vitesse avec la modulation temporelle
    :param data: Donnee2D, regroupant tous les données du problèmes
    :param opt: bool, True, tracé de la vitesse polaire
    :return: float: vitesse de phase maximale
    """
    rho0, rho1, kappa = [], [], []
    for t in data.t:
        r00_t, r11_t, kappa_t = moyennes_aniso_mt(data, t)
        rho0.append(r00_t)
        rho1.append(r11_t)
        kappa.append(kappa_t)
    rho0, rho1, kappa = np.array(rho0), np.array(rho1), np.array(kappa)
    c0, c1 = np.sqrt(kappa / rho0), np.sqrt(kappa / rho1)
    theta = np.linspace(0, 2 * np.pi, 300)
    Vp = np.zeros((data.N, 300))
    x, y = np.cos(theta), np.sin(theta)

    for i in range(data.N):
        for j in range(300):
            Vp[i, j] = np.sqrt(c0[i] ** 2 * np.cos(theta[j]) ** 2 + c1[i] ** 2 * np.sin(theta[j]) ** 2)

    if opt:
        fig, ax = plt.subplots()
        line, = ax.plot([],[], c = "blue", lw = 2, label = "Vitesse polaire", ms=0)
        linec0, = ax.plot([],[], c = 'red', lw=2, label=f'$c_{0}$', ms=0)
        linec1, = ax.plot([],[], c = 'green', lw=2, label=f'$c_{1}$', ms=0)
        ax.set_xlim(1.1*np.min([c1[n]*x[:] for n in range(data.N)]),1.1*np.max([c1[n]*x[:] for n in range(data.N)]))
        ax.set_ylim(1.1*np.min([c1[n]*y[:] for n in range(data.N)]),1.1*np.max([c1[n]*y[:] for n in range(data.N)]))
        ax.set_xlabel("Vitesse selon x")
        ax.set_ylabel("Vitesse selon y")
        ax.set_title("Vitesse polaire")
        ax.legend()
        ax.grid(True)

        def init():
            line.set_data([], [])
            linec0.set_data([], [])
            linec1.set_data([], [])
            return line, linec0, linec1

        def update(n):
            line.set_data(Vp[n,:]*x[:], Vp[n, :]*y[:])
            linec0.set_data(c0[n]*x[:], c0[n]*y[:])
            linec1.set_data(c1[n]*x[:], c1[n]*y[:])
            return line, linec0, linec1

        anim = FuncAnimation(fig, update, init_func=init, frames=data.N, interval=10, blit=True)
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
    r00, r11, kappa = moyennes_aniso(data)
    c = cmax(r00, r11, kappa)
    data.CFL_aniso(c)
    data.U = np.zeros((data.N, data.Mx, data.My, 3))

    A = np.array([[0, 0, 1 / r11],
                  [0, 0, 0],
                  [kappa, 0, 0]])
    B = np.array([[0, 0, 0],
                  [0, 0, 1 / r00],
                  [0, kappa, 0]])

    MAA = 0.5 * data.dt ** 2 * (A @ A)
    MBB = 0.5 * data.dt ** 2 * (B @ B)
    MAB = 0.5 * data.dt ** 2 * (A @ B + B @ A)
    ii = np.arange(1, data.Mx - 1)[:, None]
    jj = np.arange(1, data.My - 1)[None, :]
    gx, gy = np.abs(data.ps[0][0] - ii) * data.dx, np.abs(data.ps[0][1] - jj) * data.dy
    sigma, R = 5 * data.dx, 10 * data.dx
    r2 = gx ** 2 + gy ** 2
    Gmat = (1 / (np.pi * sigma ** 2) * np.exp(-r2 / sigma ** 2)) * ((0 <= r2) & (r2 <= R ** 2))

    for n in trange(0, data.N - 1, ncols=ncols):
        Un = data.U[n]
        xp, xm = Un[2:, 1:-1, :], Un[:-2, 1:-1, :]
        yp, ym = Un[1:-1, 2:, :], Un[1:-1, :-2, :]
        c0 = Un[1:-1, 1:-1, :]
        cross = Un[2:, 2:, :] - Un[2:, :-2, :] - Un[:-2, 2:, :] + Un[:-2, :-2, :]
        a1 = (data.dt / (2 * data.dx)) * ((xp - xm) @ A.T)
        a2 = (data.dt / (2 * data.dy)) * ((yp - ym) @ B.T)
        b1 = ((xp + xm - 2 * c0) / data.dx ** 2) @ MAA.T
        b2 = ((yp + ym - 2 * c0) / data.dy ** 2) @ MBB.T
        b3 = 1 / (4 * data.dx * data.dy) * (cross @ MAB.T)
        s = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * Gmat[..., None] * np.array([0, 0, 1])
        data.U[n + 1, 1:-1, 1:-1, :] = c0 - a1 - a2 + b1 + b2 + b3 + s

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
    r00, r11, kappa = moyennes_aniso(data)
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
    ii = np.arange(2, data.Mx - 2)[:, None]
    jj = np.arange(2, data.My - 2)[None, :]
    gx, gy = np.abs(data.ps[0][0] - ii) * data.dx, np.abs(data.ps[0][1] - jj) * data.dy
    sigma, R = 5 * data.dx, 10 * data.dx
    r2 = gx ** 2 + gy ** 2
    Gmat = (1 / (np.pi * sigma ** 2) * np.exp(-r2 / sigma ** 2)) * ((0 <= r2) & (r2 <= R ** 2))

    for n in trange(0, data.N - 1, ncols = ncols):
        Un = data.U[n]
        def sh(pp, qq):
            return Un[2 + pp:data.Mx - 2 + pp, 2 + qq:data.My - 2 + qq, :]
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
                            + sh(2,-2) -16*sh(2,-1) +30*sh(2,0) - 16*sh(2,1) + sh(2,2)),
                coeff[7] * (-sh(-2,-2) + 8*sh(-2,-1) - 8*sh(-2,1) + sh(-2,2)
                            + 2*sh(-1,-2) - 16*sh(-1,-1) + 16*sh(-1,1) - 2*sh(-1,2)
                            - 2*sh(1,-2) + 16*sh(1,-1) - 16*sh(1,1) + 2*sh(1,2)
                            + sh(2,-2) - 8*sh(2,-1) + 8*sh(2,1) - sh(2,2)),
                coeff[7] * (-sh(-2,-2) + 2*sh(-2,-1) - 2*sh(-2,1) + sh(-2,2)
                            + 8*sh(-1,-2) - 16*sh(-1,-1) + 16*sh(-1,1) - 8*sh(-1,2)
                            - 8*sh(1,-2) + 16*sh(1,-1) - 16*sh(1,1) + 8*sh(1,2)
                            + sh(2,-2) - 2*sh(2,-1) + 2*sh(2,1) - sh(2,2))]

        a1 = - data.dt * (ap(A, dxU[1]) + ap(B, dyU[1]))
        a2 = data.dt**2/2 * (ap(A @ A, dxU[2]) + ap(A @ B + B @ A, dxyU[0]) + ap(B @ B, dyU[2]))
        a3 = - data.dt**3/6 * (ap(A @ A @ A, dxU[3]) + ap(B @ B @ B, dyU[3]) + ap(A @ A @ B + B @ A @ A, dxyU[2]) + ap(B @ B @ A + A @ B @ B, dxyU[1]))
        a4 = data.dt**4/24 * (ap(A @ A @ A @ A, dxU[4]) + ap(B @ B @ B @ B, dyU[4]) + ap(A @ A @ A @ B + B @ A @ A @ A, dxyU[4]) + ap(A @ A @ B @ B + B @ B @ A @ A, dxyU[3]) + ap(B @ B @ B @ A + A @ B @ B @ B, dxyU[5]))

        s = data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * Gmat[..., None] * np.array([0, 0, 1])
        data.U[n + 1, 2:-2, 2:-2, :] = sh(0,0) + a1 + a2 + a3 + a4 + s

    ec = 0.5 * (r00 * data.U[..., 0] ** 2 + r11 * data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (2 * kappa)
    data.E = np.sum((ec + ep) * data.dx * data.dy, axis=(1, 2))

def LaxWendroff_ms(data, l, L):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 2D dans un milieu anisotrope micro structuré modulé en temps
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :param l: float, correspond à la largeur de la première couche, en m
    :param L: float, correspond à la largeur de la deuxieème couche, en m
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    l, L = l//data.dy, L//data.dy

    def G(i, j):
        sigma, R = 5 * data.dx, 10 * data.dx
        x, y = np.abs(data.ps[0][0] - i) * data.dx, np.abs(data.ps[0][1] - j) * data.dy
        return (1 / (np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)) * (0 <= x ** 2 + y ** 2 <= R ** 2)

    print("\nLaxWendroff Micro Structuré()")
    sleep(0.01)
    (rho0, rho1) = data.rho
    (kappa0, kappa1) = data.kappa
    data.U = np.zeros((data.N, data.Mx, data.My, 3))
    rho2, kappa2 = data.alpha*rho0 + (1 -data.alpha)*rho1, 1/(data.alpha/kappa0 + (1-data.alpha)/kappa1)
    A = [np.array([[0, 0, 1 / rho0], [0, 0, 0], [kappa0, 0, 0]]), np.array([[0, 0, 1 / rho1], [0, 0, 0], [kappa1, 0, 0]]), np.array([[0,0,1/rho2],[0,0,0],[kappa2,0, 0]])]
    B = [np.array([[0, 0, 0], [0, 0, 1 / rho0], [0, kappa0, 0]]), np.array([[0, 0, 0], [0, 0, 1 / rho1], [0, kappa1, 0]]), np.array([[0,0,0],[0,0,1/rho2],[0, kappa2,0]])]

    sigma, R = 5 * data.dx, 10 * data.dx
    ii = np.arange(1, data.Mx - 1)[:, None]
    jc = np.arange(1, data.My - 1)[None, :]
    gx, gy = np.abs(data.ps[0][0] - ii) * data.dx, np.abs(data.ps[0][1] - jc) * data.dy
    r2 = gx ** 2 + gy ** 2
    Gmat = (1 / (np.pi * sigma ** 2) * np.exp(-r2 / sigma ** 2)) * ((0 <= r2) & (r2 <= R ** 2))

    # selection de materiau par colonne j (le milieu ne depend que de j)
    def bande(js):
        m = (js + L // 2) % (l + L)
        return np.where((1 <= m) & (m < L - 1), 0, np.where((m > L + 1) & (m <= l + L - 2), 1, 2))
    band = bande(np.arange(1, data.My - 1))
    Aarr, Barr = np.array(A), np.array(B)
    AA = np.array([A[k] @ A[k] for k in range(3)])
    BB = np.array([B[k] @ B[k] for k in range(3)])
    ABm = np.array([A[k] @ B[k] + B[k] @ A[k] for k in range(3)])
    A_j, B_j, AA_j, BB_j, AB_j = Aarr[band], Barr[band], AA[band], BB[band], ABm[band]

    def apj(Mj, d):
        return np.einsum('jkl,ijl->ijk', Mj, d)

    for n in trange(0, data.N - 1, ncols=ncols):
        Un = data.U[n]
        xp, xm = Un[2:, 1:-1, :], Un[:-2, 1:-1, :]
        yp, ym = Un[1:-1, 2:, :], Un[1:-1, :-2, :]
        c0 = Un[1:-1, 1:-1, :]
        cross = Un[2:, 2:, :] - Un[2:, :-2, :] - Un[:-2, 2:, :] + Un[:-2, :-2, :]
        a1 = (data.dt / (2 * data.dx)) * apj(A_j, xp - xm)
        a2 = (data.dt / (2 * data.dy)) * apj(B_j, yp - ym)
        b1 = 0.5 * data.dt ** 2 * apj(AA_j, (xp + xm - 2 * c0) / data.dx ** 2)
        b2 = 0.5 * data.dt ** 2 * apj(BB_j, (yp + ym - 2 * c0) / data.dy ** 2)
        b3 = 1 / (4 * data.dx * data.dy) * 0.5 * data.dt ** 2 * apj(AB_j, cross)
        s = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * Gmat[..., None] * np.array([0, 0, 1])
        data.U[n + 1, 1:-1, 1:-1, :] = c0 - a1 - a2 + b1 + b2 + b3 + s

    band_all = (np.arange(data.My) + L // 2) % (l + L)
    sel0 = (1 <= band_all) & (band_all < L - 1)
    sel1 = (band_all > L + 1) & (band_all <= l + L - 2)
    rho_col = np.where(sel0, rho0, np.where(sel1, rho1, rho2))
    kap_col = np.where(sel0, kappa0, np.where(sel1, kappa1, kappa2))
    e = (0.5 * rho_col * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2) + 0.5 * data.U[..., 2] ** 2 / kap_col) * data.dx * data.dy
    data.E = np.sum(e, axis=(1, 2))
            
def ADER4_ms(data, l, L):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D dans un milieu anisotrope micro structuré
    :param data: Donnee2D, regroupe l'ensemble des données du probleme
    :param L: float, correspond à la largeur de la première couche, en m
    :param l: float, correspond à la largeur de la deuxieème couche, en m
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    l, L = l//data.dy, L//data.dy
    data.alpha = L/(l+L)

    def G(i, j):
        sigma, R = 5 * data.dx, 10 * data.dx
        x, y = np.abs(data.ps[0][0] - i) * data.dx, np.abs(data.ps[0][1] - j) * data.dy
        return (1 / (np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)) * (0 <= x ** 2 + y ** 2 <= R ** 2)

    sleep(0.01)
    print("\nADER4 Micro Structuré()")
    sleep(0.01)
    (rho0, rho1) = data.rho
    (kappa0, kappa1) = data.kappa
    data.U = np.zeros((data.N, data.Mx, data.My, 3))
    rho2, kappa2 = data.alpha*rho0 + (1 -data.alpha)*rho1, 1/(data.alpha/kappa0 + (1-data.alpha)/kappa1)
    A = [np.array([[0, 0, 1 / rho0], [0, 0, 0], [kappa0, 0, 0]]), np.array([[0, 0, 1 / rho1], [0, 0, 0], [kappa1, 0, 0]]), np.array([[0,0,1/rho2],[0,0,0],[kappa2,0, 0]])]
    B = [np.array([[0, 0, 0], [0, 0, 1 / rho0], [0, kappa0, 0]]), np.array([[0, 0, 0], [0, 0, 1 / rho1], [0, kappa1, 0]]), np.array([[0,0,0],[0,0,1/rho2],[0, kappa2,0]])]

    coeff = [1/(12*data.dx),1/(12*data.dx**2),1/(2*data.dx**3),1/(144*data.dx*data.dy**2),1/data.dx**4,1/(144*data.dx*data.dy),1/(144*(data.dx*data.dy)**2),1/(24*data.dx**3*data.dy)]
    sigma, R = 5 * data.dx, 10 * data.dx
    ii = np.arange(2, data.Mx - 2)[:, None]
    jc = np.arange(2, data.My - 2)[None, :]
    gx, gy = np.abs(data.ps[0][0] - ii) * data.dx, np.abs(data.ps[0][1] - jc) * data.dy
    r2 = gx ** 2 + gy ** 2
    Gmat = (1 / (np.pi * sigma ** 2) * np.exp(-r2 / sigma ** 2)) * ((0 <= r2) & (r2 <= R ** 2))

    def bande(js):
        m = (js + L // 2) % (l + L)
        return np.where((1 <= m) & (m < L - 1), 0, np.where((m > L + 1) & (m <= l + L - 2), 1, 2))
    band = bande(np.arange(2, data.My - 2))

    def st(mats):
        return np.array([mats(k) for k in range(3)])[band]
    a1A = st(lambda k: A[k]); a1B = st(lambda k: B[k])
    a2AA = st(lambda k: A[k] @ A[k]); a2BB = st(lambda k: B[k] @ B[k])
    a2AB = st(lambda k: A[k] @ B[k] + B[k] @ A[k])
    a3AAA = st(lambda k: A[k] @ A[k] @ A[k]); a3BBB = st(lambda k: B[k] @ B[k] @ B[k])
    a3AAB = st(lambda k: A[k] @ A[k] @ B[k] + B[k] @ A[k] @ A[k])
    a3BBA = st(lambda k: B[k] @ B[k] @ A[k] + A[k] @ B[k] @ B[k])
    a4AAAA = st(lambda k: A[k] @ A[k] @ A[k] @ A[k]); a4BBBB = st(lambda k: B[k] @ B[k] @ B[k] @ B[k])
    a4AAAB = st(lambda k: A[k] @ A[k] @ A[k] @ B[k] + B[k] @ A[k] @ A[k] @ A[k])
    a4AABB = st(lambda k: A[k] @ A[k] @ B[k] @ B[k] + B[k] @ B[k] @ A[k] @ A[k])
    a4BBBA = st(lambda k: B[k] @ B[k] @ B[k] @ A[k] + A[k] @ B[k] @ B[k] @ B[k])

    def apj(Mj, d):
        return np.einsum('jkl,ijl->ijk', Mj, d)

    for n in trange(0, data.N - 1, ncols = ncols):
        Un = data.U[n]
        def sh(pp, qq):
            return Un[2 + pp:data.Mx - 2 + pp, 2 + qq:data.My - 2 + qq, :]
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
                            + sh(2,-2) -16*sh(2,-1) +30*sh(2,0) - 16*sh(2,1) + sh(2,2)),
                coeff[7] * (-sh(-2,-2) + 8*sh(-2,-1) - 8*sh(-2,1) + sh(-2,2)
                            + 2*sh(-1,-2) - 16*sh(-1,-1) + 16*sh(-1,1) - 2*sh(-1,2)
                            - 2*sh(1,-2) + 16*sh(1,-1) - 16*sh(1,1) + 2*sh(1,2)
                            + sh(2,-2) - 8*sh(2,-1) + 8*sh(2,1) - sh(2,2)),
                coeff[7] * (-sh(-2,-2) + 2*sh(-2,-1) - 2*sh(-2,1) + sh(-2,2)
                            + 8*sh(-1,-2) - 16*sh(-1,-1) + 16*sh(-1,1) - 8*sh(-1,2)
                            - 8*sh(1,-2) + 16*sh(1,-1) - 16*sh(1,1) + 8*sh(1,2)
                            + sh(2,-2) - 2*sh(2,-1) + 2*sh(2,1) - sh(2,2))]

        a1 = - data.dt * (apj(a1A, dxU[1]) + apj(a1B, dyU[1]))
        a2 = data.dt**2/2 * (apj(a2AA, dxU[2]) + apj(a2AB, dxyU[0]) + apj(a2BB, dyU[2]))
        a3 = - data.dt**3/6 * (apj(a3AAA, dxU[3]) + apj(a3BBB, dyU[3]) + apj(a3AAB, dxyU[2]) + apj(a3BBA, dxyU[1]))
        a4 = data.dt**4/24 * (apj(a4AAAA, dxU[4]) + apj(a4BBBB, dyU[4]) + apj(a4AAAB, dxyU[4]) + apj(a4AABB, dxyU[3]) + apj(a4BBBA, dxyU[5]))

        s = data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * Gmat[..., None] * np.array([0, 0, 1])
        data.U[n + 1, 2:-2, 2:-2, :] = sh(0,0) + a1 + a2 + a3 + a4 + s

    band_all = (np.arange(data.My) + L // 2) % (l + L)
    sel0 = (1 <= band_all) & (band_all < L - 1)
    sel1 = (band_all > L + 1) & (band_all <= l + L - 2)
    rho_col = np.where(sel0, rho0, np.where(sel1, rho1, rho2))
    kap_col = np.where(sel0, kappa0, np.where(sel1, kappa1, kappa2))
    e = (0.5 * rho_col * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2) + 0.5 * data.U[..., 2] ** 2 / kap_col) * data.dx * data.dy
    data.E = np.sum(e, axis=(1, 2))

"""
Milieu anisotrope modulé en temps
"""

def LaxWendroff_aniso_mt(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 2D dans un milieu anisotrope modulé en temps
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """

    def G(i, j):
        sigma, R = 5 * data.dx, 10 * data.dx
        x, y = np.abs(data.ps[0][0] - i) * data.dx, np.abs(data.ps[0][1] - j) * data.dy
        return (1 / (np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)) * (0 <= x ** 2 + y ** 2 <= R ** 2)

    print("\nLaxWendroff Anisotrope Modulation Temporelle()")
    sleep(0.01)
    c = cmax_mt(data, opt = False)
    data.CFL_aniso(c)
    data.U = np.zeros((data.N, data.Mx, data.My, 3))

    (rho01, rho02) = data.rho
    (kappa01, kappa02) = data.kappa
    alpha = data.alpha

    sigma, R = 5 * data.dx, 10 * data.dx
    ii = np.arange(1, data.Mx - 1)[:, None]
    jj = np.arange(1, data.My - 1)[None, :]
    gx, gy = np.abs(data.ps[0][0] - ii) * data.dx, np.abs(data.ps[0][1] - jj) * data.dy
    r2 = gx ** 2 + gy ** 2
    Gmat = (1 / (np.pi * sigma ** 2) * np.exp(-r2 / sigma ** 2)) * ((0 <= r2) & (r2 <= R ** 2))

    for n in trange(0, data.N - 1, ncols=ncols):
        t = data.dt * n
        rho_c1 = rho01 * data.rho_mt[0](data, data.eps_r[0])(t)
        rho_c2 = rho02 * data.rho_mt[1](data, data.eps_r[1])(t)
        kappa_c1 = kappa01 * data.kappa_mt[0](data, data.eps_kappa[0])(t)
        kappa_c2 = kappa02 * data.kappa_mt[1](data, data.eps_kappa[1])(t)
        rho1 = deriv_inv(alpha * deriv_inv(rho_c1) + (1 - alpha) * deriv_inv(rho_c2))
        rho2 = alpha * rho_c1 + (1 - alpha) * rho_c2
        kappa = deriv_inv(alpha * deriv_inv(kappa_c1) + (1 - alpha) * deriv_inv(kappa_c2))
        A, A_ = A2D_aniso(data)(t)[0], A2D_aniso(data)(t)[1]
        B, B_ = B2D_aniso(data)(t)[0], B2D_aniso(data)(t)[1]
        U_temp = data.U[n, ...]
        U_temp_ = np.zeros(data.U[n,...].shape)

        S = [-rho1[1]/rho1[0], -rho2[1]/rho2[0], kappa[1]/kappa[0]]
        d0 = np.array([np.exp(-S[0] * data.dt / 2), np.exp(-S[1] * data.dt / 2), np.exp(-S[2] * data.dt / 2)])
        U_temp[:] = U_temp * d0

        MA = data.dt * A + data.dt**2/2 * A_
        MB = data.dt * B + data.dt**2/2 * B_
        MAA = 0.5 * data.dt ** 2 * (A @ A)
        MBB = 0.5 * data.dt ** 2 * (B @ B)
        MAB = 0.5 * data.dt ** 2 * (A @ B + B @ A)
        xp, xm = U_temp[2:, 1:-1, :], U_temp[:-2, 1:-1, :]
        yp, ym = U_temp[1:-1, 2:, :], U_temp[1:-1, :-2, :]
        c0 = U_temp[1:-1, 1:-1, :]
        cross = U_temp[2:, 2:, :] - U_temp[2:, :-2, :] - U_temp[:-2, 2:, :] + U_temp[:-2, :-2, :]
        a1 = (1 / (2 * data.dx)) * ((xp - xm) @ MA.T)
        a2 = (1 / (2 * data.dy)) * ((yp - ym) @ MB.T)
        b1 = ((xp + xm - 2 * c0) / data.dx ** 2) @ MAA.T
        b2 = ((yp + ym - 2 * c0) / data.dy ** 2) @ MBB.T
        b3 = 1 / (4 * data.dx * data.dy) * (cross @ MAB.T)
        s = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * Gmat[..., None] * np.array([0, 0, 1])
        U_temp_[1:-1, 1:-1, :] = c0 - a1 - a2 + b1 + b2 + b3 + s

        t += data.dt
        rho_c1 = rho01 * data.rho_mt[0](data, data.eps_r[0])(t)
        rho_c2 = rho02 * data.rho_mt[1](data, data.eps_r[1])(t)
        kappa_c1 = kappa01 * data.kappa_mt[0](data, data.eps_kappa[0])(t)
        kappa_c2 = kappa02 * data.kappa_mt[1](data, data.eps_kappa[1])(t)
        rho1 = deriv_inv(alpha * deriv_inv(rho_c1) + (1 - alpha) * deriv_inv(rho_c2))
        rho2 = alpha * rho_c1 + (1 - alpha) * rho_c2
        kappa = deriv_inv(alpha * deriv_inv(kappa_c1) + (1 - alpha) * deriv_inv(kappa_c2))

        S = [-rho1[1]/rho1[0], -rho2[1]/rho2[0], kappa[1]/kappa[0]]
        d1 = np.array([np.exp(-S[0] * data.dt / 2), np.exp(-S[1] * data.dt / 2), np.exp(-S[2] * data.dt / 2)])
        data.U[n + 1, ...] = U_temp_ * d1

    moy = [moyennes_aniso_mt(data, data.dt * n) for n in range(data.N)]
    rho0 = [m[0] for m in moy]
    rho1 = [m[1] for m in moy]
    kappa = [m[2] for m in moy]
    data.E = [np.sum((0.5 * (rho0[n] * data.U[n, ..., 0] ** 2 + rho1[n] * data.U[n, ..., 1]**2) + data.U[n,..., 2]**2 /(2*kappa[n]))*data.dx*data.dy, axis = (0,1)) for n in range(data.N)]

def ADER4_aniso_mt(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D
    :param data: Donnee2D, regroupe l'ensemble des données du probleme 2D
    :return: np.ndarray(), solution en vitesse et pression du problème 2D
    """

    def G(i, j):
        sigma, R = 5 * data.dx, 10 * data.dx
        x, y = np.abs(data.ps[0][0] - i) * data.dx, np.abs(data.ps[0][1] - j) * data.dy
        return (1 / (np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)) * (0 <= x ** 2 + y ** 2 <= R ** 2)

    sleep(0.01)
    print("\nADER4 2D Ansiotrope Modulation Temporelle()")
    sleep(0.01)
    c = cmax_mt(data, opt = False)
    data.CFL_aniso(c)


    sous_echantillonnage = data.N > 1000
    if sous_echantillonnage:
        N_frames = data.N // pas_sauvegarde
        data.U = np.zeros((N_frames, data.Mx, data.My, 3))
        Un = np.zeros((data.Mx, data.My, 3))          # solution au pas de temps n
        Un1 = np.zeros((data.Mx, data.My, 3))         # solution au pas de temps n+1
    else:
        data.U = np.zeros((data.N, data.Mx, data.My, 3))
    coeff = [1/(12*data.dx),1/(12*data.dx**2),1/(2*data.dx**3),1/(144*data.dx*data.dy**2),1/data.dx**4,1/(144*data.dx*data.dy),1/(144*(data.dx*data.dy)**2),1/(24*data.dx**3*data.dy)]

    sigma, R = 5 * data.dx, 10 * data.dx
    ii = np.arange(2, data.Mx - 2)[:, None]
    jj = np.arange(2, data.My - 2)[None, :]
    gx, gy = np.abs(data.ps[0][0] - ii) * data.dx, np.abs(data.ps[0][1] - jj) * data.dy
    r2 = gx ** 2 + gy ** 2
    Gmat = (1 / (np.pi * sigma ** 2) * np.exp(-r2 / sigma ** 2)) * ((0 <= r2) & (r2 <= R ** 2))

    for n in trange(0, data.N - 1, ncols = ncols):
        t = n * data.dt
        rho1 = data.rho_mt[0](data, data.eps_r[0])
        rho2 = data.rho_mt[1](data, data.eps_r[1])
        kappa1 = data.kappa_mt[0](data, data.eps_kappa[0])
        kappa2 = data.kappa_mt[1](data, data.eps_kappa[1])
        kappa = kappa_moy(data)   # module homogeneise module en temps (cf. A2D_aniso)
        A = A2D_aniso(data)(t)
        B = B2D_aniso(data)(t)
        if sous_echantillonnage:
            if n % pas_sauvegarde == 0 and n // pas_sauvegarde < N_frames:
                data.U[n // pas_sauvegarde, ...] = Un
            U_temp = Un
        else:
            U_temp = data.U[n, ...]
        U_temp_ = np.zeros((data.Mx, data.My, 3))

        S = [-rho1(t)[1]/rho1(t)[0], -rho2(t)[1]/rho2(t)[0], kappa(t)[1]/kappa(t)[0]]
        d0 = np.array([np.exp(-S[0] * data.dt / 2), np.exp(-S[1] * data.dt / 2), np.exp(-S[2] * data.dt / 2)])
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
                            + sh(2,-2) -16*sh(2,-1) +30*sh(2,0) - 16*sh(2,1) + sh(2,2)),
                coeff[7] * (-sh(-2,-2) + 8*sh(-2,-1) - 8*sh(-2,1) + sh(-2,2)
                            + 2*sh(-1,-2) - 16*sh(-1,-1) + 16*sh(-1,1) - 2*sh(-1,2)
                            - 2*sh(1,-2) + 16*sh(1,-1) - 16*sh(1,1) + 2*sh(1,2)
                            + sh(2,-2) - 8*sh(2,-1) + 8*sh(2,1) - sh(2,2)),
                coeff[7] * (-sh(-2,-2) + 2*sh(-2,-1) - 2*sh(-2,1) + sh(-2,2)
                            + 8*sh(-1,-2) - 16*sh(-1,-1) + 16*sh(-1,1) - 8*sh(-1,2)
                            - 8*sh(1,-2) + 16*sh(1,-1) - 16*sh(1,1) + 8*sh(1,2)
                            + sh(2,-2) - 2*sh(2,-1) + 2*sh(2,1) - sh(2,2))]

        a1 = - data.dt * (ap(A[0], dxU[1]) + ap(B[0], dyU[1]))
        a2 = data.dt**2/2 * (- ap(A[1], dxU[1]) - ap(B[1], dyU[1]) + ap(A[0] @ A[0], dxU[2]) + ap(B[0] @ B[0], dyU[2]) + ap(A[0] @ B[0] + B[0] @ A[0], dxyU[0]))
        a3 = data.dt**3/6 * (- ap(A[2], dxU[1]) - ap(B[2], dyU[1]) + ap(2 * A[1] @ A[0] + A[0] @ A[1], dxU[2]) + ap(2 * B[1] @ B[0] + B[0] @ B[1], dyU[2])
                             + ap(2 * A[1] @ B[0] + A[0] @ B[1] + 2 * B[1] @ A[0] + B[0] @ A[1], dxyU[0])
                             - ap(A[0] @ A[0] @ B[0] + B[0] @ A[0] @ A[0], dxyU[2]) - ap(B[0] @ B[0] @ A[0] + A[0] @ B[0] @ B[0], dxyU[1])
                             - ap(A[0] @ A[0] @ A[0], dxU[3]) - ap(B[0] @ B[0] @ B[0], dyU[3]))
        a4 = data.dt**4/24 * (- ap(A[3], dxU[1]) - ap(B[3], dyU[1])
                              + ap(3 * A[2] @ B[0] + 3 * A[1] @ B[1] + A[0] @ B[2] + 3 * B[2] @ A[0] + 3 * B[1] @ A[1] + B[0] @ A[2], dxyU[0])
                              + ap(3 * A[2] @ A[0] + 3 * A[1] @ A[1] + A[0] @ A[2], dxU[2]) + ap(3 * B[2] @ B[0] + 3 * B[1] @ B[1] + B[0] @ B[2], dyU[2])
                              - ap(3 * A[1] @ A[0] @ A[0] + 2 * A[0] @ A[1] @ A[0] + A[0] @ A[0] @ A[1], dxU[3])
                              - ap(3 * B[1] @ B[0] @ B[0] + 2 * B[0] @ B[1] @ B[0] + B[0] @ B[0] @ B[1], dyU[3])
                              - ap(3 * A[1] @ A[0] @ B[0] + 2 * A[0] @ A[1] @ B[0] + A[0] @ A[0] @ B[1] + 3 * B[0] @ B[0] @ A[1] + 2 * B[0] @ A[1] @ A[0] + B[0] @ A[0] @ A[1], dxyU[2])
                              - ap(3 * B[1] @ B[0] @ A[0] + 2 * B[0] @ B[1] @ A[0] + B[0] @ B[0] @ A[1] + 3 * A[0] @ A[0] @ B[1] + 2 * A[0] @ B[1] @ B[0] + A[0] @ B[0] @ B[1], dxyU[1])
                              + ap(A[0] @ A[0] @ A[0] @ B[0] + B[0] @ A[0] @ A[0] @ A[0], dxyU[4]) + ap(B[0] @ B[0] @ B[0] @ A[0] + A[0] @ B[0] @ B[0] @ B[0], dxyU[5])
                              + ap(A[0] @ A[0] @ A[0] @ A[0], dxU[4]) + ap(B[0] @ B[0] @ B[0] @ B[0], dyU[4]))

        s = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * Gmat[..., None] * np.array([0, 0, 1])
        U_temp_[2:-2, 2:-2, :] = sh(0,0) + a1 + a2 + a3 + a4 + s

        S = [-rho1(t+data.dt)[1]/rho1(t+data.dt)[0], -rho2(t+data.dt)[1]/rho2(t+data.dt)[0], kappa(t+data.dt)[1]/kappa(t+data.dt)[0]]
        d1 = np.array([np.exp(-S[0] * data.dt / 2), np.exp(-S[1] * data.dt / 2), np.exp(-S[2] * data.dt / 2)])
        if sous_echantillonnage:
            Un1[...] = U_temp_ * d1
            Un, Un1 = Un1, Un
        else:
            data.U[n + 1, ...] = U_temp_ * d1

    if sous_echantillonnage:
        data.t = data.t[:N_frames * pas_sauvegarde:pas_sauvegarde]
        data.dt = data.dt * pas_sauvegarde
        data.N = N_frames
        moy = [moyennes_aniso_mt(data, data.t[n]) for n in range(data.N)]
    else:
        moy = [moyennes_aniso_mt(data, data.dt * n) for n in range(data.N)]
    rho0 = [m[0] for m in moy]
    rho1 = [m[1] for m in moy]
    kappa = [m[2] for m in moy]
    data.E = [np.sum((0.5 * (rho0[n] * data.U[n, ..., 0] ** 2 + rho1[n] * data.U[n, ..., 1]**2) + data.U[n,..., 2]**2 /(2*kappa[n]))*data.dx*data.dy, axis = (0,1)) for n in range(data.N)]

def LaxWendroff_ms_mt(data, l, L):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 2D dans un milieu anisotrope modulé en temps micro structuré
    FONCTIONNEL UNIQUEMENT AVEC UNE MODULATION TEMPORELLE EN ECHELON !!!
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :param l: float, correspond à la largeur de la première couche, en m
    :param L: float, correspond à la largeur de la deuxieème couche, en m
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    l, L = l//data.dy, L//data.dy
    rho1 = data.rho_mt[0](data, data.eps_r[0])
    rho2 = data.rho_mt[1](data, data.eps_r[1])
    rho3 = rho_moy(data)
    kappa1 = data.kappa_mt[0](data, data.eps_kappa[0])
    kappa2 = data.kappa_mt[1](data, data.eps_kappa[1])
    kappa3 = kappa_moy(data)
    c1 = np.array([np.sqrt(data.kappa[0]*kappa1(data.t[n])[0]/(data.rho[0]*rho1(data.t[n])[0])) for n in range(data.N)])
    c2 = np.array([np.sqrt(data.kappa[1]*kappa2(data.t[n])[0]/(data.rho[1]*rho2(data.t[n])[0])) for n in range(data.N)])
    c = max(np.max(c1), np.max(c2))
    data.CFL_maj(c = c)

    def G(i, j):
        sigma, R = 3 * data.dx, 6 * data.dx
        x, y = np.abs(data.ps[0][0] - i) * data.dx, np.abs(data.ps[0][1] - j) * data.dy
        return (1 / (np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)) * (0 <= x ** 2 + y ** 2 <= R ** 2)

    print("\nLaxWendroff Micro Structuré Modulation Temporelle()")
    print(data.N)


    sous_echantillonnage = data.N > 1000
    if sous_echantillonnage:
        N_frames = data.N // pas_sauvegarde
        data.U = np.zeros((N_frames, data.Mx, data.My, 3))
        Un = np.zeros((data.Mx, data.My, 3))          # solution au pas de temps n
        Un1 = np.zeros((data.Mx, data.My, 3))         # solution au pas de temps n+1
    else:
        data.U = np.zeros((data.N, data.Mx, data.My, 3))

    m_all = (np.arange(data.My) + L // 2) % (l + L)
    band_in = np.where(m_all <= L, 0, np.where((m_all > L + 1) & (m_all <= l + L - 2), 1, 2))
    band_out = np.where((1 <= m_all) & (m_all <= L), 0, np.where((m_all > L + 1) & (m_all <= l + L - 2), 1, 2))
    m_st = (np.arange(1, data.My - 1) + L // 2) % (l + L)
    band_st = np.where((1 <= m_st) & (m_st < L - 1), 0, np.where((m_st > L + 1) & (m_st <= l + L - 2), 1, 2))
    sigma, R = 3 * data.dx, 6 * data.dx
    ii = np.arange(1, data.Mx - 1)[:, None]
    jc = np.arange(1, data.My - 1)[None, :]
    gx, gy = np.abs(data.ps[0][0] - ii) * data.dx, np.abs(data.ps[0][1] - jc) * data.dy
    r2 = gx ** 2 + gy ** 2
    Gmat = (1 / (np.pi * sigma ** 2) * np.exp(-r2 / sigma ** 2)) * ((0 <= r2) & (r2 <= R ** 2))

    for n in trange(0, data.N - 1, ncols=ncols):
        t = data.dt * n
        #ATTENTION A_ et B_ SONT LES MATRICES DANS LE MILIEU 2, PAS LA DERIVEE TEMPORELLE COMME AVANT !!!!
        A, A_, _A = A2D_ms(data)(t)[0], A2D_ms(data)(t)[1], A2D_ms(data)(t)[2]
        B, B_, _B = B2D_ms(data)(t)[0], B2D_ms(data)(t)[1], B2D_ms(data)(t)[2]
        if sous_echantillonnage:
            if n % pas_sauvegarde == 0 and n // pas_sauvegarde < N_frames:
                data.U[n // pas_sauvegarde, ...] = Un
            U_temp = Un
        else:
            U_temp = data.U[n, ...]
        U_temp_ = np.zeros((data.Mx, data.My, 3))

        S = [-rho1(t)[1]/rho1(t)[0], -rho1(t)[1]/rho1(t)[0], kappa1(t)[1]/kappa1(t)[0]]
        S_ = [-rho2(t)[1]/rho2(t)[0], -rho2(t)[1]/rho2(t)[0], kappa2(t)[1]/kappa2(t)[0]]
        _S = [-rho3(t)[1]/rho3(t)[0], -rho3(t)[1]/rho3(t)[0], kappa3(t)[1]/kappa3(t)[0]]
        d_in = np.exp(-np.array([S, S_, _S]) * data.dt / 2)
        U_temp[:] = U_temp * d_in[band_in][None, :, :]

        Amats, Bmats = [A, A_, _A], [B, B_, _B]
        def stk(fn):
            return np.array([fn(Amats[k], Bmats[k]) for k in range(3)])[band_st]
        def apj(Mj, d):
            return np.einsum('jkl,ijl->ijk', Mj, d)
        xp, xm = U_temp[2:, 1:-1, :], U_temp[:-2, 1:-1, :]
        yp, ym = U_temp[1:-1, 2:, :], U_temp[1:-1, :-2, :]
        c0 = U_temp[1:-1, 1:-1, :]
        cross = U_temp[2:, 2:, :] - U_temp[2:, :-2, :] - U_temp[:-2, 2:, :] + U_temp[:-2, :-2, :]
        a1 = (1 / (2 * data.dx)) * apj(stk(lambda a, b: data.dt * a[0] + data.dt**2/2 * a[1]), xp - xm)
        a2 = (1 / (2 * data.dy)) * apj(stk(lambda a, b: data.dt * b[0] + data.dt**2/2 * b[1]), yp - ym)
        b1 = apj(stk(lambda a, b: 0.5 * data.dt ** 2 * (a[0] @ a[0])), (xp + xm - 2 * c0) / data.dx ** 2)
        b2 = apj(stk(lambda a, b: 0.5 * data.dt ** 2 * (b[0] @ b[0])), (yp + ym - 2 * c0) / data.dy ** 2)
        b3 = 1 / (4 * data.dx * data.dy) * apj(stk(lambda a, b: 0.5 * data.dt ** 2 * (a[0] @ b[0] + b[0] @ a[0])), cross)
        s = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * Gmat[..., None] * np.array([0, 0, 1])
        U_temp_[1:-1, 1:-1, :] = c0 - a1 - a2 + b1 + b2 + b3 + s

        t = (n + 1)*data.dt
        S = [-rho1(t)[1]/rho1(t)[0], -rho1(t)[1]/rho1(t)[0], kappa1(t)[1]/kappa1(t)[0]]
        S_ = [-rho2(t)[1]/rho2(t)[0], -rho2(t)[1]/rho2(t)[0], kappa2(t)[1]/kappa2(t)[0]]
        _S = [-rho3(t)[1]/rho3(t)[0], -rho3(t)[1]/rho3(t)[0], kappa3(t)[1]/kappa3(t)[0]]
        d_out = np.exp(-np.array([S, S_, _S]) * data.dt / 2)
        if sous_echantillonnage:
            Un1[...] = U_temp_ * d_out[band_out][None, :, :]
            Un, Un1 = Un1, Un
        else:
            data.U[n + 1, ...] = U_temp_ * d_out[band_out][None, :, :]

    if sous_echantillonnage:
        data.t = data.t[:N_frames * pas_sauvegarde:pas_sauvegarde]
        data.dt = data.dt * pas_sauvegarde
        data.N = N_frames
        temps = data.t
    else:
        temps = data.dt * np.arange(data.N)
    rho1 = np.array([data.rho[0]*data.rho_mt[0](data, data.eps_r[0])(temps[n])[0] for n in range(data.N)])
    rho2 = np.array([data.rho[1]*data.rho_mt[1](data, data.eps_r[1])(temps[n])[0] for n in range(data.N)])
    rho3 = np.array([rho_moy(data)(temps[n])[0] for n in range(data.N)])
    kappa1 = np.array([data.kappa[0] * data.kappa_mt[0](data, data.eps_kappa[0])(temps[n])[0] for n in range(data.N)])
    kappa2 = np.array([data.kappa[1] * data.kappa_mt[1](data, data.eps_kappa[1])(temps[n])[0] for n in range(data.N)])
    kappa3 = np.array([kappa_moy(data)(temps[n])[0] for n in range(data.N)])
    band_e = np.where((1 <= m_all) & (m_all <= L), 0, np.where((m_all > L + 1) & (m_all <= l + L - 2), 1, 2))
    rho_col = np.stack([rho1, rho2, rho3])[band_e].T
    kap_col = np.stack([kappa1, kappa2, kappa3])[band_e].T
    e = 0.5 * rho_col[:, None, :] * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2) + 0.5 * data.U[..., 2] ** 2 / kap_col[:, None, :]
    data.E = np.sum(e, axis=(1, 2))

def ADER4_ms_mt(data, l, L):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D dans un milieu anisotrope modulé en temps micro structuré
    FONCTIONNEL UNIQUEMENT AVEC UNE MODULATION TEMPORELLE EN ECHELON !!!
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :param l: float, correspond à la largeur de la première couche, en m
    :param L: float, correspond à la largeur de la deuxieème couche, en m
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    l, L = l // data.dy, L // data.dy

    def G(i, j):
        sigma, R = 5 * data.dx, 10 * data.dx
        x, y = np.abs(data.ps[0][0] - i) * data.dx, np.abs(data.ps[0][1] - j) * data.dy
        return (1 / (np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)) * (0 <= x ** 2 + y ** 2 <= R ** 2)

    sleep(0.01)
    print("\nADER4 2D Ansiotrope Modulation Temporelle()")
    sleep(0.01)
    rho1 = rho_echelon(data, data.eps_r[0])
    rho2 = rho_echelon(data, data.eps_r[1])
    rho3 = rho_moy(data)
    kappa1 = kappa_echelon(data, data.eps_kappa[0])
    kappa2 = kappa_echelon(data, data.eps_kappa[1])
    kappa3 = kappa_moy(data)
    c1 = np.array([np.sqrt(data.kappa[0]*kappa1(data.t[n])[0]/(data.rho[0]*rho1(data.t[n])[0])) for n in range(data.N)])
    c2 = np.array([np.sqrt(data.kappa[1]*kappa2(data.t[n])[0]/(data.rho[1]*rho2(data.t[n])[0])) for n in range(data.N)])
    c = max(np.max(c1), np.max(c2))
    data.CFL_maj(c = c)


    sous_echantillonnage = data.N > 1000
    if sous_echantillonnage:
        N_frames = data.N // pas_sauvegarde
        data.U = np.zeros((N_frames, data.Mx, data.My, 3))
        Un = np.zeros((data.Mx, data.My, 3))          # solution au pas de temps n
        Un1 = np.zeros((data.Mx, data.My, 3))         # solution au pas de temps n+1
    else:
        data.U = np.zeros((data.N, data.Mx, data.My, 3))
    coeff = [1/(12*data.dx),1/(12*data.dx**2),1/(2*data.dx**3),1/(144*data.dx*data.dy**2),1/data.dx**4,1/(144*data.dx*data.dy),1/(144*(data.dx*data.dy)**2),1/(24*data.dx**3*data.dy)]

    m_all = (np.arange(data.My) + L // 2) % (l + L)
    band_in = np.where(m_all <= L, 0, np.where((m_all > L + 1) & (m_all <= l + L - 2), 1, 2))
    band_out = np.where((1 <= m_all) & (m_all <= L), 0, np.where((m_all > L + 1) & (m_all <= l + L - 2), 1, 2))
    m_st = (np.arange(2, data.My - 2) + L // 2) % (l + L)
    band_st = np.where((1 <= m_st) & (m_st < L - 1), 0, np.where((m_st > L + 1) & (m_st <= l + L - 2), 1, 2))
    sigma, R = 5 * data.dx, 10 * data.dx
    ii = np.arange(2, data.Mx - 2)[:, None]
    jc = np.arange(2, data.My - 2)[None, :]
    gx, gy = np.abs(data.ps[0][0] - ii) * data.dx, np.abs(data.ps[0][1] - jc) * data.dy
    r2 = gx ** 2 + gy ** 2
    Gmat = (1 / (np.pi * sigma ** 2) * np.exp(-r2 / sigma ** 2)) * ((0 <= r2) & (r2 <= R ** 2))

    for n in trange(0, data.N - 1, ncols = ncols):
        t = n * data.dt
        A, A_, _A  = A2D_ms(data)(t)[0],A2D_ms(data)(t)[1],A2D_ms(data)(t)[2]
        B, B_, _B  = B2D_ms(data)(t)[0],B2D_ms(data)(t)[1],B2D_ms(data)(t)[2]
        if sous_echantillonnage:
            if n % pas_sauvegarde == 0 and n // pas_sauvegarde < N_frames:
                data.U[n // pas_sauvegarde, ...] = Un
            U_temp = Un
        else:
            U_temp = data.U[n, ...]
        U_temp_ = np.zeros((data.Mx, data.My, 3))

        S = [-rho1(t)[1]/rho1(t)[0], -rho1(t)[1]/rho1(t)[0], kappa1(t)[1]/kappa1(t)[0]]
        S_ = [-rho2(t)[1]/rho2(t)[0], -rho2(t)[1]/rho2(t)[0], kappa2(t)[1]/kappa2(t)[0]]
        _S = [-rho3(t)[1]/rho3(t)[0], -rho3(t)[1]/rho3(t)[0], kappa3(t)[1]/kappa3(t)[0]]
        d_in = np.exp(-np.array([S, S_, _S]) * data.dt / 2)
        U_temp[:] = U_temp * d_in[band_in][None, :, :]

        Amats, Bmats = [A, A_, _A], [B, B_, _B]
        def stk(fn):
            return np.array([fn(Amats[k], Bmats[k]) for k in range(3)])[band_st]
        def apj(Mj, d):
            return np.einsum('jkl,ijl->ijk', Mj, d)
        def sh(pp, qq):
            return U_temp[2 + pp:data.Mx - 2 + pp, 2 + qq:data.My - 2 + qq, :]
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
                            + sh(2,-2) -16*sh(2,-1) +30*sh(2,0) - 16*sh(2,1) + sh(2,2)),
                coeff[7] * (-sh(-2,-2) + 8*sh(-2,-1) - 8*sh(-2,1) + sh(-2,2)
                            + 2*sh(-1,-2) - 16*sh(-1,-1) + 16*sh(-1,1) - 2*sh(-1,2)
                            - 2*sh(1,-2) + 16*sh(1,-1) - 16*sh(1,1) + 2*sh(1,2)
                            + sh(2,-2) - 8*sh(2,-1) + 8*sh(2,1) - sh(2,2)),
                coeff[7] * (-sh(-2,-2) + 2*sh(-2,-1) - 2*sh(-2,1) + sh(-2,2)
                            + 8*sh(-1,-2) - 16*sh(-1,-1) + 16*sh(-1,1) - 8*sh(-1,2)
                            - 8*sh(1,-2) + 16*sh(1,-1) - 16*sh(1,1) + 8*sh(1,2)
                            + sh(2,-2) - 2*sh(2,-1) + 2*sh(2,1) - sh(2,2))]

        a1 = - data.dt * (apj(stk(lambda a, b: a[0]), dxU[1]) + apj(stk(lambda a, b: b[0]), dyU[1]))
        a2 = data.dt**2/2 * (- apj(stk(lambda a, b: a[1]), dxU[1]) - apj(stk(lambda a, b: b[1]), dyU[1]) + apj(stk(lambda a, b: a[0] @ a[0]), dxU[2]) + apj(stk(lambda a, b: b[0] @ b[0]), dyU[2]) + apj(stk(lambda a, b: a[0] @ b[0] + b[0] @ a[0]), dxyU[0]))
        a3 = data.dt**3/6 * (- apj(stk(lambda a, b: a[2]), dxU[1]) - apj(stk(lambda a, b: b[2]), dyU[1]) + apj(stk(lambda a, b: 2 * a[1] @ a[0] + a[0] @ a[1]), dxU[2]) + apj(stk(lambda a, b: 2 * b[1] @ b[0] + b[0] @ b[1]), dyU[2])
                             + apj(stk(lambda a, b: 2 * a[1] @ b[0] + a[0] @ b[1] + 2 * b[1] @ a[0] + b[0] @ a[1]), dxyU[0])
                             - apj(stk(lambda a, b: a[0] @ a[0] @ b[0] + b[0] @ a[0] @ a[0]), dxyU[2]) - apj(stk(lambda a, b: b[0] @ b[0] @ a[0] + a[0] @ b[0] @ b[0]), dxyU[1])
                             - apj(stk(lambda a, b: a[0] @ a[0] @ a[0]), dxU[3]) - apj(stk(lambda a, b: b[0] @ b[0] @ b[0]), dyU[3]))
        a4 = data.dt**4/24 * (- apj(stk(lambda a, b: a[3]), dxU[1]) - apj(stk(lambda a, b: b[3]), dyU[1])
                              + apj(stk(lambda a, b: 3 * a[2] @ b[0] + 3 * a[1] @ b[1] + a[0] @ b[2] + 3 * b[2] @ a[0] + 3 * b[1] @ a[1] + b[0] @ a[2]), dxyU[0])
                              + apj(stk(lambda a, b: 3 * a[2] @ a[0] + 3 * a[1] @ a[1] + a[0] @ a[2]), dxU[2]) + apj(stk(lambda a, b: 3 * b[2] @ b[0] + 3 * b[1] @ b[1] + b[0] @ b[2]), dyU[2])
                              - apj(stk(lambda a, b: 3 * a[1] @ a[0] @ a[0] + 2 * a[0] @ a[1] @ a[0] + a[0] @ a[0] @ a[1]), dxU[3])
                              - apj(stk(lambda a, b: 3 * b[1] @ b[0] @ b[0] + 2 * b[0] @ b[1] @ b[0] + b[0] @ b[0] @ b[1]), dyU[3])
                              - apj(stk(lambda a, b: 3 * a[1] @ a[0] @ b[0] + 2 * a[0] @ a[1] @ b[0] + a[0] @ a[0] @ b[1] + 3 * b[0] @ b[0] @ a[1] + 2 * b[0] @ a[1] @ a[0] + b[0] @ a[0] @ a[1]), dxyU[2])
                              - apj(stk(lambda a, b: 3 * b[1] @ b[0] @ a[0] + 2 * b[0] @ b[1] @ a[0] + b[0] @ b[0] @ a[1] + 3 * a[0] @ a[0] @ b[1] + 2 * a[0] @ b[1] @ b[0] + a[0] @ b[0] @ b[1]), dxyU[1])
                              + apj(stk(lambda a, b: a[0] @ a[0] @ a[0] @ b[0] + b[0] @ a[0] @ a[0] @ a[0]), dxyU[4]) + apj(stk(lambda a, b: b[0] @ b[0] @ b[0] @ a[0] + a[0] @ b[0] @ b[0] @ b[0]), dxyU[5])
                              + apj(stk(lambda a, b: a[0] @ a[0] @ a[0] @ a[0]), dxU[4]) + apj(stk(lambda a, b: b[0] @ b[0] @ b[0] @ b[0]), dyU[4]))

        s = data.dt / (np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * Gmat[..., None] * np.array([0, 0, 1])
        U_temp_[2:-2, 2:-2, :] = sh(0,0) + a1 + a2 + a3 + a4 + s

        t = (n + 1)*data.dt
        S = [-rho1(t)[1]/rho1(t)[0], -rho1(t)[1]/rho1(t)[0], kappa1(t)[1]/kappa1(t)[0]]
        S_ = [-rho2(t)[1]/rho2(t)[0], -rho2(t)[1]/rho2(t)[0], kappa2(t)[1]/kappa2(t)[0]]
        _S = [-rho3(t)[1]/rho3(t)[0], -rho3(t)[1]/rho3(t)[0], kappa3(t)[1]/kappa3(t)[0]]
        d_out = np.exp(-np.array([S, S_, _S]) * data.dt / 2)
        if sous_echantillonnage:
            Un1[...] = U_temp_ * d_out[band_out][None, :, :]
            Un, Un1 = Un1, Un
        else:
            data.U[n + 1, ...] = U_temp_ * d_out[band_out][None, :, :]

    if sous_echantillonnage:
        data.t = data.t[:N_frames * pas_sauvegarde:pas_sauvegarde]
        data.dt = data.dt * pas_sauvegarde
        data.N = N_frames
        temps = data.t
    else:
        temps = data.dt * np.arange(data.N)
    rho1 = np.array([data.rho[0]*rho_echelon(data, data.eps_r[0])(temps[n])[0] for n in range(data.N)])
    rho2 = np.array([data.rho[1]*rho_echelon(data, data.eps_r[1])(temps[n])[0] for n in range(data.N)])
    rho3 = np.array([rho_moy(data)(temps[n])[0] for n in range(data.N)])
    kappa1 = np.array([data.kappa[0] * kappa_echelon(data, data.eps_kappa[0])(temps[n])[0] for n in range(data.N)])
    kappa2 = np.array([data.kappa[1] * kappa_echelon(data, data.eps_kappa[1])(temps[n])[0] for n in range(data.N)])
    kappa3 = np.array([kappa_moy(data)(temps[n])[0] for n in range(data.N)])
    band_e = np.where((1 <= m_all) & (m_all <= L), 0, np.where((m_all > L + 1) & (m_all <= l + L - 2), 1, 2))
    rho_col = np.stack([rho1, rho2, rho3])[band_e].T
    kap_col = np.stack([kappa1, kappa2, kappa3])[band_e].T
    e = 0.5 * rho_col[:, None, :] * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2) + 0.5 * data.U[..., 2] ** 2 / kap_col[:, None, :]
    data.E = np.sum(e, axis=(1, 2))