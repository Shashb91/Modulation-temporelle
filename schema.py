from donnee import *
from numpy.linalg import matrix_power
from time import sleep
from tqdm import trange
ncols = 125                                                                       #largeur de la barre de chargement
from math import factorial

"""
Schémas numériques pour la résolution du problème de propagation en 1D non modulé en temps, avec un point source ponctuel
"""

def LaxWendroff1D(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 1D
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    print("\nLaxWendroff1D()")
    sleep(0.001)
    data.U = np.zeros((data.N, data.M, 2))
    A = np.array([[0, 1/data.rho], [data.rho * data.c**2, 0]])

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(1, data.M - 1):
            a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i+1, :] - data.U[n, i-1, :])
            a2 = (0.5 * (data.dt / data.dx)**2) * (A @ A) @ (data.U[n, i+1, :] + data.U[n, i-1, :] - 2 * data.U[n, i, :])
            data.U[n+1, i, :] = data.U[n, i, :] - a1 + a2 + data.dt/data.dx * data.S(data.f,(n+1)*data.dt) * (i == data.xs) * np.array([1,0]).transpose()

    data.E = np.array([sum([(0.5 * data.rho * data.U[n, i, 0] ** 2 + data.U[n, i, 1] ** 2 / (2*data.rho * data.c ** 2))*data.dx for i in range(data.M)]) for n in range(0, data.N)])

def ADER21D(data):
    """
    Utilise le schéma d'ADER2 pour résoudre le problème de propagation 1D
    :param data: Donnee1D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 1D
    """
    print("\nADER2 1D()")
    sleep(0.001)
    gamma = np.array([[-1/2,-1/2],
                       [0, 1],
                       [1/2,-1/2]])

    s_max = gamma.shape[0]
    m_max = gamma.shape[1]
    data.U = np.zeros((data.N, data.M, 2))
    A = np.array([[0, 1 / data.rho], [data.rho * data.c ** 2, 0]])
    C = np.zeros((s_max, 2, 2))

    for s in range(0, s_max):
        a = np.zeros((2, 2))
        for m in range(0, m_max):
            a += gamma[s, m] * (data.dt/data.dx)**(m + 1) * matrix_power(-A, m + 1)/factorial(m + 1)
        C[s, :, :] = a

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(1, data.M - 1):
            a1 = sum([C[s] @ data.U[n, i + s - 1, :] for s in range(0, s_max)])
            a2 = (data.dt / data.dx) * data.S(data.f,(n+1)*data.dt) * (i == data.xs) * np.array([data.opt, not data.opt])
            data.U[n + 1, i, :] = data.U[n, i, :] - a1 + a2

    data.E = np.array([sum([(0.5 * data.rho * data.U[n, i, 0] ** 2 + data.U[n, i, 1] ** 2 / (2*data.rho * data.c ** 2))*data.dx for i in range(data.M)]) for n in range(0, data.N)])

def ADER41D(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 1D
    :param data: Donnee1D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 1D
    """
    print("\nADER4 1D()")
    sleep(0.001)
    gamma = np.array([[1 / 12, 1 / 24, -1 / 12, -1 / 24],
                      [-2 / 3, -2 / 3, 1 / 6, 1 / 6],
                      [0, 5 / 4, 0, -1 / 4],
                      [2 / 3, -2 / 3, -1 / 6, 1 / 6],
                      [-1 / 12, 1 / 24, 1 / 12, -1 / 24]])

    s_max = gamma.shape[0]
    m_max = gamma.shape[1]
    data.U = np.zeros((data.N, data.M, 2))
    A = np.array([[0, 1 / data.rho], [data.rho * data.c ** 2, 0]])
    C = np.zeros((s_max, 2, 2))

    for s in range(0, s_max):
        a = np.zeros((2, 2))
        for m in range(0, m_max):
            a += gamma[s, m] * (data.dt/data.dx)**(m + 1) * matrix_power(A, m + 1)
        C[s, :, :] = a

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(2, data.M - 2):
            a1 = sum([C[s] @ data.U[n, i + s - 2, :] for s in range(0, s_max)])
            a2 = (data.dt / data.dx) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * np.array([data.opt, not data.opt])
            data.U[n + 1, i, :] = data.U[n, i, :] - a1 + a2

    data.E = np.array([sum([(0.5 * data.rho * data.U[n, i, 0] ** 2 + data.U[n, i, 1] ** 2 / (2*data.rho * data.c ** 2))*data.dx for i in range(data.M)]) for n in range(0, data.N)])

"""
Schémas numériques pour la résolution du problème de Cauchy de propagation en 1D non modulé en temps
"""

def LaxWendroff1D_cauchy(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 1D en initialisation lointaine
    :param data: Donnee 1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    print("\nLaxWendroff 1D Cauchy()")
    sleep(0.001)
    def fct(f,t):
        omega = 2 * np.pi * f
        coeff = [1, -21 / 32, 63 / 768, -1 / 512]
        return (coeff[0] * np.sin(omega * t) + coeff[1] * np.sin(omega * 2 * t) + coeff[2] * np.sin(omega * 4 * t) + coeff[3] * np.sin(omega * 8 * t)) * (0 < t < 1 / f)

    data.U = np.zeros((data.N, data.M, 2))
    A = np.array([[0, 1/data.rho], [data.rho * data.c**2, 0]])

    # init
    for i in range(0, data.M):
        data.U[0,i,:] = 1/data.c * fct(data.f, 1/data.f +  data.tc[0] - data.dx*(i-1)/data.c) * np.array([1, data.c*data.rho])

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(1, data.M - 1):
            a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i+1, :] - data.U[n, i-1, :])
            a2 = (0.5 * (data.dt / data.dx)**2) * (A @ A) @ (data.U[n, i+1, :] + data.U[n, i-1, :] - 2 * data.U[n, i, :])
            data.U[n+1, i, :] = data.U[n, i, :] - a1 + a2

    data.E = np.array([sum([(0.5 * data.rho * data.U[n, i, 0] ** 2 + data.U[n, i, 1] ** 2 / (2*data.rho * data.c ** 2))*data.dx for i in range(data.M)]) for n in range(0, data.N)])

def ADER41D_cauchy(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 1D
    :param data: Donnee1D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 1D
    """
    print("\nADER4 1D Cauchy()")
    sleep(0.001)
    def fct(f,t):
        omega = 2 * np.pi * f
        coeff = [1, -21 / 32, 63 / 768, -1 / 512]
        return (coeff[0] * np.sin(omega * t) + coeff[1] * np.sin(omega * 2 * t) + coeff[2] * np.sin(omega * 4 * t) + coeff[3] * np.sin(omega * 8 * t)) * (0 < t < 1 / f)

    gamma = np.array([[1 / 12, 1 / 24, -1 / 12, -1 / 24],
                      [-2 / 3, -2 / 3, 1 / 6, 1 / 6],
                      [0, 5 / 4, 0, -1 / 4],
                      [2 / 3, -2 / 3, -1 / 6, 1 / 6],
                      [-1 / 12, 1 / 24, 1 / 12, -1 / 24]])

    s_max = gamma.shape[0]
    m_max = gamma.shape[1]
    data.U = np.zeros((data.N, data.M, 2))
    A = np.array([[0, 1 / data.rho], [data.rho * data.c ** 2, 0]])
    C = np.zeros((s_max, 2, 2))

    # init
    for i in range(0, data.M):
        data.U[0,i,:] = 1/data.c * fct(data.f, 1/data.f +  data.tc[0] - data.dx*(i-1)/data.c) * np.array([1, data.c*data.rho])

    for s in range(0, s_max):
        a = np.zeros((2, 2))
        for m in range(0, m_max):
            a += gamma[s, m] * (data.dt/data.dx)**(m + 1) * matrix_power(A, m + 1)
        C[s, :, :] = a

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(2, data.M - 2):
            a1 = sum([C[s] @ data.U[n, i + s - 2, :] for s in range(0, s_max)])
            data.U[n + 1, i, :] = data.U[n, i, :] - a1

    data.E = np.array([sum([(0.5 * data.rho * data.U[n, i, 0] ** 2 + data.U[n, i, 1] ** 2 / (data.rho * data.c ** 2))*data.dx for i in range(data.M)]) for n in range(0, data.N)])
 

"""
Schémas numériques pour la résolution du problème de propagation en 2D non modulé en temps, avec un point source ponctuel
"""

def LaxWendroff2D(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 2D
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    print("\nLaxWendroff 2D()")
    sleep(0.001)
    data.U = np.zeros((data.N, data.Mx, data.My, 3))
    rc2 = data.rho*data.c**2
    A = np.array([[0,0, 1 / data.rho],
                  [0,0, 0],
                  [rc2, 0, 0]])
    B = np.array([[0,0,0],
                  [0,0,1/data.rho],
                  [0, rc2, 0]])

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(1, data.Mx - 1):
            for j in range(1, data.My - 1):
                s = data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * ((i,j) in data.ps) * np.array([0, 0, 1]).transpose()
                a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i + 1, j, :] - data.U[n, i - 1, j, :])
                a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, i, j + 1, :] - data.U[n, i, j - 1, :])
                b1 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, i + 1, j, :] + data.U[n, i - 1, j, :] - 2 * data.U[n, i, j, :]) / data.dx ** 2)
                b2 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, i, j + 1, :] + data.U[n, i, j - 1, :] - 2 * data.U[n, i, j, :]) / data.dy ** 2)
                data.U[n + 1, i, j, :] = data.U[n, i, j, :] - a1 - a2 + b1 + b2 + s


    ec = 0.5 * data.rho * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (data.rho * data.c ** 2)
    data.E = np.sum((ec + ep)*data.dx*data.dy, axis=(1, 2))

def ADER42D(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D
    :param data: Donnee2D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 2D
    """
    print("\nADER4 2D()")
    sleep(0.001)
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

    c = [1/(12*data.dx),1/data.dx**2,1/(2*data.dx**3),1/(144*data.dx*data.dy**2),1/data.dx**4,1/(144*(data.dx*data.dy)**2)]
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
                           2 * c[5] * (data.U[n,i-2,j-2,:] -16*data.U[n,i-2,j-1,:] +30 * data.U[n,i-2,j,:] - 16*data.U[n,i-2,j+1,:] + data.U[n,i-2,j+2,:]
                                   -16*data.U[n,i-1,j-2,:] +256*data.U[n,i-1,j-1,:]-480*data.U[n,i-1,j,:] + 256*data.U[n,i-1,j+1,:] - 16*data.U[n,i-1,j+2,:]
                                   +30*data.U[n,i,j-2,:] -480*data.U[n,i,j-1,:] +900*data.U[n,i,j,:] -480*data.U[n,i,j+1,:] + 30*data.U[n,i,j+2,:]
                                   -16*data.U[n,i+1,j-2,:] +256*data.U[n,i+1,j-1,:]-480*data.U[n,i+1,j,:] + 256*data.U[n,i+1,j+1,:] - 16*data.U[n,i+1,j+2,:]
                                   + data.U[n,i+2,j-2,:] -16*data.U[n,i+2,j-1,:] +30*data.U[n,i+2,j,:] - 16*data.U[n,i+2,j+1,:] + data.U[n,i+2,j+2,:]))
                s = data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * ((i,j) in data.ps) * np.array([0, 0, 1]).transpose()
                data.U[n + 1, i, j, :] = data.U[n,i,j,:] - a1 - a2 - a3 - a4 + s
    ec = 0.5 * data.rho * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (data.rho * data.c ** 2)
    data.E = np.sum((ec + ep)*data.dx*data.dy, axis=(1, 2))

"""
Schémas numériques pour la résolution du problème de Cauchy de propagation en 2D non modulé en temps
"""

def LaxWendroff2D_cauchy(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 2D pour un front d'onde
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    sleep(0.01)
    print("\nLaxWendroff2D Cauchy()")
    sleep(0.01)
    data.U = np.zeros((data.N, data.Mx + 2, data.My, 3))
    theta = 0
    rc2 = data.rho * data.c ** 2
    A = np.array([[0, 0, 1 / data.rho],
                  [0, 0, 0],
                  [rc2, 0, 0]])
    B = np.array([[0, 0, 0],
                  [0, 0, 1 / data.rho],
                  [0, rc2, 0]])

    # init
    for i in range(0, data.Mx + 2):
        for j in range(0, data.My):
            data.U[0, i, j, :] = 1/data.c * data.S(data.f,1/data.f + data.tc[0] - data.dy * (j-1) / data.c) * np.array([np.sin(theta), np.cos(theta), data.rho * data.c]).transpose()


    for n in trange(0, data.N - 1, ncols=ncols):
        for j in range(1, data.My - 1):
            #condition de periodicité gauche
            a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, 1, j, :] - data.U[n, - 1, j, :])
            a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, 0, j + 1, :] - data.U[n, 0, j - 1, :])
            b1 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, 1, j, :] + data.U[n, - 1, j, :] - 2 * data.U[n, 0, j, :]) / data.dx ** 2)
            b2 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, 0, j + 1, :] + data.U[n, 0, j - 1, :] - 2 * data.U[n, 0, j, :]) / data.dy ** 2)
            data.U[n + 1, 0, j, :] = data.U[n, 0, j, :] - a1 - a2 + b1 + b2

            # condition de periodicité droite
            a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, 0, j, :] - data.U[n, -2, j, :])
            a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, -1, j + 1, :] - data.U[n, -1, j - 1, :])
            b1 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, 0, j, :] + data.U[n, -2, j, :] - 2 * data.U[n, -1, j, :]) / data.dx ** 2)
            b2 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, -1, j + 1, :] + data.U[n, -1, j - 1, :] - 2 * data.U[n, -1, j, :]) / data.dy ** 2)
            data.U[n + 1, -1, j, :] = data.U[n, -1, j, :] - a1 - a2 + b1 + b2
            for i in range(1, data.Mx + 1):
                # Lax-Wendroff 2D
                a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i + 1, j, :] - data.U[n, i - 1, j, :])
                a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, i, j + 1, :] - data.U[n, i, j - 1, :])
                b1 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, i + 1, j, :] + data.U[n, i - 1, j, :] - 2 * data.U[n, i, j, :]) / data.dx ** 2)
                b2 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, i, j + 1, :] + data.U[n, i, j - 1, :] - 2 * data.U[n, i, j, :]) / data.dy ** 2)
                data.U[n + 1, i, j, :] = data.U[n, i, j, :] - a1 - a2 + b1 + b2

    data.U = data.U[:,1:data.Mx + 1, :,:]
    ec = 0.5 * data.rho * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (data.rho * data.c ** 2)
    data.E = np.sum((ec + ep)*data.dx*data.dy, axis=(1, 2))

def ADER42D_cauchy(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D pour un front d'onde
    :param data: Donnee2D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 2D
    """
    print("\nADER4 2D Cauchy()")
    sleep(0.001)
    def fct(f,t):
        omega = 2 * np.pi * f
        coeff = [1, -21 / 32, 63 / 768, -1 / 512]
        return (coeff[0] * np.sin(omega * t) + coeff[1] * np.sin(omega * 2 * t) + coeff[2] * np.sin(omega * 4 * t) + coeff[3] * np.sin(omega * 8 * t)) * (0 < t < 1 / f)

    data.U = np.zeros((data.N, data.Mx + 4, data.My, 3))
    rc2 = data.rho*data.c**2
    theta = 0
    A = np.array([[0,0, 1 / data.rho],
                  [0,0, 0],
                  [rc2, 0, 0]])
    B = np.array([[0,0,0],
                  [0,0,1/data.rho],
                  [0, rc2, 0]])
    b2 = - (data.c*data.dt)**2/24
    b3 = data.c**2*data.dt**3/6
    b4 = - (data.c*data.dt)**4/24
    c = [1/(12*data.dx),1/data.dx**2,1/(2*data.dx**3),1/(144*data.dx*data.dy**2),1/data.dx**4,2/(144*(data.dx*data.dy)**2)]

    #init
    for i in range(0, data.Mx+4):
        for j in range(0, data.My):
            data.U[0,i,j,:] = 1/data.c * fct(data.f,1/data.f + data.tc[0] - data.dy*(j-1)/data.c)* np.array([np.sin(theta), np.cos(theta), data.rho*data.c]).transpose()


    for n in trange(0, data.N - 1, ncols = ncols):
        for j in range(2,data.My-2):
            #condition periodicité gauche
            a1 = data.dt * (c[0] * A @ (data.U[n,-2,j,:] - 8*data.U[n,-1,j,:] + 8*data.U[n,1,j,:] - data.U[n,2,j,:]) +
                            c[0] * B @ (data.U[n,0,j-2,:] - 8*data.U[n,0,j-1,:] + 8*data.U[n,0,j+1,:] - data.U[n,0,j+2,:]))
            a2 = b2 * (c[1] * (- data.U[n,-2,j,:] + 16*data.U[n,-1,j,:] - 30*data.U[n,0,j,:] + 16*data.U[n,1,j,:] - data.U[n,2,j,:]) +
                       c[1] * (- data.U[n,0,j-2,:] + 16*data.U[n,0,j-1,:] - 30*data.U[n,0,j,:] + 16*data.U[n,0,j+1,:] - data.U[n,0,j+2,:]))
            a3 = b3 * (A @ (c[2] * (- data.U[n,-2,j,:] + 2*data.U[n,-1,j,:] - 2*data.U[n,1,j,:] + data.U[n,2,j,:]) +
                            c[3] * (- data.U[n,-2,j-2,:] + 8*data.U[n,-1,j-2,:] - 8*data.U[n,1,j-2,:] + data.U[n,2,j-2,:]+
                                    16*data.U[n,-2,j-1,:] -128*data.U[n,-1,j-1,:] + 128*data.U[n,1,j-1,:] - 16*data.U[n,2,j-1,:]
                                    -30*data.U[n,-2,j,:] + 240*data.U[n,-1,j,:] - 240*data.U[n,1,j,:] + 30*data.U[n,2,j,:]+
                                    16*data.U[n,-2,j+1,:] - 128*data.U[n,-1,j+1,:] + 128*data.U[n,1,j+1,:] - 16*data.U[n,2,j+1,:]
                                    - data.U[n,-2,j+2,:] + 8*data.U[n,-1,j+2,:] - 8*data.U[n,1,j+2,:] + data.U[n,2,j+2,:])) +
                       B @ (c[2] * (- data.U[n,0,j-2,:] + 2*data.U[n,0,j-1,:] - 2*data.U[n,0,j+1,:] + data.U[n,0,j+2,:]) +
                            c[3] * (- data.U[n,-2,j-2,:] + 8*data.U[n,-2,j-1,:] - 8*data.U[n,-2,j+1,:] + data.U[n,-2,j+2,:]
                                    + 16*data.U[n,-1,j-2,:] -128*data.U[n,-1,j-1,:] + 128*data.U[n,-1,j+1,:] - 16*data.U[n,-1,j+2,:]
                                    - 30*data.U[n,0,j-2,:] + 240*data.U[n,0,j-1,:] - 240*data.U[n,0,j+1,:] + 30*data.U[n,0,j+2,:]
                                    + 16*data.U[n,1,j-2,:] - 128*data.U[n,1,j-1,:] + 128*data.U[n,1,j+1,:] - 16*data.U[n,1,j+2,:]
                                    - data.U[n,2,j-2,:] + 8*data.U[n,2,j-1,:] - 8*data.U[n,2,j+1,:] + data.U[n,2,j+2,:])))
            a4 = b4 * (c[4] * (data.U[n,-2,j,:] - 4*data.U[n,-1,j,:] + 6*data.U[n,0,j,:] - 4*data.U[n,1,j,:] + data.U[n,2,j,:]) +
                       c[4] * (data.U[n,0,j-2,:] - 4*data.U[n,0,j-1,:] + 6*data.U[n,0,j,:] - 4*data.U[n,0,j+1,:] + data.U[n,0,j+2,:]) +
                       c[5] * (data.U[n,-2,j-2,:] -16*data.U[n,-2,j-1,:] +30 * data.U[n,-2,j,:] - 16*data.U[n,-2,j+1,:] + data.U[n,-2,j+2,:]
                               - 16*data.U[n,-1,j-2,:] +256*data.U[n,-1,j-1,:]-480*data.U[n,-1,j,:] + 256*data.U[n,-1,j+1,:] - 16*data.U[n,-1,j+2,:]
                               + 30*data.U[n,0,j-2,:] -480*data.U[n,0,j-1,:] +900*data.U[n,0,j,:] -480*data.U[n,0,j+1,:] + 30*data.U[n,0,j+2,:]
                               - 16*data.U[n,1,j-2,:] +256*data.U[n,1,j-1,:]-480*data.U[n,1,j,:] + 256*data.U[n,1,j+1,:] - 16*data.U[n,1,j+2,:]
                               + data.U[n,2,j-2,:] -16*data.U[n,2,j-1,:] +30*data.U[n,2,j,:] - 16*data.U[n,2,j+1,:] + data.U[n,2,j+2,:]))
            data.U[n + 1, 0, j, :] = data.U[n,0,j,:] - a1 - a2 - a3 - a4

            a1 = data.dt * (c[0] * A @ (data.U[n,-1,j,:] - 8*data.U[n,0,j,:] + 8*data.U[n,2,j,:] - data.U[n,3,j,:]) +
                            c[0] * B @ (data.U[n,1,j-2,:] - 8*data.U[n,1,j-1,:] + 8*data.U[n,1,j+1,:] - data.U[n,1,j+2,:]))
            a2 = b2 * (c[1] * (- data.U[n,-1,j,:] + 16*data.U[n,0,j,:] - 30*data.U[n,1,j,:] + 16*data.U[n,2,j,:] - data.U[n,3,j,:]) +
                       c[1] * (- data.U[n,1,j-2,:] + 16*data.U[n,1,j-1,:] - 30*data.U[n,1,j,:] + 16*data.U[n,1,j+1,:] - data.U[n,1,j+2,:]))
            a3 = b3 * (A @ (c[2] * (- data.U[n,-1,j,:] + 2*data.U[n,0,j,:] - 2*data.U[n,2,j,:] + data.U[n,3,j,:]) +
                            c[3] * (- data.U[n,-1,j-2,:] + 8*data.U[n,0,j-2,:] - 8*data.U[n,2,j-2,:] + data.U[n,3,j-2,:]+
                                    16*data.U[n,-1,j-1,:] -128*data.U[n,0,j-1,:] + 128*data.U[n,2,j-1,:] - 16*data.U[n,3,j-1,:]
                                    -30*data.U[n,-1,j,:] + 240*data.U[n,0,j,:] - 240*data.U[n,2,j,:] + 30*data.U[n,3,j,:]+
                                    16*data.U[n,-1,j+1,:] - 128*data.U[n,0,j+1,:] + 128*data.U[n,2,j+1,:] - 16*data.U[n,3,j+1,:]
                                    - data.U[n,-1,j+2,:] + 8*data.U[n,0,j+2,:] - 8*data.U[n,2,j+2,:] + data.U[n,3,j+2,:])) +
                       B @ (c[2] * (- data.U[n,1,j-2,:] + 2*data.U[n,1,j-1,:] - 2*data.U[n,1,j+1,:] + data.U[n,1,j+2,:]) +
                            c[3] * (- data.U[n,-1,j-2,:] + 8*data.U[n,-1,j-1,:] - 8*data.U[n,-1,j+1,:] + data.U[n,-1,j+2,:]
                                    + 16*data.U[n,0,j-2,:] -128*data.U[n,0,j-1,:] + 128*data.U[n,0,j+1,:] - 16*data.U[n,0,j+2,:]
                                    - 30*data.U[n,1,j-2,:] + 240*data.U[n,1,j-1,:] - 240*data.U[n,1,j+1,:] + 30*data.U[n,1,j+2,:]
                                    + 16*data.U[n,2,j-2,:] - 128*data.U[n,2,j-1,:] + 128*data.U[n,2,j+1,:] - 16*data.U[n,2,j+2,:]
                                    - data.U[n,3,j-2,:] + 8*data.U[n,3,j-1,:] - 8*data.U[n,3,j+1,:] + data.U[n,3,j+2,:])))
            a4 = b4 * (c[4] * (data.U[n,-1,j,:] - 4*data.U[n,0,j,:] + 6*data.U[n,1,j,:] - 4*data.U[n,2,j,:] + data.U[n,3,j,:]) +
                       c[4] * (data.U[n,1,j-2,:] - 4*data.U[n,1,j-1,:] + 6*data.U[n,1,j,:] - 4*data.U[n,1,j+1,:] + data.U[n,1,j+2,:]) +
                       c[5] * (data.U[n,-1,j-2,:] -16*data.U[n,-1,j-1,:] +30 * data.U[n,-1,j,:] - 16*data.U[n,-1,j+1,:] + data.U[n,-1,j+2,:]
                               - 16*data.U[n,0,j-2,:] +256*data.U[n,0,j-1,:]-480*data.U[n,0,j,:] + 256*data.U[n,0,j+1,:] - 16*data.U[n,0,j+2,:]
                               + 30*data.U[n,1,j-2,:] -480*data.U[n,1,j-1,:] +900*data.U[n,1,j,:] -480*data.U[n,1,j+1,:] + 30*data.U[n,1,j+2,:]
                               - 16*data.U[n,2,j-2,:] +256*data.U[n,2,j-1,:]-480*data.U[n,2,j,:] + 256*data.U[n,2,j+1,:] - 16*data.U[n,2,j+2,:]
                               + data.U[n,3,j-2,:] -16*data.U[n,3,j-1,:] +30*data.U[n,3,j,:] - 16*data.U[n,3,j+1,:] + data.U[n,3,j+2,:]))
            data.U[n + 1, 1, j, :] = data.U[n,1,j,:] - a1 - a2 - a3 - a4

            #conditon de periodicité droite
            a1 = data.dt * (c[0] * A @ (data.U[n,-3,j,:] - 8*data.U[n,-2,j,:] + 8*data.U[n,0,j,:] - data.U[n,1,j,:]) +
                            c[0] * B @ (data.U[n,-1,j-2,:] - 8*data.U[n,-1,j-1,:] + 8*data.U[n,-1,j+1,:] - data.U[n,-1,j+2,:]))
            a2 = b2 * (c[1] * (- data.U[n,-3,j,:] + 16*data.U[n,-2,j,:] - 30*data.U[n,-1,j,:] + 16*data.U[n,0,j,:] - data.U[n,1,j,:]) +
                       c[1] * (- data.U[n,-1,j-2,:] + 16*data.U[n,-1,j-1,:] - 30*data.U[n,-1,j,:] + 16*data.U[n,-1,j+1,:] - data.U[n,-1,j+2,:]))
            a3 = b3 * (A @ (c[2] * (- data.U[n,-3,j,:] + 2*data.U[n,-2,j,:] - 2*data.U[n,0,j,:] + data.U[n,1,j,:]) +
                            c[3] * (- data.U[n,-3,j-2,:] + 8*data.U[n,-2,j-2,:] - 8*data.U[n,0,j-2,:] + data.U[n,1,j-2,:]+
                                    16*data.U[n,-3,j-1,:] -128*data.U[n,-2,j-1,:] + 128*data.U[n,0,j-1,:] - 16*data.U[n,1,j-1,:]
                                    -30*data.U[n,-3,j,:] + 240*data.U[n,-2,j,:] - 240*data.U[n,0,j,:] + 30*data.U[n,1,j,:]+
                                    16*data.U[n,-3,j+1,:] - 128*data.U[n,-2,j+1,:] + 128*data.U[n,0,j+1,:] - 16*data.U[n,1,j+1,:]
                                    - data.U[n,-3,j+2,:] + 8*data.U[n,-2,j+2,:] - 8*data.U[n,0,j+2,:] + data.U[n,1,j+2,:])) +
                       B @ (c[2] * (- data.U[n,-1,j-2,:] + 2*data.U[n,-1,j-1,:] - 2*data.U[n,-1,j+1,:] + data.U[n,-1,j+2,:]) +
                            c[3] * (- data.U[n,-3,j-2,:] + 8*data.U[n,-3,j-1,:] - 8*data.U[n,-3,j+1,:] + data.U[n,-3,j+2,:]
                                    + 16*data.U[n,-2,j-2,:] -128*data.U[n,-2,j-1,:] + 128*data.U[n,-2,j+1,:] - 16*data.U[n,-2,j+2,:]
                                    - 30*data.U[n,-1,j-2,:] + 240*data.U[n,-1,j-1,:] - 240*data.U[n,-1,j+1,:] + 30*data.U[n,-1,j+2,:]
                                    + 16*data.U[n,0,j-2,:] - 128*data.U[n,0,j-1,:] + 128*data.U[n,0,j+1,:] - 16*data.U[n,0,j+2,:]
                                    - data.U[n,1,j-2,:] + 8*data.U[n,1,j-1,:] - 8*data.U[n,1,j+1,:] + data.U[n,1,j+2,:])))
            a4 = b4 * (c[4] * (data.U[n,-3,j,:] - 4*data.U[n,-2,j,:] + 6*data.U[n,-1,j,:] - 4*data.U[n,0,j,:] + data.U[n,1,j,:]) +
                       c[4] * (data.U[n,-1,j-2,:] - 4*data.U[n,-1,j-1,:] + 6*data.U[n,-1,j,:] - 4*data.U[n,-1,j+1,:] + data.U[n,-1,j+2,:]) +
                       c[5] * (data.U[n,-3,j-2,:] -16*data.U[n,-3,j-1,:] +30 * data.U[n,-3,j,:] - 16*data.U[n,-3,j+1,:] + data.U[n,-3,j+2,:]
                               - 16*data.U[n,-2,j-2,:] +256*data.U[n,-2,j-1,:]-480*data.U[n,-2,j,:] + 256*data.U[n,-2,j+1,:] - 16*data.U[n,-2,j+2,:]
                               + 30*data.U[n,-1,j-2,:] -480*data.U[n,-1,j-1,:] +900*data.U[n,-1,j,:] -480*data.U[n,-1,j+1,:] + 30*data.U[n,-1,j+2,:]
                               - 16*data.U[n,0,j-2,:] +256*data.U[n,0,j-1,:]-480*data.U[n,0,j,:] + 256*data.U[n,0,j+1,:] - 16*data.U[n,0,j+2,:]
                               + data.U[n,1,j-2,:] -16*data.U[n,1,j-1,:] +30*data.U[n,1,j,:] - 16*data.U[n,1,j+1,:] + data.U[n,1,j+2,:]))
            data.U[n + 1, -1, j, :] = data.U[n,-1,j,:] - a1 - a2 - a3 - a4
            
            a1 = data.dt * (c[0] * A @ (data.U[n,-4,j,:] - 8*data.U[n,-3,j,:] + 8*data.U[n,-1,j,:] - data.U[n,0,j,:]) +
                            c[0] * B @ (data.U[n,-2,j-2,:] - 8*data.U[n,-2,j-1,:] + 8*data.U[n,-2,j+1,:] - data.U[n,-2,j+2,:]))
            a2 = b2 * (c[1] * (- data.U[n,-4,j,:] + 16*data.U[n,-3,j,:] - 30*data.U[n,-2,j,:] + 16*data.U[n,-1,j,:] - data.U[n,0,j,:]) +
                       c[1] * (- data.U[n,-2,j-2,:] + 16*data.U[n,-2,j-1,:] - 30*data.U[n,-2,j,:] + 16*data.U[n,-2,j+1,:] - data.U[n,-2,j+2,:]))
            a3 = b3 * (A @ (c[2] * (- data.U[n,-4,j,:] + 2*data.U[n,-3,j,:] - 2*data.U[n,-1,j,:] + data.U[n,0,j,:]) +
                            c[3] * (- data.U[n,-4,j-2,:] + 8*data.U[n,-3,j-2,:] - 8*data.U[n,-1,j-2,:] + data.U[n,0,j-2,:]+
                                    16*data.U[n,-4,j-1,:] -128*data.U[n,-3,j-1,:] + 128*data.U[n,-1,j-1,:] - 16*data.U[n,0,j-1,:]
                                    -30*data.U[n,-4,j,:] + 240*data.U[n,-3,j,:] - 240*data.U[n,-1,j,:] + 30*data.U[n,0,j,:]+
                                    16*data.U[n,-4,j+1,:] - 128*data.U[n,-3,j+1,:] + 128*data.U[n,-1,j+1,:] - 16*data.U[n,0,j+1,:]
                                    - data.U[n,-4,j+2,:] + 8*data.U[n,-3,j+2,:] - 8*data.U[n,-1,j+2,:] + data.U[n,0,j+2,:])) +
                       B @ (c[2] * (- data.U[n,-2,j-2,:] + 2*data.U[n,-2,j-1,:] - 2*data.U[n,-2,j+1,:] + data.U[n,-2,j+2,:]) +
                            c[3] * (- data.U[n,-4,j-2,:] + 8*data.U[n,-4,j-1,:] - 8*data.U[n,-4,j+1,:] + data.U[n,-4,j+2,:]
                                    + 16*data.U[n,-3,j-2,:] -128*data.U[n,-3,j-1,:] + 128*data.U[n,-3,j+1,:] - 16*data.U[n,-3,j+2,:]
                                    - 30*data.U[n,-2,j-2,:] + 240*data.U[n,-2,j-1,:] - 240*data.U[n,-2,j+1,:] + 30*data.U[n,-2,j+2,:]
                                    + 16*data.U[n,-1,j-2,:] - 128*data.U[n,-1,j-1,:] + 128*data.U[n,-1,j+1,:] - 16*data.U[n,-1,j+2,:]
                                    - data.U[n,0,j-2,:] + 8*data.U[n,0,j-1,:] - 8*data.U[n,0,j+1,:] + data.U[n,0,j+2,:])))
            a4 = b4 * (c[4] * (data.U[n,-4,j,:] - 4*data.U[n,-3,j,:] + 6*data.U[n,-2,j,:] - 4*data.U[n,-1,j,:] + data.U[n,0,j,:]) +
                       c[4] * (data.U[n,-2,j-2,:] - 4*data.U[n,-2,j-1,:] + 6*data.U[n,-2,j,:] - 4*data.U[n,-2,j+1,:] + data.U[n,-2,j+2,:]) +
                       c[5] * (data.U[n,-4,j-2,:] -16*data.U[n,-4,j-1,:] +30 * data.U[n,-4,j,:] - 16*data.U[n,-4,j+1,:] + data.U[n,-4,j+2,:]
                               - 16*data.U[n,-3,j-2,:] +256*data.U[n,-3,j-1,:]-480*data.U[n,-3,j,:] + 256*data.U[n,-3,j+1,:] - 16*data.U[n,-3,j+2,:]
                               + 30*data.U[n,-2,j-2,:] -480*data.U[n,-2,j-1,:] +900*data.U[n,-2,j,:] -480*data.U[n,-2,j+1,:] + 30*data.U[n,-2,j+2,:]
                               - 16*data.U[n,-1,j-2,:] +256*data.U[n,-1,j-1,:]-480*data.U[n,-1,j,:] + 256*data.U[n,-1,j+1,:] - 16*data.U[n,-1,j+2,:]
                               + data.U[n,0,j-2,:] -16*data.U[n,0,j-1,:] +30*data.U[n,0,j,:] - 16*data.U[n,0,j+1,:] + data.U[n,0,j+2,:]))
            data.U[n + 1, -2, j, :] = data.U[n,-2,j,:] - a1 - a2 - a3 - a4
            
            for i in range(2, data.Mx + 2):
                #ADER4 2D
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
                                   - 16*data.U[n,i-1,j-2,:] +256*data.U[n,i-1,j-1,:]-480*data.U[n,i-1,j,:] + 256*data.U[n,i-1,j+1,:] - 16*data.U[n,i-1,j+2,:]
                                   + 30*data.U[n,i,j-2,:] -480*data.U[n,i,j-1,:] +900*data.U[n,i,j,:] -480*data.U[n,i,j+1,:] + 30*data.U[n,i,j+2,:]
                                   - 16*data.U[n,i+1,j-2,:] +256*data.U[n,i+1,j-1,:]-480*data.U[n,i+1,j,:] + 256*data.U[n,i+1,j+1,:] - 16*data.U[n,i+1,j+2,:]
                                   + data.U[n,i+2,j-2,:] -16*data.U[n,i+2,j-1,:] +30*data.U[n,i+2,j,:] - 16*data.U[n,i+2,j+1,:] + data.U[n,i+2,j+2,:]))
                data.U[n + 1, i, j, :] = data.U[n,i,j,:] - a1 - a2 - a3 - a4
    data.U = data.U[:,2:data.Mx,:,:]
    data.U[...,0] = np.round(data.U[...,0], 16)

    ec = 0.5 * data.rho * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (data.rho * data.c ** 2)
    data.E = np.sum((ec + ep)*data.dx*data.dy, axis=(1, 2))