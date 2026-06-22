from donnee import *
from numpy.linalg import matrix_power
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
    data.U = np.zeros((data.N, data.M, 2))
    A = np.array([[0, 1/data.rho], [data.rho * data.c**2, 0]])

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(1, data.M - 1):
            a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i+1, :] - data.U[n, i-1, :])
            a2 = (0.5 * (data.dt / data.dx)**2) * (A @ A) @ (data.U[n, i+1, :] + data.U[n, i-1, :] - 2 * data.U[n, i, :])
            data.U[n+1, i, :] = data.U[n, i, :] - a1 + a2 + data.dt/data.dx * data.S(data.f,(n+1)*data.dt) * (i == data.xs) * np.array([1,0]).transpose() 

def ADER21D(data):
    """
    Utilise le schéma d'ADER2 pour résoudre le problème de propagation 1D
    :param data: Donnee1D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 1D
    """
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

def ADER41D(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 1D
    :param data: Donnee1D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 1D
    """
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

"""
Schémas numériques pour la résolution du problème de propagation en 1D non modulé en temps, avec une initialisation en ondelette tronquée lointaine
"""

def LaxWendroff1D_init(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 1D en initialisation lointaine
    :param data: Donnee 1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    data.U = np.zeros((data.N, data.M, 2))
    A = np.array([[0, 1/data.rho], [data.rho * data.c**2, 0]])

    # init
    for i in range(0, data.M):
        data.U[0,i,:] = data.dt/data.dx * data.S(data.f,  data.tc[1] - data.dx*i/data.c) * np.array([1, - data.c*data.rho])

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(1, data.M - 1):
            a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i+1, :] - data.U[n, i-1, :])
            a2 = (0.5 * (data.dt / data.dx)**2) * (A @ A) @ (data.U[n, i+1, :] + data.U[n, i-1, :] - 2 * data.U[n, i, :])
            data.U[n+1, i, :] = data.U[n, i, :] - a1 + a2 

def ADER41D_init(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 1D
    :param data: Donnee1D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 1D
    """
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
        data.U[0,i,:] = data.dt/data.dx * data.S(data.f,  data.tc[1] - data.dx*i/data.c) * np.array([1, -data.c * data.rho])

    for s in range(0, s_max):
        a = np.zeros((2, 2))
        for m in range(0, m_max):
            a += gamma[s, m] * (data.dt/data.dx)**(m + 1) * matrix_power(A, m + 1)
        C[s, :, :] = a

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(2, data.M - 2):
            a1 = sum([C[s] @ data.U[n, i + s - 2, :] for s in range(0, s_max)])
            data.U[n + 1, i, :] = data.U[n, i, :] - a1 
 

"""
Schémas numériques pour la résolution du problème de propagation en 2D non modulé en temps, avec un point source ponctuel
"""

def LaxWendroff2D(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 2D
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
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
                a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i+1, j, :] - data.U[n, i-1,j, :])
                a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, i, j+1, :] - data.U[n, i, j-1, :])
                b1 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, i+1, j, :] + data.U[n, i-1, j, :] - 2 * data.U[n, i, j, :])/data.dx**2)
                b2 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, i, j+1, :] + data.U[n, i, j-1, :] - 2 * data.U[n, i, j, :])/data.dy**2)
                s = ((data.dt/np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * ((i,j) in data.ps) * np.array([0, 0, 1]).transpose()) * (not data.opt) +
                     (data.dt/(data.rho*np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * ((i,j) in data.ps) * np.array([1, 0, 0]).transpose()) * (data.opt))
                data.U[n + 1, i, j, :] = data.U[n, i, j, :] - a1 - a2 + b1 + b2 + s 

def ADER42D(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D
    :param data: Donnee2D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 2D
    """
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
 

"""
Schémas numériques pour la résolution du problème de propagation en 2D non modulé en temps, avec un point source en onde plane
"""

def LaxWendroff2D_BC(data):
    """
       Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 2D pour un front d'onde
       :param data: Donnee2D, regroupe l'ensemble des données du problème
       :return: Donnee2D, solution en vitesse et pression du problème 2D
       """
    data.U = np.zeros((data.N, data.Mx + 2, data.My, 3))
    rc2 = data.rho * data.c ** 2
    A = np.array([[0, 0, 1 / data.rho],
                  [0, 0, 0],
                  [rc2, 0, 0]])
    B = np.array([[0, 0, 0],
                  [0, 0, 1 / data.rho],
                  [0, rc2, 0]])

    for n in trange(0, data.N - 1, ncols=ncols):
        for j in range(1, data.My - 1):
            s = ((data.dt / data.dx * data.S(data.f, (n + 1) * data.dt) * (j == 1) * np.array([0, 0, 1]).transpose()) * (not data.opt) +
                 (data.dt / data.dx * data.S(data.f, (n + 1) * data.dt) * (j == 1) * np.array([0, 1, 0]).transpose()) * (data.opt))
            #condition de periodicité gauche
            a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, 1, j, :] - data.U[n, - 1, j, :])
            a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, 0, j + 1, :] - data.U[n, 0, j - 1, :])
            b1 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, 1, j, :] + data.U[n, - 1, j, :] - 2 * data.U[n, 0, j, :]) / data.dx ** 2)
            b2 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, 0, j + 1, :] + data.U[n, 0, j - 1, :] - 2 * data.U[n, 0, j, :]) / data.dy ** 2)
            data.U[n + 1, 0, j, :] = data.U[n, 0, j, :] - a1 - a2 + b1 + b2 + s

            # condition de periodicité droite
            a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, 0, j, :] - data.U[n, -2, j, :])
            a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, -1, j + 1, :] - data.U[n, -1, j - 1, :])
            b1 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, 0, j, :] + data.U[n, -2, j, :] - 2 * data.U[n, -1, j, :]) / data.dx ** 2)
            b2 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, -1, j + 1, :] + data.U[n, -1, j - 1, :] - 2 * data.U[n, -1, j, :]) / data.dy ** 2)
            data.U[n + 1, -1, j, :] = data.U[n, -1, j, :] - a1 - a2 + b1 + b2 + s
            for i in range(1, data.Mx + 1):
                # Lax-Wendroff 2D
                a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i + 1, j, :] - data.U[n, i - 1, j, :])
                a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, i, j + 1, :] - data.U[n, i, j - 1, :])
                b1 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, i + 1, j, :] + data.U[n, i - 1, j, :] - 2 * data.U[n, i, j, :]) / data.dx ** 2)
                b2 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, i, j + 1, :] + data.U[n, i, j - 1, :] - 2 * data.U[n, i, j, :]) / data.dy ** 2)
                data.U[n + 1, i, j, :] = data.U[n, i, j, :] - a1 - a2 + b1 + b2 + s
    data.U = data.U[:,1:data.Mx + 1, :,:]

def ADER42D_BC(data):
    """
    Utilise le schéma d'ADER4 pour résoudre le problème de propagation 2D pour un front d'onde
    :param data: Donnee2D, regroupe l'ensemble des données du probleme
    :return: np.ndarray(), solution en vitesse et pression du problème 2D
    """
    data.U = np.zeros((data.N, data.Mx + 4, data.My, 3))
    rc2 = data.c**2
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
        for j in range(2,data.My-2):
            s = ((data.dt / data.dx * data.S(data.f, (n + 1) * data.dt) * (j == 2) * np.array([0, 0, 1]).transpose()) * (not data.opt) +
                 (data.dt / data.dx * data.S(data.f, (n + 1) * data.dt) * (j == 2) * np.array([1, 0, 0]).transpose()) * (data.opt))
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
            data.U[n + 1, 0, j, :] = data.U[n,0,j,:] - a1 - a2 - a3 - a4 + s

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
            data.U[n + 1, 1, j, :] = data.U[n,1,j,:] - a1 - a2 - a3 - a4 + s

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
            data.U[n + 1, -1, j, :] = data.U[n,-1,j,:] - a1 - a2 - a3 - a4 + s
            
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
            data.U[n + 1, -2, j, :] = data.U[n,-2,j,:] - a1 - a2 - a3 - a4 + s
            
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
                data.U[n + 1, i, j, :] = data.U[n,i,j,:] - a1 - a2 - a3 - a4 + s
    data.U = data.U[:,2:data.Mx,:,:] 