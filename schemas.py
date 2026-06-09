from donnee import *
from numpy.linalg import matrix_power
from tqdm import trange
ncols = 125                                                                       #largeur de la barre de chargement

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
    return data.U

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
            a += gamma[s, m] * (data.dt/data.dx)**(m + 1) * matrix_power(A, m + 1)
        C[s, :, :] = a

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(1, data.M - 1):
            a1 = sum([C[s] @ data.U[n, i + s - 1, :] for s in range(0, s_max)])
            a2 = (data.dt / data.dx) * data.S(data.f,(n+1)*data.dt) * (i == data.xs) * np.array([data.opt, not data.opt])
            data.U[n + 1, i, :] = data.U[n, i, :] - a1 + a2

    return data.U

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

    return data.U

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
                s = ((data.dt/np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * (j == data.ys) * np.array([0, 0, 1]).transpose()) * (not data.opt) +
                     (data.dt /(data.rho*np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * (j == data.ys) * np.array([1, 0, 0]).transpose()) * (data.opt))
                data.U[n + 1, i, j, :] = data.U[n, i, j, :] - a1 - a2 + b1 + b2 + s
    return data.U

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

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(2, data.Mx - 2):
            for j in range(2, data.My - 2):
                a1 = data.dt * (1/(12*data.dx) * A @ (data.U[n,i-2,j,:] - 8*data.U[n,i-1,j,:] + 8*data.U[n,i+1,j,:] - data.U[n,i+2,j,:]) +
                                1/(12*data.dy) * B @ (data.U[n,i,j-2,:] - 8*data.U[n,i,j-1,:] + 8*data.U[n,i+1,j,:] - data.U[n,i+2,j,:]))
                a2 = - (data.c*data.dt)**2/24 * (1/data.dx**2 * (- data.U[n,i-2,j,:] + 16*data.U[n,i-1,j,:] - 30*data.U[n,i,j,:] + 16*data.U[n,i+1,j,:] - data.U[n,i+2,j,:]) +
                                                 1/data.dy**2 * (- data.U[n,i,j-2,:] + 16*data.U[n,i,j-1,:] - 30*data.U[n,i,j,:] + 16*data.U[n,i,j+1,:] - data.U[n,i,j+2,:]))
                a3 = data.c**2*data.dt**3/6 * (A @ (1/(2*data.dx**3) * (- data.U[n,i-2,j,:] + 2*data.U[n,i-1,j,:] - 2*data.U[n,i+1,j,:] + data.U[n,i+2,j,:]) +
                                                    1/(144*data.dx*data.dy**2) * (- data.U[n,i-2,j-2,:] + 8*data.U[n,i-1,j-2,:] - 8*data.U[n,i+1,j-2,:] + data.U[n,i+2,j-2,:]
                                                                                  +16*data.U[n,i-2,j-1,:] -128*data.U[n,i-1,j-1,:] + 128*data.U[n,i+1,j-1,:] - 16*data.U[n,i+2,j-1,:]
                                                                                  -30*data.U[n,i-2,j,:] + 240*data.U[n,i-1,j,:] - 240*data.U[n,i+1,j,:] + 30*data.U[n,i+2,j,:]
                                                                                  +16*data.U[n,i-2,j+1,:] - 128*data.U[n,i-1,j+1,:] + 128*data.U[n,i+1,j+1,:] - 16*data.U[n,i+2,j+1,:]
                                                                                  - data.U[n,i-2,j+2,:] + 8*data.U[n,i-1,j+2,:] - 8*data.U[n,i+1,j+2,:] + data.U[n,i+2,j+2,:])) +
                                               B @ (1/(2*data.dy**3) * (- data.U[n,i,j-2,:] + 2*data.U[n,i,j-1,:] - 2*data.U[n,i,j+1,:] + data.U[n,i,j+2,:]) +
                                                    1/(144*data.dy*data.dx**2) * (- data.U[n,i-2,j-2,:] + 8*data.U[n,i-2,j-1,:] - 8*data.U[n,i-2,j+1,:] + data.U[n,i-2,j+2,:]
                                                                                  +16*data.U[n,i-1,j-2,:] -128*data.U[n,i-1,j-1,:] + 128*data.U[n,i-1,j+1,:] - 16*data.U[n,i-1,j+2,:]
                                                                                  -30*data.U[n,i,j-2,:] + 240*data.U[n,i,j-1,:] - 240*data.U[n,i,j+1,:] + 30*data.U[n,i,j+2,:]
                                                                                  +16*data.U[n,i+1,j-2,:] - 128*data.U[n,i+1,j-1,:] + 128*data.U[n,i+1,j+1,:] - 16*data.U[n,i+1,j+2,:]
                                                                                  - data.U[n,i+2,j-2,:] + 8*data.U[n,i+2,j-1,:] - 8*data.U[n,i+2,j+1,:] + data.U[n,i+2,j+2,:])))
                a4 = - (data.c*data.dt)**4/24 * (1/data.dx**4 * (data.U[n,i-2,j,:] - 4*data.U[n,i-1,j,:] + 6*data.U[n,i,j,:] - 4*data.U[n,i+1,j,:] + data.U[n,i+2,j,:]) +
                                                 1/data.dy**4 * (data.U[n,i,j-2,:] - 4*data.U[n,i,j-1,:] + 6*data.U[n,i,j,:] - 4*data.U[n,i,j+1,:] + data.U[n,i,j+2,:]) +
                                                 1/(data.dx*data.dy)**2 * (data.U[n,i-2,j-2,:] -16*data.U[n,i-2,j-1,:] +30 * data.U[n,i-2,j,:] - 16*data.U[n,i-2,j+1,:] + data.U[n,i-2,j+2,:]
                                                                          -16*data.U[n,i-1,j-2,:] +256*data.U[n,i-1,j-1,:]-480*data.U[n,i-1,j,:] + 256*data.U[n,i-1,j+1,:] - 16*data.U[n,i-1,j+2,:]
                                                                          +30*data.U[n,i,j-2,:] -480*data.U[n,i,j-1,:] +900*data.U[n,i,j,:] -480*data.U[n,i,j+1,:] + 30*data.U[n,i,j+2,:]
                                                                          -16*data.U[n,i+1,j-2,:] +256*data.U[n,i+1,j-1,:]-480*data.U[n,i+1,j,:] + 256*data.U[n,i+1,j+1,:] - 16*data.U[n,i+1,j+2,:]
                                                                          + data.U[n,i+2,j-2,:] -16*data.U[n,i+2,j-1,:] +30*data.U[n,i+2,j,:] - 16*data.U[n,i+2,j+1,:] + data.U[n,i+2,j+2,:]))
                s = ((data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * (j == data.ys) * np.array([0, 0, 1]).transpose()) * (not data.opt) +
                     (data.dt / (data.rho * np.sqrt(data.dx)) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * (j == data.ys) * np.array([1, 0, 0]).transpose()) * (data.opt))
                data.U[n + 1, i, j, :] = data.U[n,i,j,:] - a1 - a2 - a3 - a4 + s

    return data.U