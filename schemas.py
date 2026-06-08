from donnee import *
from numpy.linalg import matrix_power


def LaxWendroff1D(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 1D
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, solution en vitesse et pression du problème 1D
    """
    data.U = np.zeros((data.N, data.M, 2))
    A = np.array([[0, 1/data.rho], [data.rho * data.c**2, 0]])

    for n in range(0, data.N - 1):
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

    for n in range(0, data.N - 1):
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

    for n in range(0, data.N - 1):
        for i in range(2, data.M - 2):
            a1 = sum([C[s] @ data.U[n, i + s - 2, :] for s in range(0, s_max)])
            a2 = (data.dt / data.dx) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * np.array([data.opt, not data.opt])
            data.U[n + 1, i, :] = data.U[n, i, :] - a1 + a2

    return data.U

#def ADER41D(data): inutile
#    data.U = np.zeros((data.N, data.M, 2))
#    A = np.array([[0, 1 / data.rho], [data.rho * data.c ** 2, 0]])
#    beta = data.dt / data.dx
#
#    for n in range(0, data.N - 1):
#        for i in range(2, data.M - 2):
#            Um2 = data.U[n, i-2, :]
#            Um1 = data.U[n, i-1, :]
#            U0  = data.U[n, i,   :]
#            Up1 = data.U[n, i+1, :]
#            Up2 = data.U[n, i+2, :]
#
#            D1 = (1/12)*(Um2 - Up2) + (2/3)*(Up1 - Um1)
#            D2 = (1/24)*(Um2 + Up2) - (2/3)*(Um1 + Up1) + (5/4)*U0
#            D3 = (1/12)*(Up2 - Um2) - (1/6)*(Up1 - Um1)
#            D4 = -(1/24)*(Um2 + Up2) + (1/6)*(Um1 + Up1) - (1/4)*U0
#
#            a1 = (beta * (A @ D1) + beta**2 * (A @ A @ D2) + beta**3 * (A @ A @ A @ D3) + beta**4 * (A @ A @ A @ A @ D4))
#            a2 = (data.dt / data.dx) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * np.array([data.opt, not data.opt])
#
#            data.U[n + 1, i, :] = data.U[n, i, :] - a1 + a2
#
#    return data.U

def LaxWendroff2D(data):
    """
    Utilise le schéma de Lax-Wendroff pour résoudre le problème de propagation 2D
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, solution en vitesse et pression du problème 2D
    """
    data.U = np.zeros((data.N, data.Mx, data.My, 3))
    A = np.array([[0,0, 1 / data.rho],
                  [0,0, 0],
                  [data.rho * data.c ** 2, 0, 0]])
    B = np.array([[0,0,0],
                  [0,0,1/data.rho],
                  [0, data.rho*data.c**2, 0]])

    for n in range(0, data.N - 1):
        for i in range(1, data.Mx - 1):
            for j in range(1, data.My - 1):
                a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i+1, j, :] - data.U[n, i-1,j, :])
                b1 = (data.dt / (2 * data.dy)) * B @ (data.U[n, i, j+1, :] - data.U[n, i, j-1, :])
                c1 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, i+1, j, :] + data.U[n, i-1, j, :] - 2 * data.U[n, i, j, :])/data.dx**2)
                c2 = (0.5 * (data.dt * data.c) ** 2) * ((data.U[n, i, j+1, :] + data.U[n, i, j-1, :] - 2 * data.U[n, i, j, :])/data.dy**2)
                data.U[n + 1, i, j, :] = data.U[n, i, j, :] - a1 - b1 + c1 + c2 + data.dt**2 /(data.dx*data.dy) * data.S(data.f, (n + 1) * data.dt) * (i == data.xs) * (j == data.ys) * np.array([1,0,0]).transpose()
    return data.U
