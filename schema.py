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
    A2 = A @ A
    coef1 = data.dt / (2 * data.dx)
    coef2 = 0.5 * (data.dt / data.dx) ** 2

    for n in trange(0, data.N - 1, ncols = ncols):
        Un = data.U[n]
        diff = Un[2:, :] - Un[:-2, :]                     # i+1 - i-1
        lap = Un[2:, :] + Un[:-2, :] - 2 * Un[1:-1, :]    # i+1 + i-1 - 2i
        a1 = coef1 * (diff @ A.T)
        a2 = coef2 * (lap @ A2.T)
        data.U[n+1, 1:-1, :] = Un[1:-1, :] - a1 + a2
        if 1 <= data.xs <= data.M - 2:
            data.U[n+1, data.xs, :] += data.dt/data.dx * data.S(data.f,(n+1)*data.dt) * np.array([1,0]).transpose()

    data.E = np.sum((0.5 * data.rho * data.U[..., 0] ** 2 + data.U[..., 1] ** 2 / (2*data.rho * data.c ** 2)) * data.dx, axis=1)

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
        Un = data.U[n]
        a1 = sum(Un[s:data.M - 2 + s, :] @ C[s].T for s in range(0, s_max))
        data.U[n + 1, 1:-1, :] = Un[1:-1, :] - a1
        if 1 <= data.xs <= data.M - 2:
            data.U[n + 1, data.xs, :] += (data.dt / data.dx) * data.S(data.f,(n+1)*data.dt) * np.array([data.opt, not data.opt])

    data.E = np.sum((0.5 * data.rho * data.U[..., 0] ** 2 + data.U[..., 1] ** 2 / (2*data.rho * data.c ** 2)) * data.dx, axis=1)

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
        Un = data.U[n]
        a1 = sum(Un[s:data.M - 4 + s, :] @ C[s].T for s in range(0, s_max))
        data.U[n + 1, 2:-2, :] = Un[2:-2, :] - a1
        if 2 <= data.xs <= data.M - 3:
            data.U[n + 1, data.xs, :] += (data.dt / data.dx) * data.S(data.f, (n + 1) * data.dt) * np.array([data.opt, not data.opt])

    data.E = np.sum((0.5 * data.rho * data.U[..., 0] ** 2 + data.U[..., 1] ** 2 / (2*data.rho * data.c ** 2)) * data.dx, axis=1)

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
        return (coeff[0] * np.sin(omega * t) + coeff[1] * np.sin(omega * 2 * t) + coeff[2] * np.sin(omega * 4 * t) + coeff[3] * np.sin(omega * 8 * t)) * ((0 < t) & (t < 1 / f))

    data.U = np.zeros((data.N, data.M, 2))
    A = np.array([[0, 1/data.rho], [data.rho * data.c**2, 0]])
    A2 = A @ A
    coef1 = data.dt / (2 * data.dx)
    coef2 = 0.5 * (data.dt / data.dx) ** 2

    # init
    i = np.arange(data.M)
    arg = 1/data.f + data.tc[0] - data.dx * (i - 1) / data.c
    data.U[0] = (1/data.c * fct(data.f, arg))[:, None] * np.array([1, data.c*data.rho])

    for n in trange(0, data.N - 1, ncols = ncols):
        Un = data.U[n]
        diff = Un[2:, :] - Un[:-2, :]
        lap = Un[2:, :] + Un[:-2, :] - 2 * Un[1:-1, :]
        a1 = coef1 * (diff @ A.T)
        a2 = coef2 * (lap @ A2.T)
        data.U[n+1, 1:-1, :] = Un[1:-1, :] - a1 + a2

    data.E = np.sum((0.5 * data.rho * data.U[..., 0] ** 2 + data.U[..., 1] ** 2 / (2*data.rho * data.c ** 2)) * data.dx, axis=1)

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
        return (coeff[0] * np.sin(omega * t) + coeff[1] * np.sin(omega * 2 * t) + coeff[2] * np.sin(omega * 4 * t) + coeff[3] * np.sin(omega * 8 * t)) * ((0 < t) & (t < 1 / f))

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
    i = np.arange(data.M)
    arg = 1/data.f + data.tc[0] - data.dx * (i - 1) / data.c
    data.U[0] = (1/data.c * fct(data.f, arg))[:, None] * np.array([1, data.c*data.rho])

    for s in range(0, s_max):
        a = np.zeros((2, 2))
        for m in range(0, m_max):
            a += gamma[s, m] * (data.dt/data.dx)**(m + 1) * matrix_power(A, m + 1)
        C[s, :, :] = a

    for n in trange(0, data.N - 1, ncols = ncols):
        Un = data.U[n]
        a1 = sum(Un[s:data.M - 4 + s, :] @ C[s].T for s in range(0, s_max))
        data.U[n + 1, 2:-2, :] = Un[2:-2, :] - a1

    data.E = np.sum((0.5 * data.rho * data.U[..., 0] ** 2 + data.U[..., 1] ** 2 / (2*data.rho * data.c ** 2)) * data.dx, axis=1)


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
    ca = data.dt / (2 * data.dx)
    cb = data.dt / (2 * data.dy)
    cc = 0.5 * (data.dt * data.c) ** 2 / data.dx ** 2
    cd = 0.5 * (data.dt * data.c) ** 2 / data.dy ** 2

    for n in trange(0, data.N - 1, ncols = ncols):
        Un = data.U[n]
        xp, xm = Un[2:, 1:-1, :], Un[:-2, 1:-1, :]       # (i+1,j), (i-1,j)
        yp, ym = Un[1:-1, 2:, :], Un[1:-1, :-2, :]       # (i,j+1), (i,j-1)
        c0 = Un[1:-1, 1:-1, :]                             # (i,j)
        a1 = ca * ((xp - xm) @ A.T)
        a2 = cb * ((yp - ym) @ B.T)
        b1 = cc * (xp + xm - 2 * c0)
        b2 = cd * (yp + ym - 2 * c0)
        data.U[n + 1, 1:-1, 1:-1, :] = c0 - a1 - a2 + b1 + b2
        sval = data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * np.array([0, 0, 1]).transpose()
        for (pi, pj) in data.ps:
            if 1 <= pi <= data.Mx - 2 and 1 <= pj <= data.My - 2:
                data.U[n + 1, pi, pj, :] += sval

    ec = 0.5 * data.rho * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (2*data.rho * data.c ** 2)
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
        Un = data.U[n]
        def sh(p, q):
            return Un[2 + p:data.Mx - 2 + p, 2 + q:data.My - 2 + q, :]
        X1 = sh(-2,0) - 8*sh(-1,0) + 8*sh(1,0) - sh(2,0)
        Y1 = sh(0,-2) - 8*sh(0,-1) + 8*sh(0,1) - sh(0,2)
        a1 = data.dt * (c[0] * (X1 @ A.T) + c[0] * (Y1 @ B.T))
        X2 = - sh(-2,0) + 16*sh(-1,0) - 30*sh(0,0) + 16*sh(1,0) - sh(2,0)
        Y2 = - sh(0,-2) + 16*sh(0,-1) - 30*sh(0,0) + 16*sh(0,1) - sh(0,2)
        a2 = b2 * (c[1] * X2 + c[1] * Y2)
        X3 = - sh(-2,0) + 2*sh(-1,0) - 2*sh(1,0) + sh(2,0)
        Xmix = (- sh(-2,-2) + 8*sh(-1,-2) - 8*sh(1,-2) + sh(2,-2)
                + 16*sh(-2,-1) -128*sh(-1,-1) + 128*sh(1,-1) - 16*sh(2,-1)
                - 30*sh(-2,0) + 240*sh(-1,0) - 240*sh(1,0) + 30*sh(2,0)
                + 16*sh(-2,1) - 128*sh(-1,1) + 128*sh(1,1) - 16*sh(2,1)
                - sh(-2,2) + 8*sh(-1,2) - 8*sh(1,2) + sh(2,2))
        Y3 = - sh(0,-2) + 2*sh(0,-1) - 2*sh(0,1) + sh(0,2)
        Ymix = (- sh(-2,-2) + 8*sh(-2,-1) - 8*sh(-2,1) + sh(-2,2)
                + 16*sh(-1,-2) -128*sh(-1,-1) + 128*sh(-1,1) - 16*sh(-1,2)
                - 30*sh(0,-2) + 240*sh(0,-1) - 240*sh(0,1) + 30*sh(0,2)
                + 16*sh(1,-2) - 128*sh(1,-1) + 128*sh(1,1) - 16*sh(1,2)
                - sh(2,-2) + 8*sh(2,-1) - 8*sh(2,1) + sh(2,2))
        a3 = b3 * ((c[2] * X3 + c[3] * Xmix) @ A.T + (c[2] * Y3 + c[3] * Ymix) @ B.T)
        X4 = sh(-2,0) - 4*sh(-1,0) + 6*sh(0,0) - 4*sh(1,0) + sh(2,0)
        Y4 = sh(0,-2) - 4*sh(0,-1) + 6*sh(0,0) - 4*sh(0,1) + sh(0,2)
        XYmix = (sh(-2,-2) -16*sh(-2,-1) +30*sh(-2,0) - 16*sh(-2,1) + sh(-2,2)
                 -16*sh(-1,-2) +256*sh(-1,-1) -480*sh(-1,0) + 256*sh(-1,1) - 16*sh(-1,2)
                 +30*sh(0,-2) -480*sh(0,-1) +900*sh(0,0) -480*sh(0,1) + 30*sh(0,2)
                 -16*sh(1,-2) +256*sh(1,-1) -480*sh(1,0) + 256*sh(1,1) - 16*sh(1,2)
                 + sh(2,-2) -16*sh(2,-1) +30*sh(2,0) - 16*sh(2,1) + sh(2,2))
        a4 = b4 * (c[4] * X4 + c[4] * Y4 + 2 * c[5] * XYmix)
        data.U[n + 1, 2:-2, 2:-2, :] = sh(0,0) - a1 - a2 - a3 - a4
        sval = data.dt / np.sqrt(data.dx) * data.S(data.f, (n + 1) * data.dt) * np.array([0, 0, 1]).transpose()
        for (pi, pj) in data.ps:
            if 2 <= pi <= data.Mx - 3 and 2 <= pj <= data.My - 3:
                data.U[n + 1, pi, pj, :] += sval
    ec = 0.5 * data.rho * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (2*data.rho * data.c ** 2)
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

    ca = data.dt / (2 * data.dx)
    cb = data.dt / (2 * data.dy)
    cc = 0.5 * (data.dt * data.c) ** 2 / data.dx ** 2
    cd = 0.5 * (data.dt * data.c) ** 2 / data.dy ** 2

    # init : la valeur ne dépend que de j (identique sur tout i)
    j = np.arange(data.My)
    Sj = np.array([data.S(data.f, 1/data.f + data.tc[0] - data.dy * (jj - 1) / data.c) for jj in j])
    vec = np.array([np.sin(theta), np.cos(theta), data.rho * data.c])
    data.U[0, :, :, :] = (1/data.c) * Sj[None, :, None] * vec[None, None, :]

    for n in trange(0, data.N - 1, ncols=ncols):
        Un = data.U[n]
        xp = np.roll(Un, -1, axis=0)[:, 1:-1, :]     # U[i+1, j] (périodique en x)
        xm = np.roll(Un, 1, axis=0)[:, 1:-1, :]      # U[i-1, j]
        c0 = Un[:, 1:-1, :]                            # U[i, j]
        yp = Un[:, 2:, :]                              # U[i, j+1]
        ym = Un[:, :-2, :]                             # U[i, j-1]
        a1 = ca * ((xp - xm) @ A.T)
        a2 = cb * ((yp - ym) @ B.T)
        b1 = cc * (xp + xm - 2 * c0)
        b2 = cd * (yp + ym - 2 * c0)
        data.U[n + 1, :, 1:-1, :] = c0 - a1 - a2 + b1 + b2

    data.U = data.U[:,1:data.Mx + 1, :,:]
    ec = 0.5 * data.rho * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (2*data.rho * data.c ** 2)
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

    # init : la valeur ne depend que de j (identique sur tout i)
    j = np.arange(data.My)
    Sj = np.array([fct(data.f, 1/data.f + data.tc[0] - data.dy * (jj - 1) / data.c) for jj in j])
    vec = np.array([np.sin(theta), np.cos(theta), data.rho * data.c])
    data.U[0, :, :, :] = (1/data.c) * Sj[None, :, None] * vec[None, None, :]

    for n in trange(0, data.N - 1, ncols = ncols):
        Un = data.U[n]
        def sh(p, q):
            return np.roll(Un, -p, axis=0)[:, 2 + q:data.My - 2 + q, :]
        X1 = sh(-2,0) - 8*sh(-1,0) + 8*sh(1,0) - sh(2,0)
        Y1 = sh(0,-2) - 8*sh(0,-1) + 8*sh(0,1) - sh(0,2)
        a1 = data.dt * (c[0] * (X1 @ A.T) + c[0] * (Y1 @ B.T))
        X2 = - sh(-2,0) + 16*sh(-1,0) - 30*sh(0,0) + 16*sh(1,0) - sh(2,0)
        Y2 = - sh(0,-2) + 16*sh(0,-1) - 30*sh(0,0) + 16*sh(0,1) - sh(0,2)
        a2 = b2 * (c[1] * X2 + c[1] * Y2)
        X3 = - sh(-2,0) + 2*sh(-1,0) - 2*sh(1,0) + sh(2,0)
        Xmix = (- sh(-2,-2) + 8*sh(-1,-2) - 8*sh(1,-2) + sh(2,-2)
                + 16*sh(-2,-1) -128*sh(-1,-1) + 128*sh(1,-1) - 16*sh(2,-1)
                - 30*sh(-2,0) + 240*sh(-1,0) - 240*sh(1,0) + 30*sh(2,0)
                + 16*sh(-2,1) - 128*sh(-1,1) + 128*sh(1,1) - 16*sh(2,1)
                - sh(-2,2) + 8*sh(-1,2) - 8*sh(1,2) + sh(2,2))
        Y3 = - sh(0,-2) + 2*sh(0,-1) - 2*sh(0,1) + sh(0,2)
        Ymix = (- sh(-2,-2) + 8*sh(-2,-1) - 8*sh(-2,1) + sh(-2,2)
                + 16*sh(-1,-2) -128*sh(-1,-1) + 128*sh(-1,1) - 16*sh(-1,2)
                - 30*sh(0,-2) + 240*sh(0,-1) - 240*sh(0,1) + 30*sh(0,2)
                + 16*sh(1,-2) - 128*sh(1,-1) + 128*sh(1,1) - 16*sh(1,2)
                - sh(2,-2) + 8*sh(2,-1) - 8*sh(2,1) + sh(2,2))
        a3 = b3 * ((c[2] * X3 + c[3] * Xmix) @ A.T + (c[2] * Y3 + c[3] * Ymix) @ B.T)
        X4 = sh(-2,0) - 4*sh(-1,0) + 6*sh(0,0) - 4*sh(1,0) + sh(2,0)
        Y4 = sh(0,-2) - 4*sh(0,-1) + 6*sh(0,0) - 4*sh(0,1) + sh(0,2)
        XYmix = (sh(-2,-2) -16*sh(-2,-1) +30*sh(-2,0) - 16*sh(-2,1) + sh(-2,2)
                 -16*sh(-1,-2) +256*sh(-1,-1) -480*sh(-1,0) + 256*sh(-1,1) - 16*sh(-1,2)
                 +30*sh(0,-2) -480*sh(0,-1) +900*sh(0,0) -480*sh(0,1) + 30*sh(0,2)
                 -16*sh(1,-2) +256*sh(1,-1) -480*sh(1,0) + 256*sh(1,1) - 16*sh(1,2)
                 + sh(2,-2) -16*sh(2,-1) +30*sh(2,0) - 16*sh(2,1) + sh(2,2))
        a4 = b4 * (c[4] * X4 + c[4] * Y4 + c[5] * XYmix)
        data.U[n + 1, :, 2:-2, :] = sh(0,0) - a1 - a2 - a3 - a4

    data.U = data.U[:,2:data.Mx,:,:]
    data.U[...,0] = np.round(data.U[...,0], 16)

    ec = 0.5 * data.rho * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / (2*data.rho * data.c ** 2)
    data.E = np.sum((ec + ep)*data.dx*data.dy, axis=(1, 2))