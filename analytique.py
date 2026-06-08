from donnee import*

def signe(a):
    return -1 * (a < 0) + 1 * (a >= 0)

def analytique1D(data):
    """
    Solution analytique pour le problème de propagation
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: Donnee1D, regroupe l'ensemble des données du problème avec la solution analytique
    """
    data.U = np.zeros((data.N, data.M, 2))
    for n in range(0, data.N-1):
        for i in range(2, data.M-2):
            ind = n*data.dt - np.abs(i-data.xs)*data.dx/data.c
            a = 1/(2*data.c) * data.S(data.f, ind)
            b = signe(i - data.M//2) * data.rho/2 * data.S(data.f, ind)
            data.U[n, i, :] = np.array([a,b])
    return data.U

def analytique2D(data):
    """
    Solution analytique pour le problème de propagation
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: Donnee2D, regroupe l'ensemble des données du problème avec la solution analytique
    """
    data.U = np.zeros((data.N, data.Mx, data.My, 2))
    for n in range(0, data.N - 1):
        for i in range(2, data.Mx - 2):
            for j in range(2, data.My - 2):
                ind = n * data.dt - (np.abs(i - data.xs) * data.dx + np.abs(j - data.ys) * data.dy)/ data.c
                a = 1 / (2 * data.c) * data.S(data.f, ind)
                b = signe(i - data.xs) * signe(j-data.ys)* data.rho / 2 * data.S(data.f, ind)
                data.U[n, i, :] = np.array([a, b])
    return data.U