import numpy as np
from time import sleep
from tqdm import trange
ncols = 125
from donnee import Donnee2D

def LaxWendroff_aniso(data):
    """

    :param data:
    :return:
    """
    print("LaxWendroff Anisotrope()")
    sleep(0.001)
    (mu0, mu1) = data.e
    mu00 = mu0*data.alpha + mu1*(1-data.alpha)
    mu11 = mu0*mu1/(data.alpha*mu1 + mu0*(1-data.alpha))

    data.U = np.zeros((data.N, data.Mx, data.My, 3))
    r = data.rho[0]*data.alpha + data.rho[1]*(1-data.alpha)
    A = np.array([[0, mu00, 0],
                  [1/r, 0, 0],
                  [0, 0, 0]])
    B = np.array([[0, 0, mu11],
                  [0, 0, 0],
                  [1/r, 0, 0]])

    for n in trange(0, data.N - 1, ncols = ncols):
        for i in range(1, data.Mx - 1):
            for j in range(1, data.My - 1):
                s = data.dt / (np.sqrt(data.dx)*r) * data.S(data.f, (n + 1) * data.dt) * ((i,j) in data.ps) * np.array([1, 0, 0]).transpose()
                a1 = (data.dt / (2 * data.dx)) * A @ (data.U[n, i + 1, j, :] - data.U[n, i - 1, j, :])
                a2 = (data.dt / (2 * data.dy)) * B @ (data.U[n, i, j + 1, :] - data.U[n, i, j - 1, :])
                b1 = (0.5 * data.dt**2 * A @ A) @ ((data.U[n, i + 1, j, :] + data.U[n, i - 1, j, :] - 2 * data.U[n, i, j, :]) / data.dx ** 2)
                b2 = (0.5 * data.dt**2 * B @ B) @ ((data.U[n, i, j + 1, :] + data.U[n, i, j - 1, :] - 2 * data.U[n, i, j, :]) / data.dy ** 2)
                b3 = 1/(4*data.dx*data.dy) * (0.5 * data.dt**2 * (A @ B + B @ A)) @ (data.U[n,i + 1, j + 1] - data.U[n, i + 1, j - 1] - data.U[n, i - 1, j + 1] + data.U[n, i - 1, j - 1])
                data.U[n + 1, i, j, :] = data.U[n, i, j, :] - a1 - a2 + b1 + b2 + b3 + s


    ec = 0.5 * r * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2)
    ep = data.U[..., 2] ** 2 / mu00
    data.E = np.sum(ec + ep, axis=(1, 2))