from schema import*
from analytique import*
from tracer import*
from tqdm import trange
from sauvegarde import *
ncols = 125                                                                       #largeur de la barre de chargement

def erreur1D(t, M_li = [100, 200, 400, 800, 1600], f = "LW"):
    """
    Calcule l'erreur avec la norme L1, au temps d'indice t, avec les discretisations de M_li
    :param f: str, label
    :param M_li: [int], liste des dscretisations à calculer
    :param t: float, temps en seconde
    :return: np.darray, matrice des erreurs en vitesse et pression
    """
    nM = len(M_li)
    eps = np.zeros((len(M_li),2))
    for i in trange(nM, ncols = ncols):
        data1, data2 = Donnee1D(M = int(M_li[i]), label = f ), Donnee1D(M = int(M_li[i]), label = "Analytique")

        if f == "LW": LaxWendroff1D(data1)
        elif f == "ADER4": ADER41D(data1)
        elif f == "ADER2": ADER21D(data1)

        analytique1D(data2)
        ti = int(t/data1.dt)
        sauvegarder(data1, f + "_" + str(M_li[i]) + "_", opt=True)
        sauvegarder(data2, "Analytique_" + str(M_li[i]) + "_", opt=True)

        if f == "ADER4": eps[i, :] = np.sum([np.abs(U[ti,n,:] - u[ti,n,:]) for n in range(2,data1.M-2)])/(data1.M-4)
        else: eps[i, :] = np.sum([np.abs(U[ti,n,:] - u[ti,n,:]) for n in range(data1.M)])/data1.M
    return eps


def erreur2D(t, M_li = [100, 150, 200, 250], f = "LW"):
    """
    Calcule l'erreur avec la norme L1, au temps d'indice t, avec les discretisations de M_li
    :param f: str, label
    :param M_li: [int], liste des dscretisations à calculer
    :param t: float, temps en seconde
    :return: np.darray, matrice des erreurs en vitesse et pression
    """
    nM = len(M_li)
    eps = np.zeros((len(M_li),2))
    for i in trange(nM, ncols = ncols):
        data1 = Donnee2D(opt = True, S=pt_source_1D, xc = (0,300), yc = (0,300), c = 1500, rho = 1000, tc = (0, 0.2),CFL = 0.6, My = int(M_li[i]), Mx = int(M_li[i]), label = f)
        data2 = Donnee1D(M = int(M_li[i]), N = data1.N, label = "Analytique", xs = 0, f = 20, rho = 1000, c = 1500, CFL = 0.6, tc = (0, 0.2), xc = (0,300))

        if f == "LW": LaxWendroff2D_cauchy(data1)
        elif f == "ADER4": ADER42D_cauchy(data1)

        data1_ = Donnee1D(M = int(M_li[i]), N = data1.N, label=f + "_", xc=data1.xc, tc=data1.tc, x=data1.x, t=data1.t, c=data1.c,rho=data1.rho, CFL=0.6)
        data1_.U = data1.U[:, int(M_li[i])//2, :, 1:]

        analytique1D_cauchy(data2)
        ti = int(t/data1.dt)

        if f == "ADER4": eps[i, :] = np.sum([np.abs(data2.U[ti,n,:] - data1_.U[ti,n,:]) for n in range(2,data2.M-2)])/(data2.M-4)
        else: eps[i, :] = np.sum([np.abs(data2.U[ti,n,:] - data1_.U[ti,n,:]) for n in range(data2.M)])/data2.M
    return eps