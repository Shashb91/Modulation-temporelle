from schemas import*
from analytique import*
from tracer import*
from tqdm import trange
from sauvegarde import *

def erreur1D(t, M_li = [100, 200, 400, 800, 1600], f = "LW"):
    """
    Calcule l'erreur avec la norme L1, au temps d'indice t, avec les discretisations de M_li
    :param M_li: [int], liste des dscretisations à calculer
    :param t: float, temps en seconde
    :return: np.darray, matrice des erreurs en vitesse et pression
    """
    nM = len(M_li)
    eps = np.zeros((len(M_li),2))
    for i in trange(nM):
        data1, data2 = Donnee1D(M = int(M_li[i]), label = f ), Donnee1D(M = int(M_li[i]), label = "Analytique")

        if f == "LW": U = LaxWendroff1D(data1)
        elif f == "ADER4": U = ADER41D(data1)
        elif f == "ADER2": U = ADER21D(data1)

        u = analytique1D(data2)
        ti = int(t/data1.dt)
        sauvegarder(data1, f + "_" + M_li[i] + "_")
        sauvegarder(data2, "Analytique_" + M_li[i] + "_")

        if f == "ADER4": eps[i, :] = np.sum([np.abs(U[ti,n,:] - u[ti,n,:]) for n in range(2,data1.M-2)])/(data1.M-4)
        else: eps[i, :] = np.sum([np.abs(U[ti,n,:] - u[ti,n,:]) for n in range(data1.M)])/data1.M
    return eps