from tracer import tracer1D_analytiqueVSnumerique
from LaxWendroff import*
import numpy as np

def erreur1D(t, M_li = [100, 200, 400, 800, 1600]):
    """
    Calcule l'erreur avec la norme L1, au temps d'indice t, entre 10^Mc[0] et 10^Mc[1] points nM fois
    :param M_li: [int], liste des dscretisations à calculer
    :param t: float, temps en seconde
    :return: np.darray, matrice des erreurs en vitesse et pression
    """
    nM = len(M_li)
    eps = np.zeros((len(M_li),2))
    for i in range(nM):
        data1, data2 = Donnee1D(M = int(M_li[i])), Donnee1D(M = int(M_li[i]))
        U = LaxWendroff1D(data1)
        u = analytique1D(data2)
        ti = int(t/data1.dt)
        #tracer1D_analytiqueVSnumerique(ti, data1, data2)
        eps[i, :] = np.sum([np.abs(U[ti,n,:] - u[ti,n,:]) for n in range(data1.M)])/data1.M
    return eps