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
        print(i)
        data1, data2 = Donnee1D(M = int(M_li[i])), Donnee1D(M = int(M_li[i]))
        U = LaxWendroff1D(data1)
        u = analytique1D(data2)
        ti = int(t/data1.dt)
        print(t, ti, data1.N, data1.dt)
        tracer1D_analytiqueVSnumerique(ti, data1, data2)
        eps[i, :] = np.sum([np.abs(U[ti,n,:] - u[ti,n,:]) for n in range(data1.M)])/data1.M
    return eps

def erreur1D_temporel(tf, nt = 4, M_li = [100, 200, 400, 800, 1600]):
    """
    Calcule l'erreur avec la norme L1,  avec la fonction erreur1D, et envoie une matrice des erreurs en vitesse et pression à des temps réguliers
    :param M_li: [int], liste des dscretisations à calculer
    :param N: int, iscrétisation temporel
    :param nt: int, nombre de pas de temps
    :return: np.ndarray, matrice des erreurs en vitesse et pression à des temps réguliers
    """
    nM = len(M_li)
    eps = np.zeros((nt, nM,2))
    li_t = [int(i*tf/nt) for i in range(nt)]

    for i in range(nt):
        eps[i, : ,: ] = erreur1D(li_t[i], M_li)
    return eps