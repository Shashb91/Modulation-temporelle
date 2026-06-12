"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff et d'ADER4 1D/2D en mileu non
modulé en temps
============================================================
"""
from erreur import *
from sauvegarde import *
from schemas import*

#============================================================
#Implémentation de la résolution 1D homogène
#============================================================

data1 = Donnee1D(M = 200, label = "Lax Wendroff")                                #Lax Wendroff
data2 = Donnee1D(M = 200, label = "Analytique")                                  #ADER4
data3 = Donnee1D(M = 200, label = "ADER4")                                       #Solution analytique

U_LW = LaxWendroff1D(data1)
anim1D(data1)

u = analytique1D(data2)
anim1D(data2)

U_ADER4 = ADER41D(data3)
anim1D(data3)

data1 = charger('.save/data1_06-08_15-20-38.pkl')
data2 = charger('.save/data2_06-08_15-20-38.pkl')
anim1D_comparaison(data1, data2)                                                      #Comparaison LaxWendroff VS Analytique
anim1D_comparaison(data3, data2)                                                      #Comparaison ADER4 VS Analytique

sauvegarder(data1,"data1_")
sauvegarder(data2,"data2_")

#============================================================
#Etude de l'erreur à t = 0.05s ! (t < xf/(2*c))
#============================================================

epsLW = erreur1D(0.05, f = "LW")
erreur1D_trace(epsLW, f = "LW")

epsADER2 = erreur1D(0.05, f = "ADER2")
erreur1D_trace(epsADER2, f = "ADER2")

epsADER4 = erreur1D(0.05, M_li = [150,200,250], f = "ADER4")
erreur1D_trace(epsADER4,M_li = np.array([150,200,250]), f = "ADER4")


#============================================================
#Implémentation de la résolution 2D homogène
#============================================================

data4 = Donnee2D(label = "Lax Wendroff 2D",opt = False, source=pt_source_pression, xc = (0,300), yc = (0,300), c = 1500, rho = 1000, tc = (0, 0.08), CFL = 0.6, Mx = 300, My = 300, f = 20)
data5 = Donnee2D(label = "ADER4 2D",opt = False, source=pt_source_pression, xc = (0,300), yc = (0,300), c = 1500, rho = 1000, tc = (0, 0.08), CFL = 0.6, Mx = 300, My = 300, f = 20)

U = LaxWendroff2D(data4)
sauvegarder(data4,opt = True)

u = ADER42D(data5)
sauvegarder(data5,opt = True)
data4 = charger('.save/Lax Wendroff 2D_06-10_15-55-20.pkl')
anim2D(data4)
anim2D(data5)