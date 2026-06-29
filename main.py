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
from schema import*

"""
============================================================
Implémentation de la résolution 1D homogène
============================================================
"""
#CFL = 0.6 afin de pouvoir comparer avec les projections 2D !!! par défaut à 0.95

data1 = Donnee1D(M = 150, CFL = 0.6, xc = (0, 300), tc = (0, 0.2),c = 1500, rho = 1000, f = 20, label = "Lax Wendroff")                                #Lax Wendroff
data2 = Donnee1D(M = 150, CFL = 0.6, xc = (0, 300), tc = (0, 0.2),c = 1500, rho = 1000, f = 20, label = "Analytique")                                  #ADER4
data3 = Donnee1D(M = 150, CFL = 0.6, xc = (0, 300), tc = (0, 0.2),c = 1500, rho = 1000, f = 20, label = "ADER4")                                       #Solution analytique

LaxWendroff1D(data1)
anim1D(data1)

analytique1D(data2)
anim1D(data2)

ADER41D(data3)
anim1D(data3)

# data1 = charger('.save/data1_06-08_15-20-38.pkl')
# data2 = charger('.save/data2_06-08_15-20-38.pkl')
anim1D_comparaison(data1, data2, interval = 30)                                                      #Comparaison LaxWendroff VS Analytique
anim1D_comparaison(data3, data2, interval = 30)                                                      #Comparaison ADER4 VS Analytique

sauvegarder(data1)
sauvegarder(data2)

data2 = Donnee1D(M = 150,c = 1500, rho = 1000, f=20,xc = (0, 300), tc = (0, 0.2), label = "Analytique", CFL = 0.6)
analytique1D_cauchy(data2)
anim1D(data2)

LaxWendroff1D_cauchy(data1)
anim1D(data1, interval = 30)

ADER41D_cauchy(data3)
anim1D(data3, interval = 30)

tracer1D_comparaison(10, data1, data2)
anim1D_comparaison(data1, data2, interval = 30)                                                      #Comparaison LaxWendroff VS Analytique
anim1D_comparaison(data3, data2, interval = 30)                                                      #Comparaison ADER4 VS Analytique

"""
============================================================
Etude de l'erreur à t = 0.05s ! (t < xf/(2*c))
============================================================
"""

epsLW = erreur1D(0.05, f = "LW")
erreur_trace(epsLW, f = "LW")

epsADER2 = erreur1D(0.05, f = "ADER2")
erreur_trace(epsADER2, f = "ADER2")

epsADER4 = erreur1D(0.05, M_li = [150,200,250], f = "ADER4")
erreur_trace(epsADER4,M_li = np.array([150,200,250]), f = "ADER4")

"""
============================================================
Implémentation de la résolution 2D homogène
============================================================
"""
data1 = Donnee2D(label="Lax Wendroff 2D", S = pt_source_2D, Mx = 200, My = 200, CFL = 0.6, opt = False, f = 10, tc = (0, 0.08), xc = (0, 300), yc = (0, 300))
print(data1)
LaxWendroff2D(data1)
# data1 = charger('.save_2D_nmt/Lax Wendroff 2D_06-23_16-30-26.pkl')
sauvegarder(data1, '.save_2D_nmt/Lax Wendroff 2D_06-23_16-30-26.pkl')
anim2D(data1)


#Cas de l'onde plane avec un problème de Cauchy
M = 150
data4 = Donnee2D(label = "Lax Wendroff 2D",opt = True, S=pt_source_1D, xc = (0,300), yc = (0,300), c = 1500, rho = 1000, tc = (0, 0.2),CFL = 0.6, Mx = M, My = M, f = 20, e= 2.25e9)
data5 = Donnee2D(label = "ADER4 2D",opt = True, S=pt_source_1D, xc = (0,300), yc = (0,300), c = 1500, rho = 1000, tc = (0, 0.2), CFL = 0.6, Mx = M, My = M, f = 20, e = 2.25e9)

data2 = Donnee1D(M = M, label = "Analytique", xs = 0, f = 20, rho = 1000, c = 1500, CFL = 0.6, tc = (0, 0.2), xc = (0,300))
analytique1D_cauchy(data2)

LaxWendroff2D_cauchy(data4)
sauvegarder(data4, ".save_2D_nmt/")
# data4 = charger('.save_2D_nmt/Lax Wendroff 2D_cauchy.pkl')
anim2D(data4)

data4_ = data4.projection((data4.Mx//2,0))
anim1D(data4_, interval = 30)
anim1D_comparaison(data4_, data2, interval = 30)
anim1D_comparaison(data4_, data1, interval = 30)

ADER42D_cauchy(data5)
sauvegarder(data5)
# data5 = charger('.save_2D_nmt/ADER4 2D_cauchy.pkl')
anim2D(data5)

data5_ = data5.projection((data5.Mx//2,0))
anim1D(data5_, interval = 30)
anim1D_comparaison(data5_,data2, interval=30)
anim1D_comparaison(data5_,data3, interval=30)

erreur_LW = []
for M in [125,150,200,250]:
    LW2D = Donnee2D(label = "Lax Wendroff 2D",opt = True, S=pt_source_1D, xc = (0,300), yc = (0,300), c = 1500, rho = 1000, tc = (0, 0.2),CFL = 0.6, Mx = M, My = M, f = 20, e= 2.25e9)
    LaxWendroff2D_cauchy(LW2D)
    LW2D_ = LW2D.projection((LW2D.Mx//2,0))

    LW1D = Donnee1D(M = M, CFL = 0.6, xc = (0, 300), tc = (0, 0.2),c = 1500, rho = 1000, f = 20, label = "Lax Wendroff")
    LaxWendroff1D_cauchy(LW1D)
    print(LW2D_)
    print(LW1D)

    # anim1D_comparaison(LW1D, LW2D_)
    t = int(0.1/LW1D.dt)
    erreur_LW.append(np.sum([np.abs(LW1D.U[t,i,0] - LW2D_.U[t,i,0]) for i in range(M)])/M)

erreur_LW = np.array([erreur_LW, erreur_LW]).transpose()
erreur_trace(erreur_LW, M_li = np.array([100, 150, 200, 250]), xc = (0, 300))