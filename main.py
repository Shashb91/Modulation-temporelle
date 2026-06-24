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

data1 = Donnee1D(M = 200, label = "Lax Wendroff")                                #Lax Wendroff
data2 = Donnee1D(M = 200, label = "Analytique")                                  #ADER4
data3 = Donnee1D(M = 200, label = "ADER4")                                       #Solution analytique

# LaxWendroff1D(data1)
# anim1D(data1)

analytique1D(data2)
anim1D(data2)

# ADER41D(data3)
# anim1D(data3)

# data1 = charger('.save/data1_06-08_15-20-38.pkl')
# data2 = charger('.save/data2_06-08_15-20-38.pkl')
# anim1D_comparaison(data1, data2, interval = 30)                                                      #Comparaison LaxWendroff VS Analytique
# anim1D_comparaison(data3, data2, interval = 30)                                                      #Comparaison ADER4 VS Analytique

# sauvegarder(data1,"data1_")
# sauvegarder(data2,"data2_")

# data2 = Donnee1D(M = 200, label = "Analytique")
# analytique1D_cauchy(data2)
# anim1D(data2)

# LaxWendroff1D_cauchy(data1)
# anim1D(data1, interval = 30)

# ADER41D_cauchy(data3)
# anim1D(data3, interval = 30)

# tracer1D_comparaison(10, data1, data2)
# anim1D_comparaison(data1, data2, interval = 30)                                                      #Comparaison LaxWendroff VS Analytique
# anim1D_comparaison(data3, data2, interval = 30)                                                      #Comparaison ADER4 VS Analytique

"""
============================================================
Etude de l'erreur à t = 0.05s ! (t < xf/(2*c))
============================================================
"""

# epsLW = erreur1D(0.05, f = "LW")
# erreur1D_trace(epsLW, f = "LW")

# epsADER2 = erreur1D(0.05, f = "ADER2")
# erreur1D_trace(epsADER2, f = "ADER2")

# epsADER4 = erreur1D(0.05, M_li = [150,200,250], f = "ADER4")
# erreur1D_trace(epsADER4,M_li = np.array([150,200,250]), f = "ADER4")

"""
============================================================
Implémentation de la résolution 2D homogène
============================================================
"""
# data1 = Donnee2D(label="Lax Wendroff 2D", S = pt_source_2D)
# LaxWendroff2D(data1)
# data1 = charger('.save_2D_nmt/Lax Wendroff 2D_06-23_16-30-26.pkl')
# sauvegarder(data1, '.save_2D_nmt/Lax Wendroff 2D_06-23_16-30-26.pkl')
# anim2D(data1)


#Cas de l'onde plane avec un problème de Cauchy
M = 150
data4 = Donnee2D(label = "Lax Wendroff 2D",opt = True, S=pt_source_1D, xc = (0,300), yc = (0,300), c = 1500, rho = 1000, tc = (0, 0.2),CFL = 0.6, Mx = M, My = M, f = 20, e= 2.25e9)
data5 = Donnee2D(label = "ADER4 2D",opt = True, S=pt_source_1D, xc = (0,300), yc = (0,300), c = 1500, rho = 1000, tc = (0, 0.2), CFL = 0.6, Mx = M, My = M, f = 20, e = 2.25e9)

data2 = Donnee1D(M = M, label = "Analytique", xs = 0, f = 20, rho = 1000, c = 1500, CFL = 0.6, tc = (0, 0.2), xc = (0,300))
analytique1D_cauchy(data2)

# LaxWendroff2D_cauchy(data4)
# sauvegarder(data4, ".save_2D_nmt/")
data4 = charger('.save_2D_nmt/Lax Wendroff 2D_cauchy.pkl')
anim2D(data4)

data4_ = Donnee1D(M = M, label = "LW 2D OP", xc = data4.xc, tc = data4.tc, x = data4.x, t = data4.t, c = data4.c, rho = data4.rho, CFL = 0.6)
data4_.U = data4.U[:, M//2, :,1:]
anim1D(data4_, interval = 30)
anim1D_comparaison(data4_, data2, interval = 30)

# ADER42D_cauchy(data5)
# sauvegarder(data5)
data5 = charger('.save_2D_nmt/ADER4 2D_cauchy.pkl')
anim2D(data5)

data5_ = Donnee1D(M = M, label = "ADER4 2D OP", xc = (0,300), tc = (0, 0.2), x = data4.x, t = data4.t, c = data4.c, rho = data4.rho, CFL = 0.6)
data5_.U = data5.U[:, M//2, :,1:]
anim1D(data5_, interval = 30)
anim1D_comparaison(data5_,data2, interval=30)