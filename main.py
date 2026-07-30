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
tf = 3500 #durée en ms totale de l'animation


LW1D = Donnee1D(M = 150, CFL = 0.6, xc = (0, 300), tc = (0, 0.15), c = 1500, kappa= 2.25e9, rho = 1000, f = 20, label ="Lax Wendroff")
Analy1D = Donnee1D(M = 150, CFL = 0.6, xc = (0, 300), tc = (0, 0.15), c = 1500, kappa= 2.25e9, rho = 1000, f = 20, label ="Analytique")
ADER1D = Donnee1D(M = 150, CFL = 0.6, xc = (0, 300), tc = (0, 0.15), c = 1500, kappa= 2.25e9, rho = 1000, f = 20, label ="ADER4")
LaxWendroff1D(LW1D)
# anim1D(LW1D)

analytique1D(Analy1D)
# anim1D(Analy1D)

ADER41D(ADER1D)
# anim1D(ADER1D)

# tracer1D_comparaison(0.05, Analy1D,LW1D)
# tracer1D_comparaison(0.125, Analy1D,LW1D)
# tracer1D_comparaison(0.05, Analy1D,ADER1D)
# tracer1D_comparaison(0.125, Analy1D,ADER1D)


anim1D_comparaison(LW1D, Analy1D, interval = tf/Analy1D.N)                                                      #Comparaison LaxWendroff VS Analytique
anim1D_comparaison(ADER1D, Analy1D, interval = tf/Analy1D.N)                                                    #Comparaison ADER4 VS Analytique


Analy1D = Donnee1D(M = 150, c = 1500, rho = 1000, kappa= 2.25e9, f=20, xc = (0, 300), tc = (0, 0.15), label ="Analytique", CFL = 0.6)
analytique1D_cauchy(Analy1D)
# anim1D(Analy1D)

LaxWendroff1D_cauchy(LW1D)
# anim1D(LW1D, interval = 30)

ADER41D_cauchy(ADER1D)
# anim1D(ADER1D, interval = 30)

# tracer1D_comparaison(0.0, Analy1D,LW1D)
# tracer1D_comparaison(0.175, Analy1D,LW1D)
# tracer1D_comparaison(0.0, Analy1D,ADER1D)
# tracer1D_comparaison(0.175, Analy1D,ADER1D)

anim1D_comparaison(LW1D, Analy1D, interval = tf/Analy1D.N)                                                      #Comparaison LaxWendroff VS Analytique
anim1D_comparaison(ADER1D, Analy1D, interval = tf/Analy1D.N)                                                      #Comparaison ADER4 VS Analytique

"""
============================================================
Etude de l'erreur à t = 0.05s ! (t < xf/(2*c))
============================================================
"""

# epsLW = erreur1D(0.05, f = "LW")
# erreur_trace(epsLW, f = "LW")

# epsADER2 = erreur1D(0.05, f = "ADER2")
# erreur_trace(epsADER2, f = "ADER2")

# epsADER4 = erreur1D(0.05, M_li = [150,200,250], f = "ADER4")
# erreur_trace(epsADER4,M_li = np.array([150,200,250]), f = "ADER4")

"""
============================================================
Implémentation de la résolution 2D homogène
============================================================
"""
data1 = Donnee2D(label="Lax Wendroff 2D", S = pt_source_2D, Mx = 150, My = 150, CFL = 0.6, opt = False, f = 10, tc = (0, 0.15), xc = (0, 300), yc = (0, 300), c = 1500, rho = 1000, kappa= 2.25e9)
# print(data1)
LaxWendroff2D(data1)
# data1 = charger('.save_2D_nmt/Lax Wendroff 2D_07-16_15-20-09.pkl')
# sauvegarder(data1, '.save_2D_nmt/')
anim2D(data1, interval=tf/data1.N)
# tracer2D(data1, 0.08)
# tracer2D(data1, 0.15)


data2 = Donnee2D(label="ADER4 2D", S = pt_source_2D, Mx = 150, My = 150, CFL = 0.95, opt = False, f = 10, tc = (0, 0.15), xc = (0, 300), yc = (0, 300), c = 1500, rho = 1000, kappa= 2.25e9)
# print(data2)
ADER42D(data2)
# data2 = charger('.save_2D_nmt/Lax Wendroff 2D_06-23_16-30-26.pkl')
# sauvegarder(data2, '.save_2D_nmt/')
anim2D(data2, interval=tf/data2.N)
# tracer2D(data2,0.08)
# tracer2D(data2,0.15)

#Cas de l'onde plane avec un problème de Cauchy
M = 150
data4 = Donnee2D(label = "Lax Wendroff 2D", opt = True, S=pt_source_1D, xc = (0,300), yc = (0,300), c = 1500, rho = 1000, tc = (0, 0.15), CFL = 0.6, Mx = M, My = M, f = 20, kappa= 2.25e9)
data5 = Donnee2D(label = "ADER4 2D", opt = True, S=pt_source_1D, xc = (0,300), yc = (0,300), c = 1500, rho = 1000, tc = (0, 0.15), CFL = 0.6, Mx = M, My = M, f = 20, kappa= 2.25e9)

Analy1D = Donnee1D(M = M, label ="Analytique", xs = 0, f = 20, c = 1500, rho = 1000, kappa= 2.25e9, CFL = 0.6, tc = (0, 0.15), xc = (0, 300))
analytique1D_cauchy(Analy1D)
# anim1D(data2)

# LaxWendroff2D_cauchy(data4)
# sauvegarder(data4, ".save_2D_nmt/")
data4 = charger('.save_2D_nmt/Lax Wendroff 2D_07-17_13-25-29.pkl')
# anim2D(data4)
# tracer2D(data4,0)
# tracer2D(data4,0.175)

data4_ = data4.projection((data4.Mx//2,0))
# anim1D(data4_, interval = 30)
# anim1D_comparaison(data4_, data2, interval = 30)
# anim1D_comparaison(data4_, data1, interval = 30)
anim1D_cauchy_comparaison(Analy1D, LW1D, data4_, interval = tf/Analy1D.N)
anim1D_cauchy_comparaison(Analy1D, LW1D, data4_, interval = tf/Analy1D.N)

# ADER42D_cauchy(data5)
# sauvegarder(data5)
data5 = charger('.save/ADER4 2D_07-17_13-58-43.pkl')
# anim2D(data5)
# tracer2D(data5,0)
# tracer2D(data5,0.175)

data5_ = data5.projection((data5.Mx//2,0))
# anim1D(data5_, interval = 30)
# anim1D_comparaison(data5_,data2, interval=30)
# anim1D_comparaison(data5_,data3, interval=30)
anim1D_cauchy_comparaison(Analy1D, LW1D, data5_, interval = tf/Analy1D.N)
anim1D_cauchy_comparaison(Analy1D, LW1D, data5_, interval = tf/Analy1D.N)