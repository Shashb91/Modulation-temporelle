"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff et d'ADER4 1D/2D
============================================================
"""
from tracer import *
from erreur import *
from donnee import *
from sauvegarde import *

#============================================================
#Implémentation de la résolution 1D homogène
#============================================================

#data1 = Donnee1D(M = 200, label = "Lax Wendroff")                                #Lax Wendroff
#data2 = Donnee1D(M = 200, label = "Analytique")                                  #ADER4
#data3 = Donnee1D(M = 200, label = "ADER4")                                             #Solution analytique

#U_LW = LaxWendroff1D(data1)
#anim1D(data1)

#u = analytique1D(data2)
#anim1D(data2)

#U_ADER4 = ADER41D(data3)
#anim1D(data3)

#data1 = charger('.save/data1_06-08_15-20-38.pkl')
#data2 = charger('.save/data2_06-08_15-20-38.pkl')
#anim1D_comparaison(data1, data2)                                                       #Comparaison LaxWendroff VS Analytique
#anim1D_comparaison(data3, data2)                                                      #Comparaison ADER4 VS Analytique

#sauvegarder(data1,"data1")
#sauvegarder(data2,"data2")

#============================================================
#Etude de l'erreur à t = 0.05s ! (t < xf/(2*c))
#============================================================

#epsLW = erreur1D(0.05, f = "LW")
#erreur1D_trace(epsLW, f = "LW")

#epsADER2 = erreur1D(0.05, f = "ADER2")
#erreur1D_trace(epsADER2, f = "ADER2")

#epsADER4 = erreur1D(0.05, f = "ADER4")
#erreur1D_trace(epsADER4, f = "ADER4")


#============================================================
#Implémentation de la résolution 2D homogène
#============================================================

data4 = Donnee2D(label = "Lax Wendroff 2D")

U = LaxWendroff2D(data4)
sauvegarder(data4, "data4_")

#data4 = charger('.save/data4_06-08_16-39-58.pkl')
anim2D(data4)

tracer2D(data4,250,150)