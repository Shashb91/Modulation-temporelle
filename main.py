"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff 1D/2D
============================================================
"""
from tracer import *
from erreur import *

data1 = Donnee1D(M = 250,N = 2, label = "Lax Wendroff")                                #Lax Wendroff
data2 = Donnee1D(M = 250,N = 2, label = "Analytique")                                  #ADER4
data3 = Donnee1D(M = 250, label = "ADER4")                                       #Solution analytique

#U_LW = LaxWendroff1D(data1)
#anim1D(data1)
#
#u = analytique1D(data2)
#anim1D(data2)
#
#U_ADER4 = ADER41D(data3)
#anim1D(data3)
#
#anim1D_comparaison(data1, data2)                                                 #Comparaison LaxWendroff VS Analytique
#anim1D_comparaison(data3, data2)                                                 #Comparaison ADER4 VS Analytique

#=============================================================
#Etude de l'erreur t = 0.05s !
#=============================================================

eps = erreur1D(0.05, f = "LW")
erreur1D_trace(eps, f = "LW")

eps = erreur1D(0.05, f = "ADER4")
erreur1D_trace(eps, f = "ADER4")