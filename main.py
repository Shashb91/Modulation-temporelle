"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff 1D/2D
============================================================
"""
from tracer import *
from erreur import *
from analytique import *

data1 = Donnee1D(M = 250, label = "Lax Wendroff")                                         #Lax Wendroff
data2 = Donnee1D(M = 250, label = "Analytique")                                  #ADER4
data3 = Donnee1D(M = 250, label = "ADER4")                                       #Solution analytique

U_LW = LaxWendroff1D(data1)
#anim1D(data1)

u = analytique1D(data2)
#anim1D(data2)

U_ADER4 = ADER41D(data3)
anim1D(data3)

anim1D_comparaison(data1, data2)              #Comparaison LaxWendroff VS Analytique
anim1D_comparaison(data3, data2)              #Comparaison ADER4 VS Analytique

#=============================================================
#Etude de l'évolution de l'erreur
#=============================================================

#eps = erreur1D(0.1)
#erreur1D_trace(eps)