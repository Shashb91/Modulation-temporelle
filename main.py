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

data1 = Donnee1D()                       #Lax Wendroff
data2 = Donnee1D()                       #Solution analytique
data3 = Donnee1D(schema = "ADER4")                       #ADER4

#U_LW = LaxWendroff1D(data1)
#anim1D(data1)

#u = analytique1D(data2)
#anim1D(data2)

U_ADER4 = ADER41D(data3)
anim1D(data3)

#anim1D_analytiqueVSnumerique(data1, data2)

#=============================================================
#Etude de l'évolution de l'erreur
#=============================================================

#eps = erreur1D(0.1)
#erreur1D_trace(eps)