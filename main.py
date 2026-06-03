"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff 1D/2D
============================================================
"""
from tracer import *
from erreur import *

data1 = Donnee1D()                       #Lax Wendroff
data2 = Donnee1D()                       #Solution analytique

#U = LaxWendroff1D(data1)
#anim1D(data1)

#u = analytique1D(data2)
#anim1D(data2)

#anim1D_analytiqueVSnumerique(data1, data2)

#=============================================================
#Etude de l'évolution de l'erreur
#=============================================================

eps = erreur1D(0.1)
erreur1D_trace(eps)