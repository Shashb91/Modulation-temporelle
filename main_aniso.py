"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff et d'ADER4 2D en mileu
anisotrope

Les modules concernant la modulation anisotrope sont indicés
par _aniso

Parametres :
- CFL = 0.6
- e = 9.4e9
============================================================
"""
from source import *
from schema_aniso import*
from tracer import *

data = Donnee2D(label = "LW2D_aniso", e = (25e6, 1e9), rho = (100, 1000), alpha=0.4, Mx = 150, My = 150, S = pt_source_1D,
                xc = (0, 200), yc = (0, 200), tc = (0, 0.2), f = 20, CFL = 0.6, c = 1000)
LaxWendroff_aniso(data)
print(data)
anim2D_aniso(data)