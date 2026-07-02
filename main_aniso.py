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
from schema import LaxWendroff2D
from sauvegarde import *
from tracer import *

rho = (100, 1000)
c = (500, 1000)                                                                                                # c[0] < c[1]
e = (rho[0]*c[0]**2, rho[1]*c[1]**2)
M = 150

LW2D_ = Donnee2D(label ="LW_aniso", e = e, rho = rho, alpha=0, Mx = M, My = M, S = pt_source_2D,
                 xc = (0, 200), yc = (0, 200), tc = (0, 0.2), f = 10, CFL = 0.6)
LaxWendroff_aniso(LW2D_)
# LW2D_ = charger('.save/LW_aniso_07-02_14-28-45.pkl')
# sauvegarder(LW2D_, opt = True)
anim2D(LW2D_)

LW2D = Donnee2D(label ="LW_aniso", e = rho[1]*c[1]**2, rho = rho[1], c = c[1], alpha=0, Mx = M, My = M, S = pt_source_2D,
                 xc = (0, 200), yc = (0, 200), tc = (0, 0.2), f = 10, CFL = 0.6)
LaxWendroff2D(LW2D)
print(LW2D - LW2D_)
LW2D_x_, LW2D_y_ = LW2D_.projection((M//2,0)), LW2D_.projection((0, M//2))
anim1D_comparaison(LW2D_x_, LW2D_y_)

ADER42D_ = Donnee2D(label ="ADER2D_aniso", e = e, rho = rho, alpha=0.25, Mx = M, My = M, S = pt_source_2D,
                    xc = (0, 200), yc = (0, 200), tc = (0, 0.2), f = 10, CFL = 0.6, c = 549)
ADER4_aniso(ADER42D_)
# ADER42D_ = charger('.save/ADER2D_aniso_07-02_14-54-38.pkl')
sauvegarder(ADER42D_, opt = True)
anim2D(ADER42D_)

ADER42D_x_, ADER42D_y_ = ADER42D_.projection((M//2,0)), ADER42D_.projection((0, M//2))
anim1D_comparaison(ADER42D_x_, ADER42D_y_)

anim1D_comparaison(ADER42D_x_, LW2D_x_)
anim1D_comparaison(ADER42D_y_, LW2D_y_)