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
from modulation import *
from schema_mt import LaxWendroff2D_mt
from sauvegarde import *
from tracer import *

alpha = 0.25
rho = (1000, 1000)
c = (1000, 1000)                                                                                                # c[0] < c[1]
kappa = rho[1] * c[1] ** 2
M = 200
f_mt = 30
modulation = "echelon"
param = 0.5
eps = 0.3                            # |eps| << 1, eps = 0 -> pas de modulation

if modulation == "echelon":
    rho_ = rho_echelon
    kappa_ = kappa_echelon
elif modulation == "sinus":
    rho_ = rho_sinus
    kappa_ = kappa_sinus
elif modulation == "triangle":
    rho_ = rho_triangle
    kappa_ = kappa_triangle

if eps == 0: modulation = "0"

LW2D_ = Donnee2D(label ="LW_aniso", kappa = kappa, rho = rho, alpha=0.25, Mx = M, My = M, S = pt_source_2D,
                 xc = (0, 200), yc = (0, 200), tc = (0, 0.2), f = 10, CFL = 0.6, c = 1000)
# LaxWendroff_aniso(LW2D_)
# LW2D_ = charger('.save/LW_aniso_07-02_14-28-45.pkl')
# sauvegarder(LW2D_, opt = True)
# anim2D(LW2D_)

# LW2D = Donnee2D(label ="LW", e = rho[1]*c[1]**2, rho = rho[1], c = c[1], alpha=0, Mx = M, My = M, S = pt_source_2D,
#                  xc = (0, 200), yc = (0, 200), tc = (0, 0.2), f = 10, CFL = 0.6)
# LaxWendroff2D(LW2D)
# print(LW2D_ - LW2D)
# LW2D_x_, LW2D_y_ = LW2D_.projection((M//2,0)), LW2D_.projection((0, M//2))
# LW2D_x, LW2D_y = LW2D.projection((M//2,0)), LW2D.projection((0, M//2))
# anim1D_comparaison(LW2D_x_, LW2D_y_)
# anim1D_comparaison(LW2D_y_, LW2D_y)
# anim1D_comparaison(LW2D_x_, LW2D_x)


ADER42D_ = Donnee2D(label ="ADER2D_aniso", kappa= kappa, rho = rho, alpha=0.25, Mx = M, My = M, S = pt_source_2D,
                    xc = (0, 200), yc = (0, 200), tc = (0, 0.2), f = 10, CFL = 0.6, c = 549)
# ADER4_aniso(ADER42D_)
# ADER42D_ = charger('.save/ADER2D_aniso_07-02_14-54-38.pkl')
# sauvegarder(ADER42D_, opt = True)
# anim2D(ADER42D_)

# ADER42D_x_, ADER42D_y_ = ADER42D_.projection((M//2,0)), ADER42D_.projection((0, M//2))
# anim1D_comparaison(ADER42D_x_, ADER42D_y_)
# anim1D_comparaison(ADER42D_x_, LW2D_x_)
# anim1D_comparaison(ADER42D_y_, LW2D_y_)

"""
Anisotropie + modulation temporelle
"""

LW2D_aniso_mt = Donnee2D(label ="LW_aniso_mt="+modulation, kappa = kappa, rho = rho, alpha = alpha, Mx = M, My = M, S = pt_source_2D,
                 xc = (0, 800), yc = (0, 800), tc = (0, 0.8), f = 10, CFL = 0.6, c = 1500,
                 rho_mt = (rho_, rho_), kappa_mt = kappa_, eps_r = (eps,0), eps_kappa = -eps, omega = 2*np.pi*f_mt, param = param)
LaxWendroff_aniso_mt(LW2D_aniso_mt)
# LW2D_aniso_mt = charger('.save/LW_aniso_mt=echelon_07-06_15-12-28.pkl')
sauvegarder(LW2D_aniso_mt, opt = True)
anim2D(LW2D_aniso_mt)
tracer_energie(LW2D_aniso_mt)

# LW2D_mt = Donnee2D(label ="LW_mt="+modulation, kappa= rho[1] * c[1] ** 2, rho = rho[0], alpha=0.25, Mx = M, My = M, S = pt_source_2D,
#                  xc = (0, 400), yc = (0, 400), tc = (0, 0.8), f = 10, CFL = 0.6, c = 1500,
#                  rho_mt = rho_, kappa_mt = kappa_, eps_r = 0, eps_kappa = -eps, omega = 2*np.pi*f_mt, param = param)
# LaxWendroff2D_mt(LW2D_mt)
# LW2D_mt = charger('.save/LW_mt=echelon_07-06_15-14-19.pkl')
# print(LW2D_mt)
# sauvegarder(LW2D_mt, opt = True)
# anim2D(LW2D_mt)

LW_aniso_mt_x, LW_aniso_mt_y = LW2D_aniso_mt.projection((M//2,0)), LW2D_aniso_mt.projection((0,M//2))
# LW_mt_x, LW_mt_y = LW2D_mt.projection((M//2,0)), LW2D_mt.projection((0,M//2))

anim1D(LW_aniso_mt_x)
anim1D(LW_aniso_mt_y)
tracer_energie(LW_aniso_mt_x)
tracer_energie(LW_aniso_mt_y)
# anim1D_comparaison(LW_mt_x, LW_aniso_mt_x)
# anim1D_comparaison(LW_mt_y, LW_aniso_mt_y)