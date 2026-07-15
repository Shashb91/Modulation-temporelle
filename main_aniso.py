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
from schema import LaxWendroff2D
from sauvegarde import *
from tracer import *

alpha = 0.5
rho = (1000, 1000)
c = (1000, 1000)                                                                                                # c[0] < c[1]
kappa = (rho[0] * c[0] ** 2, rho[1] * c[1] ** 2)
M = 400
f_mt = 1/0.15
modulation = "echelon"
param = 0.5
eps = 0.3                            # |eps| << 1, eps = 0 -> pas de modulation
kappa_eff = (alpha/kappa[0] + (1-alpha)/kappa[1])**(-1)

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
                 xc = (0, 400), yc = (0, 400), tc = (0, 0.4), f = 10, CFL = 0.6, c = 1000)
# LaxWendroff_aniso(LW2D_)
# sauvegarder(LW2D_, opt = True)
# LW2D_ = charger('.save/LW_aniso_07-12_09-44-16.pkl')
# anim2D(LW2D_, interval = 5, param = 0.15, tf = 0.3)

LW2D = Donnee2D(label ="LW", kappa = kappa_eff, rho = rho[0], c = 549.8368396346259, alpha=0, Mx = M, My = M, S = pt_source_2D,
                 xc = (0, 200), yc = (0, 200), tc = (0, 0.2), f = 10, CFL = 0.6)
# LaxWendroff2D(LW2D)
# print(LW2D_ - LW2D)
# LW2D_x_, LW2D_y_ = LW2D_.projection((M//2,0)), LW2D_.projection((0, M//2))
# LW2D_x, LW2D_y = LW2D.projection((M//2,0)), LW2D.projection((0, M//2))
# anim1D_comparaison(LW2D_x_, LW2D_y_)
# anim1D_comparaison(LW2D_y_, LW2D_y)
# anim1D_comparaison(LW2D_x_, LW2D_x)


ADER42D_ = Donnee2D(label ="ADER2D_aniso", kappa= kappa_eff, rho = rho, alpha=0.25, Mx = M, My = M, S = pt_source_2D,
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

LW2D_aniso_mt = Donnee2D(label ="LW_aniso_mt=", kappa = kappa_eff, rho = rho, alpha = alpha, Mx = M, My = M, S = pt_source_2D,
                 xc = (0, 400), yc = (0, 400), tc = (0, 0.4), f = 10, CFL = 0.6, c = 1000,
                 rho_mt = (rho_, rho_), kappa_mt = kappa_, eps_r = (0,eps), eps_kappa = 0, omega = 2*np.pi*f_mt, param = param)
# LaxWendroff_aniso_mt(LW2D_aniso_mt)
# sauvegarder(LW2D_aniso_mt, opt = True)
LW2D_aniso_mt = charger('.save/LW_aniso_mt=_07-12_16-20-15.pkl')
anim2D(LW2D_aniso_mt, interval = 0.5, param = 0.15, tf = 0.3)
# tracer_energie(LW2D_aniso_mt)

# LW2D_mt = Donnee2D(label ="LW_mt="+modulation, kappa= rho[1] * c[1] ** 2, rho = rho[0], alpha=0.25, Mx = M, My = M, S = pt_source_2D,
#                  xc = (0, 400), yc = (0, 400), tc = (0, 0.8), f = 10, CFL = 0.6, c = 1500,
#                  rho_mt = rho_, kappa_mt = kappa_, eps_r = 0, eps_kappa = -eps, omega = 2*np.pi*f_mt, param = param)
# LaxWendroff2D_mt(LW2D_mt)
# LW2D_mt = charger('.save/LW_mt=echelon_07-06_15-14-19.pkl')
# print(LW2D_mt)
# sauvegarder(LW2D_mt, opt = True)
# anim2D(LW2D_mt)

# LW_aniso_mt_x, LW_aniso_mt_y = LW2D_aniso_mt.projection((M//2,0)), LW2D_aniso_mt.projection((0,M//2))
# LW_mt_x, LW_mt_y = LW2D_mt.projection((M//2,0)), LW2D_mt.projection((0,M//2))

# anim1D(LW_aniso_mt_x)
# anim1D(LW_aniso_mt_y)
# anim1D_comparaison(LW_mt_x, LW_aniso_mt_x)
# anim1D_comparaison(LW_mt_y, LW_aniso_mt_y)


ADER42D_aniso_mt = Donnee2D(label ="ADER4_aniso_mt=", kappa = kappa, rho = rho, alpha = alpha, Mx = M, My = M, S = pt_source_2D,
                            xc = (0, 400), yc = (0, 400), tc = (0, 0.15), f = 10, CFL = 0.6, c = 1000,
                            rho_mt = (rho_, rho_), kappa_mt = kappa_, eps_r = (eps,0), eps_kappa = -eps, omega = 2*np.pi*f_mt, param = param)
# ADER4_aniso_mt(ADER42D_aniso_mt)
# ADER42D_aniso_mt = charger('.save/ADER4_aniso_mt=echelon_07-07_15-52-09.pkl')
# sauvegarder(ADER42D_aniso_mt, opt = True)
# anim2D(ADER42D_aniso_mt)
# tracer_energie(ADER42D_aniso_mt)
# print(ADER42D_aniso_mt-LW2D_aniso_mt)

# ADER42D_aniso_mt_x, ADER42D_aniso_mt_y = ADER42D_aniso_mt.projection((M//2,0)), ADER42D_aniso_mt.projection((0,M//2))
# ADER4_mt_x, ADER4_mt_y = ADER4_mt.projection((M//2,0)), ADER4_mt.projection((0,M//2))

# anim1D(ADER42D_aniso_mt_x)
# anim1D(ADER42D_aniso_mt_y)
# anim1D_comparaison(ADER4_mt_x, ADER42D_aniso_mt_x)
# anim1D_comparaison(ADER4_mt_y, ADER42D_aniso_mt_y)

"""
Micro Structuré 
"""

LW2D_ms = Donnee2D(label ="LW_ms", kappa = kappa, rho = rho, Mx = M, My = M, S = pt_source_2D,
                   xc = (0, 400), yc = (0, 400), tc = (0, 0.4), f = 10, CFL = 0.6, c = 1000)
L,l = 15 * LW2D_ms.dy, 5 * LW2D_ms.dy
# LaxWendroff_ms(LW2D_ms, l, L)
# sauvegarder(LW2D_ms, opt = True)
# LW2D_ms = charger('.save/LW_ms_07-11_21-18-51.pkl')
# anim2D_ms(LW2D_ms, l, L, interval = 0.5, param = 0.15, tf = 0.3)

# data_x, data_y = LW2D_ms.projection((M//2,0)), LW2D_ms.projection((0, M//2))
# anim1D(data_x)
# anim1D(data_y)

ADER42D_ms = Donnee2D(label ="ADER4_ms", kappa = kappa, rho = rho, Mx = M, My = M, S = pt_source_2D,
                   xc = (0, 200), yc = (0, 200), tc = (0, 0.2), f = 10, CFL = 0.95, c = 1000)
L, l = 15 * ADER42D_ms.dy, 5 * ADER42D_ms.dy
# ADER4_ms(ADER42D_ms, l, L)
# sauvegarder(ADER42D_ms, opt = True)
# ADER42D_ms = charger('.save/ADER4_ms_07-09_11-38-23.pkl')
# anim2D_ms(ADER42D_ms, l, L, interval = 0.005, param=0.15)

# data_x, data_y = ADER42D_ms.projection((M//2,0)), ADER42D_ms.projection((0, M//2))
# anim1D(data_x)
# anim1D(data_y)

"""
Micro Structuré + modulation temporelle
"""

LW2D_ms_mt = Donnee2D(label ="LW_ms_mt", kappa = kappa, rho = rho, Mx = M, My = M, S = pt_source_2D,
                   xc = (0, 400), yc = (0, 400), tc = (0, 0.4), f = 10, CFL = 0.6, c = 1000,
                   rho_mt = (rho_, rho_), kappa_mt = (kappa_, kappa_), eps_r = (eps,-eps), eps_kappa = (0, 0), omega = 2*np.pi*f_mt, param = param)
L,l = 15 * LW2D_ms_mt.dy, 15 * LW2D_ms_mt.dy
# LaxWendroff_ms_mt(LW2D_ms_mt, l, L)
# sauvegarder(LW2D_ms_mt, opt = True)
LW2D_ms_mt = charger('.save/LW_ms_mt_07-12_18-09-35.pkl')
anim2D_ms(LW2D_ms_mt, l, L, interval = 0.5, param = 0.15, tf = 0.3)

# tracer_energie(LW2D_ms_mt)
# data_x, data_y = LW2D_ms_mt.projection((M//2,0)), LW2D_ms_mt.projection((0, M//2))
# anim1D(data_x)
# anim1D(data_y)

ADER42D_ms_mt = Donnee2D(label ="ADER4_ms_mt", kappa = kappa, rho = rho, Mx = M, My = M, S = pt_source_2D,
                   xc = (0, 400), yc = (0, 400), tc = (0, 0.2), f = 10, CFL = 0.95, c = 1000,
                   rho_mt = (rho_, rho_), kappa_mt = (kappa_, kappa_), eps_r = (eps,0), eps_kappa = (-eps, 0), omega = 2*np.pi*f_mt, param = param)
L,l = 15 * ADER42D_ms_mt.dy, 5 * ADER42D_ms_mt.dy
# ADER4_ms_mt(ADER42D_ms_mt, l, L)
# sauvegarder(ADER42D_ms_mt, opt = True)
# ADER42D_ms_mt = charger('.save/LW_ms_mt_07-10_10-31-42.pkl')
# anim2D_ms(ADER42D_ms_mt, l, L, interval = 5, param=0.15, tf = 0.2)

# tracer_energie(ADER42D_ms_mt)
# data_x, data_y = ADER42D_ms_mt.projection((M//2,0)), ADER42D_ms_mt.projection((0, M//2))
# anim1D(data_x)
# anim1D(data_y)