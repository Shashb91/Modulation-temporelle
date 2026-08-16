"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date   : 01/06/2026

Résolution EXHAUSTIVE en milieu homogène isotrope MODULÉ EN
TEMPS (modules suffixés par _mt).

Pour chaque configuration on effectue les résolutions sous
Lax-Wendroff et ADER4 (il n'existe pas de solution analytique
en milieu modulé), puis on trace SOIT la comparaison des deux
schémas, SOIT la solution elle-même.

Un unique cas de modulation est utilisé pour toutes les
résolutions, afin que Lax-Wendroff et ADER4 résolvent
exactement le même problème.

Paramètres physiques : rho = 1000, kappa = 2.25e9 (c = 1500).
Paramètres numériques : CFL = 0.6.
============================================================
"""
from donnee import *
from schema_mt import *
from schema import LaxWendroff2D, ADER42D
from source import pt_source_1D, pt_source_2D
from tracer import *


c, rho, kappa = 1500, 1000, 2.25e9
CFL = 0.6
duree_anim = 3                              # durée (s) de TOUTES les animations éventuelles

f_mt = 30
param = 0.5
eps = 0.3                                   # |eps| << 1
modulation = "echelon"
rho_mod, kappa_mod = rho_echelon, kappa_echelon
omega = 2 * np.pi * f_mt
mod = dict(eps_r=-eps, eps_kappa=eps, omega=omega, rho_mt=rho_mod, kappa_mt=kappa_mod, param=param)

# 1D — source ponctuelle, modulée en temps
M1, xc1, tc1, f1 = 300, (0, 600), (0, 0.2), 20
LaxWendroff_1D_mt = Donnee1D(label="Lax-Wendroff", c=c, rho=rho, kappa=kappa, f=f1, M=M1, CFL=CFL, xc=xc1, tc=tc1, **mod)
ADER4_1D_mt = Donnee1D(label="ADER4", c=c, rho=rho, kappa=kappa, f=f1, M=M1, CFL=CFL, xc=xc1, tc=tc1, **mod)

LaxWendroff1D_mt(LaxWendroff_1D_mt)
ADER41D_mt(ADER4_1D_mt)

tracer_mt(LaxWendroff_1D_mt)                              # profils rho(t), kappa(t), c(t)
anim1D_mt_comparaison(LaxWendroff_1D_mt, ADER4_1D_mt, duree=duree_anim)

# 1D — problème de Cauchy / onde plane, modulée en temps
M2, xc2, tc2, f2 = 200, (0, 500), (0, 0.2), 10
LaxWendroff_1D_cauchy_mt = Donnee1D(label="Lax-Wendroff", c=c, rho=rho, kappa=kappa, f=f2, M=M2, CFL=CFL,
                                    xc=xc2, tc=tc2, S=pt_source_1D, **mod)
ADER4_1D_cauchy_mt = Donnee1D(label="ADER4", c=c, rho=rho, kappa=kappa, f=f2, M=M2, CFL=CFL,
                                    xc=xc2, tc=tc2, S=pt_source_1D, **mod)

LaxWendroff1D_cauchy_mt(LaxWendroff_1D_cauchy_mt)
ADER41D_cauchy_mt(ADER4_1D_cauchy_mt)

anim1D_mt_comparaison(LaxWendroff_1D_cauchy_mt, ADER4_1D_cauchy_mt, duree=duree_anim)

# 2D — source ponctuelle, modulée en temps
Mxy, xc3, tc3, f3 = 300, (0, 400), (0, 0.2), 10
LaxWendroff_2D_mt = Donnee2D(label="Lax-Wendroff 2D", c=c, rho=rho, kappa=kappa, f=f3, Mx=Mxy, My=Mxy, CFL=CFL,
                             xc=xc3, yc=xc3, tc=tc3, **mod)
ADER4_2D_mt = Donnee2D(label="ADER4 2D", c=c, rho=rho, kappa=kappa, f=f3, Mx=Mxy, My=Mxy, CFL=CFL,
                       xc=xc3, yc=xc3, tc=tc3, **mod)

LaxWendroff2D_mt(LaxWendroff_2D_mt)
ADER42D_mt(ADER4_2D_mt)

anim2D(LaxWendroff_2D_mt, duree=duree_anim)
anim2D(ADER4_2D_mt, duree=duree_anim)

# 2D — problème de Cauchy / onde plane, modulée en temps
LaxWendroff_2D_cauchy_mt = Donnee2D(label="Lax-Wendroff 2D", c=c, rho=rho, kappa=kappa, f=f3, Mx=Mxy, My=Mxy, CFL=CFL,
                                    xc=xc3, yc=xc3, tc=tc3, S=pt_source_1D, **mod)
ADER4_2D_cauchy_mt = Donnee2D(label="ADER4 2D", c=c, rho=rho, kappa=kappa, f=f3, Mx=Mxy, My=Mxy, CFL=CFL,
                              xc=xc3, yc=xc3, tc=tc3, S=pt_source_1D, **mod)

LaxWendroff2D_cauchy_mt(LaxWendroff_2D_cauchy_mt)
ADER42D_cauchy_mt(ADER4_2D_cauchy_mt)

anim2D(LaxWendroff_2D_cauchy_mt, duree=duree_anim)
anim2D(ADER4_2D_cauchy_mt, duree=duree_anim)
