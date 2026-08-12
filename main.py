"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date   : 01/06/2026

Résolution EXHAUSTIVE en milieu homogène isotrope, à paramètres
constants en temps.

Pour chaque configuration on effectue les résolutions sous
Lax-Wendroff, ADER4 et (si elle existe) la solution analytique,
puis on trace SOIT la comparaison des schémas à la solution de
référence, SOIT la solution elle-même.

Paramètres physiques : rho = 1000 kg.m-3, kappa = 2.25e9 Pa,
soit c = sqrt(kappa/rho) = 1500 m.s-1.
Paramètres numériques : CFL = 0.6, M = 150.
============================================================
"""
from analytique import *
from schema import *
from tracer import *

# ---------- Paramètres physiques et numériques communs ----------
c, rho, kappa = 1500, 1000, 2.25e9          # c = sqrt(kappa/rho) = 1500 m/s
f = 20
CFL = 0.6
M = 300
xc = (0, 300)
tc = (0, 0.15)
tf = 3500                                   # durée (ms) des animations éventuelles

# ============================================================
# 1D — source ponctuelle   (solution analytique disponible)
#   -> tracé de la COMPARAISON schéma / analytique
# ============================================================
LaxWendroff_1D = Donnee1D(label="Lax-Wendroff", c=c, rho=rho, kappa=kappa, f=f, M=M, CFL=CFL, xc=xc, tc=tc)
ADER4_1D       = Donnee1D(label="ADER4",         c=c, rho=rho, kappa=kappa, f=f, M=M, CFL=CFL, xc=xc, tc=tc)
Analytique_1D  = Donnee1D(label="Analytique",    c=c, rho=rho, kappa=kappa, f=f, M=M, CFL=CFL, xc=xc, tc=tc)

LaxWendroff1D(LaxWendroff_1D)
ADER41D(ADER4_1D)
analytique1D(Analytique_1D)

tracer1D_comparaison(0.05, LaxWendroff_1D, Analytique_1D)
tracer1D_comparaison(0.05, ADER4_1D, Analytique_1D)
# anim1D_comparaison(LaxWendroff_1D, Analytique_1D, interval=tf / Analytique_1D.N)
# anim1D_comparaison(ADER4_1D, Analytique_1D, interval=tf / Analytique_1D.N)

# ============================================================
# 1D — problème de Cauchy / onde plane   (analytique disponible)
#   -> tracé de la COMPARAISON schéma / analytique
# ============================================================
LaxWendroff_1D_cauchy = Donnee1D(label="Lax-Wendroff", c=c, rho=rho, kappa=kappa, f=f, M=M, CFL=CFL, xc=xc, tc=tc)
ADER4_1D_cauchy       = Donnee1D(label="ADER4",         c=c, rho=rho, kappa=kappa, f=f, M=M, CFL=CFL, xc=xc, tc=tc)
Analytique_1D_cauchy  = Donnee1D(label="Analytique",    c=c, rho=rho, kappa=kappa, f=f, M=M, CFL=CFL, xc=xc, tc=tc)

LaxWendroff1D_cauchy(LaxWendroff_1D_cauchy)
ADER41D_cauchy(ADER4_1D_cauchy)
analytique1D_cauchy(Analytique_1D_cauchy)

tracer1D_comparaison(0.075, LaxWendroff_1D_cauchy, Analytique_1D_cauchy)
tracer1D_comparaison(0.075, ADER4_1D_cauchy, Analytique_1D_cauchy)

# ============================================================
# 2D — source ponctuelle   (pas de solution analytique)
#   -> tracé de la SOLUTION elle-même
# ============================================================
LaxWendroff_2D = Donnee2D(label="Lax-Wendroff 2D", c=c, rho=rho, kappa=kappa, f=10, Mx=M, My=M, CFL=CFL,
                          xc=xc, yc=xc, tc=tc, opt=False, S=pt_source_2D)
ADER4_2D       = Donnee2D(label="ADER4 2D",         c=c, rho=rho, kappa=kappa, f=10, Mx=M, My=M, CFL=CFL,
                          xc=xc, yc=xc, tc=tc, opt=False, S=pt_source_2D)

LaxWendroff2D(LaxWendroff_2D)
ADER42D(ADER4_2D)

tracer2D(LaxWendroff_2D, 0.08)
tracer2D(ADER4_2D, 0.08)
# anim2D(LaxWendroff_2D, interval=tf / LaxWendroff_2D.N)
# anim2D(ADER4_2D, interval=tf / ADER4_2D.N)

# ============================================================
# 2D — problème de Cauchy / onde plane
#   La coupe 1D de la solution 2D est comparée à l'analytique 1D
#   -> tracé de la COMPARAISON (analytique 1D / LW 1D / coupe 2D)
# ============================================================
LaxWendroff_2D_cauchy = Donnee2D(label="Lax-Wendroff 2D", c=c, rho=rho, kappa=kappa, f=f, Mx=M, My=M, CFL=CFL,
                                 xc=xc, yc=xc, tc=tc, opt=True, S=pt_source_1D)
ADER4_2D_cauchy       = Donnee2D(label="ADER4 2D",         c=c, rho=rho, kappa=kappa, f=f, Mx=M, My=M, CFL=CFL,
                                 xc=xc, yc=xc, tc=tc, opt=True, S=pt_source_1D)

LaxWendroff2D_cauchy(LaxWendroff_2D_cauchy)
ADER42D_cauchy(ADER4_2D_cauchy)

# références 1D (onde plane, source en x = 0)
Analytique_1D_ref   = Donnee1D(label="Analytique",    c=c, rho=rho, kappa=kappa, f=f, M=M, CFL=CFL, xc=xc, tc=tc, xs=0)
LaxWendroff_1D_ref  = Donnee1D(label="Lax-Wendroff 1D", c=c, rho=rho, kappa=kappa, f=f, M=M, CFL=CFL, xc=xc, tc=tc, xs=0)
analytique1D_cauchy(Analytique_1D_ref)
LaxWendroff1D_cauchy(LaxWendroff_1D_ref)

coupe_LW   = LaxWendroff_2D_cauchy.projection((M // 2, 0))
coupe_ADER = ADER4_2D_cauchy.projection((M // 2, 0))

tracer_cauchy_comparaison(Analytique_1D_ref, LaxWendroff_1D_ref, coupe_LW, 0.075)
tracer_cauchy_comparaison(Analytique_1D_ref, LaxWendroff_1D_ref, coupe_ADER, 0.075)
