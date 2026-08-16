"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date   : 01/06/2026

Résolution EXHAUSTIVE en milieu stratifié (modules suffixés
par _aniso). Deux approches sont couvertes :
  - milieu HOMOGÉNÉISÉ anisotrope,
  - milieu MICRO-STRUCTURÉ (interfaces explicites, suffixe ms),
chacune en régime constant puis modulé en temps.

Pour chaque configuration on effectue les résolutions sous
Lax-Wendroff et ADER4 (pas de solution analytique en milieu
stratifié), puis on trace la solution.

Un unique cas de modulation (échelon) est utilisé pour toutes
les résolutions modulées.

Paramètres : rho = (100, 1000), c = (500, 1000),
kappa = rho*c^2, alpha = 0.5, CFL = 0.6, f = 10.
============================================================
"""
from donnee import *
from schema_aniso import *
from source import pt_source_2D
from tracer import *

# ---------- Paramètres physiques et numériques communs ----------
alpha = 0.5
rho = (100, 1000)
c_couches = (500, 1000)                                   # c[0] < c[1]
kappa = (rho[0] * c_couches[0] ** 2, rho[1] * c_couches[1] ** 2)
M = 200                                                   # (200 dans l'étude ; réduit pour la rapidité)
f = 10
CFL = 0.6
xc = (0, 400)
tc = (0, 0.2)
c0 = max(c_couches)                                                 # placeholder, recalculé par CFL_aniso
duree_anim = 3                                            # durée (s) de TOUTES les animations

# ---------- Cas de modulation UNIQUE (échelon, identique pour LW et ADER4) ----------
f_mt = 1 / 0.15
param = 0.5
eps = 0.3
rho_, kappa_ = rho_echelon, kappa_echelon
mod = dict(rho_mt=(rho_, rho_), kappa_mt=(kappa_, kappa_),
           eps_r=(eps, eps), eps_kappa=(0, 0), omega=2 * np.pi * f_mt, param=param)

# ============================================================
# 1) Milieu HOMOGÉNÉISÉ anisotrope (constant en temps)
#    -> tracé de la SOLUTION (Lax-Wendroff puis ADER4)
# ============================================================
LaxWendroff_aniso_ = Donnee2D(label="Lax-Wendroff aniso", kappa=kappa, rho=rho, alpha=alpha, Mx=M, My=M,
                              S=pt_source_2D, xc=xc, yc=xc, tc=tc, f=f, CFL=CFL, c=c0)
ADER4_aniso_       = Donnee2D(label="ADER4 aniso",         kappa=kappa, rho=rho, alpha=alpha, Mx=M, My=M,
                              S=pt_source_2D, xc=xc, yc=xc, tc=tc, f=f, CFL=CFL, c=c0)

LaxWendroff_aniso(LaxWendroff_aniso_)
ADER4_aniso(ADER4_aniso_)

anim2D(LaxWendroff_aniso_, duree=duree_anim)
anim2D(ADER4_aniso_, duree=duree_anim)

# ============================================================
# 2) Milieu HOMOGÉNÉISÉ anisotrope + MODULATION TEMPORELLE
#    -> tracé de la SOLUTION (Lax-Wendroff puis ADER4)
# ============================================================
LaxWendroff_aniso_mt_ = Donnee2D(label="Lax-Wendroff aniso mt", kappa=kappa, rho=rho, alpha=alpha, Mx=M, My=M,
                                 S=pt_source_2D, xc=xc, yc=xc, tc=tc, f=f, CFL=CFL, c=c0, **mod)
ADER4_aniso_mt_       = Donnee2D(label="ADER4 aniso mt",         kappa=kappa, rho=rho, alpha=alpha, Mx=M, My=M,
                                 S=pt_source_2D, xc=xc, yc=xc, tc=tc, f=f, CFL=CFL, c=c0, **mod)

LaxWendroff_aniso_mt(LaxWendroff_aniso_mt_)
ADER4_aniso_mt(ADER4_aniso_mt_)

anim2D(LaxWendroff_aniso_mt_, duree=duree_anim)
anim2D(ADER4_aniso_mt_, duree=duree_anim)

# ============================================================
# 3) Milieu MICRO-STRUCTURÉ (constant en temps)
#    -> tracé de la SOLUTION (Lax-Wendroff puis ADER4)
# ============================================================
LaxWendroff_ms_ = Donnee2D(label="Lax-Wendroff ms", kappa=kappa, rho=rho, alpha=alpha, Mx=M, My=M,
                           S=pt_source_2D, xc=xc, yc=xc, tc=tc, f=f, CFL=CFL, c=c0)
l, L = 5 * LaxWendroff_ms_.dy, 15 * LaxWendroff_ms_.dy
LaxWendroff_ms(LaxWendroff_ms_, l, L)
anim2D_ms(LaxWendroff_ms_, l, L, duree=duree_anim)

ADER4_ms_ = Donnee2D(label="ADER4 ms", kappa=kappa, rho=rho, alpha=alpha, Mx=M, My=M,
                     S=pt_source_2D, xc=xc, yc=xc, tc=tc, f=f, CFL=CFL, c=c0)
l, L = 5 * ADER4_ms_.dy, 15 * ADER4_ms_.dy
ADER4_ms(ADER4_ms_, l, L)
anim2D_ms(ADER4_ms_, l, L, duree=duree_anim)

# ============================================================
# 4) Milieu MICRO-STRUCTURÉ + MODULATION TEMPORELLE (échelon)
#    -> tracé de la SOLUTION (Lax-Wendroff puis ADER4)
# ============================================================
LaxWendroff_ms_mt_ = Donnee2D(label="Lax-Wendroff ms mt", kappa=kappa, rho=rho, alpha=alpha, Mx=M, My=M,
                              S=pt_source_2D, xc=xc, yc=xc, tc=tc, f=f, CFL=CFL, c=c0, **mod)
l, L = 5 * LaxWendroff_ms_mt_.dy, 15 * LaxWendroff_ms_mt_.dy
LaxWendroff_ms_mt(LaxWendroff_ms_mt_, l, L)
anim2D_ms(LaxWendroff_ms_mt_, l, L, duree=duree_anim)

ADER4_ms_mt_ = Donnee2D(label="ADER4 ms mt", kappa=kappa, rho=rho, alpha=alpha, Mx=M, My=M,
                        S=pt_source_2D, xc=xc, yc=xc, tc=tc, f=f, CFL=CFL, c=c0, **mod)
l, L = 5 * ADER4_ms_mt_.dy, 15 * ADER4_ms_mt_.dy
ADER4_ms_mt(ADER4_ms_mt_, l, L)
anim2D_ms(ADER4_ms_mt_, l, L, duree=duree_anim)
