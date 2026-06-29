"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff et d'ADER4 1D/2D en mileu
modulé en temps

Les modules concernant la modulation temporelle sont indicés
par _mt

Parametres :
- CFL = 0.6
- e = 9.4e9
============================================================
"""

from schema_mt import *
from schema import LaxWendroff2D, LaxWendroff1D_cauchy, ADER42D, ADER41D_cauchy
from source import pt_source_1D
from sauvegarde import *
from tracer import *

"""
============================================================
Implémentation de la résolution 1D modulée en temps
============================================================
"""

#Paramètres
f_mt = 1/0.15
modulation = "echelon"
alpha = 0.5
eps = 0.3                            # |eps| << 1, eps = 0 -> pas de modulation

if modulation == "echelon":
    rho = rho_echelon
    E = E_echelon
elif modulation == "sinus":
    rho = rho_sinus
    E = E_sinus
elif modulation == "triangle":
    rho = rho_triangle
    E = E_triangle

if eps == 0: modulation = "0"

data1 = Donnee1D(M = 500, label ="LW_mt=" + modulation, eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi,
                          tc = (0, 1), xc = (0,1200), CFL = 0.95, f = 20, rho_mt=rho, E_mt=E, alpha = alpha)
data2 = Donnee1D(M = 500, label = "ADER4_mt=" + modulation, eps_r = -eps, eps_E=eps, omega = f_mt*2*np.pi,
                 tc = (0, 1), xc = (0,1200), CFL = 0.95, f = 20, rho_mt=rho, E_mt=E, alpha = alpha)

#Resolution
# LaxWendroff1D_mt(data1)
# sauvegarder(data1)
# print(data1)
# data1 = charger('.save/LaxWendrof_mt=sinus_06-24_16-25-06.pkl')
# anim1D_mt(data1, interval = 10)

# ADER41D_mt(data2)
# sauvegarder(data2)
# print(data2)
# data2 = charger('.save/ADER4_mt=sinus_06-24_16-27-51.pkl')
# anim1D_mt(data2, interval = 10)
# anim1D_mt_comparaison(data1, data2, interval = 10)

#Problème de Cauchy
LW1D_cauchy_mt = Donnee1D(M = 200, label ="LW1D_mt=" + modulation, eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi, S=pt_source_1D,
                          c=1500, rho=1000, CFL=0.6, e=2.25e9,tc = (0, 0.2), xc = (0,500), f = 10, rho_mt=rho, E_mt=E, alpha = alpha)
# LaxWendroff1D_cauchy_mt(LW1D_cauchy_mt)
# anim1D_mt(LW1D_cauchy_mt)

# LW1D_cauchy = Donnee1D(M = 500, label ="LW", tc = (0, 0.5), xc = (0,1200), CFL = 0.95, f = 20)
# LaxWendroff1D_cauchy(LW1D_cauchy)
# anim1D_comparaison(LW1D_cauchy, LW1D_cauchy_mt)

ADER4_1D_cauchy_mt = Donnee1D(M = 100, label ="ADER4_1D_mt=" + modulation, eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi, S=pt_source_1D,
                              c=1500, rho=1000, CFL=0.6, e=2.25e9,tc = (0, 0.15), xc = (0,400), f = 10, rho_mt=rho, E_mt=E, alpha = alpha)
ADER41D_cauchy_mt(ADER4_1D_cauchy_mt)
anim1D_mt(ADER4_1D_cauchy_mt)

# ADER4_1D_cauchy = Donnee1D(M = 200, label ="ADER4_", eps_r = -eps, eps_E=eps, omega = 0, S=pt_source_1D,
#                            c = 1500, rho=1000, CFL=0.6, e=2.25e9,tc = (0, 0.2), xc = (0,500), f = 10, rho_mt=rho, E_mt=E, alpha = alpha)
# ADER41D_cauchy(ADER4_1D_cauchy)
# anim1D_comparaison(ADER4_1D_cauchy, ADER4_1D_cauchy_mt)


"""
============================================================
Implémentation de la résolution 2D modulée en temps
============================================================
"""
LW2D = Donnee2D(Mx = 150, My = 150, label ="LW2D",tc = (0, 0.2), xc = (0,500), yc = (0, 500), CFL = 0.6, f = 10)

# LaxWendroff2D(LW2D)
# LW2D = charger('.save/LaxWendroff2D_p.pkl')
# anim2D(LW2D)
# sauvegarder(LW2D)

# LW2D_x = LW2D.projection((75, 0))
# LW2D_y = LW2D.projection((0, 75))

LW2D_mt = Donnee2D(Mx = 150, My = 150, label ="LW2D_mt=" + modulation, eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi,
                   tc = (0, 0.2), xc = (0,500), yc = (0, 500), CFL = 0.6, f = 10, rho_mt=rho, E_mt=E, alpha = alpha)
# LaxWendroff2D_mt(LW2D_mt)
# LW2D_mt = charger('.save/LW2D_mt=echelon_06-26_12-03-28.pkl')
# anim2D(LW2D_mt)
# sauvegarder(LW2D_mt)

# LW2D_mt_x = LW2D_mt.projection((75, 0))
# LW2D_mt_y = LW2D_mt.projection((0, 75))

# anim1D_comparaison(LW2D_mt_x, LW2D_x)
# anim1D_comparaison(LW2D_mt_y, LW2D_y)

# ADER4_2D = Donnee2D(Mx = 100, My = 100, label ="ADER4",tc = (0, 0.2), xc = (0,500), yc = (0, 500), CFL = 0.6, f = 10)
# ADER42D(ADER4_2D)
# # ADER4_2D = charger('.save/ADER4_06-29_10-49-44.pkl')
# anim2D(ADER4_2D)
# sauvegarder(ADER4_2D)
#
# ADER4_2D_x = ADER4_2D.projection((75, 0))
# ADER4_2D_y = ADER4_2D.projection((0, 75))

ADER4_2D_mt = Donnee2D(Mx = 150, My = 150, label ="ADER4_mt=0", eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi,
                       tc = (0, 0.2), xc = (0,500), yc = (0, 500), CFL = 0.6, f = 10, rho_mt=rho, E_mt=E, alpha = alpha)
# ADER42D_mt(ADER4_2D_mt)
# ADER4_2D_mt = charger('.save/ADER4_mt=0_06-29_11-03-13.pkl')
# anim2D(ADER4_2D_mt)
# sauvegarder(ADER4_2D_mt)

# ADER4_2D_mt_x = ADER4_2D_mt.projection((75, 0))
# ADER4_2D_mt_y = ADER4_2D_mt.projection((0, 75))
#
#
# anim1D_comparaison(ADER4_2D_mt_x, ADER4_2D_x, interval = 40)
# anim1D_comparaison(ADER4_2D_mt_y, ADER4_2D_y, interval = 40)

#Problème de Cauchy
# LW2D_cauchy_mt = Donnee2D(label = 'LW2D_mt=' + modulation, eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi, Mx = 200, My = 200, S=pt_source_1D,
#                           c=1500, rho=1000, CFL=0.6, e=2.25e9,tc = (0, 0.2), xc = (0,500), yc = (0, 500), f = 10, rho_mt=rho, E_mt=E, alpha = alpha)

# LaxWendroff2D_cauchy_mt(LW2D_cauchy_mt)
# sauvegarder(LW2D_cauchy_mt)
# LW2D_cauchy_mt = charger('.save/LW2D_mt=echelon_06-26_12-03-28.pkl')
# anim2D(LW2D_cauchy_mt)

# LW2D_cauchy_mt_x = LW2D_cauchy_mt.projection((75, 0))
# anim1D_mt_comparaison(LW2D_cauchy_mt_x, LW1D_cauchy_mt)

ADER4_2D_cauchy_mt = Donnee2D(label = 'ADER4_2D_mt=' + modulation, eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi, Mx = 100, My = 100, S=pt_source_1D,
                              c=1500, rho=1000, CFL=0.6, e=2.25e9,tc = (0, 0.15), xc = (0,400), yc = (0, 400), f = 10, rho_mt=rho, E_mt=E, alpha = alpha)

ADER42D_cauchy_mt(ADER4_2D_cauchy_mt)
sauvegarder(ADER4_2D_cauchy_mt)
# ADER4_2D_cauchy_mt = charger('.save/ADER4_2D_mt=0_06-29_12-01-58.pkl')
anim2D(ADER4_2D_cauchy_mt)

ADER4_2D_cauchy_mt_x = ADER4_2D_cauchy_mt.projection((50, 0))
anim1D_mt_comparaison(ADER4_2D_cauchy_mt_x, ADER4_1D_cauchy_mt)