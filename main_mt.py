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
from schema import LaxWendroff2D, LaxWendroff1D_cauchy
from sauvegarde import *
from tracer import *

"""
============================================================
Implémentation de la résolution 1D modulée en temps
============================================================
"""

#Paramètres
f_mt = 5
modulation = "echelon"
alpha = 0.5
eps = 0.5                             # |eps| << 1

if modulation == "echelon":
    rho = rho_echelon
    E = E_echelon
elif modulation == "sinus":
    rho = rho_sinus
    E = E_sinus
elif modulation == "triangle":
    rho = rho_triangle
    E = E_triangle

data1 = Donnee1D(M = 500, label ="LW_mt=" + modulation, eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi,
                          tc = (0, 1), xc = (0,1200), CFL = 0.95, f = 20, rho_mt=rho, E_mt=E, alpha = alpha)
data2 = Donnee1D(M = 500, label = "ADER4_mt=" + modulation, eps_r = eps/2, eps_E=eps, omega = f_mt*2*np.pi,
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

# data1_ = Donnee1D(M = 500, label = "LaxWendrof_r", eps_r = eps, eps_E=eps, omega = f_mt*2*np.pi,
#                  tc = (0, 0.6), xc = (0,1200), CFL = 0.95, f = 10, rho_mt=rho, E_mt=E, alpha = alpha, c = 3498.5051168234486)
# data2_ = Donnee1D(M = 500, label = "ADER4_r", eps_r = eps, eps_E=eps, omega = f_mt*2*np.pi,
#                   tc=(0, 0.6), xc=(0, 1200), CFL=0.95, f=10, rho_mt=rho, E_mt=E, alpha=alpha, c = 3498.5051168234486)
#
# data1_.CFL_maj()
# data2_.CFL_maj()
# data1_.U[...] = data1.U[:,::-1,:]
# data2_.U[...] = data2.U[:,::-1,:]

# anim1D_mt_comparaison(data1, data2, interval = 10)
# anim1D_mt_comparaison(data1, data1_, interval = 10)
# anim1D_mt_comparaison(data2, data2_, interval = 10)


#Problème de Cauchy
LW1D_cauchy_mt = Donnee1D(M = 150, label ="LW1D_mt=" + modulation, eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi, S=pt_source_1D,
                          c=1500, rho=1000, CFL=0.6, e=2.25e9,tc = (0, 0.2), xc = (0,500), f = 10, rho_mt=rho, E_mt=E, alpha = alpha)
LaxWendroff1D_cauchy_mt(LW1D_cauchy_mt)
anim1D_mt(LW1D_cauchy_mt)

# LW1D_cauchy = Donnee1D(M = 500, label ="LW", tc = (0, 0.5), xc = (0,1200), CFL = 0.95, f = 20)
# LaxWendroff1D_cauchy(LW1D_cauchy)
# anim1D_comparaison(LW1D_cauchy, LW1D_cauchy_mt)

LW2D_cauchy_mt = Donnee2D(label = 'LW2D_mt=' + modulation, eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi, Mx = 150, My = 150, S=pt_source_1D,
                          c=1500, rho=1000, CFL=0.6, e=2.25e9,tc = (0, 0.2), xc = (0,500), yc = (0, 500), f = 10, rho_mt=rho, E_mt=E, alpha = alpha)
LaxWendroff2D_cauchy_mt(LW2D_cauchy_mt)

"""
============================================================
Implémentation de la résolution 2D modulée en temps
============================================================
"""
LW2D = Donnee2D(Mx = 150, My = 150, label ="LW2D",tc = (0, 0.2), xc = (0,500), yc = (0, 500), CFL = 0.6, f = 10)

# LaxWendroff2D(LW2D)
LW2D = charger('.save/LaxWendroff2D_p.pkl')
# anim2D(LW2D)
# sauvegarder(LW2D)

LW2D_x = LW2D.projection((75, 0))
LW2D_y = LW2D.projection((0, 75))

LW2D_mt = Donnee2D(Mx = 150, My = 150, label ="LW2D_mt=" + modulation, eps_r = -eps, eps_E=eps, omega =f_mt * 2 * np.pi,
                   tc = (0, 0.2), xc = (0,500), yc = (0, 500), CFL = 0.6, f = 10, rho_mt=rho, E_mt=E, alpha = alpha)
# LaxWendroff2D_mt(LW2D_mt)
LW2D_mt = charger('.save/LaxWendroff2D_mt=0.pkl')
# sauvegarder(LW2D_mt)
# anim2D(LW2D_mt)

LW2D_mt_x = LW2D_mt.projection((75, 0))
LW2D_mt_y = LW2D_mt.projection((0, 75))


# anim1D_comparaison(LW2D_mt_x, LW2D_x)
# anim1D_comparaison(LW2D_mt_y, LW2D_y)

