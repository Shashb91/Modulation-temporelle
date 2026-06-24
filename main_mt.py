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
from sauvegarde import *
from tracer import *

"""
============================================================
Implémentation de la résolution 1D modulée en temps
============================================================
"""

#Paramètres
f_mt = 0
modulation = "echelon"
alpha = 0.5
eps = 0                              #0 < eps <<1

if modulation == "echelon":
    rho = rho_echelon
    E = E_echelon
elif modulation == "sinus":
    rho = rho_sinus
    E = E_sinus
elif modulation == "triangle":
    rho = rho_triangle
    E = E_triangle

data1 = Donnee1D(M = 500, label = "LaxWendrof_mt=" + modulation, eps_r = -eps, eps_E=eps, omega = f_mt*2*np.pi,
                 tc = (0, 0.6), xc = (0,1200), CFL = 0.95, f = 10, rho_mt=rho, E_mt=E, alpha = alpha)
data2 = Donnee1D(M = 500, label = "ADER4_mt=" + modulation, eps_r = eps/2, eps_E=eps, omega = f_mt*2*np.pi,
                 tc = (0, 0.6), xc = (0,1200), CFL = 0.95, f = 10, rho_mt=rho, E_mt=E, alpha = alpha)

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


"""
============================================================
Implémentation de la résolution 2D modulée en temps
============================================================
"""

data5 = Donnee2D(Mx = 150, My = 150, label = "LaxWendrof_mt=" + modulation, eps_r = -eps, eps_E=eps, omega = f_mt*2*np.pi,
                 tc = (0, 0.4), xc = (0,500), yc = (0, 500), CFL = 0.6, f = 10, rho_mt=rho, E_mt=E, alpha = alpha)
data5.CFL_maj()
print(data5)
LaxWendroff2D_mt(data5)
anim2D(data5)