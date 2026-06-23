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
f_mt = 3
modulation = "sinus"
alpha = 0.5
eps = 0.4                                  #0 < eps <<1
CFL = 0.95              #correction de la CFL avec la modulation

if modulation == "echelon":
    rho = rho_echelon
    E = E_echelon
elif modulation == "sinus":
    rho = rho_sinus
    E = E_sinus
elif modulation == "triangle":
    rho = rho_triangle
    E = E_triangle

data1 = Donnee1D(M = 750, label = "ADER4_f=" + str(f_mt) + "-Hz_mt=" + modulation, eps_r = - eps, eps_E=eps, omega = f_mt*2*np.pi,
                 tc = (0, 0.6), xc = (0,1500), CFL = CFL, f = 10, rho_mt=rho, E_mt=E, alpha = alpha)

#Resolution
U = ADER41D_mt(data1)
print(data1)


# data1 = charger('.save/ADER4_f=3-Hz_mt=echelon_06-17_14-27-24.pkl')
anim1D_mt(data1, interval = 0.00000001)
sauvegarder(data1)