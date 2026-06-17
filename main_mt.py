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
f_mt = 15
modulation = "echelon"
alpha = 0.5
eps = 0.5                                   #0 < eps <<1
CFL = 0.4              #correction de la CFL avec la modulation

if modulation == "echelon":
    rho = rho_echelon
    E = E_echelon
elif modulation == "sinus":
    rho = rho_sinus
    E = E_sinus

data1 = Donnee1D(M = 1500, label = "ADER4_f=" + str(f_mt) + "-Hz_mt=" + modulation, eps = eps, omega = f_mt*2*np.pi,
                 tc = (0, 0.185), xc = (0,600), CFL = CFL, f = 10, rho_mt=rho, E_mt=E, alpha = alpha)

#Resolution
U = ADER41D_mt(data1)


# data1 = charger('.save/ADER4_f=5-Hz_mt=echelon_06-17_09-30-51.pkl')
anim1D(data1, interval = 0.001 )
sauvegarder(data1)