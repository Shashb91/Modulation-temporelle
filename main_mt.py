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
from tracer import *
from sauvegarde import *

"""
============================================================
Implémentation de la résolution 1D modulée en temps
============================================================
"""

#Paramètres
f_mt = 20
modulation = "echelon"
alpha = 0.5
eps = 0.25                                     #0 < eps <<1
if modulation == "echelon":
    rho = rho_echelon
    E = E_echelon
elif modulation == "sinus":
    rho = rho_sinus
    E = E_sinus

data1 = Donnee1D(M = 1000, label = "ADER4_mt f=" + str(f_mt) + " Hz", e = 9.4e9, eps = eps, omega = f_mt*2*np.pi,
                 tc = (0, 0.185), xc = (0,600), CFL = 0.65, f = 10, rho_mt=rho, E_mt=E)

#Resolution
if modulation == "echelon" : U = ADER41D_mt(data1, alpha = alpha)
elif modulation == "sinus" : U = ADER41D_mt(data1)


#data1 = charger('.save/ADER4_mt f=20 Hz_06-15_16-27-44.pkl')
anim1D(data1, interval = 1)
sauvegarder(data1)